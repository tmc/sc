"""
SAE Coverage Pipeline: End-to-End Unified Pipeline

Unifies the complete flow:
1. SAE Encoding: Hidden states → Feature activations (states)
2. Statechart Building: Feature sequences → Transitions + guards
3. Coverage Simulation: Simulate statechart → Predict coverage

Single differentiable pipeline from hidden states to coverage.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet

from .sae_state_extractor import SAEConfig, SAEStateExtractor, StateConfiguration
from .guard_synthesizer import GuardSynthesizer, FeatureGuard, ActivationPattern
from .coverage_simulator import CoverageSimulator, CoveragePrediction, StatechartTransition


@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline."""
    # SAE config
    sae_input_dim: int = 256
    sae_expansion: int = 16
    sae_k_active: int = 32

    # Guard synthesis config
    guard_max_depth: int = 3
    guard_population: int = 30
    guard_generations: int = 30

    # Coverage simulation config
    simulation_max_steps: int = 100
    use_all_paths: bool = False

    # Training config
    min_transition_count: int = 2
    min_guard_examples: int = 5


@dataclass
class PipelineResult:
    """Result from running the pipeline."""
    # Extracted statechart
    n_states: int
    n_transitions: int
    n_features: int

    # Synthesized guards
    n_guards: int
    guard_avg_f1: float

    # Coverage prediction
    coverage_precision: float
    coverage_recall: float
    coverage_f1: float

    # Learned components
    guards: List[FeatureGuard] = field(default_factory=list)
    transition_map: Dict[Tuple[FrozenSet[int], FrozenSet[int]], FeatureGuard] = field(default_factory=dict)


class SAECoveragePipeline:
    """
    Unified SAE → States → Guards → Coverage pipeline.

    End-to-end flow:
    1. Feed hidden states to SAE
    2. SAE features ARE states (no clustering)
    3. Observe transitions between feature configurations
    4. Synthesize guards for each transition
    5. Simulate statechart to predict coverage
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Components
        sae_config = SAEConfig(
            input_dim=self.config.sae_input_dim,
            expansion_factor=self.config.sae_expansion,
            k_active=self.config.sae_k_active,
        )
        self.extractor = SAEStateExtractor(sae_config)
        self.simulator = CoverageSimulator()

        # Synthesized guards per transition
        self.transition_guards: Dict[Tuple[FrozenSet[int], FrozenSet[int]], FeatureGuard] = {}

        # Feature indices (discovered during extraction)
        self.feature_indices: List[int] = []

        # Statistics
        self.extraction_done = False
        self.guards_synthesized = False

    def extract_states(self, hidden_sequence: mx.array) -> List[StateConfiguration]:
        """
        Step 1: Extract states from hidden sequence.

        Args:
            hidden_sequence: (seq_len, dim) hidden states

        Returns:
            List of StateConfiguration for each step
        """
        self.extractor.reset()
        configs = self.extractor.extract_sequence(hidden_sequence)

        # Build simulator statechart
        self.simulator.build_from_observations(
            configs,
            min_transition_count=self.config.min_transition_count,
        )

        # Collect all feature indices
        self.feature_indices = list(self.extractor.get_all_active_features())

        self.extraction_done = True
        return configs

    def synthesize_guards(
        self,
        positive_transitions: Dict[Tuple[FrozenSet[int], FrozenSet[int]], List[StateConfiguration]],
        negative_transitions: Dict[Tuple[FrozenSet[int], FrozenSet[int]], List[StateConfiguration]],
        verbose: bool = True,
    ) -> Dict[Tuple[FrozenSet[int], FrozenSet[int]], FeatureGuard]:
        """
        Step 2: Synthesize guards for transitions.

        Args:
            positive_transitions: Transitions where guard should be true
            negative_transitions: Transitions where guard should be false

        Returns:
            Dict mapping (src, tgt) -> synthesized guard
        """
        if not self.feature_indices:
            raise ValueError("Must extract states first")

        synthesizer = GuardSynthesizer(
            feature_indices=self.feature_indices,
            max_depth=self.config.guard_max_depth,
            population_size=self.config.guard_population,
        )

        guards = {}

        # Get all transitions that need guards
        all_transitions = set(positive_transitions.keys()) | set(negative_transitions.keys())

        for trans_key in all_transitions:
            pos_configs = positive_transitions.get(trans_key, [])
            neg_configs = negative_transitions.get(trans_key, [])

            if len(pos_configs) < self.config.min_guard_examples:
                continue

            # Convert to activation patterns
            positive = [
                ActivationPattern(
                    active_features=config.active_features,
                    activations=config.activations,
                    label=True,
                )
                for config in pos_configs
            ]

            negative = [
                ActivationPattern(
                    active_features=config.active_features,
                    activations=config.activations,
                    label=False,
                )
                for config in neg_configs
            ]

            if verbose:
                print(f"\nSynthesizing guard for {trans_key[0][:3]}... -> {trans_key[1][:3]}...")
                print(f"  Positive: {len(positive)}, Negative: {len(negative)}")

            guard = synthesizer.synthesize(
                positive, negative,
                n_generations=self.config.guard_generations,
                verbose=False,
            )

            guards[trans_key] = guard

            if verbose:
                print(f"  Guard: {guard.to_string()[:50]}... (F1={guard.f1:.3f})")

        self.transition_guards = guards

        # Update simulator transitions with guards
        for trans in self.simulator.transitions:
            key = (trans.source, trans.target)
            if key in guards:
                trans.guard = guards[key]

        self.guards_synthesized = True
        return guards

    def predict_coverage(
        self,
        initial_hidden: mx.array,
    ) -> CoveragePrediction:
        """
        Step 3: Predict coverage via simulation.

        Args:
            initial_hidden: Initial hidden state

        Returns:
            CoveragePrediction with predicted features/states
        """
        if not self.extraction_done:
            raise ValueError("Must extract states first")

        # Extract initial state
        initial_config = self.extractor.extract_state(initial_hidden)

        # Create activation pattern
        initial_pattern = ActivationPattern(
            active_features=initial_config.active_features,
            activations=initial_config.activations,
            label=True,
        )

        # Simulate
        prediction = self.simulator.predict_coverage(
            initial_pattern,
            use_all_paths=self.config.use_all_paths,
        )

        return prediction

    def run_full_pipeline(
        self,
        hidden_sequence: mx.array,
        ground_truth_coverage: Set[int] = None,
        verbose: bool = True,
    ) -> PipelineResult:
        """
        Run complete pipeline end-to-end.

        Args:
            hidden_sequence: (seq_len, dim) hidden states
            ground_truth_coverage: Optional ground truth for evaluation
            verbose: Print progress

        Returns:
            PipelineResult with all metrics
        """
        if verbose:
            print("=" * 60)
            print("SAE COVERAGE PIPELINE: End-to-End")
            print("=" * 60)

        # Step 1: Extract states
        if verbose:
            print("\n[Step 1] Extracting states from hidden sequence...")

        configs = self.extract_states(hidden_sequence)

        if verbose:
            print(f"  Extracted {len(configs)} configurations")
            print(f"  Unique features: {len(self.feature_indices)}")
            print(f"  Statechart: {self.simulator.get_statistics()}")

        # Step 2: Build transition examples
        if verbose:
            print("\n[Step 2] Building transition examples...")

        positive_transitions = {}
        negative_transitions = {}

        # Group configurations by their transitions
        for i in range(len(configs) - 1):
            src = configs[i].active_features
            tgt = configs[i + 1].active_features
            key = (src, tgt)

            if key not in positive_transitions:
                positive_transitions[key] = []
            positive_transitions[key].append(configs[i])

        # Create negative examples (transitions that didn't happen)
        all_states = list(self.simulator.states)
        for src in all_states[:10]:  # Limit for efficiency
            for tgt in all_states[:10]:
                key = (src, tgt)
                if key not in positive_transitions:
                    if key not in negative_transitions:
                        negative_transitions[key] = []
                    # Use any config with matching source
                    for config in configs:
                        if config.active_features == src:
                            negative_transitions[key].append(config)
                            if len(negative_transitions[key]) >= 3:
                                break

        if verbose:
            print(f"  Positive transitions: {len(positive_transitions)}")
            print(f"  Negative transitions: {len(negative_transitions)}")

        # Step 3: Synthesize guards
        if verbose:
            print("\n[Step 3] Synthesizing guards...")

        guards = self.synthesize_guards(
            positive_transitions,
            negative_transitions,
            verbose=verbose,
        )

        if verbose:
            print(f"\n  Synthesized {len(guards)} guards")

        # Step 4: Predict coverage
        if verbose:
            print("\n[Step 4] Predicting coverage...")

        # Use first hidden state as initial
        prediction = self.predict_coverage(hidden_sequence[0])

        if verbose:
            print(f"  Predicted features: {len(prediction.predicted_features)}")
            print(f"  Predicted states: {len(prediction.predicted_states)}")

        # Evaluate if ground truth provided
        if ground_truth_coverage:
            prediction.actual_features = ground_truth_coverage
            if verbose:
                print(f"\n  Coverage F1: {prediction.feature_f1:.3f}")
                print(f"  Precision: {prediction.feature_precision:.3f}")
                print(f"  Recall: {prediction.feature_recall:.3f}")

        # Compute guard statistics
        guard_f1s = [g.f1 for g in guards.values()]
        avg_guard_f1 = sum(guard_f1s) / len(guard_f1s) if guard_f1s else 0.0

        result = PipelineResult(
            n_states=len(self.simulator.states),
            n_transitions=len(self.simulator.transitions),
            n_features=len(self.feature_indices),
            n_guards=len(guards),
            guard_avg_f1=avg_guard_f1,
            coverage_precision=prediction.feature_precision if ground_truth_coverage else 0.0,
            coverage_recall=prediction.feature_recall if ground_truth_coverage else 0.0,
            coverage_f1=prediction.feature_f1 if ground_truth_coverage else 0.0,
            guards=list(guards.values()),
            transition_map=guards,
        )

        if verbose:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"States: {result.n_states}, Transitions: {result.n_transitions}")
            print(f"Guards: {result.n_guards}, Avg F1: {result.guard_avg_f1:.3f}")
            if ground_truth_coverage:
                print(f"Coverage F1: {result.coverage_f1:.3f}")

        return result

    def get_learned_statechart(self) -> Dict:
        """
        Export the learned statechart.

        Returns JSON-serializable representation.
        """
        return {
            'states': [list(s) for s in self.simulator.states],
            'transitions': [
                {
                    'source': list(t.source),
                    'target': list(t.target),
                    'guard': t.guard.to_string() if t.guard else None,
                    'count': t.count,
                }
                for t in self.simulator.transitions
            ],
            'n_features': len(self.feature_indices),
            'feature_indices': self.feature_indices,
        }


def demo():
    """Demonstrate the unified pipeline."""
    print("=" * 60)
    print("SAE COVERAGE PIPELINE: Full Demo")
    print("=" * 60)

    # Create pipeline
    config = PipelineConfig(
        sae_input_dim=64,
        sae_expansion=8,
        sae_k_active=8,
        guard_generations=20,
        min_transition_count=1,
        min_guard_examples=2,
    )
    pipeline = SAECoveragePipeline(config)

    # Generate synthetic hidden sequence
    mx.random.seed(42)
    hidden_sequence = mx.random.normal((50, 64))

    # Run pipeline
    result = pipeline.run_full_pipeline(
        hidden_sequence,
        ground_truth_coverage=None,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Statechart: {result.n_states} states, {result.n_transitions} transitions")
    print(f"Features: {result.n_features}")
    print(f"Guards: {result.n_guards} (avg F1={result.guard_avg_f1:.3f})")

    # Export statechart
    statechart = pipeline.get_learned_statechart()
    print(f"\nExported statechart with {len(statechart['transitions'])} transitions")

    return pipeline, result


if __name__ == "__main__":
    demo()
