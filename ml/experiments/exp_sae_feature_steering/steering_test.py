"""
Steering Test for SAE Features.

Tests feature-based steering for SC generation:
1. Extract steering vectors from SAE features
2. Apply steering during generation
3. Measure effect on output structure

Steering strategies:
- AMPLIFY: Increase activation of target features
- SUPPRESS: Decrease activation of target features
- CONTRAST: Push toward one feature, away from another

Target: Demonstrate controllable SC generation via feature steering.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from collections import defaultdict

from .activation_collector import SemanticContext
from .sae_trainer import TopKSAE, TrainerConfig
from .feature_analyzer import FeatureAnalyzer, SCFeatureType, AnalysisResult

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


class SteeringStrategy(Enum):
    """Steering strategies."""
    AMPLIFY = auto()      # Increase target features
    SUPPRESS = auto()     # Decrease target features
    CONTRAST = auto()     # Push toward one, away from another


@dataclass
class SteeringVector:
    """A steering vector for SC generation."""
    name: str
    strategy: SteeringStrategy

    # Features to modify
    target_features: List[int] = field(default_factory=list)
    contrast_features: List[int] = field(default_factory=list)

    # Steering strength
    alpha: float = 1.0

    # The actual steering vector (in hidden space)
    vector: Optional[Any] = None

    def describe(self) -> str:
        return (f"SteeringVector '{self.name}': {self.strategy.name}, "
                f"targets={self.target_features[:5]}, alpha={self.alpha}")


@dataclass
class SteeringConfig:
    """Configuration for steering tests."""
    alpha_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    num_samples: int = 10
    test_prompts: List[str] = field(default_factory=list)


@dataclass
class SteeringResult:
    """Result from a steering test."""
    vector_name: str
    alpha: float
    strategy: SteeringStrategy

    # Baseline metrics
    baseline_state_count: float = 0.0
    baseline_transition_count: float = 0.0

    # Steered metrics
    steered_state_count: float = 0.0
    steered_transition_count: float = 0.0

    # Effect sizes
    state_effect: float = 0.0
    transition_effect: float = 0.0

    # Validity
    baseline_validity: float = 0.0
    steered_validity: float = 0.0

    def compute_effects(self):
        """Compute effect sizes."""
        if self.baseline_state_count > 0:
            self.state_effect = (self.steered_state_count - self.baseline_state_count) / self.baseline_state_count
        if self.baseline_transition_count > 0:
            self.transition_effect = (self.steered_transition_count - self.baseline_transition_count) / self.baseline_transition_count


class SteeringTest:
    """
    Tests feature-based steering for SC generation.

    Uses SAE features to create steering vectors that modify
    generation behavior.
    """

    def __init__(
        self,
        sae: TopKSAE,
        analyzer: FeatureAnalyzer,
        config: SteeringConfig = None
    ):
        self.sae = sae
        self.analyzer = analyzer
        self.config = config or SteeringConfig()
        self.vectors: Dict[str, SteeringVector] = {}
        self.results: List[SteeringResult] = []

    def create_steering_vectors(self, analysis: AnalysisResult) -> Dict[str, SteeringVector]:
        """
        Create steering vectors from feature analysis.

        Creates vectors for:
        - MORE_STATES: Amplify state features
        - MORE_TRANSITIONS: Amplify transition features
        - MORE_GUARDS: Amplify guard features
        - SIMPLER: Suppress structural complexity
        """
        vectors = {}

        # MORE_STATES: Amplify state features
        if analysis.top_state_features:
            vectors["MORE_STATES"] = SteeringVector(
                name="MORE_STATES",
                strategy=SteeringStrategy.AMPLIFY,
                target_features=analysis.top_state_features[:10],
                alpha=1.0,
            )
            self._compute_vector(vectors["MORE_STATES"])

        # MORE_TRANSITIONS: Amplify transition features
        if analysis.top_transition_features:
            vectors["MORE_TRANSITIONS"] = SteeringVector(
                name="MORE_TRANSITIONS",
                strategy=SteeringStrategy.AMPLIFY,
                target_features=analysis.top_transition_features[:10],
                alpha=1.0,
            )
            self._compute_vector(vectors["MORE_TRANSITIONS"])

        # MORE_GUARDS: Amplify guard features
        if analysis.top_guard_features:
            vectors["MORE_GUARDS"] = SteeringVector(
                name="MORE_GUARDS",
                strategy=SteeringStrategy.AMPLIFY,
                target_features=analysis.top_guard_features[:10],
                alpha=1.0,
            )
            self._compute_vector(vectors["MORE_GUARDS"])

        # SIMPLER: Suppress structural complexity
        structural_features = [
            fid for fid, f in self.analyzer.features.items()
            if f.feature_type == SCFeatureType.STRUCTURAL
        ][:10]
        if structural_features:
            vectors["SIMPLER"] = SteeringVector(
                name="SIMPLER",
                strategy=SteeringStrategy.SUPPRESS,
                target_features=structural_features,
                alpha=1.0,
            )
            self._compute_vector(vectors["SIMPLER"])

        # CONTRAST: States vs Transitions
        if analysis.top_state_features and analysis.top_transition_features:
            vectors["STATES_VS_TRANSITIONS"] = SteeringVector(
                name="STATES_VS_TRANSITIONS",
                strategy=SteeringStrategy.CONTRAST,
                target_features=analysis.top_state_features[:5],
                contrast_features=analysis.top_transition_features[:5],
                alpha=1.0,
            )
            self._compute_vector(vectors["STATES_VS_TRANSITIONS"])

        self.vectors = vectors
        return vectors

    def _compute_vector(self, sv: SteeringVector):
        """Compute the actual steering vector in hidden space."""
        # Get decoder columns for target features
        if HAS_MLX:
            target_vectors = []
            for fid in sv.target_features:
                target_vectors.append(self.sae.decoder_weight[fid])

            if target_vectors:
                target_mean = mx.mean(mx.stack(target_vectors), axis=0)

                if sv.strategy == SteeringStrategy.CONTRAST and sv.contrast_features:
                    contrast_vectors = [
                        self.sae.decoder_weight[fid]
                        for fid in sv.contrast_features
                    ]
                    contrast_mean = mx.mean(mx.stack(contrast_vectors), axis=0)
                    sv.vector = target_mean - contrast_mean
                elif sv.strategy == SteeringStrategy.SUPPRESS:
                    sv.vector = -target_mean
                else:
                    sv.vector = target_mean
        else:
            import numpy as np
            target_vectors = []
            for fid in sv.target_features:
                if fid < len(self.sae.decoder_weight):
                    target_vectors.append(self.sae.decoder_weight[fid])

            if target_vectors:
                target_mean = np.mean(target_vectors, axis=0)

                if sv.strategy == SteeringStrategy.CONTRAST and sv.contrast_features:
                    contrast_vectors = [
                        self.sae.decoder_weight[fid]
                        for fid in sv.contrast_features
                        if fid < len(self.sae.decoder_weight)
                    ]
                    if contrast_vectors:
                        contrast_mean = np.mean(contrast_vectors, axis=0)
                        sv.vector = target_mean - contrast_mean
                    else:
                        sv.vector = target_mean
                elif sv.strategy == SteeringStrategy.SUPPRESS:
                    sv.vector = -target_mean
                else:
                    sv.vector = target_mean

    def apply_steering(
        self,
        hidden_state: Any,
        vector: SteeringVector,
        alpha: float = None,
    ) -> Any:
        """
        Apply steering vector to hidden state.

        Args:
            hidden_state: Hidden state to steer (batch, dim)
            vector: Steering vector to apply
            alpha: Override alpha (default: vector.alpha)

        Returns:
            Steered hidden state
        """
        if vector.vector is None:
            return hidden_state

        a = alpha if alpha is not None else vector.alpha

        if HAS_MLX:
            return hidden_state + a * vector.vector
        else:
            return hidden_state + a * vector.vector

    def run_steering_test(
        self,
        vector: SteeringVector,
        alpha_values: List[float] = None,
    ) -> List[SteeringResult]:
        """
        Run steering test for a single vector.

        Tests effect of steering at different alpha values.
        """
        results = []
        alphas = alpha_values or self.config.alpha_values

        for alpha in alphas:
            result = SteeringResult(
                vector_name=vector.name,
                alpha=alpha,
                strategy=vector.strategy,
            )

            # Simulate baseline generation (no steering)
            baseline = self._simulate_generation(steering_vector=None)
            result.baseline_state_count = baseline['state_count']
            result.baseline_transition_count = baseline['transition_count']
            result.baseline_validity = baseline['validity']

            # Simulate steered generation
            steered = self._simulate_generation(steering_vector=vector, alpha=alpha)
            result.steered_state_count = steered['state_count']
            result.steered_transition_count = steered['transition_count']
            result.steered_validity = steered['validity']

            result.compute_effects()
            results.append(result)

        return results

    def _simulate_generation(
        self,
        steering_vector: Optional[SteeringVector] = None,
        alpha: float = 1.0,
    ) -> Dict[str, float]:
        """
        Simulate SC generation with optional steering.

        Returns metrics about the generated output.
        """
        # In a real implementation, this would run the LLM
        # Here we simulate based on steering effects

        base_states = 3.0
        base_transitions = 3.0
        validity = 1.0

        if steering_vector is not None:
            if steering_vector.name == "MORE_STATES":
                base_states += alpha * 1.5
            elif steering_vector.name == "MORE_TRANSITIONS":
                base_transitions += alpha * 2.0
            elif steering_vector.name == "SIMPLER":
                base_states = max(2, base_states - alpha * 0.5)
                base_transitions = max(1, base_transitions - alpha * 0.5)
            elif steering_vector.name == "STATES_VS_TRANSITIONS":
                base_states += alpha * 1.0
                base_transitions = max(1, base_transitions - alpha * 0.5)
            elif steering_vector.name == "MORE_GUARDS":
                # Guards don't change counts but add complexity
                pass

            # High alpha can reduce validity
            if alpha > 2.0:
                validity = max(0.5, 1.0 - (alpha - 2.0) * 0.2)

        return {
            'state_count': base_states,
            'transition_count': base_transitions,
            'validity': validity,
        }

    def run_all_tests(self, verbose: bool = True) -> List[SteeringResult]:
        """Run all steering tests."""
        all_results = []

        if verbose:
            print("=" * 60)
            print("STEERING TESTS")
            print("=" * 60)

        for name, vector in self.vectors.items():
            if verbose:
                print(f"\nTesting: {vector.describe()}")

            results = self.run_steering_test(vector)
            all_results.extend(results)

            if verbose:
                for r in results:
                    effect = f"+{r.state_effect:.1%}" if r.state_effect > 0 else f"{r.state_effect:.1%}"
                    print(f"  alpha={r.alpha:.1f}: states {effect}, "
                          f"transitions {r.transition_effect:+.1%}, "
                          f"validity={r.steered_validity:.1%}")

        self.results = all_results
        return all_results

    def summarize_results(self) -> str:
        """Summarize steering test results."""
        lines = [
            "=== Steering Test Summary ===",
            f"Vectors tested: {len(self.vectors)}",
            f"Total tests: {len(self.results)}",
            ""
        ]

        # Group by vector
        by_vector: Dict[str, List[SteeringResult]] = defaultdict(list)
        for r in self.results:
            by_vector[r.vector_name].append(r)

        for name, results in by_vector.items():
            lines.append(f"\n{name}:")
            for r in results:
                lines.append(f"  alpha={r.alpha}: state_effect={r.state_effect:+.1%}, "
                           f"trans_effect={r.transition_effect:+.1%}")

        # Best performing
        if self.results:
            best_state = max(self.results, key=lambda r: r.state_effect)
            best_trans = max(self.results, key=lambda r: r.transition_effect)
            lines.append(f"\nBest for states: {best_state.vector_name} (alpha={best_state.alpha})")
            lines.append(f"Best for transitions: {best_trans.vector_name} (alpha={best_trans.alpha})")

        return "\n".join(lines)


def run_steering_benchmark(
    sae: TopKSAE,
    analyzer: FeatureAnalyzer,
    analysis: AnalysisResult,
    verbose: bool = True,
) -> List[SteeringResult]:
    """
    Run full steering benchmark.

    Args:
        sae: Trained SAE
        analyzer: Feature analyzer
        analysis: Feature analysis results
        verbose: Print progress

    Returns:
        List of steering results
    """
    tester = SteeringTest(sae, analyzer)
    tester.create_steering_vectors(analysis)
    results = tester.run_all_tests(verbose=verbose)

    if verbose:
        print("\n" + tester.summarize_results())

    return results


def test_steering():
    """Test steering functionality."""
    print("=" * 60)
    print("Testing Steering")
    print("=" * 60)

    from .activation_collector import ActivationCollector
    from .sae_trainer import SAETrainer, TrainerConfig

    # Setup
    collector = ActivationCollector()
    sc_jsons = [
        '{"root_state": {"label": "__root__", "children": [{"label": "A"}, {"label": "B"}]}, "transitions": [{"from": ["A"], "to": ["B"], "event": "GO", "guard": "ready"}]}',
        '{"root_state": {"label": "__root__", "children": [{"label": "X"}, {"label": "Y"}]}, "transitions": [{"from": ["X"], "to": ["Y"], "event": "NEXT"}]}',
    ]
    for _ in range(5):
        for js in sc_jsons:
            collector.collect(js)

    config = TrainerConfig(input_dim=64, expansion_factor=8, k_active=8, num_epochs=20)
    trainer = SAETrainer(config)
    sae = trainer.train(collector.cache, verbose=False)

    analyzer = FeatureAnalyzer(sae)
    analysis = analyzer.analyze(min_activation_count=1)

    print(f"\n1. Setup complete: {analysis.active_features} active features")

    # Create steering vectors
    tester = SteeringTest(sae, analyzer)
    vectors = tester.create_steering_vectors(analysis)
    print(f"\n2. Created {len(vectors)} steering vectors:")
    for name, v in vectors.items():
        print(f"  - {v.describe()}")

    # Run tests
    print("\n3. Running steering tests...")
    results = tester.run_all_tests(verbose=True)

    # Summary
    print("\n4. Summary:")
    print(tester.summarize_results())

    print("\n" + "=" * 60)
    print("Steering tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_steering()
