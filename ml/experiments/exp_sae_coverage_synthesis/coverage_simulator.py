"""
Coverage Simulator: Predict Coverage via Statechart Simulation

Key insight: Coverage prediction IS statechart simulation.
- States = SAE feature configurations
- Transitions = changes in active features
- Guards = conditions on feature activations
- Coverage = which states get visited

From exp_coverage_prediction:
- Abstract interpretation over statechart
- Guard evaluation determines reachability
- Visited states map to covered code

Here we simulate over the SAE-derived statechart.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from collections import defaultdict

from .sae_state_extractor import StateConfiguration
from .guard_synthesizer import FeatureGuard, ActivationPattern


@dataclass
class StatechartTransition:
    """
    A transition in the SAE-derived statechart.

    Source/target are feature configurations.
    Guard is a FeatureGuard.
    """
    source: FrozenSet[int]  # Source feature configuration
    target: FrozenSet[int]  # Target feature configuration
    guard: Optional[FeatureGuard] = None
    probability: float = 1.0  # Transition probability (for probabilistic simulation)
    count: int = 0  # How many times we've seen this transition

    def is_enabled(self, pattern: ActivationPattern) -> bool:
        """Check if transition is enabled."""
        if self.guard is None:
            return True
        return self.guard.evaluate(pattern)


@dataclass
class SimulationResult:
    """Result of a statechart simulation."""
    visited_states: Set[FrozenSet[int]]
    visited_features: Set[int]
    transitions_taken: List[StatechartTransition]
    steps: int
    terminated_normally: bool = True


@dataclass
class CoveragePrediction:
    """
    Predicted coverage from simulation.

    Maps SAE feature states to code coverage.
    """
    predicted_features: Set[int]  # Features predicted to be active
    predicted_states: Set[FrozenSet[int]]  # Configurations predicted to be visited
    confidence: float  # Confidence in prediction (0-1)

    # If we have ground truth
    actual_features: Optional[Set[int]] = None
    actual_states: Optional[Set[FrozenSet[int]]] = None

    @property
    def feature_precision(self) -> float:
        if self.actual_features is None:
            return 0.0
        if not self.predicted_features:
            return 0.0
        correct = len(self.predicted_features & self.actual_features)
        return correct / len(self.predicted_features)

    @property
    def feature_recall(self) -> float:
        if self.actual_features is None:
            return 0.0
        if not self.actual_features:
            return 1.0
        correct = len(self.predicted_features & self.actual_features)
        return correct / len(self.actual_features)

    @property
    def feature_f1(self) -> float:
        p, r = self.feature_precision, self.feature_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


class CoverageSimulator:
    """
    Simulate statechart execution to predict coverage.

    The statechart is built from SAE features:
    - States = feature configurations
    - Transitions = observed changes in features
    - Guards = synthesized from transition patterns
    """

    def __init__(self):
        # States and transitions
        self.states: Set[FrozenSet[int]] = set()
        self.transitions: List[StatechartTransition] = []

        # State to code mapping (for actual coverage prediction)
        self.state_to_lines: Dict[FrozenSet[int], Set[int]] = defaultdict(set)
        self.feature_to_lines: Dict[int, Set[int]] = defaultdict(set)

        # Initial state
        self.initial_state: Optional[FrozenSet[int]] = None

    def add_state(self, config: FrozenSet[int], lines: Set[int] = None):
        """Add a state to the statechart."""
        self.states.add(config)
        if lines:
            self.state_to_lines[config] = lines

    def add_transition(
        self,
        source: FrozenSet[int],
        target: FrozenSet[int],
        guard: FeatureGuard = None,
        probability: float = 1.0,
    ):
        """Add a transition."""
        trans = StatechartTransition(
            source=source,
            target=target,
            guard=guard,
            probability=probability,
        )
        self.transitions.append(trans)

        # Ensure states exist
        self.states.add(source)
        self.states.add(target)

    def build_from_observations(
        self,
        configurations: List[StateConfiguration],
        min_transition_count: int = 1,
    ):
        """
        Build statechart from observed configurations.

        Args:
            configurations: Sequence of observed configurations
            min_transition_count: Minimum times a transition must be seen
        """
        # Count transitions
        trans_counts: Dict[Tuple[FrozenSet[int], FrozenSet[int]], int] = defaultdict(int)

        for i in range(len(configurations) - 1):
            src = configurations[i].active_features
            tgt = configurations[i + 1].active_features
            trans_counts[(src, tgt)] += 1
            self.states.add(src)
            self.states.add(tgt)

        # Set initial state
        if configurations:
            self.initial_state = configurations[0].active_features

        # Add transitions with sufficient count
        for (src, tgt), count in trans_counts.items():
            if count >= min_transition_count:
                self.add_transition(src, tgt)
                self.transitions[-1].count = count

    def simulate(
        self,
        initial_pattern: ActivationPattern,
        max_steps: int = 100,
    ) -> SimulationResult:
        """
        Simulate statechart execution.

        Args:
            initial_pattern: Starting feature activation pattern
            max_steps: Maximum simulation steps

        Returns:
            SimulationResult with visited states/features
        """
        current_state = initial_pattern.active_features
        visited_states: Set[FrozenSet[int]] = {current_state}
        visited_features: Set[int] = set(current_state)
        transitions_taken: List[StatechartTransition] = []

        for step in range(max_steps):
            # Find enabled transitions from current state
            enabled = []
            for trans in self.transitions:
                if trans.source == current_state:
                    pattern = ActivationPattern(
                        active_features=current_state,
                        activations={f: 1.0 for f in current_state},
                        label=True,
                    )
                    if trans.is_enabled(pattern):
                        enabled.append(trans)

            if not enabled:
                # No enabled transitions - stuck
                break

            # Take first enabled transition (deterministic)
            # For probabilistic, sample based on probability
            trans = enabled[0]
            transitions_taken.append(trans)

            # Move to target state
            current_state = trans.target
            visited_states.add(current_state)
            visited_features.update(current_state)

            # Check if we've seen this state before (loop detection)
            if len(visited_states) > len(self.states):
                break

        return SimulationResult(
            visited_states=visited_states,
            visited_features=visited_features,
            transitions_taken=transitions_taken,
            steps=len(transitions_taken),
            terminated_normally=True,
        )

    def simulate_all_paths(
        self,
        initial_pattern: ActivationPattern,
        max_depth: int = 10,
    ) -> SimulationResult:
        """
        Simulate all possible paths (for small statecharts).

        Uses BFS to explore all reachable states.
        """
        visited_states: Set[FrozenSet[int]] = set()
        visited_features: Set[int] = set()
        transitions_taken: List[StatechartTransition] = []

        # BFS
        queue = [(initial_pattern.active_features, 0)]

        while queue:
            current_state, depth = queue.pop(0)

            if depth > max_depth:
                continue

            if current_state in visited_states:
                continue

            visited_states.add(current_state)
            visited_features.update(current_state)

            # Find all enabled transitions
            for trans in self.transitions:
                if trans.source == current_state:
                    pattern = ActivationPattern(
                        active_features=current_state,
                        activations={f: 1.0 for f in current_state},
                        label=True,
                    )
                    if trans.is_enabled(pattern):
                        transitions_taken.append(trans)
                        queue.append((trans.target, depth + 1))

        return SimulationResult(
            visited_states=visited_states,
            visited_features=visited_features,
            transitions_taken=transitions_taken,
            steps=len(transitions_taken),
            terminated_normally=True,
        )

    def predict_coverage(
        self,
        initial_pattern: ActivationPattern,
        use_all_paths: bool = False,
    ) -> CoveragePrediction:
        """
        Predict coverage from initial state.

        Args:
            initial_pattern: Starting feature activations
            use_all_paths: If True, explore all paths (slower but complete)

        Returns:
            CoveragePrediction with predicted features/states
        """
        if use_all_paths:
            result = self.simulate_all_paths(initial_pattern)
        else:
            result = self.simulate(initial_pattern)

        # Convert to coverage prediction
        return CoveragePrediction(
            predicted_features=result.visited_features,
            predicted_states=result.visited_states,
            confidence=1.0 if result.terminated_normally else 0.5,
        )

    def predict_line_coverage(
        self,
        initial_pattern: ActivationPattern,
    ) -> Set[int]:
        """
        Predict which lines will be covered.

        Requires state_to_lines or feature_to_lines mapping.
        """
        prediction = self.predict_coverage(initial_pattern)

        covered_lines = set()

        # From state mappings
        for state in prediction.predicted_states:
            covered_lines.update(self.state_to_lines.get(state, set()))

        # From feature mappings
        for feature in prediction.predicted_features:
            covered_lines.update(self.feature_to_lines.get(feature, set()))

        return covered_lines

    def evaluate_predictions(
        self,
        test_cases: List[Tuple[ActivationPattern, Set[int]]],
    ) -> Dict[str, float]:
        """
        Evaluate coverage predictions against ground truth.

        Args:
            test_cases: List of (initial_pattern, actual_covered_features)

        Returns:
            Dict with precision, recall, f1
        """
        total_precision = 0.0
        total_recall = 0.0
        count = 0

        for pattern, actual in test_cases:
            prediction = self.predict_coverage(pattern)
            prediction.actual_features = actual

            total_precision += prediction.feature_precision
            total_recall += prediction.feature_recall
            count += 1

        avg_precision = total_precision / count if count > 0 else 0.0
        avg_recall = total_recall / count if count > 0 else 0.0
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'n_test_cases': count,
        }

    def get_statistics(self) -> Dict:
        """Get simulator statistics."""
        return {
            'n_states': len(self.states),
            'n_transitions': len(self.transitions),
            'n_features': len(set(f for s in self.states for f in s)),
            'has_initial': self.initial_state is not None,
        }


def demo():
    """Demonstrate coverage simulation."""
    print("=" * 60)
    print("COVERAGE SIMULATOR: Coverage via Statechart Simulation")
    print("=" * 60)

    simulator = CoverageSimulator()

    # Build simple statechart
    # States are feature configurations
    state_A = frozenset({1, 2, 3})
    state_B = frozenset({2, 3, 4})
    state_C = frozenset({3, 4, 5})
    state_D = frozenset({4, 5, 6})

    simulator.add_state(state_A)
    simulator.add_state(state_B)
    simulator.add_state(state_C)
    simulator.add_state(state_D)

    simulator.add_transition(state_A, state_B)
    simulator.add_transition(state_B, state_C)
    simulator.add_transition(state_C, state_D)
    simulator.add_transition(state_B, state_D)  # Shortcut

    simulator.initial_state = state_A

    print(f"\nStatechart: {simulator.get_statistics()}")

    # Simulate from initial state
    initial = ActivationPattern(
        active_features=state_A,
        activations={f: 1.0 for f in state_A},
        label=True,
    )

    print("\nSimulating single path...")
    result = simulator.simulate(initial)
    print(f"  Steps: {result.steps}")
    print(f"  Visited states: {len(result.visited_states)}")
    print(f"  Visited features: {result.visited_features}")

    print("\nSimulating all paths...")
    result_all = simulator.simulate_all_paths(initial)
    print(f"  Visited states: {len(result_all.visited_states)}")
    print(f"  Visited features: {result_all.visited_features}")

    # Predict coverage
    print("\nPredicting coverage...")
    prediction = simulator.predict_coverage(initial, use_all_paths=True)
    print(f"  Predicted features: {prediction.predicted_features}")
    print(f"  Confidence: {prediction.confidence}")

    return simulator


if __name__ == "__main__":
    demo()
