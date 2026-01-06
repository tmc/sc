"""
Belief State: Parallel AND-States for Belief Tracking

KEY INSIGHT:
Statechart AND-states (orthogonal regions) naturally represent
factored belief states in partially observable domains.

Each parallel region tracks beliefs about ONE hidden dimension:
- Region 1: P(opponent_hand | observations)
- Region 2: P(opponent_strategy | observations)
- Region 3: P(deck_state | observations)

The CONFIGURATION of active states across regions IS the belief state.

This connects:
- POMDP belief tracking (ML)
- Statechart semantics (formal methods)
- Bayesian inference (probability)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum, auto
import math


class BeliefValue(Enum):
    """Discrete belief levels for a dimension."""
    VERY_LOW = 0     # P < 0.1
    LOW = 1          # 0.1 <= P < 0.3
    MEDIUM = 2       # 0.3 <= P < 0.7
    HIGH = 3         # 0.7 <= P < 0.9
    VERY_HIGH = 4    # P >= 0.9

    @staticmethod
    def from_probability(p: float) -> 'BeliefValue':
        """Convert probability to discrete belief level."""
        if p < 0.1:
            return BeliefValue.VERY_LOW
        elif p < 0.3:
            return BeliefValue.LOW
        elif p < 0.7:
            return BeliefValue.MEDIUM
        elif p < 0.9:
            return BeliefValue.HIGH
        else:
            return BeliefValue.VERY_HIGH

    def to_probability_range(self) -> Tuple[float, float]:
        """Get probability range for this belief level."""
        ranges = {
            BeliefValue.VERY_LOW: (0.0, 0.1),
            BeliefValue.LOW: (0.1, 0.3),
            BeliefValue.MEDIUM: (0.3, 0.7),
            BeliefValue.HIGH: (0.7, 0.9),
            BeliefValue.VERY_HIGH: (0.9, 1.0),
        }
        return ranges[self]

    def midpoint(self) -> float:
        """Get midpoint probability."""
        lo, hi = self.to_probability_range()
        return (lo + hi) / 2


@dataclass
class BeliefDimension:
    """
    One dimension of belief (one AND-state region).

    Tracks probability distribution over possible hidden values.
    The current "state" in this region reflects the most likely value.

    Example: OpponentHandStrength dimension
    - Possible values: weak, medium, strong
    - Distribution: [0.2, 0.3, 0.5]
    - Active state: "strong" (most likely)
    """
    name: str
    possible_values: List[str]
    distribution: List[float] = field(default_factory=list)

    # Statechart state naming
    state_prefix: str = ""

    def __post_init__(self):
        if not self.distribution:
            # Uniform prior
            n = len(self.possible_values)
            self.distribution = [1.0 / n] * n

        if not self.state_prefix:
            self.state_prefix = self.name

        # Normalize
        self._normalize()

    def _normalize(self):
        """Normalize distribution to sum to 1."""
        total = sum(self.distribution)
        if total > 0:
            self.distribution = [p / total for p in self.distribution]

    def get_state_names(self) -> List[str]:
        """Get statechart state names for this dimension."""
        return [f"{self.state_prefix}_{v}" for v in self.possible_values]

    def get_active_state(self) -> str:
        """Get currently active state (most likely value)."""
        max_idx = max(range(len(self.distribution)), key=lambda i: self.distribution[i])
        return f"{self.state_prefix}_{self.possible_values[max_idx]}"

    def get_probability(self, value: str) -> float:
        """Get probability of a specific value."""
        try:
            idx = self.possible_values.index(value)
            return self.distribution[idx]
        except ValueError:
            return 0.0

    def set_probability(self, value: str, prob: float):
        """Set probability for a value (will be normalized)."""
        try:
            idx = self.possible_values.index(value)
            self.distribution[idx] = prob
            self._normalize()
        except ValueError:
            pass

    def update_bayesian(self, likelihood: Dict[str, float]):
        """
        Bayesian update: P(value|obs) ∝ P(obs|value) * P(value)

        Args:
            likelihood: P(observation | value) for each value
        """
        for i, value in enumerate(self.possible_values):
            if value in likelihood:
                self.distribution[i] *= likelihood[value]
        self._normalize()

    def entropy(self) -> float:
        """Compute entropy of belief distribution."""
        h = 0.0
        for p in self.distribution:
            if p > 0:
                h -= p * math.log2(p)
        return h

    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        n = len(self.possible_values)
        return math.log2(n) if n > 0 else 0.0

    def certainty(self) -> float:
        """How certain are we? (1 - normalized entropy)"""
        max_h = self.max_entropy()
        if max_h == 0:
            return 1.0
        return 1.0 - self.entropy() / max_h

    def copy(self) -> 'BeliefDimension':
        """Deep copy."""
        return BeliefDimension(
            name=self.name,
            possible_values=list(self.possible_values),
            distribution=list(self.distribution),
            state_prefix=self.state_prefix,
        )


@dataclass
class BeliefConfiguration:
    """
    Configuration of active states across all belief dimensions.

    This IS the joint belief state, represented as a statechart configuration.
    In AND-state semantics: ALL regions are simultaneously active.
    """
    active_states: FrozenSet[str]
    probabilities: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.active_states)

    def __eq__(self, other):
        if not isinstance(other, BeliefConfiguration):
            return False
        return self.active_states == other.active_states

    def matches_pattern(self, pattern: Set[str]) -> bool:
        """Check if configuration matches a pattern (subset check)."""
        return pattern.issubset(self.active_states)

    def to_vector(self, all_states: List[str]) -> List[float]:
        """Convert to probability vector."""
        return [self.probabilities.get(s, 0.0) for s in all_states]


@dataclass
class ParallelBeliefState:
    """
    Complete belief state using parallel AND-state regions.

    This is the core representation:
    - Multiple orthogonal regions (BeliefDimension)
    - Each region has its own state space and distribution
    - The joint configuration represents the complete belief

    Statechart structure:
    ```
    BeliefState (AND)
    ├── OpponentHand (OR): weak | medium | strong
    ├── OpponentStrategy (OR): passive | aggressive | bluffer
    └── DeckState (OR): high_rich | balanced | low_rich
    ```
    """
    name: str
    dimensions: List[BeliefDimension] = field(default_factory=list)

    def add_dimension(self, dim: BeliefDimension):
        """Add a belief dimension (parallel region)."""
        self.dimensions.append(dim)

    def get_configuration(self) -> BeliefConfiguration:
        """Get current belief configuration (active states across all regions)."""
        active = set()
        probs = {}

        for dim in self.dimensions:
            state = dim.get_active_state()
            active.add(state)
            probs[state] = max(dim.distribution)

        return BeliefConfiguration(
            active_states=frozenset(active),
            probabilities=probs,
        )

    def get_all_states(self) -> List[str]:
        """Get all possible states across all dimensions."""
        states = []
        for dim in self.dimensions:
            states.extend(dim.get_state_names())
        return states

    def update_dimension(self, dim_name: str, likelihood: Dict[str, float]):
        """Update a specific dimension with Bayesian update."""
        for dim in self.dimensions:
            if dim.name == dim_name:
                dim.update_bayesian(likelihood)
                return

    def reset_to_uniform(self):
        """Reset all dimensions to uniform prior."""
        for dim in self.dimensions:
            n = len(dim.possible_values)
            dim.distribution = [1.0 / n] * n

    def total_entropy(self) -> float:
        """Sum of entropies across all dimensions."""
        return sum(dim.entropy() for dim in self.dimensions)

    def average_certainty(self) -> float:
        """Average certainty across dimensions."""
        if not self.dimensions:
            return 0.0
        return sum(dim.certainty() for dim in self.dimensions) / len(self.dimensions)

    def to_statechart_dict(self) -> Dict:
        """Convert to statechart proto-like dictionary."""
        children = []

        for dim in self.dimensions:
            # Each dimension is an OR state
            dim_children = []
            for i, value in enumerate(dim.possible_values):
                state_name = f"{dim.state_prefix}_{value}"
                is_active = dim.distribution[i] == max(dim.distribution)
                dim_children.append({
                    'label': state_name,
                    'type': 'BASIC',
                    'is_initial': is_active,
                    'metadata': {'probability': dim.distribution[i]},
                })

            children.append({
                'label': dim.name,
                'type': 'OR',  # XOR semantics within dimension
                'children': dim_children,
            })

        return {
            'root_state': {
                'label': self.name,
                'type': 'AND',  # Parallel regions
                'children': children,
            }
        }

    def copy(self) -> 'ParallelBeliefState':
        """Deep copy."""
        return ParallelBeliefState(
            name=self.name,
            dimensions=[dim.copy() for dim in self.dimensions],
        )


def create_poker_belief_state() -> ParallelBeliefState:
    """
    Create belief state for poker with hidden cards.

    Dimensions:
    1. Opponent hand strength (what cards do they have?)
    2. Opponent strategy (how do they play?)
    3. Deck state (what's left in the deck?)
    """
    belief = ParallelBeliefState(name="PokerBelief")

    # Dimension 1: Opponent hand strength
    belief.add_dimension(BeliefDimension(
        name="opponent_hand",
        possible_values=["weak", "medium", "strong"],
        distribution=[0.4, 0.35, 0.25],  # Prior: weak more likely
        state_prefix="hand",
    ))

    # Dimension 2: Opponent strategy
    belief.add_dimension(BeliefDimension(
        name="opponent_strategy",
        possible_values=["passive", "aggressive", "bluffer"],
        distribution=[0.4, 0.4, 0.2],  # Prior: bluffers less common
        state_prefix="strategy",
    ))

    # Dimension 3: Deck state
    belief.add_dimension(BeliefDimension(
        name="deck_state",
        possible_values=["high_rich", "balanced", "low_rich"],
        distribution=[0.33, 0.34, 0.33],  # Uniform prior
        state_prefix="deck",
    ))

    return belief


def demo():
    """Demonstrate belief state representation."""
    print("=" * 60)
    print("BELIEF STATE: Parallel AND-States for POMDP")
    print("=" * 60)

    belief = create_poker_belief_state()

    print(f"\nInitial belief state: {belief.name}")
    print(f"Dimensions: {len(belief.dimensions)}")

    for dim in belief.dimensions:
        print(f"\n  {dim.name}:")
        for i, v in enumerate(dim.possible_values):
            print(f"    {v}: {dim.distribution[i]:.3f}")
        print(f"    Active state: {dim.get_active_state()}")
        print(f"    Certainty: {dim.certainty():.3f}")

    print(f"\nConfiguration: {belief.get_configuration().active_states}")
    print(f"Average certainty: {belief.average_certainty():.3f}")

    # Simulate observation: opponent raises (suggests strong hand or bluff)
    print("\n--- Observation: Opponent RAISES ---")
    belief.update_dimension("opponent_hand", {
        "weak": 0.3,    # Less likely (unless bluffing)
        "medium": 0.5,  # Possible
        "strong": 0.9,  # Most likely
    })
    belief.update_dimension("opponent_strategy", {
        "passive": 0.2,     # Unlikely
        "aggressive": 0.7,  # Likely
        "bluffer": 0.5,     # Possible
    })

    print("\nUpdated beliefs:")
    for dim in belief.dimensions:
        print(f"  {dim.name}: {dim.get_active_state()} (certainty: {dim.certainty():.3f})")

    print(f"\nNew configuration: {belief.get_configuration().active_states}")

    # Convert to statechart
    print("\nStatechart representation:")
    sc = belief.to_statechart_dict()
    print(f"  Root: {sc['root_state']['label']} (type: {sc['root_state']['type']})")
    for child in sc['root_state']['children']:
        print(f"    Region: {child['label']} (type: {child['type']})")

    return belief


if __name__ == "__main__":
    demo()
