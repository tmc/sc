"""
Belief Policy: Actions Conditioned on Belief State

Implements policies that condition on belief configurations:
1. BeliefConditionedAction - Rules mapping belief patterns to actions
2. EvolvedBeliefPolicy - Evolved policy over belief space
3. PolicyGenome - Evolvable belief-action mappings

KEY INSIGHT: Policy is a statechart transition guard over belief dimensions.
Guard: "if opponent_hand=strong AND opponent_strategy=aggressive"
Action: "fold"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
import random
import math

from .belief_state import (
    BeliefConfiguration,
    ParallelBeliefState,
    BeliefDimension,
)
from .poker_beliefs import (
    PokerBeliefState,
    PokerAction,
    HandStrength,
)


class ActionType(Enum):
    """Generic action types for belief-conditioned policy."""
    PASSIVE = "passive"       # Check, call
    AGGRESSIVE = "aggressive" # Bet, raise
    FOLD = "fold"            # Give up
    TRAP = "trap"            # Check strong hands
    BLUFF = "bluff"          # Bet weak hands


@dataclass
class BeliefPattern:
    """
    A pattern over belief dimensions.

    Matches belief configurations where specified dimensions
    are in specified states.

    Example:
    - {"opponent_hand": ["strong", "very_strong"], "opponent_strategy": ["aggressive"]}
    - Matches when opponent likely has strong hand AND plays aggressively
    """
    dimension_values: Dict[str, Set[str]] = field(default_factory=dict)
    min_certainty: float = 0.0  # Minimum certainty to match

    def matches(self, belief: ParallelBeliefState) -> bool:
        """Check if belief matches this pattern."""
        # Check certainty threshold
        if belief.average_certainty() < self.min_certainty:
            return False

        # Check each dimension constraint
        for dim in belief.dimensions:
            if dim.name in self.dimension_values:
                required_values = self.dimension_values[dim.name]
                active_state = dim.get_active_state()

                # Extract value from state name (e.g., "hand_strong" -> "strong")
                active_value = active_state.split('_', 1)[-1] if '_' in active_state else active_state

                if active_value not in required_values:
                    return False

        return True

    def to_guard_expr(self) -> str:
        """Convert to statechart guard expression."""
        clauses = []
        for dim_name, values in self.dimension_values.items():
            value_list = " OR ".join([f"{dim_name}={v}" for v in values])
            clauses.append(f"({value_list})")

        guard = " AND ".join(clauses)
        if self.min_certainty > 0:
            guard += f" AND certainty >= {self.min_certainty}"

        return guard


@dataclass
class BeliefConditionedAction:
    """
    A rule mapping belief pattern to action.

    This is a statechart transition in the belief-conditioned policy:
    - Source: any belief configuration matching pattern
    - Guard: pattern + conditions
    - Action: selected action
    """
    pattern: BeliefPattern
    action: ActionType
    priority: int = 0  # Higher priority rules checked first
    weight: float = 1.0  # Fitness weight for evolution

    def applies(self, belief: ParallelBeliefState) -> bool:
        """Check if this rule applies to current belief."""
        return self.pattern.matches(belief)

    def to_transition_dict(self) -> Dict:
        """Convert to statechart transition format."""
        return {
            'guard': self.pattern.to_guard_expr(),
            'action': self.action.value,
            'priority': self.priority,
        }


class BeliefPolicy:
    """
    Policy that selects actions based on belief configuration.

    Uses a list of BeliefConditionedAction rules, checked in priority order.
    First matching rule determines action.
    """

    def __init__(self, name: str = "BeliefPolicy"):
        self.name = name
        self.rules: List[BeliefConditionedAction] = []
        self.default_action: ActionType = ActionType.PASSIVE

    def add_rule(self, rule: BeliefConditionedAction):
        """Add a belief-conditioned action rule."""
        self.rules.append(rule)
        # Keep sorted by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def select_action(
        self,
        belief: ParallelBeliefState,
        legal_actions: Optional[Set[ActionType]] = None,
    ) -> ActionType:
        """Select action based on current belief."""
        # Check rules in priority order
        for rule in self.rules:
            if rule.applies(belief):
                if legal_actions is None or rule.action in legal_actions:
                    return rule.action

        return self.default_action

    def to_statechart(self) -> Dict:
        """Convert policy to statechart with belief-guarded transitions."""
        transitions = []

        for i, rule in enumerate(self.rules):
            transitions.append({
                'id': f"rule_{i}",
                'guard': rule.pattern.to_guard_expr(),
                'action': rule.action.value,
                'priority': rule.priority,
            })

        return {
            'name': self.name,
            'type': 'policy',
            'transitions': transitions,
            'default_action': self.default_action.value,
        }


@dataclass
class PolicyGenome:
    """
    Evolvable belief-conditioned policy.

    Genes:
    - Pattern thresholds for each dimension
    - Action probabilities for each belief region
    - Certainty thresholds for decision making
    """
    # Action probabilities by belief region
    # region_key = (hand_category, strategy_category)
    action_probs: Dict[Tuple[str, str], Dict[ActionType, float]] = field(default_factory=dict)

    # Certainty threshold for aggressive play
    certainty_threshold: float = 0.6

    # Bluff frequency when opponent seems weak
    bluff_frequency: float = 0.2

    # Value bet threshold (expected opponent strength below this)
    value_threshold: float = 0.4

    # Trap threshold (our strength above this, opponent strength below)
    trap_threshold: float = 0.7

    # Fitness tracking
    fitness: float = 0.0
    total_hands: int = 0
    total_profit: float = 0.0

    def __post_init__(self):
        # Initialize default action probabilities
        if not self.action_probs:
            self._init_default_probs()

    def _init_default_probs(self):
        """Initialize with reasonable defaults."""
        hand_cats = ["weak", "medium", "strong"]
        strategy_cats = ["passive", "aggressive", "bluffer"]

        for h in hand_cats:
            for s in strategy_cats:
                key = (h, s)

                if h == "strong":
                    # Strong opponent: be careful
                    self.action_probs[key] = {
                        ActionType.PASSIVE: 0.5,
                        ActionType.AGGRESSIVE: 0.2,
                        ActionType.FOLD: 0.25,
                        ActionType.TRAP: 0.05,
                        ActionType.BLUFF: 0.0,
                    }
                elif h == "weak":
                    # Weak opponent: be aggressive
                    self.action_probs[key] = {
                        ActionType.PASSIVE: 0.2,
                        ActionType.AGGRESSIVE: 0.5,
                        ActionType.FOLD: 0.05,
                        ActionType.TRAP: 0.1,
                        ActionType.BLUFF: 0.15,
                    }
                else:
                    # Medium: balanced
                    self.action_probs[key] = {
                        ActionType.PASSIVE: 0.35,
                        ActionType.AGGRESSIVE: 0.35,
                        ActionType.FOLD: 0.15,
                        ActionType.TRAP: 0.1,
                        ActionType.BLUFF: 0.05,
                    }

    def get_action(
        self,
        belief: PokerBeliefState,
        our_strength: float,
    ) -> ActionType:
        """Select action based on belief state and our hand strength."""
        # Get belief categories
        opp_strength = belief.expected_opponent_strength()
        opp_bluffer = belief.opponent_strategy.is_likely_bluffer()
        certainty = belief.certainty_score()

        # Determine opponent hand category
        if opp_strength < 0.35:
            hand_cat = "weak"
        elif opp_strength < 0.65:
            hand_cat = "medium"
        else:
            hand_cat = "strong"

        # Determine opponent strategy category
        strategy_dim = belief.opponent_strategy
        strategy_state = strategy_dim.get_active_state()
        if "aggressive" in strategy_state or "maniac" in strategy_state:
            strat_cat = "aggressive"
        elif "bluff" in strategy_state:
            strat_cat = "bluffer"
        else:
            strat_cat = "passive"

        # Special cases
        # 1. High certainty opponent is strong and we're weak -> fold
        if certainty > self.certainty_threshold and hand_cat == "strong" and our_strength < 0.3:
            return ActionType.FOLD

        # 2. Opponent likely bluffing and we're decent -> call/trap
        if opp_bluffer and our_strength > 0.5:
            if random.random() < 0.3:
                return ActionType.TRAP
            return ActionType.PASSIVE

        # 3. Our hand is strong -> value bet
        if our_strength > self.trap_threshold:
            if certainty < self.certainty_threshold:
                # Uncertain about opponent: bet for value
                return ActionType.AGGRESSIVE
            elif hand_cat == "weak":
                return ActionType.AGGRESSIVE  # Value bet
            else:
                # Strong opponent with strong hand: trap sometimes
                if random.random() < 0.3:
                    return ActionType.TRAP
                return ActionType.AGGRESSIVE

        # 4. Bluff opportunity
        if our_strength < 0.3 and hand_cat == "weak" and random.random() < self.bluff_frequency:
            return ActionType.BLUFF

        # 5. Default: use action probability table
        key = (hand_cat, strat_cat)
        if key in self.action_probs:
            probs = self.action_probs[key]
            r = random.random()
            cumulative = 0.0
            for action, prob in probs.items():
                cumulative += prob
                if r < cumulative:
                    return action

        return ActionType.PASSIVE

    def copy(self) -> 'PolicyGenome':
        """Deep copy."""
        new = PolicyGenome(
            certainty_threshold=self.certainty_threshold,
            bluff_frequency=self.bluff_frequency,
            value_threshold=self.value_threshold,
            trap_threshold=self.trap_threshold,
        )
        new.action_probs = {k: dict(v) for k, v in self.action_probs.items()}
        return new


def mutate_policy(genome: PolicyGenome) -> PolicyGenome:
    """Mutate a policy genome."""
    new = genome.copy()

    # Mutate thresholds
    if random.random() < 0.3:
        new.certainty_threshold = max(0.3, min(0.9, new.certainty_threshold + random.gauss(0, 0.1)))
    if random.random() < 0.3:
        new.bluff_frequency = max(0.0, min(0.5, new.bluff_frequency + random.gauss(0, 0.1)))
    if random.random() < 0.3:
        new.value_threshold = max(0.2, min(0.6, new.value_threshold + random.gauss(0, 0.1)))
    if random.random() < 0.3:
        new.trap_threshold = max(0.5, min(0.9, new.trap_threshold + random.gauss(0, 0.1)))

    # Mutate action probabilities
    for key in new.action_probs:
        if random.random() < 0.4:
            probs = new.action_probs[key]
            action = random.choice(list(probs.keys()))
            probs[action] = max(0.0, min(1.0, probs[action] + random.gauss(0, 0.15)))

            # Renormalize
            total = sum(probs.values())
            if total > 0:
                new.action_probs[key] = {k: v/total for k, v in probs.items()}

    return new


def crossover_policy(g1: PolicyGenome, g2: PolicyGenome) -> PolicyGenome:
    """Crossover two policy genomes."""
    new = PolicyGenome(
        certainty_threshold=random.choice([g1.certainty_threshold, g2.certainty_threshold]),
        bluff_frequency=random.choice([g1.bluff_frequency, g2.bluff_frequency]),
        value_threshold=random.choice([g1.value_threshold, g2.value_threshold]),
        trap_threshold=random.choice([g1.trap_threshold, g2.trap_threshold]),
    )

    # Crossover action probabilities
    for key in g1.action_probs:
        if key in g2.action_probs:
            new.action_probs[key] = random.choice([g1.action_probs[key], g2.action_probs[key]]).copy()
        else:
            new.action_probs[key] = g1.action_probs[key].copy()

    return new


class EvolvedBeliefPolicy:
    """
    Evolve belief-conditioned policies through self-play.

    Evolution discovers optimal belief → action mappings.
    """

    def __init__(
        self,
        population_size: int = 30,
        n_generations: int = 50,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.population: List[PolicyGenome] = []
        self.best_genome: Optional[PolicyGenome] = None

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            genome = PolicyGenome(
                certainty_threshold=random.uniform(0.4, 0.8),
                bluff_frequency=random.uniform(0.05, 0.3),
                value_threshold=random.uniform(0.3, 0.5),
                trap_threshold=random.uniform(0.6, 0.85),
            )
            genome._init_default_probs()

            # Add some random variation
            for key in genome.action_probs:
                for action in genome.action_probs[key]:
                    genome.action_probs[key][action] *= random.uniform(0.5, 1.5)

                # Renormalize
                total = sum(genome.action_probs[key].values())
                if total > 0:
                    genome.action_probs[key] = {
                        k: v/total for k, v in genome.action_probs[key].items()
                    }

            self.population.append(genome)

    def evaluate_against_opponent(
        self,
        genome: PolicyGenome,
        opponent: PolicyGenome,
        n_hands: int = 50,
    ) -> float:
        """Evaluate genome against an opponent."""
        total_profit = 0.0

        for _ in range(n_hands):
            # Create belief states
            our_belief = PokerBeliefState()
            opp_belief = PokerBeliefState()

            # Random hand strengths
            our_strength = random.random()
            opp_strength = random.random()

            # Simulate a betting round
            pot = 2  # Antes
            our_bet = 1
            opp_bet = 1

            # Our action
            our_action = genome.get_action(our_belief, our_strength)

            if our_action == ActionType.FOLD:
                total_profit -= our_bet
                continue

            if our_action in [ActionType.AGGRESSIVE, ActionType.BLUFF]:
                bet_size = random.choice([2, 4])  # Small or big bet
                pot += bet_size
                our_bet += bet_size

                # Opponent observes and updates belief
                opp_belief.observe_action(PokerAction.BET_BIG if bet_size > 2 else PokerAction.BET_SMALL)

            # Opponent's action
            opp_action = opponent.get_action(opp_belief, opp_strength)

            if opp_action == ActionType.FOLD:
                total_profit += pot - our_bet
                continue

            if opp_action in [ActionType.AGGRESSIVE, ActionType.BLUFF]:
                bet_size = random.choice([2, 4])
                pot += bet_size
                opp_bet += bet_size

                # We observe
                our_belief.observe_action(PokerAction.RAISE if our_bet > 1 else PokerAction.BET_BIG)

            # Showdown
            if our_strength > opp_strength:
                total_profit += pot - our_bet
            elif opp_strength > our_strength:
                total_profit -= our_bet
            # Tie: no change

        return total_profit / n_hands

    def evolve(self, verbose: bool = True) -> PolicyGenome:
        """Evolve belief-conditioned policies."""
        if not self.population:
            self.initialize_population()

        if verbose:
            print("=" * 60)
            print("EVOLVING BELIEF-CONDITIONED POLICIES")
            print("=" * 60)

        for gen in range(self.n_generations):
            # Evaluate against population
            for genome in self.population:
                opponents = random.sample(self.population, min(5, len(self.population)))
                genome.fitness = sum(
                    self.evaluate_against_opponent(genome, opp)
                    for opp in opponents
                ) / len(opponents)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].copy()
                self.best_genome.fitness = self.population[0].fitness

            if verbose and (gen % 10 == 0 or gen == self.n_generations - 1):
                best = self.population[0]
                print(f"Gen {gen:3d} | Best: {best.fitness:+.3f} "
                      f"| Bluff: {best.bluff_frequency:.2f} "
                      f"| Certainty: {best.certainty_threshold:.2f}")

            # Selection and reproduction
            elite_size = max(2, self.population_size // 10)
            new_pop = [g.copy() for g in self.population[:elite_size]]

            while len(new_pop) < self.population_size:
                if random.random() < 0.7:
                    p1 = random.choice(self.population[:self.population_size // 2])
                    p2 = random.choice(self.population[:self.population_size // 2])
                    child = crossover_policy(p1, p2)
                else:
                    parent = random.choice(self.population[:self.population_size // 2])
                    child = parent.copy()

                child = mutate_policy(child)
                new_pop.append(child)

            self.population = new_pop

        if verbose:
            print(f"\nBest policy:")
            print(f"  Certainty threshold: {self.best_genome.certainty_threshold:.2f}")
            print(f"  Bluff frequency: {self.best_genome.bluff_frequency:.2f}")
            print(f"  Value threshold: {self.best_genome.value_threshold:.2f}")
            print(f"  Trap threshold: {self.best_genome.trap_threshold:.2f}")
            print(f"  Fitness: {self.best_genome.fitness:+.3f}")

        return self.best_genome


def demo():
    """Demonstrate belief-conditioned policy."""
    print("=" * 60)
    print("BELIEF POLICY: Conditioned on Parallel AND-States")
    print("=" * 60)

    # Create belief state
    belief = PokerBeliefState()

    # Create policy
    policy = PolicyGenome()

    print("\n--- Initial Policy ---")
    print(f"Certainty threshold: {policy.certainty_threshold:.2f}")
    print(f"Bluff frequency: {policy.bluff_frequency:.2f}")

    # Simulate decisions
    print("\n--- Decision Simulation ---")

    scenarios = [
        ("Our hand weak (0.2), opponent unknown", 0.2, []),
        ("Our hand strong (0.8), opponent bet big", 0.8, [PokerAction.BET_BIG]),
        ("Our hand medium (0.5), opponent raised twice", 0.5, [PokerAction.RAISE, PokerAction.RAISE]),
        ("Our hand weak (0.1), opponent checked", 0.1, [PokerAction.CHECK]),
    ]

    for desc, our_strength, opp_actions in scenarios:
        belief.reset()
        for action in opp_actions:
            belief.observe_action(action)

        action = policy.get_action(belief, our_strength)
        print(f"\n{desc}")
        print(f"  Expected opponent strength: {belief.expected_opponent_strength():.2f}")
        print(f"  Likely bluffing: {belief.opponent_is_likely_bluffing()}")
        print(f"  Selected action: {action.value}")

    # Evolve policies
    print("\n--- Evolution ---")
    evolver = EvolvedBeliefPolicy(population_size=20, n_generations=30)
    best = evolver.evolve(verbose=True)

    return best


if __name__ == "__main__":
    demo()
