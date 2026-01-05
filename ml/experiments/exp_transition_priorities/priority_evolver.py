"""
Transition Priority Evolution

Evolves conflict resolution strategies when multiple transitions are enabled.
The Transition.priority field exists but is unused - this experiment learns
optimal priority assignments via evolution.

Two competing strategies:
1. Explicit Priority: Numeric priority per transition, highest wins
2. Specificity Priority: More specific guards (tighter conditions) win

Key insight: The "right" priority scheme depends on the domain.
Evolution discovers what works without hardcoding.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
from collections import defaultdict
from abc import ABC, abstractmethod
import random
import copy
import math


# =============================================================================
# Core Types
# =============================================================================

@dataclass
class Guard:
    """Guard condition with specificity tracking."""
    predicate: str
    params: Dict[str, Any] = field(default_factory=dict)

    # Learned specificity score (how narrow is this condition?)
    _specificity: Optional[float] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate guard in context."""
        grid = context.get("grid")
        state = context.get("state", {})

        if self.predicate == "true":
            return True
        elif self.predicate == "false":
            return False
        elif self.predicate == "has_nonzero":
            return bool(mx.any(grid != 0)) if grid is not None else False
        elif self.predicate == "all_zero":
            return bool(mx.all(grid == 0)) if grid is not None else True
        elif self.predicate == "count_gt":
            threshold = self.params.get("threshold", 0)
            return bool(mx.sum(grid != 0) > threshold) if grid is not None else False
        elif self.predicate == "count_lt":
            threshold = self.params.get("threshold", 5)
            return bool(mx.sum(grid != 0) < threshold) if grid is not None else True
        elif self.predicate == "has_color":
            color = self.params.get("color", 1)
            return bool(mx.any(grid == color)) if grid is not None else False
        elif self.predicate == "state_eq":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key) == value
        elif self.predicate == "state_gt":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key, 0) > value
        elif self.predicate == "state_lt":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key, 0) < value
        return True

    @property
    def specificity(self) -> float:
        """
        Compute guard specificity (how restrictive is this guard?).
        Higher = more specific = should have higher priority.
        """
        if self._specificity is not None:
            return self._specificity

        # Base specificity by predicate type
        specificity_map = {
            "true": 0.0,        # Always true = least specific
            "false": 1.0,       # Never true = most specific (but useless)
            "has_nonzero": 0.3,
            "all_zero": 0.3,
            "count_gt": 0.5,
            "count_lt": 0.5,
            "has_color": 0.6,   # Checks specific color
            "state_eq": 0.8,    # Exact match = very specific
            "state_gt": 0.5,
            "state_lt": 0.5,
        }

        base = specificity_map.get(self.predicate, 0.5)

        # Adjust by parameters (tighter thresholds = more specific)
        if "threshold" in self.params:
            # Higher threshold for count_gt = more specific
            # Lower threshold for count_lt = more specific
            t = self.params["threshold"]
            if self.predicate == "count_gt":
                base += min(0.3, t * 0.05)
            elif self.predicate == "count_lt":
                base += min(0.3, (10 - t) * 0.05)

        return min(1.0, base)


@dataclass
class Transition:
    """A transition with evolvable priority."""
    source: str
    target: str
    guard: Guard

    # Priority assignment (evolved)
    priority: float = 0.0

    # For tracking
    fire_count: int = 0
    conflict_wins: int = 0
    conflict_losses: int = 0

    def is_enabled(self, context: Dict[str, Any]) -> bool:
        """Check if transition is enabled."""
        return self.guard.evaluate(context)

    def effective_priority(self, strategy: "PriorityStrategy") -> float:
        """Get effective priority under given strategy."""
        return strategy.compute_priority(self)


# =============================================================================
# Priority Strategies (Evolvable)
# =============================================================================

class PriorityStrategy(ABC):
    """Base class for priority assignment strategies."""

    @abstractmethod
    def compute_priority(self, transition: Transition) -> float:
        """Compute effective priority for a transition."""
        pass

    @abstractmethod
    def mutate(self) -> "PriorityStrategy":
        """Create mutated copy of this strategy."""
        pass

    @abstractmethod
    def clone(self) -> "PriorityStrategy":
        """Create exact copy."""
        pass


@dataclass
class ExplicitPriorityStrategy(PriorityStrategy):
    """
    Explicit numeric priorities.
    Each transition has a priority value; highest wins.
    """
    # Noise added during selection (for exploration)
    selection_noise: float = 0.1

    def compute_priority(self, transition: Transition) -> float:
        """Use transition's explicit priority field."""
        noise = random.gauss(0, self.selection_noise)
        return transition.priority + noise

    def mutate(self) -> "ExplicitPriorityStrategy":
        """Mutate selection noise."""
        new_noise = max(0.01, self.selection_noise + random.gauss(0, 0.05))
        return ExplicitPriorityStrategy(selection_noise=new_noise)

    def clone(self) -> "ExplicitPriorityStrategy":
        return ExplicitPriorityStrategy(selection_noise=self.selection_noise)


@dataclass
class SpecificityPriorityStrategy(PriorityStrategy):
    """
    Specificity-based priorities.
    More specific guards (tighter conditions) get higher priority.
    """
    # Weight for specificity vs explicit priority
    specificity_weight: float = 1.0
    explicit_weight: float = 0.0

    def compute_priority(self, transition: Transition) -> float:
        """Combine specificity and explicit priority."""
        spec = transition.guard.specificity
        explicit = transition.priority
        return (self.specificity_weight * spec +
                self.explicit_weight * explicit)

    def mutate(self) -> "SpecificityPriorityStrategy":
        """Mutate weights."""
        new_spec = max(0, self.specificity_weight + random.gauss(0, 0.2))
        new_exp = max(0, self.explicit_weight + random.gauss(0, 0.2))
        # Normalize
        total = new_spec + new_exp + 0.001
        return SpecificityPriorityStrategy(
            specificity_weight=new_spec / total,
            explicit_weight=new_exp / total
        )

    def clone(self) -> "SpecificityPriorityStrategy":
        return SpecificityPriorityStrategy(
            specificity_weight=self.specificity_weight,
            explicit_weight=self.explicit_weight
        )


@dataclass
class LearnedPriorityStrategy(PriorityStrategy):
    """
    Learned priority function.
    Evolves weights for multiple factors.
    """
    # Weights for different factors
    explicit_weight: float = 0.5
    specificity_weight: float = 0.5
    fire_count_weight: float = 0.0  # Prefer frequently-fired?
    recency_weight: float = 0.0     # Prefer recently-fired?

    def compute_priority(self, transition: Transition) -> float:
        """Weighted combination of factors."""
        factors = [
            (self.explicit_weight, transition.priority),
            (self.specificity_weight, transition.guard.specificity),
            (self.fire_count_weight, math.log(1 + transition.fire_count) / 10),
            (self.recency_weight, 0.5),  # Would need recency tracking
        ]

        return sum(w * v for w, v in factors)

    def mutate(self) -> "LearnedPriorityStrategy":
        """Mutate all weights."""
        weights = [
            max(0, self.explicit_weight + random.gauss(0, 0.2)),
            max(0, self.specificity_weight + random.gauss(0, 0.2)),
            max(0, self.fire_count_weight + random.gauss(0, 0.1)),
            max(0, self.recency_weight + random.gauss(0, 0.1)),
        ]
        # Normalize
        total = sum(weights) + 0.001
        weights = [w / total for w in weights]

        return LearnedPriorityStrategy(
            explicit_weight=weights[0],
            specificity_weight=weights[1],
            fire_count_weight=weights[2],
            recency_weight=weights[3],
        )

    def clone(self) -> "LearnedPriorityStrategy":
        return LearnedPriorityStrategy(
            explicit_weight=self.explicit_weight,
            specificity_weight=self.specificity_weight,
            fire_count_weight=self.fire_count_weight,
            recency_weight=self.recency_weight,
        )


# =============================================================================
# Statechart with Priority Resolution
# =============================================================================

@dataclass
class PriorityStatechart:
    """
    Statechart that uses evolved priority strategies for conflict resolution.
    """
    states: Set[str] = field(default_factory=set)
    transitions: List[Transition] = field(default_factory=list)
    initial_state: str = "start"
    final_states: Set[str] = field(default_factory=set)

    # Current state
    current_state: str = ""

    # Priority strategy (evolved)
    strategy: PriorityStrategy = field(default_factory=ExplicitPriorityStrategy)

    # Fitness tracking
    fitness: float = 0.0
    conflicts_resolved: int = 0

    def __post_init__(self):
        if not self.current_state:
            self.current_state = self.initial_state

    def reset(self):
        """Reset to initial state."""
        self.current_state = self.initial_state
        self.conflicts_resolved = 0

    def get_enabled_transitions(self, context: Dict[str, Any]) -> List[Transition]:
        """Get all transitions enabled from current state."""
        return [
            t for t in self.transitions
            if t.source == self.current_state and t.is_enabled(context)
        ]

    def resolve_conflict(self, enabled: List[Transition]) -> Transition:
        """
        Resolve conflict when multiple transitions are enabled.
        Uses the current priority strategy.
        """
        if len(enabled) == 1:
            return enabled[0]

        self.conflicts_resolved += 1

        # Compute priorities
        priorities = [
            (t, self.strategy.compute_priority(t))
            for t in enabled
        ]

        # Select highest priority
        winner = max(priorities, key=lambda x: x[1])[0]

        # Track statistics
        winner.conflict_wins += 1
        for t in enabled:
            if t != winner:
                t.conflict_losses += 1

        return winner

    def step(self, context: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Take one step in the statechart.

        Returns: (new_state, is_done)
        """
        enabled = self.get_enabled_transitions(context)

        if not enabled:
            # No transitions enabled - check if final
            return self.current_state, self.current_state in self.final_states

        # Resolve conflict if multiple enabled
        selected = self.resolve_conflict(enabled)
        selected.fire_count += 1

        # Execute transition
        self.current_state = selected.target

        return self.current_state, self.current_state in self.final_states

    def execute(self, context: Dict[str, Any], max_steps: int = 100) -> List[str]:
        """Execute until done, return state sequence."""
        self.reset()
        sequence = [self.current_state]

        for _ in range(max_steps):
            state, done = self.step(context)
            sequence.append(state)
            if done:
                break

        return sequence

    def clone(self) -> "PriorityStatechart":
        """Deep copy."""
        new_sc = PriorityStatechart(
            states=set(self.states),
            transitions=[
                Transition(
                    source=t.source,
                    target=t.target,
                    guard=Guard(
                        predicate=t.guard.predicate,
                        params=dict(t.guard.params)
                    ),
                    priority=t.priority,
                )
                for t in self.transitions
            ],
            initial_state=self.initial_state,
            final_states=set(self.final_states),
            strategy=self.strategy.clone(),
        )
        return new_sc


# =============================================================================
# Priority Evolver
# =============================================================================

class PriorityEvolver:
    """
    Evolves both transition priorities AND priority strategies.

    Two-level evolution:
    1. Inner: Evolve transition priority values
    2. Outer: Evolve which strategy to use (explicit vs specificity vs learned)
    """

    def __init__(
        self,
        population_size: int = 50,
        strategy_pool_size: int = 10,
        mutation_rate: float = 0.3,
        priority_mutation_std: float = 0.5,
    ):
        self.population_size = population_size
        self.strategy_pool_size = strategy_pool_size
        self.mutation_rate = mutation_rate
        self.priority_mutation_std = priority_mutation_std

        # Population of statecharts
        self.population: List[PriorityStatechart] = []

        # Pool of strategies
        self.strategy_pool: List[PriorityStrategy] = []
        self._init_strategy_pool()

        # Statistics
        self.generation = 0
        self.best_fitness = 0.0
        self.strategy_fitness: Dict[str, List[float]] = defaultdict(list)

    def _init_strategy_pool(self):
        """Initialize diverse strategy pool."""
        # Explicit priority variants
        for noise in [0.0, 0.1, 0.2, 0.5]:
            self.strategy_pool.append(
                ExplicitPriorityStrategy(selection_noise=noise)
            )

        # Specificity variants
        for spec_w in [1.0, 0.8, 0.5]:
            self.strategy_pool.append(
                SpecificityPriorityStrategy(
                    specificity_weight=spec_w,
                    explicit_weight=1.0 - spec_w
                )
            )

        # Learned variants
        self.strategy_pool.append(LearnedPriorityStrategy())
        self.strategy_pool.append(
            LearnedPriorityStrategy(
                explicit_weight=0.3,
                specificity_weight=0.7,
            )
        )

    def create_random_statechart(
        self,
        n_states: int = 5,
        n_transitions: int = 10,
    ) -> PriorityStatechart:
        """Create random statechart with conflicts."""
        states = {f"s{i}" for i in range(n_states)}
        states.add("start")
        states.add("end")

        # Predicates for guards
        predicates = [
            ("true", {}),
            ("has_nonzero", {}),
            ("all_zero", {}),
            ("count_gt", {"threshold": random.randint(1, 5)}),
            ("count_lt", {"threshold": random.randint(3, 8)}),
            ("has_color", {"color": random.randint(1, 3)}),
            ("state_eq", {"key": "x", "value": random.randint(0, 3)}),
            ("state_gt", {"key": "x", "value": random.randint(0, 2)}),
        ]

        transitions = []
        state_list = list(states - {"end"})

        for _ in range(n_transitions):
            source = random.choice(state_list)
            target = random.choice(list(states - {"start"}))
            pred, params = random.choice(predicates)

            # Random initial priority (will be evolved)
            priority = random.gauss(0, 1)

            transitions.append(Transition(
                source=source,
                target=target,
                guard=Guard(predicate=pred, params=params),
                priority=priority,
            ))

        # Ensure path to end exists
        for s in state_list:
            if not any(t.source == s and t.target == "end" for t in transitions):
                if random.random() < 0.3:
                    transitions.append(Transition(
                        source=s,
                        target="end",
                        guard=Guard(predicate="true"),
                        priority=random.gauss(0, 1),
                    ))

        # Select random strategy
        strategy = random.choice(self.strategy_pool).clone()

        return PriorityStatechart(
            states=states,
            transitions=transitions,
            initial_state="start",
            final_states={"end"},
            strategy=strategy,
        )

    def mutate_priorities(self, sc: PriorityStatechart) -> PriorityStatechart:
        """Mutate transition priorities."""
        new_sc = sc.clone()

        for trans in new_sc.transitions:
            if random.random() < self.mutation_rate:
                trans.priority += random.gauss(0, self.priority_mutation_std)

        return new_sc

    def mutate_strategy(self, sc: PriorityStatechart) -> PriorityStatechart:
        """Mutate the priority strategy."""
        new_sc = sc.clone()

        if random.random() < 0.3:
            # Switch to different strategy type
            new_sc.strategy = random.choice(self.strategy_pool).clone()
        else:
            # Mutate current strategy
            new_sc.strategy = new_sc.strategy.mutate()

        return new_sc

    def crossover_priorities(
        self,
        parent1: PriorityStatechart,
        parent2: PriorityStatechart
    ) -> PriorityStatechart:
        """Crossover transition priorities between parents."""
        child = parent1.clone()

        # Blend priorities
        for i, trans in enumerate(child.transitions):
            if i < len(parent2.transitions):
                alpha = random.random()
                trans.priority = (
                    alpha * trans.priority +
                    (1 - alpha) * parent2.transitions[i].priority
                )

        # Maybe take strategy from parent2
        if random.random() < 0.5:
            child.strategy = parent2.strategy.clone()

        return child

    def evaluate(
        self,
        sc: PriorityStatechart,
        test_cases: List[Tuple[Dict[str, Any], List[str]]]
    ) -> float:
        """
        Evaluate statechart on test cases.

        test_cases: List of (context, expected_state_sequence)
        """
        if not test_cases:
            return 0.0

        total_score = 0.0

        for context, expected in test_cases:
            actual = sc.execute(context)

            # Score: how much of expected sequence was matched?
            matches = sum(
                1 for a, e in zip(actual, expected) if a == e
            )
            max_len = max(len(actual), len(expected))

            # Bonus for reaching final state
            reached_end = actual[-1] in sc.final_states

            score = matches / max_len if max_len > 0 else 0
            if reached_end:
                score += 0.2

            total_score += score

        return total_score / len(test_cases)

    def evolve(
        self,
        test_cases: List[Tuple[Dict[str, Any], List[str]]],
        n_generations: int = 100,
        verbose: bool = True,
    ) -> PriorityStatechart:
        """
        Evolve optimal priority assignment.

        Args:
            test_cases: (context, expected_sequence) pairs
            n_generations: Evolution generations
            verbose: Print progress

        Returns:
            Best evolved statechart
        """
        # Initialize population
        if not self.population:
            self.population = [
                self.create_random_statechart()
                for _ in range(self.population_size)
            ]

        best_sc = None

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate
            for sc in self.population:
                sc.fitness = self.evaluate(sc, test_cases)

                # Track strategy performance
                strategy_name = type(sc.strategy).__name__
                self.strategy_fitness[strategy_name].append(sc.fitness)

                if sc.fitness > self.best_fitness:
                    self.best_fitness = sc.fitness
                    best_sc = sc.clone()

            if verbose and gen % 10 == 0:
                avg_conflicts = sum(
                    sc.conflicts_resolved for sc in self.population
                ) / len(self.population)
                print(f"Gen {gen}: best={self.best_fitness:.3f}, "
                      f"avg_conflicts={avg_conflicts:.1f}")

            if self.best_fitness >= 1.0:
                break

            # Selection and reproduction
            self.population.sort(key=lambda sc: -sc.fitness)

            # Elitism
            next_pop = [self.population[0].clone()]

            # Tournament selection + mutation
            while len(next_pop) < self.population_size:
                # Tournament
                candidates = random.sample(
                    self.population[:self.population_size // 2],
                    min(3, len(self.population) // 2)
                )
                parent = max(candidates, key=lambda sc: sc.fitness)

                # Mutation type
                r = random.random()
                if r < 0.4:
                    child = self.mutate_priorities(parent)
                elif r < 0.7:
                    child = self.mutate_strategy(parent)
                else:
                    # Crossover
                    other = random.choice(self.population[:self.population_size // 2])
                    child = self.crossover_priorities(parent, other)

                next_pop.append(child)

            self.population = next_pop

        return best_sc

    def get_strategy_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance of different strategies."""
        result = {}

        for name, fitnesses in self.strategy_fitness.items():
            if fitnesses:
                result[name] = {
                    "mean": sum(fitnesses) / len(fitnesses),
                    "max": max(fitnesses),
                    "count": len(fitnesses),
                }

        return result


# =============================================================================
# Test Cases Generator
# =============================================================================

def generate_conflict_test_cases(
    n_cases: int = 20,
    grid_size: int = 5,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """
    Generate test cases that require conflict resolution.

    The expected sequences are generated by a "correct" priority scheme,
    and evolution should discover this scheme.
    """
    test_cases = []

    for i in range(n_cases):
        # Create context
        grid = mx.array([
            [random.randint(0, 3) for _ in range(grid_size)]
            for _ in range(grid_size)
        ])

        state = {
            "x": random.randint(0, 5),
            "y": random.randint(0, 5),
        }

        context = {"grid": grid, "state": state}

        # Expected sequence depends on context
        # More specific conditions should win
        nonzero_count = int(mx.sum(grid != 0))
        has_ones = bool(mx.any(grid == 1))

        if nonzero_count > grid_size * grid_size // 2:
            expected = ["start", "s1", "s2", "end"]
        elif has_ones:
            expected = ["start", "s0", "s1", "end"]
        elif state["x"] > 2:
            expected = ["start", "s2", "s3", "end"]
        else:
            expected = ["start", "s0", "end"]

        test_cases.append((context, expected))

    return test_cases


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate priority evolution."""
    print("=" * 60)
    print("Transition Priority Evolution")
    print("=" * 60)

    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = generate_conflict_test_cases(n_cases=30)
    print(f"Created {len(test_cases)} test cases")

    # Create evolver
    evolver = PriorityEvolver(
        population_size=30,
        mutation_rate=0.3,
    )

    print("\nEvolving priorities...")
    best = evolver.evolve(test_cases, n_generations=50, verbose=True)

    print(f"\nBest statechart:")
    print(f"  States: {len(best.states)}")
    print(f"  Transitions: {len(best.transitions)}")
    print(f"  Strategy: {type(best.strategy).__name__}")
    print(f"  Fitness: {best.fitness:.3f}")
    print(f"  Conflicts resolved: {best.conflicts_resolved}")

    # Show strategy comparison
    print("\n--- Strategy Comparison ---")
    comparison = evolver.get_strategy_comparison()
    for name, stats in sorted(comparison.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {name}: mean={stats['mean']:.3f}, "
              f"max={stats['max']:.3f}, count={stats['count']}")

    # Show top transitions by priority
    print("\n--- Top Transitions by Priority ---")
    sorted_trans = sorted(best.transitions, key=lambda t: -t.priority)[:5]
    for t in sorted_trans:
        print(f"  {t.source} -> {t.target}: "
              f"priority={t.priority:.2f}, "
              f"guard={t.guard.predicate}, "
              f"specificity={t.guard.specificity:.2f}")

    return evolver, best


def compare_strategies():
    """Compare explicit vs specificity strategies."""
    print("=" * 60)
    print("Strategy Comparison: Explicit vs Specificity")
    print("=" * 60)

    test_cases = generate_conflict_test_cases(n_cases=50)

    strategies = [
        ("Explicit (no noise)", ExplicitPriorityStrategy(selection_noise=0.0)),
        ("Explicit (noise=0.2)", ExplicitPriorityStrategy(selection_noise=0.2)),
        ("Specificity-only", SpecificityPriorityStrategy(specificity_weight=1.0, explicit_weight=0.0)),
        ("Hybrid (50/50)", SpecificityPriorityStrategy(specificity_weight=0.5, explicit_weight=0.5)),
        ("Learned", LearnedPriorityStrategy()),
    ]

    results = []

    for name, strategy in strategies:
        evolver = PriorityEvolver(population_size=20)

        # Initialize population with same strategy
        evolver.population = [
            evolver.create_random_statechart()
            for _ in range(evolver.population_size)
        ]
        for sc in evolver.population:
            sc.strategy = strategy.clone()

        # Evolve priorities only (not strategy)
        for gen in range(30):
            for sc in evolver.population:
                sc.fitness = evolver.evaluate(sc, test_cases)

            evolver.population.sort(key=lambda sc: -sc.fitness)

            next_pop = [evolver.population[0].clone()]
            while len(next_pop) < evolver.population_size:
                parent = random.choice(evolver.population[:10])
                child = evolver.mutate_priorities(parent)
                child.strategy = strategy.clone()
                next_pop.append(child)

            evolver.population = next_pop

        best_fitness = max(sc.fitness for sc in evolver.population)
        avg_fitness = sum(sc.fitness for sc in evolver.population) / len(evolver.population)

        results.append((name, best_fitness, avg_fitness))
        print(f"  {name}: best={best_fitness:.3f}, avg={avg_fitness:.3f}")

    # Winner
    winner = max(results, key=lambda x: x[1])
    print(f"\n*** Best strategy: {winner[0]} (fitness={winner[1]:.3f}) ***")

    return results


if __name__ == "__main__":
    demo()
    print("\n" + "=" * 60 + "\n")
    compare_strategies()
