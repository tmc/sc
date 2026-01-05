"""
History Strategy Evolver

Evolves WHEN to use deep (H*) vs shallow (H) history through learning.

KEY INSIGHT:
- Deep history restores the ENTIRE nested configuration
- Shallow history restores only the DIRECT child state
- No history starts from initial state

This module extends the topology genome with a HistoryType gene per state
and evolves the optimal strategy through fitness evaluation.

REFERENCE: semantics/v1/machine.go:resolveHistory
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Type, Optional, Callable
from enum import IntEnum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# History Types (matching proto/Go semantics)
# =============================================================================

class HistoryType(IntEnum):
    """
    History restoration strategy.

    Matches sc.HistoryType from statecharts.proto:
    - NONE: No history (start from initial state)
    - SHALLOW: Restore direct child only (H)
    - DEEP: Restore full nested configuration (H*)
    """
    NONE = 0      # No history - enter via initial state
    SHALLOW = 1   # H - restore direct child only
    DEEP = 2      # H* - restore entire nested configuration


# =============================================================================
# History-Aware Genome
# =============================================================================

@dataclass
class HistoryGenome:
    """
    Statechart genome with evolvable history strategy per state.

    Extends StatechartGenome with:
    - history_type[i]: HistoryType for each composite state

    The evolution learns which states benefit from:
    - No history (stateless re-entry)
    - Shallow history (remember mode but not sub-mode)
    - Deep history (remember exact nested configuration)
    """
    n_states: int
    parent: List[int]                    # parent[i] = -1 for root
    state_type: List[int]                # 0=BASIC, 1=OR, 2=AND
    history_type: List[HistoryType]      # History strategy per state
    transitions: List[Tuple[int, int, int]]  # (src, tgt, event)
    initial_state: int = 1

    # Fitness tracking
    fitness: float = 0.0
    accuracy: float = 0.0

    def __post_init__(self):
        """Validate and ensure correct sizes."""
        assert len(self.parent) == self.n_states
        assert len(self.state_type) == self.n_states
        assert len(self.history_type) == self.n_states
        assert self.parent[0] == -1  # Root has no parent

    def get_children(self, state_idx: int) -> List[int]:
        """Get children of a state."""
        return [i for i in range(self.n_states) if self.parent[i] == state_idx]

    def get_leaves(self) -> List[int]:
        """Get leaf states (no children)."""
        return [i for i in range(self.n_states) if not self.get_children(i)]

    def get_depth(self, state_idx: int) -> int:
        """Get depth of state in hierarchy."""
        depth = 0
        curr = state_idx
        while self.parent[curr] != -1:
            curr = self.parent[curr]
            depth += 1
        return depth

    def get_max_depth(self) -> int:
        """Get maximum hierarchy depth."""
        return max(self.get_depth(i) for i in range(self.n_states))

    def is_composite(self, state_idx: int) -> bool:
        """Check if state has children (composite)."""
        return len(self.get_children(state_idx)) > 0

    def is_valid(self) -> bool:
        """Check if genome represents valid statechart."""
        # Check for cycles in parent hierarchy
        for i in range(self.n_states):
            visited = set()
            curr = i
            while curr != -1:
                if curr in visited:
                    return False
                visited.add(curr)
                curr = self.parent[curr]

        # Check transitions reference valid states
        for src, tgt, _ in self.transitions:
            if src < 0 or src >= self.n_states:
                return False
            if tgt < 0 or tgt >= self.n_states:
                return False

        return True

    def copy(self) -> 'HistoryGenome':
        """Create a deep copy."""
        new = HistoryGenome(
            n_states=self.n_states,
            parent=self.parent.copy(),
            state_type=self.state_type.copy(),
            history_type=self.history_type.copy(),
            transitions=[t for t in self.transitions],
            initial_state=self.initial_state
        )
        new.fitness = self.fitness
        new.accuracy = self.accuracy
        return new

    def count_history_types(self) -> Dict[str, int]:
        """Count occurrences of each history type."""
        counts = {h.name: 0 for h in HistoryType}
        for ht in self.history_type:
            counts[HistoryType(ht).name] += 1
        return counts


# =============================================================================
# Genome Creation and Initialization
# =============================================================================

def create_random_history_genome(
    n_states: int = 6,
    n_events: int = 5,
    max_depth: int = 3
) -> HistoryGenome:
    """
    Create a random genome with evolvable history types.

    Hierarchy is built to ensure depth for testing nested history.
    """
    if n_states < 3:
        n_states = 3

    # Build hierarchical structure with controlled depth
    parent = [-1]  # Root
    state_type = [1]  # Root is OR

    # First pass: create states with some depth
    for i in range(1, n_states):
        # Pick parent from existing states, preferring shallower for variety
        possible_parents = list(range(i))
        weights = [1 / (1 + sum(1 for p in parent if p == pp)) for pp in possible_parents]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen_parent = random.choices(possible_parents, weights=weights)[0]
        parent.append(chosen_parent)

        # State type: composite if early, leaf if late
        if i < n_states // 2:
            state_type.append(random.choice([0, 1]))  # BASIC or OR
        else:
            state_type.append(0)  # BASIC (leaf)

    # Fix state types based on children
    for i in range(n_states):
        children = [j for j in range(n_states) if parent[j] == i]
        if children:
            state_type[i] = random.choice([1, 2])  # OR or AND
        else:
            state_type[i] = 0  # BASIC

    # Initialize history types randomly
    # NONE is more common since not all states need history
    history_type = []
    for i in range(n_states):
        if i == 0:
            # Root typically doesn't use history
            history_type.append(HistoryType.NONE)
        elif state_type[i] != 0:  # Composite state
            # 50% NONE, 25% SHALLOW, 25% DEEP
            r = random.random()
            if r < 0.5:
                history_type.append(HistoryType.NONE)
            elif r < 0.75:
                history_type.append(HistoryType.SHALLOW)
            else:
                history_type.append(HistoryType.DEEP)
        else:
            # Leaf states don't use history
            history_type.append(HistoryType.NONE)

    # Create some transitions
    leaves = [i for i in range(n_states) if state_type[i] == 0]
    transitions = []

    if len(leaves) >= 2:
        n_transitions = random.randint(len(leaves), len(leaves) * 2)
        for _ in range(n_transitions):
            src = random.choice(leaves)
            tgt = random.choice(leaves)
            event = random.randint(0, n_events - 1)
            transitions.append((src, tgt, event))

    # Initial state
    initial = leaves[0] if leaves else 1

    return HistoryGenome(
        n_states=n_states,
        parent=parent,
        state_type=state_type,
        history_type=history_type,
        transitions=transitions,
        initial_state=initial
    )


# =============================================================================
# History Execution Simulator
# =============================================================================

class HistoryMachine:
    """
    Simulates statechart execution with history semantics.

    Implements the same logic as semantics/v1/machine.go:resolveHistory
    """

    def __init__(self, genome: HistoryGenome):
        self.genome = genome
        self.current_state: int = genome.initial_state
        self.history: Dict[int, List[int]] = {}  # parent -> stored configuration

        # Build transition map
        self.transition_map: Dict[Tuple[int, int], int] = {}
        for src, tgt, event in genome.transitions:
            self.transition_map[(src, event)] = tgt

    def get_configuration(self) -> List[int]:
        """Get current active states (leaf + ancestors)."""
        config = []
        curr = self.current_state
        while curr != -1:
            config.append(curr)
            curr = self.genome.parent[curr]
        return config

    def get_active_descendants(self, state_idx: int) -> List[int]:
        """Get all active descendants of a state."""
        config = self.get_configuration()
        descendants = []

        for s in config:
            # Check if s is a descendant of state_idx
            curr = s
            while curr != -1:
                if self.genome.parent[curr] == state_idx:
                    descendants.append(s)
                    break
                curr = self.genome.parent[curr]

        return descendants

    def save_history(self, state_idx: int):
        """
        Save history when exiting a composite state.

        Stores all active descendants of this state.
        """
        if self.genome.is_composite(state_idx):
            active_descendants = self.get_active_descendants(state_idx)
            if active_descendants:
                self.history[state_idx] = active_descendants

    def resolve_history(self, state_idx: int) -> int:
        """
        Resolve which state to enter based on history type.

        IMPLEMENTS semantics/v1/machine.go:resolveHistory logic:
        - DEEP: Return stored nested state
        - SHALLOW: Return only direct child from storage
        - NONE: Return initial child
        """
        ht = self.genome.history_type[state_idx]

        if ht == HistoryType.NONE:
            # No history: find initial child
            return self._get_initial_child(state_idx)

        if state_idx not in self.history:
            # No stored history: fall back to initial
            return self._get_initial_child(state_idx)

        stored = self.history[state_idx]

        if ht == HistoryType.DEEP:
            # Deep history: return deepest stored state
            # (the one with maximum depth)
            if stored:
                return max(stored, key=lambda s: self.genome.get_depth(s))
            return self._get_initial_child(state_idx)

        elif ht == HistoryType.SHALLOW:
            # Shallow history: return only DIRECT child
            direct_children = self.genome.get_children(state_idx)
            for s in stored:
                # Check if s is a direct child
                if s in direct_children:
                    return s
                # Or find ancestor that is direct child
                curr = s
                while curr != -1:
                    if self.genome.parent[curr] == state_idx:
                        return curr
                    curr = self.genome.parent[curr]
            return self._get_initial_child(state_idx)

        return self._get_initial_child(state_idx)

    def _get_initial_child(self, state_idx: int) -> int:
        """Get initial child of a composite state."""
        children = self.genome.get_children(state_idx)
        if not children:
            return state_idx  # Already a leaf

        # Recursively find leaf
        child = children[0]
        while True:
            next_children = self.genome.get_children(child)
            if not next_children:
                return child
            child = next_children[0]

    def step(self, event: int) -> Tuple[int, int]:
        """
        Take a step with the given event.

        Returns (old_state, new_state).
        """
        old_state = self.current_state

        key = (self.current_state, event)
        if key in self.transition_map:
            target = self.transition_map[key]

            # Save history for states we're exiting
            config = self.get_configuration()
            new_config = []
            curr = target
            while curr != -1:
                new_config.append(curr)
                curr = self.genome.parent[curr]

            # States being exited
            for s in config:
                if s not in new_config:
                    self.save_history(s)

            # If entering composite state, resolve history
            while True:
                children = self.genome.get_children(target)
                if not children:
                    break
                target = self.resolve_history(target)

            self.current_state = target

        return old_state, self.current_state

    def reset(self):
        """Reset to initial state."""
        self.current_state = self.genome.initial_state
        self.history = {}


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_history_type(genome: HistoryGenome, mutation_rate: float = 0.3) -> HistoryGenome:
    """
    Mutate history type genes.

    Only composite states can have meaningful history types.
    """
    g = genome.copy()

    for i in range(g.n_states):
        if random.random() < mutation_rate:
            if g.is_composite(i):
                # Composite state: can have any history type
                if random.random() < 0.5:
                    # Random new type
                    g.history_type[i] = random.choice(list(HistoryType))
                else:
                    # Cycle to next type
                    current = g.history_type[i]
                    next_type = (current + 1) % 3
                    g.history_type[i] = HistoryType(next_type)
            else:
                # Leaf state: always NONE
                g.history_type[i] = HistoryType.NONE

    return g


def mutate_structure(genome: HistoryGenome, mutation_rate: float = 0.2) -> HistoryGenome:
    """Mutate structural aspects (states, transitions)."""
    g = genome.copy()

    # Mutation: Toggle OR/AND
    if random.random() < mutation_rate:
        composites = [i for i in range(g.n_states) if g.is_composite(i)]
        if composites:
            idx = random.choice(composites)
            g.state_type[idx] = 2 if g.state_type[idx] == 1 else 1

    # Mutation: Add transition
    if random.random() < mutation_rate:
        leaves = g.get_leaves()
        if len(leaves) >= 2:
            src = random.choice(leaves)
            tgt = random.choice(leaves)
            event = random.randint(0, 4)
            g.transitions.append((src, tgt, event))

    # Mutation: Remove transition
    if random.random() < mutation_rate and len(g.transitions) > 1:
        idx = random.randint(0, len(g.transitions) - 1)
        g.transitions.pop(idx)

    return g


def crossover_history(g1: HistoryGenome, g2: HistoryGenome) -> HistoryGenome:
    """
    Crossover history type genes between two genomes.

    Structure from g1, history types mixed from both.
    """
    child = g1.copy()

    # Mix history types
    min_states = min(g1.n_states, g2.n_states)
    for i in range(min_states):
        if random.random() < 0.5:
            child.history_type[i] = g2.history_type[i]

    return child


# =============================================================================
# History Evolver
# =============================================================================

@dataclass
class HistoryEvolverConfig:
    """Configuration for history evolution."""
    population_size: int = 30
    n_generations: int = 50
    elite_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    n_states: int = 8
    n_events: int = 5
    n_scenarios: int = 100
    verbose: bool = True
    log_every: int = 10


@dataclass
class HistoryEvolutionResult:
    """Result of history evolution."""
    best_genome: HistoryGenome
    final_accuracy: float
    generations_to_target: int  # -1 if not reached
    total_generations: int
    elapsed_time: float
    accuracy_history: List[float] = field(default_factory=list)
    history_type_evolution: List[Dict[str, int]] = field(default_factory=list)


class HistoryEvolver:
    """
    Evolves history restoration strategy through learning.

    Given scenarios that require specific history behavior, evolves
    the optimal history_type per state.
    """

    def __init__(
        self,
        scenario_generator: Callable,  # () -> List[scenario]
        config: HistoryEvolverConfig = None
    ):
        self.scenario_generator = scenario_generator
        self.config = config or HistoryEvolverConfig()

    def evaluate_fitness(
        self,
        genome: HistoryGenome,
        scenarios: List
    ) -> Tuple[float, float]:
        """
        Evaluate genome fitness on scenarios.

        Each scenario is (events, expected_final_state).
        """
        if not genome.is_valid():
            return 0.0, 0.0

        correct = 0
        total = 0

        for events, expected_state in scenarios:
            machine = HistoryMachine(genome)

            for event in events:
                machine.step(event)

            actual_state = machine.current_state

            # Check if reached expected state
            if actual_state == expected_state:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        # Parsimony bonus for simpler history usage
        history_counts = genome.count_history_types()
        n_history = history_counts['SHALLOW'] + history_counts['DEEP']
        parsimony = 1.0 - (n_history / genome.n_states) * 0.1

        fitness = accuracy * parsimony
        return fitness, accuracy

    def evolve(
        self,
        target_accuracy: float = 0.9,
        callback: Callable = None
    ) -> HistoryEvolutionResult:
        """
        Run evolution to learn optimal history strategy.

        Returns best genome found.
        """
        config = self.config
        start_time = time.time()

        if config.verbose:
            print("=" * 60)
            print("DEEP vs SHALLOW HISTORY EVOLUTION")
            print("=" * 60)
            print(f"Population: {config.population_size}")
            print(f"States: {config.n_states}")
            print(f"Scenarios: {config.n_scenarios}")
            print("-" * 60)

        # Generate scenarios
        scenarios = self.scenario_generator()
        if config.verbose:
            print(f"Generated {len(scenarios)} scenarios")

        # Initialize population
        population = [
            create_random_history_genome(
                n_states=config.n_states,
                n_events=config.n_events
            )
            for _ in range(config.population_size)
        ]

        # Evaluate initial population
        for genome in population:
            genome.fitness, genome.accuracy = self.evaluate_fitness(genome, scenarios)

        # Track results
        accuracy_history = []
        history_type_evolution = []
        generations_to_target = -1
        best_ever = max(population, key=lambda g: g.fitness)

        # Evolution loop
        for gen in range(config.n_generations):
            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if population[0].fitness > best_ever.fitness:
                best_ever = population[0].copy()

            accuracy_history.append(best_ever.accuracy)
            history_type_evolution.append(best_ever.count_history_types())

            # Check target
            if generations_to_target < 0 and best_ever.accuracy >= target_accuracy:
                generations_to_target = gen

            # Progress report
            if config.verbose and (gen % config.log_every == 0 or gen == config.n_generations - 1):
                counts = best_ever.count_history_types()
                print(f"Gen {gen:3d} | Best: {population[0].accuracy:.3f} | "
                      f"H={counts['SHALLOW']}, H*={counts['DEEP']}, None={counts['NONE']}")

            # Elite
            new_pop = [g.copy() for g in population[:config.elite_size]]

            # Generate offspring
            while len(new_pop) < config.population_size:
                if random.random() < config.crossover_rate:
                    p1 = random.choice(population[:config.population_size//2])
                    p2 = random.choice(population[:config.population_size//2])
                    child = crossover_history(p1, p2)
                else:
                    parent = random.choice(population[:config.population_size//2])
                    child = parent.copy()

                # Mutate
                child = mutate_history_type(child, config.mutation_rate)
                child = mutate_structure(child, config.mutation_rate * 0.5)

                if child.is_valid():
                    child.fitness, child.accuracy = self.evaluate_fitness(child, scenarios)
                    new_pop.append(child)

            population = new_pop

            # Callback
            if callback:
                callback(gen, population, best_ever)

        elapsed = time.time() - start_time

        if config.verbose:
            print("-" * 60)
            counts = best_ever.count_history_types()
            print(f"FINAL: accuracy={best_ever.accuracy:.3f}")
            print(f"History types: H={counts['SHALLOW']}, H*={counts['DEEP']}, None={counts['NONE']}")
            print(f"Elapsed: {elapsed:.2f}s")
            print("=" * 60)

        return HistoryEvolutionResult(
            best_genome=best_ever,
            final_accuracy=best_ever.accuracy,
            generations_to_target=generations_to_target,
            total_generations=config.n_generations,
            elapsed_time=elapsed,
            accuracy_history=accuracy_history,
            history_type_evolution=history_type_evolution
        )


# =============================================================================
# Convenience Function
# =============================================================================

def evolve_history_strategy(
    scenario_generator: Callable,
    n_generations: int = 50,
    population_size: int = 30,
    target_accuracy: float = 0.9,
    verbose: bool = True
) -> HistoryEvolutionResult:
    """
    Convenience function to evolve history strategy.

    Args:
        scenario_generator: Function returning list of (events, expected_state) pairs
        n_generations: Maximum generations
        population_size: Population size
        target_accuracy: Stop early if reached
        verbose: Print progress

    Returns:
        HistoryEvolutionResult with best genome and statistics
    """
    config = HistoryEvolverConfig(
        population_size=population_size,
        n_generations=n_generations,
        verbose=verbose
    )

    evolver = HistoryEvolver(scenario_generator, config)
    return evolver.evolve(target_accuracy=target_accuracy)


# =============================================================================
# Testing
# =============================================================================

def test_history_machine():
    """Test history machine execution."""
    print("=" * 60)
    print("HISTORY MACHINE TEST")
    print("=" * 60)

    # Create a simple hierarchical genome
    # Root(0) -> [Composite(1) -> [Leaf(3), Leaf(4)], Leaf(2)]
    genome = HistoryGenome(
        n_states=5,
        parent=[-1, 0, 0, 1, 1],
        state_type=[1, 1, 0, 0, 0],  # OR, OR, BASIC, BASIC, BASIC
        history_type=[
            HistoryType.NONE,     # Root
            HistoryType.DEEP,     # Composite - DEEP history
            HistoryType.NONE,     # Leaf
            HistoryType.NONE,     # Leaf
            HistoryType.NONE      # Leaf
        ],
        transitions=[
            (3, 4, 0),  # Leaf3 -> Leaf4 on event 0
            (4, 3, 1),  # Leaf4 -> Leaf3 on event 1
            (3, 2, 2),  # Leaf3 -> Leaf2 on event 2 (exit composite)
            (4, 2, 2),  # Leaf4 -> Leaf2 on event 2 (exit composite)
            (2, 3, 3),  # Leaf2 -> Leaf3 on event 3 (enter composite via history)
        ],
        initial_state=3
    )

    print(f"Genome: {genome.n_states} states")
    print(f"History types: {genome.count_history_types()}")

    # Test execution
    machine = HistoryMachine(genome)
    print(f"\nInitial state: {machine.current_state}")

    # Navigate within composite
    old, new = machine.step(0)
    print(f"Event 0: {old} -> {new}")

    # Exit to Leaf2
    old, new = machine.step(2)
    print(f"Event 2: {old} -> {new}")

    # Return via history - should restore to Leaf4 (DEEP) or Leaf4's parent's initial (SHALLOW)
    old, new = machine.step(3)
    print(f"Event 3 (history): {old} -> {new}")

    if new == 4:
        print("DEEP history worked! Restored to exact nested state.")
    else:
        print(f"Returned to state {new}")

    print("\nHistory machine test complete!")


def test_basic_evolution():
    """Test basic evolution functionality."""
    print("=" * 60)
    print("BASIC EVOLUTION TEST")
    print("=" * 60)

    # Simple scenario generator
    def simple_scenario_generator():
        scenarios = []
        for _ in range(50):
            events = [random.randint(0, 4) for _ in range(random.randint(3, 8))]
            expected = random.randint(1, 4)  # Random expected (evolution will learn)
            scenarios.append((events, expected))
        return scenarios

    config = HistoryEvolverConfig(
        population_size=20,
        n_generations=20,
        n_states=6,
        n_events=5,
        verbose=True,
        log_every=5
    )

    evolver = HistoryEvolver(simple_scenario_generator, config)
    result = evolver.evolve(target_accuracy=0.5)

    print(f"\nResult: accuracy={result.final_accuracy:.3f}")
    print(f"History types: {result.best_genome.count_history_types()}")

    print("\nBasic evolution test complete!")
    return result


if __name__ == "__main__":
    test_history_machine()
    print("\n")
    test_basic_evolution()
