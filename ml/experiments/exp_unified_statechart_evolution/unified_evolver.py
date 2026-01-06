"""
Unified Statechart Evolver

THE FLAGSHIP EVOLVER: Co-evolves all statechart components using NSGA-II.

Combines evolution of:
- Topology (states, hierarchy)
- History types
- Guards
- Priorities
- Actions
- Events

Multi-objective optimization finds Pareto-optimal solutions
trading off correctness, minimality, and interpretability.
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

from .unified_genome import (
    UnifiedGenome, UnifiedTransition, UnifiedGuard, UnifiedAction,
    StateType, HistoryType, ActionType, GuardOp,
    create_random_unified_genome
)
from .unified_executor import execute_scenarios, UnifiedMachine
from .unified_fitness import (
    UnifiedFitnessComponents, compute_unified_fitness,
    nsga2_select, tournament_select_nsga2, compute_pareto_fronts
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedEvolverConfig:
    """Configuration for unified evolution."""
    # Population
    population_size: int = 50
    n_generations: int = 100
    elite_size: int = 5

    # Genome limits
    max_states: int = 15
    min_states: int = 3
    max_transitions: int = 30
    max_actions_per_transition: int = 2

    # Events and context
    n_events: int = 5
    context_variables: List[str] = field(default_factory=lambda: ['x', 'y', 'mode', 'count'])

    # Component toggles
    evolve_topology: bool = True
    evolve_history: bool = True
    evolve_guards: bool = True
    evolve_priorities: bool = True
    evolve_actions: bool = True

    # Mutation rates
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7

    # Specific mutation weights
    add_state_prob: float = 0.1
    remove_state_prob: float = 0.08
    change_type_prob: float = 0.1
    change_history_prob: float = 0.1
    add_transition_prob: float = 0.15
    remove_transition_prob: float = 0.1
    mutate_guard_prob: float = 0.2
    mutate_priority_prob: float = 0.1
    mutate_action_prob: float = 0.15

    # Selection
    use_nsga2: bool = True
    tournament_size: int = 3

    # Fitness weights (for single-objective fallback)
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'correctness': 1.0,
        'minimality': 0.2,
        'interpretability': 0.1
    })

    # Logging
    verbose: bool = True
    log_every: int = 10


# =============================================================================
# Evolution Result
# =============================================================================

@dataclass
class UnifiedEvolutionResult:
    """Result of unified evolution."""
    best_genome: UnifiedGenome
    pareto_front: List[UnifiedGenome]
    best_fitness: UnifiedFitnessComponents
    pareto_fitness: List[UnifiedFitnessComponents]
    generations: int
    elapsed_time: float
    fitness_history: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_topology(genome: UnifiedGenome, config: UnifiedEvolverConfig) -> UnifiedGenome:
    """Mutate topology: states and hierarchy."""
    g = genome.copy()

    # Add state
    if random.random() < config.add_state_prob and g.n_states < config.max_states:
        parent_idx = random.randint(0, g.n_states - 1)
        g.n_states += 1
        g.parent.append(parent_idx)
        g.state_type.append(StateType.BASIC)
        g.history_type.append(HistoryType.NONE)
        g.state_labels.append(f"S{g.n_states - 1}")

        # Update parent type if it was BASIC
        if g.state_type[parent_idx] == StateType.BASIC:
            g.state_type[parent_idx] = random.choice([StateType.OR, StateType.AND])

    # Remove state
    if random.random() < config.remove_state_prob and g.n_states > config.min_states:
        removable = [i for i in range(1, g.n_states) if not g.get_children(i)]
        if removable:
            to_remove = random.choice(removable)

            # Remove transitions
            g.transitions = [t for t in g.transitions
                           if t.src != to_remove and t.tgt != to_remove]

            # Update indices
            g.parent = [p if p < to_remove else (p - 1 if p > to_remove else p)
                       for i, p in enumerate(g.parent) if i != to_remove]
            g.state_type = [g.state_type[i] for i in range(g.n_states) if i != to_remove]
            g.history_type = [g.history_type[i] for i in range(g.n_states) if i != to_remove]
            g.state_labels = [g.state_labels[i] for i in range(g.n_states) if i != to_remove]
            g.n_states -= 1

            # Update transition indices
            for t in g.transitions:
                if t.src > to_remove:
                    t.src -= 1
                if t.tgt > to_remove:
                    t.tgt -= 1

            # Update initial state
            if g.initial_state == to_remove:
                leaves = g.get_leaves()
                g.initial_state = leaves[0] if leaves else 1
            elif g.initial_state > to_remove:
                g.initial_state -= 1

    # Change state type
    if random.random() < config.change_type_prob:
        idx = random.randint(1, g.n_states - 1)
        children = g.get_children(idx)
        if children:
            g.state_type[idx] = StateType.AND if g.state_type[idx] == StateType.OR else StateType.OR

    return g


def mutate_history(genome: UnifiedGenome, config: UnifiedEvolverConfig) -> UnifiedGenome:
    """Mutate history types."""
    g = genome.copy()

    if not config.evolve_history:
        return g

    for i in range(g.n_states):
        if random.random() < config.change_history_prob:
            if g.is_composite(i):
                g.history_type[i] = random.choice(list(HistoryType))
            else:
                g.history_type[i] = HistoryType.NONE

    return g


def mutate_transitions(genome: UnifiedGenome, config: UnifiedEvolverConfig) -> UnifiedGenome:
    """Mutate transitions, guards, priorities, actions."""
    g = genome.copy()
    leaves = g.get_leaves()

    if not leaves:
        return g

    # Add transition
    if random.random() < config.add_transition_prob and len(g.transitions) < config.max_transitions:
        src = random.choice(leaves)
        tgt = random.choice(leaves)
        event = random.randint(0, g.n_events - 1)

        guard = UnifiedGuard()
        if config.evolve_guards and random.random() < 0.4:
            var = random.choice(g.context_variables)
            op = random.choice([GuardOp.EQ, GuardOp.GT, GuardOp.LT])
            value = random.randint(0, 5)
            guard = UnifiedGuard(variable=var, op=op, value=value)

        priority = random.uniform(0, 1) if config.evolve_priorities else 0.0

        actions = []
        if config.evolve_actions and random.random() < 0.3:
            var = random.choice(g.context_variables)
            action_type = random.choice([ActionType.SET, ActionType.INCREMENT])
            value = random.randint(0, 3)
            actions.append(UnifiedAction(action_type=action_type, variable=var, value=value))

        g.transitions.append(UnifiedTransition(
            src=src, tgt=tgt, event=event,
            guard=guard, priority=priority, actions=actions
        ))

    # Remove transition
    if random.random() < config.remove_transition_prob and len(g.transitions) > 1:
        idx = random.randint(0, len(g.transitions) - 1)
        g.transitions.pop(idx)

    # Mutate existing transitions
    for t in g.transitions:
        # Mutate guard
        if config.evolve_guards and random.random() < config.mutate_guard_prob:
            if random.random() < 0.3:
                t.guard = UnifiedGuard()  # Reset to TRUE
            else:
                t.guard.variable = random.choice(g.context_variables)
                t.guard.op = random.choice(list(GuardOp))
                t.guard.value = random.randint(0, 10)

        # Mutate priority
        if config.evolve_priorities and random.random() < config.mutate_priority_prob:
            t.priority = max(0, min(1, t.priority + random.gauss(0, 0.2)))

        # Mutate actions
        if config.evolve_actions and random.random() < config.mutate_action_prob:
            if random.random() < 0.3 and t.actions:
                # Remove action
                t.actions.pop(random.randint(0, len(t.actions) - 1))
            elif random.random() < 0.3 and len(t.actions) < config.max_actions_per_transition:
                # Add action
                var = random.choice(g.context_variables)
                action_type = random.choice(list(ActionType))
                value = random.randint(0, 5) if action_type != ActionType.TOGGLE else None
                t.actions.append(UnifiedAction(action_type=action_type, variable=var, value=value))
            elif t.actions:
                # Modify action
                action = random.choice(t.actions)
                action.variable = random.choice(g.context_variables)

    return g


def mutate_unified(genome: UnifiedGenome, config: UnifiedEvolverConfig) -> UnifiedGenome:
    """Apply all mutations."""
    g = genome.copy()

    if config.evolve_topology:
        g = mutate_topology(g, config)

    if config.evolve_history:
        g = mutate_history(g, config)

    g = mutate_transitions(g, config)

    return g


def crossover_unified(g1: UnifiedGenome, g2: UnifiedGenome) -> UnifiedGenome:
    """Crossover two genomes."""
    # Take structure from g1, mix transitions from both
    child = g1.copy()

    # Mix transitions
    n_from_g2 = len(g2.transitions) // 2
    child.transitions = child.transitions[:len(child.transitions) // 2]

    for t in random.sample(g2.transitions, min(n_from_g2, len(g2.transitions))):
        if t.src < child.n_states and t.tgt < child.n_states:
            child.transitions.append(t.copy())

    # Mix history types
    min_states = min(g1.n_states, g2.n_states)
    for i in range(min_states):
        if random.random() < 0.5:
            child.history_type[i] = g2.history_type[i]

    return child


# =============================================================================
# Unified Evolver
# =============================================================================

class UnifiedEvolver:
    """
    THE FLAGSHIP EVOLVER: Co-evolves all statechart components.

    Uses NSGA-II for multi-objective optimization of:
    - Correctness
    - Minimality
    - Interpretability
    """

    def __init__(
        self,
        scenario_generator: Callable[[], List[Tuple[List[int], int]]],
        config: UnifiedEvolverConfig = None
    ):
        self.scenario_generator = scenario_generator
        self.config = config or UnifiedEvolverConfig()

    def create_initial_population(self) -> List[UnifiedGenome]:
        """Create initial random population."""
        population = []
        attempts = 0
        max_attempts = self.config.population_size * 3

        while len(population) < self.config.population_size and attempts < max_attempts:
            genome = create_random_unified_genome(
                max_states=self.config.max_states,
                n_events=self.config.n_events,
                context_variables=self.config.context_variables,
                include_history=self.config.evolve_history,
                include_actions=self.config.evolve_actions,
                include_guards=self.config.evolve_guards
            )
            if genome.is_valid():
                population.append(genome)
            attempts += 1

        return population

    def evaluate_population(
        self,
        population: List[UnifiedGenome],
        scenarios: List[Tuple[List[int], int]]
    ) -> List[UnifiedFitnessComponents]:
        """Evaluate fitness for all genomes."""
        components_list = []

        for genome in population:
            if not genome.is_valid():
                # Invalid genome gets worst fitness
                components = UnifiedFitnessComponents()
            else:
                components = compute_unified_fitness(
                    genome, scenarios,
                    max_states=self.config.max_states,
                    max_transitions=self.config.max_transitions
                )

            components_list.append(components)

            # Store in genome
            genome.accuracy = components.accuracy
            genome.fitness = components.weighted_sum(self.config.fitness_weights)
            genome.complexity = genome.compute_complexity()

        return components_list

    def select_parents(
        self,
        population: List[UnifiedGenome],
        components_list: List[UnifiedFitnessComponents]
    ) -> List[int]:
        """Select parents for next generation."""
        if self.config.use_nsga2:
            return tournament_select_nsga2(
                population, components_list,
                n_parents=self.config.population_size // 2,
                tournament_size=self.config.tournament_size
            )
        else:
            # Simple tournament selection on weighted sum
            parents = []
            for _ in range(self.config.population_size // 2):
                candidates = random.sample(range(len(population)),
                                          min(self.config.tournament_size, len(population)))
                winner = max(candidates, key=lambda i: population[i].fitness)
                parents.append(winner)
            return parents

    def evolve(self, callback: Callable = None) -> UnifiedEvolutionResult:
        """
        Run unified evolution.

        Returns Pareto front of solutions.
        """
        config = self.config
        start_time = time.time()

        if config.verbose:
            print("=" * 70)
            print("UNIFIED STATECHART EVOLUTION - FLAGSHIP")
            print("=" * 70)
            print(f"Population: {config.population_size}")
            print(f"Generations: {config.n_generations}")
            print(f"Max states: {config.max_states}")
            print(f"Components: topology={config.evolve_topology}, "
                  f"history={config.evolve_history}, guards={config.evolve_guards}, "
                  f"priorities={config.evolve_priorities}, actions={config.evolve_actions}")
            print(f"Selection: {'NSGA-II' if config.use_nsga2 else 'Tournament'}")
            print("-" * 70)

        # Generate scenarios
        scenarios = self.scenario_generator()
        if config.verbose:
            print(f"Scenarios: {len(scenarios)}")

        # Initialize population
        population = self.create_initial_population()
        if config.verbose:
            print(f"Initial population: {len(population)} valid genomes")

        # Evaluate
        components_list = self.evaluate_population(population, scenarios)

        # Track results
        fitness_history = []
        best_ever = max(population, key=lambda g: g.fitness)
        best_components = max(components_list, key=lambda c: c.weighted_sum(config.fitness_weights))

        # Evolution loop
        for gen in range(config.n_generations):
            # Select parents
            parent_indices = self.select_parents(population, components_list)
            parents = [population[i] for i in parent_indices]

            # Elite
            elite_indices = nsga2_select(population, components_list, config.elite_size)
            elite = [population[i].copy() for i in elite_indices]

            # Generate offspring
            offspring = []
            while len(offspring) < config.population_size - config.elite_size:
                if random.random() < config.crossover_rate and len(parents) >= 2:
                    p1, p2 = random.sample(parents, 2)
                    child = crossover_unified(p1, p2)
                else:
                    parent = random.choice(parents)
                    child = parent.copy()

                if random.random() < config.mutation_rate:
                    child = mutate_unified(child, config)

                if child.is_valid():
                    offspring.append(child)

            # New population
            population = elite + offspring

            # Evaluate
            components_list = self.evaluate_population(population, scenarios)

            # Track best
            gen_best_idx = max(range(len(population)), key=lambda i: population[i].fitness)
            if population[gen_best_idx].fitness > best_ever.fitness:
                best_ever = population[gen_best_idx].copy()
                best_components = components_list[gen_best_idx]

            # Pareto front
            fronts = compute_pareto_fronts(components_list)
            pareto_size = len(fronts[0]) if fronts else 0

            # History
            gen_best = population[gen_best_idx]
            fitness_history.append({
                'generation': gen,
                'best_fitness': gen_best.fitness,
                'best_accuracy': gen_best.accuracy,
                'pareto_size': pareto_size,
                'avg_states': sum(g.n_states for g in population) / len(population)
            })

            # Callback
            if callback:
                callback(gen, population, best_ever)

            # Progress
            if config.verbose and (gen % config.log_every == 0 or gen == config.n_generations - 1):
                avg_fitness = sum(g.fitness for g in population) / len(population)
                print(f"Gen {gen:3d} | "
                      f"Best: {gen_best.fitness:.4f} (acc={gen_best.accuracy:.3f}) | "
                      f"Avg: {avg_fitness:.4f} | "
                      f"Pareto: {pareto_size} | "
                      f"States: {gen_best.n_states}")

        elapsed = time.time() - start_time

        # Get final Pareto front
        fronts = compute_pareto_fronts(components_list)
        pareto_indices = fronts[0] if fronts else [0]
        pareto_front = [population[i].copy() for i in pareto_indices]
        pareto_fitness = [components_list[i] for i in pareto_indices]

        if config.verbose:
            print("-" * 70)
            print(f"FINAL RESULTS")
            print(f"Best fitness: {best_ever.fitness:.4f}")
            print(f"Best accuracy: {best_ever.accuracy:.3f}")
            print(f"Best complexity: {best_ever.compute_complexity():.2f}")
            print(f"Pareto front size: {len(pareto_front)}")
            print(f"Elapsed: {elapsed:.2f}s")
            print("=" * 70)

        return UnifiedEvolutionResult(
            best_genome=best_ever,
            pareto_front=pareto_front,
            best_fitness=best_components,
            pareto_fitness=pareto_fitness,
            generations=config.n_generations,
            elapsed_time=elapsed,
            fitness_history=fitness_history
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_unified_evolution(
    scenario_generator: Callable,
    n_generations: int = 50,
    population_size: int = 30,
    verbose: bool = True
) -> UnifiedEvolutionResult:
    """Quick unified evolution for testing."""
    config = UnifiedEvolverConfig(
        population_size=population_size,
        n_generations=n_generations,
        max_states=10,
        verbose=verbose,
        log_every=10
    )

    evolver = UnifiedEvolver(scenario_generator, config)
    return evolver.evolve()


# =============================================================================
# Testing
# =============================================================================

def test_unified_evolver():
    """Test unified evolver."""
    print("=" * 70)
    print("UNIFIED EVOLVER TEST")
    print("=" * 70)

    # Simple scenario generator
    def simple_scenarios():
        scenarios = []
        for _ in range(50):
            events = [random.randint(0, 4) for _ in range(random.randint(2, 5))]
            expected = random.randint(1, 3)
            scenarios.append((events, expected))
        return scenarios

    result = quick_unified_evolution(
        simple_scenarios,
        n_generations=20,
        population_size=20,
        verbose=True
    )

    print(f"\nResult:")
    print(f"  Best accuracy: {result.best_genome.accuracy:.3f}")
    print(f"  Pareto front: {len(result.pareto_front)} solutions")
    print(f"  Components: {result.best_genome.count_components()}")

    print("\nUnified evolver test complete!")
    return result


if __name__ == "__main__":
    test_unified_evolver()
