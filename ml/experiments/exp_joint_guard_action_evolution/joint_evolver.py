"""
Joint Guard-Action Evolver

Co-evolves guards and actions together using:
1. CAUSAL-AWARE MUTATION: Mutate causally-related pairs together
2. COHERENCE SELECTION: Prefer individuals with high coherence
3. JOINT CROSSOVER: Exchange causal clusters, not random pairs

Key insight: Independent evolution of guards and actions leads to
incoherent statecharts. Joint evolution maintains causal relationships.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from collections import defaultdict

from .joint_genome import (
    JointGenome, GuardActionPair, Guard, GuardClause, GuardOp, GuardCombinator,
    Action, ActionEffect, EffectOp,
    random_joint_genome, random_guard_clause, random_action_effect
)
from .causal_graph import (
    build_causal_graph, analyze_causal_graph, CausalGraph,
    get_mutation_cluster, suggest_coherent_mutation
)
from .coherence_fitness import (
    CoherenceFitness, compute_coherence_fitness, Scenario
)


# =============================================================================
# Evolution Config
# =============================================================================

@dataclass
class JointEvolverConfig:
    """Configuration for joint evolution."""
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elite_size: int = 5
    tournament_size: int = 3

    # Causal-aware evolution
    causal_mutation_prob: float = 0.7  # Prefer causal-guided mutations
    cluster_mutation_depth: int = 1     # Depth for mutation clusters
    cluster_crossover: bool = True      # Exchange causal clusters

    # Variables
    variables: List[str] = field(default_factory=list)
    possible_values: Dict[str, List[Any]] = field(default_factory=dict)

    # Problem size
    n_states: int = 4
    n_events: int = 3
    n_pairs: int = 8


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_guard_clause(
    clause: GuardClause,
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> GuardClause:
    """Mutate a single guard clause."""
    mutation_type = random.choice(['variable', 'operator', 'value', 'new'])

    if mutation_type == 'new':
        return random_guard_clause(variables, possible_values)

    new_clause = copy.deepcopy(clause)

    if mutation_type == 'variable':
        new_clause.variable = random.choice(variables)
        # Update value to match variable type
        values = possible_values.get(new_clause.variable, [0, 1, True, False])
        new_clause.value = random.choice(values)

    elif mutation_type == 'operator':
        new_clause.op = random.choice([
            GuardOp.EQ, GuardOp.NE, GuardOp.LT, GuardOp.LE,
            GuardOp.GT, GuardOp.GE
        ])

    elif mutation_type == 'value':
        values = possible_values.get(clause.variable, [0, 1, True, False])
        new_clause.value = random.choice(values)

    return new_clause


def mutate_guard(
    guard: Guard,
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> Guard:
    """Mutate a guard."""
    new_guard = copy.deepcopy(guard)
    mutation_type = random.choice(['clause', 'add', 'remove', 'combinator'])

    if mutation_type == 'clause' and new_guard.clauses:
        idx = random.randint(0, len(new_guard.clauses) - 1)
        new_guard.clauses[idx] = mutate_guard_clause(
            new_guard.clauses[idx], variables, possible_values)

    elif mutation_type == 'add' and len(new_guard.clauses) < 4:
        new_guard.clauses.append(random_guard_clause(variables, possible_values))

    elif mutation_type == 'remove' and len(new_guard.clauses) > 1:
        idx = random.randint(0, len(new_guard.clauses) - 1)
        new_guard.clauses.pop(idx)

    elif mutation_type == 'combinator':
        new_guard.combinator = (GuardCombinator.OR if new_guard.combinator == GuardCombinator.AND
                                 else GuardCombinator.AND)

    return new_guard


def mutate_action_effect(
    effect: ActionEffect,
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> ActionEffect:
    """Mutate a single action effect."""
    mutation_type = random.choice(['variable', 'operator', 'value', 'new'])

    if mutation_type == 'new':
        return random_action_effect(variables, possible_values)

    new_effect = ActionEffect(
        variable=effect.variable,
        effect_op=effect.effect_op,
        value=effect.value
    )

    if mutation_type == 'variable':
        new_effect.variable = random.choice(variables)
        values = possible_values.get(new_effect.variable, [0, 1])
        new_effect.value = random.choice(values)

    elif mutation_type == 'operator':
        new_effect.effect_op = random.choice([
            EffectOp.SET, EffectOp.INCREMENT, EffectOp.DECREMENT,
            EffectOp.TOGGLE
        ])

    elif mutation_type == 'value':
        values = possible_values.get(effect.variable, [0, 1])
        new_effect.value = random.choice(values)

    return new_effect


def mutate_action(
    action: Action,
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> Action:
    """Mutate an action."""
    new_action = Action(name=action.name, effects=copy.deepcopy(action.effects))
    mutation_type = random.choice(['effect', 'add', 'remove'])

    if mutation_type == 'effect' and new_action.effects:
        idx = random.randint(0, len(new_action.effects) - 1)
        new_action.effects[idx] = mutate_action_effect(
            new_action.effects[idx], variables, possible_values)

    elif mutation_type == 'add' and len(new_action.effects) < 3:
        new_action.effects.append(random_action_effect(variables, possible_values))

    elif mutation_type == 'remove' and len(new_action.effects) > 0:
        idx = random.randint(0, len(new_action.effects) - 1)
        new_action.effects.pop(idx)

    return new_action


def mutate_pair(
    pair: GuardActionPair,
    config: JointEvolverConfig
) -> GuardActionPair:
    """Mutate a guard-action pair."""
    new_pair = GuardActionPair(
        guard=copy.deepcopy(pair.guard),
        action=copy.deepcopy(pair.action),
        source_state=pair.source_state,
        target_state=pair.target_state,
        event=pair.event,
        priority=pair.priority
    )

    mutation_type = random.choice(['guard', 'action', 'states', 'both'])

    if mutation_type == 'guard' or mutation_type == 'both':
        new_pair.guard = mutate_guard(new_pair.guard, config.variables,
                                       config.possible_values)

    if mutation_type == 'action' or mutation_type == 'both':
        new_pair.action = mutate_action(new_pair.action, config.variables,
                                         config.possible_values)

    if mutation_type == 'states':
        choice = random.choice(['source', 'target', 'event', 'priority'])
        if choice == 'source':
            new_pair.source_state = random.randint(0, config.n_states - 1)
        elif choice == 'target':
            new_pair.target_state = random.randint(0, config.n_states - 1)
        elif choice == 'event':
            new_pair.event = random.randint(0, config.n_events - 1)
        else:
            new_pair.priority = random.randint(0, 3)

    return new_pair


# =============================================================================
# Causal-Aware Mutation
# =============================================================================

def causal_aware_mutate(
    genome: JointGenome,
    config: JointEvolverConfig
) -> JointGenome:
    """Mutate genome with causal awareness.

    Instead of mutating random pairs, we:
    1. Build causal graph
    2. Select a pair to mutate
    3. Also mutate causally-related pairs to maintain coherence
    """
    new_genome = genome.copy()
    graph = build_causal_graph(new_genome)

    # Select pair to mutate
    pair_idx = random.randint(0, len(new_genome.pairs) - 1)

    # Get mutation cluster
    cluster = get_mutation_cluster(graph, pair_idx, depth=config.cluster_mutation_depth)

    # Get coherent mutation suggestion
    suggestion = suggest_coherent_mutation(
        new_genome, graph, pair_idx,
        config.variables, config.possible_values
    )

    # Apply mutations to cluster
    for idx in cluster:
        if random.random() < config.mutation_rate:
            new_genome.pairs[idx] = mutate_pair(new_genome.pairs[idx], config)

    # Apply suggested coherent mutation
    target_idx, mutation_type = suggestion
    if mutation_type.startswith("add_write:"):
        var = mutation_type.split(":")[1]
        # Add effect that writes this variable
        effect = ActionEffect(var, EffectOp.SET,
                              random.choice(config.possible_values.get(var, [0, 1])))
        new_genome.pairs[target_idx].action.effects.append(effect)

    elif mutation_type.startswith("add_read:"):
        var = mutation_type.split(":")[1]
        # Add clause that reads this variable
        clause = GuardClause(var, GuardOp.EQ,
                             random.choice(config.possible_values.get(var, [0, 1])))
        new_genome.pairs[target_idx].guard.clauses.append(clause)

    return new_genome


def standard_mutate(
    genome: JointGenome,
    config: JointEvolverConfig
) -> JointGenome:
    """Standard mutation without causal awareness."""
    new_genome = genome.copy()

    for i in range(len(new_genome.pairs)):
        if random.random() < config.mutation_rate:
            new_genome.pairs[i] = mutate_pair(new_genome.pairs[i], config)

    return new_genome


# =============================================================================
# Crossover Operators
# =============================================================================

def cluster_crossover(
    parent1: JointGenome,
    parent2: JointGenome,
    config: JointEvolverConfig
) -> Tuple[JointGenome, JointGenome]:
    """Exchange causal clusters between parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    graph1 = build_causal_graph(parent1)
    graph2 = build_causal_graph(parent2)

    # Select a cluster from each parent
    components1 = [graph1.get_connected_component(i) for i in range(len(parent1.pairs))]
    components2 = [graph2.get_connected_component(i) for i in range(len(parent2.pairs))]

    if components1 and components2:
        cluster1 = random.choice(components1)
        cluster2 = random.choice(components2)

        # Exchange clusters (swap pairs at matching indices)
        for idx in cluster1:
            if idx < len(child2.pairs):
                child1.pairs[idx] = copy.deepcopy(parent2.pairs[idx])

        for idx in cluster2:
            if idx < len(child1.pairs):
                child2.pairs[idx] = copy.deepcopy(parent1.pairs[idx])

    return child1, child2


def uniform_crossover(
    parent1: JointGenome,
    parent2: JointGenome,
    config: JointEvolverConfig
) -> Tuple[JointGenome, JointGenome]:
    """Standard uniform crossover."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(min(len(parent1.pairs), len(parent2.pairs))):
        if random.random() < 0.5:
            child1.pairs[i] = copy.deepcopy(parent2.pairs[i])
            child2.pairs[i] = copy.deepcopy(parent1.pairs[i])

    return child1, child2


# =============================================================================
# Selection
# =============================================================================

def tournament_select(
    population: List[JointGenome],
    fitness_list: List[CoherenceFitness],
    tournament_size: int
) -> JointGenome:
    """Tournament selection."""
    candidates = random.sample(range(len(population)), tournament_size)
    best = max(candidates, key=lambda i: fitness_list[i].total_fitness())
    return population[best].copy()


# =============================================================================
# Joint Evolver
# =============================================================================

@dataclass
class EvolutionResult:
    """Results from evolution run."""
    best_genome: JointGenome
    best_fitness: CoherenceFitness
    fitness_history: List[float]
    generation_stats: List[Dict[str, float]]


class JointEvolver:
    """Co-evolves guards and actions for coherent statecharts."""

    def __init__(self, config: JointEvolverConfig):
        self.config = config
        self.population: List[JointGenome] = []
        self.fitness_list: List[CoherenceFitness] = []

    def initialize_population(self):
        """Create initial random population."""
        self.population = [
            random_joint_genome(
                n_states=self.config.n_states,
                n_events=self.config.n_events,
                n_pairs=self.config.n_pairs,
                variables=self.config.variables,
                possible_values=self.config.possible_values
            )
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(self, scenarios: List[Scenario] = None):
        """Evaluate fitness of all individuals."""
        self.fitness_list = [
            compute_coherence_fitness(genome, scenarios)
            for genome in self.population
        ]

    def evolve_generation(self, scenarios: List[Scenario] = None):
        """Evolve one generation."""
        # Sort by fitness
        sorted_indices = sorted(
            range(len(self.population)),
            key=lambda i: self.fitness_list[i].total_fitness(),
            reverse=True
        )

        # Elite selection
        new_population = [
            self.population[i].copy()
            for i in sorted_indices[:self.config.elite_size]
        ]

        # Generate rest of population
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = tournament_select(self.population, self.fitness_list,
                                         self.config.tournament_size)
            parent2 = tournament_select(self.population, self.fitness_list,
                                         self.config.tournament_size)

            # Crossover
            if random.random() < self.config.crossover_rate:
                if self.config.cluster_crossover:
                    child1, child2 = cluster_crossover(parent1, parent2, self.config)
                else:
                    child1, child2 = uniform_crossover(parent1, parent2, self.config)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < self.config.causal_mutation_prob:
                child1 = causal_aware_mutate(child1, self.config)
                child2 = causal_aware_mutate(child2, self.config)
            else:
                child1 = standard_mutate(child1, self.config)
                child2 = standard_mutate(child2, self.config)

            new_population.extend([child1, child2])

        # Trim to population size
        self.population = new_population[:self.config.population_size]

        # Re-evaluate
        self.evaluate_population(scenarios)

    def run(
        self,
        scenarios: List[Scenario] = None,
        verbose: bool = True
    ) -> EvolutionResult:
        """Run full evolution."""
        self.initialize_population()
        self.evaluate_population(scenarios)

        fitness_history = []
        generation_stats = []

        for gen in range(self.config.n_generations):
            # Get stats
            best_idx = max(range(len(self.fitness_list)),
                           key=lambda i: self.fitness_list[i].total_fitness())
            best_fitness = self.fitness_list[best_idx]
            avg_fitness = sum(f.total_fitness() for f in self.fitness_list) / len(self.fitness_list)

            fitness_history.append(best_fitness.total_fitness())
            generation_stats.append({
                'best': best_fitness.total_fitness(),
                'avg': avg_fitness,
                'causal': best_fitness.causal_coherence,
                'temporal': best_fitness.temporal_coherence,
                'behavioral': best_fitness.behavioral_coherence,
                'structural': best_fitness.structural_coherence
            })

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: best={best_fitness.total_fitness():.3f} "
                      f"avg={avg_fitness:.3f} "
                      f"causal={best_fitness.causal_coherence:.2f} "
                      f"behav={best_fitness.behavioral_coherence:.2f}")

            # Evolve
            self.evolve_generation(scenarios)

        # Final best
        best_idx = max(range(len(self.fitness_list)),
                       key=lambda i: self.fitness_list[i].total_fitness())

        return EvolutionResult(
            best_genome=self.population[best_idx],
            best_fitness=self.fitness_list[best_idx],
            fitness_history=fitness_history,
            generation_stats=generation_stats
        )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate joint evolution."""
    print("=" * 60)
    print("JOINT GUARD-ACTION EVOLUTION DEMO")
    print("=" * 60)

    config = JointEvolverConfig(
        population_size=30,
        n_generations=50,
        mutation_rate=0.3,
        crossover_rate=0.5,
        elite_size=3,
        causal_mutation_prob=0.7,
        cluster_crossover=True,
        variables=["turn", "phase", "score", "ready"],
        possible_values={
            "turn": [0, 1, 2],
            "phase": ["init", "play", "end"],
            "score": [0, 1, 2, 3],
            "ready": [True, False]
        },
        n_states=4,
        n_events=3,
        n_pairs=8
    )

    # Create test scenarios
    scenarios = [
        Scenario(
            initial_context={"turn": 0, "phase": "init", "score": 0, "ready": False},
            events=[0, 1, 2],
            expected_final_context={"ready": True}
        ),
        Scenario(
            initial_context={"turn": 1, "phase": "play", "score": 0, "ready": True},
            events=[1, 1],
            expected_final_context={"score": 2}
        ),
    ]

    evolver = JointEvolver(config)
    result = evolver.run(scenarios, verbose=True)

    print(f"\n--- Final Results ---")
    print(f"Best fitness: {result.best_fitness.total_fitness():.3f}")
    print(f"  Causal coherence:     {result.best_fitness.causal_coherence:.3f}")
    print(f"  Temporal coherence:   {result.best_fitness.temporal_coherence:.3f}")
    print(f"  Behavioral coherence: {result.best_fitness.behavioral_coherence:.3f}")
    print(f"  Structural coherence: {result.best_fitness.structural_coherence:.3f}")

    print(f"\n--- Best Genome ---")
    for i, pair in enumerate(result.best_genome.pairs):
        print(f"{i}: s{pair.source_state}--[e{pair.event}]-->s{pair.target_state}")
        print(f"   {pair.guard.to_string()} -> {pair.action.to_string()}")

    # Improvement over generations
    if result.fitness_history:
        improvement = (result.fitness_history[-1] - result.fitness_history[0]) / max(0.01, result.fitness_history[0])
        print(f"\n--- Improvement ---")
        print(f"Initial: {result.fitness_history[0]:.3f}")
        print(f"Final:   {result.fitness_history[-1]:.3f}")
        print(f"Improvement: {improvement*100:.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
