"""
Unified Topology Evolution Framework

Ties together all components into a single, configurable evolution system:
- GameEnvironment abstraction for multiple games
- Parallel evaluation with fitness caching
- Multi-objective fitness (NSGA-II)
- Guard and signal evolution
- Configurable operators and parameters

Usage:
    from experiments.exp_topology_evolution.framework import TopologyEvolver
    from experiments.exp_topology_evolution.environments import TicTacToeEnv

    evolver = TopologyEvolver(TicTacToeEnv)
    best = evolver.evolve(n_generations=100)
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import Type, List, Tuple, Dict, Callable, Optional, Any
from enum import Enum

from .environments import GameEnvironment, TicTacToeEnv, Connect4Env
from .fitness import (
    FitnessComponents, MultiObjectiveFitness, compute_fitness_components
)
from .guards import GuardExpr, create_random_guard, mutate_guard


# =============================================================================
# Evolution Configuration
# =============================================================================

class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    NSGA2 = "nsga2"
    ROULETTE = "roulette"


@dataclass
class EvolutionConfig:
    """Configuration for topology evolution."""
    # Population
    population_size: int = 50
    elite_size: int = 5
    n_generations: int = 100

    # Genome limits
    max_states: int = 20
    min_states: int = 3

    # Mutation rates
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7

    # Mutation operator weights
    add_state_weight: float = 0.15
    remove_state_weight: float = 0.10
    toggle_type_weight: float = 0.10
    toggle_history_weight: float = 0.05
    add_transition_weight: float = 0.20
    remove_transition_weight: float = 0.10
    mutate_transition_weight: float = 0.30

    # Selection
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3

    # Fitness
    use_multi_objective: bool = True
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 1.0,
        "parsimony": 0.1,
        "hierarchy_score": 0.05,
        "parallelism_score": 0.05
    })

    # Dataset
    dataset_size: int = 500

    # Parallel
    n_workers: Optional[int] = None
    use_caching: bool = True

    # Logging
    log_every: int = 10
    verbose: bool = True


# =============================================================================
# Evolution Statistics
# =============================================================================

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    best_accuracy: float
    avg_fitness: float
    avg_accuracy: float
    avg_states: float
    n_unique: int
    elapsed_time: float


@dataclass
class EvolutionStats:
    """Statistics for entire evolution run."""
    generations: List[GenerationStats] = field(default_factory=list)
    total_time: float = 0.0
    total_evaluations: int = 0
    cache_hit_rate: float = 0.0

    def add_generation(self, stats: GenerationStats):
        self.generations.append(stats)

    def summary(self) -> str:
        """Generate summary string."""
        if not self.generations:
            return "No generations completed"

        final = self.generations[-1]
        return (
            f"Evolution Summary:\n"
            f"  Generations: {len(self.generations)}\n"
            f"  Best Fitness: {final.best_fitness:.4f}\n"
            f"  Best Accuracy: {final.best_accuracy:.3f}\n"
            f"  Avg States: {final.avg_states:.1f}\n"
            f"  Total Time: {self.total_time:.2f}s\n"
            f"  Cache Hit Rate: {self.cache_hit_rate:.1%}"
        )


# =============================================================================
# Unified Topology Evolver
# =============================================================================

class TopologyEvolver:
    """
    Unified topology evolution framework.

    Combines all evolution components into a single, configurable system.
    """

    def __init__(
        self,
        env_class: Type[GameEnvironment],
        config: EvolutionConfig = None
    ):
        """
        Initialize evolver.

        Args:
            env_class: Game environment class
            config: Evolution configuration
        """
        self.env_class = env_class
        self.config = config or EvolutionConfig()

        # Multi-objective fitness
        self.fitness_evaluator = MultiObjectiveFitness(
            weights=self.config.fitness_weights,
            use_pareto=(self.config.selection_method == SelectionMethod.NSGA2)
        )

        # Statistics
        self.stats = EvolutionStats()

        # Import here to avoid circular imports
        from .evolve import (
            create_genome_for_game, mutate, crossover, evaluate_fitness
        )
        from .parallel import ParallelEvaluator, EvolutionConfig as ParallelConfig

        self._create_genome = lambda: create_genome_for_game(
            env_class, self.config.max_states
        )
        self._mutate = mutate
        self._crossover = crossover
        self._evaluate = evaluate_fitness

        # Setup parallel evaluator if caching enabled
        if self.config.use_caching:
            par_config = ParallelConfig(
                n_workers=self.config.n_workers,
                use_caching=True
            )
            self._parallel = ParallelEvaluator(par_config)
        else:
            self._parallel = None

    def generate_dataset(self) -> List[Tuple]:
        """Generate evaluation dataset."""
        from .evolve import generate_env_dataset
        return generate_env_dataset(self.env_class, self.config.dataset_size)

    def initialize_population(self) -> List:
        """Create initial population."""
        population = []
        attempts = 0
        max_attempts = self.config.population_size * 3

        while len(population) < self.config.population_size and attempts < max_attempts:
            genome = self._create_genome()
            if genome.is_valid():
                population.append(genome)
            attempts += 1

        return population

    def evaluate_population(
        self,
        population: List,
        dataset: List[Tuple]
    ) -> Tuple[List[float], List[FitnessComponents]]:
        """
        Evaluate fitness for entire population.

        Returns (fitness_list, components_list).
        """
        if self._parallel:
            # Use parallel evaluator with caching
            results = self._parallel.evaluate_population(
                population, dataset, self.env_class, self.config.max_states
            )
            fitness_list = []
            components_list = []
            for genome, (raw_fitness, accuracy) in zip(population, results):
                components = compute_fitness_components(
                    genome, accuracy, self.config.max_states
                )
                fitness = components.weighted_sum(self.config.fitness_weights)
                fitness_list.append(fitness)
                components_list.append(components)
            return fitness_list, components_list
        else:
            # Sequential evaluation
            fitness_list = []
            components_list = []
            for genome in population:
                raw_fitness, accuracy = self._evaluate(
                    genome, dataset, self.env_class, self.config.max_states
                )
                components = compute_fitness_components(
                    genome, accuracy, self.config.max_states
                )
                fitness = components.weighted_sum(self.config.fitness_weights)
                fitness_list.append(fitness)
                components_list.append(components)
            return fitness_list, components_list

    def select_parents(
        self,
        population: List,
        fitness_list: List[float],
        components_list: List[FitnessComponents]
    ) -> List:
        """Select parents for next generation."""
        if self.config.selection_method == SelectionMethod.NSGA2:
            # Multi-objective selection
            indices = self.fitness_evaluator.select(
                population, components_list, self.config.population_size // 2
            )
            return [population[i] for i in indices]

        elif self.config.selection_method == SelectionMethod.TOURNAMENT:
            # Tournament selection
            parents = []
            n_parents = self.config.population_size // 2
            for _ in range(n_parents):
                candidates = random.sample(
                    list(range(len(population))),
                    min(self.config.tournament_size, len(population))
                )
                winner = max(candidates, key=lambda i: fitness_list[i])
                parents.append(population[winner])
            return parents

        else:  # ROULETTE
            # Roulette wheel selection
            total = sum(fitness_list)
            if total <= 0:
                return random.sample(population, self.config.population_size // 2)

            probs = [f / total for f in fitness_list]
            parents = random.choices(
                population,
                weights=probs,
                k=self.config.population_size // 2
            )
            return parents

    def create_offspring(self, parents: List) -> List:
        """Create offspring from parents."""
        offspring = []

        while len(offspring) < self.config.population_size - self.config.elite_size:
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
                child = self._crossover(p1, p2)
            else:
                parent = random.choice(parents)
                child = copy.deepcopy(parent)

            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)

            if child.is_valid():
                offspring.append(child)

        return offspring

    def evolve(
        self,
        n_generations: int = None,
        callback: Callable = None
    ):
        """
        Run evolution.

        Args:
            n_generations: Override config n_generations
            callback: Called each generation with (gen, population, best)

        Returns:
            Best genome found
        """
        n_gen = n_generations or self.config.n_generations
        self.stats = EvolutionStats()
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print(f"TOPOLOGY EVOLUTION: {self.env_class.config.name}")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {n_gen}")
            print(f"Max States: {self.config.max_states}")
            print(f"Selection: {self.config.selection_method.value}")
            print("-" * 60)

        # Generate dataset
        if self.config.verbose:
            print("Generating dataset...")
        dataset = self.generate_dataset()
        if self.config.verbose:
            print(f"Generated {len(dataset)} positions")

        # Initialize population
        if self.config.verbose:
            print("Initializing population...")
        population = self.initialize_population()
        if self.config.verbose:
            print(f"Created {len(population)} valid genomes")

        # Evaluate initial population
        fitness_list, components_list = self.evaluate_population(population, dataset)
        for genome, fitness, comp in zip(population, fitness_list, components_list):
            genome.fitness = fitness
            genome.accuracy = comp.accuracy

        # Track best ever
        best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
        best_ever = copy.deepcopy(population[best_idx])

        if self.config.verbose:
            print("-" * 60)

        # Evolution loop
        for gen in range(n_gen):
            gen_start = time.time()

            # Sort by fitness
            sorted_pairs = sorted(
                zip(population, fitness_list, components_list),
                key=lambda x: x[1],
                reverse=True
            )
            population = [p for p, _, _ in sorted_pairs]
            fitness_list = [f for _, f, _ in sorted_pairs]
            components_list = [c for _, _, c in sorted_pairs]

            # Elitism
            elite = population[:self.config.elite_size]

            # Select parents
            parents = self.select_parents(population, fitness_list, components_list)

            # Create offspring
            offspring = self.create_offspring(parents)

            # New population
            population = elite + offspring

            # Evaluate
            fitness_list, components_list = self.evaluate_population(population, dataset)
            for genome, fitness, comp in zip(population, fitness_list, components_list):
                genome.fitness = fitness
                genome.accuracy = comp.accuracy

            # Track best
            best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
            if fitness_list[best_idx] > best_ever.fitness:
                best_ever = copy.deepcopy(population[best_idx])

            # Statistics
            gen_stats = GenerationStats(
                generation=gen,
                best_fitness=fitness_list[best_idx],
                best_accuracy=components_list[best_idx].accuracy,
                avg_fitness=sum(fitness_list) / len(fitness_list),
                avg_accuracy=sum(c.accuracy for c in components_list) / len(components_list),
                avg_states=sum(g.n_states for g in population) / len(population),
                n_unique=len(set(g.n_states for g in population)),
                elapsed_time=time.time() - gen_start
            )
            self.stats.add_generation(gen_stats)

            # Callback
            if callback:
                callback(gen, population, best_ever)

            # Progress report
            if self.config.verbose and (gen % self.config.log_every == 0 or gen == n_gen - 1):
                print(
                    f"Gen {gen:3d} | "
                    f"Best: {gen_stats.best_fitness:.4f} (acc={gen_stats.best_accuracy:.3f}) | "
                    f"Avg: {gen_stats.avg_fitness:.4f} | "
                    f"States: {gen_stats.avg_states:.1f}"
                )

        # Final stats
        self.stats.total_time = time.time() - start_time
        if self._parallel:
            par_stats = self._parallel.get_stats()
            self.stats.total_evaluations = par_stats["total_evaluations"]
            self.stats.cache_hit_rate = par_stats["cache_hit_rate"]

        if self.config.verbose:
            print("-" * 60)
            print(self.stats.summary())
            print("=" * 60)

        return best_ever

    def evolve_with_guard_coevolution(
        self,
        n_generations: int = None
    ):
        """
        Evolve topology and guards together.

        This is an experimental extension that co-evolves guard expressions
        alongside the statechart topology.
        """
        # TODO: Implement guard co-evolution
        # For now, just run standard evolution
        return self.evolve(n_generations)


# =============================================================================
# CLI Interface
# =============================================================================

def run_evolution(
    game: str = "tictactoe",
    n_generations: int = 100,
    population_size: int = 50,
    max_states: int = 20,
    verbose: bool = True
):
    """
    Run evolution from command line.

    Args:
        game: Game name ("tictactoe" or "connect4")
        n_generations: Number of generations
        population_size: Population size
        max_states: Maximum states per genome
        verbose: Print progress
    """
    from .environments import ENVIRONMENTS

    if game.lower() not in ENVIRONMENTS:
        print(f"Unknown game: {game}")
        print(f"Available: {list(ENVIRONMENTS.keys())}")
        return None

    env_class = ENVIRONMENTS[game.lower()]

    config = EvolutionConfig(
        population_size=population_size,
        n_generations=n_generations,
        max_states=max_states,
        verbose=verbose
    )

    evolver = TopologyEvolver(env_class, config)
    best = evolver.evolve()

    return best


# =============================================================================
# Testing
# =============================================================================

def test_framework():
    """Test the unified evolution framework."""
    print("=" * 60)
    print("FRAMEWORK TEST")
    print("=" * 60)

    # Quick test with small parameters
    config = EvolutionConfig(
        population_size=20,
        n_generations=10,
        max_states=8,
        dataset_size=100,
        log_every=5,
        verbose=True
    )

    evolver = TopologyEvolver(TicTacToeEnv, config)
    best = evolver.evolve()

    print(f"\nBest genome: {best.n_states} states, accuracy={best.accuracy:.3f}")

    # Test with NSGA-II
    print("\n" + "-" * 60)
    print("Testing NSGA-II selection...")

    config2 = EvolutionConfig(
        population_size=20,
        n_generations=10,
        max_states=8,
        dataset_size=100,
        selection_method=SelectionMethod.NSGA2,
        log_every=5,
        verbose=True
    )

    evolver2 = TopologyEvolver(TicTacToeEnv, config2)
    best2 = evolver2.evolve()

    print(f"\nBest genome (NSGA-II): {best2.n_states} states, accuracy={best2.accuracy:.3f}")

    print("\nFramework test complete!")
    return best, best2


if __name__ == "__main__":
    test_framework()
