"""
Parallel Evolution Infrastructure

Provides scalable evaluation of statechart genomes through:
- Fitness caching (avoid re-evaluating unchanged genomes)
- Parallel dataset generation
- Parallel fitness evaluation
- Batch processing
"""

import hashlib
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Dict, Type, Optional, Callable
import time

from .environments import GameEnvironment, TicTacToeEnv, Connect4Env


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for parallel evolution."""
    n_workers: int = None           # Number of parallel workers (None = CPU count)
    use_caching: bool = True        # Cache fitness values
    cache_size: int = 10000         # Max cache entries
    batch_size: int = 50            # Batch size for evaluation
    timeout_per_genome: float = 5.0 # Timeout per genome evaluation (seconds)


# =============================================================================
# Genome Hashing for Caching
# =============================================================================

def hash_genome(genome) -> str:
    """
    Create a hash of a genome for caching.

    Uses structural hash based on topology, not fitness.
    """
    # Create hashable representation
    key_parts = [
        genome.n_states,
        tuple(genome.parent),
        tuple(int(t) for t in genome.state_type),
        tuple(genome.has_history),
        tuple((t.src, t.tgt, t.event, t.guard_idx) for t in genome.transitions),
        genome.initial_state,
        genome.n_events,
        genome.n_guards,
    ]

    # Hash the pickled representation
    data = pickle.dumps(key_parts)
    return hashlib.md5(data).hexdigest()


# =============================================================================
# Fitness Cache
# =============================================================================

class FitnessCache:
    """LRU cache for genome fitness values."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.access_order: List[str] = []

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, genome) -> Optional[Tuple[float, float]]:
        """Get cached fitness for genome. Returns (fitness, accuracy) or None."""
        key = hash_genome(genome)
        if key in self.cache:
            self.hits += 1
            # Update access order (move to end)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, genome, fitness: float, accuracy: float):
        """Cache fitness for genome."""
        key = hash_genome(genome)

        # Evict oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = (fitness, accuracy)
        self.access_order.append(key)

    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


# =============================================================================
# Parallel Dataset Generation
# =============================================================================

def _generate_single_position(args) -> Optional[Tuple]:
    """Generate a single random position. Worker function."""
    env_class, max_moves = args
    env = env_class.generate_random_position(max_moves)
    if env.is_terminal():
        return None
    return (env, env.get_legal_mask())


def generate_dataset_parallel(
    env_class: Type[GameEnvironment],
    n_positions: int,
    n_workers: int = None,
    max_moves: int = None
) -> List[Tuple]:
    """
    Generate dataset in parallel.

    Args:
        env_class: Game environment class
        n_positions: Number of positions to generate
        n_workers: Number of parallel workers
        max_moves: Max moves for random position generation
    """
    n_workers = n_workers or mp.cpu_count()

    # Generate more than needed to account for terminal positions
    n_attempts = int(n_positions * 1.5)

    args = [(env_class, max_moves) for _ in range(n_attempts)]

    dataset = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(_generate_single_position, args)
        for result in results:
            if result is not None:
                dataset.append(result)
                if len(dataset) >= n_positions:
                    break

    return dataset[:n_positions]


# =============================================================================
# Parallel Fitness Evaluation
# =============================================================================

def _evaluate_single_genome(args) -> Tuple[int, float, float]:
    """Evaluate a single genome. Worker function."""
    from .evolve import evaluate_fitness

    idx, genome, dataset, env_class, max_states = args
    try:
        fitness, accuracy = evaluate_fitness(genome, dataset, env_class, max_states)
        return (idx, fitness, accuracy)
    except Exception as e:
        print(f"Error evaluating genome {idx}: {e}")
        return (idx, 0.0, 0.0)


class ParallelEvaluator:
    """
    Parallel fitness evaluator with caching.
    """

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.n_workers = self.config.n_workers or mp.cpu_count()
        self.cache = FitnessCache(self.config.cache_size) if self.config.use_caching else None

        # Statistics
        self.total_evaluations = 0
        self.cached_evaluations = 0
        self.total_time = 0.0

    def evaluate_population(
        self,
        population: List,
        dataset: List[Tuple],
        env_class: Type[GameEnvironment],
        max_states: int
    ) -> List[Tuple[float, float]]:
        """
        Evaluate fitness for entire population.

        Returns list of (fitness, accuracy) tuples.
        """
        start_time = time.time()
        results = [None] * len(population)

        # Check cache first
        uncached_indices = []
        for i, genome in enumerate(population):
            if self.cache:
                cached = self.cache.get(genome)
                if cached is not None:
                    results[i] = cached
                    self.cached_evaluations += 1
                    continue
            uncached_indices.append(i)

        self.total_evaluations += len(uncached_indices)

        # Evaluate uncached genomes
        if uncached_indices:
            if len(uncached_indices) <= 2 or self.n_workers <= 1:
                # Sequential evaluation for small batches
                for i in uncached_indices:
                    from .evolve import evaluate_fitness
                    fitness, accuracy = evaluate_fitness(
                        population[i], dataset, env_class, max_states
                    )
                    results[i] = (fitness, accuracy)
                    if self.cache:
                        self.cache.put(population[i], fitness, accuracy)
            else:
                # Parallel evaluation
                args = [
                    (i, population[i], dataset, env_class, max_states)
                    for i in uncached_indices
                ]

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    for idx, fitness, accuracy in executor.map(_evaluate_single_genome, args):
                        results[idx] = (fitness, accuracy)
                        if self.cache:
                            self.cache.put(population[idx], fitness, accuracy)

        self.total_time += time.time() - start_time
        return results

    def get_stats(self) -> Dict:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "cached_evaluations": self.cached_evaluations,
            "cache_hit_rate": self.cache.hit_rate() if self.cache else 0.0,
            "total_time": self.total_time,
            "avg_time_per_eval": self.total_time / max(1, self.total_evaluations),
        }


# =============================================================================
# Parallel Evolution Loop
# =============================================================================

class ParallelEvolver:
    """
    Complete parallel evolution system.
    """

    def __init__(
        self,
        env_class: Type[GameEnvironment],
        config: EvolutionConfig = None
    ):
        self.env_class = env_class
        self.config = config or EvolutionConfig()
        self.evaluator = ParallelEvaluator(self.config)

    def evolve(
        self,
        n_generations: int = 100,
        population_size: int = 50,
        elite_size: int = 5,
        max_states: int = 20,
        dataset_size: int = 500,
        callback: Callable = None,
        verbose: bool = True
    ):
        """
        Run parallel evolution.

        Args:
            n_generations: Number of generations
            population_size: Population size
            elite_size: Number of elites to preserve
            max_states: Maximum states per genome
            dataset_size: Number of positions for evaluation
            callback: Optional callback(gen, population, best) called each generation
            verbose: Print progress
        """
        import random
        import copy
        from .evolve import (
            create_genome_for_game, mutate, crossover,
            tournament_select
        )

        config = self.env_class.config

        if verbose:
            print(f"Game: {config.name}")
            print(f"Workers: {self.config.n_workers or mp.cpu_count()}")
            print(f"Caching: {self.config.use_caching}")

        # Generate dataset (can also be parallel)
        if verbose:
            print("Generating dataset...")

        start = time.time()
        if self.config.n_workers and self.config.n_workers > 1:
            dataset = generate_dataset_parallel(
                self.env_class, dataset_size, self.config.n_workers
            )
        else:
            from .evolve import generate_env_dataset
            dataset = generate_env_dataset(self.env_class, dataset_size)

        if verbose:
            print(f"Generated {len(dataset)} positions in {time.time()-start:.2f}s")

        # Initialize population
        if verbose:
            print("Initializing population...")

        population = [
            create_genome_for_game(self.env_class, max_states)
            for _ in range(population_size)
        ]

        # Evaluate initial population
        fitnesses = self.evaluator.evaluate_population(
            population, dataset, self.env_class, max_states
        )
        for g, (f, a) in zip(population, fitnesses):
            g.fitness, g.accuracy = f, a

        best_ever = max(population, key=lambda g: g.fitness)

        if verbose:
            print(f"\nStarting evolution for {n_generations} generations...")
            print("=" * 60)

        for gen in range(n_generations):
            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Keep elite
            new_population = population[:elite_size]

            # Generate offspring
            while len(new_population) < population_size:
                if random.random() < 0.7:
                    p1 = tournament_select(population)
                    p2 = tournament_select(population)
                    child = crossover(p1, p2)
                else:
                    parent = tournament_select(population)
                    child = copy.deepcopy(parent)

                child = mutate(child)

                if child.is_valid():
                    new_population.append(child)

            # Evaluate new population (parallel with caching)
            fitnesses = self.evaluator.evaluate_population(
                new_population, dataset, self.env_class, max_states
            )
            for g, (f, a) in zip(new_population, fitnesses):
                g.fitness, g.accuracy = f, a

            population = new_population

            # Track best
            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(gen_best)

            # Callback
            if callback:
                callback(gen, population, best_ever)

            # Progress report
            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                avg_fitness = sum(g.fitness for g in population) / len(population)
                avg_states = sum(g.n_states for g in population) / len(population)
                stats = self.evaluator.get_stats()
                print(f"Gen {gen:3d} | Best: {gen_best.fitness:.4f} (acc={gen_best.accuracy:.3f}) "
                      f"| Cache: {stats['cache_hit_rate']:.1%}")

        if verbose:
            print("=" * 60)
            stats = self.evaluator.get_stats()
            print(f"\nEvolution complete:")
            print(f"  Best fitness: {best_ever.fitness:.4f}")
            print(f"  Best accuracy: {best_ever.accuracy:.3f}")
            print(f"  Total evaluations: {stats['total_evaluations']}")
            print(f"  Cached evaluations: {stats['cached_evaluations']}")
            print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
            print(f"  Total time: {stats['total_time']:.2f}s")

        return best_ever


# =============================================================================
# Testing
# =============================================================================

def test_parallel():
    """Test parallel evolution infrastructure."""
    print("=" * 60)
    print("PARALLEL EVOLUTION TEST")
    print("=" * 60)

    config = EvolutionConfig(
        n_workers=4,
        use_caching=True,
        cache_size=1000
    )

    evolver = ParallelEvolver(TicTacToeEnv, config)

    best = evolver.evolve(
        n_generations=30,
        population_size=40,
        max_states=10,
        dataset_size=300,
        verbose=True
    )

    print(f"\nBest genome: {best.n_states} states, accuracy={best.accuracy:.3f}")
    return best


if __name__ == "__main__":
    test_parallel()
