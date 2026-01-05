"""
Guard Fine-tuning with Frozen Topology

KEY INSIGHT:
After transferring topology, we only need to learn the GUARD parameters
for the new game. The structural decisions (AND/OR, hierarchy, history)
have already been made.

This dramatically reduces the search space:
- Full evolution: O(states * types * hierarchy * transitions * guards)
- Guard fine-tuning: O(transitions * guard_params)

Process:
1. FREEZE: Lock topology (states, types, hierarchy, transition structure)
2. FINE-TUNE: Evolve only guard parameters
3. COMPARE: Measure convergence speed vs full evolution
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

from exp_topology_evolution.environments import (
    GameEnvironment, TicTacToeEnv, Connect4Env, OthelloEnv
)
from exp_topology_evolution.evolve import (
    StatechartGenome, StateType, Transition, create_random_genome,
    evaluate_fitness, generate_env_dataset
)
from exp_topology_evolution.fitness import compute_fitness_components


# =============================================================================
# Frozen Topology Wrapper
# =============================================================================

@dataclass
class FrozenTopologyGenome:
    """
    Genome with frozen topology - only guards can change.

    Wraps a StatechartGenome but prevents structural mutations.
    """
    base_genome: StatechartGenome

    # Only these can evolve
    guard_indices: List[int] = field(default_factory=list)

    # Fitness tracking
    fitness: float = 0.0
    accuracy: float = 0.0

    def __post_init__(self):
        """Initialize guard indices from base genome."""
        if not self.guard_indices:
            self.guard_indices = [t.guard_idx for t in self.base_genome.transitions]

    @property
    def n_states(self) -> int:
        return self.base_genome.n_states

    def copy(self) -> 'FrozenTopologyGenome':
        """Create a copy with shared frozen structure."""
        new = FrozenTopologyGenome(
            base_genome=self.base_genome,  # Shared reference - frozen
            guard_indices=self.guard_indices.copy()  # Only guards copied
        )
        new.fitness = self.fitness
        new.accuracy = self.accuracy
        return new

    def to_genome(self) -> StatechartGenome:
        """Convert to full genome with current guards."""
        genome = StatechartGenome(
            n_states=self.base_genome.n_states,
            parent=self.base_genome.parent,  # Shared
            state_type=self.base_genome.state_type,  # Shared
            has_history=self.base_genome.has_history,  # Shared
            transitions=[
                Transition(t.src, t.tgt, t.event, self.guard_indices[i])
                for i, t in enumerate(self.base_genome.transitions)
            ],
            initial_state=self.base_genome.initial_state,
            n_events=self.base_genome.n_events,
            n_guards=self.base_genome.n_guards,
            max_states=self.base_genome.max_states
        )
        genome.fitness = self.fitness
        genome.accuracy = self.accuracy
        return genome

    def is_valid(self) -> bool:
        """Always valid if base is valid."""
        return self.base_genome.is_valid()


def freeze_topology(genome: StatechartGenome) -> FrozenTopologyGenome:
    """
    Freeze a genome's topology, allowing only guard evolution.

    Args:
        genome: Source genome to freeze

    Returns:
        FrozenTopologyGenome wrapper
    """
    return FrozenTopologyGenome(base_genome=genome)


# =============================================================================
# Guard-Only Mutations
# =============================================================================

def mutate_guards(
    frozen: FrozenTopologyGenome,
    mutation_rate: float = 0.3,
    n_guards: int = 4
) -> FrozenTopologyGenome:
    """
    Mutate only guard indices (topology is frozen).

    Args:
        frozen: Frozen genome
        mutation_rate: Probability of mutating each guard
        n_guards: Number of available guard types

    Returns:
        New frozen genome with mutated guards
    """
    new = frozen.copy()

    for i in range(len(new.guard_indices)):
        if random.random() < mutation_rate:
            # Either random change or incremental
            if random.random() < 0.5:
                new.guard_indices[i] = random.randint(0, n_guards - 1)
            else:
                # Incremental mutation
                delta = random.choice([-1, 1])
                new.guard_indices[i] = (new.guard_indices[i] + delta) % n_guards

    return new


def crossover_guards(
    frozen1: FrozenTopologyGenome,
    frozen2: FrozenTopologyGenome
) -> FrozenTopologyGenome:
    """
    Crossover guard indices between two frozen genomes.

    Both must share the same base topology!
    """
    assert frozen1.base_genome is frozen2.base_genome, \
        "Crossover requires same frozen topology"

    new = frozen1.copy()
    n_guards = len(new.guard_indices)

    # Two-point crossover
    if n_guards >= 3:
        pt1 = random.randint(0, n_guards - 2)
        pt2 = random.randint(pt1 + 1, n_guards - 1)
        new.guard_indices[pt1:pt2] = frozen2.guard_indices[pt1:pt2]
    else:
        # Uniform crossover for small genomes
        for i in range(n_guards):
            if random.random() < 0.5:
                new.guard_indices[i] = frozen2.guard_indices[i]

    return new


# =============================================================================
# Guard Fine-tuner
# =============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for guard fine-tuning."""
    population_size: int = 30
    n_generations: int = 50
    elite_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    dataset_size: int = 500
    target_accuracy: float = 0.9
    verbose: bool = True
    log_every: int = 10


@dataclass
class FinetuneResult:
    """Result of guard fine-tuning."""
    best_genome: StatechartGenome
    final_accuracy: float
    generations_to_target: int  # -1 if not reached
    total_generations: int
    elapsed_time: float
    accuracy_history: List[float] = field(default_factory=list)


class GuardFinetuner:
    """
    Fine-tune guards for transferred topology.

    Usage:
        finetuner = GuardFinetuner(transferred_genome, Connect4Env)
        result = finetuner.finetune()
    """

    def __init__(
        self,
        base_genome: StatechartGenome,
        env_class: Type[GameEnvironment],
        config: FinetuneConfig = None
    ):
        self.base_genome = base_genome
        self.env_class = env_class
        self.config = config or FinetuneConfig()

        # Ensure genome is valid
        if not base_genome.is_valid():
            raise ValueError("Base genome must be valid")

    def finetune(self) -> FinetuneResult:
        """
        Run guard fine-tuning evolution.

        Returns:
            FinetuneResult with best genome and statistics
        """
        config = self.config
        start_time = time.time()

        if config.verbose:
            print("=" * 60)
            print(f"GUARD FINE-TUNING: {self.env_class.config.name}")
            print("=" * 60)
            print(f"Topology: {self.base_genome.n_states} states, "
                  f"{len(self.base_genome.transitions)} transitions")
            print(f"Population: {config.population_size}")
            print(f"Target accuracy: {config.target_accuracy:.1%}")
            print("-" * 60)

        # Generate dataset
        dataset = generate_env_dataset(self.env_class, config.dataset_size)

        # Initialize population with frozen topology
        population = [freeze_topology(self.base_genome) for _ in range(config.population_size)]

        # Randomize initial guards
        for frozen in population:
            for i in range(len(frozen.guard_indices)):
                frozen.guard_indices[i] = random.randint(0, self.base_genome.n_guards - 1)

        # Evaluate initial population
        for frozen in population:
            genome = frozen.to_genome()
            _, accuracy = evaluate_fitness(
                genome, dataset, self.env_class, self.base_genome.max_states
            )
            frozen.fitness = accuracy
            frozen.accuracy = accuracy

        # Track results
        accuracy_history = []
        generations_to_target = -1
        best_ever = max(population, key=lambda f: f.fitness)

        # Evolution loop
        for gen in range(config.n_generations):
            # Sort by fitness
            population.sort(key=lambda f: f.fitness, reverse=True)

            # Track best
            if population[0].fitness > best_ever.fitness:
                best_ever = population[0].copy()

            accuracy_history.append(best_ever.accuracy)

            # Check target
            if generations_to_target < 0 and best_ever.accuracy >= config.target_accuracy:
                generations_to_target = gen

            # Progress report
            if config.verbose and (gen % config.log_every == 0 or gen == config.n_generations - 1):
                print(f"Gen {gen:3d} | Best: {population[0].accuracy:.3f} | "
                      f"Avg: {sum(f.accuracy for f in population)/len(population):.3f}")

            # Elite
            new_pop = [f.copy() for f in population[:config.elite_size]]

            # Generate offspring
            while len(new_pop) < config.population_size:
                if random.random() < config.crossover_rate:
                    p1 = random.choice(population[:config.population_size//2])
                    p2 = random.choice(population[:config.population_size//2])
                    child = crossover_guards(p1, p2)
                else:
                    parent = random.choice(population[:config.population_size//2])
                    child = parent.copy()

                child = mutate_guards(child, config.mutation_rate, self.base_genome.n_guards)
                new_pop.append(child)

            # Evaluate
            for frozen in new_pop:
                genome = frozen.to_genome()
                _, accuracy = evaluate_fitness(
                    genome, dataset, self.env_class, self.base_genome.max_states
                )
                frozen.fitness = accuracy
                frozen.accuracy = accuracy

            population = new_pop

        elapsed = time.time() - start_time

        if config.verbose:
            print("-" * 60)
            print(f"Final accuracy: {best_ever.accuracy:.3f}")
            print(f"Generations to {config.target_accuracy:.0%}: "
                  f"{generations_to_target if generations_to_target >= 0 else 'Not reached'}")
            print(f"Elapsed: {elapsed:.2f}s")
            print("=" * 60)

        return FinetuneResult(
            best_genome=best_ever.to_genome(),
            final_accuracy=best_ever.accuracy,
            generations_to_target=generations_to_target,
            total_generations=config.n_generations,
            elapsed_time=elapsed,
            accuracy_history=accuracy_history
        )


def finetune_guards(
    genome: StatechartGenome,
    env_class: Type[GameEnvironment],
    n_generations: int = 50,
    target_accuracy: float = 0.9,
    verbose: bool = True
) -> FinetuneResult:
    """
    Convenience function for guard fine-tuning.

    Args:
        genome: Genome with topology to fine-tune
        env_class: Target game environment
        n_generations: Maximum generations
        target_accuracy: Stop early if reached
        verbose: Print progress

    Returns:
        FinetuneResult
    """
    config = FinetuneConfig(
        n_generations=n_generations,
        target_accuracy=target_accuracy,
        verbose=verbose
    )
    finetuner = GuardFinetuner(genome, env_class, config)
    return finetuner.finetune()


# =============================================================================
# Testing
# =============================================================================

def test_guard_finetuning():
    """Test guard fine-tuning functionality."""
    print("=" * 60)
    print("GUARD FINE-TUNING TEST")
    print("=" * 60)

    # Create a random genome for TicTacToe
    genome = create_random_genome(max_states=8, n_events=9, n_guards=4)
    while not genome.is_valid():
        genome = create_random_genome(max_states=8, n_events=9, n_guards=4)

    print(f"\nBase genome: {genome.n_states} states, {len(genome.transitions)} transitions")

    # Test freezing
    frozen = freeze_topology(genome)
    print(f"Frozen topology: {len(frozen.guard_indices)} guards to tune")

    # Test mutation
    mutated = mutate_guards(frozen)
    n_changed = sum(1 for a, b in zip(frozen.guard_indices, mutated.guard_indices) if a != b)
    print(f"After mutation: {n_changed} guards changed")

    # Test crossover
    frozen2 = freeze_topology(genome)
    for i in range(len(frozen2.guard_indices)):
        frozen2.guard_indices[i] = random.randint(0, 3)
    crossed = crossover_guards(frozen, frozen2)
    print(f"After crossover: guards from both parents")

    # Quick fine-tuning test
    print("\n" + "-" * 60)
    print("Running quick fine-tuning test...")

    result = finetune_guards(
        genome,
        TicTacToeEnv,
        n_generations=20,
        target_accuracy=0.7,
        verbose=True
    )

    print(f"\nResult: accuracy={result.final_accuracy:.3f}, "
          f"reached target at gen {result.generations_to_target}")

    print("\nGuard fine-tuning test complete!")
    return result


if __name__ == "__main__":
    test_guard_finetuning()
