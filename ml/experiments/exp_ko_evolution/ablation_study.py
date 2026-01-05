"""
Ablation Study: Impact of History Learning

Compare:
1. FULL: Evolvable history (depth 0-5, can evolve)
2. NO_HISTORY: Fixed depth=0 (history disabled)
3. FIXED_1: Fixed depth=1 (hand-coded Ko rule)
4. FIXED_3: Fixed depth=3 (hand-coded super-Ko)

This demonstrates that:
- Without history, loops occur
- Evolution DISCOVERS the optimal depth
- Learned depth matches or exceeds hand-coded
"""

import random
import copy
from dataclasses import dataclass
from typing import List, Dict
import time

from .topology_ko import (
    KoGenome, HistoryGene, HistoryEncoding,
    create_random_ko_genome, mutate_ko_genome, crossover_ko,
    evaluate_ko_genome, GoGame, play_game_with_genome
)


# =============================================================================
# Ablation Conditions
# =============================================================================

def create_no_history_genome(n_states: int = 5) -> KoGenome:
    """Create genome with history DISABLED (depth=0 fixed)."""
    genome = create_random_ko_genome(n_states)
    # Force all history to depth=0
    for gene in genome.history_genes:
        gene.depth = 0
        gene.encoding = HistoryEncoding.NONE
    return genome


def create_fixed_depth_genome(n_states: int = 5, depth: int = 1) -> KoGenome:
    """Create genome with FIXED history depth (hand-coded)."""
    genome = create_random_ko_genome(n_states)
    # Set all history to fixed depth
    for gene in genome.history_genes:
        gene.depth = depth
        gene.encoding = HistoryEncoding.RAW if depth > 0 else HistoryEncoding.NONE
    return genome


def mutate_no_history(genome: KoGenome) -> KoGenome:
    """Mutate but keep history disabled."""
    new = mutate_ko_genome(genome)
    # Force history to stay at 0
    for gene in new.history_genes:
        gene.depth = 0
        gene.encoding = HistoryEncoding.NONE
    return new


def mutate_fixed_depth(genome: KoGenome, depth: int) -> KoGenome:
    """Mutate but keep history at fixed depth."""
    new = mutate_ko_genome(genome)
    for gene in new.history_genes:
        gene.depth = depth
        gene.encoding = HistoryEncoding.RAW if depth > 0 else HistoryEncoding.NONE
    return new


# =============================================================================
# Run Single Condition
# =============================================================================

@dataclass
class AblationResult:
    """Results from one ablation condition."""
    condition: str
    generations: int
    final_fitness: float
    games_completed: int
    games_looped: int
    ko_prevented: int
    avg_history_depth: float
    elapsed_time: float


def run_condition(
    condition: str,
    n_generations: int = 30,
    population_size: int = 20,
    games_per_eval: int = 10,
    fixed_depth: int = 0
) -> AblationResult:
    """Run evolution for one ablation condition."""

    start_time = time.time()

    # Initialize population based on condition
    if condition == "FULL":
        population = [create_random_ko_genome(5) for _ in range(population_size)]
        mutate_fn = mutate_ko_genome
    elif condition == "NO_HISTORY":
        population = [create_no_history_genome(5) for _ in range(population_size)]
        mutate_fn = mutate_no_history
    else:  # FIXED_N
        population = [create_fixed_depth_genome(5, fixed_depth) for _ in range(population_size)]
        mutate_fn = lambda g: mutate_fixed_depth(g, fixed_depth)

    # Evaluate initial
    for genome in population:
        evaluate_ko_genome(genome, games_per_eval)

    # Evolution
    for gen in range(n_generations):
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Elite
        new_pop = [g.copy() for g in population[:3]]

        # Offspring
        while len(new_pop) < population_size:
            if random.random() < 0.7 and condition == "FULL":
                p1 = random.choice(population[:population_size//2])
                p2 = random.choice(population[:population_size//2])
                child = crossover_ko(p1, p2)
            else:
                parent = random.choice(population[:population_size//2])
                child = parent.copy()

            child = mutate_fn(child)
            if child.is_valid():
                new_pop.append(child)

        for genome in new_pop:
            evaluate_ko_genome(genome, games_per_eval)

        population = new_pop

    # Best result
    best = max(population, key=lambda g: g.fitness)
    avg_depth = sum(g.max_history_depth() for g in population) / len(population)

    return AblationResult(
        condition=condition,
        generations=n_generations,
        final_fitness=best.fitness,
        games_completed=best.games_completed,
        games_looped=best.games_looped,
        ko_prevented=best.ko_violations_prevented,
        avg_history_depth=avg_depth,
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Full Ablation Study
# =============================================================================

def run_ablation_study(
    n_runs: int = 3,
    n_generations: int = 30,
    population_size: int = 20,
    games_per_eval: int = 10,
    verbose: bool = True
) -> Dict[str, List[AblationResult]]:
    """
    Run complete ablation study.

    Conditions:
    - FULL: Evolvable history (our method)
    - NO_HISTORY: History disabled (depth=0 fixed)
    - FIXED_1: Hand-coded Ko (depth=1)
    - FIXED_3: Hand-coded super-Ko (depth=3)
    """

    conditions = [
        ("FULL", 0),
        ("NO_HISTORY", 0),
        ("FIXED_1", 1),
        ("FIXED_3", 3),
    ]

    results = {c[0]: [] for c in conditions}

    if verbose:
        print("=" * 70)
        print("ABLATION STUDY: Impact of History Learning")
        print("=" * 70)
        print(f"Conditions: {[c[0] for c in conditions]}")
        print(f"Runs per condition: {n_runs}")
        print(f"Generations: {n_generations}")
        print("-" * 70)

    for run in range(n_runs):
        if verbose:
            print(f"\n--- Run {run + 1}/{n_runs} ---")

        for condition, fixed_depth in conditions:
            result = run_condition(
                condition=condition,
                n_generations=n_generations,
                population_size=population_size,
                games_per_eval=games_per_eval,
                fixed_depth=fixed_depth
            )
            results[condition].append(result)

            if verbose:
                print(f"  {condition:12s} | fit={result.final_fitness:+.2f} "
                      f"| looped={result.games_looped}/{games_per_eval} "
                      f"| depth={result.avg_history_depth:.1f} "
                      f"| time={result.elapsed_time:.1f}s")

    # Aggregate results
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION RESULTS (averaged over {} runs)".format(n_runs))
        print("=" * 70)
        print(f"{'Condition':<12} | {'Fitness':>8} | {'Loops':>6} | {'Ko Prev':>8} | {'Depth':>6}")
        print("-" * 70)

        for condition, _ in conditions:
            runs = results[condition]
            avg_fit = sum(r.final_fitness for r in runs) / len(runs)
            avg_loops = sum(r.games_looped for r in runs) / len(runs)
            avg_ko = sum(r.ko_prevented for r in runs) / len(runs)
            avg_depth = sum(r.avg_history_depth for r in runs) / len(runs)

            print(f"{condition:<12} | {avg_fit:>+8.2f} | {avg_loops:>6.1f} | {avg_ko:>8.1f} | {avg_depth:>6.1f}")

        print("-" * 70)

        # Statistical comparison
        full_fits = [r.final_fitness for r in results["FULL"]]
        no_hist_fits = [r.final_fitness for r in results["NO_HISTORY"]]
        fixed1_fits = [r.final_fitness for r in results["FIXED_1"]]

        full_loops = [r.games_looped for r in results["FULL"]]
        no_hist_loops = [r.games_looped for r in results["NO_HISTORY"]]

        print("\nKEY COMPARISONS:")
        print(f"  FULL vs NO_HISTORY fitness: {sum(full_fits)/len(full_fits):+.2f} vs {sum(no_hist_fits)/len(no_hist_fits):+.2f}")
        print(f"  FULL vs NO_HISTORY loops:   {sum(full_loops)/len(full_loops):.1f} vs {sum(no_hist_loops)/len(no_hist_loops):.1f}")
        print(f"  FULL vs FIXED_1 fitness:    {sum(full_fits)/len(full_fits):+.2f} vs {sum(fixed1_fits)/len(fixed1_fits):+.2f}")

        # Conclusion
        if sum(full_fits) > sum(no_hist_fits):
            print("\n*** HISTORY LEARNING IMPROVES FITNESS ***")
        if sum(full_loops) < sum(no_hist_loops):
            print("*** HISTORY LEARNING REDUCES LOOPS ***")
        if sum(full_fits) >= sum(fixed1_fits):
            print("*** LEARNED HISTORY MATCHES/EXCEEDS HAND-CODED ***")

    return results


# =============================================================================
# Quick Validation Test
# =============================================================================

def quick_validation():
    """Quick test to validate ablation works."""
    print("Quick validation of ablation conditions...\n")

    # Test each condition briefly
    for condition, depth in [("FULL", 0), ("NO_HISTORY", 0), ("FIXED_1", 1)]:
        result = run_condition(
            condition=condition,
            n_generations=10,
            population_size=10,
            games_per_eval=5,
            fixed_depth=depth
        )
        print(f"{condition}: fitness={result.final_fitness:+.2f}, "
              f"loops={result.games_looped}, depth={result.avg_history_depth:.1f}")

    print("\nValidation complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_validation()
    else:
        results = run_ablation_study(
            n_runs=3,
            n_generations=30,
            population_size=20,
            games_per_eval=10,
            verbose=True
        )
