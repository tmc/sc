"""
Transfer Learning Benchmark

Compare training efficiency:
1. SCRATCH: Full topology evolution from random initialization
2. TRANSFER: Topology transfer + guard fine-tuning only

Metrics:
- Steps to 90% accuracy
- Final accuracy
- Wall-clock time
- Generalization (test on unseen positions)

Test chains:
- TicTacToe -> Connect4
- TicTacToe -> Connect4 -> Othello
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Type, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_topology_evolution.environments import (
    GameEnvironment, TicTacToeEnv, Connect4Env, OthelloEnv,
    ENVIRONMENTS
)
from exp_topology_evolution.evolve import (
    StatechartGenome, create_random_genome, evaluate_fitness
)
from exp_topology_evolution.framework import TopologyEvolver, EvolutionConfig

from .topology_transfer import TopologyTransfer, extract_topology, transfer_chain
from .guard_finetuning import GuardFinetuner, FinetuneConfig, FinetuneResult, finetune_guards


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    method: str  # "scratch" or "transfer"
    source_game: str
    target_game: str
    final_accuracy: float
    steps_to_90: int  # -1 if not reached
    total_steps: int
    elapsed_time: float
    accuracy_curve: List[float] = field(default_factory=list)


@dataclass
class TransferResults:
    """Complete transfer learning benchmark results."""
    scratch_results: List[BenchmarkResult]
    transfer_results: List[BenchmarkResult]

    def summary(self) -> str:
        """Generate summary comparison."""
        lines = ["=" * 70, "TRANSFER LEARNING BENCHMARK RESULTS", "=" * 70]

        # Group by target game
        targets = set(r.target_game for r in self.scratch_results + self.transfer_results)

        for target in sorted(targets):
            scratch = [r for r in self.scratch_results if r.target_game == target]
            transfer = [r for r in self.transfer_results if r.target_game == target]

            lines.append(f"\n--- {target} ---")

            if scratch:
                avg_scratch_acc = sum(r.final_accuracy for r in scratch) / len(scratch)
                avg_scratch_steps = sum(r.steps_to_90 for r in scratch if r.steps_to_90 > 0)
                n_reached = sum(1 for r in scratch if r.steps_to_90 > 0)
                avg_scratch_steps = avg_scratch_steps / n_reached if n_reached > 0 else -1
                avg_scratch_time = sum(r.elapsed_time for r in scratch) / len(scratch)

                lines.append(f"  SCRATCH:")
                lines.append(f"    Final accuracy: {avg_scratch_acc:.3f}")
                lines.append(f"    Steps to 90%: {avg_scratch_steps:.0f}" if avg_scratch_steps > 0 else "    Steps to 90%: Not reached")
                lines.append(f"    Time: {avg_scratch_time:.1f}s")

            if transfer:
                avg_transfer_acc = sum(r.final_accuracy for r in transfer) / len(transfer)
                avg_transfer_steps = sum(r.steps_to_90 for r in transfer if r.steps_to_90 > 0)
                n_reached = sum(1 for r in transfer if r.steps_to_90 > 0)
                avg_transfer_steps = avg_transfer_steps / n_reached if n_reached > 0 else -1
                avg_transfer_time = sum(r.elapsed_time for r in transfer) / len(transfer)

                lines.append(f"  TRANSFER:")
                lines.append(f"    Final accuracy: {avg_transfer_acc:.3f}")
                lines.append(f"    Steps to 90%: {avg_transfer_steps:.0f}" if avg_transfer_steps > 0 else "    Steps to 90%: Not reached")
                lines.append(f"    Time: {avg_transfer_time:.1f}s")

            # Speedup calculation
            if scratch and transfer and avg_scratch_steps > 0 and avg_transfer_steps > 0:
                speedup = avg_scratch_steps / avg_transfer_steps
                lines.append(f"  SPEEDUP: {speedup:.1f}x")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_scratch(
    env_class: Type[GameEnvironment],
    n_generations: int = 50,
    population_size: int = 30,
    target_accuracy: float = 0.9,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Benchmark training from scratch (full topology evolution).

    Args:
        env_class: Target game environment
        n_generations: Maximum generations
        population_size: Population size
        target_accuracy: Target accuracy threshold
        verbose: Print progress

    Returns:
        BenchmarkResult
    """
    start_time = time.time()

    config = EvolutionConfig(
        population_size=population_size,
        n_generations=n_generations,
        max_states=12,
        dataset_size=500,
        verbose=verbose,
        log_every=10
    )

    evolver = TopologyEvolver(env_class, config)

    # Track accuracy per generation
    accuracy_curve = []
    steps_to_90 = -1

    def callback(gen, population, best):
        accuracy_curve.append(best.accuracy)
        nonlocal steps_to_90
        if steps_to_90 < 0 and best.accuracy >= target_accuracy:
            steps_to_90 = gen

    best = evolver.evolve(callback=callback)
    elapsed = time.time() - start_time

    return BenchmarkResult(
        method="scratch",
        source_game="None",
        target_game=env_class.config.name,
        final_accuracy=best.accuracy,
        steps_to_90=steps_to_90,
        total_steps=n_generations,
        elapsed_time=elapsed,
        accuracy_curve=accuracy_curve
    )


def benchmark_transfer(
    source_genome: StatechartGenome,
    source_env: Type[GameEnvironment],
    target_env: Type[GameEnvironment],
    n_generations: int = 50,
    target_accuracy: float = 0.9,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Benchmark topology transfer + guard fine-tuning.

    Args:
        source_genome: Pre-trained source genome
        source_env: Source game environment
        target_env: Target game environment
        n_generations: Maximum fine-tuning generations
        target_accuracy: Target accuracy threshold
        verbose: Print progress

    Returns:
        BenchmarkResult
    """
    start_time = time.time()

    # Transfer topology
    transfer = TopologyTransfer(source_genome, source_env, target_env)
    target_genome = transfer.transfer()

    if verbose:
        print(f"Transferred topology: {target_genome.n_states} states, "
              f"{len(target_genome.transitions)} transitions")

    # Fine-tune guards
    result = finetune_guards(
        target_genome,
        target_env,
        n_generations=n_generations,
        target_accuracy=target_accuracy,
        verbose=verbose
    )

    elapsed = time.time() - start_time

    return BenchmarkResult(
        method="transfer",
        source_game=source_env.config.name,
        target_game=target_env.config.name,
        final_accuracy=result.final_accuracy,
        steps_to_90=result.generations_to_target,
        total_steps=n_generations,
        elapsed_time=elapsed,
        accuracy_curve=result.accuracy_history
    )


# =============================================================================
# Full Benchmark Suite
# =============================================================================

def run_transfer_benchmark(
    n_runs: int = 3,
    n_generations: int = 50,
    population_size: int = 30,
    target_accuracy: float = 0.9,
    test_chain: bool = True,
    verbose: bool = True
) -> TransferResults:
    """
    Run complete transfer learning benchmark.

    Tests:
    1. TicTacToe -> Connect4 (transfer vs scratch)
    2. TicTacToe -> Connect4 -> Othello (chain transfer)

    Args:
        n_runs: Number of runs per condition
        n_generations: Generations per run
        population_size: Population size
        target_accuracy: Target accuracy (90%)
        test_chain: Include chain transfer test
        verbose: Print progress

    Returns:
        TransferResults with all benchmark data
    """
    scratch_results = []
    transfer_results = []

    if verbose:
        print("=" * 70)
        print("TRANSFER LEARNING BENCHMARK")
        print("=" * 70)
        print(f"Runs per condition: {n_runs}")
        print(f"Generations: {n_generations}")
        print(f"Target accuracy: {target_accuracy:.0%}")
        print("-" * 70)

    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUN {run + 1}/{n_runs}")
            print("=" * 70)

        # =================================================================
        # Step 1: Train source on TicTacToe
        # =================================================================
        if verbose:
            print("\n--- Training source on TicTacToe ---")

        source_config = EvolutionConfig(
            population_size=population_size,
            n_generations=n_generations,
            max_states=10,
            dataset_size=500,
            verbose=verbose,
            log_every=10
        )
        source_evolver = TopologyEvolver(TicTacToeEnv, source_config)
        source_genome = source_evolver.evolve()

        if verbose:
            print(f"Source trained: accuracy={source_genome.accuracy:.3f}")

        # =================================================================
        # Step 2: Benchmark Connect4 - SCRATCH
        # =================================================================
        if verbose:
            print("\n--- Connect4: Training from SCRATCH ---")

        c4_scratch = benchmark_scratch(
            Connect4Env,
            n_generations=n_generations,
            population_size=population_size,
            target_accuracy=target_accuracy,
            verbose=verbose
        )
        scratch_results.append(c4_scratch)

        # =================================================================
        # Step 3: Benchmark Connect4 - TRANSFER
        # =================================================================
        if verbose:
            print("\n--- Connect4: TRANSFER from TicTacToe ---")

        c4_transfer = benchmark_transfer(
            source_genome,
            TicTacToeEnv,
            Connect4Env,
            n_generations=n_generations,
            target_accuracy=target_accuracy,
            verbose=verbose
        )
        transfer_results.append(c4_transfer)

        # =================================================================
        # Step 4: Chain transfer to Othello (optional)
        # =================================================================
        if test_chain:
            if verbose:
                print("\n--- Othello: Training from SCRATCH ---")

            othello_scratch = benchmark_scratch(
                OthelloEnv,
                n_generations=n_generations,
                population_size=population_size,
                target_accuracy=target_accuracy,
                verbose=verbose
            )
            scratch_results.append(othello_scratch)

            if verbose:
                print("\n--- Othello: CHAIN TRANSFER (TicTacToe -> Connect4 -> Othello) ---")

            # First transfer to Connect4
            c4_genome = TopologyTransfer(source_genome, TicTacToeEnv, Connect4Env).transfer()

            # Then transfer to Othello
            othello_transfer = benchmark_transfer(
                c4_genome,
                Connect4Env,
                OthelloEnv,
                n_generations=n_generations,
                target_accuracy=target_accuracy,
                verbose=verbose
            )
            transfer_results.append(othello_transfer)

    results = TransferResults(
        scratch_results=scratch_results,
        transfer_results=transfer_results
    )

    if verbose:
        print("\n" + results.summary())

    return results


# =============================================================================
# Quick Benchmark (for testing)
# =============================================================================

def quick_benchmark():
    """Quick benchmark for testing."""
    print("=" * 70)
    print("QUICK TRANSFER LEARNING BENCHMARK")
    print("=" * 70)

    return run_transfer_benchmark(
        n_runs=1,
        n_generations=20,
        population_size=20,
        target_accuracy=0.8,
        test_chain=False,
        verbose=True
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        results = quick_benchmark()
    else:
        results = run_transfer_benchmark(
            n_runs=3,
            n_generations=50,
            population_size=30,
            target_accuracy=0.9,
            test_chain=True,
            verbose=True
        )

    print("\nBenchmark complete!")
