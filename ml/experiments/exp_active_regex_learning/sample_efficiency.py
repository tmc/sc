"""
Sample Efficiency Comparison: Active vs Passive Learning

Compares sample efficiency between:
1. Passive Learning - random example selection
2. Active Learning - strategic query selection

Target: Demonstrate 50% reduction in examples needed to reach target accuracy.

Metrics:
- Queries to reach X% accuracy
- Final accuracy given fixed query budget
- Learning curve (accuracy vs queries)
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

from .oracle_interface import (
    Oracle,
    BENCHMARK_ORACLES,
    get_oracle,
    list_oracles,
)
from .query_strategy import get_strategy, list_strategies
from .active_learner import ActiveLearner, LearningState


# =============================================================================
# Experiment Results
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    oracle_name: str
    strategy_name: str
    n_queries: int
    final_accuracy: float
    accuracy_history: List[float]
    queries_to_80: Optional[int] = None
    queries_to_90: Optional[int] = None
    queries_to_95: Optional[int] = None
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "oracle": self.oracle_name,
            "strategy": self.strategy_name,
            "n_queries": self.n_queries,
            "final_accuracy": self.final_accuracy,
            "queries_to_80": self.queries_to_80,
            "queries_to_90": self.queries_to_90,
            "queries_to_95": self.queries_to_95,
            "runtime": self.runtime_seconds,
        }


@dataclass
class ComparisonResult:
    """Comparison between passive and active learning."""
    oracle_name: str
    passive_results: List[ExperimentResult]
    active_results: Dict[str, List[ExperimentResult]]

    # Summary statistics
    passive_mean_queries_90: float = 0.0
    active_mean_queries_90: Dict[str, float] = field(default_factory=dict)
    reduction_percent: Dict[str, float] = field(default_factory=dict)

    def compute_summary(self):
        """Compute summary statistics."""
        # Passive baseline
        queries_90 = [r.queries_to_90 for r in self.passive_results if r.queries_to_90]
        self.passive_mean_queries_90 = sum(queries_90) / len(queries_90) if queries_90 else float('inf')

        # Active strategies
        for strat, results in self.active_results.items():
            queries_90 = [r.queries_to_90 for r in results if r.queries_to_90]
            if queries_90:
                mean = sum(queries_90) / len(queries_90)
                self.active_mean_queries_90[strat] = mean

                # Compute reduction
                if self.passive_mean_queries_90 > 0:
                    reduction = (self.passive_mean_queries_90 - mean) / self.passive_mean_queries_90 * 100
                    self.reduction_percent[strat] = reduction

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=== {self.oracle_name} ===",
            f"Passive (random): {self.passive_mean_queries_90:.1f} queries to 90%",
        ]
        for strat, mean in sorted(self.active_mean_queries_90.items()):
            reduction = self.reduction_percent.get(strat, 0)
            lines.append(f"Active ({strat}): {mean:.1f} queries ({reduction:+.1f}% vs passive)")
        return "\n".join(lines)


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(
    oracle: Oracle,
    strategy_name: str,
    max_queries: int = 100,
    batch_size: int = 5,
    seed: Optional[int] = None,
) -> ExperimentResult:
    """Run a single learning experiment."""
    if seed is not None:
        random.seed(seed)

    start_time = time.time()

    # Create learner
    strategy = get_strategy(strategy_name)
    learner = ActiveLearner(
        oracle=oracle,
        strategy=strategy,
        max_queries=max_queries,
        batch_size=batch_size,
    )
    learner.initialize_candidate_pool()

    # Run learning
    state = learner.learn(verbose=False)
    final_accuracy = learner.get_final_accuracy()

    runtime = time.time() - start_time

    # Find queries to reach accuracy thresholds
    def queries_to_threshold(history: List[float], threshold: float) -> Optional[int]:
        for i, acc in enumerate(history):
            if acc >= threshold:
                return (i + 1) * batch_size + 5  # +5 for initial examples
        return None

    return ExperimentResult(
        oracle_name=oracle.name,
        strategy_name=strategy_name,
        n_queries=state.n_queries,
        final_accuracy=final_accuracy,
        accuracy_history=state.accuracy_history,
        queries_to_80=queries_to_threshold(state.accuracy_history, 0.80),
        queries_to_90=queries_to_threshold(state.accuracy_history, 0.90),
        queries_to_95=queries_to_threshold(state.accuracy_history, 0.95),
        runtime_seconds=runtime,
    )


def run_comparison(
    oracle_name: str,
    active_strategies: List[str] = ["uncertainty", "hybrid", "diversity"],
    n_runs: int = 5,
    max_queries: int = 100,
    verbose: bool = True,
) -> ComparisonResult:
    """
    Compare passive vs active learning on an oracle.

    Args:
        oracle_name: Name of benchmark oracle
        active_strategies: List of active strategies to test
        n_runs: Number of runs per configuration
        max_queries: Maximum queries per run
        verbose: Print progress

    Returns:
        ComparisonResult with all metrics
    """
    oracle_factory = BENCHMARK_ORACLES[oracle_name]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing on: {oracle_name}")
        print(f"{'='*60}")

    # Run passive (random) baseline
    if verbose:
        print("\nPassive (Random) baseline:")
    passive_results = []
    for run in range(n_runs):
        oracle = oracle_factory()
        result = run_single_experiment(
            oracle, "random", max_queries, seed=run
        )
        passive_results.append(result)
        if verbose:
            print(f"  Run {run+1}: {result.n_queries} queries, "
                  f"acc={result.final_accuracy:.3f}, "
                  f"to_90={result.queries_to_90}")

    # Run active strategies
    active_results = {}
    for strat in active_strategies:
        if verbose:
            print(f"\nActive ({strat}):")
        strat_results = []
        for run in range(n_runs):
            oracle = oracle_factory()
            result = run_single_experiment(
                oracle, strat, max_queries, seed=run
            )
            strat_results.append(result)
            if verbose:
                print(f"  Run {run+1}: {result.n_queries} queries, "
                      f"acc={result.final_accuracy:.3f}, "
                      f"to_90={result.queries_to_90}")
        active_results[strat] = strat_results

    # Create comparison result
    comparison = ComparisonResult(
        oracle_name=oracle_name,
        passive_results=passive_results,
        active_results=active_results,
    )
    comparison.compute_summary()

    if verbose:
        print(f"\n{comparison.summary()}")

    return comparison


def run_full_benchmark(
    oracle_names: Optional[List[str]] = None,
    strategies: List[str] = ["uncertainty", "hybrid"],
    n_runs: int = 3,
    max_queries: int = 80,
    verbose: bool = True,
) -> Dict[str, ComparisonResult]:
    """
    Run full benchmark across multiple oracles.

    Returns:
        Dict mapping oracle name to comparison results
    """
    if oracle_names is None:
        oracle_names = ["ab_star", "even_zeros", "no_consecutive", "binary_div3"]

    results = {}
    for oracle_name in oracle_names:
        comparison = run_comparison(
            oracle_name,
            active_strategies=strategies,
            n_runs=n_runs,
            max_queries=max_queries,
            verbose=verbose,
        )
        results[oracle_name] = comparison

    # Print overall summary
    if verbose:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)

        for strat in strategies:
            reductions = []
            for oracle_name, comp in results.items():
                if strat in comp.reduction_percent:
                    reductions.append(comp.reduction_percent[strat])

            if reductions:
                mean_reduction = sum(reductions) / len(reductions)
                print(f"\n{strat}: {mean_reduction:.1f}% average reduction in queries")
                for oracle_name, comp in results.items():
                    red = comp.reduction_percent.get(strat, 0)
                    print(f"  {oracle_name}: {red:.1f}%")

    return results


# =============================================================================
# Learning Curve Visualization
# =============================================================================

def generate_learning_curves(
    oracle_name: str,
    strategies: List[str] = ["random", "uncertainty", "hybrid"],
    max_queries: int = 100,
    n_runs: int = 3,
) -> Dict[str, List[List[float]]]:
    """
    Generate learning curves for visualization.

    Returns:
        Dict mapping strategy to list of accuracy histories
    """
    oracle_factory = BENCHMARK_ORACLES[oracle_name]
    curves = {}

    for strat in strategies:
        curves[strat] = []
        for run in range(n_runs):
            oracle = oracle_factory()
            result = run_single_experiment(oracle, strat, max_queries, seed=run)
            curves[strat].append(result.accuracy_history)

    return curves


def print_learning_curves(curves: Dict[str, List[List[float]]]):
    """Print ASCII visualization of learning curves."""
    # Find max length
    max_len = max(
        len(h) for histories in curves.values() for h in histories
    )

    # Average across runs
    avg_curves = {}
    for strat, histories in curves.items():
        avg = []
        for i in range(max_len):
            vals = [h[i] for h in histories if i < len(h)]
            avg.append(sum(vals) / len(vals) if vals else 0)
        avg_curves[strat] = avg

    # Print
    print("\nLearning Curves (accuracy vs queries):")
    print("-" * 60)

    for queries in range(0, max_len, 2):
        line = f"Q{queries*5:3d}: "
        for strat in sorted(avg_curves.keys()):
            acc = avg_curves[strat][queries] if queries < len(avg_curves[strat]) else 0
            bar_len = int(acc * 20)
            line += f"{strat[:8]:8s} {'#' * bar_len}{' ' * (20-bar_len)} {acc:.2f}  "
        print(line)


# =============================================================================
# Main Demo
# =============================================================================

def demo():
    """Run demonstration of sample efficiency comparison."""
    print("="*60)
    print("Active vs Passive Learning: Sample Efficiency Comparison")
    print("="*60)
    print("\nGoal: Demonstrate 50% fewer examples with active learning")

    # Quick benchmark on simple patterns
    results = run_full_benchmark(
        oracle_names=["ab_star", "even_zeros", "no_consecutive"],
        strategies=["uncertainty", "hybrid"],
        n_runs=3,
        max_queries=60,
        verbose=True,
    )

    # Check if we achieved target
    print("\n" + "="*60)
    print("TARGET CHECK: 50% reduction")
    print("="*60)

    achieved_target = False
    for strat in ["uncertainty", "hybrid"]:
        reductions = [
            comp.reduction_percent.get(strat, 0)
            for comp in results.values()
        ]
        mean_red = sum(reductions) / len(reductions) if reductions else 0

        status = "ACHIEVED" if mean_red >= 50 else "NOT YET"
        print(f"\n{strat}: {mean_red:.1f}% reduction [{status}]")

        if mean_red >= 50:
            achieved_target = True

    if achieved_target:
        print("\n*** TARGET ACHIEVED: 50%+ reduction with active learning ***")
    else:
        print("\n*** Working toward 50% target - results vary by pattern ***")

    return results


if __name__ == "__main__":
    demo()
