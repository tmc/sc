"""
Domain Benchmark - Measure Transfer Efficiency Across Regex Domains

Target: 50% sample efficiency improvement via transfer learning.

Metrics:
1. Sample efficiency: Samples needed to reach target accuracy
2. Transfer gain: Accuracy improvement from transfer
3. Negative transfer: Cases where transfer hurts
4. Cross-domain matrix: All pairwise transfer results

Domains tested:
- Email: user@domain.tld
- URL: protocol://domain/path
- Phone: +1-area-prefix-line
- Date: YYYY-MM-DD variants
- IPv4: 192.168.1.1
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import random
import time

from .domain_encoder import DomainEncoder, DomainPatterns, compute_domain_similarity
from .transfer_learner import (
    TransferLearner, BaselineLearner, TransferStats,
    run_transfer_experiment, compare_transfer_vs_baseline
)


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for domain benchmark."""
    # Sample sizes to test
    sample_sizes: List[int] = field(default_factory=lambda: [10, 25, 50, 100])

    # Training settings
    source_epochs: int = 50
    fine_tune_epochs: int = 20
    baseline_epochs: int = 50

    # Target accuracy threshold
    target_accuracy: float = 0.80

    # Number of runs for averaging
    n_runs: int = 3

    # Domains to benchmark
    domains: List[str] = field(default_factory=lambda: ['email', 'url', 'phone', 'date', 'ipv4'])


@dataclass
class DomainResult:
    """Result for a single domain pair."""
    source_domain: str
    target_domain: str

    # Accuracy metrics
    source_accuracy: float
    transfer_accuracy: float
    baseline_accuracy: float

    # Sample efficiency
    transfer_samples_to_target: int = 0
    baseline_samples_to_target: int = 0
    efficiency_gain: float = 0.0  # (baseline_samples - transfer_samples) / baseline_samples

    # Transfer metrics
    transfer_gain: float = 0.0  # transfer_accuracy - baseline_accuracy
    is_negative_transfer: bool = False


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    config: BenchmarkConfig
    results: List[DomainResult]

    # Summary statistics
    avg_efficiency_gain: float = 0.0
    best_transfer_pair: Tuple[str, str] = ("", "")
    worst_transfer_pair: Tuple[str, str] = ("", "")
    negative_transfer_count: int = 0

    # Timing
    total_time: float = 0.0


# =============================================================================
# Benchmark Runner
# =============================================================================

class DomainBenchmark:
    """
    Run comprehensive transfer learning benchmark across domains.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.domain_data: Dict[str, List[str]] = {}
        self._generate_domain_data()

    def _generate_domain_data(self):
        """Generate data for all domains."""
        n_samples = 200  # Generate enough for testing

        self.domain_data = {
            'email': DomainPatterns.generate_email(n_samples),
            'url': DomainPatterns.generate_url(n_samples),
            'phone': DomainPatterns.generate_phone(n_samples),
            'date': DomainPatterns.generate_date(n_samples),
            'ipv4': DomainPatterns.generate_ipv4(n_samples),
        }

    def _samples_to_target(
        self,
        learner,
        examples: List[str],
        target_acc: float,
        max_samples: int = 100,
    ) -> int:
        """
        Find number of samples needed to reach target accuracy.
        """
        for n in self.config.sample_sizes:
            if n > min(max_samples, len(examples)):
                continue

            subset = examples[:n]
            acc = learner.evaluate(subset)

            if acc >= target_acc:
                return n

        return max_samples  # Didn't reach target

    def run_pair(
        self,
        source_domain: str,
        target_domain: str,
    ) -> DomainResult:
        """
        Run benchmark for a single source-target pair.
        """
        source_examples = self.domain_data[source_domain]
        target_examples = self.domain_data[target_domain]

        # Transfer learning
        transfer_learner = TransferLearner()
        source_acc = transfer_learner.train_source(
            source_examples[:100],
            epochs=self.config.source_epochs
        )

        _, transfer_acc = transfer_learner.transfer(
            target_examples[:100],
            fine_tune_epochs=self.config.fine_tune_epochs
        )

        # Baseline (from scratch)
        baseline = BaselineLearner()
        baseline_acc = baseline.train_and_evaluate(
            target_examples[:100],
            epochs=self.config.baseline_epochs
        )

        # Sample efficiency comparison
        transfer_samples = 0
        baseline_samples = 0

        for n in self.config.sample_sizes:
            target_subset = target_examples[:n]

            # Transfer with n target samples
            tl = TransferLearner()
            tl.train_source(source_examples[:100], epochs=self.config.source_epochs)
            _, t_acc = tl.transfer(target_subset, fine_tune_epochs=self.config.fine_tune_epochs)

            if t_acc >= self.config.target_accuracy and transfer_samples == 0:
                transfer_samples = n

            # Baseline with n samples
            bl = BaselineLearner()
            b_acc = bl.train_and_evaluate(target_subset, epochs=self.config.baseline_epochs)

            if b_acc >= self.config.target_accuracy and baseline_samples == 0:
                baseline_samples = n

        # If didn't reach target, use max
        if transfer_samples == 0:
            transfer_samples = max(self.config.sample_sizes)
        if baseline_samples == 0:
            baseline_samples = max(self.config.sample_sizes)

        # Compute efficiency gain
        if baseline_samples > 0:
            efficiency_gain = (baseline_samples - transfer_samples) / baseline_samples
        else:
            efficiency_gain = 0.0

        transfer_gain = transfer_acc - baseline_acc
        is_negative = transfer_acc < baseline_acc

        return DomainResult(
            source_domain=source_domain,
            target_domain=target_domain,
            source_accuracy=source_acc,
            transfer_accuracy=transfer_acc,
            baseline_accuracy=baseline_acc,
            transfer_samples_to_target=transfer_samples,
            baseline_samples_to_target=baseline_samples,
            efficiency_gain=efficiency_gain,
            transfer_gain=transfer_gain,
            is_negative_transfer=is_negative,
        )

    def run_all_pairs(self) -> BenchmarkResult:
        """
        Run benchmark for all domain pairs.
        """
        start_time = time.time()
        results = []

        domains = self.config.domains

        for source in domains:
            for target in domains:
                if source == target:
                    continue

                print(f"Running {source} -> {target}...")
                result = self.run_pair(source, target)
                results.append(result)

        total_time = time.time() - start_time

        # Compute summary statistics
        efficiency_gains = [r.efficiency_gain for r in results if not r.is_negative_transfer]
        avg_gain = sum(efficiency_gains) / len(efficiency_gains) if efficiency_gains else 0.0

        # Find best and worst pairs
        if results:
            best = max(results, key=lambda r: r.transfer_gain)
            worst = min(results, key=lambda r: r.transfer_gain)
            best_pair = (best.source_domain, best.target_domain)
            worst_pair = (worst.source_domain, worst.target_domain)
        else:
            best_pair = ("", "")
            worst_pair = ("", "")

        negative_count = sum(1 for r in results if r.is_negative_transfer)

        return BenchmarkResult(
            config=self.config,
            results=results,
            avg_efficiency_gain=avg_gain,
            best_transfer_pair=best_pair,
            worst_transfer_pair=worst_pair,
            negative_transfer_count=negative_count,
            total_time=total_time,
        )


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_transfer_matrix(results: List[DomainResult]) -> Dict[str, Dict[str, float]]:
    """
    Create transfer gain matrix.

    Returns: {source: {target: transfer_gain}}
    """
    matrix: Dict[str, Dict[str, float]] = {}

    for r in results:
        if r.source_domain not in matrix:
            matrix[r.source_domain] = {}
        matrix[r.source_domain][r.target_domain] = r.transfer_gain

    return matrix


def find_optimal_transfer_path(
    results: List[DomainResult],
    start_domain: str,
    end_domain: str,
) -> List[str]:
    """
    Find optimal path from start to end domain for transfer.

    Uses greedy selection of best transfer gains.
    """
    # Build graph
    gains: Dict[Tuple[str, str], float] = {}
    for r in results:
        gains[(r.source_domain, r.target_domain)] = r.transfer_gain

    # Greedy path
    path = [start_domain]
    current = start_domain
    visited = {start_domain}

    while current != end_domain:
        # Find best next step
        best_next = None
        best_gain = float('-inf')

        for (src, tgt), gain in gains.items():
            if src == current and tgt not in visited:
                if gain > best_gain:
                    best_gain = gain
                    best_next = tgt

        if best_next is None:
            break  # No path found

        path.append(best_next)
        visited.add(best_next)
        current = best_next

    return path


def compute_sample_efficiency_improvement(results: List[DomainResult]) -> float:
    """
    Compute overall sample efficiency improvement from transfer.

    Target: 50% improvement
    """
    improvements = []

    for r in results:
        if r.baseline_samples_to_target > 0:
            improvement = (r.baseline_samples_to_target - r.transfer_samples_to_target) / r.baseline_samples_to_target
            improvements.append(improvement)

    if improvements:
        return sum(improvements) / len(improvements)
    return 0.0


# =============================================================================
# Reporting
# =============================================================================

def print_benchmark_report(result: BenchmarkResult):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("REGEX TRANSFER BENCHMARK REPORT")
    print("=" * 70)

    print(f"\nDomains tested: {', '.join(result.config.domains)}")
    print(f"Sample sizes: {result.config.sample_sizes}")
    print(f"Total time: {result.total_time:.2f}s")

    # Transfer matrix
    print("\n--- Transfer Gain Matrix ---")
    matrix = analyze_transfer_matrix(result.results)
    domains = result.config.domains

    # Header
    print(f"{'Source\\Target':>12}", end='')
    for d in domains:
        print(f"{d[:8]:>10}", end='')
    print()

    # Rows
    for src in domains:
        print(f"{src[:12]:>12}", end='')
        for tgt in domains:
            if src == tgt:
                print(f"{'---':>10}", end='')
            elif src in matrix and tgt in matrix[src]:
                gain = matrix[src][tgt]
                color = "+" if gain > 0 else ""
                print(f"{color}{gain:>9.1%}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()

    # Sample efficiency
    print("\n--- Sample Efficiency ---")
    print(f"{'Source':>10} -> {'Target':>10}  {'Transfer':>10} {'Baseline':>10} {'Improve':>10}")
    print("-" * 60)

    for r in sorted(result.results, key=lambda x: -x.efficiency_gain):
        print(
            f"{r.source_domain[:10]:>10} -> {r.target_domain[:10]:>10}  "
            f"{r.transfer_samples_to_target:>10} "
            f"{r.baseline_samples_to_target:>10} "
            f"{r.efficiency_gain:>+10.1%}"
        )

    # Summary
    print("\n--- Summary ---")
    print(f"Average efficiency gain: {result.avg_efficiency_gain:.1%}")
    print(f"Best transfer pair: {result.best_transfer_pair[0]} -> {result.best_transfer_pair[1]}")
    print(f"Worst transfer pair: {result.worst_transfer_pair[0]} -> {result.worst_transfer_pair[1]}")
    print(f"Negative transfer cases: {result.negative_transfer_count}")

    # Check target
    overall_improvement = compute_sample_efficiency_improvement(result.results)
    target_met = overall_improvement >= 0.50

    print(f"\n{'=' * 70}")
    print(f"TARGET: 50% sample efficiency improvement")
    print(f"RESULT: {overall_improvement:.1%} improvement")
    print(f"STATUS: {'ACHIEVED' if target_met else 'NOT MET'}")
    print(f"{'=' * 70}")


# =============================================================================
# Quick Benchmark
# =============================================================================

def run_quick_benchmark() -> BenchmarkResult:
    """
    Run a quick benchmark with reduced settings.
    """
    config = BenchmarkConfig(
        sample_sizes=[10, 25, 50],
        source_epochs=30,
        fine_tune_epochs=10,
        baseline_epochs=30,
        n_runs=1,
        domains=['email', 'url', 'phone'],  # Subset for speed
    )

    benchmark = DomainBenchmark(config)
    return benchmark.run_all_pairs()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run domain benchmark demonstration."""
    print("=" * 60)
    print("Domain Benchmark Demo")
    print("=" * 60)

    # Run quick benchmark
    print("\nRunning quick benchmark (3 domains)...")
    result = run_quick_benchmark()

    # Print report
    print_benchmark_report(result)

    # Show structural similarity for context
    print("\n--- Domain Structural Similarity ---")
    similarity = compute_domain_similarity()

    for (d1, d2), sim in sorted(similarity.items(), key=lambda x: -x[1]):
        if d1 < d2:
            print(f"  {d1} <-> {d2}: {sim:.3f}")

    return result


if __name__ == "__main__":
    demo()
