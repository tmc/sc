"""
Benchmark Runner for Statechart Synthesis

Orchestrates the full benchmark:
1. Load all methods and domains
2. Run each method on each domain
3. Compute comprehensive metrics
4. Generate leaderboard
5. Export results

Usage:
    python -m exp_statechart_synthesis_benchmark.benchmark_runner
    python -m exp_statechart_synthesis_benchmark.benchmark_runner --quick
    python -m exp_statechart_synthesis_benchmark.benchmark_runner --domain games
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from .synthesis_methods import (
    SynthesisMethod, SynthesisConfig, SynthesisResult,
    get_all_methods, get_methods_by_category, MethodCategory
)
from .domain_suite import (
    Domain, DomainSuite, DomainType, Difficulty,
    get_all_domains, get_domains_by_type, get_domains_by_difficulty,
    create_full_suite, create_quick_suite
)
from .metrics import BenchmarkMetrics, compute_all_metrics
from .leaderboard import Leaderboard, aggregate_experiment_results


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    # Method configuration
    n_generations: int = 50
    population_size: int = 30
    timeout_seconds: float = 60.0

    # Benchmark scope
    quick_mode: bool = False
    domain_filter: Optional[DomainType] = None
    difficulty_filter: Optional[Difficulty] = None
    method_filter: Optional[MethodCategory] = None

    # Output
    output_file: Optional[str] = None
    verbose: bool = True

    def to_synthesis_config(self) -> SynthesisConfig:
        """Convert to SynthesisConfig."""
        return SynthesisConfig(
            n_generations=self.n_generations,
            population_size=self.population_size,
            timeout_seconds=self.timeout_seconds
        )


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class MethodDomainResult:
    """Result of running one method on one domain."""
    method_name: str
    method_category: MethodCategory
    domain_name: str
    domain_type: DomainType

    synthesis_result: SynthesisResult
    metrics: BenchmarkMetrics

    success: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    config: BenchmarkConfig
    start_time: datetime
    end_time: Optional[datetime] = None

    results: List[MethodDomainResult] = field(default_factory=list)
    leaderboard: Optional[Leaderboard] = None

    # Summary stats
    n_methods: int = 0
    n_domains: int = 0
    n_runs: int = 0
    n_successes: int = 0
    total_time: float = 0.0

    def success_rate(self) -> float:
        return self.n_successes / self.n_runs if self.n_runs > 0 else 0.0


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs the complete benchmark suite."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.methods: List[SynthesisMethod] = []
        self.domains: List[Domain] = []

    def load_methods(self):
        """Load synthesis methods."""
        synthesis_config = self.config.to_synthesis_config()

        if self.config.method_filter:
            self.methods = get_methods_by_category(
                self.config.method_filter, synthesis_config)
        else:
            self.methods = get_all_methods(synthesis_config)

        if self.config.verbose:
            print(f"Loaded {len(self.methods)} methods")

    def load_domains(self):
        """Load test domains."""
        if self.config.quick_mode:
            suite = create_quick_suite()
            self.domains = suite.domains
        elif self.config.domain_filter:
            self.domains = get_domains_by_type(self.config.domain_filter)
        elif self.config.difficulty_filter:
            self.domains = get_domains_by_difficulty(self.config.difficulty_filter)
        else:
            self.domains = get_all_domains()

        if self.config.verbose:
            print(f"Loaded {len(self.domains)} domains")

    def run(self) -> BenchmarkRun:
        """Run the complete benchmark."""
        self.load_methods()
        self.load_domains()

        run = BenchmarkRun(
            config=self.config,
            start_time=datetime.now(),
            n_methods=len(self.methods),
            n_domains=len(self.domains),
            n_runs=len(self.methods) * len(self.domains)
        )

        if self.config.verbose:
            print(f"\nRunning {run.n_runs} experiments "
                  f"({run.n_methods} methods x {run.n_domains} domains)")
            print("=" * 70)

        leaderboard = Leaderboard()

        for domain in self.domains:
            if self.config.verbose:
                print(f"\n--- Domain: {domain.name} ({domain.domain_type.name}) ---")

            for method in self.methods:
                result = self._run_single(method, domain)
                run.results.append(result)

                if result.success:
                    run.n_successes += 1
                    leaderboard.add_entry(
                        method.name, method.category, domain, result.metrics)

                if self.config.verbose:
                    status = "✓" if result.success else "✗"
                    score = result.metrics.overall_score() if result.success else 0.0
                    print(f"  {status} {method.name:<25} "
                          f"score={score:.3f} "
                          f"acc={result.metrics.accuracy:.2f} "
                          f"time={result.synthesis_result.synthesis_time:.2f}s")

        run.end_time = datetime.now()
        run.total_time = (run.end_time - run.start_time).total_seconds()

        # Compute rankings
        leaderboard.compute_rankings()
        run.leaderboard = leaderboard

        return run

    def _run_single(
        self,
        method: SynthesisMethod,
        domain: Domain
    ) -> MethodDomainResult:
        """Run a single method on a single domain."""
        try:
            examples = domain.get_train_pairs()
            synthesis_result = method.synthesize(examples)
            metrics = compute_all_metrics(synthesis_result, domain)

            return MethodDomainResult(
                method_name=method.name,
                method_category=method.category,
                domain_name=domain.name,
                domain_type=domain.domain_type,
                synthesis_result=synthesis_result,
                metrics=metrics,
                success=True
            )

        except Exception as e:
            return MethodDomainResult(
                method_name=method.name,
                method_category=method.category,
                domain_name=domain.name,
                domain_type=domain.domain_type,
                synthesis_result=SynthesisResult(
                    method_name=method.name,
                    category=method.category,
                    statechart=None,
                    synthesis_time=0,
                    iterations=0,
                    success=False,
                    error_message=str(e)
                ),
                metrics=BenchmarkMetrics(),
                success=False,
                error=str(e)
            )


# =============================================================================
# Result Reporting
# =============================================================================

def print_summary(run: BenchmarkRun):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\nRun Statistics:")
    print(f"  Methods:    {run.n_methods}")
    print(f"  Domains:    {run.n_domains}")
    print(f"  Total runs: {run.n_runs}")
    print(f"  Successes:  {run.n_successes} ({run.success_rate():.1%})")
    print(f"  Total time: {run.total_time:.1f}s")

    if run.leaderboard:
        run.leaderboard.print_overall_leaderboard()
        run.leaderboard.print_category_comparison()


def export_results(run: BenchmarkRun, filename: str):
    """Export results to JSON file."""
    data = {
        'timestamp': run.start_time.isoformat(),
        'config': {
            'n_generations': run.config.n_generations,
            'population_size': run.config.population_size,
            'quick_mode': run.config.quick_mode
        },
        'stats': {
            'n_methods': run.n_methods,
            'n_domains': run.n_domains,
            'n_runs': run.n_runs,
            'n_successes': run.n_successes,
            'success_rate': run.success_rate(),
            'total_time': run.total_time
        },
        'results': [
            {
                'method': r.method_name,
                'category': r.method_category.name,
                'domain': r.domain_name,
                'domain_type': r.domain_type.name,
                'success': r.success,
                'metrics': r.metrics.to_dict() if r.success else None
            }
            for r in run.results
        ]
    }

    if run.leaderboard:
        data['leaderboard'] = json.loads(run.leaderboard.to_json())

    # Add known experiment results
    data['experiment_results'] = aggregate_experiment_results()

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults exported to: {filename}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description='Statechart Synthesis Benchmark')

    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark (easy domains only)')
    parser.add_argument('--domain', type=str,
                        choices=['games', 'regex', 'dialogue', 'code'],
                        help='Filter by domain type')
    parser.add_argument('--difficulty', type=str,
                        choices=['easy', 'medium', 'hard'],
                        help='Filter by difficulty')
    parser.add_argument('--generations', type=int, default=30,
                        help='Number of generations for evolutionary methods')
    parser.add_argument('--population', type=int, default=20,
                        help='Population size')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Build config
    config = BenchmarkConfig(
        n_generations=args.generations,
        population_size=args.population,
        quick_mode=args.quick,
        verbose=not args.quiet,
        output_file=args.output
    )

    if args.domain:
        config.domain_filter = DomainType[args.domain.upper()]
    if args.difficulty:
        config.difficulty_filter = Difficulty[args.difficulty.upper()]

    # Run benchmark
    runner = BenchmarkRunner(config)
    run = runner.run()

    # Report results
    print_summary(run)

    # Export
    if config.output_file:
        export_results(run, config.output_file)


def demo():
    """Quick demo of benchmark runner."""
    print("=" * 60)
    print("BENCHMARK RUNNER DEMO")
    print("=" * 60)

    config = BenchmarkConfig(
        n_generations=20,
        population_size=15,
        quick_mode=True,  # Fast
        verbose=True
    )

    runner = BenchmarkRunner(config)
    run = runner.run()

    print_summary(run)

    # Show aggregated experiment results
    print("\n--- Results from Completed Experiments ---")
    results = aggregate_experiment_results()
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['accuracy'])

    print(f"{'Method':<30} {'Accuracy':>10} {'Category':<15}")
    print("-" * 60)
    for method, data in sorted_results:
        print(f"{method:<30} {data['accuracy']:>10.3f} {data['category']:<15}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
