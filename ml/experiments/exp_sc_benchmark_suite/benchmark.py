"""
Benchmark Runner: Main Orchestration for SC Benchmark Suite.

Runs all test cases against all methods and generates comparison report.

Usage:
    from ml.experiments.exp_sc_benchmark_suite import run_benchmark
    results = run_benchmark()

    # Or with custom config
    config = BenchmarkConfig(methods=[GenerationMethod.BASELINE])
    results = run_benchmark(config)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path

from .test_suite import (
    TestCase,
    TestSuite,
    Complexity,
    Feature,
    get_test_suite,
)
from .method_runner import (
    GenerationMethod,
    MethodResult,
    MethodRunner,
    get_runner,
)
from .report_generator import (
    ReportGenerator,
    MethodStats,
    generate_markdown_report,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    methods: List[GenerationMethod] = field(
        default_factory=lambda: list(GenerationMethod)
    )
    complexities: List[Complexity] = field(
        default_factory=lambda: list(Complexity)
    )
    max_tests: Optional[int] = None  # Limit number of tests
    output_dir: str = "."
    save_report: bool = True
    verbose: bool = True


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    total_tests: int = 0
    total_runs: int = 0
    duration_seconds: float = 0.0

    # By method
    method_stats: Dict[GenerationMethod, MethodStats] = field(default_factory=dict)

    # By complexity
    validity_by_complexity: Dict[Complexity, Dict[GenerationMethod, float]] = field(
        default_factory=dict
    )

    # Raw results
    results: Dict[str, Dict[GenerationMethod, MethodResult]] = field(default_factory=dict)

    # Report path
    report_path: Optional[str] = None

    def summary(self) -> str:
        """Get brief summary."""
        lines = [
            f"Total tests: {self.total_tests}",
            f"Total runs: {self.total_runs}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "Validity by Method:",
        ]

        for method, stats in sorted(
            self.method_stats.items(),
            key=lambda x: x[1].validity_rate,
            reverse=True
        ):
            lines.append(f"  {method.name}: {stats.validity_rate:.1%}")

        return "\n".join(lines)


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Coordinates running all tests across all methods
    and generates comparison reports.
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.suite = get_test_suite()
        self.runners: Dict[GenerationMethod, MethodRunner] = {}
        self.report_generator = ReportGenerator()

        # Initialize runners
        for method in self.config.methods:
            self.runners[method] = get_runner(method)

    def run(self) -> BenchmarkResults:
        """Run complete benchmark suite."""
        start_time = time.time()

        if self.config.verbose:
            print("=" * 70)
            print("SC BENCHMARK SUITE")
            print("=" * 70)
            print(f"Methods: {[m.name for m in self.config.methods]}")
            print(f"Complexities: {[c.name for c in self.config.complexities]}")
            print("-" * 70)

        # Filter tests by complexity
        tests = [
            t for t in self.suite.tests
            if t.complexity in self.config.complexities
        ]

        # Apply max_tests limit
        if self.config.max_tests:
            tests = tests[:self.config.max_tests]

        if self.config.verbose:
            print(f"Running {len(tests)} tests x {len(self.config.methods)} methods "
                  f"= {len(tests) * len(self.config.methods)} total runs")
            print("-" * 70)

        results = BenchmarkResults()
        results.total_tests = len(tests)
        results.total_runs = len(tests) * len(self.config.methods)

        # Run all tests
        for i, test in enumerate(tests):
            if self.config.verbose:
                progress = (i + 1) / len(tests) * 100
                print(f"\r[{progress:5.1f}%] Testing: {test.id:20s}", end="", flush=True)

            results.results[test.id] = {}

            for method in self.config.methods:
                runner = self.runners[method]
                result = runner.generate(test.prompt, test.min_states)
                results.results[test.id][method] = result
                self.report_generator.add_result(test, method, result)

        if self.config.verbose:
            print("\r" + " " * 60 + "\r", end="")  # Clear progress line

        # Compute statistics
        self.report_generator.compute_statistics()
        results.method_stats = self.report_generator.method_stats.copy()

        # Compute validity by complexity
        for complexity in self.config.complexities:
            results.validity_by_complexity[complexity] = {}
            cs = self.report_generator.complexity_stats[complexity]
            for method in self.config.methods:
                ms = cs.by_method[method]
                results.validity_by_complexity[complexity][method] = ms.validity_rate

        results.duration_seconds = time.time() - start_time

        # Generate and save report
        if self.config.save_report:
            report_path = Path(self.config.output_dir) / "benchmark_report.md"
            report = self.report_generator.generate_report(str(report_path))
            results.report_path = str(report_path)

            if self.config.verbose:
                print(f"\nReport saved to: {report_path}")

        # Print summary
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)
            print(results.summary())
            print("=" * 70)

        return results


def run_benchmark(
    config: BenchmarkConfig = None,
    verbose: bool = True
) -> BenchmarkResults:
    """
    Convenience function to run benchmark.

    Args:
        config: Optional configuration
        verbose: Print progress

    Returns:
        BenchmarkResults with all data
    """
    if config is None:
        config = BenchmarkConfig(verbose=verbose)
    else:
        config.verbose = verbose

    runner = BenchmarkRunner(config)
    return runner.run()


def run_quick_benchmark(verbose: bool = True) -> BenchmarkResults:
    """Run quick benchmark with subset of tests."""
    config = BenchmarkConfig(
        max_tests=15,  # 3 per complexity
        verbose=verbose,
    )
    return run_benchmark(config)


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Run quick benchmark
    print("\n1. Running quick benchmark (15 tests)...")
    results = run_quick_benchmark(verbose=False)

    print(f"   Tests: {results.total_tests}")
    print(f"   Runs: {results.total_runs}")
    print(f"   Duration: {results.duration_seconds:.2f}s")

    # Check validity rates
    print("\n2. Validity rates:")
    for method, stats in results.method_stats.items():
        print(f"   {method.name}: {stats.validity_rate:.1%}")

    # Check complexity breakdown
    print("\n3. By complexity:")
    for complexity, method_rates in results.validity_by_complexity.items():
        rates = [f"{m.name}:{r:.0%}" for m, r in method_rates.items()]
        print(f"   {complexity.name}: {', '.join(rates)}")

    # Check report was generated
    print("\n4. Report:")
    if results.report_path:
        print(f"   Saved to: {results.report_path}")
    else:
        print("   Not saved")

    # Validate results structure
    print("\n5. Validation:")
    assert results.total_tests > 0, "No tests run"
    assert results.total_runs > 0, "No runs completed"
    assert len(results.method_stats) == len(GenerationMethod), "Missing method stats"
    print("   All validations passed!")

    print("\n" + "=" * 60)
    print("Benchmark system tests complete!")
    print("=" * 60)

    return results


def run_full_benchmark():
    """Run complete benchmark with all 55 tests."""
    print("=" * 70)
    print("FULL SC BENCHMARK SUITE")
    print("=" * 70)

    config = BenchmarkConfig(
        verbose=True,
        save_report=True,
    )

    results = run_benchmark(config)

    # Print detailed summary
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    print("\n### Method Comparison ###")
    print(f"{'Method':<12} {'Valid':>6} {'Total':>6} {'Rate':>8} {'Latency':>10}")
    print("-" * 50)

    for method in GenerationMethod:
        stats = results.method_stats[method]
        print(f"{method.name:<12} {stats.valid_count:>6} {stats.total_tests:>6} "
              f"{stats.validity_rate:>7.1%} {stats.avg_latency_ms:>9.2f}ms")

    print("\n### Validity by Complexity ###")
    for complexity in Complexity:
        print(f"\n{complexity.name}:")
        for method, rate in results.validity_by_complexity[complexity].items():
            print(f"  {method.name}: {rate:.1%}")

    print("\n" + "=" * 70)
    print(f"Benchmark complete! Report: {results.report_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Run full benchmark when executed directly
    run_full_benchmark()
