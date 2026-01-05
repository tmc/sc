"""
Benchmark for SC debugging accuracy.

Measures how well the LLM can identify bugs in statecharts.
"""

from dataclasses import dataclass
from typing import Dict, List

from .test_cases import generate_test_cases, DebugTestCase
from .debugger import SCDebugger, DebugResult


@dataclass
class DebugBenchmarkResult:
    """Results from debugging benchmark."""
    total_cases: int
    correct_bug_type: int
    correct_location: int
    fully_correct: int  # Both type and location correct
    results: List[DebugResult]

    @property
    def type_accuracy(self) -> float:
        return self.correct_bug_type / self.total_cases if self.total_cases > 0 else 0

    @property
    def location_accuracy(self) -> float:
        return self.correct_location / self.total_cases if self.total_cases > 0 else 0

    @property
    def fix_accuracy(self) -> float:
        """Overall fix accuracy - both type and location correct."""
        return self.fully_correct / self.total_cases if self.total_cases > 0 else 0


def run_debug_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    max_cases: int = None,
) -> DebugBenchmarkResult:
    """
    Run the debugging benchmark.

    Args:
        model_id: Model to use for debugging
        max_cases: Limit number of test cases (None = all)

    Returns:
        DebugBenchmarkResult with accuracy metrics
    """
    print("=" * 70)
    print("SC Debugging Benchmark")
    print("=" * 70)

    # Generate test cases
    cases = generate_test_cases()
    if max_cases:
        cases = cases[:max_cases]

    print(f"\nTest cases: {len(cases)}")
    print(f"Model: {model_id}")

    # Create debugger
    debugger = SCDebugger(model_id)

    # Run debugging on each case
    results = []
    correct_type = 0
    correct_loc = 0
    fully_correct = 0

    print("\n" + "-" * 70)
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.name}")
        print(f"  Expected bug: {case.bug_type.value}")

        result = debugger.debug(case)
        results.append(result)

        print(f"  Identified: {result.identified_bug_type or 'None'}")
        print(f"  Location: {result.identified_location or 'None'}")

        type_ok = "✓" if result.correct_bug_type else "✗"
        loc_ok = "✓" if result.correct_location else "✗"
        print(f"  Type correct: {type_ok}, Location correct: {loc_ok}")

        if result.correct_bug_type:
            correct_type += 1
        if result.correct_location:
            correct_loc += 1
        if result.correct_bug_type and result.correct_location:
            fully_correct += 1

    # Create result
    benchmark_result = DebugBenchmarkResult(
        total_cases=len(cases),
        correct_bug_type=correct_type,
        correct_location=correct_loc,
        fully_correct=fully_correct,
        results=results,
    )

    return benchmark_result


def print_benchmark_results(result: DebugBenchmarkResult):
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nTotal cases: {result.total_cases}")
    print(f"\nAccuracy Metrics:")
    print(f"  Bug type identification: {result.correct_bug_type}/{result.total_cases} = {result.type_accuracy:.0%}")
    print(f"  Bug location identification: {result.correct_location}/{result.total_cases} = {result.location_accuracy:.0%}")
    print(f"  Full fix accuracy (both correct): {result.fully_correct}/{result.total_cases} = {result.fix_accuracy:.0%}")

    print("\n" + "-" * 70)
    print("Per-case breakdown:")
    print("-" * 70)
    print(f"{'Case':<30} {'Type':^8} {'Location':^8} {'Overall':^8}")
    print("-" * 70)

    for r in result.results:
        type_mark = "✓" if r.correct_bug_type else "✗"
        loc_mark = "✓" if r.correct_location else "✗"
        overall = "✓" if r.correct_bug_type and r.correct_location else "✗"
        print(f"{r.test_name:<30} {type_mark:^8} {loc_mark:^8} {overall:^8}")

    print("=" * 70)

    return result.fix_accuracy


def test_benchmark():
    """Run full benchmark and report."""
    result = run_debug_benchmark()
    fix_accuracy = print_benchmark_results(result)

    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)
    print(f"\nSC_DEBUG: fix_accuracy={fix_accuracy:.0%}")
    print(f"  type_accuracy={result.type_accuracy:.0%}")
    print(f"  location_accuracy={result.location_accuracy:.0%}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    test_benchmark()
