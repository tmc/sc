"""
Benchmark for Terminal State Prediction.

Evaluates LLM's ability to predict final configuration after
executing an event sequence on various statechart types.
"""

import sys
import time
from typing import List, Dict, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

from . import (
    TestCase,
    generate_all_test_cases,
    configuration_accuracy,
    execute_to_completion,
    ALL_TEST_MACHINES,
)
from .state_predictor import TerminalStatePredictor, PredictionResult


@dataclass
class CategoryResult:
    """Results for a single category."""
    category: str
    total: int = 0
    exact_matches: int = 0
    jaccard_sum: float = 0.0
    parse_failures: int = 0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.total if self.total > 0 else 0.0

    @property
    def avg_jaccard(self) -> float:
        return self.jaccard_sum / self.total if self.total > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    total_cases: int
    exact_matches: int
    avg_jaccard: float
    by_category: Dict[str, CategoryResult] = field(default_factory=dict)
    inference_time: float = 0.0
    parse_failures: int = 0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.total_cases if self.total_cases > 0 else 0.0


def run_benchmark(
    cases_per_machine: int = 10,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run the terminal state prediction benchmark."""
    print("=" * 60)
    print("TERMINAL STATE PREDICTION BENCHMARK")
    print("=" * 60)

    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = generate_all_test_cases(cases_per_machine)
    print(f"Generated {len(test_cases)} test cases")

    # Count by category
    category_counts = defaultdict(int)
    for case in test_cases:
        category_counts[case.category] += 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} cases")

    # Initialize predictor
    print("\nLoading model...")
    predictor = TerminalStatePredictor()
    predictor.load_model()

    # Run predictions
    print("\nRunning predictions...")
    start_time = time.time()

    results_by_category: Dict[str, CategoryResult] = {}
    total_exact = 0
    total_jaccard = 0.0
    total_parse_fail = 0

    for i, case in enumerate(test_cases):
        # Get prediction
        result = predictor.predict(
            case.sc_json,
            case.initial_state,
            case.events,
        )

        # Compute metrics
        metrics = configuration_accuracy(
            result.predicted_states,
            case.expected_final,
        )

        # Update category stats
        cat = case.category
        if cat not in results_by_category:
            results_by_category[cat] = CategoryResult(category=cat)

        cat_result = results_by_category[cat]
        cat_result.total += 1
        cat_result.exact_matches += int(metrics["exact_match"])
        cat_result.jaccard_sum += metrics["jaccard"]
        if not result.parse_success:
            cat_result.parse_failures += 1
            total_parse_fail += 1

        total_exact += int(metrics["exact_match"])
        total_jaccard += metrics["jaccard"]

        # Progress
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_cases)}")

        # Detailed output for failures (limited)
        if verbose and metrics["exact_match"] == 0 and i < 5:
            print(f"\n  MISMATCH [{cat}]:")
            print(f"    Events: {case.events}")
            print(f"    Expected: {case.expected_final}")
            print(f"    Predicted: {result.predicted_states}")
            print(f"    Raw: {result.raw_output[:80]}...")

    inference_time = time.time() - start_time

    # Compute overall results
    benchmark_result = BenchmarkResult(
        total_cases=len(test_cases),
        exact_matches=total_exact,
        avg_jaccard=total_jaccard / len(test_cases) if test_cases else 0.0,
        by_category=results_by_category,
        inference_time=inference_time,
        parse_failures=total_parse_fail,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Exact match: {benchmark_result.exact_matches}/{benchmark_result.total_cases} ({benchmark_result.exact_accuracy*100:.1f}%)")
    print(f"  Avg Jaccard: {benchmark_result.avg_jaccard*100:.1f}%")
    print(f"  Parse failures: {benchmark_result.parse_failures}")
    print(f"  Inference time: {benchmark_result.inference_time:.2f}s")

    print(f"\nBy Category:")
    for cat in ["linear", "cycle", "hierarchy", "parallel"]:
        if cat in results_by_category:
            r = results_by_category[cat]
            print(f"  {cat}: {r.exact_matches}/{r.total} exact ({r.exact_accuracy*100:.1f}%), jaccard={r.avg_jaccard*100:.1f}%")

    return benchmark_result


def quick_test():
    """Run a quick sanity test."""
    print("=" * 60)
    print("QUICK TEST")
    print("=" * 60)

    from . import (
        LINEAR_CHAIN_SC, CYCLE_SC, HIERARCHY_SC, PARALLEL_SC,
        TraceExecutor
    )

    predictor = TerminalStatePredictor()
    predictor.load_model()

    test_cases = [
        (LINEAR_CHAIN_SC, "A", ["NEXT", "NEXT"], {"C"}),
        (LINEAR_CHAIN_SC, "A", ["NEXT", "NEXT", "NEXT"], {"D"}),
        (CYCLE_SC, "S1", ["TICK"], {"S2"}),
        (CYCLE_SC, "S1", ["TICK", "TICK", "TICK"], {"S1"}),  # Full cycle
        (HIERARCHY_SC, "Idle", ["START"], {"Running"}),
        (HIERARCHY_SC, "Idle", ["START", "PAUSE"], {"Paused"}),
        (PARALLEL_SC, "Off", ["POWER"], {"MotorIdle", "DisplayOff"}),
        (PARALLEL_SC, "Off", ["POWER", "RUN"], {"MotorRunning", "DisplayOff"}),
    ]

    correct = 0
    total = len(test_cases)

    for sc, initial, events, expected in test_cases:
        # Get ground truth
        executor = TraceExecutor(sc)
        actual = executor.execute_trace(events)

        # Get prediction
        result = predictor.predict(sc, initial, events)

        # Check
        is_correct = result.predicted_states == expected
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "FAIL"
        print(f"[{status}] {sc['name']}: {initial} + {events}")
        print(f"      Expected: {expected}")
        print(f"      Actual:   {actual}")
        print(f"      Predicted: {result.predicted_states}")

    print(f"\nQuick test: {correct}/{total} correct")
    return correct, total


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Terminal State Prediction Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--cases", type=int, default=10, help="Cases per machine")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        result = run_benchmark(
            cases_per_machine=args.cases,
            verbose=not args.quiet,
        )

        # Format for reporting
        by_type = []
        for cat in ["linear", "cycle", "hierarchy", "parallel"]:
            if cat in result.by_category:
                r = result.by_category[cat]
                by_type.append(f"{cat}:{r.exact_accuracy*100:.0f}%")

        print(f"\n--- REPORT FORMAT ---")
        print(f"TERMINAL_STATE exact={result.exact_accuracy*100:.0f}%, jaccard={result.avg_jaccard*100:.0f}%, by_type=[{', '.join(by_type)}]")


if __name__ == "__main__":
    main()
