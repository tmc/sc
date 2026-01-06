"""
Benchmark for comprehensive action execution testing.

Runs all test cases and reports accuracy by category.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from .test_cases import TEST_CASES, TEST_CASES_BY_CATEGORY, TestCase
from .action_executor import (
    ActionExecutor,
    ExecutionResult,
    compare_contexts,
    compare_action_order,
)


def run_single_test(executor: ActionExecutor, tc: TestCase) -> Dict[str, Any]:
    """Run a single test case and return results."""
    result = executor.execute(
        states=tc.states,
        hierarchy=tc.hierarchy,
        initial_state=tc.initial_state,
        entry_actions=tc.entry_actions,
        exit_actions=tc.exit_actions,
        transition_actions=tc.transition_actions,
        transitions=tc.transitions,
        initial_context=tc.initial_context,
        events=tc.events,
        guards=tc.guards,
    )

    # Check correctness
    state_correct = result.final_state == tc.expected_final_state
    context_correct = compare_contexts(tc.expected_context, result.final_context)
    order_correct = compare_action_order(tc.expected_action_order, result.action_order)

    return {
        "name": tc.name,
        "category": tc.category,
        "success": result.success,
        "state_correct": state_correct,
        "context_correct": context_correct,
        "order_correct": order_correct,
        "expected_state": tc.expected_final_state,
        "actual_state": result.final_state,
        "expected_context": tc.expected_context,
        "actual_context": result.final_context,
        "expected_order": tc.expected_action_order,
        "actual_order": result.action_order,
        "errors": result.errors,
    }


def run_benchmark(
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full benchmark."""
    executor = ActionExecutor(model_path)

    results = []
    category_stats = {}

    for tc in TEST_CASES:
        if verbose:
            print(f"  Testing: {tc.name}...", end=" ", flush=True)

        result = run_single_test(executor, tc)
        results.append(result)

        # Update category stats
        cat = tc.category
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0,
                "state_correct": 0,
                "context_correct": 0,
                "order_correct": 0,
            }
        category_stats[cat]["total"] += 1
        if result["state_correct"]:
            category_stats[cat]["state_correct"] += 1
        if result["context_correct"]:
            category_stats[cat]["context_correct"] += 1
        if result["order_correct"]:
            category_stats[cat]["order_correct"] += 1

        if verbose:
            status = ""
            if result["state_correct"]:
                status += "S"
            if result["context_correct"]:
                status += "C"
            if result["order_correct"]:
                status += "O"
            print(f"[{status or 'FAIL'}]")

    # Calculate overall stats
    total = len(results)
    overall = {
        "total": total,
        "state_correct": sum(1 for r in results if r["state_correct"]),
        "context_correct": sum(1 for r in results if r["context_correct"]),
        "order_correct": sum(1 for r in results if r["order_correct"]),
        "all_correct": sum(
            1 for r in results
            if r["state_correct"] and r["context_correct"] and r["order_correct"]
        ),
    }

    # Calculate percentages
    def pct(n, d):
        return round(100 * n / d) if d > 0 else 0

    summary = {
        "overall": {
            "state": pct(overall["state_correct"], total),
            "context": pct(overall["context_correct"], total),
            "order": pct(overall["order_correct"], total),
            "all": pct(overall["all_correct"], total),
        },
        "by_category": {},
    }

    for cat, stats in category_stats.items():
        t = stats["total"]
        summary["by_category"][cat] = {
            "state": pct(stats["state_correct"], t),
            "context": pct(stats["context_correct"], t),
            "order": pct(stats["order_correct"], t),
        }

    return {
        "results": results,
        "category_stats": category_stats,
        "overall": overall,
        "summary": summary,
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
    }


def print_summary(benchmark_result: Dict[str, Any]):
    """Print a summary of benchmark results."""
    summary = benchmark_result["summary"]

    print("\n" + "=" * 60)
    print("ACTION EXECUTION BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nModel: {benchmark_result['model']}")
    print(f"Total tests: {benchmark_result['overall']['total']}")

    print("\nOVERALL:")
    print(f"  State:   {summary['overall']['state']}%")
    print(f"  Context: {summary['overall']['context']}%")
    print(f"  Order:   {summary['overall']['order']}%")
    print(f"  All:     {summary['overall']['all']}%")

    print("\nBY CATEGORY:")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}:")
        print(f"    State: {stats['state']}% | Context: {stats['context']}% | Order: {stats['order']}%")

    # Show failures
    failures = [r for r in benchmark_result["results"] if not (
        r["state_correct"] and r["context_correct"] and r["order_correct"]
    )]
    if failures:
        print("\nFAILURES:")
        for f in failures[:5]:  # Show first 5
            issues = []
            if not f["state_correct"]:
                issues.append(f"state: {f['actual_state']} vs {f['expected_state']}")
            if not f["context_correct"]:
                issues.append(f"context mismatch")
            if not f["order_correct"]:
                issues.append(f"order: {f['actual_order']} vs {f['expected_order']}")
            print(f"  {f['name']}: {', '.join(issues)}")


def save_results(benchmark_result: Dict[str, Any], output_dir: str = None):
    """Save results to file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(benchmark_result, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    import sys
    import time

    print("=" * 60)
    print("exp_action_execution_comprehensive Benchmark")
    print("=" * 60)

    # Parse args for subset testing
    subset = None
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run only one test per category for quick validation
        from .test_cases import TEST_CASES_BY_CATEGORY
        subset_tests = []
        for cat, tests in TEST_CASES_BY_CATEGORY.items():
            subset_tests.append(tests[0])  # First test from each category
        TEST_CASES.clear()
        TEST_CASES.extend(subset_tests)
        print(f"Quick mode: {len(TEST_CASES)} tests")

    start = time.time()
    result = run_benchmark(verbose=True)
    elapsed = time.time() - start

    print_summary(result)
    print(f"\nTime: {elapsed:.1f}s ({elapsed/len(result['results']):.1f}s/test)")
    save_results(result)

    # Output for orchestrator
    s = result["summary"]
    entry_pct = s["by_category"].get("entry", {}).get("context", 0)
    exit_pct = s["by_category"].get("exit", {}).get("context", 0)
    trans_pct = s["by_category"].get("transition", {}).get("context", 0)
    order_pct = s["by_category"].get("order", {}).get("order", 0)
    ctx_pct = s["by_category"].get("context_mutation", {}).get("context", 0)

    print(f"\n[DDB5]: ACTION_EXEC entry={entry_pct}%, exit={exit_pct}%, trans={trans_pct}%, order={order_pct}%, context_mut={ctx_pct}%")
