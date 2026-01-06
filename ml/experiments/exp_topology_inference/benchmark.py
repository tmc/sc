"""
Topology Inference Benchmark.

Tests LLM's ability to infer SC topology from execution traces.
Target: 50%+ overall accuracy.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

from . import (
    ALL_TEST_CASES,
    TopologyTestCase,
    InferredTopology,
    evaluate_topology,
    ground_truth_topology,
)
from .topology_inferrer import TopologyInferrer, InferenceResult


@dataclass
class BenchmarkResult:
    """Result for a single test case."""
    test_case: TopologyTestCase
    inference_result: InferenceResult
    metrics: Dict[str, float]
    elapsed: float


def run_benchmark(
    model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    methods: List[str] = None,
) -> Dict:
    """Run the topology inference benchmark."""
    if methods is None:
        methods = ["enumeration", "simple"]

    print("=" * 70)
    print("TOPOLOGY INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Test cases: {len(ALL_TEST_CASES)}")

    inferrer = TopologyInferrer(model_name=model_name)

    results_by_method: Dict[str, List[BenchmarkResult]] = {m: [] for m in methods}

    for method in methods:
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print("=" * 70)

        for tc in ALL_TEST_CASES:
            print(f"\n--- {tc.name} ({tc.category}) ---")
            print(f"Traces: {tc.traces[:2]}{'...' if len(tc.traces) > 2 else ''}")

            t0 = time.time()
            result = inferrer.infer(tc.traces, method=method)
            elapsed = time.time() - t0

            # Evaluate
            metrics = evaluate_topology(result.topology, tc)

            # Determine pass/fail
            status = "PASS" if metrics["overall"] >= 0.8 else "FAIL"

            print(f"[{status}] States: {result.topology.states}")
            print(f"      Trans: {result.topology.transitions}")
            print(f"      Initial: {result.topology.initial_state}")
            print(f"      Features: {result.topology.features}")
            print(f"      Expected states: {tc.expected_states}")
            print(f"      Expected trans: {tc.expected_transitions}")
            print(f"      Metrics: states={metrics['states_acc']:.0%} trans={metrics['transitions_acc']:.0%} init={metrics['initial_acc']:.0%}")
            print(f"      Overall: {metrics['overall']:.0%} | Time: {elapsed:.1f}s")

            if not result.parse_success:
                print(f"      Parse failed. Raw: {result.raw_output[:100]}...")

            results_by_method[method].append(BenchmarkResult(
                test_case=tc,
                inference_result=result,
                metrics=metrics,
                elapsed=elapsed,
            ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY METHOD")
    print("=" * 70)

    method_summaries = {}
    for method in methods:
        results = results_by_method[method]
        n = len(results)

        avg_states = sum(r.metrics["states_acc"] for r in results) / n
        avg_trans = sum(r.metrics["transitions_acc"] for r in results) / n
        avg_initial = sum(r.metrics["initial_acc"] for r in results) / n
        avg_overall = sum(r.metrics["overall"] for r in results) / n
        avg_time = sum(r.elapsed for r in results) / n

        pass_count = sum(1 for r in results if r.metrics["overall"] >= 0.8)

        method_summaries[method] = {
            "states_acc": avg_states,
            "transitions_acc": avg_trans,
            "initial_acc": avg_initial,
            "overall": avg_overall,
            "pass_count": pass_count,
            "total": n,
            "avg_time": avg_time,
        }

        print(f"\n{method.upper()}:")
        print(f"  States: {avg_states:.0%}")
        print(f"  Transitions: {avg_trans:.0%}")
        print(f"  Initial: {avg_initial:.0%}")
        print(f"  Overall: {avg_overall:.0%}")
        print(f"  Pass rate (>80%): {pass_count}/{n} ({pass_count/n:.0%})")
        print(f"  Avg time: {avg_time:.1f}s")

    # Best method
    best_method = max(method_summaries.keys(), key=lambda m: method_summaries[m]["overall"])
    best_overall = method_summaries[best_method]["overall"]

    print(f"\nBest method: {best_method} ({best_overall:.0%})")

    # Breakdown by category
    print("\n" + "-" * 70)
    print(f"BREAKDOWN BY CATEGORY (using {best_method})")
    print("-" * 70)

    best_results = results_by_method[best_method]
    by_category = {}
    for r in best_results:
        cat = r.test_case.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "sum_overall": 0, "pass": 0}
        by_category[cat]["total"] += 1
        by_category[cat]["sum_overall"] += r.metrics["overall"]
        if r.metrics["overall"] >= 0.8:
            by_category[cat]["pass"] += 1

    for cat, stats in sorted(by_category.items()):
        avg = stats["sum_overall"] / stats["total"]
        print(f"  {cat}: {avg:.0%} ({stats['pass']}/{stats['total']} pass)")

    return {
        "method_summaries": method_summaries,
        "best_method": best_method,
        "best_overall": best_overall,
        "by_category": by_category,
        "results": results_by_method,
    }


def run_baseline_comparison():
    """Compare LLM inference vs deterministic baseline."""
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    inferrer = TopologyInferrer()

    llm_correct = 0
    baseline_correct = 0
    total = len(ALL_TEST_CASES)

    for tc in ALL_TEST_CASES:
        # LLM inference
        llm_result = inferrer.infer(tc.traces, method="enumeration")
        llm_metrics = evaluate_topology(llm_result.topology, tc)

        # Deterministic baseline
        baseline = ground_truth_topology(tc.traces)
        baseline_metrics = evaluate_topology(baseline, tc)

        if llm_metrics["overall"] >= 0.8:
            llm_correct += 1
        if baseline_metrics["overall"] >= 0.8:
            baseline_correct += 1

        print(f"{tc.name}: LLM={llm_metrics['overall']:.0%} Baseline={baseline_metrics['overall']:.0%}")

    print(f"\nLLM: {llm_correct}/{total} ({llm_correct/total:.0%})")
    print(f"Baseline: {baseline_correct}/{total} ({baseline_correct/total:.0%})")

    return {
        "llm": llm_correct / total,
        "baseline": baseline_correct / total,
    }


def main():
    """Run full benchmark and report."""
    import os

    # Run benchmark
    results = run_benchmark()

    # Extract metrics for report
    best = results["best_method"]
    summary = results["method_summaries"][best]

    states_acc = summary["states_acc"]
    trans_acc = summary["transitions_acc"]
    overall_acc = summary["overall"]

    by_cat = results["by_category"]
    linear_acc = by_cat.get("linear", {}).get("sum_overall", 0) / max(by_cat.get("linear", {}).get("total", 1), 1)
    cycle_acc = by_cat.get("cycle", {}).get("sum_overall", 0) / max(by_cat.get("cycle", {}).get("total", 1), 1)
    branch_acc = by_cat.get("branch", {}).get("sum_overall", 0) / max(by_cat.get("branch", {}).get("total", 1), 1)
    selfloop_acc = by_cat.get("selfloop", {}).get("sum_overall", 0) / max(by_cat.get("selfloop", {}).get("total", 1), 1)

    # Print report
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)

    report = f"[9D1B]: TOPOLOGY_INFERENCE states={states_acc:.0%}, transitions={trans_acc:.0%}, overall={overall_acc:.0%}, by_pattern=[linear:{linear_acc:.0%}, cycle:{cycle_acc:.0%}, branch:{branch_acc:.0%}, selfloop:{selfloop_acc:.0%}]"
    print(report)

    # Send to orchestrator if available
    orchestrator_sid = "B90CCCD4"
    try:
        import subprocess
        subprocess.run(
            ["it2", "session", "send-text", orchestrator_sid, report],
            capture_output=True,
            timeout=5,
        )
        print(f"\nReport sent to orchestrator {orchestrator_sid}")
    except Exception as e:
        print(f"\nCould not send to orchestrator: {e}")
        print("Report text above can be sent manually.")

    return results


if __name__ == "__main__":
    main()
