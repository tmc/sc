"""
History States Comprehensive Benchmark.

Tests LLM understanding of shallow (H) vs deep (H*) history semantics.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any

from . import (
    ALL_TEST_CASES,
    HistoryQuestion,
    categorize_test,
    evaluate_prediction,
)
from .history_predictor import HistoryPredictor, PredictionResult


@dataclass
class TestResult:
    """Result for a single test case."""
    name: str
    category: str
    question: HistoryQuestion
    prediction: PredictionResult
    evaluation: Dict[str, Any]
    elapsed: float


def run_benchmark(
    model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
) -> Dict:
    """Run the history states benchmark."""
    print("=" * 70)
    print("HISTORY STATES COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Test cases: {len(ALL_TEST_CASES)}")

    predictor = HistoryPredictor(model_name=model_name)
    predictor.load_model()

    results: List[TestResult] = []

    for name, tc in ALL_TEST_CASES:
        category = categorize_test(tc)
        print(f"\n--- {name} ({category}) ---")
        print(f"History: {tc.history_type}")
        print(f"Before: {tc.active_before_exit}")
        print(f"Expected: {tc.expected_config}")

        t0 = time.time()
        prediction = predictor.predict(tc)
        elapsed = time.time() - t0

        evaluation = evaluate_prediction(prediction.predicted_config, tc.expected_config)

        status = "PASS" if evaluation["correct"] else "FAIL"
        print(f"Predicted: {prediction.predicted_config}")
        print(f"[{status}] Jaccard: {evaluation['jaccard']:.0%}")
        if evaluation["missing"]:
            print(f"  Missing: {evaluation['missing']}")
        if evaluation["extra"]:
            print(f"  Extra: {evaluation['extra']}")
        print(f"Time: {elapsed:.1f}s")

        results.append(TestResult(
            name=name,
            category=category,
            question=tc,
            prediction=prediction,
            evaluation=evaluation,
            elapsed=elapsed,
        ))

    # Summary by category
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)

    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"correct": 0, "total": 0, "jaccard_sum": 0}
        categories[r.category]["total"] += 1
        categories[r.category]["jaccard_sum"] += r.evaluation["jaccard"]
        if r.evaluation["correct"]:
            categories[r.category]["correct"] += 1

    for cat, stats in sorted(categories.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        avg_jaccard = stats["jaccard_sum"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {acc:.0%} exact ({stats['correct']}/{stats['total']}), {avg_jaccard:.0%} jaccard")

    # Overall metrics
    print("\n" + "-" * 70)
    total = len(results)
    correct = sum(1 for r in results if r.evaluation["correct"])
    avg_jaccard = sum(r.evaluation["jaccard"] for r in results) / total if total > 0 else 0
    avg_time = sum(r.elapsed for r in results) / total if total > 0 else 0

    print(f"Overall: {correct}/{total} ({correct/total:.0%}) exact match")
    print(f"Average Jaccard: {avg_jaccard:.0%}")
    print(f"Average time: {avg_time:.1f}s")

    # Compute category-specific metrics for report
    shallow_acc = categories.get("shallow", {}).get("correct", 0) / max(categories.get("shallow", {}).get("total", 1), 1)
    deep_acc = categories.get("deep", {}).get("correct", 0) / max(categories.get("deep", {}).get("total", 1), 1)
    no_prior_acc = categories.get("no_prior", {}).get("correct", 0) / max(categories.get("no_prior", {}).get("total", 1), 1)
    parallel_acc = categories.get("parallel", {}).get("correct", 0) / max(categories.get("parallel", {}).get("total", 1), 1)
    overall_acc = correct / total if total > 0 else 0

    return {
        "results": results,
        "categories": categories,
        "shallow_acc": shallow_acc,
        "deep_acc": deep_acc,
        "no_prior_acc": no_prior_acc,
        "parallel_acc": parallel_acc,
        "overall_acc": overall_acc,
        "avg_jaccard": avg_jaccard,
    }


def main():
    """Run benchmark and report."""
    results = run_benchmark()

    # Print report
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)

    shallow = results["shallow_acc"]
    deep = results["deep_acc"]
    no_prior = results["no_prior_acc"]
    parallel = results["parallel_acc"]
    overall = results["overall_acc"]

    report = f"[9D1B]: HISTORY_STATES shallow={shallow:.0%}, deep={deep:.0%}, no_prior={no_prior:.0%}, parallel={parallel:.0%}, overall={overall:.0%}"
    print(report)

    # Send to orchestrator
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

    return results


if __name__ == "__main__":
    main()
