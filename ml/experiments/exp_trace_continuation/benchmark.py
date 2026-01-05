"""
Trace Continuation Benchmark.

Evaluates LLM ability to predict next N states given partial trace.
"""

import time
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .trace_continuer import (
    TraceContinuer,
    ContinuationResult,
    generate_continuation_dataset,
)


@dataclass
class BenchmarkConfig:
    """Configuration for continuation benchmark."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    n_samples: int = 15
    continuation_lengths: List[int] = None
    max_tokens: int = 100

    def __post_init__(self):
        if self.continuation_lengths is None:
            self.continuation_lengths = [1, 3, 5]


def evaluate_continuation(
    predicted: List[str],
    actual: List[str],
    n_requested: int,
) -> Dict:
    """Evaluate continuation prediction quality."""
    # Exact matches
    n_correct = 0
    for i, (p, a) in enumerate(zip(predicted, actual)):
        if p == a:
            n_correct += 1
        else:
            break  # Stop at first mismatch for strict evaluation

    # First state correct
    first_correct = len(predicted) > 0 and len(actual) > 0 and predicted[0] == actual[0]

    # Any correct (relaxed)
    any_correct = sum(1 for p in predicted if p in actual)

    return {
        "n_correct_prefix": n_correct,
        "first_correct": first_correct,
        "any_correct": any_correct,
        "predicted_count": len(predicted),
        "actual_count": len(actual),
        "n_requested": n_requested,
    }


def run_continuation_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run the full continuation benchmark."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 70)
    print("TRACE CONTINUATION BENCHMARK")
    print("=" * 70)

    # Generate dataset
    print(f"\nGenerating dataset with {config.n_samples} samples...")
    dataset = generate_continuation_dataset(
        n_samples=config.n_samples,
        continuation_lengths=config.continuation_lengths,
    )
    print(f"Generated {len(dataset)} continuation tasks")

    if not dataset:
        print("ERROR: No dataset generated")
        return {"error": "No dataset"}

    # Load model
    continuer = TraceContinuer(model_path=config.model_path)

    # Results by continuation length
    results_by_n = defaultdict(list)

    print("\n" + "-" * 70)
    print("Running predictions...")
    print("-" * 70)

    for i, task in enumerate(dataset):
        n_steps = task["n_steps"]
        partial = task["partial_trace"]
        actual = task["actual_continuation"]

        t0 = time.time()
        result = continuer.predict_next_states(
            task["sc_json"],
            partial,
            n_steps=n_steps,
            max_tokens=config.max_tokens,
        )
        elapsed = time.time() - t0

        # Evaluate
        eval_result = evaluate_continuation(
            result.predicted_states,
            actual,
            n_steps,
        )

        # Update result
        result.actual_states = actual
        result.n_correct = eval_result["n_correct_prefix"]

        results_by_n[n_steps].append({
            "task": task,
            "result": result,
            "eval": eval_result,
            "time": elapsed,
        })

        # Print progress
        status = "OK" if eval_result["first_correct"] else "FAIL"
        pred_str = result.predicted_states[:3] if result.predicted_states else ["?"]
        print(f"[{i+1:2d}] n={n_steps} [{status}] {' -> '.join(partial[-2:])} ... "
              f"pred={pred_str} actual={actual[:3]} ({elapsed:.1f}s)")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("RESULTS BY CONTINUATION LENGTH")
    print("=" * 70)

    summary = {}
    for n_steps in sorted(results_by_n.keys()):
        results = results_by_n[n_steps]
        n_total = len(results)

        if n_total == 0:
            continue

        first_correct = sum(1 for r in results if r["eval"]["first_correct"])
        all_correct = sum(1 for r in results if r["eval"]["n_correct_prefix"] == n_steps)
        avg_correct = sum(r["eval"]["n_correct_prefix"] for r in results) / n_total
        avg_time = sum(r["time"] for r in results) / n_total

        first_rate = first_correct / n_total
        all_rate = all_correct / n_total

        summary[f"next_{n_steps}"] = {
            "total": n_total,
            "first_correct": first_correct,
            "first_rate": first_rate,
            "all_correct": all_correct,
            "all_rate": all_rate,
            "avg_correct": avg_correct,
            "avg_time": avg_time,
        }

        print(f"\nNext {n_steps} states:")
        print(f"  First correct:  {first_rate:6.1%} ({first_correct}/{n_total})")
        print(f"  All correct:    {all_rate:6.1%} ({all_correct}/{n_total})")
        print(f"  Avg correct:    {avg_correct:.2f}/{n_steps}")
        print(f"  Avg time:       {avg_time:.2f}s")

    # Overall summary
    all_results = [r for results in results_by_n.values() for r in results]
    total = len(all_results)
    if total > 0:
        overall_first = sum(1 for r in all_results if r["eval"]["first_correct"])
        overall_first_rate = overall_first / total

        print("\n" + "=" * 70)
        print(f"OVERALL: First state correct = {overall_first_rate:.1%} ({overall_first}/{total})")
        print("=" * 70)

        summary["overall"] = {
            "total": total,
            "first_correct": overall_first,
            "first_rate": overall_first_rate,
        }

    return summary


def main():
    """Run benchmark and print report."""
    config = BenchmarkConfig(
        n_samples=15,
        continuation_lengths=[1, 3, 5],
    )

    summary = run_continuation_benchmark(config)

    # Print report for orchestrator
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)

    rates = []
    for n in [1, 3, 5]:
        key = f"next_{n}"
        if key in summary:
            rate = summary[key]["first_rate"]
            rates.append(f"next_{n}={rate:.0%}")

    print(f"TRACE_CONTINUATION {', '.join(rates)}")


if __name__ == "__main__":
    main()
