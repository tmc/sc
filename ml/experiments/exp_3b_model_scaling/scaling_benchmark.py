#!/usr/bin/env python3
"""
Scaling Benchmark: Compare 0.5B, 1.5B, and 3B models on semantic SC tasks.

Runs:
1. SC Repair benchmark
2. SC Debugging benchmark
3. SC Completion benchmark

For each model size and compares results.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

# MLX imports
from mlx_lm import load, generate

# Import sys path for local experiments
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


class ModelSize(str, Enum):
    """Available model sizes."""
    SMALL = "0.5B"
    MEDIUM = "1.5B"
    LARGE = "3B"


# Model IDs for each size
MODEL_IDS = {
    ModelSize.SMALL: "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ModelSize.MEDIUM: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ModelSize.LARGE: "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
}


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_size: str
    model_id: str
    task: str
    accuracy: float
    total_cases: int
    successful: int
    inference_time_s: float
    avg_time_per_sample: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class ScalingResults:
    """Full scaling benchmark results."""
    results: List[BenchmarkResult]
    comparison_table: Dict[str, Dict[str, float]]
    summary: str


def run_repair_benchmark(model_id: str, model_size: str) -> BenchmarkResult:
    """Run SC repair benchmark with specified model."""
    from experiments.exp_sc_repair.repair_engine import SCRepairEngine, RepairConfig
    from experiments.exp_sc_repair.benchmark import get_benchmark_cases, RepairBenchmark

    print(f"\n--- SC REPAIR ({model_size}) ---")
    print(f"Model: {model_id}")

    # Configure with specific model
    config = RepairConfig(model_name=model_id, use_llm=True)
    benchmark = RepairBenchmark(config)

    start = time.time()
    result = benchmark.run(use_llm=True)
    elapsed = time.time() - start

    accuracy = result.accuracy * 100
    print(f"Accuracy: {accuracy:.0f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/result.total_cases:.1f}s/sample)")

    return BenchmarkResult(
        model_size=model_size,
        model_id=model_id,
        task="repair",
        accuracy=accuracy,
        total_cases=result.total_cases,
        successful=result.successful_repairs,
        inference_time_s=elapsed,
        avg_time_per_sample=elapsed / result.total_cases if result.total_cases > 0 else 0,
        details={"per_category": result.per_category}
    )


def run_debugging_benchmark(model_id: str, model_size: str) -> BenchmarkResult:
    """Run SC debugging benchmark with specified model."""
    from experiments.exp_sc_debugging.benchmark import run_debug_benchmark

    print(f"\n--- SC DEBUGGING ({model_size}) ---")
    print(f"Model: {model_id}")

    start = time.time()
    result = run_debug_benchmark(model_id=model_id)
    elapsed = time.time() - start

    # Fix accuracy = both type and location correct
    accuracy = result.fix_accuracy * 100
    print(f"Fix Accuracy: {accuracy:.0f}%")
    print(f"Type Accuracy: {result.type_accuracy * 100:.0f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/result.total_cases:.1f}s/sample)")

    return BenchmarkResult(
        model_size=model_size,
        model_id=model_id,
        task="debug",
        accuracy=accuracy,
        total_cases=result.total_cases,
        successful=result.fully_correct,
        inference_time_s=elapsed,
        avg_time_per_sample=elapsed / result.total_cases if result.total_cases > 0 else 0,
        details={
            "type_accuracy": result.type_accuracy * 100,
            "location_accuracy": result.location_accuracy * 100,
        }
    )


def run_completion_benchmark(model_id: str, model_size: str) -> BenchmarkResult:
    """Run SC completion benchmark with specified model."""
    from experiments.exp_schema_guided_sampling.sc_completion_benchmark import run_benchmark

    print(f"\n--- SC COMPLETION ({model_size}) ---")
    print(f"Model: {model_id}")

    start = time.time()
    result = run_benchmark(model_id=model_id)
    elapsed = time.time() - start

    accuracy = result["accuracy"]
    print(f"Accuracy: {accuracy:.0f}%")
    print(f"Time: {elapsed:.1f}s")

    return BenchmarkResult(
        model_size=model_size,
        model_id=model_id,
        task="complete",
        accuracy=accuracy,
        total_cases=result["total"],
        successful=result["valid_count"],
        inference_time_s=elapsed,
        avg_time_per_sample=elapsed / result["total"] if result["total"] > 0 else 0,
    )


def run_single_model_benchmarks(model_size: ModelSize) -> List[BenchmarkResult]:
    """Run all benchmarks for a single model size."""
    model_id = MODEL_IDS[model_size]
    results = []

    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_size.value}")
    print(f"{'='*60}")

    # Run each benchmark
    try:
        results.append(run_repair_benchmark(model_id, model_size.value))
    except Exception as e:
        print(f"Repair benchmark failed: {e}")
        results.append(BenchmarkResult(
            model_size=model_size.value, model_id=model_id,
            task="repair", accuracy=0, total_cases=0, successful=0,
            inference_time_s=0, avg_time_per_sample=0,
            details={"error": str(e)}
        ))

    try:
        results.append(run_debugging_benchmark(model_id, model_size.value))
    except Exception as e:
        print(f"Debug benchmark failed: {e}")
        results.append(BenchmarkResult(
            model_size=model_size.value, model_id=model_id,
            task="debug", accuracy=0, total_cases=0, successful=0,
            inference_time_s=0, avg_time_per_sample=0,
            details={"error": str(e)}
        ))

    try:
        results.append(run_completion_benchmark(model_id, model_size.value))
    except Exception as e:
        print(f"Completion benchmark failed: {e}")
        results.append(BenchmarkResult(
            model_size=model_size.value, model_id=model_id,
            task="complete", accuracy=0, total_cases=0, successful=0,
            inference_time_s=0, avg_time_per_sample=0,
            details={"error": str(e)}
        ))

    return results


def build_comparison_table(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Build comparison table: task -> {model_size -> accuracy}."""
    table = {
        "repair": {},
        "debug": {},
        "complete": {},
    }

    for r in results:
        if r.task in table:
            table[r.task][r.model_size] = r.accuracy

    return table


def format_comparison_table(table: Dict[str, Dict[str, float]]) -> str:
    """Format comparison table as markdown."""
    lines = [
        "| Task | 0.5B | 1.5B | 3B |",
        "|------|------|------|-----|",
    ]

    for task in ["repair", "debug", "complete"]:
        row = f"| {task.capitalize()} |"
        for size in ["0.5B", "1.5B", "3B"]:
            val = table.get(task, {}).get(size, "-")
            if isinstance(val, (int, float)):
                row += f" {val:.0f}% |"
            else:
                row += f" {val} |"
        lines.append(row)

    return "\n".join(lines)


def run_full_scaling_benchmark(
    sizes: Optional[List[ModelSize]] = None,
    save_results: bool = True,
) -> ScalingResults:
    """
    Run full scaling benchmark across model sizes.

    Args:
        sizes: Model sizes to test (default: all)
        save_results: Save to results.json

    Returns:
        ScalingResults with all data
    """
    if sizes is None:
        sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]

    print("=" * 70)
    print("SC MODEL SCALING BENCHMARK")
    print("=" * 70)
    print(f"Testing sizes: {[s.value for s in sizes]}")

    all_results = []

    for size in sizes:
        results = run_single_model_benchmarks(size)
        all_results.extend(results)

    # Build comparison
    comparison = build_comparison_table(all_results)
    table_str = format_comparison_table(comparison)

    # Summary
    print("\n" + "=" * 70)
    print("SCALING RESULTS")
    print("=" * 70)
    print(table_str)

    # Calculate inference time comparison
    time_by_size = {}
    for r in all_results:
        if r.model_size not in time_by_size:
            time_by_size[r.model_size] = []
        time_by_size[r.model_size].append(r.avg_time_per_sample)

    avg_times = {s: sum(t)/len(t) if t else 0 for s, t in time_by_size.items()}

    print("\nInference Time (avg per sample):")
    for size, t in sorted(avg_times.items()):
        print(f"  {size}: {t:.1f}s")

    # Build summary string
    summary_parts = []
    for task in ["repair", "debug", "complete"]:
        for size, acc in comparison.get(task, {}).items():
            summary_parts.append(f"{task}_{size}={acc:.0f}%")

    summary = ", ".join(summary_parts)

    result = ScalingResults(
        results=all_results,
        comparison_table=comparison,
        summary=summary,
    )

    # Save results
    if save_results:
        results_path = Path(__file__).parent / "results.json"
        with open(results_path, "w") as f:
            json.dump({
                "results": [asdict(r) for r in all_results],
                "comparison": comparison,
                "summary": summary,
                "avg_inference_time": avg_times,
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return result


def run_3b_only() -> ScalingResults:
    """Quick benchmark of just the 3B model."""
    return run_full_scaling_benchmark(sizes=[ModelSize.LARGE])


def demo():
    """Run demo with all model sizes."""
    result = run_full_scaling_benchmark()

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"\n{format_comparison_table(result.comparison_table)}")
    print(f"\nSummary: {result.summary}")


if __name__ == "__main__":
    demo()
