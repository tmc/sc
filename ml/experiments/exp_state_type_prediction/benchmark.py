#!/usr/bin/env python3
"""
State Type Prediction Benchmark

Tests BASIC/OR/PARALLEL classification accuracy using template-based prompting.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_state_type_prediction.type_predictor import (
    StateTypePredictor,
    TypePredictionResult,
    StateType,
    BASIC_TESTS,
    OR_TESTS,
    PARALLEL_TESTS,
    EDGE_TESTS,
)


@dataclass
class CategoryResults:
    """Results for a test category."""
    name: str
    correct: int
    total: int
    results: List[TypePredictionResult]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total * 100 if self.total > 0 else 0


class StateTypeBenchmark:
    """Benchmark for state type prediction."""

    def __init__(self, model=None, tokenizer=None):
        self.predictor = StateTypePredictor(model, tokenizer)

    def run(self) -> Dict[str, CategoryResults]:
        """Run all test categories."""
        categories = {
            "BASIC": BASIC_TESTS,
            "OR": OR_TESTS,
            "PARALLEL": PARALLEL_TESTS,
            "EDGE": EDGE_TESTS,
        }

        results = {}

        print("\n" + "=" * 70)
        print("STATE TYPE PREDICTION BENCHMARK")
        print("=" * 70)

        for cat_name, tests in categories.items():
            print(f"\n{'=' * 70}")
            print(f"Category: {cat_name}")
            print("=" * 70)

            cat_results = []

            for desc, expected in tests:
                result = self.predictor.predict(desc, expected)
                cat_results.append(result)

                status = "OK" if result.correct else "FAIL"
                pred_str = result.predicted.name if result.predicted else "None"
                exp_str = expected.name

                marker = "[OK]" if result.correct else "[FAIL]"
                print(f"  {marker} {desc[:45]}...")
                if not result.correct:
                    print(f"        Expected: {exp_str}, Got: {pred_str}")

            correct = sum(1 for r in cat_results if r.correct)
            results[cat_name] = CategoryResults(
                name=cat_name,
                correct=correct,
                total=len(tests),
                results=cat_results,
            )

            print(f"\n  {cat_name}: {correct}/{len(tests)} ({results[cat_name].accuracy:.0f}%)")

        return results


def print_summary(results: Dict[str, CategoryResults]) -> Dict[str, float]:
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Category':<12} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 50)

    total_correct = 0
    total_tests = 0

    for cat_name, cat_result in results.items():
        print(f"{cat_name:<12} {cat_result.correct:>10} {cat_result.total:>10} {cat_result.accuracy:>11.0f}%")
        total_correct += cat_result.correct
        total_tests += cat_result.total

    overall = total_correct / total_tests * 100 if total_tests > 0 else 0
    print("-" * 50)
    print(f"{'OVERALL':<12} {total_correct:>10} {total_tests:>10} {overall:>11.0f}%")

    return {
        "basic": results.get("BASIC", CategoryResults("", 0, 1, [])).accuracy,
        "or": results.get("OR", CategoryResults("", 0, 1, [])).accuracy,
        "parallel": results.get("PARALLEL", CategoryResults("", 0, 1, [])).accuracy,
        "edge": results.get("EDGE", CategoryResults("", 0, 1, [])).accuracy,
        "overall": overall,
    }


def analyze_errors(results: Dict[str, CategoryResults]):
    """Analyze prediction errors."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    confusion = {
        "BASIC->OR": 0,
        "BASIC->PARALLEL": 0,
        "OR->BASIC": 0,
        "OR->PARALLEL": 0,
        "PARALLEL->BASIC": 0,
        "PARALLEL->OR": 0,
        "None": 0,
    }

    for cat_name, cat_result in results.items():
        errors = [r for r in cat_result.results if not r.correct]
        if not errors:
            continue

        print(f"\n{cat_name} errors ({len(errors)}):")
        for err in errors:
            pred_str = err.predicted.name if err.predicted else "None"
            exp_str = err.expected.name
            print(f"  - \"{err.description[:40]}...\"")
            print(f"    Expected: {exp_str}, Got: {pred_str}")

            if err.predicted is None:
                confusion["None"] += 1
            else:
                key = f"{exp_str}->{pred_str}"
                if key in confusion:
                    confusion[key] += 1

    print("\nConfusion Matrix:")
    for key, count in confusion.items():
        if count > 0:
            print(f"  {key}: {count}")


def save_results(results: Dict[str, CategoryResults], metrics: Dict[str, float]):
    """Save results to JSON."""
    results_dir = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_state_type_prediction/results"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "BENCHMARK_RESULTS.json")

    # Convert results to serializable format
    serializable = {
        "session": "DDB5",
        "model": "Qwen2.5-Coder-1.5B-Instruct-4bit",
        "metrics": metrics,
        "by_category": {},
    }

    for cat_name, cat_result in results.items():
        serializable["by_category"][cat_name] = {
            "correct": cat_result.correct,
            "total": cat_result.total,
            "accuracy": cat_result.accuracy,
            "errors": [
                {
                    "description": r.description,
                    "expected": r.expected.name,
                    "predicted": r.predicted.name if r.predicted else None,
                }
                for r in cat_result.results if not r.correct
            ],
        }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def run_benchmark(model=None, tokenizer=None) -> Dict[str, float]:
    """Run the full benchmark."""
    benchmark = StateTypeBenchmark(model, tokenizer)
    results = benchmark.run()
    metrics = print_summary(results)
    analyze_errors(results)
    save_results(results, metrics)
    return metrics


if __name__ == "__main__":
    print("[DDB5]: State Type Prediction Benchmark")

    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-1.5B-Instruct-4bit...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None
        print("Running with mock model")

    metrics = run_benchmark(model, tokenizer)

    # Report
    print(f"\n[DDB5]: STATE_TYPE_PRED basic={metrics['basic']:.0f}%, or={metrics['or']:.0f}%, parallel={metrics['parallel']:.0f}%, overall={metrics['overall']:.0f}%")
