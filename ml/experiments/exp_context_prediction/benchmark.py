#!/usr/bin/env python3
"""
Context Prediction Benchmark

Evaluates LLM ability to predict final context values.
Tests across complexity levels L1-L4.
"""

import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_execution_prediction import (
    CounterMachine,
    AccumulatorMachine,
    BranchingMachine,
    execute_to_completion,
)
from .context_predictor import ContextPredictor, PredictionResult


@dataclass
class TestCase:
    """A single test case."""
    name: str
    level: str  # L1, L2, L3, L4
    sc_json: Dict
    initial_context: Dict[str, Any]
    events: List[str]
    expected_context: Dict[str, Any]


@dataclass
class LevelResult:
    """Results for a complexity level."""
    level: str
    total: int
    exact_matches: int
    partial_accuracy: float
    mae: float

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.total * 100 if self.total > 0 else 0


def create_test_cases() -> List[TestCase]:
    """Create test cases for each complexity level."""
    cases = []

    # L1: Single increment (count++)
    for target in [3, 5, 7]:
        machine = CounterMachine(target=target)
        sc = machine.to_json()
        ctx = machine.initial_context.copy()
        events = ["BEGIN"] + ["TICK"] * target
        result = execute_to_completion(sc, events, ctx)

        cases.append(TestCase(
            name=f"counter_{target}",
            level="L1",
            sc_json=sc,
            initial_context=ctx,
            events=events,
            expected_context=result.final_context,
        ))

    # L2: Arithmetic (score = count * multiplier style)
    # Using branching machine with different scores
    for score, threshold in [(75, 50), (30, 50), (80, 60)]:
        machine = BranchingMachine(threshold=threshold)
        ctx = machine.initial_context.copy()
        ctx["score"] = score
        sc = machine.to_json()
        events = ["EVALUATE", "CONTINUE", "FINISH"]
        result = execute_to_completion(sc, events, ctx)

        cases.append(TestCase(
            name=f"branch_score{score}_thresh{threshold}",
            level="L2",
            sc_json=sc,
            initial_context=ctx,
            events=events,
            expected_context=result.final_context,
        ))

    # L3: Conditionals - branch based on value
    for score in [25, 55, 90]:
        machine = BranchingMachine(threshold=50)
        ctx = machine.initial_context.copy()
        ctx["score"] = score
        sc = machine.to_json()
        events = ["EVALUATE", "CONTINUE", "FINISH"]
        result = execute_to_completion(sc, events, ctx)

        cases.append(TestCase(
            name=f"conditional_{score}",
            level="L3",
            sc_json=sc,
            initial_context=ctx,
            events=events,
            expected_context=result.final_context,
        ))

    # L4: Accumulation (sum over loop)
    for values in [[1, 2, 3], [10, 20, 30], [5, 5, 5, 5]]:
        machine = AccumulatorMachine(values=values)
        sc = machine.to_json()
        ctx = machine.initial_context.copy()
        events = ["START"] + ["NEXT"] * len(values)
        result = execute_to_completion(sc, events, ctx)

        cases.append(TestCase(
            name=f"accumulator_{sum(values)}",
            level="L4",
            sc_json=sc,
            initial_context=ctx,
            events=events,
            expected_context=result.final_context,
        ))

    return cases


class ContextPredictionBenchmark:
    """Benchmark for context prediction."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = ContextPredictor(model, tokenizer)
        self.results: Dict[str, List[PredictionResult]] = {
            "L1": [], "L2": [], "L3": [], "L4": []
        }

    def run(self, test_cases: Optional[List[TestCase]] = None) -> Dict[str, LevelResult]:
        """Run benchmark on test cases."""
        if test_cases is None:
            test_cases = create_test_cases()

        for case in test_cases:
            print(f"\n[{case.level}] {case.name}:")
            print(f"  Events: {' -> '.join(case.events[:5])}{'...' if len(case.events) > 5 else ''}")

            result = self.predictor.predict(
                case.sc_json,
                case.initial_context,
                case.events,
                ground_truth=case.expected_context,
            )

            self.results[case.level].append(result)

            status = "✓" if result.exact_match else "✗"
            print(f"  {status} Expected: {case.expected_context}")
            print(f"    Predicted: {result.predicted_context}")
            if result.exact_match:
                print("    EXACT MATCH")
            else:
                print(f"    Partial: {result.partial_matches}/{result.total_variables}")

        return self._compute_level_results()

    def _compute_level_results(self) -> Dict[str, LevelResult]:
        """Compute results per level."""
        level_results = {}

        for level, results in self.results.items():
            if not results:
                level_results[level] = LevelResult(level, 0, 0, 0.0, 0.0)
                continue

            total = len(results)
            exact = sum(1 for r in results if r.exact_match)
            partial_sum = sum(r.partial_matches / r.total_variables
                             for r in results if r.total_variables > 0)
            partial_acc = partial_sum / total if total > 0 else 0.0
            mae_sum = sum(r.mae for r in results)
            mae_avg = mae_sum / total if total > 0 else 0.0

            level_results[level] = LevelResult(
                level=level,
                total=total,
                exact_matches=exact,
                partial_accuracy=partial_acc * 100,
                mae=mae_avg,
            )

        return level_results

    def print_summary(self) -> Dict[str, float]:
        """Print summary and return key metrics."""
        level_results = self._compute_level_results()

        print("\n" + "=" * 60)
        print("CONTEXT PREDICTION RESULTS")
        print("=" * 60)

        print(f"\n{'Level':<8} {'Total':>6} {'Exact%':>10} {'Partial%':>10} {'MAE':>10}")
        print("-" * 50)

        total_exact = 0
        total_partial = 0
        total_tests = 0
        total_mae = 0

        for level in ["L1", "L2", "L3", "L4"]:
            r = level_results[level]
            print(f"{r.level:<8} {r.total:>6} {r.exact_accuracy:>9.0f}% {r.partial_accuracy:>9.0f}% {r.mae:>10.2f}")
            total_exact += r.exact_matches
            total_partial += r.partial_accuracy * r.total
            total_tests += r.total
            total_mae += r.mae * r.total

        avg_exact = total_exact / total_tests * 100 if total_tests > 0 else 0
        avg_partial = total_partial / total_tests if total_tests > 0 else 0
        avg_mae = total_mae / total_tests if total_tests > 0 else 0

        print("-" * 50)
        print(f"{'OVERALL':<8} {total_tests:>6} {avg_exact:>9.0f}% {avg_partial:>9.0f}% {avg_mae:>10.2f}")

        return {
            "exact_accuracy": avg_exact,
            "partial_accuracy": avg_partial,
            "mae": avg_mae,
            "by_level": {
                level: r.exact_accuracy for level, r in level_results.items()
            }
        }


def run_benchmark(model=None, tokenizer=None) -> Dict[str, Any]:
    """Run full benchmark."""
    benchmark = ContextPredictionBenchmark(model, tokenizer)
    benchmark.run()
    return benchmark.print_summary()


if __name__ == "__main__":
    print("[DDB5]: Context Prediction Benchmark")

    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-1.5B-Instruct-4bit...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None
        print("Running with no model (will produce empty predictions)")

    summary = run_benchmark(model, tokenizer)

    print(f"\n[DDB5] FINAL: exact={summary['exact_accuracy']:.0f}%, partial={summary['partial_accuracy']:.0f}%, mae={summary['mae']:.2f}")
    print(f"[DDB5] By level: L1={summary['by_level']['L1']:.0f}%, L2={summary['by_level']['L2']:.0f}%, L3={summary['by_level']['L3']:.0f}%, L4={summary['by_level']['L4']:.0f}%")

    # Save results
    results_path = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_context_prediction/results/REAL_RESULTS.json"
    with open(results_path, "w") as f:
        json.dump({
            "session": "DDB5",
            "mode": "REAL",
            "model": "Qwen2.5-Coder-1.5B",
            "exact_accuracy": summary["exact_accuracy"],
            "partial_accuracy": summary["partial_accuracy"],
            "mae": summary["mae"],
            "by_level": summary["by_level"],
        }, f, indent=2)
    print(f"[DDB5] Results saved to {results_path}")
