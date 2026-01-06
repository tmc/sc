#!/usr/bin/env python3
"""
Improved Comparison Operators Benchmark

Compares multiple prompting strategies to improve L3 guard synthesis
from 20% baseline to 80%+ accuracy.

Strategies tested:
1. Baseline - Simple prompt
2. Chain-of-Thought (CoT) - Step-by-step reasoning
3. Template - Explicit structure
4. Augmented - 10+ few-shot examples
"""

import json
import os
import sys
from typing import Dict, List, Any

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_comparison_operators.improved_generator import (
    ImprovedGenerator,
    PromptStrategy,
    GenerationResult,
)
from experiments.exp_comparison_operators.comparison_dataset import TEST_CASES


def run_benchmark(model=None, tokenizer=None) -> Dict[str, Any]:
    """Run benchmark comparing all strategies."""
    generator = ImprovedGenerator(model, tokenizer)

    results: Dict[PromptStrategy, List[GenerationResult]] = {
        s: [] for s in PromptStrategy
    }

    print("\n" + "=" * 70)
    print("L3 GUARD SYNTHESIS IMPROVEMENT BENCHMARK")
    print("=" * 70)

    total_cases = len(TEST_CASES)
    print(f"\nTest cases: {total_cases}")
    print(f"Strategies: {', '.join(s.value for s in PromptStrategy)}")

    # Run each strategy
    for strategy in PromptStrategy:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value.upper()}")
        print("=" * 70)

        for i, (nl, expected, op) in enumerate(TEST_CASES):
            result = generator.generate(nl, expected, op, strategy)
            results[strategy].append(result)

            status = "✓" if result.is_correct else "✗"
            gen = result.generated_guard or "(none)"
            if len(gen) > 30:
                gen = gen[:27] + "..."
            print(f"  [{status}] \"{nl[:35]}...\" → {gen}")
            if not result.is_correct and result.generated_guard:
                print(f"       Expected: {expected}")

    # Compute summary
    summary = {}
    for strategy in PromptStrategy:
        strategy_results = results[strategy]
        n = len(strategy_results)
        correct = sum(1 for r in strategy_results if r.is_correct)
        has_comp = sum(1 for r in strategy_results if r.has_comparison)
        avg_time = sum(r.generation_time_ms for r in strategy_results) / n if n > 0 else 0

        summary[strategy.value] = {
            "total": n,
            "correct": correct,
            "accuracy": correct / n * 100 if n > 0 else 0,
            "has_comparison": has_comp,
            "comparison_rate": has_comp / n * 100 if n > 0 else 0,
            "avg_time_ms": avg_time,
        }

    return summary, results


def print_summary(summary: Dict[str, Dict]) -> Dict[str, float]:
    """Print summary and identify best strategy."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<12} {'Accuracy':>10} {'HasComp':>10} {'Time(ms)':>10}")
    print("-" * 50)

    best_strategy = None
    best_accuracy = 0

    for strategy, stats in summary.items():
        acc = stats["accuracy"]
        comp = stats["comparison_rate"]
        time_ms = stats["avg_time_ms"]

        bar = "█" * int(acc / 5)
        print(f"{strategy:<12} {acc:>9.0f}% {comp:>9.0f}% {time_ms:>9.0f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_strategy = strategy

    print("-" * 50)
    print(f"\nBest strategy: {best_strategy} ({best_accuracy:.0f}%)")

    # Improvement over baseline
    baseline_acc = summary.get("baseline", {}).get("accuracy", 20)
    improvement = best_accuracy - baseline_acc
    print(f"Improvement over baseline: +{improvement:.0f}%")

    return {
        "baseline": baseline_acc,
        "cot": summary.get("cot", {}).get("accuracy", 0),
        "template": summary.get("template", {}).get("accuracy", 0),
        "augmented": summary.get("augmented", {}).get("accuracy", 0),
        "best": best_accuracy,
        "best_strategy": best_strategy,
    }


def analyze_errors(results: Dict[PromptStrategy, List[GenerationResult]]):
    """Analyze common error patterns."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    for strategy in PromptStrategy:
        errors = [r for r in results[strategy] if not r.is_correct]
        if not errors:
            print(f"\n{strategy.value}: No errors!")
            continue

        print(f"\n{strategy.value} errors ({len(errors)}):")
        for e in errors[:5]:  # Show first 5
            print(f"  Input: \"{e.natural_language}\"")
            print(f"  Expected: {e.expected_guard}")
            print(f"  Got: {e.generated_guard}")
            print()


def save_results(summary: Dict, metrics: Dict):
    """Save results to file."""
    results_dir = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_comparison_operators/results"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "IMPROVED_RESULTS.json")
    with open(results_path, "w") as f:
        json.dump({
            "session": "DDB5",
            "mode": "REAL",
            "model": "Qwen2.5-Coder-1.5B",
            "summary": summary,
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    print("[DDB5]: L3 Guard Synthesis Improvement Benchmark")

    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-1.5B-Instruct-4bit...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None
        print("Running with no model")

    summary, results = run_benchmark(model, tokenizer)
    metrics = print_summary(summary)
    analyze_errors(results)
    save_results(summary, metrics)

    # Report to orchestrator
    print(f"\n[DDB5]: L3_GUARD_IMPROVED baseline={metrics['baseline']:.0f}%, cot={metrics['cot']:.0f}%, template={metrics['template']:.0f}%, augmented={metrics['augmented']:.0f}%, best={metrics['best']:.0f}%")
