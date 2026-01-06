#!/usr/bin/env python3
"""
Improved Entry/Exit Actions Benchmark

Compares prompting strategies to fix exit action generation (50% → 90%+).

Strategies:
1. Baseline - Original prompt
2. Exit-heavy - More exit examples than entry
3. Explicit - Clear markers "on_exit means LEAVING"
4. Contrast - Side-by-side entry vs exit
5. Direct - Simple focused prompt (learned from L3 success)
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_entry_exit_actions.improved_generator import (
    ImprovedGenerator,
    PromptStrategy,
    GenerationResult,
    ENTRY_TESTS,
    EXIT_TESTS,
    BOTH_TESTS,
)


@dataclass
class StrategyResults:
    """Results for a single strategy."""
    strategy: str
    entry_correct: int
    entry_total: int
    exit_correct: int
    exit_total: int
    both_correct: int
    both_total: int

    @property
    def entry_pct(self) -> float:
        return self.entry_correct / self.entry_total * 100 if self.entry_total > 0 else 0

    @property
    def exit_pct(self) -> float:
        return self.exit_correct / self.exit_total * 100 if self.exit_total > 0 else 0

    @property
    def both_pct(self) -> float:
        return self.both_correct / self.both_total * 100 if self.both_total > 0 else 0


def run_benchmark(model=None, tokenizer=None) -> Dict[str, StrategyResults]:
    """Run benchmark comparing all strategies."""
    generator = ImprovedGenerator(model, tokenizer)

    results: Dict[PromptStrategy, List[GenerationResult]] = {
        s: [] for s in PromptStrategy
    }

    print("\n" + "=" * 70)
    print("EXIT ACTION IMPROVEMENT BENCHMARK")
    print("=" * 70)

    all_tests = [
        ("entry", ENTRY_TESTS),
        ("exit", EXIT_TESTS),
        ("both", BOTH_TESTS),
    ]

    # Run each strategy
    for strategy in PromptStrategy:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value.upper()}")
        print("=" * 70)

        for action_type, tests in all_tests:
            print(f"\n  [{action_type.upper()}]")

            for desc, expected_type in tests:
                result = generator.generate(desc, expected_type, strategy)
                results[strategy].append(result)

                status = "✓" if result.correct else "✗"
                detail = ""
                if not result.correct:
                    if result.error:
                        detail = f" ({result.error})"
                    elif expected_type == "exit" and result.has_entry and not result.has_exit:
                        detail = " (generated on_entry instead of on_exit!)"
                    elif expected_type == "entry" and result.has_exit and not result.has_entry:
                        detail = " (generated on_exit instead of on_entry!)"

                print(f"    [{status}] {desc[:45]}...{detail}")

    # Compute summary per strategy
    summary = {}
    for strategy in PromptStrategy:
        strategy_results = results[strategy]

        entry_results = [r for r in strategy_results if r.action_type == "entry"]
        exit_results = [r for r in strategy_results if r.action_type == "exit"]
        both_results = [r for r in strategy_results if r.action_type == "both"]

        summary[strategy.value] = StrategyResults(
            strategy=strategy.value,
            entry_correct=sum(1 for r in entry_results if r.correct),
            entry_total=len(entry_results),
            exit_correct=sum(1 for r in exit_results if r.correct),
            exit_total=len(exit_results),
            both_correct=sum(1 for r in both_results if r.correct),
            both_total=len(both_results),
        )

    return summary, results


def print_summary(summary: Dict[str, StrategyResults]) -> Dict[str, float]:
    """Print summary and find best strategy."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<12} {'Entry%':>10} {'Exit%':>10} {'Both%':>10}")
    print("-" * 50)

    best_exit_strategy = None
    best_exit_accuracy = 0

    for strategy_name, stats in summary.items():
        print(f"{strategy_name:<12} {stats.entry_pct:>9.0f}% {stats.exit_pct:>9.0f}% {stats.both_pct:>9.0f}%")

        if stats.exit_pct > best_exit_accuracy:
            best_exit_accuracy = stats.exit_pct
            best_exit_strategy = strategy_name

    print("-" * 50)

    baseline_exit = summary.get("baseline", StrategyResults("", 0, 1, 0, 1, 0, 1)).exit_pct
    improvement = best_exit_accuracy - baseline_exit

    print(f"\nBest EXIT strategy: {best_exit_strategy} ({best_exit_accuracy:.0f}%)")
    print(f"Improvement over baseline: {baseline_exit:.0f}% → {best_exit_accuracy:.0f}% (+{improvement:.0f}%)")

    return {
        "entry": summary.get("direct", summary.get("baseline")).entry_pct if summary else 0,
        "exit_baseline": baseline_exit,
        "exit_best": best_exit_accuracy,
        "exit_best_strategy": best_exit_strategy,
        "both": summary.get(best_exit_strategy, StrategyResults("", 0, 1, 0, 1, 0, 1)).both_pct,
        "by_strategy": {
            name: {"entry": s.entry_pct, "exit": s.exit_pct, "both": s.both_pct}
            for name, s in summary.items()
        }
    }


def analyze_exit_errors(results: Dict[PromptStrategy, List[GenerationResult]]):
    """Analyze exit action errors."""
    print("\n" + "=" * 70)
    print("EXIT ERROR ANALYSIS")
    print("=" * 70)

    for strategy in PromptStrategy:
        exit_results = [r for r in results[strategy] if r.action_type == "exit"]
        errors = [r for r in exit_results if not r.correct]

        if not errors:
            print(f"\n{strategy.value}: No exit errors!")
            continue

        confused = [r for r in errors if r.has_entry and not r.has_exit]
        no_action = [r for r in errors if not r.has_entry and not r.has_exit]
        other = [r for r in errors if r not in confused and r not in no_action]

        print(f"\n{strategy.value} exit errors ({len(errors)}/{len(exit_results)}):")
        if confused:
            print(f"  - Confused (generated entry instead): {len(confused)}")
        if no_action:
            print(f"  - No action generated: {len(no_action)}")
        if other:
            print(f"  - Other errors: {len(other)}")


def save_results(summary: Dict, metrics: Dict):
    """Save results to file."""
    results_dir = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_entry_exit_actions/results"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "IMPROVED_RESULTS.json")
    with open(results_path, "w") as f:
        json.dump({
            "session": "DDB5",
            "mode": "REAL",
            "model": "Qwen2.5-Coder-1.5B",
            "metrics": metrics,
            "summary": {k: {"entry": v.entry_pct, "exit": v.exit_pct, "both": v.both_pct}
                       for k, v in summary.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    print("[DDB5]: Exit Action Improvement Benchmark")

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
    analyze_exit_errors(results)
    save_results(summary, metrics)

    # Report
    print(f"\n[DDB5]: EXIT_ACTION_IMPROVED entry={metrics['entry']:.0f}%, exit_baseline={metrics['exit_baseline']:.0f}%, exit_improved={metrics['exit_best']:.0f}%, both={metrics['both']:.0f}%")
