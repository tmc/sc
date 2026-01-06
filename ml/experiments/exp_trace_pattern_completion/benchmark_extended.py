#!/usr/bin/env python3
"""
Extended Benchmark: Compare differential learning approaches.

Tests both simple patterns and complex cases that challenge
the basic transition-map approach.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .benchmark import TEST_CASES as SIMPLE_CASES
from .complex_cases import COMPLEX_TEST_CASES
from .differential_learner import (
    DifferentialLearner,
    UnigramLearner,
    NGramLearner,
    StackAwareLearner,
    ModeAwareLearner,
    CounterAwareLearner,
    EnsembleLearner,
)


@dataclass
class ExtendedResult:
    """Result for one test case with multiple learners."""
    name: str
    challenge: str
    expected: List[str]
    predictions: Dict[str, List[str]]
    correct: Dict[str, bool]
    time_ms: float


def run_extended_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Run extended benchmark comparing all learners.
    """
    print("=" * 70)
    print("EXTENDED TRACE PATTERN COMPLETION BENCHMARK")
    print("=" * 70)

    # Combine simple and complex cases
    all_cases = []

    # Add simple cases with challenge type
    for case in SIMPLE_CASES:
        case_copy = dict(case)
        case_copy["challenge"] = case.get("pattern_type", "simple")
        all_cases.append(case_copy)

    # Add complex cases
    all_cases.extend(COMPLEX_TEST_CASES)

    print(f"\nTotal test cases: {len(all_cases)}")
    print(f"  Simple: {len(SIMPLE_CASES)}")
    print(f"  Complex: {len(COMPLEX_TEST_CASES)}")

    # Initialize learners
    learner_names = ["unigram", "2gram", "3gram", "stack", "mode", "counter", "ensemble"]

    # Track results
    results: List[ExtendedResult] = []
    by_challenge: Dict[str, List[ExtendedResult]] = {}
    by_learner: Dict[str, Dict[str, int]] = {name: {"correct": 0, "total": 0} for name in learner_names}

    for case in all_cases:
        name = case["name"]
        challenge = case.get("challenge", "simple")
        examples = case["examples"]
        partial = case["partial"]
        expected_1 = case.get("expected_1", [])

        if verbose:
            print(f"\n[{name}] Challenge: {challenge}")
            print(f"  Partial: {partial[-5:] if len(partial) > 5 else partial}...")
            print(f"  Expected: {expected_1}")

        start = time.time()

        # Train differential learner
        diff_learner = DifferentialLearner()
        diff_learner.learn(examples)

        # Get predictions from all approaches
        all_preds = diff_learner.predict_all(partial, n=1)

        elapsed = (time.time() - start) * 1000

        # Check correctness
        predictions = {name: r.predicted for name, r in all_preds.items()}
        correct = {name: predictions[name] == expected_1 for name in predictions}

        # Record result
        result = ExtendedResult(
            name=name,
            challenge=challenge,
            expected=expected_1,
            predictions=predictions,
            correct=correct,
            time_ms=elapsed,
        )
        results.append(result)

        # Track by challenge type
        if challenge not in by_challenge:
            by_challenge[challenge] = []
        by_challenge[challenge].append(result)

        # Track by learner
        for learner_name in learner_names:
            by_learner[learner_name]["total"] += 1
            if correct.get(learner_name, False):
                by_learner[learner_name]["correct"] += 1

        if verbose:
            for learner_name in learner_names:
                pred = predictions.get(learner_name, [])
                status = "✓" if correct.get(learner_name, False) else "✗"
                print(f"    {learner_name:12}: {pred} {status}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY BY LEARNER")
    print("=" * 70)

    print(f"\n{'Learner':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 42)

    learner_accuracies = {}
    for learner_name in learner_names:
        correct = by_learner[learner_name]["correct"]
        total = by_learner[learner_name]["total"]
        acc = correct / total * 100 if total > 0 else 0
        learner_accuracies[learner_name] = acc
        print(f"{learner_name:<12} {correct:>8} {total:>8} {acc:>9.1f}%")

    # Best learner
    best_learner = max(learner_accuracies.keys(), key=lambda k: learner_accuracies[k])
    print(f"\nBest learner: {best_learner} ({learner_accuracies[best_learner]:.1f}%)")

    # === BY CHALLENGE TYPE ===
    print("\n" + "=" * 70)
    print("SUMMARY BY CHALLENGE TYPE")
    print("=" * 70)

    challenge_summary = {}
    for challenge, challenge_results in sorted(by_challenge.items()):
        n = len(challenge_results)

        # Get accuracy per learner for this challenge
        learner_acc = {}
        for learner_name in learner_names:
            correct = sum(1 for r in challenge_results if r.correct.get(learner_name, False))
            learner_acc[learner_name] = correct / n * 100 if n > 0 else 0

        best = max(learner_acc.keys(), key=lambda k: learner_acc[k])
        challenge_summary[challenge] = {
            "n": n,
            "learner_acc": learner_acc,
            "best": best,
            "best_acc": learner_acc[best],
        }

        print(f"\n{challenge.upper()} ({n} cases):")
        print(f"  {'Learner':<12} {'Accuracy':>10}")
        print(f"  {'-'*24}")
        for learner_name in learner_names:
            acc = learner_acc[learner_name]
            marker = " *" if learner_name == best else ""
            print(f"  {learner_name:<12} {acc:>9.1f}%{marker}")

    # === DIFFERENTIAL ANALYSIS ===
    print("\n" + "=" * 70)
    print("DIFFERENTIAL ANALYSIS")
    print("=" * 70)

    # Where did ensemble beat unigram?
    ensemble_wins = []
    unigram_wins = []
    ties = []

    for r in results:
        ensemble_correct = r.correct.get("ensemble", False)
        unigram_correct = r.correct.get("unigram", False)

        if ensemble_correct and not unigram_correct:
            ensemble_wins.append(r)
        elif unigram_correct and not ensemble_correct:
            unigram_wins.append(r)
        elif ensemble_correct == unigram_correct:
            ties.append(r)

    print(f"\nEnsemble vs Unigram:")
    print(f"  Ensemble wins: {len(ensemble_wins)}")
    print(f"  Unigram wins: {len(unigram_wins)}")
    print(f"  Ties: {len(ties)}")

    if ensemble_wins:
        print(f"\n  Ensemble wins on:")
        for r in ensemble_wins[:5]:
            print(f"    - {r.name} ({r.challenge})")

    if unigram_wins:
        print(f"\n  Unigram wins on:")
        for r in unigram_wins[:5]:
            print(f"    - {r.name} ({r.challenge})")

    # Which challenges need specialized learners?
    print("\n\nSpecialized Learner Wins:")
    specialized = ["stack", "mode", "counter"]
    for spec in specialized:
        spec_only_wins = [
            r for r in results
            if r.correct.get(spec, False) and not r.correct.get("unigram", False)
        ]
        if spec_only_wins:
            print(f"  {spec}: {len(spec_only_wins)} unique wins")
            for r in spec_only_wins[:3]:
                print(f"    - {r.name}")

    return {
        "learner_accuracies": learner_accuracies,
        "by_challenge": challenge_summary,
        "results": results,
        "ensemble_vs_unigram": {
            "ensemble_wins": len(ensemble_wins),
            "unigram_wins": len(unigram_wins),
            "ties": len(ties),
        },
    }


def format_report(results: Dict[str, Any]) -> str:
    """Format for orchestrator report."""
    acc = results["learner_accuracies"]
    evsu = results["ensemble_vs_unigram"]

    parts = [f"{k}:{v:.0f}%" for k, v in sorted(acc.items())]
    learners_str = ", ".join(parts)

    return (
        f"TRACE_PATTERN_EXTENDED "
        f"[{learners_str}] "
        f"ensemble_vs_unigram=[+{evsu['ensemble_wins']}/-{evsu['unigram_wins']}/={evsu['ties']}]"
    )


if __name__ == "__main__":
    print("[12FF]: Extended Trace Pattern Completion Benchmark")
    print()

    results = run_extended_benchmark(verbose=True)

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(format_report(results))
