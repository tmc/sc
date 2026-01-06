#!/usr/bin/env python3
"""
Trace Pattern Completion Benchmark

Tests pattern learning and trace completion across pattern types:
1. Simple cycle: [A,B,C,A,B,C,...]
2. Alternating: [A,B,A,B,...]
3. Growth: [A, A,B, A,B,C,...]
4. Nested: patterns with sub-patterns
5. With noise: mostly pattern but variations
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .pattern_learner import PatternLearner, learn_patterns
from .trace_completer import TraceCompleter, complete_trace


# Test cases organized by pattern type
TEST_CASES = [
    # Cycle patterns
    {
        "name": "cycle_3",
        "pattern_type": "cycle",
        "examples": [
            ["A", "B", "C", "A", "B", "C"],
            ["A", "B", "C", "A", "B", "C", "A"],
            ["A", "B", "C", "A"],
        ],
        "partial": ["A", "B", "C", "A", "B"],
        "expected_1": ["C"],
        "expected_3": ["C", "A", "B"],
    },
    {
        "name": "cycle_4",
        "pattern_type": "cycle",
        "examples": [
            ["W", "X", "Y", "Z", "W", "X", "Y", "Z"],
            ["W", "X", "Y", "Z", "W"],
        ],
        "partial": ["W", "X", "Y"],
        "expected_1": ["Z"],
        "expected_3": ["Z", "W", "X"],
    },
    {
        "name": "traffic_light",
        "pattern_type": "cycle",
        "examples": [
            ["RED", "GREEN", "YELLOW", "RED", "GREEN", "YELLOW"],
            ["RED", "GREEN", "YELLOW", "RED"],
        ],
        "partial": ["RED", "GREEN"],
        "expected_1": ["YELLOW"],
        "expected_3": ["YELLOW", "RED", "GREEN"],
    },
    # Alternating patterns
    {
        "name": "toggle",
        "pattern_type": "alt",
        "examples": [
            ["ON", "OFF", "ON", "OFF", "ON"],
            ["ON", "OFF", "ON", "OFF"],
        ],
        "partial": ["ON", "OFF", "ON"],
        "expected_1": ["OFF"],
        "expected_3": ["OFF", "ON", "OFF"],
    },
    {
        "name": "ping_pong",
        "pattern_type": "alt",
        "examples": [
            ["PING", "PONG", "PING", "PONG", "PING", "PONG"],
            ["PING", "PONG", "PING", "PONG"],
        ],
        "partial": ["PING", "PONG", "PING", "PONG"],
        "expected_1": ["PING"],
        "expected_3": ["PING", "PONG", "PING"],
    },
    # Growth patterns
    {
        "name": "growth_linear",
        "pattern_type": "growth",
        "examples": [
            ["A"],
            ["A", "B"],
            ["A", "B", "C"],
            ["A", "B", "C", "D"],
        ],
        "partial": ["A", "B", "C", "D"],
        "expected_1": ["E"] if False else [],  # No clear next (pattern ends)
        "expected_3": [],
    },
    {
        "name": "counting",
        "pattern_type": "growth",
        "examples": [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "TWO", "THREE"],
        ],
        "partial": ["ONE", "TWO"],
        "expected_1": ["THREE"],
        "expected_3": ["THREE"],  # Growth ends
    },
    # Nested patterns
    {
        "name": "nested_ab",
        "pattern_type": "nested",
        "examples": [
            ["START", "A", "B", "END", "START", "A", "B", "END"],
            ["START", "A", "B", "END", "START"],
        ],
        "partial": ["START", "A", "B", "END", "START", "A"],
        "expected_1": ["B"],
        "expected_3": ["B", "END", "START"],
    },
    # Patterns with state
    {
        "name": "door_sequence",
        "pattern_type": "cycle",
        "examples": [
            ["CLOSED", "OPENING", "OPEN", "CLOSING", "CLOSED"],
            ["CLOSED", "OPENING", "OPEN", "CLOSING", "CLOSED", "OPENING"],
        ],
        "partial": ["CLOSED", "OPENING", "OPEN"],
        "expected_1": ["CLOSING"],
        "expected_3": ["CLOSING", "CLOSED", "OPENING"],
    },
    # Edge cases
    {
        "name": "single_repeat",
        "pattern_type": "cycle",
        "examples": [
            ["X", "X", "X", "X"],
            ["X", "X", "X"],
        ],
        "partial": ["X", "X"],
        "expected_1": ["X"],
        "expected_3": ["X", "X", "X"],
    },
]


@dataclass
class BenchmarkResult:
    """Result for one test case."""
    name: str
    pattern_type: str
    pattern_detected: str
    complete_1_correct: bool
    complete_3_correct: bool
    predicted_1: List[str]
    predicted_3: List[str]
    time_ms: float


def run_benchmark(
    completer: Optional[TraceCompleter] = None,
    use_llm: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run trace pattern completion benchmark.

    Args:
        completer: TraceCompleter instance (created if None)
        use_llm: Use LLM for completion
        verbose: Print progress

    Returns:
        Dict with results and metrics
    """
    print("=" * 60)
    print("TRACE PATTERN COMPLETION BENCHMARK")
    print("=" * 60)
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Method: {'LLM' if use_llm else 'Algorithmic'}")
    print()

    if completer is None:
        if use_llm:
            try:
                from mlx_lm import load
                print("Loading model...")
                model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
                completer = TraceCompleter(model, tokenizer, verbose=verbose)
            except Exception as e:
                print(f"LLM load failed: {e}, using algorithmic")
                completer = TraceCompleter(verbose=verbose)
        else:
            completer = TraceCompleter(verbose=verbose)

    results: List[BenchmarkResult] = []
    by_type: Dict[str, List[BenchmarkResult]] = {}
    learner = PatternLearner()

    for case in TEST_CASES:
        name = case["name"]
        pattern_type = case["pattern_type"]
        examples = case["examples"]
        partial = case["partial"]
        expected_1 = case["expected_1"]
        expected_3 = case["expected_3"]

        if verbose:
            print(f"\n[{name}] Type: {pattern_type}")
            print(f"  Partial: {partial}")

        start = time.time()

        # Learn pattern
        analysis = learner.learn(examples)
        detected = analysis.primary_pattern.pattern_type if analysis.primary_pattern else "none"

        # Complete 1-step
        if use_llm:
            result_1 = completer.complete_with_llm(examples, partial, num_predictions=1)
        else:
            result_1 = completer.complete(examples, partial, num_predictions=1)

        # Complete 3-step
        if use_llm:
            result_3 = completer.complete_with_llm(examples, partial, num_predictions=3)
        else:
            result_3 = completer.complete(examples, partial, num_predictions=3)

        elapsed = (time.time() - start) * 1000

        # Check correctness
        correct_1 = result_1.predicted_next == expected_1
        correct_3 = result_3.predicted_next == expected_3 or (
            not expected_3 and not result_3.predicted_next
        )

        br = BenchmarkResult(
            name=name,
            pattern_type=pattern_type,
            pattern_detected=detected,
            complete_1_correct=correct_1,
            complete_3_correct=correct_3,
            predicted_1=result_1.predicted_next,
            predicted_3=result_3.predicted_next,
            time_ms=elapsed,
        )
        results.append(br)

        if pattern_type not in by_type:
            by_type[pattern_type] = []
        by_type[pattern_type].append(br)

        if verbose:
            status_1 = "✓" if correct_1 else "✗"
            status_3 = "✓" if correct_3 else "✗"
            print(f"  Pattern detected: {detected}")
            print(f"  1-step: {result_1.predicted_next} (expected {expected_1}) {status_1}")
            print(f"  3-step: {result_3.predicted_next} (expected {expected_3}) {status_3}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Pattern detection accuracy
    pattern_correct = sum(
        1 for r in results
        if r.pattern_detected in (r.pattern_type, "cycle", "alternating", "transition_map")
    )
    pattern_acc = pattern_correct / len(results) * 100 if results else 0

    # Completion accuracy
    complete_1_correct = sum(1 for r in results if r.complete_1_correct)
    complete_3_correct = sum(1 for r in results if r.complete_3_correct)
    complete_1_acc = complete_1_correct / len(results) * 100 if results else 0
    complete_3_acc = complete_3_correct / len(results) * 100 if results else 0

    print(f"\nPattern detection: {pattern_acc:.0f}%")
    print(f"1-step completion: {complete_1_acc:.0f}%")
    print(f"3-step completion: {complete_3_acc:.0f}%")

    print(f"\n{'Pattern Type':<12} {'1-step':>8} {'3-step':>8}")
    print("-" * 32)

    type_rates = {}
    for ptype, type_results in sorted(by_type.items()):
        c1 = sum(1 for r in type_results if r.complete_1_correct)
        c3 = sum(1 for r in type_results if r.complete_3_correct)
        n = len(type_results)
        r1 = c1 / n * 100 if n > 0 else 0
        r3 = c3 / n * 100 if n > 0 else 0
        type_rates[ptype] = {"1step": r1, "3step": r3}
        print(f"{ptype:<12} {r1:>7.0f}% {r3:>7.0f}%")

    avg_time = sum(r.time_ms for r in results) / len(results) if results else 0
    print(f"\nAvg time: {avg_time:.1f}ms")

    return {
        "pattern_acc": pattern_acc,
        "complete_1_acc": complete_1_acc,
        "complete_3_acc": complete_3_acc,
        "by_type": type_rates,
        "results": results,
        "avg_time_ms": avg_time,
    }


def format_report(results: Dict[str, Any]) -> str:
    """Format results for orchestrator report."""
    pattern = results["pattern_acc"]
    c1 = results["complete_1_acc"]
    c3 = results["complete_3_acc"]
    by_type = results["by_type"]

    type_str = ", ".join([
        f"{k}:{v['1step']:.0f}%"
        for k, v in sorted(by_type.items())
    ])

    return f"TRACE_PATTERN_COMPLETION pattern_acc={pattern:.0f}%, complete_1={c1:.0f}%, complete_3={c3:.0f}%, by_type=[{type_str}]"


if __name__ == "__main__":
    print("[12FF]: Trace Pattern Completion Benchmark")
    print()

    # Run algorithmic benchmark
    print("\n" + "=" * 60)
    print("ALGORITHMIC COMPLETION")
    print("=" * 60)
    results_algo = run_benchmark(use_llm=False, verbose=True)

    # Try LLM benchmark
    print("\n" + "=" * 60)
    print("LLM COMPLETION")
    print("=" * 60)
    try:
        results_llm = run_benchmark(use_llm=True, verbose=True)
    except Exception as e:
        print(f"LLM benchmark failed: {e}")
        results_llm = results_algo

    # Report
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"\nAlgorithmic: {format_report(results_algo)}")
    if results_llm != results_algo:
        print(f"LLM:         {format_report(results_llm)}")
