#!/usr/bin/env python3
"""
Chain-of-Thought Induction Benchmark

Compares CoT vs Direct prompting for trace-to-SC induction.
Tests across model sizes: 0.5B, 1.5B, 3B.

Goal: Improve from 20% to 60%+ generalization.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

from .cot_inducer import CoTInducer, DirectInducer, CoTInductionResult
from .inducer import compare_statecharts


# Test cases optimized for induction patterns
COT_TEST_CASES = [
    {
        "name": "toggle",
        "pattern": "2-state cycle",
        "traces": [
            ["ON", "OFF", "ON"],
            ["ON", "OFF", "ON", "OFF"],
            ["ON", "OFF"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "ON"},
                {"from": ["On"], "to": ["Off"], "event": "OFF"},
            ]
        },
    },
    {
        "name": "counter",
        "pattern": "3-state linear",
        "traces": [
            ["INC", "INC", "DEC"],
            ["INC", "INC", "INC", "DEC", "DEC"],
            ["INC", "DEC"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Zero", "type": 1, "is_initial": True},
                    {"label": "Low", "type": 1},
                    {"label": "High", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Zero"], "to": ["Low"], "event": "INC"},
                {"from": ["Low"], "to": ["High"], "event": "INC"},
                {"from": ["High"], "to": ["Low"], "event": "DEC"},
                {"from": ["Low"], "to": ["Zero"], "event": "DEC"},
            ]
        },
    },
    {
        "name": "cycle",
        "pattern": "3-state cycle",
        "traces": [
            ["A", "B", "C", "A", "B", "C"],
            ["A", "B", "C", "A"],
            ["A", "B", "C"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1},
                    {"label": "C", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "A"},
                {"from": ["B"], "to": ["C"], "event": "B"},
                {"from": ["C"], "to": ["A"], "event": "C"},
            ]
        },
    },
    {
        "name": "branch",
        "pattern": "star pattern",
        "traces": [
            ["LEFT", "CENTER"],
            ["RIGHT", "CENTER"],
            ["LEFT", "CENTER", "RIGHT", "CENTER"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Center", "type": 1, "is_initial": True},
                    {"label": "Left", "type": 1},
                    {"label": "Right", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Center"], "to": ["Left"], "event": "LEFT"},
                {"from": ["Center"], "to": ["Right"], "event": "RIGHT"},
                {"from": ["Left"], "to": ["Center"], "event": "CENTER"},
                {"from": ["Right"], "to": ["Center"], "event": "CENTER"},
            ]
        },
    },
    {
        "name": "door",
        "pattern": "3-state lock",
        "traces": [
            ["OPEN", "CLOSE", "LOCK"],
            ["OPEN", "CLOSE", "LOCK", "UNLOCK", "OPEN"],
            ["OPEN", "CLOSE"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Closed", "type": 1, "is_initial": True},
                    {"label": "Open", "type": 1},
                    {"label": "Locked", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"},
                {"from": ["Closed"], "to": ["Locked"], "event": "LOCK"},
                {"from": ["Locked"], "to": ["Closed"], "event": "UNLOCK"},
            ]
        },
    },
]


@dataclass
class CoTBenchmarkResult:
    """Result for one model/method combination."""
    model: str
    method: str  # "cot" or "direct"
    avg_states_acc: float
    avg_trans_acc: float
    valid_rate: float
    avg_time_ms: float
    details: List[Dict] = None


def run_single_benchmark(
    inducer: Any,
    method: str,
    test_cases: List[Dict],
    verbose: bool = False,
) -> CoTBenchmarkResult:
    """Run benchmark with a single inducer."""
    results = []
    total_states = 0.0
    total_trans = 0.0
    valid_count = 0
    total_time = 0.0

    for case in test_cases:
        name = case["name"]
        traces = case["traces"]
        ground_truth = case["ground_truth"]

        if verbose:
            print(f"  [{name}] Pattern: {case['pattern']}")

        # Induce
        if method == "cot":
            result = inducer.induce(traces, use_analysis_hint=True)
        else:
            result = inducer.induce(traces)

        # Compare
        if result.induced_sc:
            states_acc, trans_acc = compare_statecharts(
                result.induced_sc, ground_truth
            )
            valid_count += 1
        else:
            states_acc, trans_acc = 0.0, 0.0

        total_states += states_acc
        total_trans += trans_acc
        total_time += result.generation_time_ms

        results.append({
            "name": name,
            "valid": result.is_valid_json,
            "states_acc": states_acc,
            "trans_acc": trans_acc,
            "num_states": result.num_states,
            "time_ms": result.generation_time_ms,
        })

        if verbose:
            status = "OK" if result.is_valid_json else "FAIL"
            print(f"    {status} states={states_acc*100:.0f}% trans={trans_acc*100:.0f}%")

    n = len(test_cases)
    return CoTBenchmarkResult(
        model="",  # Set by caller
        method=method,
        avg_states_acc=total_states / n if n > 0 else 0.0,
        avg_trans_acc=total_trans / n if n > 0 else 0.0,
        valid_rate=valid_count / n if n > 0 else 0.0,
        avg_time_ms=total_time / n if n > 0 else 0.0,
        details=results,
    )


def run_cot_benchmark(
    models: List[Tuple[str, Any, Any]] = None,
    verbose: bool = True,
) -> Dict[str, List[CoTBenchmarkResult]]:
    """
    Run full CoT benchmark comparing methods and model sizes.

    Args:
        models: List of (name, model, tokenizer) tuples
        verbose: Print progress

    Returns:
        Dict mapping model_name -> [direct_result, cot_result]
    """
    print("=" * 60)
    print("CHAIN-OF-THOUGHT INDUCTION BENCHMARK")
    print("=" * 60)
    print(f"\nTest cases: {len(COT_TEST_CASES)}")
    print("Methods: direct, cot")
    print()

    results = {}

    if models is None:
        # Default: try to load standard sizes
        models = []
        model_ids = [
            ("0.5B", "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"),
            ("1.5B", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"),
            ("3B", "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"),
        ]

        try:
            from mlx_lm import load

            for name, model_id in model_ids:
                print(f"Loading {name}...")
                try:
                    model, tokenizer = load(model_id)
                    models.append((name, model, tokenizer))
                    print(f"  Loaded: {model_id}")
                except Exception as e:
                    print(f"  Skip {name}: {e}")
        except ImportError:
            print("mlx_lm not available")

    if not models:
        print("No models available!")
        return results

    # Run benchmarks
    for model_name, model, tokenizer in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print("=" * 60)

        model_results = []

        # Direct prompting
        print("\n[Direct]")
        direct_inducer = DirectInducer(model, tokenizer, verbose=False)
        direct_result = run_single_benchmark(
            direct_inducer, "direct", COT_TEST_CASES, verbose
        )
        direct_result.model = model_name
        model_results.append(direct_result)

        print(f"  Avg states: {direct_result.avg_states_acc*100:.0f}%")
        print(f"  Avg trans: {direct_result.avg_trans_acc*100:.0f}%")
        print(f"  Valid: {direct_result.valid_rate*100:.0f}%")

        # Chain-of-thought
        print("\n[CoT]")
        cot_inducer = CoTInducer(model, tokenizer, verbose=False)
        cot_result = run_single_benchmark(
            cot_inducer, "cot", COT_TEST_CASES, verbose
        )
        cot_result.model = model_name
        model_results.append(cot_result)

        print(f"  Avg states: {cot_result.avg_states_acc*100:.0f}%")
        print(f"  Avg trans: {cot_result.avg_trans_acc*100:.0f}%")
        print(f"  Valid: {cot_result.valid_rate*100:.0f}%")

        # Improvement
        states_delta = (cot_result.avg_states_acc - direct_result.avg_states_acc) * 100
        trans_delta = (cot_result.avg_trans_acc - direct_result.avg_trans_acc) * 100
        print(f"\n  Delta: states={states_delta:+.0f}% trans={trans_delta:+.0f}%")

        results[model_name] = model_results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<8} {'Direct':>8} {'CoT':>8} {'Delta':>8}")
    print("-" * 36)

    for model_name, model_results in results.items():
        direct = model_results[0].avg_states_acc * 100
        cot = model_results[1].avg_states_acc * 100
        delta = cot - direct
        print(f"{model_name:<8} {direct:>7.0f}% {cot:>7.0f}% {delta:>+7.0f}%")

    return results


def format_report(results: Dict[str, List[CoTBenchmarkResult]]) -> str:
    """Format results for orchestrator report."""
    parts = []
    for model_name, model_results in results.items():
        direct = model_results[0].avg_states_acc * 100
        cot = model_results[1].avg_states_acc * 100
        parts.append(f"direct_{model_name}={direct:.0f}%, cot_{model_name}={cot:.0f}%")
    return "TRACE_INDUCTION_COT " + ", ".join(parts)


if __name__ == "__main__":
    print("[12FF]: Chain-of-Thought Induction Benchmark")
    print()

    results = run_cot_benchmark(verbose=True)

    if results:
        print("\n" + "=" * 60)
        print("REPORT:")
        print(format_report(results))
