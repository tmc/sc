#!/usr/bin/env python3
"""
Trace-to-SC Induction Benchmark

Measures accuracy of inducing statecharts from observed traces.
Uses REAL inference.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any

from .inducer import SCInducer, InductionResult, compare_statecharts


# Test cases: traces and ground truth statecharts
TEST_CASES = [
    {
        "name": "toggle",
        "traces": [
            ["TURN_ON", "TURN_OFF", "TURN_ON"],
            ["TURN_ON", "TURN_OFF", "TURN_ON", "TURN_OFF"],
            ["TURN_ON"],
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
                {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
                {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"},
            ]
        },
    },
    {
        "name": "traffic_light",
        "traces": [
            ["NEXT", "NEXT", "NEXT"],
            ["NEXT", "NEXT", "NEXT", "NEXT", "NEXT", "NEXT"],
            ["NEXT", "NEXT"],
        ],
        "hints": {"num_states": 3},
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Yellow", "type": 1},
                    {"label": "Green", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"},
            ]
        },
    },
    {
        "name": "player",
        "traces": [
            ["PLAY", "PAUSE", "PLAY", "STOP"],
            ["PLAY", "STOP"],
            ["PLAY", "PAUSE", "STOP"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Stopped", "type": 1, "is_initial": True},
                    {"label": "Playing", "type": 1},
                    {"label": "Paused", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Paused"], "to": ["Stopped"], "event": "STOP"},
            ]
        },
    },
    {
        "name": "door",
        "traces": [
            ["OPEN", "CLOSE", "OPEN", "CLOSE"],
            ["OPEN", "CLOSE"],
            ["OPEN"],
        ],
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Closed", "type": 1, "is_initial": True},
                    {"label": "Open", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"},
            ]
        },
    },
    {
        "name": "counter",
        "traces": [
            ["INCREMENT", "INCREMENT", "RESET"],
            ["INCREMENT", "INCREMENT", "INCREMENT", "RESET"],
            ["RESET"],
        ],
        "hints": {"num_states": 2},
        "ground_truth": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Counting", "type": 1, "is_initial": True},
                    {"label": "Zero", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Zero"], "to": ["Counting"], "event": "INCREMENT"},
                {"from": ["Counting"], "to": ["Counting"], "event": "INCREMENT"},
                {"from": ["Counting"], "to": ["Zero"], "event": "RESET"},
            ]
        },
    },
]


@dataclass
class BenchmarkResult:
    """Result for a single test case."""
    name: str
    is_valid: bool
    states_accuracy: float
    transitions_accuracy: float
    num_induced_states: int
    num_truth_states: int
    generation_time_ms: float


class InductionBenchmark:
    """Benchmark for trace-to-SC induction."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.results: List[BenchmarkResult] = []

    def run(self, test_cases=None) -> List[BenchmarkResult]:
        """Run benchmark on test cases."""
        if test_cases is None:
            test_cases = TEST_CASES

        self.results = []
        inducer = SCInducer(self.model, self.tokenizer)

        for case in test_cases:
            name = case["name"]
            traces = case["traces"]
            ground_truth = case["ground_truth"]
            hints = case.get("hints")

            print(f"\nInducing: {name}")
            print(f"  Traces: {len(traces)}, Events: {set(e for t in traces for e in t)}")

            # Induce
            result = inducer.induce(traces, hints)

            # Compare to ground truth
            if result.induced_sc:
                states_acc, trans_acc = compare_statecharts(
                    result.induced_sc, ground_truth
                )
            else:
                states_acc, trans_acc = 0.0, 0.0

            # Count ground truth states
            def count_gt_states(sc):
                count = 0
                def recurse(s):
                    nonlocal count
                    if s.get("label") and s.get("label") != "__root__":
                        count += 1
                    for c in s.get("children", []):
                        recurse(c)
                if "root_state" in sc:
                    recurse(sc["root_state"])
                return count

            gt_states = count_gt_states(ground_truth)

            bench_result = BenchmarkResult(
                name=name,
                is_valid=result.is_valid_json,
                states_accuracy=states_acc,
                transitions_accuracy=trans_acc,
                num_induced_states=result.num_states,
                num_truth_states=gt_states,
                generation_time_ms=result.generation_time_ms,
            )
            self.results.append(bench_result)

            status = "✓" if result.is_valid_json else "✗"
            print(f"  {status} States: {result.num_states}/{gt_states} (acc={states_acc*100:.0f}%)")
            print(f"    Transitions acc: {trans_acc*100:.0f}%")

        return self.results

    def print_summary(self):
        """Print summary."""
        if not self.results:
            return 0, 0

        print("\n" + "=" * 60)
        print("TRACE-TO-SC INDUCTION RESULTS")
        print("=" * 60)

        print(f"\n{'Case':<15} {'Valid':>6} {'States%':>10} {'Trans%':>10}")
        print("-" * 45)

        total_states = 0
        total_trans = 0
        valid_count = 0

        for r in self.results:
            status = "✓" if r.is_valid else "✗"
            print(f"{r.name:<15} {status:>6} {r.states_accuracy*100:>9.0f}% {r.transitions_accuracy*100:>9.0f}%")
            total_states += r.states_accuracy
            total_trans += r.transitions_accuracy
            if r.is_valid:
                valid_count += 1

        n = len(self.results)
        avg_states = total_states / n * 100
        avg_trans = total_trans / n * 100

        print("-" * 45)
        print(f"{'AVERAGE':<15} {valid_count}/{n}   {avg_states:>9.0f}% {avg_trans:>9.0f}%")

        return avg_states, avg_trans


def run_benchmark(model=None, tokenizer=None) -> Dict[str, Any]:
    """Run full benchmark."""
    benchmark = InductionBenchmark(model, tokenizer)
    results = benchmark.run()
    avg_states, avg_trans = benchmark.print_summary()

    return {
        "average_states_accuracy": avg_states,
        "average_transitions_accuracy": avg_trans,
        "results": [
            {
                "name": r.name,
                "valid": r.is_valid,
                "states_acc": r.states_accuracy,
                "trans_acc": r.transitions_accuracy,
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    print("[DDB5]: Trace-to-SC Induction Benchmark")

    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-0.5B...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
    except:
        model, tokenizer = None, None
        print("No model available, using mock")

    summary = run_benchmark(model, tokenizer)
    print(f"\n[DDB5] States accuracy: {summary['average_states_accuracy']:.0f}%")
    print(f"[DDB5] Transitions accuracy: {summary['average_transitions_accuracy']:.0f}%")
