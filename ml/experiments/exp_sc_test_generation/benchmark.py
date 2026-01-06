#!/usr/bin/env python3
"""
SC Test Generation Benchmark

Measures coverage achieved by LLM-generated test traces.
Uses REAL inference - no mocking.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .coverage_analyzer import CoverageAnalyzer, compute_state_coverage, compute_transition_coverage
from .trace_generator import TraceGenerator, TraceGeneratorConfig, generate_test_traces


# Test statecharts of varying complexity
TEST_STATECHARTS = [
    {
        "name": "toggle",
        "description": "Simple on/off toggle",
        "statechart": {
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
        "expected_states": 2,
        "expected_transitions": 2,
    },
    {
        "name": "traffic_light",
        "description": "Traffic light with 3 states",
        "statechart": {
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
        "expected_states": 3,
        "expected_transitions": 3,
    },
    {
        "name": "media_player",
        "description": "Media player with multiple controls",
        "statechart": {
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
        "expected_states": 3,
        "expected_transitions": 5,
    },
    {
        "name": "order_processing",
        "description": "Order with multiple stages",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Pending", "type": 1, "is_initial": True},
                    {"label": "Confirmed", "type": 1},
                    {"label": "Shipped", "type": 1},
                    {"label": "Delivered", "type": 1},
                    {"label": "Cancelled", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Pending"], "to": ["Confirmed"], "event": "CONFIRM"},
                {"from": ["Pending"], "to": ["Cancelled"], "event": "CANCEL"},
                {"from": ["Confirmed"], "to": ["Shipped"], "event": "SHIP"},
                {"from": ["Confirmed"], "to": ["Cancelled"], "event": "CANCEL"},
                {"from": ["Shipped"], "to": ["Delivered"], "event": "DELIVER"},
            ]
        },
        "expected_states": 5,
        "expected_transitions": 5,
    },
    {
        "name": "authentication",
        "description": "Login flow with retry",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "LoggedOut", "type": 1, "is_initial": True},
                    {"label": "Authenticating", "type": 1},
                    {"label": "LoggedIn", "type": 1},
                    {"label": "Error", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["LoggedOut"], "to": ["Authenticating"], "event": "LOGIN"},
                {"from": ["Authenticating"], "to": ["LoggedIn"], "event": "SUCCESS"},
                {"from": ["Authenticating"], "to": ["Error"], "event": "FAILURE"},
                {"from": ["Error"], "to": ["Authenticating"], "event": "RETRY"},
                {"from": ["Error"], "to": ["LoggedOut"], "event": "CANCEL"},
                {"from": ["LoggedIn"], "to": ["LoggedOut"], "event": "LOGOUT"},
            ]
        },
        "expected_states": 4,
        "expected_transitions": 6,
    },
]


@dataclass
class BenchmarkResult:
    """Results for a single statechart test."""
    name: str
    state_coverage: float
    transition_coverage: float
    num_traces: int
    total_events: int
    generation_time_ms: float
    traces_generated: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state_coverage": self.state_coverage,
            "transition_coverage": self.transition_coverage,
            "num_traces": self.num_traces,
            "total_events": self.total_events,
            "generation_time_ms": self.generation_time_ms,
        }


class TestGenBenchmark:
    """Benchmark runner for test trace generation."""

    def __init__(self, model=None, tokenizer=None, num_traces: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.num_traces = num_traces
        self.results: List[BenchmarkResult] = []

    def run(self, statecharts: Optional[List[Dict]] = None) -> List[BenchmarkResult]:
        """Run benchmark on statecharts."""
        if statecharts is None:
            statecharts = TEST_STATECHARTS

        self.results = []

        for sc_config in statecharts:
            name = sc_config["name"]
            statechart = sc_config["statechart"]

            print(f"\nTesting: {name}")

            start = time.time()
            traces, state_cov, trans_cov = generate_test_traces(
                statechart,
                self.model,
                self.tokenizer,
                num_traces=self.num_traces,
            )
            gen_time = (time.time() - start) * 1000

            total_events = sum(len(t.events) for t in traces)

            result = BenchmarkResult(
                name=name,
                state_coverage=state_cov,
                transition_coverage=trans_cov,
                num_traces=len(traces),
                total_events=total_events,
                generation_time_ms=gen_time,
                traces_generated=[t.events for t in traces],
            )
            self.results.append(result)

            print(f"  State coverage:      {state_cov*100:.0f}%")
            print(f"  Transition coverage: {trans_cov*100:.0f}%")
            print(f"  Traces: {len(traces)}, Events: {total_events}")

        return self.results

    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 60)
        print("TEST GENERATION BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\n{'Statechart':<20} {'States%':>10} {'Trans%':>10} {'Traces':>8} {'Events':>8}")
        print("-" * 60)

        total_state_cov = 0
        total_trans_cov = 0

        for r in self.results:
            print(f"{r.name:<20} {r.state_coverage*100:>9.0f}% {r.transition_coverage*100:>9.0f}% "
                  f"{r.num_traces:>8} {r.total_events:>8}")
            total_state_cov += r.state_coverage
            total_trans_cov += r.transition_coverage

        n = len(self.results)
        avg_state = total_state_cov / n * 100
        avg_trans = total_trans_cov / n * 100

        print("-" * 60)
        print(f"{'AVERAGE':<20} {avg_state:>9.0f}% {avg_trans:>9.0f}%")

        return avg_state, avg_trans


def run_benchmark(
    model=None,
    tokenizer=None,
    num_traces: int = 3,
) -> Dict[str, Any]:
    """Run the full benchmark."""
    benchmark = TestGenBenchmark(model, tokenizer, num_traces)
    results = benchmark.run()
    avg_state, avg_trans = benchmark.print_summary()

    return {
        "results": [r.to_dict() for r in results],
        "average_state_coverage": avg_state,
        "average_transition_coverage": avg_trans,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SC Test Generation Benchmark")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--traces", type=int, default=3, help="Traces per statechart")
    parser.add_argument("--output", default="./results/test_gen.json", help="Output file")

    args = parser.parse_args()

    model = None
    tokenizer = None

    if args.model:
        try:
            from mlx_lm import load
            print(f"Loading model: {args.model}")
            model, tokenizer = load(args.model)
        except Exception as e:
            print(f"Failed to load model: {e}")

    print(f"\n[DDB5] Running SC Test Generation Benchmark")
    print(f"Model: {args.model or 'programmatic (no LLM)'}")
    print(f"Traces per SC: {args.traces}")

    summary = run_benchmark(model, tokenizer, args.traces)

    # Save results
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
