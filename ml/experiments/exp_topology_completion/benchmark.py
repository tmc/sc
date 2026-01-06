#!/usr/bin/env python3
"""
Topology Completion Benchmark

Tests completion of various incomplete statechart patterns:
1. Missing transition - states disconnected
2. Missing terminal - no final state
3. Missing initial - no start state
4. Orphan states - no connections
5. Dead ends - can enter but not exit
6. Complex - multiple issues
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .topology_completer import TopologyCompleter, TopologyAnalyzer


# Test cases for incomplete SCs
TEST_CASES = [
    {
        "name": "missing_transition",
        "description": "State C unreachable from A→B",
        "partial_sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1},
                    {"label": "C", "type": 1, "is_terminal": True},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "NEXT"}
            ]
        },
        "issue_type": "missing_trans",
    },
    {
        "name": "no_terminal",
        "description": "Linear A→B→C but no terminal",
        "partial_sc": {
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
                {"from": ["A"], "to": ["B"], "event": "STEP1"},
                {"from": ["B"], "to": ["C"], "event": "STEP2"},
            ]
        },
        "issue_type": "no_terminal",
    },
    {
        "name": "no_initial",
        "description": "States exist but no initial marked",
        "partial_sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1},
                    {"label": "Running", "type": 1},
                    {"label": "Done", "type": 1, "is_terminal": True},
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Running"], "event": "START"},
                {"from": ["Running"], "to": ["Done"], "event": "FINISH"},
            ]
        },
        "issue_type": "no_initial",
    },
    {
        "name": "orphan_state",
        "description": "State D has no connections",
        "partial_sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Start", "type": 1, "is_initial": True},
                    {"label": "Middle", "type": 1},
                    {"label": "End", "type": 1, "is_terminal": True},
                    {"label": "Orphan", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Start"], "to": ["Middle"], "event": "GO"},
                {"from": ["Middle"], "to": ["End"], "event": "FINISH"},
            ]
        },
        "issue_type": "orphan",
    },
    {
        "name": "dead_end",
        "description": "State B is a dead end (not terminal)",
        "partial_sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Entry", "type": 1, "is_initial": True},
                    {"label": "Trap", "type": 1},
                    {"label": "Exit", "type": 1, "is_terminal": True},
                ]
            },
            "transitions": [
                {"from": ["Entry"], "to": ["Trap"], "event": "FALL"},
                {"from": ["Entry"], "to": ["Exit"], "event": "ESCAPE"},
            ]
        },
        "issue_type": "dead_end",
    },
    {
        "name": "complex_multiple",
        "description": "Multiple issues: no initial, orphan, dead end",
        "partial_sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "S1", "type": 1},
                    {"label": "S2", "type": 1},
                    {"label": "S3", "type": 1},
                    {"label": "Isolated", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["S1"], "to": ["S2"], "event": "A"},
            ]
        },
        "issue_type": "complex",
    },
    {
        "name": "valid_simple",
        "description": "Already valid - simple toggle",
        "partial_sc": {
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
        "issue_type": "valid",
    },
    {
        "name": "cycle_no_terminal",
        "description": "Valid cycle but might want terminal",
        "partial_sc": {
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
                {"from": ["Red"], "to": ["Green"], "event": "GO"},
                {"from": ["Green"], "to": ["Yellow"], "event": "SLOW"},
                {"from": ["Yellow"], "to": ["Red"], "event": "STOP"},
            ]
        },
        "issue_type": "valid",  # Cycles are valid
    },
]


@dataclass
class BenchmarkResult:
    """Result for one test case."""
    name: str
    issue_type: str
    original_valid: bool
    completed_valid: bool
    num_gaps: int
    num_fixes: int
    time_ms: float


def run_benchmark(
    completer: Optional[TopologyCompleter] = None,
    use_llm: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run topology completion benchmark.

    Args:
        completer: TopologyCompleter instance (created if None)
        use_llm: Use LLM for completion (vs algorithmic)
        verbose: Print progress

    Returns:
        Dict with results by issue type
    """
    print("=" * 60)
    print("TOPOLOGY COMPLETION BENCHMARK")
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
                completer = TopologyCompleter(model, tokenizer, verbose=verbose)
            except Exception as e:
                print(f"LLM load failed: {e}, using algorithmic")
                completer = TopologyCompleter(verbose=verbose)
        else:
            completer = TopologyCompleter(verbose=verbose)

    results: List[BenchmarkResult] = []
    by_issue: Dict[str, List[BenchmarkResult]] = {}

    for case in TEST_CASES:
        name = case["name"]
        issue_type = case["issue_type"]
        partial_sc = case["partial_sc"]

        if verbose:
            print(f"\n[{name}] {case['description']}")

        # Analyze original
        analyzer = TopologyAnalyzer()
        original_analysis = analyzer.analyze(partial_sc)
        original_valid = original_analysis.is_valid

        # Complete
        start = time.time()
        if use_llm:
            result = completer.complete(partial_sc)
        else:
            result = completer.complete_algorithmic(partial_sc)
        elapsed = (time.time() - start) * 1000

        br = BenchmarkResult(
            name=name,
            issue_type=issue_type,
            original_valid=original_valid,
            completed_valid=result.is_valid,
            num_gaps=len(original_analysis.gaps),
            num_fixes=len(original_analysis.fixes),
            time_ms=elapsed,
        )
        results.append(br)

        if issue_type not in by_issue:
            by_issue[issue_type] = []
        by_issue[issue_type].append(br)

        status = "VALID" if result.is_valid else "INVALID"
        if verbose:
            print(f"  Original: {'valid' if original_valid else 'invalid'} ({br.num_gaps} gaps)")
            print(f"  Completed: {status} in {elapsed:.1f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    valid_count = sum(1 for r in results if r.completed_valid)
    valid_rate = valid_count / total * 100 if total > 0 else 0

    print(f"\nOverall: {valid_count}/{total} valid ({valid_rate:.0f}%)")

    print(f"\n{'Issue Type':<15} {'Valid':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 45)

    issue_rates = {}
    for issue_type, issue_results in sorted(by_issue.items()):
        valid = sum(1 for r in issue_results if r.completed_valid)
        total_i = len(issue_results)
        rate = valid / total_i * 100 if total_i > 0 else 0
        issue_rates[issue_type] = rate
        print(f"{issue_type:<15} {valid:>8} {total_i:>8} {rate:>7.0f}%")

    avg_time = sum(r.time_ms for r in results) / len(results) if results else 0
    print(f"\nAvg time: {avg_time:.1f}ms")

    return {
        "valid_rate": valid_rate,
        "by_issue": issue_rates,
        "results": results,
        "avg_time_ms": avg_time,
    }


def format_report(results: Dict[str, Any]) -> str:
    """Format results for orchestrator report."""
    valid = results["valid_rate"]
    by_issue = results["by_issue"]

    issue_str = ", ".join([
        f"{k}:{v:.0f}%" for k, v in sorted(by_issue.items())
    ])

    return f"TOPOLOGY_COMPLETION valid={valid:.0f}%, by_issue=[{issue_str}]"


if __name__ == "__main__":
    print("[12FF]: Topology Completion Benchmark")
    print()

    # Run algorithmic benchmark first (fast)
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
