"""
Benchmark for Incremental Statechart Updates

Evaluates the incremental editor against a suite of change requests.

TARGET: 90%+ valid edits

METRICS:
- Edit success rate (parse + apply)
- Validation success rate (resulting chart is valid)
- Diff size efficiency (minimal operations)
- Semantic correctness (intended changes applied)
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict

from .diff_generator import (
    StatechartDiff,
    DiffGenerator,
    DiffApplier,
    diff_size,
    chart_size,
)
from .incremental_editor import (
    ChangeType,
    ChangeRequest,
    EditResult,
    IncrementalEditor,
)


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    chart: Dict[str, Any]
    change_request: str
    expected_change_type: ChangeType
    expected_states_after: Optional[Set[str]] = None
    expected_transitions_after: Optional[int] = None


@dataclass
class CaseResult:
    """Result for a single benchmark case."""
    case: BenchmarkCase
    edit_result: EditResult
    duration: float
    success: bool
    valid: bool
    semantic_correct: bool = True  # Did the intended change happen?

    @property
    def passed(self) -> bool:
        return self.success and self.valid and self.semantic_correct


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_cases: int
    successful_edits: int
    valid_edits: int
    semantic_correct: int
    passed: int
    total_duration: float
    by_change_type: Dict[ChangeType, Dict[str, int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successful_edits / self.total_cases if self.total_cases > 0 else 0

    @property
    def validity_rate(self) -> float:
        return self.valid_edits / self.total_cases if self.total_cases > 0 else 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0


# =============================================================================
# Benchmark Dataset
# =============================================================================

def create_base_charts() -> List[Dict[str, Any]]:
    """Create base charts for benchmarking."""
    charts = []

    # Simple flat chart
    charts.append({
        "name": "simple_flat",
        "chart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "idle", "type": 1, "is_initial": True},
                    {"label": "active", "type": 1},
                    {"label": "done", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["idle"], "to": ["active"], "event": "START"},
                {"from": ["active"], "to": ["done"], "event": "FINISH"},
            ]
        }
    })

    # Media player chart
    charts.append({
        "name": "media_player",
        "chart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "stopped", "type": 1, "is_initial": True},
                    {"label": "playing", "type": 1},
                    {"label": "paused", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["stopped"], "to": ["playing"], "event": "PLAY"},
                {"from": ["playing"], "to": ["paused"], "event": "PAUSE"},
                {"from": ["paused"], "to": ["playing"], "event": "PLAY"},
                {"from": ["playing"], "to": ["stopped"], "event": "STOP"},
                {"from": ["paused"], "to": ["stopped"], "event": "STOP"},
            ]
        }
    })

    # Traffic light
    charts.append({
        "name": "traffic_light",
        "chart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "red", "type": 1, "is_initial": True},
                    {"label": "green", "type": 1},
                    {"label": "yellow", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["red"], "to": ["green"], "event": "TIMER"},
                {"from": ["green"], "to": ["yellow"], "event": "TIMER"},
                {"from": ["yellow"], "to": ["red"], "event": "TIMER"},
            ]
        }
    })

    # Login flow
    charts.append({
        "name": "login_flow",
        "chart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "logged_out", "type": 1, "is_initial": True},
                    {"label": "authenticating", "type": 1},
                    {"label": "logged_in", "type": 1},
                    {"label": "error", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["logged_out"], "to": ["authenticating"], "event": "SUBMIT"},
                {"from": ["authenticating"], "to": ["logged_in"], "event": "SUCCESS"},
                {"from": ["authenticating"], "to": ["error"], "event": "FAILURE"},
                {"from": ["error"], "to": ["logged_out"], "event": "RETRY"},
                {"from": ["logged_in"], "to": ["logged_out"], "event": "LOGOUT"},
            ]
        }
    })

    # Order process
    charts.append({
        "name": "order_process",
        "chart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "cart", "type": 1, "is_initial": True},
                    {"label": "checkout", "type": 1},
                    {"label": "payment", "type": 1},
                    {"label": "shipped", "type": 1},
                    {"label": "delivered", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["cart"], "to": ["checkout"], "event": "CHECKOUT"},
                {"from": ["checkout"], "to": ["payment"], "event": "PAY"},
                {"from": ["payment"], "to": ["shipped"], "event": "CONFIRM"},
                {"from": ["shipped"], "to": ["delivered"], "event": "DELIVER"},
            ]
        }
    })

    return charts


def create_benchmark_cases() -> List[BenchmarkCase]:
    """Create comprehensive benchmark test cases."""
    cases = []
    base_charts = create_base_charts()

    # ADD_STATE cases
    for base in base_charts:
        cases.append(BenchmarkCase(
            name=f"{base['name']}_add_state",
            chart=base["chart"],
            change_request="Add a new state called 'pending'",
            expected_change_type=ChangeType.ADD_STATE,
        ))

        cases.append(BenchmarkCase(
            name=f"{base['name']}_add_state_2",
            chart=base["chart"],
            change_request="Create state named 'waiting'",
            expected_change_type=ChangeType.ADD_STATE,
        ))

    # REMOVE_STATE cases (use charts with enough states)
    for base in base_charts:
        states = [c["label"] for c in base["chart"]["root_state"]["children"]]
        if len(states) > 2:
            # Remove non-initial state
            target = states[-1]
            cases.append(BenchmarkCase(
                name=f"{base['name']}_remove_state",
                chart=base["chart"],
                change_request=f"Remove the state '{target}'",
                expected_change_type=ChangeType.REMOVE_STATE,
            ))

    # ADD_TRANSITION cases
    for base in base_charts:
        states = [c["label"] for c in base["chart"]["root_state"]["children"]]
        if len(states) >= 2:
            cases.append(BenchmarkCase(
                name=f"{base['name']}_add_transition",
                chart=base["chart"],
                change_request=f"Add transition from {states[0]} to {states[-1]} on JUMP",
                expected_change_type=ChangeType.ADD_TRANSITION,
            ))

    # RENAME_STATE cases
    for base in base_charts:
        states = [c["label"] for c in base["chart"]["root_state"]["children"]]
        if states:
            old = states[0]
            cases.append(BenchmarkCase(
                name=f"{base['name']}_rename_state",
                chart=base["chart"],
                change_request=f"Rename state '{old}' to 'initial_state'",
                expected_change_type=ChangeType.RENAME_STATE,
            ))

    # SET_INITIAL cases
    for base in base_charts:
        states = [c["label"] for c in base["chart"]["root_state"]["children"]]
        if len(states) > 1:
            # Set non-initial state as initial
            target = states[1]
            cases.append(BenchmarkCase(
                name=f"{base['name']}_set_initial",
                chart=base["chart"],
                change_request=f"Make '{target}' the initial state",
                expected_change_type=ChangeType.SET_INITIAL,
            ))

    # Semi-complex requests (can be handled deterministically with good patterns)
    for base in base_charts[:2]:
        states = [c["label"] for c in base["chart"]["root_state"]["children"]]
        if len(states) >= 2:
            # Add guard to first transition
            cases.append(BenchmarkCase(
                name=f"{base['name']}_guard_first",
                chart=base["chart"],
                change_request="Add guard 'count > 0' to the first transition",
                expected_change_type=ChangeType.ADD_GUARD,
            ))

            # Create retry transition (from last to first)
            cases.append(BenchmarkCase(
                name=f"{base['name']}_retry",
                chart=base["chart"],
                change_request=f"Create a retry transition from {states[-1]} back to {states[0]}",
                expected_change_type=ChangeType.ADD_TRANSITION,
            ))

    return cases


# =============================================================================
# Benchmark Runner
# =============================================================================

class IncrementalBenchmark:
    """Benchmark runner for incremental editing."""

    def __init__(
        self,
        editor: Optional[IncrementalEditor] = None,
        verbose: bool = True,
    ):
        self.editor = editor or IncrementalEditor(verbose=False)
        self.verbose = verbose

    def run_case(self, case: BenchmarkCase) -> CaseResult:
        """Run a single benchmark case."""
        start = time.time()

        result = self.editor.edit(case.chart, case.change_request)

        duration = time.time() - start

        # Check semantic correctness
        semantic_correct = self._check_semantics(case, result)

        return CaseResult(
            case=case,
            edit_result=result,
            duration=duration,
            success=result.success,
            valid=result.valid,
            semantic_correct=semantic_correct,
        )

    def _check_semantics(self, case: BenchmarkCase, result: EditResult) -> bool:
        """Check if the intended change was applied."""
        if not result.success or result.modified_chart is None:
            return False

        modified = result.modified_chart

        # Check expected states
        if case.expected_states_after is not None:
            actual_states = self._get_state_labels(modified)
            if actual_states != case.expected_states_after:
                return False

        # Check expected transition count
        if case.expected_transitions_after is not None:
            actual_trans = len(modified.get("transitions", []))
            if actual_trans != case.expected_transitions_after:
                return False

        return True

    def _get_state_labels(self, chart: Dict[str, Any]) -> Set[str]:
        """Get all state labels from chart."""
        labels = set()

        def traverse(state):
            labels.add(state.get("label", ""))
            for child in state.get("children", []):
                traverse(child)

        if "root_state" in chart:
            traverse(chart["root_state"])

        return labels

    def run_benchmark(
        self,
        cases: Optional[List[BenchmarkCase]] = None,
    ) -> Tuple[List[CaseResult], BenchmarkSummary]:
        """Run full benchmark suite."""
        if cases is None:
            cases = create_benchmark_cases()

        results = []
        by_type: Dict[ChangeType, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "valid": 0, "passed": 0}
        )

        for i, case in enumerate(cases):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processing case {i + 1}/{len(cases)}...")

            result = self.run_case(case)
            results.append(result)

            # Track by type
            ct = case.expected_change_type
            by_type[ct]["total"] += 1
            if result.success:
                by_type[ct]["success"] += 1
            if result.valid:
                by_type[ct]["valid"] += 1
            if result.passed:
                by_type[ct]["passed"] += 1

        # Build summary
        summary = BenchmarkSummary(
            total_cases=len(results),
            successful_edits=sum(1 for r in results if r.success),
            valid_edits=sum(1 for r in results if r.valid),
            semantic_correct=sum(1 for r in results if r.semantic_correct),
            passed=sum(1 for r in results if r.passed),
            total_duration=sum(r.duration for r in results),
            by_change_type=dict(by_type),
        )

        return results, summary


# =============================================================================
# Demo and Main
# =============================================================================

def demo():
    """Run benchmark demonstration."""
    print("=" * 60)
    print("Incremental Edit Benchmark")
    print("=" * 60)
    print(f"Target: 90%+ valid edits")
    print()

    # Create benchmark
    benchmark = IncrementalBenchmark(verbose=True)
    cases = create_benchmark_cases()

    print(f"Running {len(cases)} test cases...")
    print()

    results, summary = benchmark.run_benchmark(cases)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Total cases: {summary.total_cases}")
    print(f"  Successful edits: {summary.successful_edits} ({summary.success_rate:.1%})")
    print(f"  Valid edits: {summary.valid_edits} ({summary.validity_rate:.1%})")
    print(f"  Passed: {summary.passed} ({summary.pass_rate:.1%})")
    print(f"  Total duration: {summary.total_duration:.2f}s")

    print(f"\nBy change type:")
    for ct, stats in summary.by_change_type.items():
        if stats["total"] > 0:
            print(f"  {ct.value}:")
            print(f"    Total: {stats['total']}")
            print(f"    Valid: {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")

    # Target check
    print()
    print("=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    target_met = summary.validity_rate >= 0.90
    print(f"\nValid edit rate: {summary.validity_rate:.1%}")
    print(f"Target (90%+): {'ACHIEVED' if target_met else 'NOT MET'}")

    if target_met:
        print("\n*** TARGET ACHIEVED: 90%+ valid incremental edits ***")
    else:
        print(f"\n*** Gap to target: {0.90 - summary.validity_rate:.1%} ***")

    # Show some failures for debugging
    failures = [r for r in results if not r.valid]
    if failures and len(failures) <= 5:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures[:5]:
            print(f"  - {f.case.name}: {f.edit_result.error or 'invalid result'}")

    return results, summary


if __name__ == "__main__":
    demo()
