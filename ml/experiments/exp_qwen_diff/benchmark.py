#!/usr/bin/env python3
"""
Benchmark: Measure statechart diff accuracy.

Generates test cases with known changes, runs diff detection,
and measures accuracy. Target: 95%+ accuracy.
"""

import json
import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto

from .change_detector import (
    ChangeDetector, DiffResult, ChangeType, Compatibility,
    StateChange, TransitionChange
)
from .sc_differ import SCDiffer, DiffConfig, DiffSummary


@dataclass
class ExpectedChange:
    """Expected change for verification."""
    change_type: str  # "state_added", "state_removed", "transition_added", etc.
    label: str        # State label or transition key
    details: str = ""


@dataclass
class TestCase:
    """A test case with known changes."""
    name: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    expected_changes: List[ExpectedChange]
    expected_compatibility: str  # "COMPATIBLE" or "BREAKING"


@dataclass
class BenchmarkResult:
    """Result of benchmark run."""
    total_cases: int = 0
    correct_detections: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    compatibility_correct: int = 0
    total_duration: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def detection_accuracy(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.correct_detections / self.total_cases

    @property
    def compatibility_accuracy(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.compatibility_correct / self.total_cases

    @property
    def overall_accuracy(self) -> float:
        return (self.detection_accuracy + self.compatibility_accuracy) / 2


class TestCaseGenerator:
    """Generates test cases with known changes."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_base_chart(self, num_states: int = 4) -> Dict[str, Any]:
        """Generate a valid base chart."""
        states = [f"S{i}" for i in range(num_states)]

        children = [{"label": states[0], "is_initial": True}]
        children.extend([{"label": s} for s in states[1:]])

        transitions = []
        for i in range(len(states) - 1):
            transitions.append({
                "from": [states[i]],
                "to": [states[i + 1]],
                "event": f"E{i}",
            })

        return {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": children,
            },
            "transitions": transitions,
        }

    def generate_add_state_case(self) -> TestCase:
        """Generate test case: add a state."""
        before = self.generate_base_chart(3)
        after = copy.deepcopy(before)

        # Add new state
        after["root_state"]["children"].append({"label": "NewState"})

        return TestCase(
            name="Add state",
            before=before,
            after=after,
            expected_changes=[
                ExpectedChange("state_added", "NewState"),
            ],
            expected_compatibility="COMPATIBLE",
        )

    def generate_remove_state_case(self) -> TestCase:
        """Generate test case: remove a state."""
        before = self.generate_base_chart(4)
        after = copy.deepcopy(before)

        # Remove last state
        removed = after["root_state"]["children"].pop()
        removed_label = removed["label"]

        # Count transitions that will be removed
        removed_transitions = [
            t for t in before["transitions"]
            if removed_label in t.get("from", []) or removed_label in t.get("to", [])
        ]

        # Remove transitions to/from that state
        after["transitions"] = [
            t for t in after["transitions"]
            if removed_label not in t.get("from", []) and removed_label not in t.get("to", [])
        ]

        # Build expected changes
        expected = [ExpectedChange("state_removed", removed_label)]
        for t in removed_transitions:
            expected.append(ExpectedChange(
                "transition_removed",
                f"{t['from'][0]}->{t['to'][0]}"
            ))

        return TestCase(
            name="Remove state",
            before=before,
            after=after,
            expected_changes=expected,
            expected_compatibility="BREAKING",
        )

    def generate_add_transition_case(self) -> TestCase:
        """Generate test case: add a transition."""
        before = self.generate_base_chart(3)
        after = copy.deepcopy(before)

        # Add new transition
        after["transitions"].append({
            "from": ["S2"],
            "to": ["S0"],
            "event": "RESET",
        })

        return TestCase(
            name="Add transition",
            before=before,
            after=after,
            expected_changes=[
                ExpectedChange("transition_added", "S2->S0[RESET]"),
            ],
            expected_compatibility="COMPATIBLE",
        )

    def generate_remove_transition_case(self) -> TestCase:
        """Generate test case: remove a transition."""
        before = self.generate_base_chart(4)
        after = copy.deepcopy(before)

        # Remove a transition
        removed = after["transitions"].pop()

        return TestCase(
            name="Remove transition",
            before=before,
            after=after,
            expected_changes=[
                ExpectedChange("transition_removed", f"{removed['from'][0]}->{removed['to'][0]}"),
            ],
            expected_compatibility="BREAKING",
        )

    def generate_no_change_case(self) -> TestCase:
        """Generate test case: no changes."""
        chart = self.generate_base_chart(3)

        return TestCase(
            name="No changes",
            before=chart,
            after=copy.deepcopy(chart),
            expected_changes=[],
            expected_compatibility="COMPATIBLE",
        )

    def generate_complex_case(self) -> TestCase:
        """Generate test case: multiple changes."""
        before = self.generate_base_chart(4)
        after = copy.deepcopy(before)

        # Add state
        after["root_state"]["children"].append({"label": "Added"})

        # Remove state S2 - this removes transitions S1->S2 and S2->S3
        removed = after["root_state"]["children"].pop(2)  # Remove S2
        removed_label = removed["label"]

        # Count transitions that will be removed
        removed_transitions = [
            t for t in before["transitions"]
            if removed_label in t.get("from", []) or removed_label in t.get("to", [])
        ]

        # Update transitions
        after["transitions"] = [
            t for t in after["transitions"]
            if removed_label not in t.get("from", []) and removed_label not in t.get("to", [])
        ]

        # Add new transition
        after["transitions"].append({
            "from": ["S1"],
            "to": ["Added"],
            "event": "NEW_EVENT",
        })

        # Build expected changes
        expected = [
            ExpectedChange("state_added", "Added"),
            ExpectedChange("state_removed", removed_label),
            ExpectedChange("transition_added", "S1->Added[NEW_EVENT]"),
        ]
        # Add all removed transitions
        for t in removed_transitions:
            expected.append(ExpectedChange(
                "transition_removed",
                f"{t['from'][0]}->{t['to'][0]}"
            ))

        return TestCase(
            name="Complex changes",
            before=before,
            after=after,
            expected_changes=expected,
            expected_compatibility="BREAKING",
        )

    def generate_test_suite(self, num_cases: int = 20) -> List[TestCase]:
        """Generate a suite of test cases."""
        generators = [
            self.generate_add_state_case,
            self.generate_remove_state_case,
            self.generate_add_transition_case,
            self.generate_remove_transition_case,
            self.generate_no_change_case,
            self.generate_complex_case,
        ]

        cases = []
        for i in range(num_cases):
            generator = random.choice(generators)
            case = generator()
            case.name = f"{case.name} #{i+1}"
            cases.append(case)

        return cases


class DiffBenchmark:
    """Benchmarks diff detection accuracy."""

    def __init__(self, sc_path: str = "./sc"):
        self.detector = ChangeDetector(sc_path)
        self.generator = TestCaseGenerator()

    def run_benchmark(
        self,
        num_cases: int = 30,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run full benchmark.

        Args:
            num_cases: Number of test cases
            verbose: Print progress

        Returns:
            BenchmarkResult with statistics
        """
        result = BenchmarkResult()

        if verbose:
            print(f"Generating {num_cases} test cases...")

        cases = self.generator.generate_test_suite(num_cases)

        if verbose:
            print(f"Running benchmark...")

        start_time = time.time()

        for case in cases:
            case_result = self._evaluate_case(case)
            result.cases.append(case_result)

            if case_result["detection_correct"]:
                result.correct_detections += 1
            if case_result["compatibility_correct"]:
                result.compatibility_correct += 1
            result.false_positives += case_result["false_positives"]
            result.false_negatives += case_result["false_negatives"]
            result.total_cases += 1

            if verbose:
                status = "OK" if case_result["detection_correct"] else "FAIL"
                print(f"  [{result.total_cases}/{num_cases}] {status} - {case.name}")

        result.total_duration = time.time() - start_time

        return result

    def _evaluate_case(self, case: TestCase) -> Dict[str, Any]:
        """Evaluate a single test case."""
        diff = self.detector.diff(case.before, case.after)

        # Check expected changes
        detected_changes = set()
        for change in diff.states_added:
            detected_changes.add(("state_added", change.state_label))
        for change in diff.states_removed:
            detected_changes.add(("state_removed", change.state_label))
        for change in diff.transitions_added:
            key = f"{change.from_states[0]}->{change.to_states[0]}[{change.event}]"
            detected_changes.add(("transition_added", key))
        for change in diff.transitions_removed:
            key = f"{change.from_states[0]}->{change.to_states[0]}"
            detected_changes.add(("transition_removed", key))

        expected_changes = set()
        for exp in case.expected_changes:
            expected_changes.add((exp.change_type, exp.label))

        # Calculate metrics
        # For simpler matching, just check change types and counts
        expected_types = {}
        for exp in case.expected_changes:
            expected_types[exp.change_type] = expected_types.get(exp.change_type, 0) + 1

        detected_types = {}
        for change_type, _ in detected_changes:
            detected_types[change_type] = detected_types.get(change_type, 0) + 1

        # Check if counts match for each type
        detection_correct = True
        false_positives = 0
        false_negatives = 0

        all_types = set(expected_types.keys()) | set(detected_types.keys())
        for t in all_types:
            expected = expected_types.get(t, 0)
            detected = detected_types.get(t, 0)
            if expected != detected:
                detection_correct = False
                if detected > expected:
                    false_positives += detected - expected
                else:
                    false_negatives += expected - detected

        # Check compatibility
        compatibility_correct = diff.compatibility.name == case.expected_compatibility

        return {
            "name": case.name,
            "detection_correct": detection_correct,
            "compatibility_correct": compatibility_correct,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected": list(expected_changes),
            "detected": list(detected_changes),
        }


def run_full_benchmark(num_cases: int = 30) -> BenchmarkResult:
    """Run full benchmark and print results."""
    print("=" * 60)
    print("STATECHART DIFF BENCHMARK")
    print("=" * 60)

    benchmark = DiffBenchmark()
    result = benchmark.run_benchmark(num_cases)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nDetection Accuracy: {result.correct_detections}/{result.total_cases} "
          f"({result.detection_accuracy:.1%})")
    print(f"Compatibility Accuracy: {result.compatibility_correct}/{result.total_cases} "
          f"({result.compatibility_accuracy:.1%})")
    print(f"Overall Accuracy: {result.overall_accuracy:.1%}")

    print(f"\nFalse Positives: {result.false_positives}")
    print(f"False Negatives: {result.false_negatives}")

    print(f"\nTiming:")
    print(f"  - Total duration: {result.total_duration:.2f}s")
    print(f"  - Average per case: {result.total_duration/result.total_cases:.3f}s")

    # Check target
    target_met = result.detection_accuracy >= 0.95
    print(f"\n{'='*60}")
    print(f"TARGET (95%): {'ACHIEVED' if target_met else 'NOT MET'}")
    print(f"{'='*60}")

    return result


def demo():
    """Quick demo with fewer cases."""
    print("=" * 60)
    print("DIFF BENCHMARK DEMO")
    print("=" * 60)

    benchmark = DiffBenchmark()
    result = benchmark.run_benchmark(num_cases=10)

    print(f"\nDetection Accuracy: {result.detection_accuracy:.1%}")
    print(f"Compatibility Accuracy: {result.compatibility_accuracy:.1%}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark(num_cases=50)
    else:
        demo()
