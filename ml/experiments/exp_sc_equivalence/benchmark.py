"""
Equivalence Benchmark: Evaluate equivalence checking accuracy.

Metrics:
- Accuracy: % correct verdicts
- False Positive Rate: % incorrectly marked equivalent
- False Negative Rate: % incorrectly marked non-equivalent
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time

from .equivalence_checker import EquivalenceChecker, EquivalenceConfig


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    sc1: Dict[str, Any]
    sc2: Dict[str, Any]
    expected_equivalent: bool
    description: str


@dataclass
class BenchmarkResult:
    """Result of running benchmark."""
    total_cases: int
    correct: int
    accuracy: float
    false_positives: int  # Said equivalent but wasn't
    false_negatives: int  # Said not equivalent but was
    fp_rate: float
    fn_rate: float
    details: List[Dict[str, Any]]
    inference_time: float


def get_benchmark_cases() -> List[BenchmarkCase]:
    """Get test cases for equivalence checking."""
    cases = []

    # === EQUIVALENT CASES ===

    # Case 1: Identical statecharts
    sc_identical = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="identical",
        sc1=sc_identical,
        sc2=sc_identical,
        expected_equivalent=True,
        description="Identical statecharts",
    ))

    # Case 2: Same behavior, different names
    sc_renamed = {
        "name": "Semaphore",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Stop", "type": 1, "is_initial": True},
                {"label": "Go", "type": 1},
                {"label": "Caution", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Stop"], "to": ["Go"], "event": "TIMER"},
            {"from": ["Go"], "to": ["Caution"], "event": "TIMER"},
            {"from": ["Caution"], "to": ["Stop"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="renamed_states",
        sc1=sc_identical,
        sc2=sc_renamed,
        expected_equivalent=True,
        description="Same behavior with renamed states",
    ))

    # Case 3: Same behavior, different transition order
    sc_reordered = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="reordered_transitions",
        sc1=sc_identical,
        sc2=sc_reordered,
        expected_equivalent=True,
        description="Same behavior with reordered transitions",
    ))

    # Case 4: Simple toggle - equivalent
    toggle1 = {
        "name": "Toggle1",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
        ]
    }

    toggle2 = {
        "name": "Toggle2",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Inactive", "type": 1, "is_initial": True},
                {"label": "Active", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Inactive"], "to": ["Active"], "event": "TOGGLE"},
            {"from": ["Active"], "to": ["Inactive"], "event": "TOGGLE"},
        ]
    }

    cases.append(BenchmarkCase(
        name="toggle_equivalent",
        sc1=toggle1,
        sc2=toggle2,
        expected_equivalent=True,
        description="Two-state toggle with different names",
    ))

    # === NON-EQUIVALENT CASES ===

    # Case 5: Different number of states
    sc_missing_yellow = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="missing_state",
        sc1=sc_identical,
        sc2=sc_missing_yellow,
        expected_equivalent=False,
        description="One has Yellow state, other doesn't",
    ))

    # Case 6: Different events
    sc_diff_event = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TICK"},  # Different event
            {"from": ["Green"], "to": ["Yellow"], "event": "TICK"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TICK"},
        ]
    }

    cases.append(BenchmarkCase(
        name="different_events",
        sc1=sc_identical,
        sc2=sc_diff_event,
        expected_equivalent=False,
        description="Different event names (TIMER vs TICK)",
    ))

    # Case 7: Different initial state
    sc_diff_initial = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1},
                {"label": "Green", "type": 1, "is_initial": True},  # Green is initial
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="different_initial",
        sc1=sc_identical,
        sc2=sc_diff_initial,
        expected_equivalent=False,
        description="Different initial states (Red vs Green)",
    ))

    # Case 8: Different transition targets
    sc_diff_target = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Yellow"], "event": "TIMER"},  # Red->Yellow instead of Red->Green
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="different_target",
        sc1=sc_identical,
        sc2=sc_diff_target,
        expected_equivalent=False,
        description="Different transition target (Red->Yellow vs Red->Green)",
    ))

    # Case 9: Extra transition
    sc_extra_trans = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
            {"from": ["Red"], "to": ["Yellow"], "event": "EMERGENCY"},  # Extra
        ]
    }

    cases.append(BenchmarkCase(
        name="extra_transition",
        sc1=sc_identical,
        sc2=sc_extra_trans,
        expected_equivalent=False,
        description="One has extra EMERGENCY transition",
    ))

    # Case 10: Self-loop difference
    sc_with_selfloop = {
        "name": "Toggle",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["On"], "event": "REFRESH"},  # Self-loop
        ]
    }

    cases.append(BenchmarkCase(
        name="selfloop_difference",
        sc1=toggle1,
        sc2=sc_with_selfloop,
        expected_equivalent=False,
        description="One has self-loop on REFRESH",
    ))

    # Case 11: Equivalent with extra unreachable state
    sc_unreachable = {
        "name": "Toggle",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1},
                {"label": "Orphan", "type": 1},  # Unreachable
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
        ]
    }

    cases.append(BenchmarkCase(
        name="unreachable_state",
        sc1=toggle1,
        sc2=sc_unreachable,
        expected_equivalent=True,  # Behaviorally same - unreachable state doesn't affect traces
        description="Equivalent despite unreachable state",
    ))

    # Case 12: Different structure, same traces
    # Linear vs cycle that produces same short traces
    linear = {
        "name": "Linear",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1, "is_initial": True},
                {"label": "B", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"},
        ]
    }

    also_linear = {
        "name": "AlsoLinear",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Start", "type": 1, "is_initial": True},
                {"label": "End", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Start"], "to": ["End"], "event": "GO"},
        ]
    }

    cases.append(BenchmarkCase(
        name="same_linear",
        sc1=linear,
        sc2=also_linear,
        expected_equivalent=True,
        description="Simple linear progression with different names",
    ))

    return cases


class EquivalenceBenchmark:
    """Benchmark runner for equivalence checking."""

    def __init__(self, config: Optional[EquivalenceConfig] = None):
        self.config = config or EquivalenceConfig()

    def run(self) -> BenchmarkResult:
        """Run full benchmark."""
        checker = EquivalenceChecker(self.config)
        cases = get_benchmark_cases()

        results = []
        correct = 0
        false_positives = 0
        false_negatives = 0

        start_time = time.time()

        for case in cases:
            result = checker.check_equivalence(case.sc1, case.sc2)

            is_correct = result.equivalent == case.expected_equivalent

            if is_correct:
                correct += 1
            else:
                if result.equivalent and not case.expected_equivalent:
                    false_positives += 1
                elif not result.equivalent and case.expected_equivalent:
                    false_negatives += 1

            results.append({
                "name": case.name,
                "description": case.description,
                "expected": case.expected_equivalent,
                "predicted": result.equivalent,
                "correct": is_correct,
                "confidence": result.confidence,
                "method": result.method_used,
                "evidence": result.evidence,
                "counterexample": result.counterexample,
            })

        inference_time = time.time() - start_time

        total = len(cases)
        # Count positives and negatives for rate calculation
        expected_equiv = sum(1 for c in cases if c.expected_equivalent)
        expected_not_equiv = total - expected_equiv

        accuracy = correct / total if total > 0 else 0.0
        fp_rate = false_positives / expected_not_equiv if expected_not_equiv > 0 else 0.0
        fn_rate = false_negatives / expected_equiv if expected_equiv > 0 else 0.0

        return BenchmarkResult(
            total_cases=total,
            correct=correct,
            accuracy=accuracy,
            false_positives=false_positives,
            false_negatives=false_negatives,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            details=results,
            inference_time=inference_time,
        )


def run_benchmark() -> BenchmarkResult:
    """Convenience function to run benchmark."""
    benchmark = EquivalenceBenchmark()
    return benchmark.run()


def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("SC EQUIVALENCE BENCHMARK")
    print("=" * 60)

    benchmark = EquivalenceBenchmark()
    result = benchmark.run()

    print(f"\n=== RESULTS ===")
    print(f"Total cases: {result.total_cases}")
    print(f"Correct: {result.correct}")
    print(f"ACCURACY: {result.accuracy * 100:.1f}%")
    print(f"False Positives: {result.false_positives} ({result.fp_rate * 100:.1f}%)")
    print(f"False Negatives: {result.false_negatives} ({result.fn_rate * 100:.1f}%)")
    print(f"Inference time: {result.inference_time:.2f}s")

    print(f"\n=== CASE DETAILS ===")
    for detail in result.details:
        status = "✓" if detail["correct"] else "✗"
        pred = "EQUIV" if detail["predicted"] else "NOT-EQUIV"
        exp = "EQUIV" if detail["expected"] else "NOT-EQUIV"
        print(f"  {status} {detail['name']}: predicted={pred}, expected={exp}")
        if not detail["correct"]:
            print(f"      {detail['description']}")
            if detail["counterexample"]:
                print(f"      Counterexample: {detail['counterexample']}")

    return result


if __name__ == "__main__":
    demo()
