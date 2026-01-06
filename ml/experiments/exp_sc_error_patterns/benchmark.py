"""
Benchmark: Error Detection Accuracy Evaluation

Evaluates error detection on:
- Known error cases (ground truth)
- Synthetic error injection
- LLM-generated statecharts
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
import time
import random
import copy

from .error_taxonomy import ErrorCategory, ErrorSeverity
from .error_detector import ErrorDetector, DetectionResult, detect_errors
from .error_patterns import PatternMatcher, COMMON_PATTERNS
from .error_fixer import ErrorFixer, FixSuggestion


@dataclass
class BenchmarkResult:
    """Result from a single benchmark case."""
    name: str
    category: str

    # Ground truth
    expected_errors: int
    expected_categories: List[str]

    # Detection results
    detected_errors: int
    detected_categories: List[str]

    # Metrics
    precision: float
    recall: float
    f1_score: float
    detection_time_ms: float

    # Details
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    @property
    def avg_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision for r in self.results) / len(self.results)

    @property
    def avg_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def avg_f1(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.f1_score for r in self.results) / len(self.results)

    def summary(self) -> Dict[str, Any]:
        by_category = {}
        for r in self.results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r.f1_score)

        return {
            "total_cases": len(self.results),
            "avg_precision": round(self.avg_precision, 3),
            "avg_recall": round(self.avg_recall, 3),
            "avg_f1": round(self.avg_f1, 3),
            "by_category": {
                k: round(sum(v) / len(v), 3)
                for k, v in by_category.items()
            }
        }


# Ground truth test cases
GROUND_TRUTH_CASES: List[Dict[str, Any]] = [
    # Missing state cases
    {
        "name": "missing_target_state",
        "category": "missing_state",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "GO"},
            ]
        },
        "expected_errors": [
            {"category": "missing_state", "state": "B"},
        ]
    },
    {
        "name": "missing_source_state",
        "category": "missing_state",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "B", "type": 1, "is_initial": True},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "GO"},
            ]
        },
        "expected_errors": [
            {"category": "missing_state", "state": "A"},
        ]
    },

    # Dangling transition cases
    {
        "name": "empty_source",
        "category": "dangling_transition",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                ]
            },
            "transitions": [
                {"from": [], "to": ["A"], "event": "RESET"},
            ]
        },
        "expected_errors": [
            {"category": "dangling_transition", "issue": "no_source"},
        ]
    },

    # Invalid guard cases
    {
        "name": "guard_syntax_error",
        "category": "invalid_guard",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "GO",
                 "guard": "count >> 5"},  # Invalid >>
            ]
        },
        "expected_errors": [
            {"category": "invalid_guard", "guard": "count >> 5"},
        ]
    },

    # Hierarchy violation cases
    {
        "name": "missing_initial_in_composite",
        "category": "hierarchy_violation",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {
                        "label": "Composite",
                        "type": 2,
                        "children": [
                            {"label": "Sub1", "type": 1},
                            {"label": "Sub2", "type": 1},
                        ]
                    }
                ]
            },
            "transitions": []
        },
        "expected_errors": [
            {"category": "hierarchy_violation", "issue": "no_initial"},
        ]
    },
    {
        "name": "multiple_initial_states",
        "category": "hierarchy_violation",
        "statechart": {
            "name": "Test",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1, "is_initial": True},
                ]
            },
            "transitions": []
        },
        "expected_errors": [
            {"category": "hierarchy_violation", "issue": "multiple_initial"},
        ]
    },

    # Valid statechart (no errors)
    {
        "name": "valid_simple",
        "category": "valid",
        "statechart": {
            "name": "Valid",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Start", "type": 1, "is_initial": True},
                    {"label": "End", "type": 1, "is_final": True},
                ]
            },
            "transitions": [
                {"from": ["Start"], "to": ["End"], "event": "FINISH"},
            ]
        },
        "expected_errors": []
    },
]


class ErrorInjector:
    """Inject errors into valid statecharts for testing."""

    def inject_missing_state(self, sc: Dict) -> Tuple[Dict, Dict]:
        """Inject a missing state error."""
        sc = copy.deepcopy(sc)
        transitions = sc.get("transitions", [])

        if transitions:
            # Pick a transition and add non-existent target
            trans = random.choice(transitions)
            fake_state = f"NonExistent_{random.randint(100, 999)}"
            trans["to"] = [fake_state]

            return sc, {
                "category": "missing_state",
                "state": fake_state,
            }

        return sc, {}

    def inject_dangling_transition(self, sc: Dict) -> Tuple[Dict, Dict]:
        """Inject a dangling transition."""
        sc = copy.deepcopy(sc)

        sc.setdefault("transitions", []).append({
            "from": [],
            "to": ["SomeState"],
            "event": "ORPHAN",
        })

        return sc, {
            "category": "dangling_transition",
            "event": "ORPHAN",
        }

    def inject_invalid_guard(self, sc: Dict) -> Tuple[Dict, Dict]:
        """Inject an invalid guard."""
        sc = copy.deepcopy(sc)
        transitions = sc.get("transitions", [])

        if transitions:
            trans = random.choice(transitions)
            trans["guard"] = "x >> 5"  # Invalid syntax

            return sc, {
                "category": "invalid_guard",
                "guard": "x >> 5",
            }

        return sc, {}

    def inject_hierarchy_violation(self, sc: Dict) -> Tuple[Dict, Dict]:
        """Inject a hierarchy violation."""
        sc = copy.deepcopy(sc)

        # Remove is_initial from all states
        def remove_initial(state):
            state["is_initial"] = False
            for child in state.get("children", []):
                remove_initial(child)

        remove_initial(sc.get("root_state", {}))

        return sc, {
            "category": "hierarchy_violation",
            "issue": "no_initial",
        }


class ErrorDetectionBenchmark:
    """Benchmark error detection accuracy."""

    def __init__(self):
        self.detector = ErrorDetector()
        self.injector = ErrorInjector()
        self.suite = BenchmarkSuite()

    def run_ground_truth(self) -> BenchmarkSuite:
        """Run benchmark on ground truth cases."""
        for case in GROUND_TRUTH_CASES:
            result = self._evaluate_case(case)
            self.suite.add_result(result)

        return self.suite

    def run_injection_tests(
        self,
        base_statechart: Dict[str, Any],
        n_tests: int = 10,
    ) -> BenchmarkSuite:
        """Run benchmark with error injection."""
        injectors = [
            ("missing_state", self.injector.inject_missing_state),
            ("dangling_transition", self.injector.inject_dangling_transition),
            ("invalid_guard", self.injector.inject_invalid_guard),
            ("hierarchy_violation", self.injector.inject_hierarchy_violation),
        ]

        for i in range(n_tests):
            category, inject_fn = random.choice(injectors)
            injected, expected = inject_fn(base_statechart)

            if expected:
                case = {
                    "name": f"injection_{category}_{i}",
                    "category": category,
                    "statechart": injected,
                    "expected_errors": [expected],
                }
                result = self._evaluate_case(case)
                self.suite.add_result(result)

        return self.suite

    def _evaluate_case(self, case: Dict) -> BenchmarkResult:
        """Evaluate a single test case."""
        t0 = time.time()
        detection = self.detector.detect(case["statechart"])
        detection_time = (time.time() - t0) * 1000

        expected = case.get("expected_errors", [])
        expected_categories = [e.get("category", "") for e in expected]

        detected_categories = [
            e.category.value for e in detection.errors + detection.warnings
        ]

        # Calculate metrics
        tp, fp, fn = self._calculate_confusion(
            expected_categories,
            detected_categories,
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return BenchmarkResult(
            name=case["name"],
            category=case["category"],
            expected_errors=len(expected),
            expected_categories=expected_categories,
            detected_errors=len(detection.errors) + len(detection.warnings),
            detected_categories=detected_categories,
            precision=precision,
            recall=recall,
            f1_score=f1,
            detection_time_ms=detection_time,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def _calculate_confusion(
        self,
        expected: List[str],
        detected: List[str],
    ) -> Tuple[int, int, int]:
        """Calculate TP, FP, FN."""
        expected_set = set(expected)
        detected_set = set(detected)

        tp = len(expected_set & detected_set)
        fp = len(detected_set - expected_set)
        fn = len(expected_set - detected_set)

        return tp, fp, fn


def run_benchmark(verbose: bool = True) -> BenchmarkSuite:
    """Run comprehensive benchmark."""
    benchmark = ErrorDetectionBenchmark()

    if verbose:
        print("=" * 60)
        print("ERROR DETECTION BENCHMARK")
        print("=" * 60)

    # Ground truth tests
    if verbose:
        print("\n--- Ground Truth Cases ---")

    benchmark.run_ground_truth()

    if verbose:
        for r in benchmark.suite.results:
            status = "PASS" if r.f1_score >= 0.8 else "FAIL"
            print(f"  [{status}] {r.name}: P={r.precision:.2f} R={r.recall:.2f} F1={r.f1_score:.2f}")

    # Injection tests
    if verbose:
        print("\n--- Injection Tests ---")

    base_sc = {
        "name": "Base",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "Active", "type": 1},
                {"label": "Done", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Active"], "event": "START"},
            {"from": ["Active"], "to": ["Done"], "event": "FINISH"},
        ]
    }

    initial_count = len(benchmark.suite.results)
    benchmark.run_injection_tests(base_sc, n_tests=8)

    if verbose:
        for r in benchmark.suite.results[initial_count:]:
            status = "PASS" if r.f1_score >= 0.8 else "FAIL"
            print(f"  [{status}] {r.name}: P={r.precision:.2f} R={r.recall:.2f} F1={r.f1_score:.2f}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        summary = benchmark.suite.summary()
        print(f"Total cases: {summary['total_cases']}")
        print(f"Avg Precision: {summary['avg_precision']}")
        print(f"Avg Recall: {summary['avg_recall']}")
        print(f"Avg F1: {summary['avg_f1']}")
        print(f"\nBy category:")
        for cat, f1 in summary['by_category'].items():
            print(f"  {cat}: F1={f1}")

    return benchmark.suite


def demo():
    """Quick benchmark demo."""
    print("Running error detection benchmark...")
    return run_benchmark(verbose=True)


if __name__ == "__main__":
    demo()
