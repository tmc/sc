"""
Repair Benchmark: Evaluate repair accuracy with REAL inference.

Tests repair engine across error categories:
1. Duplicate labels
2. Unreachable states
3. Missing initial
4. Invalid hierarchy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import time

# Import repair components
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')
from experiments.exp_sc_repair.repair_engine import SCRepairEngine, RepairConfig
from experiments.exp_sc_repair.repair_validator import validate_repair
from experiments.exp_sc_error_patterns.error_detector import detect_errors


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    error_category: str
    broken_sc: Dict[str, Any]
    description: str


@dataclass
class BenchmarkResult:
    """Result of running benchmark."""
    total_cases: int
    successful_repairs: int
    partial_repairs: int
    failed_repairs: int
    accuracy: float
    per_category: Dict[str, Dict[str, Any]]
    details: List[Dict[str, Any]]
    inference_time: float


def get_benchmark_cases() -> List[BenchmarkCase]:
    """Get test cases for each error category."""
    cases = []

    # === DUPLICATE LABELS ===
    cases.append(BenchmarkCase(
        name="duplicate_simple",
        error_category="duplicate_element",
        broken_sc={
            "name": "DuplicateTest",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "A", "type": 1},  # Duplicate!
                    {"label": "B", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "GO"},
            ]
        },
        description="Simple duplicate state label",
    ))

    cases.append(BenchmarkCase(
        name="duplicate_nested",
        error_category="duplicate_element",
        broken_sc={
            "name": "NestedDuplicate",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {
                        "label": "Parent",
                        "type": 2,
                        "is_initial": True,
                        "children": [
                            {"label": "Child", "type": 1, "is_initial": True},
                            {"label": "Child", "type": 1},  # Duplicate!
                        ]
                    },
                ]
            },
            "transitions": []
        },
        description="Duplicate labels in nested states",
    ))

    # === UNREACHABLE STATES ===
    cases.append(BenchmarkCase(
        name="unreachable_simple",
        error_category="unreachable_state",
        broken_sc={
            "name": "UnreachableTest",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Start", "type": 1, "is_initial": True},
                    {"label": "Middle", "type": 1},
                    {"label": "Orphan", "type": 1},  # Unreachable!
                ]
            },
            "transitions": [
                {"from": ["Start"], "to": ["Middle"], "event": "GO"},
            ]
        },
        description="State with no incoming transitions",
    ))

    cases.append(BenchmarkCase(
        name="unreachable_island",
        error_category="unreachable_state",
        broken_sc={
            "name": "IslandTest",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Main", "type": 1, "is_initial": True},
                    {"label": "Island1", "type": 1},  # Unreachable island
                    {"label": "Island2", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Island1"], "to": ["Island2"], "event": "HOP"},
            ]
        },
        description="Island of states disconnected from main flow",
    ))

    # === MISSING INITIAL ===
    cases.append(BenchmarkCase(
        name="missing_initial_root",
        error_category="hierarchy_violation",
        broken_sc={
            "name": "NoInitialTest",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1},  # No initial!
                    {"label": "B", "type": 1},
                    {"label": "C", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "GO"},
            ]
        },
        description="No initial state at root level",
    ))

    cases.append(BenchmarkCase(
        name="missing_initial_composite",
        error_category="hierarchy_violation",
        broken_sc={
            "name": "CompositeNoInitial",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {
                        "label": "Outer",
                        "type": 2,
                        "is_initial": True,
                        "children": [
                            {"label": "Inner1", "type": 1},  # No initial in composite!
                            {"label": "Inner2", "type": 1},
                        ]
                    },
                ]
            },
            "transitions": []
        },
        description="Composite state without initial child",
    ))

    # === INVALID HIERARCHY ===
    cases.append(BenchmarkCase(
        name="hierarchy_empty_composite",
        error_category="hierarchy_violation",
        broken_sc={
            "name": "EmptyComposite",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Normal", "type": 1, "is_initial": True},
                    {"label": "Empty", "type": 2, "children": []},  # Empty composite!
                ]
            },
            "transitions": []
        },
        description="Composite state with no children",
    ))

    cases.append(BenchmarkCase(
        name="hierarchy_multiple_initial",
        error_category="hierarchy_violation",
        broken_sc={
            "name": "MultipleInitial",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1, "is_initial": True},  # Multiple initial!
                    {"label": "C", "type": 1},
                ]
            },
            "transitions": []
        },
        description="Multiple initial states at same level",
    ))

    # === DANGLING TRANSITIONS ===
    cases.append(BenchmarkCase(
        name="dangling_source",
        error_category="dangling_reference",
        broken_sc={
            "name": "DanglingSource",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["NonExistent"], "to": ["B"], "event": "GO"},  # Bad source!
            ]
        },
        description="Transition from non-existent state",
    ))

    cases.append(BenchmarkCase(
        name="dangling_target",
        error_category="dangling_reference",
        broken_sc={
            "name": "DanglingTarget",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["A"], "to": ["Missing"], "event": "GO"},  # Bad target!
            ]
        },
        description="Transition to non-existent state",
    ))

    return cases


class RepairBenchmark:
    """Benchmark runner for repair accuracy."""

    def __init__(self, config: Optional[RepairConfig] = None):
        self.config = config or RepairConfig()
        self.engine = None

    def run(self, use_llm: bool = True) -> BenchmarkResult:
        """
        Run full benchmark.

        Args:
            use_llm: If True, use REAL LLM inference

        Returns:
            BenchmarkResult with accuracy metrics
        """
        # Initialize engine
        self.engine = SCRepairEngine(self.config)

        cases = get_benchmark_cases()
        results = []
        per_category = {}

        start_time = time.time()

        for case in cases:
            result = self._run_case(case, use_llm)
            results.append(result)

            # Track per-category stats
            cat = case.error_category
            if cat not in per_category:
                per_category[cat] = {"total": 0, "success": 0, "partial": 0, "failed": 0}

            per_category[cat]["total"] += 1
            if result["status"] == "success":
                per_category[cat]["success"] += 1
            elif result["status"] == "partial":
                per_category[cat]["partial"] += 1
            else:
                per_category[cat]["failed"] += 1

        inference_time = time.time() - start_time

        # Calculate metrics
        successful = sum(1 for r in results if r["status"] == "success")
        partial = sum(1 for r in results if r["status"] == "partial")
        failed = sum(1 for r in results if r["status"] == "failed")

        accuracy = successful / len(cases) if cases else 0.0

        return BenchmarkResult(
            total_cases=len(cases),
            successful_repairs=successful,
            partial_repairs=partial,
            failed_repairs=failed,
            accuracy=accuracy,
            per_category=per_category,
            details=results,
            inference_time=inference_time,
        )

    def _run_case(self, case: BenchmarkCase, use_llm: bool) -> Dict[str, Any]:
        """Run a single benchmark case."""
        result = {
            "name": case.name,
            "category": case.error_category,
            "description": case.description,
            "status": "failed",
            "original_errors": 0,
            "repaired_errors": 0,
            "validation": None,
            "error": None,
        }

        try:
            # Get original error count
            orig_detect = detect_errors(case.broken_sc)
            orig_errors = len(orig_detect.errors) + len(orig_detect.warnings)
            result["original_errors"] = orig_errors

            # Attempt repair
            repair_result = self.engine.repair(case.broken_sc)

            if not repair_result.success:
                result["status"] = "failed"
                result["error"] = "Repair engine returned failure"
                return result

            # Validate repair
            validation = validate_repair(case.broken_sc, repair_result.repaired)
            result["validation"] = {
                "is_valid": validation.is_valid,
                "errors_fixed": validation.errors_fixed,
                "errors_introduced": validation.errors_introduced,
                "semantic_preserved": validation.semantic_preserved,
            }
            result["repaired_errors"] = validation.repaired_error_count

            # Determine status
            if validation.is_valid and validation.repaired_error_count == 0:
                result["status"] = "success"
            elif validation.errors_fixed > 0 and validation.errors_introduced == 0:
                result["status"] = "partial"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        return result


def run_benchmark(use_llm: bool = True) -> BenchmarkResult:
    """Convenience function to run benchmark."""
    benchmark = RepairBenchmark()
    return benchmark.run(use_llm=use_llm)


def demo():
    """Run benchmark demo with REAL inference."""
    print("=" * 60)
    print("SC REPAIR BENCHMARK: REAL LLM Inference")
    print("=" * 60)

    benchmark = RepairBenchmark()
    result = benchmark.run(use_llm=True)

    print(f"\n=== RESULTS ===")
    print(f"Total cases: {result.total_cases}")
    print(f"Successful: {result.successful_repairs}")
    print(f"Partial: {result.partial_repairs}")
    print(f"Failed: {result.failed_repairs}")
    print(f"ACCURACY: {result.accuracy * 100:.1f}%")
    print(f"Inference time: {result.inference_time:.2f}s")

    print(f"\n=== PER-CATEGORY ===")
    for cat, stats in result.per_category.items():
        cat_acc = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {stats['success']}/{stats['total']} ({cat_acc*100:.0f}%)")

    print(f"\n=== CASE DETAILS ===")
    for detail in result.details:
        status_icon = "✓" if detail["status"] == "success" else ("~" if detail["status"] == "partial" else "✗")
        print(f"  {status_icon} {detail['name']}: {detail['status']}")
        if detail.get("error"):
            print(f"      Error: {detail['error']}")

    return result


if __name__ == "__main__":
    demo()
