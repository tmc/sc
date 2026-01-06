"""
Hierarchy-Focused Benchmark for Terminal State Prediction.

Tests LLM's ability to:
1. Enter composite states correctly (cascade to initial child)
2. Exit composite states (all descendants exit)
3. Understand leaf vs composite state distinction
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from . import TraceExecutor, configuration_accuracy
from .hierarchy_predictor import HierarchyPredictor, HierarchyPredictionResult


# =============================================================================
# Hierarchy-Focused Test Cases
# =============================================================================

# Test 1: Simple hierarchy - entering composite state
SIMPLE_HIERARCHY_SC = {
    "name": "SimpleHierarchy",
    "category": "hierarchy",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Idle", "type": 1, "is_initial": True},
            {
                "label": "Active",
                "type": 2,
                "children": [
                    {"label": "Running", "type": 1, "is_initial": True},
                    {"label": "Paused", "type": 1},
                ]
            },
            {"label": "Done", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["Idle"], "to": ["Active"], "event": "START"},
        {"from": ["Running"], "to": ["Paused"], "event": "PAUSE"},
        {"from": ["Paused"], "to": ["Running"], "event": "RESUME"},
        {"from": ["Active"], "to": ["Done"], "event": "STOP"},
    ]
}

# Test 2: Deep hierarchy - multiple levels
DEEP_HIERARCHY_SC = {
    "name": "DeepHierarchy",
    "category": "hierarchy",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Off", "type": 1, "is_initial": True},
            {
                "label": "On",
                "type": 2,
                "children": [
                    {
                        "label": "Ready",
                        "type": 2,
                        "is_initial": True,
                        "children": [
                            {"label": "Waiting", "type": 1, "is_initial": True},
                            {"label": "Processing", "type": 1},
                        ]
                    },
                    {"label": "Error", "type": 1},
                ]
            },
        ]
    },
    "transitions": [
        {"from": ["Off"], "to": ["On"], "event": "POWER"},
        {"from": ["Waiting"], "to": ["Processing"], "event": "START"},
        {"from": ["Processing"], "to": ["Waiting"], "event": "DONE"},
        {"from": ["Ready"], "to": ["Error"], "event": "FAIL"},
        {"from": ["Error"], "to": ["Ready"], "event": "RESET"},
        {"from": ["On"], "to": ["Off"], "event": "POWER"},
    ]
}

# Test 3: Branching hierarchy - multiple child paths
BRANCHING_HIERARCHY_SC = {
    "name": "BranchingHierarchy",
    "category": "hierarchy",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Start", "type": 1, "is_initial": True},
            {
                "label": "Phase1",
                "type": 2,
                "children": [
                    {"label": "P1_Init", "type": 1, "is_initial": True},
                    {"label": "P1_Work", "type": 1},
                    {"label": "P1_Done", "type": 1},
                ]
            },
            {
                "label": "Phase2",
                "type": 2,
                "children": [
                    {"label": "P2_Init", "type": 1, "is_initial": True},
                    {"label": "P2_Work", "type": 1},
                ]
            },
            {"label": "Finish", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["Start"], "to": ["Phase1"], "event": "BEGIN"},
        {"from": ["P1_Init"], "to": ["P1_Work"], "event": "WORK"},
        {"from": ["P1_Work"], "to": ["P1_Done"], "event": "COMPLETE"},
        {"from": ["Phase1"], "to": ["Phase2"], "event": "ADVANCE"},
        {"from": ["P2_Init"], "to": ["P2_Work"], "event": "WORK"},
        {"from": ["Phase2"], "to": ["Finish"], "event": "END"},
    ]
}

# Test 4: Flat terminal (control case)
FLAT_TERMINAL_SC = {
    "name": "FlatTerminal",
    "category": "flat",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "A", "type": 1, "is_initial": True},
            {"label": "B", "type": 1},
            {"label": "C", "type": 1},
            {"label": "Terminal", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["A"], "to": ["B"], "event": "NEXT"},
        {"from": ["B"], "to": ["C"], "event": "NEXT"},
        {"from": ["C"], "to": ["Terminal"], "event": "NEXT"},
    ]
}


HIERARCHY_TEST_MACHINES = [
    SIMPLE_HIERARCHY_SC,
    DEEP_HIERARCHY_SC,
    BRANCHING_HIERARCHY_SC,
    FLAT_TERMINAL_SC,
]


# =============================================================================
# Specific Test Cases
# =============================================================================

@dataclass
class HierarchyTestCase:
    """A hierarchy-specific test case."""
    sc_json: dict
    initial: str
    events: List[str]
    expected: Set[str]
    description: str


def generate_hierarchy_test_cases() -> List[HierarchyTestCase]:
    """Generate focused hierarchy test cases."""
    cases = []

    # Simple Hierarchy tests
    cases.extend([
        HierarchyTestCase(
            SIMPLE_HIERARCHY_SC, "Idle", ["START"],
            {"Running"}, "Enter composite -> expect leaf (Running)"
        ),
        HierarchyTestCase(
            SIMPLE_HIERARCHY_SC, "Idle", ["START", "PAUSE"],
            {"Paused"}, "Enter composite, then substate transition"
        ),
        HierarchyTestCase(
            SIMPLE_HIERARCHY_SC, "Idle", ["START", "STOP"],
            {"Done"}, "Exit composite state entirely"
        ),
        HierarchyTestCase(
            SIMPLE_HIERARCHY_SC, "Idle", ["START", "PAUSE", "RESUME"],
            {"Running"}, "Multiple transitions within composite"
        ),
    ])

    # Deep Hierarchy tests
    cases.extend([
        HierarchyTestCase(
            DEEP_HIERARCHY_SC, "Off", ["POWER"],
            {"Waiting"}, "Deep cascade: On -> Ready -> Waiting"
        ),
        HierarchyTestCase(
            DEEP_HIERARCHY_SC, "Off", ["POWER", "START"],
            {"Processing"}, "Cascade then transition within"
        ),
        HierarchyTestCase(
            DEEP_HIERARCHY_SC, "Off", ["POWER", "FAIL"],
            {"Error"}, "Exit nested composite to sibling"
        ),
        HierarchyTestCase(
            DEEP_HIERARCHY_SC, "Off", ["POWER", "FAIL", "RESET"],
            {"Waiting"}, "Re-enter nested composite"
        ),
    ])

    # Branching Hierarchy tests
    cases.extend([
        HierarchyTestCase(
            BRANCHING_HIERARCHY_SC, "Start", ["BEGIN"],
            {"P1_Init"}, "Enter Phase1 -> expect P1_Init"
        ),
        HierarchyTestCase(
            BRANCHING_HIERARCHY_SC, "Start", ["BEGIN", "WORK"],
            {"P1_Work"}, "Transition within Phase1"
        ),
        HierarchyTestCase(
            BRANCHING_HIERARCHY_SC, "Start", ["BEGIN", "ADVANCE"],
            {"P2_Init"}, "Move from Phase1 to Phase2"
        ),
        HierarchyTestCase(
            BRANCHING_HIERARCHY_SC, "Start", ["BEGIN", "WORK", "COMPLETE", "ADVANCE"],
            {"P2_Init"}, "Complete Phase1 substates, advance"
        ),
    ])

    # Flat terminal (control)
    cases.extend([
        HierarchyTestCase(
            FLAT_TERMINAL_SC, "A", ["NEXT"],
            {"B"}, "Simple flat transition"
        ),
        HierarchyTestCase(
            FLAT_TERMINAL_SC, "A", ["NEXT", "NEXT"],
            {"C"}, "Two flat transitions"
        ),
        HierarchyTestCase(
            FLAT_TERMINAL_SC, "A", ["NEXT", "NEXT", "NEXT"],
            {"Terminal"}, "Reach terminal leaf state"
        ),
    ])

    return cases


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class MethodResult:
    """Results for a specific prompting method."""
    method: str
    exact_matches: int
    total: int
    accuracy: float
    avg_time: float


def run_hierarchy_benchmark(
    model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    methods: List[str] = None,
) -> Dict:
    """Run the hierarchy-focused benchmark."""
    if methods is None:
        methods = ["cascade", "viz", "stepwise"]

    print("=" * 70)
    print("HIERARCHY TERMINAL STATE PREDICTION BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")

    test_cases = generate_hierarchy_test_cases()
    print(f"Test cases: {len(test_cases)}")

    predictor = HierarchyPredictor(model_name=model_name)

    results_by_method: Dict[str, List[Tuple[HierarchyTestCase, HierarchyPredictionResult, float]]] = {
        m: [] for m in methods
    }

    for method in methods:
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print("=" * 70)

        for i, tc in enumerate(test_cases):
            t0 = time.time()
            result = predictor.predict(
                tc.sc_json,
                tc.initial,
                tc.events,
                method=method,
            )
            elapsed = time.time() - t0

            # Check accuracy
            correct = result.predicted_states == tc.expected
            status = "OK" if correct else "FAIL"

            print(f"[{i+1:2d}] [{status}] {tc.description}")
            print(f"     Events: {tc.events}")
            print(f"     Expected: {tc.expected} | Got: {result.predicted_states}")
            if not correct:
                print(f"     Raw: {result.raw_output[:80]}...")
            print(f"     Time: {elapsed:.1f}s")

            results_by_method[method].append((tc, result, elapsed))

    # Compute summary
    print("\n" + "=" * 70)
    print("SUMMARY BY METHOD")
    print("=" * 70)

    method_summaries: List[MethodResult] = []
    for method in methods:
        results = results_by_method[method]
        correct = sum(1 for tc, r, _ in results if r.predicted_states == tc.expected)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        avg_time = sum(t for _, _, t in results) / total if total > 0 else 0

        summary = MethodResult(method, correct, total, accuracy, avg_time)
        method_summaries.append(summary)

        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Avg time: {avg_time:.1f}s")

    # Find best method
    best = max(method_summaries, key=lambda x: x.accuracy)
    print(f"\nBest method: {best.method} ({best.accuracy:.1%})")

    # Breakdown by SC type
    print("\n" + "-" * 70)
    print("BREAKDOWN BY SC TYPE (using best method: {best.method})")
    print("-" * 70)

    best_results = results_by_method[best.method]
    by_sc = {}
    for tc, result, _ in best_results:
        sc_name = tc.sc_json.get("name", "Unknown")
        if sc_name not in by_sc:
            by_sc[sc_name] = {"correct": 0, "total": 0}
        by_sc[sc_name]["total"] += 1
        if result.predicted_states == tc.expected:
            by_sc[sc_name]["correct"] += 1

    for sc_name, stats in by_sc.items():
        rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {sc_name}: {rate:.0%} ({stats['correct']}/{stats['total']})")

    return {
        "methods": {m.method: {"accuracy": m.accuracy, "correct": m.exact_matches, "total": m.total}
                    for m in method_summaries},
        "best_method": best.method,
        "best_accuracy": best.accuracy,
        "by_sc": by_sc,
    }


def run_baseline_comparison():
    """Compare baseline (original) vs hierarchy-aware predictor."""
    from .state_predictor import TerminalStatePredictor

    print("=" * 70)
    print("BASELINE vs CASCADE COMPARISON")
    print("=" * 70)

    test_cases = generate_hierarchy_test_cases()

    # Filter to hierarchy-only cases
    hierarchy_cases = [tc for tc in test_cases if tc.sc_json.get("category") == "hierarchy"]
    print(f"Hierarchy test cases: {len(hierarchy_cases)}")

    baseline_predictor = TerminalStatePredictor()
    hierarchy_predictor = HierarchyPredictor()

    baseline_correct = 0
    cascade_correct = 0
    total = len(hierarchy_cases)

    for tc in hierarchy_cases:
        # Baseline
        baseline_result = baseline_predictor.predict(tc.sc_json, tc.initial, tc.events)
        if baseline_result.predicted_states == tc.expected:
            baseline_correct += 1

        # Cascade
        cascade_result = hierarchy_predictor.predict(tc.sc_json, tc.initial, tc.events, method="cascade")
        if cascade_result.predicted_states == tc.expected:
            cascade_correct += 1

    baseline_rate = baseline_correct / total if total > 0 else 0
    cascade_rate = cascade_correct / total if total > 0 else 0

    print(f"\nBaseline: {baseline_rate:.0%} ({baseline_correct}/{total})")
    print(f"Cascade:  {cascade_rate:.0%} ({cascade_correct}/{total})")
    print(f"Improvement: {(cascade_rate - baseline_rate)*100:+.0f}pp")

    return {
        "baseline": baseline_rate,
        "cascade": cascade_rate,
        "improvement": cascade_rate - baseline_rate,
    }


def main():
    """Run full benchmark."""
    # Run baseline comparison first
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline Comparison (Hierarchy cases only)")
    print("=" * 70)
    comparison = run_baseline_comparison()

    # Run full benchmark
    print("\n" + "=" * 70)
    print("PHASE 2: Full Benchmark (All methods)")
    print("=" * 70)
    results = run_hierarchy_benchmark()

    # Print report
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)

    cascade_acc = results["methods"].get("cascade", {}).get("accuracy", 0)
    viz_acc = results["methods"].get("viz", {}).get("accuracy", 0)
    stepwise_acc = results["methods"].get("stepwise", {}).get("accuracy", 0)
    best_acc = results["best_accuracy"]

    print(f"TERMINAL_HIERARCHY baseline={comparison['baseline']*100:.0f}%, cascade={cascade_acc*100:.0f}%, viz={viz_acc*100:.0f}%, best={best_acc*100:.0f}%")


if __name__ == "__main__":
    main()
