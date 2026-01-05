"""
Reachability Prediction Benchmark.

Evaluates LLM reachability prediction against BFS ground truth.

Metrics:
- F1 score for reachability classification
- Precision and recall
- Steps accuracy (when reachable)
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Tuple

from .reachability_predictor import (
    BFSReachabilityAnalyzer,
    LLMReachabilityPredictor,
    ReachabilityResult,
)

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_execution_prediction import (
    CounterMachine,
    BranchingMachine,
    HierarchicalMachine,
)
from experiments.exp_execution_prediction.metrics import reachability_f1


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    sc_json: Dict
    start_config: Set[str]
    target_state: str
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_cases: int = 0
    llm_predictions: List[Tuple[bool, int]] = field(default_factory=list)
    bfs_ground_truth: List[Tuple[bool, int]] = field(default_factory=list)
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    steps_accuracy: float = 0.0
    total_time_ms: float = 0.0


def create_benchmark_cases() -> List[BenchmarkCase]:
    """Create diverse benchmark cases."""
    cases = []

    # === Counter Machine Cases ===

    # 1. Counter: Start -> Done (reachable)
    counter = CounterMachine(target=3)
    cases.append(BenchmarkCase(
        name="counter_start_to_done",
        sc_json=counter.to_json(),
        start_config={"Start"},
        target_state="Done",
        context=counter.initial_context,
        description="Counter: Start to Done (reachable in 4 steps)"
    ))

    # 2. Counter: Counting -> Done (reachable)
    cases.append(BenchmarkCase(
        name="counter_counting_to_done",
        sc_json=counter.to_json(),
        start_config={"Counting"},
        target_state="Done",
        context={"count": 1, "target": 3},
        description="Counter: Counting to Done (reachable in 2 steps)"
    ))

    # 3. Counter: Done -> Start (not reachable)
    cases.append(BenchmarkCase(
        name="counter_done_to_start",
        sc_json=counter.to_json(),
        start_config={"Done"},
        target_state="Start",
        context={"count": 3, "target": 3},
        description="Counter: Done to Start (not reachable)"
    ))

    # === Branching Machine Cases ===

    # 4. Branching: Check -> Final (reachable)
    branching = BranchingMachine(threshold=50)
    cases.append(BenchmarkCase(
        name="branching_check_to_final",
        sc_json=branching.to_json(),
        start_config={"Check"},
        target_state="Final",
        context={"score": 75, "result": None, "bonus": 0, "final_score": 0},
        description="Branching: Check to Final (reachable in 3 steps)"
    ))

    # 5. Branching: Check -> PassPath (reachable with high score)
    cases.append(BenchmarkCase(
        name="branching_check_to_passpath",
        sc_json=branching.to_json(),
        start_config={"Check"},
        target_state="PassPath",
        context={"score": 80, "result": None, "bonus": 0, "final_score": 0},
        description="Branching: Check to PassPath (reachable in 1 step)"
    ))

    # 6. Branching: Check -> PassPath (not reachable with low score)
    cases.append(BenchmarkCase(
        name="branching_check_to_passpath_low",
        sc_json=branching.to_json(),
        start_config={"Check"},
        target_state="PassPath",
        context={"score": 30, "result": None, "bonus": 0, "final_score": 0},
        description="Branching: Check to PassPath (not reachable - score too low)"
    ))

    # 7. Branching: Merge -> Check (not reachable)
    cases.append(BenchmarkCase(
        name="branching_merge_to_check",
        sc_json=branching.to_json(),
        start_config={"Merge"},
        target_state="Check",
        context={"score": 50, "result": "pass", "bonus": 10, "final_score": 0},
        description="Branching: Merge to Check (not reachable)"
    ))

    # === Hierarchical Machine Cases ===

    # 8. Hierarchical: P1_Init -> Final (reachable)
    hier = HierarchicalMachine()
    cases.append(BenchmarkCase(
        name="hier_p1init_to_final",
        sc_json=hier.to_json(),
        start_config={"P1_Init"},
        target_state="Final",
        context=hier.initial_context,
        description="Hierarchical: P1_Init to Final (reachable in 7 steps)"
    ))

    # 9. Hierarchical: P1_Process -> P2_Complete (reachable)
    cases.append(BenchmarkCase(
        name="hier_p1process_to_p2complete",
        sc_json=hier.to_json(),
        start_config={"P1_Process"},
        target_state="P2_Complete",
        context={"phase": 1, "step": 1, "validated": False, "complete": False},
        description="Hierarchical: P1_Process to P2_Complete (reachable)"
    ))

    # 10. Hierarchical: Final -> P1_Init (not reachable)
    cases.append(BenchmarkCase(
        name="hier_final_to_p1init",
        sc_json=hier.to_json(),
        start_config={"Final"},
        target_state="P1_Init",
        context={"phase": 2, "step": 0, "validated": True, "complete": True},
        description="Hierarchical: Final to P1_Init (not reachable)"
    ))

    # === Edge Cases ===

    # 11. Self-loop: already at target
    cases.append(BenchmarkCase(
        name="already_at_target",
        sc_json=counter.to_json(),
        start_config={"Done"},
        target_state="Done",
        context={"count": 3, "target": 3},
        description="Already at target state (0 steps)"
    ))

    # 12. Non-existent state
    cases.append(BenchmarkCase(
        name="nonexistent_target",
        sc_json=counter.to_json(),
        start_config={"Start"},
        target_state="Nonexistent",
        context=counter.initial_context,
        description="Target state doesn't exist (not reachable)"
    ))

    return cases


def run_benchmark(verbose: bool = True) -> BenchmarkResults:
    """
    Run the reachability prediction benchmark.

    Compares LLM predictions to BFS ground truth.
    """
    print("=" * 60)
    print("REACHABILITY PREDICTION BENCHMARK")
    print("=" * 60)
    print()

    # Initialize LLM predictor
    print("Loading model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    llm = LLMReachabilityPredictor("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")

    if not llm.load_model():
        print("ERROR: Failed to load model")
        return BenchmarkResults()

    print("Model loaded successfully")
    print()

    # Get test cases
    cases = create_benchmark_cases()
    print(f"Running {len(cases)} test cases...")
    print()

    # Initialize results
    results = BenchmarkResults()
    start_time = time.time()

    for i, case in enumerate(cases):
        if verbose:
            print(f"[{i+1}/{len(cases)}] {case.name}")
            print(f"  {case.description}")

        # Get BFS ground truth
        bfs = BFSReachabilityAnalyzer(case.sc_json)
        bfs_result = bfs.analyze(
            case.target_state,
            case.start_config,
            case.context,
        )

        # Get LLM prediction
        llm_result = llm.predict(
            case.sc_json,
            case.start_config,
            case.target_state,
            case.context,
        )

        # Record results
        results.total_cases += 1
        results.bfs_ground_truth.append((bfs_result.is_reachable, bfs_result.min_steps))
        results.llm_predictions.append((llm_result.is_reachable, llm_result.min_steps))

        if verbose:
            bfs_str = f"steps={bfs_result.min_steps}" if bfs_result.is_reachable else "NOT REACHABLE"
            llm_str = f"steps={llm_result.min_steps}" if llm_result.is_reachable else "NOT REACHABLE"

            status = "MATCH" if bfs_result.is_reachable == llm_result.is_reachable else "MISMATCH"

            print(f"  BFS: {bfs_str}")
            print(f"  LLM: {llm_str}")
            print(f"  Status: {status}")
            print(f"  Time: BFS={bfs_result.analysis_time_ms:.1f}ms, LLM={llm_result.analysis_time_ms:.1f}ms")
            print()

    results.total_time_ms = (time.time() - start_time) * 1000

    # Compute metrics using reachability_f1
    metric_result = reachability_f1(results.llm_predictions, results.bfs_ground_truth)

    results.f1 = metric_result.details["f1"]
    results.precision = metric_result.details["precision"]
    results.recall = metric_result.details["recall"]
    results.steps_accuracy = metric_result.details["steps_accuracy"]

    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"Total cases: {results.total_cases}")
    print(f"F1 Score: {results.f1 * 100:.1f}%")
    print(f"Precision: {results.precision * 100:.1f}%")
    print(f"Recall: {results.recall * 100:.1f}%")
    print(f"Steps Accuracy: {results.steps_accuracy * 100:.1f}%")
    print()
    print(f"Confusion Matrix:")
    conf = metric_result.details["confusion"]
    print(f"  TP={conf['tp']}, FP={conf['fp']}, TN={conf['tn']}, FN={conf['fn']}")
    print()
    print(f"Total time: {results.total_time_ms:.1f}ms")
    print(f"Avg per case: {results.total_time_ms / max(results.total_cases, 1):.1f}ms")

    return results


def format_report(results: BenchmarkResults) -> str:
    """Format results for orchestrator report."""
    return (
        f"REACHABILITY f1={results.f1 * 100:.0f}%, "
        f"precision={results.precision * 100:.0f}%, "
        f"recall={results.recall * 100:.0f}%, "
        f"steps_acc={results.steps_accuracy * 100:.0f}%"
    )


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
    print()
    print("=" * 60)
    print("REPORT:")
    print(format_report(results))
