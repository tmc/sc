"""
FIM Statecharts Benchmark - Real MLX evaluation.

Tests Fill-in-Middle generation on statechart holes using
mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

from .fim_generator import FIMGenerator, HoleType, FIMResult, StructureContext


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    prefix: str
    suffix: str
    hole_type: HoleType
    context: StructureContext
    description: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_cases: int = 0
    valid_count: int = 0
    by_hole_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    results: List[FIMResult] = field(default_factory=list)
    total_time_ms: float = 0.0

    @property
    def valid_pct(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return (self.valid_count / self.total_cases) * 100

    def type_pct(self, hole_type: str) -> float:
        if hole_type not in self.by_hole_type:
            return 0.0
        stats = self.by_hole_type[hole_type]
        if stats['total'] == 0:
            return 0.0
        return (stats['valid'] / stats['total']) * 100


def create_benchmark_cases() -> List[BenchmarkCase]:
    """Create comprehensive benchmark cases for all hole types."""
    cases = []

    # ========== STATE HOLES ==========

    # Simple state insertion
    cases.append(BenchmarkCase(
        name="state_simple",
        prefix='{"root_state": {"label": "__root__", "type": 2, "children": [',
        suffix=', {"label": "Off", "type": 1, "is_initial": false}]}}',
        hole_type=HoleType.STATE,
        context=StructureContext(
            existing_states=["__root__", "Off"],
            parent_state="__root__",
            sibling_states=["Off"]
        ),
        description="Insert initial On state"
    ))

    # State with specific name needed
    cases.append(BenchmarkCase(
        name="state_loading",
        prefix='{"root_state": {"label": "App", "type": 2, "children": [{"label": "Idle", "type": 1, "is_initial": true}, ',
        suffix=', {"label": "Error", "type": 1}]}}',
        hole_type=HoleType.STATE,
        context=StructureContext(
            existing_states=["App", "Idle", "Error"],
            parent_state="App",
            sibling_states=["Idle", "Error"]
        ),
        description="Insert Loading state"
    ))

    # Nested state
    cases.append(BenchmarkCase(
        name="state_nested",
        prefix='{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Active", "type": 2, "children": [',
        suffix=']}]}}',
        hole_type=HoleType.STATE,
        context=StructureContext(
            existing_states=["__root__", "Active"],
            parent_state="Active",
            sibling_states=[]
        ),
        description="Insert first child of Active composite state"
    ))

    # ========== TRANSITION HOLES ==========

    # Simple transition
    cases.append(BenchmarkCase(
        name="trans_simple",
        prefix='{"transitions": [',
        suffix=']}',
        hole_type=HoleType.TRANSITION,
        context=StructureContext(
            existing_states=["On", "Off"],
            existing_events=["TOGGLE"]
        ),
        description="Insert On->Off transition"
    ))

    # Second transition in list
    cases.append(BenchmarkCase(
        name="trans_second",
        prefix='{"transitions": [{"from": ["Off"], "to": ["On"], "event": "TURN_ON"}, ',
        suffix=']}',
        hole_type=HoleType.TRANSITION,
        context=StructureContext(
            existing_states=["On", "Off"],
            existing_events=["TURN_ON", "TURN_OFF"]
        ),
        description="Insert On->Off TURN_OFF transition"
    ))

    # Transition with guard context
    cases.append(BenchmarkCase(
        name="trans_guarded",
        prefix='{"transitions": [{"from": ["Idle"], "to": ["Running"], "event": "START", "guard": {"expression": "ready"}}, ',
        suffix=']}',
        hole_type=HoleType.TRANSITION,
        context=StructureContext(
            existing_states=["Idle", "Running", "Stopped"],
            existing_events=["START", "STOP", "PAUSE"]
        ),
        description="Insert Running->Stopped transition"
    ))

    # ========== GUARD HOLES ==========

    # Simple guard
    cases.append(BenchmarkCase(
        name="guard_simple",
        prefix='{"from": ["Idle"], "to": ["Active"], "event": "START", "guard": {"expression": "',
        suffix='"}}',
        hole_type=HoleType.GUARD,
        context=StructureContext(
            existing_states=["Idle", "Active"]
        ),
        description="Insert guard condition"
    ))

    # Numeric guard
    cases.append(BenchmarkCase(
        name="guard_numeric",
        prefix='{"from": ["Counting"], "to": ["Done"], "event": "INCREMENT", "guard": {"expression": "',
        suffix='"}}',
        hole_type=HoleType.GUARD,
        context=StructureContext(
            existing_states=["Counting", "Done"]
        ),
        description="Insert count comparison guard"
    ))

    # Boolean guard
    cases.append(BenchmarkCase(
        name="guard_boolean",
        prefix='{"from": ["Waiting"], "to": ["Processing"], "event": "SUBMIT", "guard": {"expression": "',
        suffix='"}}',
        hole_type=HoleType.GUARD,
        context=StructureContext(
            existing_states=["Waiting", "Processing"]
        ),
        description="Insert boolean flag guard"
    ))

    # ========== ACTION HOLES ==========

    # Entry action
    cases.append(BenchmarkCase(
        name="action_entry",
        prefix='{"label": "Loading", "type": 1, "entry_actions": [',
        suffix=']}',
        hole_type=HoleType.ACTION,
        context=StructureContext(
            existing_states=["Loading", "Ready", "Error"]
        ),
        description="Insert entry action for Loading state"
    ))

    # Exit action
    cases.append(BenchmarkCase(
        name="action_exit",
        prefix='{"label": "Active", "type": 1, "exit_actions": [',
        suffix=']}',
        hole_type=HoleType.ACTION,
        context=StructureContext(
            existing_states=["Active", "Inactive"]
        ),
        description="Insert exit action for Active state"
    ))

    # Second action
    cases.append(BenchmarkCase(
        name="action_second",
        prefix='{"label": "Processing", "type": 1, "entry_actions": [{"name": "startTimer"}, ',
        suffix=']}',
        hole_type=HoleType.ACTION,
        context=StructureContext(
            existing_states=["Processing", "Complete"]
        ),
        description="Insert second entry action"
    ))

    return cases


def run_benchmark(verbose: bool = True) -> BenchmarkResults:
    """
    Run the FIM statecharts benchmark with real MLX inference.

    Returns:
        BenchmarkResults with validation statistics
    """
    print("=" * 60)
    print("FIM STATECHARTS BENCHMARK")
    print("=" * 60)
    print()

    # Initialize generator
    print("Loading model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    generator = FIMGenerator("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")

    if not generator.load_model():
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
    results.by_hole_type = {
        "STATE": {"total": 0, "valid": 0},
        "TRANSITION": {"total": 0, "valid": 0},
        "GUARD": {"total": 0, "valid": 0},
        "ACTION": {"total": 0, "valid": 0},
    }

    # Run each case
    start_time = time.time()

    for i, case in enumerate(cases):
        if verbose:
            print(f"[{i+1}/{len(cases)}] {case.name}: {case.description}")

        result = generator.generate_fim(
            prefix=case.prefix,
            suffix=case.suffix,
            hole_type=case.hole_type,
            context=case.context,
            max_tokens=80
        )

        results.results.append(result)
        results.total_cases += 1

        # Update stats
        type_key = case.hole_type.name
        results.by_hole_type[type_key]["total"] += 1

        if result.is_valid:
            results.valid_count += 1
            results.by_hole_type[type_key]["valid"] += 1
            status = "VALID"
        else:
            status = f"INVALID: {result.error}"

        if verbose:
            print(f"  Generated: {result.generated[:60]}...")
            print(f"  Status: {status}")
            print(f"  Time: {result.generation_time_ms:.1f}ms")
            print()

    results.total_time_ms = (time.time() - start_time) * 1000

    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"Overall: {results.valid_count}/{results.total_cases} valid ({results.valid_pct:.1f}%)")
    print()
    print("By hole type:")
    for hole_type in ["STATE", "TRANSITION", "GUARD", "ACTION"]:
        stats = results.by_hole_type[hole_type]
        pct = results.type_pct(hole_type)
        print(f"  {hole_type}: {stats['valid']}/{stats['total']} ({pct:.1f}%)")
    print()
    print(f"Total time: {results.total_time_ms:.1f}ms")
    print(f"Avg per case: {results.total_time_ms / max(results.total_cases, 1):.1f}ms")

    return results


def format_report(results: BenchmarkResults) -> str:
    """Format results for orchestrator report."""
    state_pct = results.type_pct("STATE")
    trans_pct = results.type_pct("TRANSITION")
    guard_pct = results.type_pct("GUARD")
    action_pct = results.type_pct("ACTION")

    return (
        f"FIM_STATECHARTS valid_insertion={results.valid_pct:.0f}%, "
        f"by_hole_type=[STATE:{state_pct:.0f}%, TRANSITION:{trans_pct:.0f}%, "
        f"GUARD:{guard_pct:.0f}%, ACTION:{action_pct:.0f}%]"
    )


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
    print()
    print("=" * 60)
    print("REPORT:")
    print(format_report(results))
