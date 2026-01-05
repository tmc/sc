"""
Path Length Predictor: Predict steps until terminal state.

Given a statechart and initial context, predict how many steps
are needed to reach a terminal (final) state.

Variations:
- Fixed path: Deterministic, same steps every time
- Variable path: Depends on guards and context values
- Unbounded: May not terminate (predict "infinite")
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Shared infrastructure
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_execution_prediction import (
    CounterMachine,
    AccumulatorMachine,
    BranchingMachine,
    ParallelCounterMachine,
    HierarchicalMachine,
    TraceExecutor,
    execute_to_completion,
)


@dataclass
class PathLengthExample:
    """A path length prediction example."""
    name: str
    sc_json: Dict
    initial_context: Dict[str, Any]
    events: List[str]  # Events to drive execution
    expected_length: int  # Expected steps to terminal
    path_type: str  # "fixed", "variable", "unbounded"


def generate_fixed_path_examples() -> List[PathLengthExample]:
    """Generate examples with deterministic path lengths."""
    examples = []

    # Counter machines with different targets
    for target in [3, 5, 7]:
        counter = CounterMachine(target=target)
        # Fixed: BEGIN (1) + target loop TICKs + exit TICK (1) = target + 2
        events = ["BEGIN"] + ["TICK"] * (target + 1)

        examples.append(PathLengthExample(
            name=f"counter_target_{target}",
            sc_json=counter.to_json(),
            initial_context=counter.initial_context,
            events=events,
            expected_length=target + 2,  # BEGIN + target loops + exit
            path_type="fixed",
        ))

    # Accumulator with fixed values
    for n in [3, 5]:
        values = list(range(1, n + 1))  # [1,2,3] or [1,2,3,4,5]
        acc = AccumulatorMachine(values=values)
        # Fixed: START + n NEXTs + 1 final NEXT
        events = ["START"] + ["NEXT"] * (n + 1)

        examples.append(PathLengthExample(
            name=f"accumulator_n_{n}",
            sc_json=acc.to_json(),
            initial_context=acc.initial_context,
            events=events,
            expected_length=n + 1,  # START + n-1 loops + 1 to Complete
            path_type="fixed",
        ))

    return examples


def generate_variable_path_examples() -> List[PathLengthExample]:
    """Generate examples where path length depends on context."""
    examples = []

    # Branching machine - path depends on score
    for score, threshold in [(75, 50), (25, 50), (60, 70)]:
        branch = BranchingMachine(threshold=threshold)
        branch.initial_context["score"] = score

        # Both paths have same length: EVALUATE -> CONTINUE -> FINISH
        events = ["EVALUATE", "CONTINUE", "FINISH"]

        examples.append(PathLengthExample(
            name=f"branch_score_{score}_thresh_{threshold}",
            sc_json=branch.to_json(),
            initial_context=branch.initial_context,
            events=events,
            expected_length=3,  # Check->Pass/Fail, Pass/Fail->Merge, Merge->Final
            path_type="variable",
        ))

    # Hierarchical machine with variable work steps
    for steps in [3, 4]:
        hier = HierarchicalMachine()
        # START + steps WORKs + ADVANCE + BEGIN + VALIDATE + FINISH
        events = ["START"] + ["WORK"] * (steps + 1) + ["ADVANCE", "BEGIN", "VALIDATE", "FINISH"]

        examples.append(PathLengthExample(
            name=f"hierarchical_steps_{steps}",
            sc_json=hier.to_json(),
            initial_context=hier.initial_context,
            events=events,
            expected_length=steps + 5,  # START + steps-1 loops + 1 P1_Done + ADVANCE + BEGIN + VALIDATE + FINISH
            path_type="variable",
        ))

    return examples


def generate_unbounded_examples() -> List[PathLengthExample]:
    """Generate examples with NO terminal state (truly unbounded)."""
    examples = []

    # Machine with no terminal state at all
    no_terminal_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1, "is_initial": True},
                {"label": "B", "type": 1},
                {"label": "C", "type": 1},
            ],
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"},
            {"from": ["B"], "to": ["C"], "event": "GO"},
            {"from": ["C"], "to": ["A"], "event": "GO"},  # Cycle back
        ],
    }

    examples.append(PathLengthExample(
        name="no_terminal_cycle",
        sc_json=no_terminal_sc,
        initial_context={},
        events=["GO"] * 10,
        expected_length=-1,  # -1 means "infinite"
        path_type="unbounded",
    ))

    # Self-loop with no exit
    infinite_loop_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Loop", "type": 1, "is_initial": True},
            ],
        },
        "transitions": [
            {"from": ["Loop"], "to": ["Loop"], "event": "TICK"},
        ],
    }

    examples.append(PathLengthExample(
        name="infinite_loop",
        sc_json=infinite_loop_sc,
        initial_context={},
        events=["TICK"] * 20,
        expected_length=-1,  # Truly unbounded - no terminal exists
        path_type="unbounded",
    ))

    return examples


def get_all_examples() -> List[PathLengthExample]:
    """Get all path length prediction examples."""
    return (
        generate_fixed_path_examples() +
        generate_variable_path_examples() +
        generate_unbounded_examples()
    )


def compute_actual_path_length(
    sc_json: Dict,
    events: List[str],
    initial_context: Optional[Dict[str, Any]] = None,
) -> Tuple[int, bool]:
    """
    Compute actual path length by executing the statechart.

    Returns: (path_length, is_terminal)
    """
    result = execute_to_completion(sc_json, events, initial_context)
    return result.path_length, result.is_terminal


def create_path_length_prompt(
    sc_json: Dict,
    initial_context: Dict[str, Any],
    events: List[str],
) -> str:
    """Create prompt for path length prediction."""
    sc_str = json.dumps(sc_json, separators=(',', ':'))
    ctx_str = json.dumps(initial_context, separators=(',', ':'))
    events_str = ", ".join(events[:5])  # Show first 5 events
    if len(events) > 5:
        events_str += f", ... ({len(events)} total)"

    return f"""Given a statechart and initial context, predict the number of steps to reach a terminal (final) state.

Statechart:
{sc_str}

Initial context: {ctx_str}
Events available: [{events_str}]

Analyze the statechart structure, transitions, and guards to determine the path length.
If the statechart may not terminate or path is very long (>50 steps), answer "infinite".

Path length (integer or "infinite"):"""


def parse_path_length_response(response: str) -> int:
    """
    Parse model response to extract path length.

    Returns: integer path length, or -1 for "infinite"
    """
    response = response.strip().lower()

    # Check for infinite/unbounded
    if any(word in response for word in ["infinite", "unbounded", "never", "forever", "indefinite"]):
        return -1

    # Extract first number
    match = re.search(r'\b(\d+)\b', response)
    if match:
        return int(match.group(1))

    return -1  # Default to infinite if can't parse


def format_examples_for_few_shot(examples: List[PathLengthExample], n: int = 3) -> str:
    """Format examples for few-shot prompting."""
    lines = []

    for ex in examples[:n]:
        sc_str = json.dumps(ex.sc_json, separators=(',', ':'))
        ctx_str = json.dumps(ex.initial_context, separators=(',', ':'))

        # Compute actual length
        actual_len, is_terminal = compute_actual_path_length(
            ex.sc_json, ex.events, ex.initial_context
        )

        answer = str(actual_len) if actual_len >= 0 else "infinite"

        lines.append(f"Example ({ex.path_type} path):")
        lines.append(f"SC: {sc_str[:200]}...")
        lines.append(f"Context: {ctx_str}")
        lines.append(f"Answer: {answer}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Path Length Predictor Test")
    print("=" * 60)

    # Test each category
    for category, examples in [
        ("Fixed Path", generate_fixed_path_examples()),
        ("Variable Path", generate_variable_path_examples()),
        ("Unbounded", generate_unbounded_examples()),
    ]:
        print(f"\n{category}:")
        for ex in examples[:2]:  # Show first 2
            actual_len, is_terminal = compute_actual_path_length(
                ex.sc_json, ex.events, ex.initial_context
            )
            print(f"  {ex.name}:")
            print(f"    Expected: {ex.expected_length}, Actual: {actual_len}")
            print(f"    Terminal: {is_terminal}")
