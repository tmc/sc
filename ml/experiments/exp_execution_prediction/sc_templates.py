"""
Statechart Templates for Execution Prediction.

Complex templates (10+ states) with context manipulation via Starlark.
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SCTemplate:
    """Base template for statecharts."""
    name: str
    states: List[Dict]
    transitions: List[Dict]
    initial_context: Dict[str, Any]

    def to_json(self) -> Dict:
        """Convert to SC JSON format."""
        return {
            "name": self.name,
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": self.states,
            },
            "transitions": self.transitions,
            "initial_context": self.initial_context,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2)


def CounterMachine(target: int = 5) -> SCTemplate:
    """
    Counter machine: counts from 0 to target.

    States: Start -> Counting -> Done (terminal)
    Context: {count: 0, target: N}
    Actions: count = count + 1 on TICK
    Terminal: count >= target
    """
    return SCTemplate(
        name="CounterMachine",
        states=[
            {"label": "Start", "type": 1, "is_initial": True},
            {"label": "Counting", "type": 1},
            {"label": "Done", "type": 1, "is_final": True},
        ],
        transitions=[
            {
                "from": ["Start"],
                "to": ["Counting"],
                "event": "BEGIN",
                "actions": [{"expression": "count = 0"}],
            },
            {
                "from": ["Counting"],
                "to": ["Counting"],
                "event": "TICK",
                "guard": {"expression": f"count < {target}"},
                "actions": [{"expression": "count = count + 1"}],
            },
            {
                "from": ["Counting"],
                "to": ["Done"],
                "event": "TICK",
                "guard": {"expression": f"count >= {target}"},
            },
        ],
        initial_context={"count": 0, "target": target},
    )


def AccumulatorMachine(values: List[int] = None) -> SCTemplate:
    """
    Accumulator: sums a list of values.

    States: Idle -> Processing -> Complete
    Context: {sum: 0, index: 0, values: [...]}
    Actions: sum = sum + values[index]; index = index + 1
    Terminal: index >= len(values)
    """
    if values is None:
        values = [random.randint(1, 10) for _ in range(5)]

    n = len(values)

    return SCTemplate(
        name="AccumulatorMachine",
        states=[
            {"label": "Idle", "type": 1, "is_initial": True},
            {"label": "Processing", "type": 1},
            {"label": "Complete", "type": 1, "is_final": True},
        ],
        transitions=[
            {
                "from": ["Idle"],
                "to": ["Processing"],
                "event": "START",
                "actions": [{"expression": "sum = 0"}, {"expression": "index = 0"}],
            },
            {
                "from": ["Processing"],
                "to": ["Processing"],
                "event": "NEXT",
                "guard": {"expression": f"index < {n}"},
                "actions": [
                    {"expression": "sum = sum + values[index]"},
                    {"expression": "index = index + 1"},
                ],
            },
            {
                "from": ["Processing"],
                "to": ["Complete"],
                "event": "NEXT",
                "guard": {"expression": f"index >= {n}"},
            },
        ],
        initial_context={"sum": 0, "index": 0, "values": values},
    )


def BranchingMachine(threshold: int = 50) -> SCTemplate:
    """
    Branching machine: takes different paths based on guards.

    States: Check -> (PassPath/FailPath) -> Merge -> Final
    Context: {score: random, result: None}
    Guards: score >= threshold -> Pass, else -> Fail
    """
    return SCTemplate(
        name="BranchingMachine",
        states=[
            {"label": "Check", "type": 1, "is_initial": True},
            {"label": "PassPath", "type": 1},
            {"label": "FailPath", "type": 1},
            {"label": "Merge", "type": 1},
            {"label": "Final", "type": 1, "is_final": True},
        ],
        transitions=[
            {
                "from": ["Check"],
                "to": ["PassPath"],
                "event": "EVALUATE",
                "guard": {"expression": f"score >= {threshold}"},
                "actions": [{"expression": "result = 'pass'"}],
            },
            {
                "from": ["Check"],
                "to": ["FailPath"],
                "event": "EVALUATE",
                "guard": {"expression": f"score < {threshold}"},
                "actions": [{"expression": "result = 'fail'"}],
            },
            {
                "from": ["PassPath"],
                "to": ["Merge"],
                "event": "CONTINUE",
                "actions": [{"expression": "bonus = 10"}],
            },
            {
                "from": ["FailPath"],
                "to": ["Merge"],
                "event": "CONTINUE",
                "actions": [{"expression": "bonus = 0"}],
            },
            {
                "from": ["Merge"],
                "to": ["Final"],
                "event": "FINISH",
                "actions": [{"expression": "final_score = score + bonus"}],
            },
        ],
        initial_context={"score": random.randint(0, 100), "result": None, "bonus": 0, "final_score": 0},
    )


def ParallelCounterMachine(target_a: int = 3, target_b: int = 4) -> SCTemplate:
    """
    Parallel counter: two independent counters in parallel regions.

    Regions: CounterA (counts to target_a), CounterB (counts to target_b)
    Context: {a: 0, b: 0}
    Terminal: both regions complete
    """
    return SCTemplate(
        name="ParallelCounterMachine",
        states=[
            {
                "label": "Parallel",
                "type": 3,  # AND-state (parallel)
                "is_initial": True,
                "children": [
                    {
                        "label": "RegionA",
                        "type": 2,
                        "children": [
                            {"label": "CountingA", "type": 1, "is_initial": True},
                            {"label": "DoneA", "type": 1, "is_final": True},
                        ],
                    },
                    {
                        "label": "RegionB",
                        "type": 2,
                        "children": [
                            {"label": "CountingB", "type": 1, "is_initial": True},
                            {"label": "DoneB", "type": 1, "is_final": True},
                        ],
                    },
                ],
            },
        ],
        transitions=[
            {
                "from": ["CountingA"],
                "to": ["CountingA"],
                "event": "TICK_A",
                "guard": {"expression": f"a < {target_a}"},
                "actions": [{"expression": "a = a + 1"}],
            },
            {
                "from": ["CountingA"],
                "to": ["DoneA"],
                "event": "TICK_A",
                "guard": {"expression": f"a >= {target_a}"},
            },
            {
                "from": ["CountingB"],
                "to": ["CountingB"],
                "event": "TICK_B",
                "guard": {"expression": f"b < {target_b}"},
                "actions": [{"expression": "b = b + 1"}],
            },
            {
                "from": ["CountingB"],
                "to": ["DoneB"],
                "event": "TICK_B",
                "guard": {"expression": f"b >= {target_b}"},
            },
        ],
        initial_context={"a": 0, "b": 0, "target_a": target_a, "target_b": target_b},
    )


def HierarchicalMachine() -> SCTemplate:
    """
    Hierarchical machine: nested composite states (10+ states).

    Structure:
    - Main (OR)
      - Phase1 (OR)
        - P1_Init
        - P1_Process
        - P1_Done
      - Phase2 (OR)
        - P2_Init
        - P2_Validate
        - P2_Complete
      - Final

    Context: {phase: 0, step: 0, validated: False}
    """
    return SCTemplate(
        name="HierarchicalMachine",
        states=[
            {
                "label": "Phase1",
                "type": 2,
                "is_initial": True,
                "children": [
                    {"label": "P1_Init", "type": 1, "is_initial": True},
                    {"label": "P1_Process", "type": 1},
                    {"label": "P1_Done", "type": 1},
                ],
            },
            {
                "label": "Phase2",
                "type": 2,
                "children": [
                    {"label": "P2_Init", "type": 1, "is_initial": True},
                    {"label": "P2_Validate", "type": 1},
                    {"label": "P2_Complete", "type": 1},
                ],
            },
            {"label": "Final", "type": 1, "is_final": True},
        ],
        transitions=[
            # Phase 1 internal transitions
            {
                "from": ["P1_Init"],
                "to": ["P1_Process"],
                "event": "START",
                "actions": [{"expression": "phase = 1"}, {"expression": "step = 0"}],
            },
            {
                "from": ["P1_Process"],
                "to": ["P1_Process"],
                "event": "WORK",
                "guard": {"expression": "step < 3"},
                "actions": [{"expression": "step = step + 1"}],
            },
            {
                "from": ["P1_Process"],
                "to": ["P1_Done"],
                "event": "WORK",
                "guard": {"expression": "step >= 3"},
            },
            # Phase 1 -> Phase 2 transition
            {
                "from": ["P1_Done"],
                "to": ["Phase2"],
                "event": "ADVANCE",
                "actions": [{"expression": "phase = 2"}, {"expression": "step = 0"}],
            },
            # Phase 2 internal transitions
            {
                "from": ["P2_Init"],
                "to": ["P2_Validate"],
                "event": "BEGIN",
            },
            {
                "from": ["P2_Validate"],
                "to": ["P2_Complete"],
                "event": "VALIDATE",
                "actions": [{"expression": "validated = True"}],
            },
            # Phase 2 -> Final
            {
                "from": ["P2_Complete"],
                "to": ["Final"],
                "event": "FINISH",
                "actions": [{"expression": "complete = True"}],
            },
        ],
        initial_context={"phase": 0, "step": 0, "validated": False, "complete": False},
    )


def generate_random_sc(
    min_states: int = 10,
    max_states: int = 15,
    complexity: str = "high",
) -> SCTemplate:
    """Generate a random complex statechart."""
    templates = [
        lambda: CounterMachine(target=random.randint(3, 10)),
        lambda: AccumulatorMachine(values=[random.randint(1, 20) for _ in range(random.randint(3, 8))]),
        lambda: BranchingMachine(threshold=random.randint(30, 70)),
        lambda: ParallelCounterMachine(target_a=random.randint(2, 5), target_b=random.randint(2, 5)),
        lambda: HierarchicalMachine(),
    ]

    if complexity == "high":
        # Prefer hierarchical and parallel
        weights = [0.1, 0.1, 0.2, 0.3, 0.3]
    else:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    template_fn = random.choices(templates, weights=weights)[0]
    return template_fn()


if __name__ == "__main__":
    print("SC Templates Test")
    print("=" * 60)

    templates = [
        ("Counter", CounterMachine(target=5)),
        ("Accumulator", AccumulatorMachine([1, 2, 3, 4, 5])),
        ("Branching", BranchingMachine(threshold=50)),
        ("Parallel", ParallelCounterMachine(3, 4)),
        ("Hierarchical", HierarchicalMachine()),
    ]

    for name, template in templates:
        print(f"\n{name}:")
        print(f"  States: {len(template.states)}")
        print(f"  Transitions: {len(template.transitions)}")
        print(f"  Initial context: {template.initial_context}")
