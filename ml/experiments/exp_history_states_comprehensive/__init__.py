"""
exp_history_states_comprehensive: Test shallow (H) and deep (H*) history states.

History Semantics:
- SHALLOW (H): Remembers ONLY immediate child, nested goes to default
- DEEP (H*): Remembers FULL configuration including all nested states
- NONE: Always enter default substate

Reference: proto/statecharts/v1/statecharts.proto - HistoryType, State.history_type
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional, Tuple
from enum import Enum
import json


class HistoryType(Enum):
    """History type for composite states."""
    NONE = 0
    SHALLOW = 1  # H
    DEEP = 2     # H*


@dataclass
class State:
    """State in a statechart."""
    name: str
    is_default: bool = False
    children: List["State"] = field(default_factory=list)
    is_parallel: bool = False  # AND-state vs OR-state

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_default_child(self) -> Optional["State"]:
        for c in self.children:
            if c.is_default:
                return c
        return self.children[0] if self.children else None


@dataclass
class HistoryTestCase:
    """Test case for history state behavior."""
    name: str
    category: str  # "shallow", "deep", "no_prior", "parallel", "nested"
    statechart: State  # Root state
    history_type: HistoryType
    # Scenario: sequence of (action, expected_config)
    scenario: List[Tuple[str, Set[str]]]
    description: str


@dataclass
class HistoryQuestion:
    """A question about history state behavior."""
    statechart_desc: str  # ASCII description of SC
    history_type: str  # "shallow" or "deep"
    active_before_exit: Set[str]  # Configuration before exiting
    exit_to: str  # State exited to
    re_enter_via: str  # Re-entry point
    expected_config: Set[str]  # Expected configuration after re-entry


# =============================================================================
# Test Cases
# =============================================================================

def build_power_sc() -> State:
    """Power state machine: On{Low, Medium, High}, Off"""
    return State("Power", children=[
        State("On", children=[
            State("Low", is_default=True),
            State("Medium"),
            State("High"),
        ]),
        State("Off", is_default=True),
    ])


def build_nested_sc() -> State:
    """3-level nested: A{B{C,D}, E{F,G}}"""
    return State("A", children=[
        State("B", is_default=True, children=[
            State("C", is_default=True),
            State("D"),
        ]),
        State("E", children=[
            State("F", is_default=True),
            State("G"),
        ]),
    ])


def build_game_sc() -> State:
    """Parallel regions: Game{Movement{Walk,Run}, Combat{Idle,Fight}}"""
    return State("Game", is_parallel=True, children=[
        State("Movement", children=[
            State("Walking", is_default=True),
            State("Running"),
        ]),
        State("Combat", children=[
            State("Idle", is_default=True),
            State("Fighting"),
        ]),
    ])


def build_deep_nested_sc() -> State:
    """4-level: Root{L1{L2{L3{A,B}}}}"""
    return State("Root", children=[
        State("L1", is_default=True, children=[
            State("L2", is_default=True, children=[
                State("L3", is_default=True, children=[
                    State("A", is_default=True),
                    State("B"),
                ]),
            ]),
        ]),
        State("Outside"),
    ])


# =============================================================================
# Test Case Definitions
# =============================================================================

TC1_BASIC_SHALLOW = HistoryQuestion(
    statechart_desc="""
Power [OR]
├── On [OR]
│   ├── Low (default)
│   ├── Medium
│   └── High
└── Off (default)
""",
    history_type="shallow",
    active_before_exit={"Power", "On", "High"},
    exit_to="Off",
    re_enter_via="On (via H)",
    expected_config={"Power", "On", "Low"},  # Shallow: default child, not High!
)

TC2_BASIC_DEEP = HistoryQuestion(
    statechart_desc="""
Power [OR]
├── On [OR]
│   ├── Low (default)
│   ├── Medium
│   └── High
└── Off (default)
""",
    history_type="deep",
    active_before_exit={"Power", "On", "High"},
    exit_to="Off",
    re_enter_via="On (via H*)",
    expected_config={"Power", "On", "High"},  # Deep: exact restore
)

TC3_SHALLOW_VS_DEEP_NESTED = HistoryQuestion(
    statechart_desc="""
A [OR]
├── B [OR] (default)
│   ├── C (default)
│   └── D
└── E [OR]
    ├── F (default)
    └── G
""",
    history_type="shallow",
    active_before_exit={"A", "B", "D"},  # Was in B.D
    exit_to="E",  # Exit to sibling
    re_enter_via="B (via H)",
    expected_config={"A", "B", "C"},  # Shallow: B restored, but C (default) not D
)

TC4_NO_PRIOR_HISTORY = HistoryQuestion(
    statechart_desc="""
Power [OR]
├── On [OR]
│   ├── Low (default)
│   ├── Medium
│   └── High
└── Off (default)
""",
    history_type="deep",
    active_before_exit=set(),  # No prior entry!
    exit_to="Off",
    re_enter_via="On (via H*, first time)",
    expected_config={"Power", "On", "Low"},  # No history -> use default
)

TC5_MULTIPLE_CYCLES = HistoryQuestion(
    statechart_desc="""
Power [OR]
├── On [OR]
│   ├── Low (default)
│   ├── Medium
│   └── High
└── Off

History after multiple cycles:
1. Enter On.Low
2. Transition to On.High
3. Exit to Off (history saves High)
4. Re-enter On via H* (restore High)
5. Transition to On.Medium
6. Exit to Off (history now saves Medium)
7. Re-enter On via H*
""",
    history_type="deep",
    active_before_exit={"Power", "On", "Medium"},  # Last was Medium
    exit_to="Off",
    re_enter_via="On (via H*)",
    expected_config={"Power", "On", "Medium"},  # Most recent history
)

TC6_DEEP_NESTING = HistoryQuestion(
    statechart_desc="""
Root [OR]
├── L1 [OR] (default)
│   └── L2 [OR] (default)
│       └── L3 [OR] (default)
│           ├── A (default)
│           └── B
└── Outside
""",
    history_type="deep",
    active_before_exit={"Root", "L1", "L2", "L3", "B"},
    exit_to="Outside",
    re_enter_via="L1 (via H*)",
    expected_config={"Root", "L1", "L2", "L3", "B"},  # Deep: full restore
)

TC7_PARALLEL_HISTORY = HistoryQuestion(
    statechart_desc="""
Game [PARALLEL]
├── Movement [OR]
│   ├── Walking (default)
│   └── Running
└── Combat [OR]
    ├── Idle (default)
    └── Fighting

Note: Parallel state - both regions active simultaneously
""",
    history_type="deep",
    active_before_exit={"Game", "Movement", "Running", "Combat", "Fighting"},
    exit_to="Paused (outside Game)",
    re_enter_via="Game (via H*)",
    expected_config={"Game", "Movement", "Running", "Combat", "Fighting"},
)

TC8_SHALLOW_PARALLEL = HistoryQuestion(
    statechart_desc="""
Game [PARALLEL]
├── Movement [OR]
│   ├── Walking (default)
│   └── Running
└── Combat [OR]
    ├── Idle (default)
    └── Fighting
""",
    history_type="shallow",
    active_before_exit={"Game", "Movement", "Running", "Combat", "Fighting"},
    exit_to="Paused",
    re_enter_via="Game (via H)",
    # Shallow on parallel: remembers immediate children (Movement, Combat)
    # but nested states go to defaults
    expected_config={"Game", "Movement", "Walking", "Combat", "Idle"},
)

TC9_SHALLOW_DEEP_COMPARE = HistoryQuestion(
    statechart_desc="""
A [OR]
├── B [OR] (default)
│   └── Inner [OR]
│       ├── X (default)
│       └── Y
└── C

Configuration before exit: A.B.Inner.Y
Exit to C
""",
    history_type="shallow",
    active_before_exit={"A", "B", "Inner", "Y"},
    exit_to="C",
    re_enter_via="B (via H)",
    # Shallow: B remembered, Inner goes to default child
    expected_config={"A", "B", "Inner", "X"},
)

TC10_DEEP_SAME_SCENARIO = HistoryQuestion(
    statechart_desc="""
A [OR]
├── B [OR] (default)
│   └── Inner [OR]
│       ├── X (default)
│       └── Y
└── C

Configuration before exit: A.B.Inner.Y
Exit to C
""",
    history_type="deep",
    active_before_exit={"A", "B", "Inner", "Y"},
    exit_to="C",
    re_enter_via="B (via H*)",
    expected_config={"A", "B", "Inner", "Y"},  # Deep: exact
)

ALL_TEST_CASES = [
    ("TC1_basic_shallow", TC1_BASIC_SHALLOW),
    ("TC2_basic_deep", TC2_BASIC_DEEP),
    ("TC3_shallow_nested", TC3_SHALLOW_VS_DEEP_NESTED),
    ("TC4_no_prior", TC4_NO_PRIOR_HISTORY),
    ("TC5_multiple_cycles", TC5_MULTIPLE_CYCLES),
    ("TC6_deep_nesting", TC6_DEEP_NESTING),
    ("TC7_parallel_deep", TC7_PARALLEL_HISTORY),
    ("TC8_parallel_shallow", TC8_SHALLOW_PARALLEL),
    ("TC9_shallow_vs_deep_A", TC9_SHALLOW_DEEP_COMPARE),
    ("TC10_shallow_vs_deep_B", TC10_DEEP_SAME_SCENARIO),
]


def categorize_test(tc: HistoryQuestion) -> str:
    """Categorize test case."""
    if "PARALLEL" in tc.statechart_desc:
        return "parallel"
    if not tc.active_before_exit:
        return "no_prior"
    if tc.history_type == "shallow":
        return "shallow"
    return "deep"


def evaluate_prediction(predicted: Set[str], expected: Set[str]) -> Dict[str, Any]:
    """Evaluate prediction against expected configuration."""
    if not predicted and not expected:
        return {"correct": True, "jaccard": 1.0, "missing": set(), "extra": set()}

    intersection = predicted & expected
    union = predicted | expected
    jaccard = len(intersection) / len(union) if union else 1.0

    return {
        "correct": predicted == expected,
        "jaccard": jaccard,
        "missing": expected - predicted,
        "extra": predicted - expected,
    }


def demo():
    """Demonstrate history test cases."""
    print("=" * 60)
    print("History States Test Cases")
    print("=" * 60)

    for name, tc in ALL_TEST_CASES:
        cat = categorize_test(tc)
        print(f"\n--- {name} ({cat}) ---")
        print(f"History type: {tc.history_type}")
        print(f"Before exit: {tc.active_before_exit}")
        print(f"Re-enter via: {tc.re_enter_via}")
        print(f"Expected: {tc.expected_config}")


if __name__ == "__main__":
    demo()
