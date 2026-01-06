"""
Test cases for comprehensive action execution coverage.

Covers:
- Action placement (entry/exit/transition/combined/nested)
- Context mutations (assignment, increment, conditional, multi-var, nested)
- Execution order (Harel semantics)
- Guard-action interaction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ActionType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    TRANSITION = "transition"


@dataclass
class Action:
    """An action to execute."""
    type: ActionType
    expression: str  # e.g., "count = 0", "x = x + 1"
    state: str  # Which state this action belongs to


@dataclass
class TestCase:
    """A test case for action execution."""
    name: str
    description: str
    category: str  # entry, exit, transition, order, context_mutation

    # Statechart definition
    states: List[str]
    hierarchy: Dict[str, List[str]]  # parent -> children
    initial_state: str

    # Actions
    entry_actions: Dict[str, List[str]]  # state -> [expressions]
    exit_actions: Dict[str, List[str]]  # state -> [expressions]
    transition_actions: Dict[str, str]  # "src->tgt" -> expression

    # Transitions
    transitions: Dict[str, str]  # "src,event" -> "tgt"
    guards: Dict[str, str] = field(default_factory=dict)  # "src,event" -> guard expr

    # Test scenario
    initial_context: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    # Expected results
    expected_context: Dict[str, Any] = field(default_factory=dict)
    expected_action_order: List[str] = field(default_factory=list)  # ["A.exit", "trans:A->B", "B.entry"]
    expected_final_state: str = ""


# =============================================================================
# TC1: Entry Actions
# =============================================================================

TC_ENTRY_SIMPLE = TestCase(
    name="entry_simple",
    description="Simple entry action sets count to 0",
    category="entry",
    states=["Idle", "Active"],
    hierarchy={},
    initial_state="Idle",
    entry_actions={"Active": ["count = 0"]},
    exit_actions={},
    transition_actions={},
    transitions={"Idle,START": "Active"},
    events=["START"],
    expected_context={"count": 0},
    expected_action_order=["Active.entry"],
    expected_final_state="Active",
)

TC_ENTRY_MULTIPLE = TestCase(
    name="entry_multiple",
    description="Entry action with multiple assignments",
    category="entry",
    states=["Off", "On"],
    hierarchy={},
    initial_state="Off",
    entry_actions={"On": ["power = true", "brightness = 100"]},
    exit_actions={},
    transition_actions={},
    transitions={"Off,TURN_ON": "On"},
    events=["TURN_ON"],
    expected_context={"power": True, "brightness": 100},
    expected_action_order=["On.entry"],
    expected_final_state="On",
)

TC_ENTRY_INCREMENT = TestCase(
    name="entry_increment",
    description="Entry action increments existing value",
    category="entry",
    states=["A", "B"],
    hierarchy={},
    initial_state="A",
    entry_actions={"B": ["visits = visits + 1"]},
    exit_actions={},
    transition_actions={},
    transitions={"A,GO": "B"},
    initial_context={"visits": 5},
    events=["GO"],
    expected_context={"visits": 6},
    expected_action_order=["B.entry"],
    expected_final_state="B",
)

# =============================================================================
# TC2: Exit Actions
# =============================================================================

TC_EXIT_SIMPLE = TestCase(
    name="exit_simple",
    description="Exit action sets cleanup flag",
    category="exit",
    states=["Running", "Stopped"],
    hierarchy={},
    initial_state="Running",
    entry_actions={},
    exit_actions={"Running": ["cleaned_up = true"]},
    transition_actions={},
    transitions={"Running,STOP": "Stopped"},
    events=["STOP"],
    expected_context={"cleaned_up": True},
    expected_action_order=["Running.exit"],
    expected_final_state="Stopped",
)

TC_EXIT_SAVE_STATE = TestCase(
    name="exit_save_state",
    description="Exit action saves current state",
    category="exit",
    states=["Editing", "Viewing"],
    hierarchy={},
    initial_state="Editing",
    entry_actions={},
    exit_actions={"Editing": ["saved_data = current_data"]},
    transition_actions={},
    transitions={"Editing,VIEW": "Viewing"},
    initial_context={"current_data": "hello world"},
    events=["VIEW"],
    expected_context={"current_data": "hello world", "saved_data": "hello world"},
    expected_action_order=["Editing.exit"],
    expected_final_state="Viewing",
)

# =============================================================================
# TC3: Transition Actions
# =============================================================================

TC_TRANS_SIMPLE = TestCase(
    name="trans_simple",
    description="Transition action logs event",
    category="transition",
    states=["A", "B"],
    hierarchy={},
    initial_state="A",
    entry_actions={},
    exit_actions={},
    transition_actions={"A->B": "transition_count = transition_count + 1"},
    transitions={"A,GO": "B"},
    initial_context={"transition_count": 0},
    events=["GO"],
    expected_context={"transition_count": 1},
    expected_action_order=["trans:A->B"],
    expected_final_state="B",
)

TC_TRANS_SET_TARGET = TestCase(
    name="trans_set_target",
    description="Transition action sets target-related data",
    category="transition",
    states=["Login", "Dashboard"],
    hierarchy={},
    initial_state="Login",
    entry_actions={},
    exit_actions={},
    transition_actions={"Login->Dashboard": "logged_in_at = 1234567890"},
    transitions={"Login,AUTH_SUCCESS": "Dashboard"},
    events=["AUTH_SUCCESS"],
    expected_context={"logged_in_at": 1234567890},
    expected_action_order=["trans:Login->Dashboard"],
    expected_final_state="Dashboard",
)

# =============================================================================
# TC4: Execution Order (Critical!)
# =============================================================================

TC_ORDER_EXIT_ENTRY = TestCase(
    name="order_exit_entry",
    description="Exit before entry (Harel semantics)",
    category="order",
    states=["A", "B"],
    hierarchy={},
    initial_state="A",
    entry_actions={"B": ["order = order + 'B.entry,'"]},
    exit_actions={"A": ["order = order + 'A.exit,'"]},
    transition_actions={},
    transitions={"A,GO": "B"},
    initial_context={"order": ""},
    events=["GO"],
    expected_context={"order": "A.exit,B.entry,"},
    expected_action_order=["A.exit", "B.entry"],
    expected_final_state="B",
)

TC_ORDER_EXIT_TRANS_ENTRY = TestCase(
    name="order_exit_trans_entry",
    description="Exit -> Transition -> Entry order",
    category="order",
    states=["A", "B"],
    hierarchy={},
    initial_state="A",
    entry_actions={"B": ["order = order + 'B.entry,'"]},
    exit_actions={"A": ["order = order + 'A.exit,'"]},
    transition_actions={"A->B": "order = order + 'trans,'"},
    transitions={"A,GO": "B"},
    initial_context={"order": ""},
    events=["GO"],
    expected_context={"order": "A.exit,trans,B.entry,"},
    expected_action_order=["A.exit", "trans:A->B", "B.entry"],
    expected_final_state="B",
)

TC_ORDER_NESTED_EXIT = TestCase(
    name="order_nested_exit",
    description="Nested states exit bottom-up (child before parent)",
    category="order",
    states=["Parent", "Child", "Other"],
    hierarchy={"Parent": ["Child"]},
    initial_state="Child",  # Start in nested child
    entry_actions={},
    exit_actions={
        "Child": ["order = order + 'Child.exit,'"],
        "Parent": ["order = order + 'Parent.exit,'"],
    },
    transition_actions={},
    transitions={"Child,LEAVE": "Other"},
    initial_context={"order": ""},
    events=["LEAVE"],
    expected_context={"order": "Child.exit,Parent.exit,"},
    expected_action_order=["Child.exit", "Parent.exit"],
    expected_final_state="Other",
)

TC_ORDER_NESTED_ENTRY = TestCase(
    name="order_nested_entry",
    description="Nested states enter top-down (parent before child)",
    category="order",
    states=["Start", "Parent", "Child"],
    hierarchy={"Parent": ["Child"]},
    initial_state="Start",
    entry_actions={
        "Parent": ["order = order + 'Parent.entry,'"],
        "Child": ["order = order + 'Child.entry,'"],
    },
    exit_actions={},
    transition_actions={},
    transitions={"Start,ENTER": "Child"},
    initial_context={"order": ""},
    events=["ENTER"],
    expected_context={"order": "Parent.entry,Child.entry,"},
    expected_action_order=["Parent.entry", "Child.entry"],
    expected_final_state="Child",
)

# =============================================================================
# TC5: Context Mutations
# =============================================================================

TC_CONTEXT_CONDITIONAL = TestCase(
    name="context_conditional",
    description="Conditional assignment in action",
    category="context_mutation",
    states=["Check", "High", "Low"],
    hierarchy={},
    initial_state="Check",
    entry_actions={
        "High": ["result = 'above threshold'"],
        "Low": ["result = 'below threshold'"],
    },
    exit_actions={},
    transition_actions={},
    transitions={"Check,EVAL": "High"},  # Simplified - guard determines path
    guards={"Check,EVAL": "value > 50"},
    initial_context={"value": 75},
    events=["EVAL"],
    expected_context={"value": 75, "result": "above threshold"},
    expected_action_order=["High.entry"],
    expected_final_state="High",
)

TC_CONTEXT_CHAIN = TestCase(
    name="context_chain",
    description="Action sets value used by subsequent entry",
    category="context_mutation",
    states=["A", "B"],
    hierarchy={},
    initial_state="A",
    entry_actions={"B": ["y = x + 1"]},
    exit_actions={"A": ["x = 10"]},
    transition_actions={},
    transitions={"A,GO": "B"},
    initial_context={},
    events=["GO"],
    expected_context={"x": 10, "y": 11},
    expected_action_order=["A.exit", "B.entry"],
    expected_final_state="B",
)

TC_CONTEXT_NESTED_ACCESS = TestCase(
    name="context_nested_access",
    description="Nested object access in actions",
    category="context_mutation",
    states=["Game", "Score"],
    hierarchy={},
    initial_state="Game",
    entry_actions={"Score": ["player.score = player.score + 100"]},
    exit_actions={},
    transition_actions={},
    transitions={"Game,WIN": "Score"},
    initial_context={"player": {"name": "Alice", "score": 500}},
    events=["WIN"],
    expected_context={"player": {"name": "Alice", "score": 600}},
    expected_action_order=["Score.entry"],
    expected_final_state="Score",
)

TC_CONTEXT_MULTI_VAR = TestCase(
    name="context_multi_var",
    description="Multiple variables with dependencies",
    category="context_mutation",
    states=["Init", "Computed"],
    hierarchy={},
    initial_state="Init",
    entry_actions={"Computed": ["a = 10", "b = 20", "c = a + b"]},
    exit_actions={},
    transition_actions={},
    transitions={"Init,COMPUTE": "Computed"},
    events=["COMPUTE"],
    expected_context={"a": 10, "b": 20, "c": 30},
    expected_action_order=["Computed.entry"],
    expected_final_state="Computed",
)

# =============================================================================
# Combined/Complex Cases
# =============================================================================

TC_COMBINED_ALL = TestCase(
    name="combined_all",
    description="Entry + Exit + Transition actions together",
    category="combined",
    states=["Ready", "Processing", "Done"],
    hierarchy={},
    initial_state="Ready",
    entry_actions={
        "Processing": ["started = true"],
        "Done": ["completed = true"],
    },
    exit_actions={
        "Ready": ["prepared = true"],
        "Processing": ["result = 42"],
    },
    transition_actions={
        "Ready->Processing": "step = 1",
        "Processing->Done": "step = 2",
    },
    transitions={
        "Ready,START": "Processing",
        "Processing,FINISH": "Done",
    },
    events=["START", "FINISH"],
    expected_context={
        "prepared": True,
        "step": 2,
        "started": True,
        "result": 42,
        "completed": True,
    },
    expected_action_order=[
        "Ready.exit", "trans:Ready->Processing", "Processing.entry",
        "Processing.exit", "trans:Processing->Done", "Done.entry",
    ],
    expected_final_state="Done",
)

TC_SELF_TRANSITION = TestCase(
    name="self_transition",
    description="Self-transition executes exit and entry",
    category="combined",
    states=["Counter"],
    hierarchy={},
    initial_state="Counter",
    entry_actions={"Counter": ["count = count + 1"]},
    exit_actions={"Counter": ["exited = exited + 1"]},
    transition_actions={"Counter->Counter": "looped = looped + 1"},
    transitions={"Counter,INCREMENT": "Counter"},
    initial_context={"count": 0, "exited": 0, "looped": 0},
    events=["INCREMENT", "INCREMENT"],
    expected_context={"count": 2, "exited": 2, "looped": 2},
    expected_action_order=[
        "Counter.exit", "trans:Counter->Counter", "Counter.entry",
        "Counter.exit", "trans:Counter->Counter", "Counter.entry",
    ],
    expected_final_state="Counter",
)

# =============================================================================
# All Test Cases
# =============================================================================

TEST_CASES = [
    # Entry actions
    TC_ENTRY_SIMPLE,
    TC_ENTRY_MULTIPLE,
    TC_ENTRY_INCREMENT,
    # Exit actions
    TC_EXIT_SIMPLE,
    TC_EXIT_SAVE_STATE,
    # Transition actions
    TC_TRANS_SIMPLE,
    TC_TRANS_SET_TARGET,
    # Execution order
    TC_ORDER_EXIT_ENTRY,
    TC_ORDER_EXIT_TRANS_ENTRY,
    TC_ORDER_NESTED_EXIT,
    TC_ORDER_NESTED_ENTRY,
    # Context mutations
    TC_CONTEXT_CONDITIONAL,
    TC_CONTEXT_CHAIN,
    TC_CONTEXT_NESTED_ACCESS,
    TC_CONTEXT_MULTI_VAR,
    # Combined
    TC_COMBINED_ALL,
    TC_SELF_TRANSITION,
]

# Group test cases by category
TEST_CASES_BY_CATEGORY = {
    "entry": [TC_ENTRY_SIMPLE, TC_ENTRY_MULTIPLE, TC_ENTRY_INCREMENT],
    "exit": [TC_EXIT_SIMPLE, TC_EXIT_SAVE_STATE],
    "transition": [TC_TRANS_SIMPLE, TC_TRANS_SET_TARGET],
    "order": [TC_ORDER_EXIT_ENTRY, TC_ORDER_EXIT_TRANS_ENTRY, TC_ORDER_NESTED_EXIT, TC_ORDER_NESTED_ENTRY],
    "context_mutation": [TC_CONTEXT_CONDITIONAL, TC_CONTEXT_CHAIN, TC_CONTEXT_NESTED_ACCESS, TC_CONTEXT_MULTI_VAR],
    "combined": [TC_COMBINED_ALL, TC_SELF_TRANSITION],
}
