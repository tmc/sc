"""
exp_terminal_state_prediction: Predict final configuration after SC execution.

Given: (SC_json, initial_state, event_sequence)
Output: predicted final_configuration (set of active states)

Test Categories:
- Linear chains: A -> B -> C -> D
- Cycles: detect steady state after cycling
- Hierarchy: composite state entry/exit
- Parallel: both regions reach final states

Uses LLM to predict the terminal state given SC definition and event trace.
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Tuple, Any
import random


# =============================================================================
# Test Machines
# =============================================================================

# Category 1: Linear Chain - simple sequential states
LINEAR_CHAIN_SC = {
    "name": "LinearChain",
    "category": "linear",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "A", "type": 1, "is_initial": True},
            {"label": "B", "type": 1},
            {"label": "C", "type": 1},
            {"label": "D", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["A"], "to": ["B"], "event": "NEXT"},
        {"from": ["B"], "to": ["C"], "event": "NEXT"},
        {"from": ["C"], "to": ["D"], "event": "NEXT"},
    ]
}

# Category 2: Cycle - state cycles back
CYCLE_SC = {
    "name": "Cycle",
    "category": "cycle",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "S1", "type": 1, "is_initial": True},
            {"label": "S2", "type": 1},
            {"label": "S3", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["S1"], "to": ["S2"], "event": "TICK"},
        {"from": ["S2"], "to": ["S3"], "event": "TICK"},
        {"from": ["S3"], "to": ["S1"], "event": "TICK"},  # Cycle back
    ]
}

# Category 3: Hierarchy - composite states with default entry
HIERARCHY_SC = {
    "name": "Hierarchy",
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

# Category 4: Parallel - orthogonal regions
PARALLEL_SC = {
    "name": "Parallel",
    "category": "parallel",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Off", "type": 1, "is_initial": True},
            {
                "label": "On",
                "type": 3,  # PARALLEL
                "children": [
                    {
                        "label": "Motor",
                        "type": 2,
                        "children": [
                            {"label": "MotorIdle", "type": 1, "is_initial": True},
                            {"label": "MotorRunning", "type": 1},
                        ]
                    },
                    {
                        "label": "Display",
                        "type": 2,
                        "children": [
                            {"label": "DisplayOff", "type": 1, "is_initial": True},
                            {"label": "DisplayOn", "type": 1},
                        ]
                    },
                ]
            },
        ]
    },
    "transitions": [
        {"from": ["Off"], "to": ["On"], "event": "POWER"},
        {"from": ["On"], "to": ["Off"], "event": "POWER"},
        {"from": ["MotorIdle"], "to": ["MotorRunning"], "event": "RUN"},
        {"from": ["MotorRunning"], "to": ["MotorIdle"], "event": "STOP_MOTOR"},
        {"from": ["DisplayOff"], "to": ["DisplayOn"], "event": "LIGHT"},
        {"from": ["DisplayOn"], "to": ["DisplayOff"], "event": "DIM"},
    ]
}

# Additional test cases for each category
EXTENDED_LINEAR_SC = {
    "name": "ExtendedLinear",
    "category": "linear",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Init", "type": 1, "is_initial": True},
            {"label": "Step1", "type": 1},
            {"label": "Step2", "type": 1},
            {"label": "Step3", "type": 1},
            {"label": "Step4", "type": 1},
            {"label": "Final", "type": 1},
        ]
    },
    "transitions": [
        {"from": ["Init"], "to": ["Step1"], "event": "GO"},
        {"from": ["Step1"], "to": ["Step2"], "event": "GO"},
        {"from": ["Step2"], "to": ["Step3"], "event": "GO"},
        {"from": ["Step3"], "to": ["Step4"], "event": "GO"},
        {"from": ["Step4"], "to": ["Final"], "event": "GO"},
    ]
}

TRAFFIC_LIGHT_CYCLE_SC = {
    "name": "TrafficLight",
    "category": "cycle",
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

NESTED_HIERARCHY_SC = {
    "name": "NestedHierarchy",
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

ALL_TEST_MACHINES = [
    LINEAR_CHAIN_SC,
    CYCLE_SC,
    HIERARCHY_SC,
    PARALLEL_SC,
    EXTENDED_LINEAR_SC,
    TRAFFIC_LIGHT_CYCLE_SC,
    NESTED_HIERARCHY_SC,
]


# =============================================================================
# Trace Executor
# =============================================================================

@dataclass
class StateInfo:
    """Information about a state."""
    label: str
    type: int  # 1=BASIC, 2=NORMAL, 3=PARALLEL
    parent: Optional[str]
    children: List[str]
    is_initial: bool


class TraceExecutor:
    """Execute event traces on statecharts to compute final configuration."""

    def __init__(self, sc_json: dict):
        self.sc_json = sc_json
        self.states: Dict[str, StateInfo] = {}
        self.transitions = sc_json.get("transitions", [])
        self._parse_states(sc_json.get("root_state", {}), None)

    def _parse_states(self, state: dict, parent: Optional[str]):
        """Parse state hierarchy."""
        label = state.get("label", "")
        state_type = state.get("type", 1)
        children = state.get("children", [])

        if label and not label.startswith("__"):
            child_labels = [c.get("label") for c in children if c.get("label")]
            self.states[label] = StateInfo(
                label=label,
                type=state_type,
                parent=parent,
                children=child_labels,
                is_initial=state.get("is_initial", False),
            )

        for child in children:
            child_label = child.get("label", "")
            self._parse_states(
                child,
                label if label and not label.startswith("__") else parent
            )

    def initial_config(self) -> Set[str]:
        """Compute initial configuration."""
        active = set()

        # Find root-level initial states
        for label, info in self.states.items():
            if info.parent is None and info.is_initial:
                active.add(label)
                active = active | self._enter_substates(label)

        return active

    def _enter_substates(self, state_label: str) -> Set[str]:
        """Enter initial substates of a composite state."""
        result = set()
        info = self.states.get(state_label)

        if not info or info.type == 1:  # BASIC
            return result

        if info.type == 3:  # PARALLEL - enter all children's initial states
            for child in info.children:
                child_info = self.states.get(child)
                if child_info:
                    # For parallel, enter the region (OR state) and its initial
                    result = result | self._enter_region(child)
        else:  # NORMAL (OR) - enter initial child
            result = result | self._enter_region(state_label)

        return result

    def _enter_region(self, region_label: str) -> Set[str]:
        """Enter the initial state within a region."""
        result = set()
        region_info = self.states.get(region_label)
        if not region_info:
            return result

        for child in region_info.children:
            child_info = self.states.get(child)
            if child_info and child_info.is_initial:
                result.add(child)
                result = result | self._enter_substates(child)
                break

        return result

    def _get_descendant_states(self, state_label: str) -> Set[str]:
        """Get all descendant states of a composite state."""
        result = set()
        info = self.states.get(state_label)
        if not info:
            return result

        for child in info.children:
            result.add(child)
            result = result | self._get_descendant_states(child)

        return result

    def _transition_enabled(self, from_states: Set[str], active: Set[str]) -> bool:
        """Check if a transition is enabled."""
        for from_state in from_states:
            if from_state in active:
                return True
            # Check if from_state is a composite containing active substates
            descendants = self._get_descendant_states(from_state)
            if descendants & active:
                return True
        return False

    def step(self, active: Set[str], event: str) -> Optional[Set[str]]:
        """Execute one event, returning new configuration."""
        for t in self.transitions:
            from_states = set(t.get("from", []))
            to_states = set(t.get("to", []))
            t_event = t.get("event", "")

            if t_event == event and self._transition_enabled(from_states, active):
                # Find states to exit
                new_active = active.copy()
                for from_state in from_states:
                    if from_state in active:
                        new_active.discard(from_state)
                    else:
                        # Composite state - remove its active descendants
                        descendants = self._get_descendant_states(from_state)
                        new_active = new_active - descendants

                # Enter target states
                for target in to_states:
                    entered = self._enter_state(target)
                    new_active = new_active | entered

                return new_active

        return None  # No matching transition

    def _enter_state(self, state_label: str) -> Set[str]:
        """Enter a state, returning only leaf states in configuration."""
        info = self.states.get(state_label)
        if not info:
            return {state_label}

        if info.type == 1:  # BASIC - leaf state
            return {state_label}
        elif info.type == 3:  # PARALLEL - enter all regions
            result = set()
            for child in info.children:
                result = result | self._enter_region(child)
            return result
        else:  # NORMAL (OR) - enter initial child
            return self._enter_region(state_label)

    def execute_trace(self, events: List[str]) -> Set[str]:
        """Execute a sequence of events and return final configuration."""
        active = self.initial_config()

        for event in events:
            result = self.step(active, event)
            if result is not None:
                active = result
            # If event doesn't match, configuration stays the same

        return active


def execute_to_completion(sc_json: dict, events: List[str]) -> Set[str]:
    """Execute events on SC and return final configuration."""
    executor = TraceExecutor(sc_json)
    return executor.execute_trace(events)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def configuration_accuracy(predicted: Set[str], actual: Set[str]) -> Dict[str, float]:
    """Compute accuracy metrics between predicted and actual configurations."""
    if not predicted and not actual:
        return {"exact_match": 1.0, "jaccard": 1.0, "precision": 1.0, "recall": 1.0}

    exact = 1.0 if predicted == actual else 0.0

    intersection = len(predicted & actual)
    union = len(predicted | actual)
    jaccard = intersection / union if union > 0 else 0.0

    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(actual) if actual else 0.0

    return {
        "exact_match": exact,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
    }


# =============================================================================
# Test Case Generator
# =============================================================================

@dataclass
class TestCase:
    """A single test case for terminal state prediction."""
    sc_json: dict
    initial_state: str
    events: List[str]
    expected_final: Set[str]
    category: str


def generate_test_cases(sc_json: dict, num_cases: int = 10) -> List[TestCase]:
    """Generate test cases for a given statechart."""
    executor = TraceExecutor(sc_json)
    category = sc_json.get("category", "unknown")
    cases = []

    # Collect all events
    all_events = set()
    for t in sc_json.get("transitions", []):
        event = t.get("event", "")
        if event:
            all_events.add(event)
    all_events = list(all_events)

    if not all_events:
        return cases

    for _ in range(num_cases):
        # Generate random event sequence of varying length
        length = random.randint(1, 8)
        events = [random.choice(all_events) for _ in range(length)]

        # Execute to get expected final state
        active = executor.initial_config()
        for event in events:
            result = executor.step(active, event)
            if result is not None:
                active = result

        # Get initial state name
        initial = list(executor.initial_config())
        initial_name = initial[0] if initial else "unknown"

        cases.append(TestCase(
            sc_json=sc_json,
            initial_state=initial_name,
            events=events,
            expected_final=active,
            category=category,
        ))

    return cases


def generate_all_test_cases(cases_per_machine: int = 15) -> List[TestCase]:
    """Generate test cases for all test machines."""
    all_cases = []
    for sc in ALL_TEST_MACHINES:
        cases = generate_test_cases(sc, cases_per_machine)
        all_cases.extend(cases)
    return all_cases


# =============================================================================
# Demo
# =============================================================================

# Hierarchy-aware prediction exports
from .hierarchy_predictor import (
    HierarchyPredictor,
    HierarchyPredictionResult,
    build_ascii_tree,
    get_state_info,
)
from .benchmark_hierarchy import (
    run_hierarchy_benchmark,
    generate_hierarchy_test_cases,
    HIERARCHY_TEST_MACHINES,
)


def demo():
    """Demonstrate the terminal state prediction infrastructure."""
    print("=" * 60)
    print("Terminal State Prediction Infrastructure")
    print("=" * 60)

    for sc in ALL_TEST_MACHINES:
        print(f"\n--- {sc['name']} ({sc.get('category', 'unknown')}) ---")
        executor = TraceExecutor(sc)
        initial = executor.initial_config()
        print(f"Initial config: {initial}")

        # Generate a test case
        cases = generate_test_cases(sc, num_cases=3)
        for i, case in enumerate(cases):
            print(f"  Case {i+1}: {case.events} -> {case.expected_final}")


if __name__ == "__main__":
    demo()
