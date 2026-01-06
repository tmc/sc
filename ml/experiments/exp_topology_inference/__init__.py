"""
exp_topology_inference: Infer SC topology from execution traces.

Given: List of execution traces (state sequences)
Output: Inferred topology (states, transitions, initial state, patterns)

Test Categories:
- Linear: A -> B -> C chain
- Cycle: A -> B -> C -> A cycle
- Branch: Diamond pattern (A -> B/C -> X)
- Self-loop: State transitions to itself
- Complex: Multiple patterns combined
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Tuple, Optional
import json


@dataclass
class InferredTopology:
    """Inferred topology from traces."""
    states: Set[str]
    transitions: Set[Tuple[str, str]]
    initial_state: Optional[str]
    features: List[str]  # ["self-loop on B", "cycle", "branching from A"]


@dataclass
class TopologyTestCase:
    """Test case for topology inference."""
    name: str
    category: str
    traces: List[List[str]]
    expected_states: Set[str]
    expected_transitions: Set[Tuple[str, str]]
    expected_initial: str
    expected_features: List[str]


def ground_truth_topology(traces: List[List[str]]) -> InferredTopology:
    """Compute ground truth topology from traces deterministically."""
    states = set()
    transitions = set()

    for trace in traces:
        for state in trace:
            states.add(state)
        for i in range(len(trace) - 1):
            transitions.add((trace[i], trace[i + 1]))

    # Initial state is first state of first trace
    initial = traces[0][0] if traces and traces[0] else None

    # Detect features
    features = []

    # Self-loops
    for s in states:
        if (s, s) in transitions:
            features.append(f"self-loop on {s}")

    # Branching (state with multiple outgoing)
    for s in states:
        outgoing = [t for t in transitions if t[0] == s]
        if len(outgoing) > 1:
            targets = sorted([t[1] for t in outgoing])
            features.append(f"branch from {s} to {','.join(targets)}")

    # Cycles (can reach initial from some state)
    if initial:
        for s in states:
            if s != initial and (s, initial) in transitions:
                features.append("cycle")
                break

    return InferredTopology(
        states=states,
        transitions=transitions,
        initial_state=initial,
        features=sorted(features),
    )


# =============================================================================
# Test Cases
# =============================================================================

LINEAR_TEST = TopologyTestCase(
    name="Linear",
    category="linear",
    traces=[
        ["A", "B", "C", "D"],
        ["A", "B", "C"],
        ["A", "B"],
    ],
    expected_states={"A", "B", "C", "D"},
    expected_transitions={("A", "B"), ("B", "C"), ("C", "D")},
    expected_initial="A",
    expected_features=[],
)

CYCLE_TEST = TopologyTestCase(
    name="Cycle3",
    category="cycle",
    traces=[
        ["A", "B", "C", "A", "B", "C"],
        ["A", "B", "C", "A"],
        ["A", "B", "C", "A", "B"],
    ],
    expected_states={"A", "B", "C"},
    expected_transitions={("A", "B"), ("B", "C"), ("C", "A")},
    expected_initial="A",
    expected_features=["cycle"],
)

BRANCH_TEST = TopologyTestCase(
    name="Diamond",
    category="branch",
    traces=[
        ["A", "B", "D"],
        ["A", "C", "D"],
        ["A", "B", "D"],
        ["A", "C", "D"],
    ],
    expected_states={"A", "B", "C", "D"},
    expected_transitions={("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")},
    expected_initial="A",
    expected_features=["branch from A to B,C"],
)

SELFLOOP_TEST = TopologyTestCase(
    name="SelfLoop",
    category="selfloop",
    traces=[
        ["A", "A", "A", "B"],
        ["A", "B"],
        ["A", "A", "B"],
    ],
    expected_states={"A", "B"},
    expected_transitions={("A", "A"), ("A", "B")},
    expected_initial="A",
    expected_features=["branch from A to A,B", "self-loop on A"],
)

COMPLEX_TEST = TopologyTestCase(
    name="Complex",
    category="complex",
    traces=[
        ["Start", "A", "B", "C", "A"],  # Cycle
        ["Start", "A", "B", "B", "C"],  # Self-loop on B
        ["Start", "A", "C"],            # Skip B
        ["Start", "A", "B", "C", "End"],
    ],
    expected_states={"Start", "A", "B", "C", "End"},
    expected_transitions={
        ("Start", "A"),
        ("A", "B"), ("A", "C"),  # Branch from A
        ("B", "B"), ("B", "C"),  # Self-loop on B
        ("C", "A"), ("C", "End"),
    },
    expected_initial="Start",
    expected_features=["branch from A to B,C", "branch from B to B,C", "branch from C to A,End", "cycle", "self-loop on B"],
)

# Additional edge cases
SINGLE_STATE_TEST = TopologyTestCase(
    name="SingleState",
    category="edge",
    traces=[
        ["X"],
        ["X"],
    ],
    expected_states={"X"},
    expected_transitions=set(),
    expected_initial="X",
    expected_features=[],
)

TWO_STATE_TOGGLE = TopologyTestCase(
    name="Toggle",
    category="cycle",
    traces=[
        ["On", "Off", "On", "Off"],
        ["On", "Off", "On"],
    ],
    expected_states={"On", "Off"},
    expected_transitions={("On", "Off"), ("Off", "On")},
    expected_initial="On",
    expected_features=["cycle"],
)

MULTI_BRANCH_TEST = TopologyTestCase(
    name="MultiBranch",
    category="branch",
    traces=[
        ["Root", "A", "End"],
        ["Root", "B", "End"],
        ["Root", "C", "End"],
    ],
    expected_states={"Root", "A", "B", "C", "End"},
    expected_transitions={
        ("Root", "A"), ("Root", "B"), ("Root", "C"),
        ("A", "End"), ("B", "End"), ("C", "End"),
    },
    expected_initial="Root",
    expected_features=["branch from Root to A,B,C"],
)

ALL_TEST_CASES = [
    LINEAR_TEST,
    CYCLE_TEST,
    BRANCH_TEST,
    SELFLOOP_TEST,
    COMPLEX_TEST,
    SINGLE_STATE_TEST,
    TWO_STATE_TOGGLE,
    MULTI_BRANCH_TEST,
]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_topology(
    predicted: InferredTopology,
    expected: TopologyTestCase,
) -> Dict[str, float]:
    """Evaluate predicted topology against expected."""
    # States accuracy (Jaccard)
    states_intersection = len(predicted.states & expected.expected_states)
    states_union = len(predicted.states | expected.expected_states)
    states_acc = states_intersection / states_union if states_union > 0 else 1.0

    # Transitions accuracy (Jaccard)
    trans_intersection = len(predicted.transitions & expected.expected_transitions)
    trans_union = len(predicted.transitions | expected.expected_transitions)
    trans_acc = trans_intersection / trans_union if trans_union > 0 else 1.0

    # Initial state accuracy (exact match)
    initial_acc = 1.0 if predicted.initial_state == expected.expected_initial else 0.0

    # Features accuracy (set overlap)
    pred_features = set(predicted.features)
    exp_features = set(expected.expected_features)
    if not pred_features and not exp_features:
        features_acc = 1.0
    else:
        feat_intersection = len(pred_features & exp_features)
        feat_union = len(pred_features | exp_features)
        features_acc = feat_intersection / feat_union if feat_union > 0 else 0.0

    # Overall accuracy (weighted average)
    overall = (states_acc * 0.3 + trans_acc * 0.4 + initial_acc * 0.2 + features_acc * 0.1)

    return {
        "states_acc": states_acc,
        "transitions_acc": trans_acc,
        "initial_acc": initial_acc,
        "features_acc": features_acc,
        "overall": overall,
    }


def demo():
    """Demonstrate topology inference infrastructure."""
    print("=" * 60)
    print("Topology Inference Infrastructure Demo")
    print("=" * 60)

    for tc in ALL_TEST_CASES:
        print(f"\n--- {tc.name} ({tc.category}) ---")
        print(f"Traces: {tc.traces[:2]}...")

        # Compute ground truth
        gt = ground_truth_topology(tc.traces)
        print(f"States: {gt.states}")
        print(f"Transitions: {gt.transitions}")
        print(f"Initial: {gt.initial_state}")
        print(f"Features: {gt.features}")

        # Evaluate against expected
        metrics = evaluate_topology(gt, tc)
        print(f"Self-eval: states={metrics['states_acc']:.0%} trans={metrics['transitions_acc']:.0%}")


if __name__ == "__main__":
    demo()
