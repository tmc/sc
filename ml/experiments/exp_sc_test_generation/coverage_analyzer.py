"""
Coverage Analyzer for Statecharts

Tracks state and transition coverage during test trace execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class CoverageReport:
    """Coverage metrics for a statechart."""
    # State coverage
    total_states: int = 0
    covered_states: Set[str] = field(default_factory=set)
    uncovered_states: Set[str] = field(default_factory=set)

    # Transition coverage
    total_transitions: int = 0
    covered_transitions: Set[Tuple[str, str, str]] = field(default_factory=set)  # (from, to, event)
    uncovered_transitions: Set[Tuple[str, str, str]] = field(default_factory=set)

    @property
    def state_coverage(self) -> float:
        """State coverage percentage."""
        if self.total_states == 0:
            return 0.0
        return len(self.covered_states) / self.total_states

    @property
    def transition_coverage(self) -> float:
        """Transition coverage percentage."""
        if self.total_transitions == 0:
            return 0.0
        return len(self.covered_transitions) / self.total_transitions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_coverage": self.state_coverage,
            "transition_coverage": self.transition_coverage,
            "states": {
                "total": self.total_states,
                "covered": len(self.covered_states),
                "covered_list": list(self.covered_states),
                "uncovered_list": list(self.uncovered_states),
            },
            "transitions": {
                "total": self.total_transitions,
                "covered": len(self.covered_transitions),
                "covered_list": [list(t) for t in self.covered_transitions],
                "uncovered_list": [list(t) for t in self.uncovered_transitions],
            },
        }


class CoverageAnalyzer:
    """
    Analyzes coverage achieved by test traces on a statechart.
    """

    def __init__(self, statechart: Dict):
        """
        Initialize with a statechart definition.

        Args:
            statechart: Statechart JSON with root_state and transitions
        """
        self.statechart = statechart
        self.all_states: Set[str] = set()
        self.all_transitions: Set[Tuple[str, str, str]] = set()
        self.initial_state: Optional[str] = None

        self._extract_states_and_transitions()

    def _extract_states_and_transitions(self):
        """Extract all states and transitions from statechart."""
        # Extract states recursively
        def extract_states(state: Dict, parent_path: str = ""):
            label = state.get("label", "")
            if label and label != "__root__":
                self.all_states.add(label)

            if state.get("is_initial") and self.initial_state is None:
                self.initial_state = label

            for child in state.get("children", []):
                extract_states(child, f"{parent_path}/{label}")
                if child.get("is_initial") and self.initial_state is None:
                    self.initial_state = child.get("label")

        if "root_state" in self.statechart:
            extract_states(self.statechart["root_state"])

        # Extract transitions
        for trans in self.statechart.get("transitions", []):
            from_states = trans.get("from", [])
            to_states = trans.get("to", [])
            event = trans.get("event", "")

            for f in from_states:
                for t in to_states:
                    self.all_transitions.add((f, t, event))

    def execute_trace(self, events: List[str]) -> Tuple[CoverageReport, List[str]]:
        """
        Execute a trace and compute coverage.

        Args:
            events: List of event names to execute

        Returns:
            (coverage_report, state_sequence)
        """
        report = CoverageReport(
            total_states=len(self.all_states),
            total_transitions=len(self.all_transitions),
            uncovered_states=self.all_states.copy(),
            uncovered_transitions=self.all_transitions.copy(),
        )

        # Start from initial state
        current_state = self.initial_state
        state_sequence = []

        if current_state:
            state_sequence.append(current_state)
            report.covered_states.add(current_state)
            report.uncovered_states.discard(current_state)

        # Execute each event
        for event in events:
            # Find matching transition
            next_state = None
            for (f, t, e) in self.all_transitions:
                if f == current_state and e == event:
                    next_state = t
                    # Mark transition covered
                    trans_key = (f, t, e)
                    report.covered_transitions.add(trans_key)
                    report.uncovered_transitions.discard(trans_key)
                    break

            if next_state:
                current_state = next_state
                state_sequence.append(current_state)
                report.covered_states.add(current_state)
                report.uncovered_states.discard(current_state)

        return report, state_sequence

    def get_available_events(self, current_state: str) -> List[str]:
        """Get events available from current state."""
        events = []
        for (f, t, e) in self.all_transitions:
            if f == current_state:
                events.append(e)
        return events

    def get_uncovered_transitions(self, report: CoverageReport) -> List[Dict]:
        """Get list of uncovered transitions with details."""
        uncovered = []
        for (f, t, e) in report.uncovered_transitions:
            uncovered.append({
                "from": f,
                "to": t,
                "event": e,
            })
        return uncovered


def compute_state_coverage(statechart: Dict, traces: List[List[str]]) -> float:
    """
    Compute state coverage for multiple traces.

    Args:
        statechart: Statechart definition
        traces: List of event sequences

    Returns:
        State coverage percentage (0.0 to 1.0)
    """
    analyzer = CoverageAnalyzer(statechart)
    all_covered = set()

    for trace in traces:
        report, _ = analyzer.execute_trace(trace)
        all_covered.update(report.covered_states)

    if len(analyzer.all_states) == 0:
        return 0.0
    return len(all_covered) / len(analyzer.all_states)


def compute_transition_coverage(statechart: Dict, traces: List[List[str]]) -> float:
    """
    Compute transition coverage for multiple traces.

    Args:
        statechart: Statechart definition
        traces: List of event sequences

    Returns:
        Transition coverage percentage (0.0 to 1.0)
    """
    analyzer = CoverageAnalyzer(statechart)
    all_covered = set()

    for trace in traces:
        report, _ = analyzer.execute_trace(trace)
        all_covered.update(report.covered_transitions)

    if len(analyzer.all_transitions) == 0:
        return 0.0
    return len(all_covered) / len(analyzer.all_transitions)
