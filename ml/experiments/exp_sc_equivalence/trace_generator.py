"""
Trace Generator: Enumerate execution traces for statecharts.

A trace is a sequence of (state, event, next_state) tuples representing
possible executions through the statechart.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, FrozenSet
from collections import deque


@dataclass
class Trace:
    """An execution trace through a statechart."""
    steps: List[Tuple[str, str, str]]  # (from_state, event, to_state)

    def __hash__(self):
        return hash(tuple(self.steps))

    def __eq__(self, other):
        if not isinstance(other, Trace):
            return False
        return self.steps == other.steps

    @property
    def events(self) -> Tuple[str, ...]:
        """Get sequence of events in trace."""
        return tuple(step[1] for step in self.steps)

    @property
    def states(self) -> Tuple[str, ...]:
        """Get sequence of states visited."""
        if not self.steps:
            return ()
        states = [self.steps[0][0]]
        states.extend(step[2] for step in self.steps)
        return tuple(states)

    @property
    def final_state(self) -> Optional[str]:
        """Get final state of trace."""
        if not self.steps:
            return None
        return self.steps[-1][2]


@dataclass
class TraceSet:
    """Set of traces with comparison operations."""
    traces: Set[Trace] = field(default_factory=set)

    def add(self, trace: Trace):
        self.traces.add(trace)

    @property
    def event_sequences(self) -> Set[Tuple[str, ...]]:
        """Get all unique event sequences."""
        return {t.events for t in self.traces}

    def accepts(self, events: Tuple[str, ...]) -> bool:
        """Check if trace set accepts event sequence."""
        return events in self.event_sequences


class TraceGenerator:
    """
    Generate execution traces from statechart.

    Uses BFS to enumerate traces up to a maximum depth.
    """

    def __init__(self, max_depth: int = 5, max_traces: int = 1000):
        self.max_depth = max_depth
        self.max_traces = max_traces

    def generate(self, statechart: Dict[str, Any]) -> TraceSet:
        """
        Generate all traces up to max_depth.

        Args:
            statechart: Statechart definition

        Returns:
            TraceSet containing all enumerated traces
        """
        trace_set = TraceSet()

        # Get initial state
        initial = self._get_initial_state(statechart)
        if not initial:
            return trace_set

        # Get transitions indexed by source state
        trans_by_source = self._index_transitions(statechart)

        # BFS to enumerate traces
        # Queue: (current_state, trace_so_far)
        queue = deque([(initial, [])])
        visited_configs = set()  # (state, trace_length) to limit explosion

        while queue and len(trace_set.traces) < self.max_traces:
            current, trace = queue.popleft()

            # Add current trace if non-empty
            if trace:
                trace_set.add(Trace(steps=list(trace)))

            # Check depth limit
            if len(trace) >= self.max_depth:
                continue

            # Avoid revisiting same state at same depth
            config = (current, len(trace))
            if config in visited_configs:
                continue
            visited_configs.add(config)

            # Explore outgoing transitions
            for trans in trans_by_source.get(current, []):
                event = trans.get("event", "")
                targets = trans.get("to", [])

                for target in targets:
                    new_step = (current, event, target)
                    new_trace = trace + [new_step]
                    queue.append((target, new_trace))

        return trace_set

    def generate_event_sequences(
        self,
        statechart: Dict[str, Any],
    ) -> Set[Tuple[str, ...]]:
        """Generate just the event sequences (without state info)."""
        trace_set = self.generate(statechart)
        return trace_set.event_sequences

    def get_accepted_events(
        self,
        statechart: Dict[str, Any],
        from_state: str,
    ) -> Set[str]:
        """Get events accepted from a specific state."""
        events = set()
        for trans in statechart.get("transitions", []):
            if from_state in trans.get("from", []):
                event = trans.get("event", "")
                if event:
                    events.add(event)
        return events

    def get_reachable_states(
        self,
        statechart: Dict[str, Any],
    ) -> Set[str]:
        """Get all reachable states from initial."""
        initial = self._get_initial_state(statechart)
        if not initial:
            return set()

        reachable = {initial}
        trans_by_source = self._index_transitions(statechart)

        queue = deque([initial])
        while queue:
            current = queue.popleft()
            for trans in trans_by_source.get(current, []):
                for target in trans.get("to", []):
                    if target not in reachable:
                        reachable.add(target)
                        queue.append(target)

        return reachable

    def _get_initial_state(self, statechart: Dict[str, Any]) -> Optional[str]:
        """Get initial state label."""
        def find_initial(state):
            if state.get("is_initial") and not state.get("label", "").startswith("__"):
                return state.get("label")
            for child in state.get("children", []):
                result = find_initial(child)
                if result:
                    return result
            return None

        return find_initial(statechart.get("root_state", {}))

    def _index_transitions(
        self,
        statechart: Dict[str, Any],
    ) -> Dict[str, List[Dict]]:
        """Index transitions by source state."""
        index = {}
        for trans in statechart.get("transitions", []):
            for source in trans.get("from", []):
                if source not in index:
                    index[source] = []
                index[source].append(trans)
        return index


def generate_traces(
    statechart: Dict[str, Any],
    max_depth: int = 5,
) -> TraceSet:
    """Convenience function to generate traces."""
    generator = TraceGenerator(max_depth=max_depth)
    return generator.generate(statechart)


def demo():
    """Demonstrate trace generation."""
    print("=" * 60)
    print("TRACE GENERATOR: Enumerate Execution Traces")
    print("=" * 60)

    # Simple traffic light
    traffic_light = {
        "name": "Traffic Light",
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

    generator = TraceGenerator(max_depth=4)
    traces = generator.generate(traffic_light)

    print(f"\n--- Traffic Light Traces (depth=4) ---")
    print(f"Total traces: {len(traces.traces)}")

    for trace in sorted(traces.traces, key=lambda t: len(t.steps)):
        events = " -> ".join(trace.events) if trace.events else "(empty)"
        states = " -> ".join(trace.states)
        print(f"  Events: {events}")
        print(f"  States: {states}")
        print()

    print(f"Unique event sequences: {len(traces.event_sequences)}")

    return traces


if __name__ == "__main__":
    demo()
