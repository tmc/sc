"""
Temporal Behavior Modeling for Statecharts.

Models time-based statechart behaviors:
- AFTER(duration) events: trigger after time in state
- Timeout patterns: idle timeout, retry with backoff
- Delayed transitions: wait before firing
- Time guards: conditions based on elapsed time

Temporal semantics follow Harel statecharts and UML state machines.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class TimeEvent:
    """A time-based event."""
    event_type: str  # "after", "at", "when"
    duration_ms: int = 0  # Duration in milliseconds
    expression: str = ""  # Original expression

    def __post_init__(self):
        if self.event_type == "after":
            self.expression = f"after({self.duration_ms}ms)"
        elif self.event_type == "at":
            self.expression = f"at({self.duration_ms})"


@dataclass
class TemporalState:
    """A state with temporal behaviors."""
    label: str
    entry_time: int = 0  # When state was entered (simulation time)
    timers: Dict[str, int] = field(default_factory=dict)  # timer_id -> expiry_time

    def enter(self, current_time: int):
        """Enter this state at the given time."""
        self.entry_time = current_time

    def elapsed(self, current_time: int) -> int:
        """Time elapsed since entering state."""
        return current_time - self.entry_time

    def set_timer(self, timer_id: str, duration_ms: int, current_time: int):
        """Set a timer to fire after duration."""
        self.timers[timer_id] = current_time + duration_ms

    def clear_timer(self, timer_id: str):
        """Clear a timer."""
        if timer_id in self.timers:
            del self.timers[timer_id]

    def get_expired_timers(self, current_time: int) -> List[str]:
        """Get list of expired timer IDs."""
        expired = []
        for timer_id, expiry in self.timers.items():
            if current_time >= expiry:
                expired.append(timer_id)
        return expired


@dataclass
class TemporalTransition:
    """A transition with temporal aspects."""
    source: str
    target: str
    event: str = ""  # Regular event
    after_ms: Optional[int] = None  # AFTER duration
    guard: str = ""  # Guard condition including time guards
    action: str = ""  # Action to execute
    resets_timer: bool = False  # Does this reset the source state timer?

    @property
    def is_temporal(self) -> bool:
        """Is this a time-triggered transition?"""
        return self.after_ms is not None or "time" in self.guard.lower()

    @property
    def trigger(self) -> str:
        """Get the trigger description."""
        if self.after_ms is not None:
            return f"after({self.after_ms}ms)"
        return self.event


@dataclass
class TemporalStatechart:
    """A statechart with temporal behaviors."""
    states: Dict[str, TemporalState] = field(default_factory=dict)
    transitions: List[TemporalTransition] = field(default_factory=list)
    initial_state: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)  # For guards/actions

    def add_state(self, label: str) -> TemporalState:
        """Add a state to the statechart."""
        state = TemporalState(label=label)
        self.states[label] = state
        return state

    def add_transition(
        self,
        source: str,
        target: str,
        event: str = "",
        after_ms: Optional[int] = None,
        guard: str = "",
        action: str = "",
        resets_timer: bool = False
    ) -> TemporalTransition:
        """Add a transition."""
        trans = TemporalTransition(
            source=source,
            target=target,
            event=event,
            after_ms=after_ms,
            guard=guard,
            action=action,
            resets_timer=resets_timer
        )
        self.transitions.append(trans)
        return trans

    def get_after_transitions(self, state: str) -> List[TemporalTransition]:
        """Get all AFTER transitions from a state."""
        return [t for t in self.transitions
                if t.source == state and t.after_ms is not None]

    def get_event_transitions(self, state: str, event: str) -> List[TemporalTransition]:
        """Get transitions triggered by an event."""
        return [t for t in self.transitions
                if t.source == state and t.event == event]


class TemporalSimulator:
    """Simulates temporal statechart execution."""

    def __init__(self, sc: TemporalStatechart):
        self.sc = sc
        self.current_state: str = ""
        self.current_time: int = 0
        self.variables: Dict[str, Any] = dict(sc.variables)
        self.history: List[Tuple[int, str, str]] = []  # (time, event, new_state)

    def reset(self):
        """Reset to initial state."""
        self.current_state = self.sc.initial_state
        self.current_time = 0
        self.variables = dict(self.sc.variables)
        self.history = [(0, "INIT", self.current_state)]

        # Enter initial state
        if self.current_state in self.sc.states:
            self.sc.states[self.current_state].enter(0)

    def advance_time(self, delta_ms: int) -> List[str]:
        """
        Advance simulation time and fire any expired AFTER transitions.
        Returns list of states visited.
        """
        self.current_time += delta_ms
        visited = []

        # Check for expired AFTER transitions
        after_transitions = self.sc.get_after_transitions(self.current_state)
        for trans in after_transitions:
            if trans.after_ms is None:
                continue

            state = self.sc.states.get(self.current_state)
            if state and state.elapsed(self.current_time) >= trans.after_ms:
                # Fire the transition
                self._execute_transition(trans)
                visited.append(self.current_state)

        return visited

    def send_event(self, event: str) -> bool:
        """
        Send an event to the statechart.
        Returns True if a transition fired.
        """
        transitions = self.sc.get_event_transitions(self.current_state, event)

        for trans in transitions:
            # Check guard
            if trans.guard and not self._eval_guard(trans.guard):
                continue

            # Fire transition
            self._execute_transition(trans)

            # Handle timer reset
            if trans.resets_timer and trans.target == trans.source:
                state = self.sc.states.get(self.current_state)
                if state:
                    state.enter(self.current_time)

            return True

        return False

    def _execute_transition(self, trans: TemporalTransition):
        """Execute a transition."""
        # Execute action
        if trans.action:
            self._execute_action(trans.action)

        # Record in history
        self.history.append((self.current_time, trans.trigger, trans.target))

        # Move to new state
        self.current_state = trans.target
        if self.current_state in self.sc.states:
            self.sc.states[self.current_state].enter(self.current_time)

    def _eval_guard(self, guard: str) -> bool:
        """Evaluate a guard condition."""
        # Simple guard evaluation
        try:
            # Replace variable references
            expr = guard
            for var, val in self.variables.items():
                expr = expr.replace(var, str(val))

            # Handle time-based guards
            if "elapsed" in expr.lower():
                state = self.sc.states.get(self.current_state)
                if state:
                    elapsed = state.elapsed(self.current_time)
                    expr = expr.replace("elapsed", str(elapsed))

            return eval(expr)
        except Exception:
            return False

    def _execute_action(self, action: str):
        """Execute an action."""
        # Simple action execution: var = expr or var++ or var--
        if "++" in action:
            var = action.replace("++", "").strip()
            if var in self.variables:
                self.variables[var] += 1
        elif "--" in action:
            var = action.replace("--", "").strip()
            if var in self.variables:
                self.variables[var] -= 1
        elif "=" in action:
            parts = action.split("=")
            if len(parts) == 2:
                var = parts[0].strip()
                try:
                    val = eval(parts[1].strip())
                    self.variables[var] = val
                except Exception:
                    pass


def parse_after_duration(expr: str) -> Optional[int]:
    """
    Parse an AFTER duration expression.
    Returns duration in milliseconds.

    Examples:
    - after(5s) -> 5000
    - after(100ms) -> 100
    - after(1m) -> 60000
    """
    match = re.match(r"after\((\d+)(ms|s|m|h)?\)", expr.lower())
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2) or "ms"

    multipliers = {
        "ms": 1,
        "s": 1000,
        "m": 60000,
        "h": 3600000
    }

    return value * multipliers.get(unit, 1)


def build_temporal_sc(spec: Dict[str, Any]) -> TemporalStatechart:
    """
    Build a TemporalStatechart from a specification dict.

    Spec format:
    {
        "states": ["S1", "S2", ...],
        "initial": "S1",
        "transitions": [
            {"from": "S1", "to": "S2", "event": "go"},
            {"from": "S2", "to": "S1", "after_ms": 5000},
            ...
        ],
        "variables": {"attempts": 0, ...}
    }
    """
    sc = TemporalStatechart()

    # Add states
    for state_label in spec.get("states", []):
        sc.add_state(state_label)

    # Set initial state
    sc.initial_state = spec.get("initial", "")

    # Add transitions
    for trans_spec in spec.get("transitions", []):
        sc.add_transition(
            source=trans_spec.get("from", ""),
            target=trans_spec.get("to", ""),
            event=trans_spec.get("event", ""),
            after_ms=trans_spec.get("after_ms"),
            guard=trans_spec.get("guard", ""),
            action=trans_spec.get("action", ""),
            resets_timer=trans_spec.get("resets_timer", False)
        )

    # Set variables
    sc.variables = dict(spec.get("variables", {}))

    return sc


def predict_temporal_behavior(
    sc: TemporalStatechart,
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Predict the temporal behavior given a scenario.

    Scenario format:
    {
        "duration_ms": 10000,  # How long to simulate
        "events": [(1000, "click"), (3000, "click"), ...]  # (time, event) pairs
    }

    Returns predicted state sequence and timing.
    """
    sim = TemporalSimulator(sc)
    sim.reset()

    events = sorted(scenario.get("events", []), key=lambda x: x[0])
    duration = scenario.get("duration_ms", 10000)

    # Simulate in small time steps
    step_ms = 100  # 100ms steps
    event_idx = 0

    state_sequence = [sim.current_state]

    for t in range(0, duration, step_ms):
        # Send any events at this time
        while event_idx < len(events) and events[event_idx][0] <= t:
            event_time, event_name = events[event_idx]
            sim.current_time = event_time
            if sim.send_event(event_name):
                state_sequence.append(sim.current_state)
            event_idx += 1

        # Advance time and check for timeouts
        visited = sim.advance_time(step_ms)
        state_sequence.extend(visited)

    return {
        "final_state": sim.current_state,
        "state_sequence": state_sequence,
        "history": sim.history,
        "variables": sim.variables
    }


if __name__ == "__main__":
    print("Temporal Behavior Test")
    print("=" * 60)

    # Test: Simple timeout
    spec = {
        "states": ["Idle", "Screensaver"],
        "initial": "Idle",
        "transitions": [
            {"from": "Idle", "to": "Screensaver", "after_ms": 30000},
            {"from": "Screensaver", "to": "Idle", "event": "activity"}
        ]
    }

    sc = build_temporal_sc(spec)
    result = predict_temporal_behavior(sc, {"duration_ms": 35000, "events": []})
    print(f"Simple timeout: {result['state_sequence']}")
    print(f"Final state: {result['final_state']}")

    # Test: Activity reset
    print("\n" + "=" * 60)
    spec2 = {
        "states": ["Active", "Idle"],
        "initial": "Active",
        "transitions": [
            {"from": "Active", "to": "Idle", "after_ms": 5000},
            {"from": "Active", "to": "Active", "event": "activity", "resets_timer": True},
            {"from": "Idle", "to": "Active", "event": "activity"}
        ]
    }

    sc2 = build_temporal_sc(spec2)
    # Activity at 2s and 4s should prevent timeout
    result2 = predict_temporal_behavior(
        sc2,
        {"duration_ms": 10000, "events": [(2000, "activity"), (4000, "activity")]}
    )
    print(f"Activity reset: {result2['state_sequence']}")
    print(f"History: {result2['history']}")
