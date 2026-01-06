"""
Breakpoint System for Statechart Debugging

Breakpoint types:
1. State breakpoints - Break when entering/exiting a state
2. Transition breakpoints - Break when a transition fires
3. Event breakpoints - Break when an event is received
4. Conditional breakpoints - Break when condition is true
5. Watchpoints - Break when variable changes

Based on traditional debugger concepts adapted for statecharts.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
import re


# =============================================================================
# Breakpoint Types
# =============================================================================

class BreakpointType(Enum):
    """Types of breakpoints."""
    STATE_ENTRY = auto()      # Break on entering a state
    STATE_EXIT = auto()       # Break on exiting a state
    STATE_ACTIVE = auto()     # Break when state is active
    TRANSITION = auto()        # Break when transition fires
    EVENT = auto()            # Break on event received
    CONDITION = auto()        # Break when condition true
    WATCHPOINT = auto()       # Break when variable changes


class BreakpointState(Enum):
    """State of a breakpoint."""
    ENABLED = auto()
    DISABLED = auto()
    HIT = auto()             # Currently at this breakpoint
    EXPIRED = auto()         # One-shot breakpoint that has fired


@dataclass
class BreakpointHit:
    """Information about a breakpoint being hit."""
    breakpoint_id: int
    breakpoint_type: BreakpointType
    location: str              # State/transition name
    reason: str                # Why it was hit
    context: Dict[str, Any]    # Current context
    timestamp: float = 0.0
    sequence: int = 0


# =============================================================================
# Breakpoint Classes
# =============================================================================

@dataclass
class Breakpoint:
    """Base breakpoint class."""
    id: int
    bp_type: BreakpointType
    state: BreakpointState = BreakpointState.ENABLED
    hit_count: int = 0
    ignore_count: int = 0      # Skip this many hits before breaking
    one_shot: bool = False     # Delete after first hit
    condition: Optional[str] = None  # Condition expression

    def should_break(self, context: Dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if self.state != BreakpointState.ENABLED:
            return False

        # Check ignore count
        if self.ignore_count > 0:
            self.ignore_count -= 1
            self.hit_count += 1
            return False

        # Check condition
        if self.condition:
            try:
                if not eval(self.condition, {"ctx": context, **context}):
                    return False
            except Exception:
                return False

        self.hit_count += 1

        # Handle one-shot
        if self.one_shot:
            self.state = BreakpointState.EXPIRED

        return True


@dataclass
class StateBreakpoint(Breakpoint):
    """Breakpoint on state entry/exit/active."""
    state_name: str = ""
    state_pattern: Optional[str] = None  # Regex pattern

    def matches_state(self, state: str) -> bool:
        """Check if state matches this breakpoint."""
        if self.state_pattern:
            return bool(re.match(self.state_pattern, state))
        return state == self.state_name


@dataclass
class TransitionBreakpoint(Breakpoint):
    """Breakpoint on transition firing."""
    transition_label: str = ""
    source_state: Optional[str] = None
    target_state: Optional[str] = None
    event_type: Optional[str] = None

    def matches_transition(
        self,
        label: str,
        source: str,
        target: str,
        event: str
    ) -> bool:
        """Check if transition matches this breakpoint."""
        if self.transition_label and self.transition_label != label:
            return False
        if self.source_state and self.source_state != source:
            return False
        if self.target_state and self.target_state != target:
            return False
        if self.event_type and self.event_type != event:
            return False
        return True


@dataclass
class EventBreakpoint(Breakpoint):
    """Breakpoint on event reception."""
    event_type: str = ""
    event_pattern: Optional[str] = None

    def matches_event(self, event_type: str) -> bool:
        """Check if event matches this breakpoint."""
        if self.event_pattern:
            return bool(re.match(self.event_pattern, event_type))
        return event_type == self.event_type


@dataclass
class Watchpoint(Breakpoint):
    """Breakpoint on variable change."""
    variable_name: str = ""
    previous_value: Any = None
    watch_for: str = "change"  # "change", "read", "write"

    def check_change(self, new_value: Any) -> bool:
        """Check if value changed."""
        changed = new_value != self.previous_value
        self.previous_value = new_value
        return changed


# =============================================================================
# Breakpoint Manager
# =============================================================================

class BreakpointManager:
    """
    Manages all breakpoints for a debug session.

    Provides:
    - Add/remove breakpoints
    - Enable/disable breakpoints
    - Check if should break
    - List all breakpoints
    """

    def __init__(self):
        self.breakpoints: Dict[int, Breakpoint] = {}
        self.next_id = 1
        self.hit_history: List[BreakpointHit] = []

        # Index for fast lookup
        self.state_breakpoints: Dict[str, List[int]] = {}
        self.transition_breakpoints: List[int] = []
        self.event_breakpoints: Dict[str, List[int]] = {}
        self.watchpoints: Dict[str, List[int]] = {}

    def _allocate_id(self) -> int:
        """Allocate a unique breakpoint ID."""
        bp_id = self.next_id
        self.next_id += 1
        return bp_id

    def add_state_breakpoint(
        self,
        state_name: str,
        bp_type: BreakpointType = BreakpointType.STATE_ENTRY,
        condition: Optional[str] = None,
        one_shot: bool = False,
    ) -> int:
        """Add a state breakpoint."""
        bp_id = self._allocate_id()
        bp = StateBreakpoint(
            id=bp_id,
            bp_type=bp_type,
            state_name=state_name,
            condition=condition,
            one_shot=one_shot,
        )
        self.breakpoints[bp_id] = bp

        if state_name not in self.state_breakpoints:
            self.state_breakpoints[state_name] = []
        self.state_breakpoints[state_name].append(bp_id)

        return bp_id

    def add_transition_breakpoint(
        self,
        label: str = "",
        source: Optional[str] = None,
        target: Optional[str] = None,
        event: Optional[str] = None,
        condition: Optional[str] = None,
        one_shot: bool = False,
    ) -> int:
        """Add a transition breakpoint."""
        bp_id = self._allocate_id()
        bp = TransitionBreakpoint(
            id=bp_id,
            bp_type=BreakpointType.TRANSITION,
            transition_label=label,
            source_state=source,
            target_state=target,
            event_type=event,
            condition=condition,
            one_shot=one_shot,
        )
        self.breakpoints[bp_id] = bp
        self.transition_breakpoints.append(bp_id)
        return bp_id

    def add_event_breakpoint(
        self,
        event_type: str,
        condition: Optional[str] = None,
        one_shot: bool = False,
    ) -> int:
        """Add an event breakpoint."""
        bp_id = self._allocate_id()
        bp = EventBreakpoint(
            id=bp_id,
            bp_type=BreakpointType.EVENT,
            event_type=event_type,
            condition=condition,
            one_shot=one_shot,
        )
        self.breakpoints[bp_id] = bp

        if event_type not in self.event_breakpoints:
            self.event_breakpoints[event_type] = []
        self.event_breakpoints[event_type].append(bp_id)

        return bp_id

    def add_watchpoint(
        self,
        variable: str,
        watch_for: str = "change",
        condition: Optional[str] = None,
    ) -> int:
        """Add a watchpoint on a variable."""
        bp_id = self._allocate_id()
        bp = Watchpoint(
            id=bp_id,
            bp_type=BreakpointType.WATCHPOINT,
            variable_name=variable,
            watch_for=watch_for,
            condition=condition,
        )
        self.breakpoints[bp_id] = bp

        if variable not in self.watchpoints:
            self.watchpoints[variable] = []
        self.watchpoints[variable].append(bp_id)

        return bp_id

    def remove_breakpoint(self, bp_id: int) -> bool:
        """Remove a breakpoint."""
        if bp_id not in self.breakpoints:
            return False

        bp = self.breakpoints[bp_id]

        # Remove from indices
        if isinstance(bp, StateBreakpoint):
            if bp.state_name in self.state_breakpoints:
                self.state_breakpoints[bp.state_name].remove(bp_id)
        elif isinstance(bp, TransitionBreakpoint):
            self.transition_breakpoints.remove(bp_id)
        elif isinstance(bp, EventBreakpoint):
            if bp.event_type in self.event_breakpoints:
                self.event_breakpoints[bp.event_type].remove(bp_id)
        elif isinstance(bp, Watchpoint):
            if bp.variable_name in self.watchpoints:
                self.watchpoints[bp.variable_name].remove(bp_id)

        del self.breakpoints[bp_id]
        return True

    def enable_breakpoint(self, bp_id: int) -> bool:
        """Enable a breakpoint."""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].state = BreakpointState.ENABLED
            return True
        return False

    def disable_breakpoint(self, bp_id: int) -> bool:
        """Disable a breakpoint."""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].state = BreakpointState.DISABLED
            return True
        return False

    def check_state_entry(
        self,
        state: str,
        context: Dict[str, Any],
        timestamp: float = 0.0,
        sequence: int = 0,
    ) -> Optional[BreakpointHit]:
        """Check if we should break on state entry."""
        bp_ids = self.state_breakpoints.get(state, [])

        for bp_id in bp_ids:
            bp = self.breakpoints.get(bp_id)
            if not isinstance(bp, StateBreakpoint):
                continue
            if bp.bp_type != BreakpointType.STATE_ENTRY:
                continue
            if not bp.matches_state(state):
                continue
            if bp.should_break(context):
                hit = BreakpointHit(
                    breakpoint_id=bp_id,
                    breakpoint_type=BreakpointType.STATE_ENTRY,
                    location=state,
                    reason=f"Entered state '{state}'",
                    context=context.copy(),
                    timestamp=timestamp,
                    sequence=sequence,
                )
                self.hit_history.append(hit)
                return hit

        return None

    def check_state_exit(
        self,
        state: str,
        context: Dict[str, Any],
        timestamp: float = 0.0,
        sequence: int = 0,
    ) -> Optional[BreakpointHit]:
        """Check if we should break on state exit."""
        bp_ids = self.state_breakpoints.get(state, [])

        for bp_id in bp_ids:
            bp = self.breakpoints.get(bp_id)
            if not isinstance(bp, StateBreakpoint):
                continue
            if bp.bp_type != BreakpointType.STATE_EXIT:
                continue
            if not bp.matches_state(state):
                continue
            if bp.should_break(context):
                hit = BreakpointHit(
                    breakpoint_id=bp_id,
                    breakpoint_type=BreakpointType.STATE_EXIT,
                    location=state,
                    reason=f"Exited state '{state}'",
                    context=context.copy(),
                    timestamp=timestamp,
                    sequence=sequence,
                )
                self.hit_history.append(hit)
                return hit

        return None

    def check_transition(
        self,
        label: str,
        source: str,
        target: str,
        event: str,
        context: Dict[str, Any],
        timestamp: float = 0.0,
        sequence: int = 0,
    ) -> Optional[BreakpointHit]:
        """Check if we should break on transition."""
        for bp_id in self.transition_breakpoints:
            bp = self.breakpoints.get(bp_id)
            if not isinstance(bp, TransitionBreakpoint):
                continue
            if not bp.matches_transition(label, source, target, event):
                continue
            if bp.should_break(context):
                hit = BreakpointHit(
                    breakpoint_id=bp_id,
                    breakpoint_type=BreakpointType.TRANSITION,
                    location=f"{source} -> {target}",
                    reason=f"Transition '{label}' fired on '{event}'",
                    context=context.copy(),
                    timestamp=timestamp,
                    sequence=sequence,
                )
                self.hit_history.append(hit)
                return hit

        return None

    def check_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        timestamp: float = 0.0,
        sequence: int = 0,
    ) -> Optional[BreakpointHit]:
        """Check if we should break on event."""
        bp_ids = self.event_breakpoints.get(event_type, [])

        for bp_id in bp_ids:
            bp = self.breakpoints.get(bp_id)
            if not isinstance(bp, EventBreakpoint):
                continue
            if not bp.matches_event(event_type):
                continue
            if bp.should_break(context):
                hit = BreakpointHit(
                    breakpoint_id=bp_id,
                    breakpoint_type=BreakpointType.EVENT,
                    location=event_type,
                    reason=f"Event '{event_type}' received",
                    context=context.copy(),
                    timestamp=timestamp,
                    sequence=sequence,
                )
                self.hit_history.append(hit)
                return hit

        return None

    def check_watchpoint(
        self,
        variable: str,
        new_value: Any,
        context: Dict[str, Any],
        timestamp: float = 0.0,
        sequence: int = 0,
    ) -> Optional[BreakpointHit]:
        """Check if we should break on variable change."""
        bp_ids = self.watchpoints.get(variable, [])

        for bp_id in bp_ids:
            bp = self.breakpoints.get(bp_id)
            if not isinstance(bp, Watchpoint):
                continue
            if not bp.check_change(new_value):
                continue
            if bp.should_break(context):
                hit = BreakpointHit(
                    breakpoint_id=bp_id,
                    breakpoint_type=BreakpointType.WATCHPOINT,
                    location=variable,
                    reason=f"Variable '{variable}' changed to {new_value}",
                    context=context.copy(),
                    timestamp=timestamp,
                    sequence=sequence,
                )
                self.hit_history.append(hit)
                return hit

        return None

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints."""
        result = []
        for bp_id, bp in self.breakpoints.items():
            info = {
                "id": bp_id,
                "type": bp.bp_type.name,
                "state": bp.state.name,
                "hit_count": bp.hit_count,
                "one_shot": bp.one_shot,
                "condition": bp.condition,
            }

            if isinstance(bp, StateBreakpoint):
                info["location"] = bp.state_name
            elif isinstance(bp, TransitionBreakpoint):
                info["location"] = bp.transition_label or f"{bp.source_state} -> {bp.target_state}"
            elif isinstance(bp, EventBreakpoint):
                info["location"] = bp.event_type
            elif isinstance(bp, Watchpoint):
                info["location"] = bp.variable_name

            result.append(info)

        return result

    def clear_all(self):
        """Clear all breakpoints."""
        self.breakpoints.clear()
        self.state_breakpoints.clear()
        self.transition_breakpoints.clear()
        self.event_breakpoints.clear()
        self.watchpoints.clear()
        self.hit_history.clear()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate breakpoint system."""
    print("=" * 60)
    print("Breakpoint System Demo")
    print("=" * 60)

    manager = BreakpointManager()

    # Add various breakpoints
    bp1 = manager.add_state_breakpoint("RUNNING", BreakpointType.STATE_ENTRY)
    print(f"Added state entry breakpoint on RUNNING: #{bp1}")

    bp2 = manager.add_transition_breakpoint(source="IDLE", target="RUNNING")
    print(f"Added transition breakpoint IDLE->RUNNING: #{bp2}")

    bp3 = manager.add_event_breakpoint("start")
    print(f"Added event breakpoint on 'start': #{bp3}")

    bp4 = manager.add_watchpoint("counter", watch_for="change")
    print(f"Added watchpoint on 'counter': #{bp4}")

    bp5 = manager.add_state_breakpoint(
        "ERROR",
        BreakpointType.STATE_ENTRY,
        condition="ctx.get('error_count', 0) > 3",
    )
    print(f"Added conditional breakpoint on ERROR: #{bp5}")

    # List breakpoints
    print("\n--- All Breakpoints ---")
    for bp in manager.list_breakpoints():
        print(f"  #{bp['id']}: {bp['type']} at '{bp['location']}' [{bp['state']}]")

    # Simulate execution
    print("\n--- Simulating Execution ---")
    context = {"counter": 0, "error_count": 0}

    # Check event breakpoint
    hit = manager.check_event("start", context, timestamp=1.0, sequence=1)
    if hit:
        print(f"BREAK: {hit.reason}")

    # Check transition breakpoint
    hit = manager.check_transition(
        "t1", "IDLE", "RUNNING", "start", context, timestamp=2.0, sequence=2
    )
    if hit:
        print(f"BREAK: {hit.reason}")

    # Check state entry
    hit = manager.check_state_entry("RUNNING", context, timestamp=3.0, sequence=3)
    if hit:
        print(f"BREAK: {hit.reason}")

    # Check watchpoint
    context["counter"] = 1
    hit = manager.check_watchpoint("counter", 1, context, timestamp=4.0, sequence=4)
    if hit:
        print(f"BREAK: {hit.reason}")

    # Conditional breakpoint (should not trigger)
    context["error_count"] = 2
    hit = manager.check_state_entry("ERROR", context, timestamp=5.0, sequence=5)
    if hit:
        print(f"BREAK: {hit.reason}")
    else:
        print("Conditional breakpoint NOT hit (error_count=2, need >3)")

    # Now it should trigger
    context["error_count"] = 5
    hit = manager.check_state_entry("ERROR", context, timestamp=6.0, sequence=6)
    if hit:
        print(f"BREAK: {hit.reason}")

    # Show hit history
    print("\n--- Breakpoint Hit History ---")
    for i, hit in enumerate(manager.hit_history):
        print(f"  {i+1}. #{hit.breakpoint_id} at seq={hit.sequence}: {hit.reason}")

    return manager


if __name__ == "__main__":
    demo()
