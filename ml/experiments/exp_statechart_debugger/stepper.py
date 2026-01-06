"""
Stepper - Step-by-Step Execution for Statechart Debugging

Execution modes:
1. Run - Execute until breakpoint or end
2. Step - Execute one transition
3. Step Into - Step into nested statechart
4. Step Over - Step over nested statechart
5. Step Out - Run until exiting current state
6. Reverse Step - Step backward in history

Provides control flow for interactive debugging.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable, Iterator
from enum import Enum, auto
import copy


# =============================================================================
# Execution State
# =============================================================================

class ExecutionMode(Enum):
    """Modes of execution."""
    RUNNING = auto()      # Free running
    PAUSED = auto()       # Paused at breakpoint
    STEPPING = auto()     # Single stepping
    STOPPED = auto()      # Execution terminated
    REVERSE = auto()      # Reverse execution


class StepType(Enum):
    """Types of step commands."""
    STEP = auto()         # Single step (one transition)
    STEP_INTO = auto()    # Step into nested state
    STEP_OVER = auto()    # Step over nested state
    STEP_OUT = auto()     # Run until exit current state
    CONTINUE = auto()     # Continue running
    REVERSE = auto()      # Step backward


@dataclass
class ExecutionFrame:
    """
    A frame in the execution stack.

    Represents a point in execution that can be returned to.
    """
    sequence: int
    timestamp: float
    active_states: List[str]
    context: Dict[str, Any]
    parent_state: Optional[str] = None  # For nested statecharts
    event: Optional[str] = None
    transition: Optional[str] = None


@dataclass
class StepResult:
    """Result of a step operation."""
    step_type: StepType
    old_states: List[str]
    new_states: List[str]
    event: Optional[str] = None
    transition: Optional[str] = None
    context_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    completed: bool = False
    error: Optional[str] = None


# =============================================================================
# Simulated Statechart for Debugging
# =============================================================================

@dataclass
class DebugTransition:
    """A transition in the debuggable statechart."""
    label: str
    source: str
    target: str
    event: str
    guard: Optional[str] = None
    action: Optional[str] = None


@dataclass
class DebugState:
    """A state in the debuggable statechart."""
    name: str
    is_initial: bool = False
    is_final: bool = False
    is_composite: bool = False
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    entry_action: Optional[str] = None
    exit_action: Optional[str] = None


class DebugStatechart:
    """
    A statechart configured for debugging.

    Provides step-by-step execution with full state inspection.
    """

    def __init__(self):
        self.states: Dict[str, DebugState] = {}
        self.transitions: List[DebugTransition] = []
        self.initial_state: str = ""
        self.active_states: Set[str] = set()
        self.context: Dict[str, Any] = {}

        # Event queue
        self.event_queue: List[str] = []

        # Execution tracking
        self.sequence = 0
        self.timestamp = 0.0

    def add_state(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        is_composite: bool = False,
        parent: Optional[str] = None,
    ) -> DebugState:
        """Add a state to the statechart."""
        state = DebugState(
            name=name,
            is_initial=is_initial,
            is_final=is_final,
            is_composite=is_composite,
            parent=parent,
        )
        self.states[name] = state

        if is_initial and not self.initial_state:
            self.initial_state = name

        if parent and parent in self.states:
            self.states[parent].children.append(name)

        return state

    def add_transition(
        self,
        label: str,
        source: str,
        target: str,
        event: str,
        guard: Optional[str] = None,
        action: Optional[str] = None,
    ) -> DebugTransition:
        """Add a transition."""
        trans = DebugTransition(
            label=label,
            source=source,
            target=target,
            event=event,
            guard=guard,
            action=action,
        )
        self.transitions.append(trans)
        return trans

    def reset(self):
        """Reset to initial state."""
        self.active_states = {self.initial_state}
        self.context = {}
        self.event_queue = []
        self.sequence = 0
        self.timestamp = 0.0

    def get_enabled_transitions(self, event: str) -> List[DebugTransition]:
        """Get transitions enabled for given event from active states."""
        enabled = []
        for trans in self.transitions:
            if trans.event != event:
                continue
            if trans.source not in self.active_states:
                continue

            # Check guard
            if trans.guard:
                try:
                    if not eval(trans.guard, {"ctx": self.context, **self.context}):
                        continue
                except Exception:
                    continue

            enabled.append(trans)

        return enabled

    def fire_transition(self, trans: DebugTransition) -> StepResult:
        """Fire a single transition."""
        old_states = list(self.active_states)
        context_before = copy.deepcopy(self.context)

        # Exit source state
        self.active_states.discard(trans.source)

        # Execute action
        if trans.action:
            try:
                exec(trans.action, {"ctx": self.context})
            except Exception as e:
                pass

        # Enter target state
        self.active_states.add(trans.target)

        # Track context changes
        context_changes = {}
        for key in set(context_before.keys()) | set(self.context.keys()):
            old_val = context_before.get(key)
            new_val = self.context.get(key)
            if old_val != new_val:
                context_changes[key] = (old_val, new_val)

        self.sequence += 1
        self.timestamp += 0.1

        # Check if reached final state
        completed = any(
            self.states[s].is_final for s in self.active_states
            if s in self.states
        )

        return StepResult(
            step_type=StepType.STEP,
            old_states=old_states,
            new_states=list(self.active_states),
            event=trans.event,
            transition=trans.label,
            context_changes=context_changes,
            completed=completed,
        )

    def send_event(self, event: str):
        """Add event to queue."""
        self.event_queue.append(event)

    def has_pending_events(self) -> bool:
        """Check if there are pending events."""
        return len(self.event_queue) > 0


# =============================================================================
# Execution Stepper
# =============================================================================

class ExecutionStepper:
    """
    Controls step-by-step execution of a statechart.

    Features:
    - Step forward/backward
    - Step into/over nested states
    - Run to breakpoint
    - Execution history
    """

    def __init__(self, statechart: DebugStatechart):
        self.statechart = statechart
        self.mode = ExecutionMode.PAUSED

        # Execution history for reverse stepping
        self.history: List[ExecutionFrame] = []
        self.history_index = -1
        self.max_history = 1000

        # Step tracking
        self.step_depth = 0  # For step over/out
        self.target_state: Optional[str] = None  # For step out

        # Callbacks
        self.on_step: Optional[Callable[[StepResult], None]] = None
        self.on_state_change: Optional[Callable[[Set[str], Set[str]], None]] = None

    def _save_frame(self):
        """Save current state to history."""
        frame = ExecutionFrame(
            sequence=self.statechart.sequence,
            timestamp=self.statechart.timestamp,
            active_states=list(self.statechart.active_states),
            context=copy.deepcopy(self.statechart.context),
        )

        # Truncate history if at max
        if len(self.history) >= self.max_history:
            self.history = self.history[1:]

        self.history.append(frame)
        self.history_index = len(self.history) - 1

    def _restore_frame(self, index: int) -> bool:
        """Restore state from history."""
        if index < 0 or index >= len(self.history):
            return False

        frame = self.history[index]
        self.statechart.active_states = set(frame.active_states)
        self.statechart.context = copy.deepcopy(frame.context)
        self.statechart.sequence = frame.sequence
        self.statechart.timestamp = frame.timestamp
        self.history_index = index
        return True

    def start(self):
        """Start execution (initialize)."""
        self.statechart.reset()
        self.history.clear()
        self._save_frame()
        self.mode = ExecutionMode.PAUSED

    def step(self) -> Optional[StepResult]:
        """Execute one step (one transition)."""
        if self.mode == ExecutionMode.STOPPED:
            return None

        # Need an event to process
        if not self.statechart.has_pending_events():
            return StepResult(
                step_type=StepType.STEP,
                old_states=list(self.statechart.active_states),
                new_states=list(self.statechart.active_states),
                error="No pending events",
            )

        event = self.statechart.event_queue.pop(0)
        enabled = self.statechart.get_enabled_transitions(event)

        if not enabled:
            return StepResult(
                step_type=StepType.STEP,
                old_states=list(self.statechart.active_states),
                new_states=list(self.statechart.active_states),
                event=event,
                error=f"No enabled transitions for event '{event}'",
            )

        # Fire first enabled transition
        trans = enabled[0]
        result = self.statechart.fire_transition(trans)

        # Save to history
        self._save_frame()

        # Notify callback
        if self.on_step:
            self.on_step(result)

        if result.completed:
            self.mode = ExecutionMode.STOPPED

        return result

    def step_into(self) -> Optional[StepResult]:
        """Step into nested state."""
        result = self.step()
        if result:
            result.step_type = StepType.STEP_INTO
            self.step_depth += 1
        return result

    def step_over(self) -> Optional[StepResult]:
        """Step over nested state (run until same depth)."""
        start_depth = self.step_depth
        result = self.step()

        while result and not result.completed:
            # Check if back at same depth
            if self.step_depth <= start_depth:
                break
            result = self.step()

        if result:
            result.step_type = StepType.STEP_OVER
        return result

    def step_out(self) -> Optional[StepResult]:
        """Step out of current state."""
        if not self.statechart.active_states:
            return None

        # Track the state we're exiting
        current_states = set(self.statechart.active_states)
        result = None

        while True:
            result = self.step()
            if not result or result.completed:
                break

            # Check if we exited any of the original states
            if not current_states.intersection(self.statechart.active_states):
                break

        if result:
            result.step_type = StepType.STEP_OUT
        return result

    def continue_run(
        self,
        check_breakpoint: Optional[Callable[[], bool]] = None,
        max_steps: int = 10000,
    ) -> Optional[StepResult]:
        """Continue running until breakpoint or completion."""
        self.mode = ExecutionMode.RUNNING
        result = None
        steps = 0

        while steps < max_steps:
            if not self.statechart.has_pending_events():
                break

            result = self.step()
            steps += 1

            if not result or result.completed:
                break

            # Check for breakpoint
            if check_breakpoint and check_breakpoint():
                self.mode = ExecutionMode.PAUSED
                break

        return result

    def reverse_step(self) -> bool:
        """Step backward in history."""
        if self.history_index <= 0:
            return False

        self.mode = ExecutionMode.REVERSE
        return self._restore_frame(self.history_index - 1)

    def pause(self):
        """Pause execution."""
        self.mode = ExecutionMode.PAUSED

    def stop(self):
        """Stop execution."""
        self.mode = ExecutionMode.STOPPED

    def get_current_frame(self) -> Optional[ExecutionFrame]:
        """Get current execution frame."""
        if self.history_index >= 0 and self.history_index < len(self.history):
            return self.history[self.history_index]
        return None

    def get_history_length(self) -> int:
        """Get length of execution history."""
        return len(self.history)

    def can_step_back(self) -> bool:
        """Check if we can step backward."""
        return self.history_index > 0

    def can_step_forward(self) -> bool:
        """Check if we can step forward."""
        return (
            self.mode != ExecutionMode.STOPPED and
            self.statechart.has_pending_events()
        )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the stepper."""
    print("=" * 60)
    print("Execution Stepper Demo")
    print("=" * 60)

    # Create a simple statechart
    sc = DebugStatechart()

    sc.add_state("IDLE", is_initial=True)
    sc.add_state("RUNNING")
    sc.add_state("PAUSED")
    sc.add_state("STOPPED", is_final=True)

    sc.add_transition("t1", "IDLE", "RUNNING", "start")
    sc.add_transition("t2", "RUNNING", "PAUSED", "pause")
    sc.add_transition("t3", "PAUSED", "RUNNING", "resume")
    sc.add_transition("t4", "RUNNING", "STOPPED", "stop")
    sc.add_transition("t5", "PAUSED", "STOPPED", "stop")

    # Create stepper
    stepper = ExecutionStepper(sc)
    stepper.start()

    print(f"\nInitial state: {sc.active_states}")

    # Send events
    sc.send_event("start")
    sc.send_event("pause")
    sc.send_event("resume")
    sc.send_event("stop")

    print(f"Events queued: {sc.event_queue}")

    # Step through execution
    print("\n--- Stepping Through Execution ---")

    step_num = 0
    while stepper.can_step_forward():
        result = stepper.step()
        step_num += 1

        if result:
            print(f"\nStep {step_num}:")
            print(f"  Event: {result.event}")
            print(f"  Transition: {result.transition}")
            print(f"  States: {result.old_states} -> {result.new_states}")

            if result.completed:
                print("  EXECUTION COMPLETE")
                break

    # Show history
    print(f"\n--- Execution History ({stepper.get_history_length()} frames) ---")
    for i, frame in enumerate(stepper.history):
        print(f"  {i}: seq={frame.sequence} states={frame.active_states}")

    # Reverse stepping
    print("\n--- Reverse Stepping ---")
    while stepper.can_step_back():
        stepper.reverse_step()
        frame = stepper.get_current_frame()
        if frame:
            print(f"  <- seq={frame.sequence} states={frame.active_states}")

    return stepper


if __name__ == "__main__":
    demo()
