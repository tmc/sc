"""
Debug Session - Main Debugger for Statecharts

Combines all debugging components:
- Breakpoints (state, transition, event, watchpoint)
- Stepper (step, step into/over/out, continue, reverse)
- Inspector (configuration, context, history, diff)

Provides a unified debugging interface similar to GDB/LLDB for statecharts.

Commands:
- break <state/transition> - Set breakpoint
- delete <id> - Delete breakpoint
- step / next / continue - Control execution
- print <var> - Print variable
- info states - Show active states
- backtrace - Show execution history
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto

try:
    from .breakpoint import (
        BreakpointManager, BreakpointHit, BreakpointType,
        StateBreakpoint, TransitionBreakpoint, EventBreakpoint, Watchpoint
    )
    from .stepper import (
        ExecutionStepper, DebugStatechart, DebugState, DebugTransition,
        ExecutionMode, StepType, StepResult, ExecutionFrame
    )
    from .inspector import (
        ConfigurationInspector, ContextInspector, HistoryInspector, DiffInspector,
        ConfigurationSnapshot
    )
except ImportError:
    from breakpoint import (
        BreakpointManager, BreakpointHit, BreakpointType,
        StateBreakpoint, TransitionBreakpoint, EventBreakpoint, Watchpoint
    )
    from stepper import (
        ExecutionStepper, DebugStatechart, DebugState, DebugTransition,
        ExecutionMode, StepType, StepResult, ExecutionFrame
    )
    from inspector import (
        ConfigurationInspector, ContextInspector, HistoryInspector, DiffInspector,
        ConfigurationSnapshot
    )


# =============================================================================
# Debug Session State
# =============================================================================

class DebugCommand(Enum):
    """Debug commands."""
    # Execution control
    RUN = auto()
    STEP = auto()
    STEP_INTO = auto()
    STEP_OVER = auto()
    STEP_OUT = auto()
    CONTINUE = auto()
    REVERSE = auto()
    STOP = auto()
    RESTART = auto()

    # Breakpoints
    BREAK_STATE = auto()
    BREAK_TRANSITION = auto()
    BREAK_EVENT = auto()
    WATCH = auto()
    DELETE_BREAK = auto()
    ENABLE_BREAK = auto()
    DISABLE_BREAK = auto()
    LIST_BREAKS = auto()

    # Inspection
    INFO_STATES = auto()
    INFO_TRANSITIONS = auto()
    INFO_CONTEXT = auto()
    PRINT = auto()
    BACKTRACE = auto()
    DIFF = auto()

    # Session
    HELP = auto()
    QUIT = auto()


@dataclass
class CommandResult:
    """Result of executing a debug command."""
    command: DebugCommand
    success: bool
    message: str
    data: Any = None


# =============================================================================
# Debug Session
# =============================================================================

class DebugSession:
    """
    Main debugging session for a statechart.

    Integrates:
    - BreakpointManager for breakpoints
    - ExecutionStepper for step execution
    - Inspectors for state viewing
    """

    def __init__(self, statechart: Optional[DebugStatechart] = None):
        # Create or use provided statechart
        self.statechart = statechart or DebugStatechart()

        # Components
        self.breakpoints = BreakpointManager()
        self.stepper = ExecutionStepper(self.statechart)
        self.config_inspector = ConfigurationInspector()
        self.context_inspector = ContextInspector()
        self.history_inspector = HistoryInspector()
        self.diff_inspector = DiffInspector()

        # Session state
        self.is_running = False
        self.current_hit: Optional[BreakpointHit] = None
        self.command_history: List[str] = []

        # Setup callbacks
        self.stepper.on_step = self._on_step

    def _on_step(self, result: StepResult):
        """Callback when step completes."""
        # Update inspectors
        self.config_inspector.update_active_states(self.statechart.active_states)
        self.config_inspector.update_enabled_transitions(
            self.statechart.active_states,
            self.statechart.context
        )
        self.context_inspector.update_context(
            self.statechart.sequence,
            self.statechart.context
        )

        if result.transition:
            self.config_inspector.record_transition_fired(result.transition)

        self.history_inspector.add_entry(
            sequence=self.statechart.sequence,
            timestamp=self.statechart.timestamp,
            event=result.event,
            transition=result.transition,
            source_states=result.old_states,
            target_states=result.new_states,
            context_changes=result.context_changes,
        )

        self.config_inspector.take_snapshot(
            self.statechart.sequence,
            self.statechart.timestamp,
            self.statechart.active_states,
            self.statechart.context,
            result.event,
            result.transition,
        )

    def load_statechart(self, statechart: DebugStatechart):
        """Load a statechart for debugging."""
        self.statechart = statechart
        self.stepper = ExecutionStepper(statechart)
        self.stepper.on_step = self._on_step

        # Register states and transitions with inspector
        for name, state in statechart.states.items():
            self.config_inspector.register_state(
                name,
                is_initial=state.is_initial,
                is_final=state.is_final,
                is_composite=state.is_composite,
                parent=state.parent,
            )

        for trans in statechart.transitions:
            self.config_inspector.register_transition(
                trans.label,
                trans.source,
                trans.target,
                trans.event,
                trans.guard,
            )

    def start(self):
        """Start debug session."""
        self.stepper.start()
        self.is_running = True
        self._on_step(StepResult(
            step_type=StepType.STEP,
            old_states=[],
            new_states=list(self.statechart.active_states),
        ))

    def _check_breakpoints(self) -> Optional[BreakpointHit]:
        """Check all breakpoints for current state."""
        context = self.statechart.context
        seq = self.statechart.sequence
        ts = self.statechart.timestamp

        # Check state breakpoints for active states
        for state in self.statechart.active_states:
            hit = self.breakpoints.check_state_entry(state, context, ts, seq)
            if hit:
                return hit

        return None

    def step(self) -> CommandResult:
        """Execute one step."""
        result = self.stepper.step()

        if not result:
            return CommandResult(
                DebugCommand.STEP, False, "Cannot step (no pending events)"
            )

        # Check breakpoints
        hit = self._check_breakpoints()
        if hit:
            self.current_hit = hit
            return CommandResult(
                DebugCommand.STEP, True,
                f"Breakpoint hit: {hit.reason}",
                data={"step": result, "breakpoint": hit}
            )

        return CommandResult(
            DebugCommand.STEP, True,
            f"Step: {result.old_states} -> {result.new_states}",
            data=result
        )

    def step_into(self) -> CommandResult:
        """Step into nested state."""
        result = self.stepper.step_into()
        if not result:
            return CommandResult(DebugCommand.STEP_INTO, False, "Cannot step into")
        return CommandResult(DebugCommand.STEP_INTO, True, f"Stepped into", data=result)

    def step_over(self) -> CommandResult:
        """Step over nested state."""
        result = self.stepper.step_over()
        if not result:
            return CommandResult(DebugCommand.STEP_OVER, False, "Cannot step over")
        return CommandResult(DebugCommand.STEP_OVER, True, "Stepped over", data=result)

    def step_out(self) -> CommandResult:
        """Step out of current state."""
        result = self.stepper.step_out()
        if not result:
            return CommandResult(DebugCommand.STEP_OUT, False, "Cannot step out")
        return CommandResult(DebugCommand.STEP_OUT, True, "Stepped out", data=result)

    def continue_run(self) -> CommandResult:
        """Continue running until breakpoint."""
        def check_bp():
            return self._check_breakpoints() is not None

        result = self.stepper.continue_run(check_breakpoint=check_bp)

        hit = self._check_breakpoints()
        if hit:
            self.current_hit = hit
            return CommandResult(
                DebugCommand.CONTINUE, True,
                f"Stopped at breakpoint: {hit.reason}",
                data=hit
            )

        if result and result.completed:
            return CommandResult(
                DebugCommand.CONTINUE, True,
                "Execution completed",
                data=result
            )

        return CommandResult(
            DebugCommand.CONTINUE, True,
            "Stopped (no pending events)"
        )

    def reverse_step(self) -> CommandResult:
        """Step backward in history."""
        if self.stepper.reverse_step():
            frame = self.stepper.get_current_frame()
            return CommandResult(
                DebugCommand.REVERSE, True,
                f"Reversed to seq={frame.sequence}",
                data=frame
            )
        return CommandResult(DebugCommand.REVERSE, False, "Cannot step backward")

    def stop(self) -> CommandResult:
        """Stop execution."""
        self.stepper.stop()
        self.is_running = False
        return CommandResult(DebugCommand.STOP, True, "Execution stopped")

    def restart(self) -> CommandResult:
        """Restart execution."""
        self.start()
        return CommandResult(DebugCommand.RESTART, True, "Execution restarted")

    # =========================================================================
    # Breakpoint Commands
    # =========================================================================

    def break_on_state(
        self,
        state: str,
        bp_type: BreakpointType = BreakpointType.STATE_ENTRY,
        condition: Optional[str] = None,
    ) -> CommandResult:
        """Set breakpoint on state."""
        bp_id = self.breakpoints.add_state_breakpoint(state, bp_type, condition)
        return CommandResult(
            DebugCommand.BREAK_STATE, True,
            f"Breakpoint #{bp_id} set on state '{state}'",
            data=bp_id
        )

    def break_on_transition(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        event: Optional[str] = None,
        label: str = "",
    ) -> CommandResult:
        """Set breakpoint on transition."""
        bp_id = self.breakpoints.add_transition_breakpoint(
            label=label, source=source, target=target, event=event
        )
        desc = f"{source or '*'} -> {target or '*'}" if not label else label
        return CommandResult(
            DebugCommand.BREAK_TRANSITION, True,
            f"Breakpoint #{bp_id} set on transition '{desc}'",
            data=bp_id
        )

    def break_on_event(self, event: str) -> CommandResult:
        """Set breakpoint on event."""
        bp_id = self.breakpoints.add_event_breakpoint(event)
        return CommandResult(
            DebugCommand.BREAK_EVENT, True,
            f"Breakpoint #{bp_id} set on event '{event}'",
            data=bp_id
        )

    def watch_variable(self, variable: str) -> CommandResult:
        """Set watchpoint on variable."""
        bp_id = self.breakpoints.add_watchpoint(variable)
        self.context_inspector.add_watch(variable)
        return CommandResult(
            DebugCommand.WATCH, True,
            f"Watchpoint #{bp_id} set on '{variable}'",
            data=bp_id
        )

    def delete_breakpoint(self, bp_id: int) -> CommandResult:
        """Delete a breakpoint."""
        if self.breakpoints.remove_breakpoint(bp_id):
            return CommandResult(
                DebugCommand.DELETE_BREAK, True,
                f"Breakpoint #{bp_id} deleted"
            )
        return CommandResult(
            DebugCommand.DELETE_BREAK, False,
            f"Breakpoint #{bp_id} not found"
        )

    def list_breakpoints(self) -> CommandResult:
        """List all breakpoints."""
        bps = self.breakpoints.list_breakpoints()
        return CommandResult(
            DebugCommand.LIST_BREAKS, True,
            f"{len(bps)} breakpoint(s)",
            data=bps
        )

    # =========================================================================
    # Inspection Commands
    # =========================================================================

    def info_states(self) -> CommandResult:
        """Get information about states."""
        active = self.config_inspector.get_active_states()
        data = {
            "active": [s.name for s in active],
            "all": list(self.config_inspector.state_info.keys()),
        }
        return CommandResult(
            DebugCommand.INFO_STATES, True,
            f"Active: {', '.join(data['active'])}",
            data=data
        )

    def info_transitions(self) -> CommandResult:
        """Get information about transitions."""
        enabled = self.config_inspector.get_enabled_transitions()
        data = {
            "enabled": [t.label for t in enabled],
            "all": list(self.config_inspector.transition_info.keys()),
        }
        return CommandResult(
            DebugCommand.INFO_TRANSITIONS, True,
            f"Enabled: {', '.join(data['enabled'])}",
            data=data
        )

    def info_context(self) -> CommandResult:
        """Get context information."""
        ctx = self.context_inspector.format_context()
        return CommandResult(
            DebugCommand.INFO_CONTEXT, True,
            ctx,
            data=self.statechart.context
        )

    def print_variable(self, variable: str) -> CommandResult:
        """Print a variable value."""
        value = self.statechart.context.get(variable)
        if value is not None:
            return CommandResult(
                DebugCommand.PRINT, True,
                f"{variable} = {value}",
                data=value
            )
        return CommandResult(
            DebugCommand.PRINT, False,
            f"Variable '{variable}' not found"
        )

    def backtrace(self, limit: int = 10) -> CommandResult:
        """Show execution backtrace."""
        timeline = self.history_inspector.format_ascii_timeline()
        return CommandResult(
            DebugCommand.BACKTRACE, True,
            timeline,
            data=self.history_inspector.entries[-limit:]
        )

    def diff(self, seq1: int, seq2: int) -> CommandResult:
        """Diff two snapshots."""
        snaps = self.config_inspector.snapshots

        if seq1 >= len(snaps) or seq2 >= len(snaps):
            return CommandResult(
                DebugCommand.DIFF, False,
                f"Invalid snapshot indices"
            )

        diff = self.diff_inspector.diff_snapshots(snaps[seq1], snaps[seq2])
        formatted = self.diff_inspector.format_diff(diff)

        return CommandResult(
            DebugCommand.DIFF, True,
            formatted,
            data=diff
        )

    # =========================================================================
    # Event Sending
    # =========================================================================

    def send_event(self, event: str):
        """Send an event to the statechart."""
        self.statechart.send_event(event)

    def send_events(self, events: List[str]):
        """Send multiple events."""
        for event in events:
            self.statechart.send_event(event)

    # =========================================================================
    # Session State
    # =========================================================================

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current debug state."""
        return {
            "mode": self.stepper.mode.name,
            "sequence": self.statechart.sequence,
            "active_states": list(self.statechart.active_states),
            "pending_events": len(self.statechart.event_queue),
            "breakpoints": len(self.breakpoints.breakpoints),
            "history_length": self.stepper.get_history_length(),
            "can_step_forward": self.stepper.can_step_forward(),
            "can_step_back": self.stepper.can_step_back(),
        }

    def format_prompt(self) -> str:
        """Format debug prompt."""
        mode = self.stepper.mode.name[:3]
        seq = self.statechart.sequence
        states = ", ".join(list(self.statechart.active_states)[:3])
        return f"[{mode}:{seq}] ({states})> "


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the debug session."""
    print("=" * 60)
    print("Debug Session Demo")
    print("=" * 60)

    # Create a statechart
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

    # Create debug session
    session = DebugSession()
    session.load_statechart(sc)

    # Set breakpoints
    print("\n--- Setting Breakpoints ---")
    result = session.break_on_state("RUNNING")
    print(result.message)

    result = session.break_on_event("stop")
    print(result.message)

    # Start session
    print("\n--- Starting Debug Session ---")
    session.start()
    print(f"Initial state: {session.info_states().message}")

    # Send events
    session.send_events(["start", "pause", "resume", "stop"])
    print(f"Events queued: {sc.event_queue}")

    # Step through
    print("\n--- Stepping Through Execution ---")

    for i in range(10):
        print(f"\n{session.format_prompt()}", end="")

        result = session.step()
        print(result.message)

        if result.data and isinstance(result.data, dict) and "breakpoint" in result.data:
            print(f"  [BREAKPOINT HIT]")

        if not session.stepper.can_step_forward():
            print("  [NO MORE STEPS]")
            break

    # Show backtrace
    print("\n--- Backtrace ---")
    result = session.backtrace()
    print(result.message)

    # Show state summary
    print("\n--- Session Summary ---")
    summary = session.get_state_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return session


if __name__ == "__main__":
    demo()
