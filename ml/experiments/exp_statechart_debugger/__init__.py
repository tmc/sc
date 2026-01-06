"""
exp_statechart_debugger - Interactive Debugging for Statechart Execution

This experiment implements a GDB/LLDB-style debugger for statecharts:

Components:
- breakpoint.py: Breakpoint system (state, transition, event, watchpoint)
- stepper.py: Step-by-step execution control
- inspector.py: Configuration and state inspection
- debug_session.py: Main debugger integrating all components
- benchmark.py: Performance testing

Features:
- Breakpoints on states (entry/exit/active) and transitions
- Conditional breakpoints with guard expressions
- Step execution (step, step-into, step-over, step-out)
- Reverse stepping through execution history
- Configuration inspection and diff
- Context variable watching
- ASCII timeline visualization

Usage:
    from exp_statechart_debugger import DebugSession

    session = DebugSession()
    session.add_state("IDLE", is_initial=True)
    session.add_state("RUNNING")
    session.add_state("STOPPED", is_final=True)
    session.add_transition("t1", "IDLE", "RUNNING", "start")
    session.add_transition("t2", "RUNNING", "STOPPED", "stop")

    session.break_on_state("RUNNING", BreakpointType.STATE_ENTRY)
    session.start()
    session.send_event("start")
    session.run()  # Runs until breakpoint at RUNNING entry
"""

from .breakpoint import (
    BreakpointType,
    Breakpoint,
    StateBreakpoint,
    TransitionBreakpoint,
    EventBreakpoint,
    Watchpoint,
    BreakpointHit,
    BreakpointManager,
)

from .stepper import (
    ExecutionMode,
    StepType,
    ExecutionFrame,
    StepResult,
    DebugState,
    DebugTransition,
    DebugStatechart,
    ExecutionStepper,
)

from .inspector import (
    StateInfo,
    TransitionInfo,
    ConfigurationSnapshot,
    ConfigurationInspector,
    ContextInspector,
    HistoryInspector,
    DiffInspector,
)

from .debug_session import (
    DebugCommand,
    CommandResult,
    DebugSession,
)

__all__ = [
    # Breakpoint module
    "BreakpointType",
    "Breakpoint",
    "StateBreakpoint",
    "TransitionBreakpoint",
    "EventBreakpoint",
    "Watchpoint",
    "BreakpointHit",
    "BreakpointManager",
    # Stepper module
    "ExecutionMode",
    "StepType",
    "ExecutionFrame",
    "StepResult",
    "DebugState",
    "DebugTransition",
    "DebugStatechart",
    "ExecutionStepper",
    # Inspector module
    "StateInfo",
    "TransitionInfo",
    "ConfigurationSnapshot",
    "ConfigurationInspector",
    "ContextInspector",
    "HistoryInspector",
    "DiffInspector",
    # Debug session
    "DebugCommand",
    "CommandResult",
    "DebugSession",
]
