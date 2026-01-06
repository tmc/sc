"""
Parent Trigger - Trigger parent transitions when children complete.

When a PARALLEL state's children all reach final states, the parent
should take a completion transition (τ event).

FORMAL SEMANTICS:
1. Completion event τ is generated when:
   - A BASIC final state is entered, OR
   - A NORMAL state's active child completes, OR
   - A PARALLEL state's ALL children complete

2. Completion transition enabled when:
   enabled(t, σ) ⟺ t.event = τ ∧ t.source ∈ σ ∧ complete(t.source, σ)

3. Completion transitions have priority (fire immediately).

This module:
- Monitors completion events
- Finds enabled completion transitions
- Executes parent transitions
- Tracks completion chains
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable, Tuple, Any
from enum import Enum, auto
import copy

from .terminal_state import State, StateType, Configuration, Transition
from .completion_detector import CompletionDetector, CompletionEvent


@dataclass
class TransitionExecution:
    """Record of a triggered transition."""
    transition: Transition
    source_config: Set[str]
    target_config: Set[str]
    triggered_by: str  # "event", "completion", "propagation"
    step: int


class ParentTriggerEngine:
    """Execute completion-triggered transitions."""

    def __init__(
        self,
        root: State,
        transitions: List[Transition],
    ):
        self.root = root
        self.transitions = transitions
        self.state_map = self._build_state_map(root)
        self.detector = CompletionDetector(root)
        self.config = Configuration(active_states=set())
        self.step = 0
        self.history: List[TransitionExecution] = []

    def _build_state_map(self, state: State) -> Dict[str, State]:
        result = {state.label: state}
        for child in state.children:
            result.update(self._build_state_map(child))
        return result

    def initialize(self):
        """Initialize to default configuration."""
        self.config = Configuration(active_states=set())
        self._enter_default(self.root)
        self.step = 0
        self.history = []
        self.detector.reset()

    def _enter_default(self, state: State):
        """Enter state and its default substates."""
        self.config.add(state.label)

        if state.state_type == StateType.NORMAL:
            # Enter initial child
            for child in state.children:
                if child.is_initial:
                    self._enter_default(child)
                    break
        elif state.state_type == StateType.PARALLEL:
            # Enter all children
            for child in state.children:
                self._enter_default(child)

    def get_enabled_event_transitions(self, event: str) -> List[Transition]:
        """Get transitions enabled by event."""
        enabled = []
        for t in self.transitions:
            if t.event == event:
                for src in t.source:
                    if self.config.is_active(src):
                        enabled.append(t)
                        break
        return enabled

    def get_enabled_completion_transitions(self) -> List[Transition]:
        """Get completion transitions enabled by current state."""
        enabled = []

        for t in self.transitions:
            if not t.is_completion_transition():
                continue

            for src_label in t.source:
                if not self.config.is_active(src_label):
                    continue

                state = self.state_map.get(src_label)
                if state and self._is_complete(state):
                    enabled.append(t)
                    break

        return enabled

    def _is_complete(self, state: State) -> bool:
        """Check if state is in completion state."""
        if state.is_final:
            return True

        if state.state_type == StateType.BASIC:
            return state.is_final

        elif state.state_type == StateType.NORMAL:
            # Complete if active child is complete
            for child in state.children:
                if self.config.is_active(child.label):
                    return self._is_complete(child)
            return False

        elif state.state_type == StateType.PARALLEL:
            # Complete if ALL children are complete
            for child in state.children:
                if not self._is_complete(child):
                    return False
            return True

        return False

    def fire_transition(
        self,
        transition: Transition,
        triggered_by: str = "event",
    ) -> TransitionExecution:
        """Execute a transition."""
        self.step += 1

        old_config = copy.copy(self.config.active_states)

        # Exit source states
        for src_label in transition.source:
            self._exit_state(src_label)

        # Enter target states
        for tgt_label in transition.target:
            if tgt_label in self.state_map:
                self._enter_state(tgt_label)

        execution = TransitionExecution(
            transition=transition,
            source_config=old_config,
            target_config=copy.copy(self.config.active_states),
            triggered_by=triggered_by,
            step=self.step,
        )
        self.history.append(execution)

        return execution

    def _exit_state(self, label: str):
        """Exit state and all its active descendants."""
        state = self.state_map.get(label)
        if not state:
            return

        # Exit children first (bottom-up)
        for child in state.children:
            if self.config.is_active(child.label):
                self._exit_state(child.label)

        self.config.remove(label)

    def _enter_state(self, label: str):
        """Enter state and its default substates."""
        state = self.state_map.get(label)
        if not state:
            return

        # Enter ancestors first
        if state.parent and not self.config.is_active(state.parent.label):
            self._enter_state(state.parent.label)

        self.config.add(label)
        self._enter_default(state)

    def process_event(self, event: str) -> List[TransitionExecution]:
        """
        Process event and any resulting completions.
        Returns all transitions fired.
        """
        executions = []

        # First, fire event-triggered transitions
        enabled = self.get_enabled_event_transitions(event)
        for t in enabled:
            exec_record = self.fire_transition(t, "event")
            executions.append(exec_record)

        # Then, fire any completion transitions (chain reaction)
        executions.extend(self._process_completions())

        return executions

    def _process_completions(self) -> List[TransitionExecution]:
        """Process completion transitions until stable."""
        executions = []

        while True:
            # Update completion status
            self.detector.update(self.config)

            # Find enabled completion transitions
            enabled = self.get_enabled_completion_transitions()
            if not enabled:
                break

            # Fire first enabled completion transition
            t = enabled[0]
            exec_record = self.fire_transition(t, "completion")
            executions.append(exec_record)

        return executions

    def get_state_status(self) -> Dict[str, str]:
        """Get current state status."""
        status = {}
        for label, state in self.state_map.items():
            if self.config.is_active(label):
                if state.is_final:
                    status[label] = "FINAL"
                else:
                    status[label] = "ACTIVE"
            else:
                status[label] = "INACTIVE"
        return status


class CompletionChainTracker:
    """Track chains of completion transitions."""

    def __init__(self):
        self.chains: List[List[TransitionExecution]] = []
        self.current_chain: List[TransitionExecution] = []

    def start_chain(self):
        """Start a new completion chain."""
        if self.current_chain:
            self.chains.append(self.current_chain)
        self.current_chain = []

    def add_execution(self, execution: TransitionExecution):
        """Add execution to current chain."""
        self.current_chain.append(execution)

    def end_chain(self):
        """End current chain."""
        if self.current_chain:
            self.chains.append(self.current_chain)
            self.current_chain = []

    def get_chain_lengths(self) -> List[int]:
        """Get lengths of all chains."""
        lengths = [len(c) for c in self.chains]
        if self.current_chain:
            lengths.append(len(self.current_chain))
        return lengths

    def get_longest_chain(self) -> List[TransitionExecution]:
        """Get longest completion chain."""
        all_chains = self.chains + ([self.current_chain] if self.current_chain else [])
        if not all_chains:
            return []
        return max(all_chains, key=len)


def build_completion_chain_example() -> Tuple[State, List[Transition]]:
    """
    Build statechart that demonstrates completion chain.

    Outer (PARALLEL)
    ├── Region1 (NORMAL)
    │   ├── R1_Start (initial)
    │   ├── R1_Working
    │   └── R1_Done (final)
    └── Region2 (NORMAL)
        ├── R2_Start (initial)
        ├── R2_Working
        └── R2_Done (final)

    When Outer completes -> transition to AllDone
    """
    # Region 1
    r1_start = State(label="R1_Start", is_initial=True)
    r1_working = State(label="R1_Working")
    r1_done = State(label="R1_Done", is_final=True)
    region1 = State(
        label="Region1",
        state_type=StateType.NORMAL,
        children=[r1_start, r1_working, r1_done],
    )

    # Region 2
    r2_start = State(label="R2_Start", is_initial=True)
    r2_working = State(label="R2_Working")
    r2_done = State(label="R2_Done", is_final=True)
    region2 = State(
        label="Region2",
        state_type=StateType.NORMAL,
        children=[r2_start, r2_working, r2_done],
    )

    # Outer parallel
    outer = State(
        label="Outer",
        state_type=StateType.PARALLEL,
        children=[region1, region2],
        is_initial=True,
    )

    # Final state
    all_done = State(label="AllDone", is_final=True)

    # Root
    root = State(
        label="Root",
        state_type=StateType.NORMAL,
        children=[outer, all_done],
    )

    # Transitions
    transitions = [
        # Region 1 flow
        Transition(source=["R1_Start"], target=["R1_Working"], event="start1"),
        Transition(source=["R1_Working"], target=["R1_Done"], event="finish1"),

        # Region 2 flow
        Transition(source=["R2_Start"], target=["R2_Working"], event="start2"),
        Transition(source=["R2_Working"], target=["R2_Done"], event="finish2"),

        # Completion: Outer -> AllDone when all regions done
        Transition(source=["Outer"], target=["AllDone"], event=""),  # τ
    ]

    return root, transitions


def demo():
    """Demonstrate parent trigger engine."""
    print("=" * 60)
    print("PARENT TRIGGER ENGINE")
    print("=" * 60)

    root, transitions = build_completion_chain_example()

    print("\nStatechart structure:")
    def print_tree(state, indent=0):
        mark = "*" if state.is_final else ""
        init = "→" if state.is_initial else ""
        type_str = f" ({state.state_type.name})" if state.state_type != StateType.BASIC else ""
        print("  " * indent + f"{init}{state.label}{mark}{type_str}")
        for child in state.children:
            print_tree(child, indent + 1)
    print_tree(root)

    print("\nTransitions:")
    for t in transitions:
        event = t.event if t.event else "τ (completion)"
        print(f"  {t.source} --[{event}]--> {t.target}")

    # Create engine
    engine = ParentTriggerEngine(root, transitions)
    engine.initialize()

    print("\n" + "-" * 60)
    print("EXECUTION SIMULATION")
    print("-" * 60)

    print("\nInitial configuration:")
    status = engine.get_state_status()
    active = [k for k, v in status.items() if v in ("ACTIVE", "FINAL")]
    print(f"  Active: {active}")

    # Process events
    events_to_send = [
        "start1",   # R1: Start -> Working
        "start2",   # R2: Start -> Working
        "finish1",  # R1: Working -> Done (final)
        "finish2",  # R2: Working -> Done (final) -> triggers Outer completion -> AllDone
    ]

    for event in events_to_send:
        print(f"\n--- Event: '{event}' ---")
        executions = engine.process_event(event)

        for exec_record in executions:
            t = exec_record.transition
            event_str = t.event if t.event else "τ"
            print(f"  Fired: {t.source} --[{event_str}]--> {t.target} ({exec_record.triggered_by})")

        status = engine.get_state_status()
        active = [k for k, v in status.items() if v in ("ACTIVE", "FINAL")]
        finals = [k for k, v in status.items() if v == "FINAL"]
        print(f"  Active: {active}")
        print(f"  Finals: {finals}")

    print("\n" + "-" * 60)
    print("EXECUTION HISTORY")
    print("-" * 60)

    for exec_record in engine.history:
        t = exec_record.transition
        event_str = t.event if t.event else "τ"
        print(f"  Step {exec_record.step}: {t.source} --[{event_str}]--> {t.target} ({exec_record.triggered_by})")

    return engine


if __name__ == "__main__":
    demo()
