"""
Completion Detector - Detect when all parallel regions reach final.

FORMAL DEFINITION:
A PARALLEL state p is complete iff:
  complete(p, σ) ⟺ ∀r ∈ children(p): complete(r, σ)

where for each region r:
  complete(r, σ) ⟺ ∃f ∈ descendants(r): is_final(f) ∧ f ∈ σ

This detector:
1. Monitors configuration changes
2. Detects when final states become active
3. Checks completion conditions bottom-up
4. Generates completion events for parent transitions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable, Tuple
from enum import Enum, auto

from .terminal_state import State, StateType, Configuration, Transition


class CompletionStatus(Enum):
    """Status of region/state completion."""
    NOT_STARTED = auto()    # Not yet entered
    IN_PROGRESS = auto()    # Active but not final
    COMPLETED = auto()      # Reached final state
    TERMINATED = auto()     # Exited without reaching final


@dataclass
class RegionStatus:
    """Track completion status of a region."""
    region_label: str
    status: CompletionStatus = CompletionStatus.NOT_STARTED
    final_state_reached: Optional[str] = None
    completion_time: Optional[int] = None  # Step number when completed

    def is_complete(self) -> bool:
        return self.status == CompletionStatus.COMPLETED


@dataclass
class CompletionEvent:
    """Event generated when completion condition met."""
    source_state: str
    completion_type: str  # "region" or "parallel"
    details: Dict[str, any] = field(default_factory=dict)


class CompletionDetector:
    """Detect completion of parallel regions."""

    def __init__(self, root: State):
        self.root = root
        self.state_map = self._build_state_map(root)
        self.final_states = self._collect_final_states(root)
        self.parallel_states = self._collect_parallel_states(root)
        self.region_status: Dict[str, RegionStatus] = {}
        self.step_counter = 0
        self.completion_listeners: List[Callable[[CompletionEvent], None]] = []

        # Initialize region status for each parallel state's regions
        for p_label in self.parallel_states:
            p_state = self.state_map[p_label]
            for region in p_state.children:
                self.region_status[region.label] = RegionStatus(
                    region_label=region.label
                )

    def _build_state_map(self, state: State) -> Dict[str, State]:
        result = {state.label: state}
        for child in state.children:
            result.update(self._build_state_map(child))
        return result

    def _collect_final_states(self, state: State) -> Set[str]:
        finals = set()
        if state.is_final:
            finals.add(state.label)
        for child in state.children:
            finals.update(self._collect_final_states(child))
        return finals

    def _collect_parallel_states(self, state: State) -> Set[str]:
        parallels = set()
        if state.state_type == StateType.PARALLEL:
            parallels.add(state.label)
        for child in state.children:
            parallels.update(self._collect_parallel_states(child))
        return parallels

    def add_listener(self, listener: Callable[[CompletionEvent], None]):
        """Add completion event listener."""
        self.completion_listeners.append(listener)

    def _notify_listeners(self, event: CompletionEvent):
        for listener in self.completion_listeners:
            listener(event)

    def get_region_for_state(self, state_label: str) -> Optional[str]:
        """Get the parent region for a state (under a PARALLEL parent)."""
        state = self.state_map.get(state_label)
        if not state or not state.parent:
            return None

        parent = state.parent
        if parent.parent and parent.parent.state_type == StateType.PARALLEL:
            return parent.label

        # Recurse up
        return self.get_region_for_state(parent.label)

    def update(self, config: Configuration) -> List[CompletionEvent]:
        """
        Update completion status based on current configuration.
        Returns list of newly generated completion events.
        """
        self.step_counter += 1
        events = []

        # Check each final state
        for final_label in self.final_states:
            if config.is_active(final_label):
                region_label = self.get_region_for_state(final_label)
                if region_label and region_label in self.region_status:
                    status = self.region_status[region_label]
                    if status.status != CompletionStatus.COMPLETED:
                        # Mark region as complete
                        status.status = CompletionStatus.COMPLETED
                        status.final_state_reached = final_label
                        status.completion_time = self.step_counter

                        event = CompletionEvent(
                            source_state=region_label,
                            completion_type="region",
                            details={
                                "final_state": final_label,
                                "step": self.step_counter,
                            }
                        )
                        events.append(event)
                        self._notify_listeners(event)

        # Check parallel state completion
        for p_label in self.parallel_states:
            if config.is_active(p_label):
                if self.is_parallel_complete(p_label):
                    event = CompletionEvent(
                        source_state=p_label,
                        completion_type="parallel",
                        details={
                            "regions": self._get_region_details(p_label),
                            "step": self.step_counter,
                        }
                    )
                    events.append(event)
                    self._notify_listeners(event)

        return events

    def is_parallel_complete(self, parallel_label: str) -> bool:
        """
        Check if PARALLEL state is complete.
        complete(p) ⟺ ∀r ∈ children(p): complete(r)
        """
        p_state = self.state_map.get(parallel_label)
        if not p_state or p_state.state_type != StateType.PARALLEL:
            return False

        for region in p_state.children:
            status = self.region_status.get(region.label)
            if not status or not status.is_complete():
                return False

        return True

    def _get_region_details(self, parallel_label: str) -> Dict[str, any]:
        """Get completion details for all regions."""
        p_state = self.state_map[parallel_label]
        details = {}
        for region in p_state.children:
            status = self.region_status.get(region.label)
            if status:
                details[region.label] = {
                    "status": status.status.name,
                    "final": status.final_state_reached,
                    "step": status.completion_time,
                }
        return details

    def get_incomplete_regions(self, parallel_label: str) -> List[str]:
        """Get list of regions that haven't completed."""
        p_state = self.state_map.get(parallel_label)
        if not p_state:
            return []

        incomplete = []
        for region in p_state.children:
            status = self.region_status.get(region.label)
            if not status or not status.is_complete():
                incomplete.append(region.label)
        return incomplete

    def get_completion_progress(self, parallel_label: str) -> Tuple[int, int]:
        """Get (completed, total) regions for parallel state."""
        p_state = self.state_map.get(parallel_label)
        if not p_state:
            return (0, 0)

        completed = 0
        total = len(p_state.children)

        for region in p_state.children:
            status = self.region_status.get(region.label)
            if status and status.is_complete():
                completed += 1

        return (completed, total)

    def reset(self):
        """Reset all completion status."""
        self.step_counter = 0
        for status in self.region_status.values():
            status.status = CompletionStatus.NOT_STARTED
            status.final_state_reached = None
            status.completion_time = None


class MultiLevelCompletionDetector:
    """Detect completion across nested parallel states."""

    def __init__(self, root: State):
        self.root = root
        self.detector = CompletionDetector(root)
        self.pending_completions: List[str] = []

    def update(self, config: Configuration) -> List[CompletionEvent]:
        """Update and propagate completions up the hierarchy."""
        events = self.detector.update(config)

        # Track which parallel states completed
        for event in events:
            if event.completion_type == "parallel":
                self.pending_completions.append(event.source_state)

        # Check if parent parallel states now complete
        propagated = self._propagate_completions(config)
        events.extend(propagated)

        return events

    def _propagate_completions(self, config: Configuration) -> List[CompletionEvent]:
        """Propagate completions to parent parallel states."""
        events = []

        for p_label in list(self.pending_completions):
            state = self.detector.state_map.get(p_label)
            if not state or not state.parent:
                continue

            # Check if parent is a region of a parallel
            parent = state.parent
            if parent.parent and parent.parent.state_type == StateType.PARALLEL:
                grandparent_label = parent.parent.label
                if self.detector.is_parallel_complete(grandparent_label):
                    event = CompletionEvent(
                        source_state=grandparent_label,
                        completion_type="parallel_propagated",
                        details={
                            "triggered_by": p_label,
                        }
                    )
                    events.append(event)
                    self.pending_completions.append(grandparent_label)

            self.pending_completions.remove(p_label)

        return events


def build_nested_parallel() -> State:
    """
    Build nested parallel statechart.

    Root (PARALLEL)
    ├── RegionA (NORMAL)
    │   ├── A_Active
    │   └── A_Final*
    └── RegionB (NORMAL)
        ├── B_Parallel (PARALLEL)
        │   ├── B1 (NORMAL)
        │   │   ├── B1_Active
        │   │   └── B1_Final*
        │   └── B2 (NORMAL)
        │       ├── B2_Active
        │       └── B2_Final*
        └── B_Done*
    """
    # Region A
    a_active = State(label="A_Active", is_initial=True)
    a_final = State(label="A_Final", is_final=True)
    region_a = State(
        label="RegionA",
        state_type=StateType.NORMAL,
        children=[a_active, a_final],
    )

    # Nested parallel in Region B
    b1_active = State(label="B1_Active", is_initial=True)
    b1_final = State(label="B1_Final", is_final=True)
    b1 = State(
        label="B1",
        state_type=StateType.NORMAL,
        children=[b1_active, b1_final],
    )

    b2_active = State(label="B2_Active", is_initial=True)
    b2_final = State(label="B2_Final", is_final=True)
    b2 = State(
        label="B2",
        state_type=StateType.NORMAL,
        children=[b2_active, b2_final],
    )

    b_parallel = State(
        label="B_Parallel",
        state_type=StateType.PARALLEL,
        children=[b1, b2],
        is_initial=True,
    )

    b_done = State(label="B_Done", is_final=True)

    region_b = State(
        label="RegionB",
        state_type=StateType.NORMAL,
        children=[b_parallel, b_done],
    )

    root = State(
        label="Root",
        state_type=StateType.PARALLEL,
        children=[region_a, region_b],
    )

    return root


def demo():
    """Demonstrate completion detection."""
    print("=" * 60)
    print("COMPLETION DETECTOR")
    print("=" * 60)

    root = build_nested_parallel()

    print("\nStatechart structure:")
    def print_tree(state, indent=0):
        mark = "*" if state.is_final else ""
        type_str = f" ({state.state_type.name})" if state.state_type != StateType.BASIC else ""
        print("  " * indent + f"{state.label}{mark}{type_str}")
        for child in state.children:
            print_tree(child, indent + 1)
    print_tree(root)

    detector = CompletionDetector(root)

    # Track events
    received_events = []
    detector.add_listener(lambda e: received_events.append(e))

    print("\n" + "-" * 60)
    print("COMPLETION DETECTION SIMULATION")
    print("-" * 60)

    # Step 1: Initial config
    config1 = Configuration(active_states={
        "Root", "RegionA", "A_Active", "RegionB", "B_Parallel",
        "B1", "B1_Active", "B2", "B2_Active"
    })
    print("\nStep 1: Initial state")
    events = detector.update(config1)
    prog = detector.get_completion_progress("Root")
    print(f"  Root progress: {prog[0]}/{prog[1]} regions complete")
    print(f"  Events: {len(events)}")

    # Step 2: RegionA completes
    config2 = Configuration(active_states={
        "Root", "RegionA", "A_Final", "RegionB", "B_Parallel",
        "B1", "B1_Active", "B2", "B2_Active"
    })
    print("\nStep 2: RegionA reaches final")
    events = detector.update(config2)
    prog = detector.get_completion_progress("Root")
    print(f"  Root progress: {prog[0]}/{prog[1]} regions complete")
    print(f"  Events: {[e.source_state for e in events]}")

    # Step 3: B1 completes
    config3 = Configuration(active_states={
        "Root", "RegionA", "A_Final", "RegionB", "B_Parallel",
        "B1", "B1_Final", "B2", "B2_Active"
    })
    print("\nStep 3: B1 reaches final")
    events = detector.update(config3)
    prog_b = detector.get_completion_progress("B_Parallel")
    print(f"  B_Parallel progress: {prog_b[0]}/{prog_b[1]} regions complete")
    print(f"  Events: {[e.source_state for e in events]}")

    # Step 4: B2 completes -> B_Parallel completes
    config4 = Configuration(active_states={
        "Root", "RegionA", "A_Final", "RegionB", "B_Parallel",
        "B1", "B1_Final", "B2", "B2_Final"
    })
    print("\nStep 4: B2 reaches final -> B_Parallel completes")
    events = detector.update(config4)
    print(f"  B_Parallel complete: {detector.is_parallel_complete('B_Parallel')}")
    print(f"  Events: {[(e.source_state, e.completion_type) for e in events]}")

    # Summary
    print("\n" + "-" * 60)
    print("EVENT SUMMARY")
    print("-" * 60)
    print(f"Total events received: {len(received_events)}")
    for e in received_events:
        print(f"  {e.completion_type}: {e.source_state}")

    return detector


if __name__ == "__main__":
    demo()
