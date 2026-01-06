"""
Unified Statechart Executor

Executes UnifiedGenome with all components:
- State machine transitions
- History save/restore
- Guard evaluation
- Priority-based conflict resolution
- Action execution

Implements full Harel semantics as in semantics/v1/machine.go.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from .unified_genome import (
    UnifiedGenome, UnifiedTransition, UnifiedGuard, UnifiedAction,
    StateType, HistoryType, GuardOp, ActionType
)


# =============================================================================
# Context
# =============================================================================

@dataclass
class UnifiedContext:
    """
    Execution context with variables for guards and actions.
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    history: Dict[int, List[int]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any):
        self.variables[key] = value

    def copy(self) -> 'UnifiedContext':
        return UnifiedContext(
            variables=copy.deepcopy(self.variables),
            history=copy.deepcopy(self.history)
        )


# =============================================================================
# Unified Machine
# =============================================================================

class UnifiedMachine:
    """
    Executes a UnifiedGenome with full statechart semantics.

    Implements:
    - Hierarchical state machine
    - History (shallow and deep)
    - Guard evaluation
    - Priority-based conflict resolution
    - Action execution
    """

    def __init__(self, genome: UnifiedGenome):
        self.genome = genome
        self.current_state: int = genome.initial_state
        self.context = UnifiedContext(
            variables=copy.deepcopy(genome.initial_context)
        )

        # Build transition lookup
        self._build_transition_map()

    def _build_transition_map(self):
        """Build lookup structure for transitions."""
        # (src, event) -> [transitions]
        self.transition_map: Dict[Tuple[int, int], List[UnifiedTransition]] = {}

        for t in self.genome.transitions:
            key = (t.src, t.event)
            if key not in self.transition_map:
                self.transition_map[key] = []
            self.transition_map[key].append(t)

    def get_configuration(self) -> List[int]:
        """Get current active states (leaf + ancestors)."""
        config = [self.current_state]
        curr = self.genome.parent[self.current_state]
        while curr != -1:
            config.append(curr)
            curr = self.genome.parent[curr]
        return config

    def get_enabled_transitions(self, event: int) -> List[UnifiedTransition]:
        """Get all transitions enabled by event in current state."""
        key = (self.current_state, event)
        candidates = self.transition_map.get(key, [])

        enabled = []
        for t in candidates:
            if t.guard.evaluate(self.context.variables):
                enabled.append(t)

        return enabled

    def resolve_conflicts(self, transitions: List[UnifiedTransition]) -> Optional[UnifiedTransition]:
        """
        Resolve conflicts when multiple transitions are enabled.

        Uses priority + guard specificity.
        """
        if not transitions:
            return None
        if len(transitions) == 1:
            return transitions[0]

        # Sort by effective priority (descending)
        sorted_trans = sorted(
            transitions,
            key=lambda t: t.effective_priority(use_specificity=True),
            reverse=True
        )

        return sorted_trans[0]

    def save_history(self, state_idx: int):
        """Save history when exiting composite state."""
        if not self.genome.is_composite(state_idx):
            return

        # Find active descendants
        config = self.get_configuration()
        descendants = []
        for s in config:
            # Check if s is descendant of state_idx
            curr = s
            while curr != -1:
                if self.genome.parent[curr] == state_idx:
                    descendants.append(s)
                    break
                curr = self.genome.parent[curr]

        if descendants:
            self.context.history[state_idx] = descendants

    def resolve_history(self, state_idx: int) -> int:
        """
        Resolve which state to enter based on history type.

        DEEP: Return deepest stored state
        SHALLOW: Return direct child from storage
        NONE: Return initial child
        """
        ht = self.genome.history_type[state_idx]

        if ht == HistoryType.NONE:
            return self._get_initial_child(state_idx)

        if state_idx not in self.context.history:
            return self._get_initial_child(state_idx)

        stored = self.context.history[state_idx]

        if ht == HistoryType.DEEP:
            # Return deepest stored state
            if stored:
                return max(stored, key=lambda s: self.genome.get_depth(s))
            return self._get_initial_child(state_idx)

        elif ht == HistoryType.SHALLOW:
            # Return direct child from storage
            children = self.genome.get_children(state_idx)
            for s in stored:
                if s in children:
                    return s
                # Find ancestor that is direct child
                curr = s
                while curr != -1:
                    if self.genome.parent[curr] == state_idx:
                        return curr
                    curr = self.genome.parent[curr]
            return self._get_initial_child(state_idx)

        return self._get_initial_child(state_idx)

    def _get_initial_child(self, state_idx: int) -> int:
        """Get initial child of composite state (recursively to leaf)."""
        children = self.genome.get_children(state_idx)
        if not children:
            return state_idx

        child = children[0]
        while True:
            next_children = self.genome.get_children(child)
            if not next_children:
                return child
            child = next_children[0]

    def execute_actions(self, actions: List[UnifiedAction]):
        """Execute transition actions."""
        for action in actions:
            self.context.variables = action.execute(self.context.variables)

    def step(self, event: int) -> Tuple[bool, int, int]:
        """
        Process event and execute transition.

        Returns: (transition_fired, old_state, new_state)
        """
        old_state = self.current_state

        # Get enabled transitions
        enabled = self.get_enabled_transitions(event)
        if not enabled:
            return False, old_state, old_state

        # Resolve conflicts
        transition = self.resolve_conflicts(enabled)
        if not transition:
            return False, old_state, old_state

        # Save history for exited composite states
        config = self.get_configuration()
        target = transition.tgt

        # Find new configuration for target
        new_config = []
        curr = target
        while curr != -1:
            new_config.append(curr)
            curr = self.genome.parent[curr]

        # States being exited
        for s in config:
            if s not in new_config:
                self.save_history(s)

        # Execute actions
        self.execute_actions(transition.actions)

        # Enter target (resolve history if composite)
        while True:
            children = self.genome.get_children(target)
            if not children:
                break
            target = self.resolve_history(target)

        self.current_state = target
        return True, old_state, self.current_state

    def reset(self):
        """Reset machine to initial state."""
        self.current_state = self.genome.initial_state
        self.context = UnifiedContext(
            variables=copy.deepcopy(self.genome.initial_context)
        )

    def run_trace(self, events: List[int]) -> List[int]:
        """
        Run a sequence of events and return state trace.
        """
        trace = [self.current_state]
        for event in events:
            fired, old, new = self.step(event)
            trace.append(new)
        return trace


# =============================================================================
# Scenario Execution
# =============================================================================

@dataclass
class ScenarioResult:
    """Result of executing a scenario."""
    events: List[int]
    expected_state: int
    actual_state: int
    trace: List[int]
    correct: bool


def execute_scenario(
    genome: UnifiedGenome,
    events: List[int],
    expected_state: int,
    initial_context: Dict[str, Any] = None
) -> ScenarioResult:
    """
    Execute a single scenario and check result.
    """
    machine = UnifiedMachine(genome)
    if initial_context:
        machine.context.variables = copy.deepcopy(initial_context)

    trace = machine.run_trace(events)
    actual = machine.current_state

    return ScenarioResult(
        events=events,
        expected_state=expected_state,
        actual_state=actual,
        trace=trace,
        correct=(actual == expected_state)
    )


def execute_scenarios(
    genome: UnifiedGenome,
    scenarios: List[Tuple[List[int], int]]
) -> Tuple[float, List[ScenarioResult]]:
    """
    Execute multiple scenarios and compute accuracy.

    Args:
        genome: The statechart genome
        scenarios: List of (events, expected_state) pairs

    Returns:
        (accuracy, results)
    """
    results = []
    correct = 0

    for events, expected in scenarios:
        result = execute_scenario(genome, events, expected)
        results.append(result)
        if result.correct:
            correct += 1

    accuracy = correct / len(scenarios) if scenarios else 0.0
    return accuracy, results


# =============================================================================
# Testing
# =============================================================================

def test_unified_executor():
    """Test unified executor."""
    print("=" * 60)
    print("UNIFIED EXECUTOR TEST")
    print("=" * 60)

    from .unified_genome import create_random_unified_genome

    genome = create_random_unified_genome(
        max_states=6,
        n_events=3,
        include_history=True,
        include_actions=True
    )

    print(f"Genome: {genome.n_states} states, {len(genome.transitions)} transitions")

    machine = UnifiedMachine(genome)
    print(f"Initial state: {machine.current_state}")

    # Run some events
    for event in range(3):
        fired, old, new = machine.step(event)
        print(f"Event {event}: {old} -> {new} (fired={fired})")

    # Test trace
    machine.reset()
    trace = machine.run_trace([0, 1, 2, 0, 1])
    print(f"Trace: {trace}")

    print("\nUnified executor test complete!")
    return machine


if __name__ == "__main__":
    test_unified_executor()
