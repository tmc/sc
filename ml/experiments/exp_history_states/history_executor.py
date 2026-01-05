"""
History State Executor for Statecharts.

Implements shallow (H) and deep (H*) history state semantics:
- Shallow history: Remembers only the immediate child of a composite state
- Deep history: Remembers the full nested configuration within a composite state

When exiting a composite state, the active configuration is saved to history.
When entering via a history pseudostate, the saved configuration is restored.
If no history exists, falls back to the default initial state.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Any, Tuple
from enum import Enum


class HistoryType(Enum):
    """History pseudostate types."""
    SHALLOW = "shallow"  # H - remembers immediate child only
    DEEP = "deep"        # H* - remembers full nested configuration


@dataclass
class HistoryState:
    """Represents a history pseudostate."""
    label: str
    parent: str  # The composite state this history belongs to
    history_type: HistoryType
    default_state: Optional[str] = None  # Fallback if no history


@dataclass
class StateInfo:
    """Information about a state."""
    label: str
    type: int  # 1=BASIC, 2=NORMAL (OR), 3=PARALLEL (AND)
    parent: Optional[str]
    children: List[str]
    is_initial: bool
    is_history: bool = False
    history_type: Optional[HistoryType] = None


class HistoryExecutor:
    """
    Statechart executor with history state support.

    Key semantics:
    1. When exiting a composite state, save current substates to history
    2. When entering via history pseudostate, restore saved configuration
    3. If no history, use default initial state
    """

    def __init__(self, sc_json: dict):
        self.sc_json = sc_json
        self.states: Dict[str, StateInfo] = {}
        self.transitions = sc_json.get("transitions", [])
        self.history_states: Dict[str, HistoryState] = {}

        # History storage: parent_label -> saved configuration
        self.shallow_history: Dict[str, str] = {}
        self.deep_history: Dict[str, Set[str]] = {}

        self._parse_states(sc_json.get("root_state", {}), None)
        self._identify_history_states()

    def _parse_states(self, state: dict, parent: Optional[str]):
        """Parse state hierarchy."""
        label = state.get("label", "")
        state_type = state.get("type", 1)
        children = state.get("children", [])

        if label and not label.startswith("__"):
            child_labels = [c.get("label") for c in children if c.get("label")]

            self.states[label] = StateInfo(
                label=label,
                type=state_type,
                parent=parent,
                children=child_labels,
                is_initial=state.get("is_initial", False),
                is_history=state.get("is_history", False),
                history_type=self._parse_history_type(state),
            )

        for child in children:
            child_label = child.get("label", "")
            self._parse_states(
                child,
                label if label and not label.startswith("__") else parent
            )

    def _parse_history_type(self, state: dict) -> Optional[HistoryType]:
        """Parse history type from state definition."""
        if not state.get("is_history"):
            return None

        h_type = state.get("history_type", "shallow")
        if h_type in ("deep", "HISTORY_TYPE_DEEP", 1):
            return HistoryType.DEEP
        return HistoryType.SHALLOW

    def _identify_history_states(self):
        """Identify and register history pseudostates."""
        for label, info in self.states.items():
            if info.is_history:
                # Find the parent composite state
                parent = info.parent
                if parent:
                    # Find default state (initial sibling)
                    parent_info = self.states.get(parent)
                    default = None
                    if parent_info:
                        for sibling in parent_info.children:
                            sibling_info = self.states.get(sibling)
                            if sibling_info and sibling_info.is_initial:
                                default = sibling
                                break

                    self.history_states[label] = HistoryState(
                        label=label,
                        parent=parent,
                        history_type=info.history_type or HistoryType.SHALLOW,
                        default_state=default,
                    )

    def initial_config(self) -> Set[str]:
        """Compute initial configuration."""
        active = set()

        # Find root-level initial states
        for label, info in self.states.items():
            if info.parent is None and info.is_initial:
                active.add(label)
                active = active | self._enter_substates(label)

        return active

    def _enter_substates(self, state_label: str) -> Set[str]:
        """Enter initial substates of a composite state."""
        result = set()
        info = self.states.get(state_label)

        if not info or info.type == 1:  # BASIC
            return result

        if info.type == 3:  # PARALLEL - enter all children
            for child in info.children:
                child_info = self.states.get(child)
                if child_info and not child_info.is_history:
                    result.add(child)
                    result = result | self._enter_substates(child)
        else:  # NORMAL (OR) - enter initial child
            for child in info.children:
                child_info = self.states.get(child)
                if child_info and child_info.is_initial and not child_info.is_history:
                    result.add(child)
                    result = result | self._enter_substates(child)
                    break

        return result

    def _save_history(self, composite_label: str, active: Set[str]):
        """Save history when exiting a composite state."""
        info = self.states.get(composite_label)
        if not info:
            return

        # Find substates of this composite that are active
        substates = self._get_descendant_states(composite_label)
        active_substates = active & substates

        if not active_substates:
            return

        # Save shallow history: the immediate child that's active
        for child in info.children:
            if child in active or any(
                self._is_ancestor(child, s) for s in active_substates
            ):
                child_info = self.states.get(child)
                if child_info and not child_info.is_history:
                    self.shallow_history[composite_label] = child
                    break

        # Save deep history: full nested configuration
        self.deep_history[composite_label] = active_substates.copy()

    def _get_descendant_states(self, state_label: str) -> Set[str]:
        """Get all descendant states of a composite state."""
        result = set()
        info = self.states.get(state_label)

        if not info:
            return result

        for child in info.children:
            result.add(child)
            result = result | self._get_descendant_states(child)

        return result

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        current = descendant
        while current:
            info = self.states.get(current)
            if not info:
                return False
            if info.parent == ancestor:
                return True
            current = info.parent
        return False

    def _restore_history(
        self,
        history_label: str,
        active: Set[str]
    ) -> Set[str]:
        """Restore configuration from history pseudostate."""
        h_state = self.history_states.get(history_label)
        if not h_state:
            return active

        parent = h_state.parent

        if h_state.history_type == HistoryType.DEEP:
            # Deep history: restore full nested configuration
            if parent in self.deep_history:
                saved = self.deep_history[parent]
                return active | saved
            # No history - use default
            if h_state.default_state:
                active.add(h_state.default_state)
                active = active | self._enter_substates(h_state.default_state)
        else:
            # Shallow history: restore immediate child only
            if parent in self.shallow_history:
                child = self.shallow_history[parent]
                active.add(child)
                active = active | self._enter_substates(child)
            # No history - use default
            elif h_state.default_state:
                active.add(h_state.default_state)
                active = active | self._enter_substates(h_state.default_state)

        return active

    def _transition_enabled(self, from_states: Set[str], active: Set[str]) -> bool:
        """
        Check if a transition is enabled from current configuration.

        A transition is enabled if:
        1. Any from_state is directly in active, OR
        2. Any from_state is a composite containing an active substate
        """
        for from_state in from_states:
            if from_state in active:
                return True
            # Check if from_state is a composite containing active substates
            descendants = self._get_descendant_states(from_state)
            if descendants & active:
                return True
        return False

    def _get_active_from_state(
        self, from_states: Set[str], active: Set[str]
    ) -> Optional[str]:
        """Get the actual active state that matches a transition source."""
        for from_state in from_states:
            if from_state in active:
                return from_state
            # Check if from_state is a composite containing active substates
            descendants = self._get_descendant_states(from_state)
            active_descendants = descendants & active
            if active_descendants:
                return from_state  # Return the composite state
        return None

    def enabled_events(self, active: Set[str]) -> Set[str]:
        """Get events enabled from current configuration."""
        enabled = set()
        for t in self.transitions:
            from_states = set(t.get("from", []))
            if self._transition_enabled(from_states, active):
                event = t.get("event", "")
                if event:
                    enabled.add(event)
        return enabled

    def step(self, active: Set[str], event: str) -> Optional[Set[str]]:
        """Execute event, returning new configuration with history handling."""
        for t in self.transitions:
            from_states = set(t.get("from", []))
            to_states = set(t.get("to", []))
            t_event = t.get("event", "")

            if t_event == event and self._transition_enabled(from_states, active):
                # Find actual states being exited
                states_to_exit = self._get_states_to_exit(from_states, active)

                # Save history for composite states we're exiting
                exited_composites = self._find_exited_composites(
                    active, from_states, to_states, states_to_exit
                )
                for composite in exited_composites:
                    self._save_history(composite, active)

                # Execute transition - remove exited states and their descendants
                new_active = active.copy()
                for state in states_to_exit:
                    new_active.discard(state)
                    descendants = self._get_descendant_states(state)
                    new_active = new_active - descendants

                # Handle each target state
                for target in to_states:
                    if target in self.history_states:
                        # Target is a history pseudostate
                        new_active = self._restore_history(target, new_active)
                    else:
                        # Regular state - enter it
                        new_active.add(target)
                        new_active = new_active | self._enter_substates(target)

                return new_active

        return None

    def _get_states_to_exit(
        self, from_states: Set[str], active: Set[str]
    ) -> Set[str]:
        """Get actual states to exit based on transition source."""
        to_exit = set()
        for from_state in from_states:
            if from_state in active:
                to_exit.add(from_state)
            else:
                # from_state is a composite - find active substates
                descendants = self._get_descendant_states(from_state)
                to_exit.update(descendants & active)
                # Also mark the composite as being exited for history purposes
                if descendants & active:
                    to_exit.add(from_state)
        return to_exit

    def _find_exited_composites(
        self,
        active: Set[str],
        from_states: Set[str],
        to_states: Set[str],
        states_to_exit: Set[str]
    ) -> List[str]:
        """Find composite states being exited (need history save)."""
        exited = []

        # Check if any from_state itself is a composite being exited
        for from_state in from_states:
            info = self.states.get(from_state)
            if info and info.type in (2, 3):  # Composite state
                descendants = self._get_descendant_states(from_state)
                if descendants & active:  # Has active substates
                    # Check if target is outside this composite
                    all_in_composite = descendants | {from_state}
                    targets_outside = any(
                        t not in all_in_composite
                        for t in to_states
                        if t not in self.history_states
                    )
                    if targets_outside and from_state not in exited:
                        exited.append(from_state)

        # Also check ancestors of exited states
        for state in states_to_exit:
            current = state
            while current:
                info = self.states.get(current)
                if not info:
                    break

                parent = info.parent
                if parent:
                    parent_info = self.states.get(parent)
                    if parent_info and parent_info.type in (2, 3):
                        # Check if we're exiting this parent
                        descendants = self._get_descendant_states(parent)
                        all_in_parent = descendants | {parent}
                        targets_outside = any(
                            t not in all_in_parent
                            for t in to_states
                            if t not in self.history_states
                        )
                        if targets_outside and parent not in exited:
                            exited.append(parent)

                current = parent

        return exited

    def reset_history(self):
        """Clear all saved history."""
        self.shallow_history.clear()
        self.deep_history.clear()

    def get_shallow_history(self, composite: str) -> Optional[str]:
        """Get saved shallow history for a composite state."""
        return self.shallow_history.get(composite)

    def get_deep_history(self, composite: str) -> Optional[Set[str]]:
        """Get saved deep history for a composite state."""
        return self.deep_history.get(composite)
