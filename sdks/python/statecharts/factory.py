"""
Factory functions for creating statechart components.

This module provides convenient factory functions for creating statechart
components with sensible defaults and Pythonic interfaces.
"""

from typing import List, Optional, Dict, Any, Union
from .core import State, Transition, Event, Guard, Action, Statechart, Machine, StateType, Configuration, StateRef


def basic_state(label: str, is_initial: bool = False, is_final: bool = False) -> State:
    """
    Create a basic state with no children.
    
    Args:
        label: The label/name of the state
        is_initial: Whether this is an initial state (default child of parent)
        is_final: Whether this is a final/terminal state
        
    Returns:
        A basic State instance
        
    Example:
        >>> off_state = basic_state("Off", is_initial=True)
        >>> on_state = basic_state("On")
    """
    return State(
        label=label,
        type=StateType.BASIC,
        children=[],
        is_initial=is_initial,
        is_final=is_final,
    )


def normal_state(
    label: str, 
    children: List[State], 
    is_initial: bool = False, 
    is_final: bool = False
) -> State:
    """
    Create a normal (OR/XOR) composite state with children.
    
    In normal states, exactly one child is active at any time (XOR semantics).
    
    Args:
        label: The label/name of the state
        children: List of child states  
        is_initial: Whether this is an initial state (default child of parent)
        is_final: Whether this is a final/terminal state
        
    Returns:
        A normal State instance
        
    Example:
        >>> child1 = basic_state("Off", is_initial=True)
        >>> child2 = basic_state("On")
        >>> power_state = normal_state("Power", [child1, child2])
    """
    return State(
        label=label,
        type=StateType.NORMAL,
        children=children,
        is_initial=is_initial,
        is_final=is_final,
    )


def parallel_state(
    label: str, 
    children: List[State], 
    is_initial: bool = False, 
    is_final: bool = False
) -> State:
    """
    Create a parallel (AND/orthogonal) composite state with children.
    
    In parallel states, all children are active simultaneously (AND semantics).
    
    Args:
        label: The label/name of the state
        children: List of child states (all will be active concurrently)
        is_initial: Whether this is an initial state (default child of parent)
        is_final: Whether this is a final/terminal state
        
    Returns:
        A parallel State instance
        
    Example:
        >>> display = basic_state("DisplayOn", is_initial=True)
        >>> audio = basic_state("AudioOn", is_initial=True)  
        >>> device = parallel_state("Device", [display, audio])
    """
    return State(
        label=label,
        type=StateType.PARALLEL,
        children=children,
        is_initial=is_initial,
        is_final=is_final,
    )


def orthogonal_state(
    label: str, 
    children: List[State], 
    is_initial: bool = False, 
    is_final: bool = False
) -> State:
    """
    Create an orthogonal state (alias for parallel_state).
    
    This is an alias for parallel_state() using the academic terminology.
    In orthogonal states, all children are active simultaneously (AND semantics).
    
    Args:
        label: The label/name of the state
        children: List of child states (all will be active concurrently)
        is_initial: Whether this is an initial state (default child of parent)
        is_final: Whether this is a final/terminal state
        
    Returns:
        A parallel State instance
    """
    return parallel_state(label, children, is_initial, is_final)


def transition(
    label: str,
    from_states: Union[str, List[str]],
    to_states: Union[str, List[str]],
    event: str,
    guard: Optional[Union[str, Guard]] = None,
    actions: Optional[Union[str, List[str], List[Action]]] = None,
) -> Transition:
    """
    Create a transition between states.
    
    Args:
        label: The label/name of the transition
        from_states: Source state(s) - can be a single state label or list of labels
        to_states: Target state(s) - can be a single state label or list of labels  
        event: The event that triggers this transition
        guard: Optional guard condition (string expression or Guard object)
        actions: Optional actions to execute (string, list of strings, or list of Action objects)
        
    Returns:
        A Transition instance
        
    Example:
        >>> trans = transition(
        ...     "PowerOn",
        ...     from_states="Off", 
        ...     to_states="On",
        ...     event="POWER_BUTTON",
        ...     guard="battery_level > 10",
        ...     actions=["log_power_on", "start_display"]
        ... )
    """
    # Normalize from_states to list
    if isinstance(from_states, str):
        from_list = [from_states]
    else:
        from_list = from_states
    
    # Normalize to_states to list
    if isinstance(to_states, str):
        to_list = [to_states]
    else:
        to_list = to_states
    
    # Handle guard
    guard_obj = None
    if guard:
        if isinstance(guard, str):
            guard_obj = Guard(expression=guard)
        else:
            guard_obj = guard
    
    # Handle actions
    actions_list = []
    if actions:
        if isinstance(actions, str):
            actions_list = [Action(label=actions)]
        elif isinstance(actions, list):
            actions_list = []
            for action in actions:
                if isinstance(action, str):
                    actions_list.append(Action(label=action))
                else:
                    actions_list.append(action)
    
    return Transition(
        label=label,
        from_states=from_list,
        to_states=to_list,
        event=event,
        guard=guard_obj,
        actions=actions_list,
    )


def event(label: str) -> Event:
    """
    Create an event.
    
    Args:
        label: The label/name of the event
        
    Returns:
        An Event instance
        
    Example:
        >>> power_event = event("POWER_BUTTON")
        >>> timer_event = event("TIMEOUT")
    """
    return Event(label=label)


def guard(expression: str) -> Guard:
    """
    Create a guard condition.
    
    Args:
        expression: The guard expression (e.g., "x > 10", "enabled == true")
        
    Returns:
        A Guard instance
        
    Example:
        >>> battery_guard = guard("battery_level > 10")
        >>> enabled_guard = guard("system_enabled == true")
    """
    return Guard(expression=expression)


def action(label: str) -> Action:
    """
    Create an action.
    
    Args:
        label: The label/name of the action
        
    Returns:
        An Action instance
        
    Example:
        >>> log_action = action("log_transition") 
        >>> notify_action = action("send_notification")
    """
    return Action(label=label)


def statechart(root_state: State, transitions: Optional[List[Transition]] = None, events: Optional[List[Event]] = None) -> Statechart:
    """
    Create a statechart with a root state, transitions, and events.
    
    Args:
        root_state: The root state of the statechart
        transitions: List of transitions (optional)
        events: List of events (optional)
        
    Returns:
        A Statechart instance
        
    Example:
        >>> root = normal_state("System", [basic_state("Off", is_initial=True), basic_state("On")])
        >>> trans = transition("PowerOn", "Off", "On", "POWER_BUTTON")
        >>> chart = statechart(root, transitions=[trans], events=[event("POWER_BUTTON")])
    """
    return Statechart(
        root_state=root_state,
        transitions=transitions or [],
        events=events or [],
    )


def machine(id: str, statechart: Statechart, context: Optional[Dict[str, Any]] = None) -> Machine:
    """
    Create a machine instance from a statechart.
    
    Args:
        id: Unique identifier for the machine
        statechart: The statechart definition
        context: Optional context/data for the machine
        
    Returns:
        A Machine instance
        
    Example:
        >>> chart = statechart(basic_state("Idle"))
        >>> my_machine = machine("machine-1", chart, context={"user_id": 123})
    """
    return Machine(id=id, statechart=statechart, context=context)


def configuration(state_labels: List[str]) -> Configuration:
    """
    Create a configuration from a list of state labels.
    
    Args:
        state_labels: List of active state labels
        
    Returns:
        A Configuration instance
        
    Example:
        >>> config = configuration(["System", "Power", "On"])
    """
    return Configuration(states=[StateRef(label=label) for label in state_labels])


# Builder pattern helper classes for more complex statechart construction

class StatechartBuilder:
    """
    Builder class for constructing complex statecharts with a fluent interface.
    
    Example:
        >>> chart = (StatechartBuilder()
        ...     .root_state("System")
        ...     .add_state("Off", parent="System", is_initial=True)
        ...     .add_state("On", parent="System")
        ...     .add_transition("PowerOn", from_state="Off", to_state="On", event="POWER")
        ...     .add_event("POWER")
        ...     .build())
    """
    
    def __init__(self):
        self._states: Dict[str, State] = {}
        self._root_label: Optional[str] = None
        self._transitions: List[Transition] = []
        self._events: List[Event] = []
    
    def root_state(self, label: str, state_type: StateType = StateType.NORMAL) -> 'StatechartBuilder':
        """Set the root state."""
        self._root_label = label
        self._states[label] = State(label=label, type=state_type, children=[])
        return self
    
    def add_state(
        self, 
        label: str, 
        parent: Optional[str] = None,
        state_type: StateType = StateType.BASIC,
        is_initial: bool = False,
        is_final: bool = False
    ) -> 'StatechartBuilder':
        """Add a state to the statechart."""
        state = State(
            label=label,
            type=state_type,
            children=[],
            is_initial=is_initial,
            is_final=is_final,
        )
        self._states[label] = state
        
        if parent and parent in self._states:
            self._states[parent].add_child(state)
        
        return self
    
    def add_transition(
        self,
        label: str,
        from_state: str,
        to_state: str,
        event: str,
        guard: Optional[str] = None,
        actions: Optional[List[str]] = None,
    ) -> 'StatechartBuilder':
        """Add a transition to the statechart."""
        trans = transition(
            label=label,
            from_states=from_state,
            to_states=to_state,
            event=event,
            guard=guard,
            actions=actions,
        )
        self._transitions.append(trans)
        return self
    
    def add_event(self, label: str) -> 'StatechartBuilder':
        """Add an event to the statechart."""
        self._events.append(event(label))
        return self
    
    def build(self) -> Statechart:
        """Build and return the statechart."""
        if not self._root_label or self._root_label not in self._states:
            raise ValueError("Root state must be set before building")
        
        root_state = self._states[self._root_label]
        return Statechart(
            root_state=root_state,
            transitions=self._transitions,
            events=self._events,
        )


# Convenience functions for common patterns

def simple_toggle_statechart(
    on_state_label: str = "On",
    off_state_label: str = "Off", 
    toggle_event: str = "TOGGLE"
) -> Statechart:
    """
    Create a simple two-state toggle statechart.
    
    Args:
        on_state_label: Label for the "on" state
        off_state_label: Label for the "off" state  
        toggle_event: Event that toggles between states
        
    Returns:
        A simple toggle Statechart
        
    Example:
        >>> toggle_chart = simple_toggle_statechart("Active", "Inactive", "TOGGLE")
    """
    off_state = basic_state(off_state_label, is_initial=True)
    on_state = basic_state(on_state_label)
    root = normal_state("Root", [off_state, on_state])
    
    on_transition = transition("TurnOn", off_state_label, on_state_label, toggle_event)
    off_transition = transition("TurnOff", on_state_label, off_state_label, toggle_event)
    
    toggle_event_obj = event(toggle_event)
    
    return statechart(root, [on_transition, off_transition], [toggle_event_obj])


def hierarchical_statechart_example() -> Statechart:
    """
    Create an example hierarchical statechart for demonstration.
    
    Returns:
        A sample hierarchical Statechart
    """
    # Create leaf states
    off = basic_state("Off", is_initial=True)
    idle = basic_state("Idle", is_initial=True)
    active = basic_state("Active")
    
    # Create composite states
    on_state = normal_state("On", [idle, active])
    root = normal_state("System", [off, on_state])
    
    # Create transitions
    transitions = [
        transition("PowerOn", "Off", "On", "POWER"),
        transition("PowerOff", "On", "Off", "POWER"),
        transition("Activate", "Idle", "Active", "START"),
        transition("Deactivate", "Active", "Idle", "STOP"),
    ]
    
    # Create events
    events = [
        event("POWER"),
        event("START"), 
        event("STOP"),
    ]
    
    return statechart(root, transitions, events)