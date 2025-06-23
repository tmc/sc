"""
Core Pythonic API for statecharts.

This module provides a high-level, Pythonic interface for working with statecharts,
wrapping the lower-level Protocol Buffer types with more convenient Python classes.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .generated import (
    Statechart as PbStatechart,
    State as PbState,
    Transition as PbTransition,
    Event as PbEvent,
    Guard as PbGuard,
    Action as PbAction,
    StateRef as PbStateRef,
    Configuration as PbConfiguration,
    Machine as PbMachine,
    Step as PbStep,
    StateType as PbStateType,
    MachineState as PbMachineState,
)


class StateType(Enum):
    """State types for statechart states."""
    UNSPECIFIED = PbStateType.STATE_TYPE_UNSPECIFIED
    BASIC = PbStateType.STATE_TYPE_BASIC
    NORMAL = PbStateType.STATE_TYPE_NORMAL
    PARALLEL = PbStateType.STATE_TYPE_PARALLEL
    ORTHOGONAL = PbStateType.STATE_TYPE_ORTHOGONAL  # Alias for PARALLEL


class MachineState(Enum):
    """Machine states for statechart machines."""
    UNSPECIFIED = PbMachineState.MACHINE_STATE_UNSPECIFIED
    RUNNING = PbMachineState.MACHINE_STATE_RUNNING
    STOPPED = PbMachineState.MACHINE_STATE_STOPPED


@dataclass
class Event:
    """
    Represents an event in a statechart.
    
    Events trigger transitions between states.
    """
    label: str
    
    @classmethod
    def from_pb(cls, pb_event: PbEvent) -> 'Event':
        """Create an Event from a protobuf Event."""
        return cls(label=pb_event.label)
    
    def to_pb(self) -> PbEvent:
        """Convert to protobuf Event."""
        return PbEvent(label=self.label)


@dataclass
class Guard:
    """
    Represents a guard condition for a transition.
    
    Guards are conditions that must be satisfied for a transition to occur.
    """
    expression: str
    
    @classmethod
    def from_pb(cls, pb_guard: PbGuard) -> 'Guard':
        """Create a Guard from a protobuf Guard."""
        return cls(expression=pb_guard.expression)
    
    def to_pb(self) -> PbGuard:
        """Convert to protobuf Guard."""
        return PbGuard(expression=self.expression)


@dataclass
class Action:
    """
    Represents an action associated with a transition.
    
    Actions are executed when a transition occurs.
    """
    label: str
    
    @classmethod
    def from_pb(cls, pb_action: PbAction) -> 'Action':
        """Create an Action from a protobuf Action."""
        return cls(label=pb_action.label)
    
    def to_pb(self) -> PbAction:
        """Convert to protobuf Action."""
        return PbAction(label=self.label)


@dataclass
class State:
    """
    Represents a state in a statechart.
    
    States can be basic (no children), normal (OR semantics), or parallel (AND semantics).
    """
    label: str
    type: StateType = StateType.BASIC
    children: List['State'] = None
    is_initial: bool = False
    is_final: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @classmethod
    def from_pb(cls, pb_state: PbState) -> 'State':
        """Create a State from a protobuf State."""
        return cls(
            label=pb_state.label,
            type=StateType(pb_state.type),
            children=[cls.from_pb(child) for child in pb_state.children],
            is_initial=pb_state.is_initial,
            is_final=pb_state.is_final,
        )
    
    def to_pb(self) -> PbState:
        """Convert to protobuf State."""
        return PbState(
            label=self.label,
            type=self.type.value,
            children=[child.to_pb() for child in self.children],
            is_initial=self.is_initial,
            is_final=self.is_final,
        )
    
    def add_child(self, child: 'State') -> None:
        """Add a child state."""
        self.children.append(child)
        # Update type if needed
        if self.type == StateType.BASIC and self.children:
            self.type = StateType.NORMAL
    
    def find_state(self, label: str) -> Optional['State']:
        """Find a state by label in this state's hierarchy."""
        if self.label == label:
            return self
        for child in self.children:
            found = child.find_state(label)
            if found:
                return found
        return None
    
    def get_all_states(self) -> List['State']:
        """Get all states in this state's hierarchy."""
        states = [self]
        for child in self.children:
            states.extend(child.get_all_states())
        return states


@dataclass
class Transition:
    """
    Represents a transition between states in a statechart.
    
    Transitions connect source states to target states and are triggered by events.
    """
    label: str
    from_states: List[str]
    to_states: List[str]
    event: str
    guard: Optional[Guard] = None
    actions: List[Action] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
    
    @classmethod
    def from_pb(cls, pb_transition: PbTransition) -> 'Transition':
        """Create a Transition from a protobuf Transition."""
        guard = Guard.from_pb(pb_transition.guard) if pb_transition.guard else None
        actions = [Action.from_pb(action) for action in pb_transition.actions]
        
        return cls(
            label=pb_transition.label,
            from_states=list(pb_transition.from_),
            to_states=list(pb_transition.to),
            event=pb_transition.event,
            guard=guard,
            actions=actions,
        )
    
    def to_pb(self) -> PbTransition:
        """Convert to protobuf Transition."""
        pb_guard = self.guard.to_pb() if self.guard else None
        pb_actions = [action.to_pb() for action in self.actions]
        
        return PbTransition(
            label=self.label,
            from_=self.from_states,
            to=self.to_states,
            event=self.event,
            guard=pb_guard,
            actions=pb_actions,
        )


@dataclass
class StateRef:
    """Reference to a state."""
    label: str
    
    @classmethod
    def from_pb(cls, pb_state_ref: PbStateRef) -> 'StateRef':
        """Create a StateRef from a protobuf StateRef."""
        return cls(label=pb_state_ref.label)
    
    def to_pb(self) -> PbStateRef:
        """Convert to protobuf StateRef."""
        return PbStateRef(label=self.label)


@dataclass
class Configuration:
    """
    Represents the active states of a statechart at a given point in time.
    """
    states: List[StateRef]
    
    @classmethod
    def from_pb(cls, pb_config: PbConfiguration) -> 'Configuration':
        """Create a Configuration from a protobuf Configuration."""
        return cls(states=[StateRef.from_pb(state) for state in pb_config.states])
    
    def to_pb(self) -> PbConfiguration:
        """Convert to protobuf Configuration."""
        return PbConfiguration(states=[state.to_pb() for state in self.states])
    
    def has_state(self, label: str) -> bool:
        """Check if a state is active in this configuration."""
        return any(state.label == label for state in self.states)
    
    def get_state_labels(self) -> List[str]:
        """Get all active state labels."""
        return [state.label for state in self.states]


@dataclass
class Step:
    """Represents a step in the execution of a statechart."""
    events: List[Event]
    transitions: List[Transition]
    starting_configuration: Configuration
    resulting_configuration: Configuration
    context: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_pb(cls, pb_step: PbStep) -> 'Step':
        """Create a Step from a protobuf Step."""
        # Convert protobuf Struct to dict
        context = None
        if pb_step.context:
            context = dict(pb_step.context)
        
        return cls(
            events=[Event.from_pb(event) for event in pb_step.events],
            transitions=[Transition.from_pb(trans) for trans in pb_step.transitions],
            starting_configuration=Configuration.from_pb(pb_step.starting_configuration),
            resulting_configuration=Configuration.from_pb(pb_step.resulting_configuration),
            context=context,
        )
    
    def to_pb(self) -> PbStep:
        """Convert to protobuf Step."""
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = None
        if self.context:
            pb_context = Struct()
            pb_context.update(self.context)
        
        return PbStep(
            events=[event.to_pb() for event in self.events],
            transitions=[trans.to_pb() for trans in self.transitions],
            starting_configuration=self.starting_configuration.to_pb(),
            resulting_configuration=self.resulting_configuration.to_pb(),
            context=pb_context,
        )


class Statechart:
    """
    Complete, static description of a statechart.
    
    A statechart consists of a root state, transitions, and events.
    """
    
    def __init__(self, root_state: State, transitions: List[Transition] = None, events: List[Event] = None):
        self.root_state = root_state
        self.transitions = transitions or []
        self.events = events or []
    
    @classmethod
    def from_pb(cls, pb_statechart: PbStatechart) -> 'Statechart':
        """Create a Statechart from a protobuf Statechart."""
        root_state = State.from_pb(pb_statechart.root_state)
        transitions = [Transition.from_pb(trans) for trans in pb_statechart.transitions]
        events = [Event.from_pb(event) for event in pb_statechart.events]
        
        return cls(root_state=root_state, transitions=transitions, events=events)
    
    def to_pb(self) -> PbStatechart:
        """Convert to protobuf Statechart."""
        return PbStatechart(
            root_state=self.root_state.to_pb(),
            transitions=[trans.to_pb() for trans in self.transitions],
            events=[event.to_pb() for event in self.events],
        )
    
    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the statechart."""
        self.transitions.append(transition)
    
    def add_event(self, event: Event) -> None:
        """Add an event to the statechart."""
        self.events.append(event)
    
    def find_state(self, label: str) -> Optional[State]:
        """Find a state by label."""
        return self.root_state.find_state(label)
    
    def get_all_states(self) -> List[State]:
        """Get all states in the statechart."""
        return self.root_state.get_all_states()
    
    def get_events_for_state(self, state_label: str) -> List[str]:
        """Get all events that can trigger transitions from a given state."""
        events = []
        for transition in self.transitions:
            if state_label in transition.from_states:
                events.append(transition.event)
        return list(set(events))  # Remove duplicates
    
    def get_transitions_for_event(self, event_label: str) -> List[Transition]:
        """Get all transitions that are triggered by a given event."""
        return [trans for trans in self.transitions if trans.event == event_label]


class Machine:
    """
    An instance of a statechart with current configuration and execution history.
    """
    
    def __init__(self, id: str, statechart: Statechart, context: Optional[Dict[str, Any]] = None):
        self.id = id
        self.state = MachineState.STOPPED
        self.context = context or {}
        self.statechart = statechart
        self.configuration = Configuration(states=[])
        self.step_history: List[Step] = []
    
    @classmethod
    def from_pb(cls, pb_machine: PbMachine) -> 'Machine':
        """Create a Machine from a protobuf Machine."""
        # Convert protobuf Struct to dict
        context = dict(pb_machine.context) if pb_machine.context else {}
        
        machine = cls(
            id=pb_machine.id,
            statechart=Statechart.from_pb(pb_machine.statechart),
            context=context,
        )
        machine.state = MachineState(pb_machine.state)
        machine.configuration = Configuration.from_pb(pb_machine.configuration)
        machine.step_history = [Step.from_pb(step) for step in pb_machine.step_history]
        
        return machine
    
    def to_pb(self) -> PbMachine:
        """Convert to protobuf Machine."""
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = Struct()
        pb_context.update(self.context)
        
        return PbMachine(
            id=self.id,
            state=self.state.value,
            context=pb_context,
            statechart=self.statechart.to_pb(),
            configuration=self.configuration.to_pb(),
            step_history=[step.to_pb() for step in self.step_history],
        )
    
    def is_running(self) -> bool:
        """Check if the machine is running."""
        return self.state == MachineState.RUNNING
    
    def is_stopped(self) -> bool:
        """Check if the machine is stopped."""
        return self.state == MachineState.STOPPED
    
    def get_active_states(self) -> List[str]:
        """Get the labels of currently active states."""
        return self.configuration.get_state_labels()
    
    def is_state_active(self, label: str) -> bool:
        """Check if a specific state is currently active."""
        return self.configuration.has_state(label)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert machine to a dictionary representation."""
        return {
            'id': self.id,
            'state': self.state.name,
            'context': self.context,
            'active_states': self.get_active_states(),
            'step_count': len(self.step_history),
        }
    
    def to_json(self) -> str:
        """Convert machine to JSON representation."""
        return json.dumps(self.to_dict(), indent=2)