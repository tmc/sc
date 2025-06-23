"""
Unit tests for core statechart functionality.

This module tests the core classes and functions of the statecharts library,
including State, Transition, Event, Statechart, and Machine classes.
"""

import pytest
from typing import Dict, Any

from statecharts.core import (
    State,
    Transition,
    Event,
    Guard,
    Action,
    StateRef,
    Configuration,
    Statechart,
    Machine,
    StateType,
    MachineState,
)
from statecharts.factory import (
    basic_state,
    normal_state,
    parallel_state,
    transition,
    event,
    guard,
    action,
    statechart,
    machine,
)


class TestStateType:
    """Test StateType enum."""
    
    def test_state_type_values(self):
        """Test that StateType enum has correct values."""
        assert StateType.BASIC.name == "BASIC"
        assert StateType.NORMAL.name == "NORMAL"
        assert StateType.PARALLEL.name == "PARALLEL"
        assert StateType.ORTHOGONAL.name == "ORTHOGONAL"
        
        # ORTHOGONAL should be an alias for PARALLEL
        assert StateType.ORTHOGONAL.value == StateType.PARALLEL.value


class TestMachineState:
    """Test MachineState enum."""
    
    def test_machine_state_values(self):
        """Test that MachineState enum has correct values."""
        assert MachineState.UNSPECIFIED.name == "UNSPECIFIED"
        assert MachineState.RUNNING.name == "RUNNING"
        assert MachineState.STOPPED.name == "STOPPED"


class TestEvent:
    """Test Event class."""
    
    def test_event_creation(self):
        """Test creating an Event."""
        evt = Event("TEST_EVENT")
        assert evt.label == "TEST_EVENT"
    
    def test_event_protobuf_conversion(self):
        """Test Event to/from protobuf conversion."""
        evt = Event("TEST_EVENT")
        pb_event = evt.to_pb()
        assert pb_event.label == "TEST_EVENT"
        
        restored_event = Event.from_pb(pb_event)
        assert restored_event.label == evt.label


class TestGuard:
    """Test Guard class."""
    
    def test_guard_creation(self):
        """Test creating a Guard."""
        g = Guard("x > 5")
        assert g.expression == "x > 5"
    
    def test_guard_protobuf_conversion(self):
        """Test Guard to/from protobuf conversion."""
        g = Guard("enabled == true")
        pb_guard = g.to_pb()
        assert pb_guard.expression == "enabled == true"
        
        restored_guard = Guard.from_pb(pb_guard)
        assert restored_guard.expression == g.expression


class TestAction:
    """Test Action class."""
    
    def test_action_creation(self):
        """Test creating an Action."""
        a = Action("log_message")
        assert a.label == "log_message"
    
    def test_action_protobuf_conversion(self):
        """Test Action to/from protobuf conversion."""
        a = Action("send_notification")
        pb_action = a.to_pb()
        assert pb_action.label == "send_notification"
        
        restored_action = Action.from_pb(pb_action)
        assert restored_action.label == a.label


class TestState:
    """Test State class."""
    
    def test_basic_state_creation(self):
        """Test creating a basic state."""
        state = State("TestState", StateType.BASIC)
        assert state.label == "TestState"
        assert state.type == StateType.BASIC
        assert state.children == []
        assert not state.is_initial
        assert not state.is_final
    
    def test_state_with_children(self):
        """Test creating a state with children."""
        child1 = State("Child1", StateType.BASIC, is_initial=True)
        child2 = State("Child2", StateType.BASIC)
        parent = State("Parent", StateType.NORMAL, children=[child1, child2])
        
        assert len(parent.children) == 2
        assert parent.children[0].label == "Child1"
        assert parent.children[1].label == "Child2"
    
    def test_add_child(self):
        """Test adding a child to a state."""
        parent = State("Parent", StateType.BASIC)
        child = State("Child", StateType.BASIC)
        
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert parent.type == StateType.NORMAL  # Should auto-promote from BASIC
    
    def test_find_state(self):
        """Test finding states in hierarchy."""
        child1 = State("Child1", StateType.BASIC, is_initial=True)
        child2 = State("Child2", StateType.BASIC)
        parent = State("Parent", StateType.NORMAL, children=[child1, child2])
        
        # Find existing states
        found_parent = parent.find_state("Parent")
        assert found_parent == parent
        
        found_child = parent.find_state("Child1")
        assert found_child == child1
        
        # Try to find non-existent state
        not_found = parent.find_state("NonExistent")
        assert not_found is None
    
    def test_get_all_states(self):
        """Test getting all states in hierarchy."""
        child1 = State("Child1", StateType.BASIC)
        child2 = State("Child2", StateType.BASIC)
        parent = State("Parent", StateType.NORMAL, children=[child1, child2])
        
        all_states = parent.get_all_states()
        assert len(all_states) == 3
        assert parent in all_states
        assert child1 in all_states
        assert child2 in all_states
    
    def test_state_protobuf_conversion(self):
        """Test State to/from protobuf conversion."""
        child = State("Child", StateType.BASIC, is_initial=True)
        parent = State("Parent", StateType.NORMAL, children=[child], is_final=True)
        
        pb_state = parent.to_pb()
        assert pb_state.label == "Parent"
        assert pb_state.type == StateType.NORMAL.value
        assert pb_state.is_final == True
        assert len(pb_state.children) == 1
        
        restored_state = State.from_pb(pb_state)
        assert restored_state.label == parent.label
        assert restored_state.type == parent.type
        assert restored_state.is_final == parent.is_final
        assert len(restored_state.children) == 1
        assert restored_state.children[0].label == "Child"


class TestTransition:
    """Test Transition class."""
    
    def test_transition_creation(self):
        """Test creating a Transition."""
        trans = Transition(
            label="TestTransition",
            from_states=["State1"],
            to_states=["State2"],
            event="TEST_EVENT"
        )
        
        assert trans.label == "TestTransition"
        assert trans.from_states == ["State1"]
        assert trans.to_states == ["State2"]
        assert trans.event == "TEST_EVENT"
        assert trans.guard is None
        assert trans.actions == []
    
    def test_transition_with_guard_and_actions(self):
        """Test creating a Transition with guard and actions."""
        g = Guard("x > 0")
        a1 = Action("action1")
        a2 = Action("action2")
        
        trans = Transition(
            label="ComplexTransition",
            from_states=["State1", "State2"],
            to_states=["State3"],
            event="COMPLEX_EVENT",
            guard=g,
            actions=[a1, a2]
        )
        
        assert trans.guard == g
        assert len(trans.actions) == 2
        assert trans.actions[0] == a1
        assert trans.actions[1] == a2
    
    def test_transition_protobuf_conversion(self):
        """Test Transition to/from protobuf conversion."""
        g = Guard("enabled == true")
        a = Action("log_transition")
        
        trans = Transition(
            label="TestTrans",
            from_states=["From1", "From2"],
            to_states=["To1"],
            event="EVENT",
            guard=g,
            actions=[a]
        )
        
        pb_trans = trans.to_pb()
        assert pb_trans.label == "TestTrans"
        assert list(pb_trans.from_) == ["From1", "From2"]
        assert list(pb_trans.to) == ["To1"]
        assert pb_trans.event == "EVENT"
        assert pb_trans.guard.expression == "enabled == true"
        assert len(pb_trans.actions) == 1
        
        restored_trans = Transition.from_pb(pb_trans)
        assert restored_trans.label == trans.label
        assert restored_trans.from_states == trans.from_states
        assert restored_trans.to_states == trans.to_states
        assert restored_trans.event == trans.event
        assert restored_trans.guard.expression == trans.guard.expression
        assert len(restored_trans.actions) == 1


class TestConfiguration:
    """Test Configuration class."""
    
    def test_configuration_creation(self):
        """Test creating a Configuration."""
        ref1 = StateRef("State1")
        ref2 = StateRef("State2")
        config = Configuration([ref1, ref2])
        
        assert len(config.states) == 2
        assert config.states[0].label == "State1"
        assert config.states[1].label == "State2"
    
    def test_has_state(self):
        """Test checking if configuration has a state."""
        ref = StateRef("ActiveState")
        config = Configuration([ref])
        
        assert config.has_state("ActiveState")
        assert not config.has_state("InactiveState")
    
    def test_get_state_labels(self):
        """Test getting state labels from configuration."""
        refs = [StateRef("State1"), StateRef("State2"), StateRef("State3")]
        config = Configuration(refs)
        
        labels = config.get_state_labels()
        assert labels == ["State1", "State2", "State3"]


class TestStatechart:
    """Test Statechart class."""
    
    def test_statechart_creation(self):
        """Test creating a Statechart."""
        root = State("Root", StateType.BASIC)
        trans = Transition("Trans", ["A"], ["B"], "EVENT")
        evt = Event("EVENT")
        
        chart = Statechart(root, [trans], [evt])
        
        assert chart.root_state == root
        assert len(chart.transitions) == 1
        assert chart.transitions[0] == trans
        assert len(chart.events) == 1
        assert chart.events[0] == evt
    
    def test_add_transition(self):
        """Test adding transitions to statechart."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        
        trans = Transition("Trans", ["A"], ["B"], "EVENT")
        chart.add_transition(trans)
        
        assert len(chart.transitions) == 1
        assert chart.transitions[0] == trans
    
    def test_add_event(self):
        """Test adding events to statechart."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        
        evt = Event("TEST_EVENT")
        chart.add_event(evt)
        
        assert len(chart.events) == 1
        assert chart.events[0] == evt
    
    def test_find_state(self):
        """Test finding states in statechart."""
        child = State("Child", StateType.BASIC)
        root = State("Root", StateType.NORMAL, children=[child])
        chart = Statechart(root)
        
        found_root = chart.find_state("Root")
        assert found_root == root
        
        found_child = chart.find_state("Child")
        assert found_child == child
        
        not_found = chart.find_state("NotFound")
        assert not_found is None
    
    def test_get_events_for_state(self):
        """Test getting events that can trigger from a state."""
        root = State("Root", StateType.BASIC)
        
        trans1 = Transition("Trans1", ["StateA"], ["StateB"], "EVENT1")
        trans2 = Transition("Trans2", ["StateA"], ["StateC"], "EVENT2")
        trans3 = Transition("Trans3", ["StateB"], ["StateC"], "EVENT1")
        
        chart = Statechart(root, [trans1, trans2, trans3])
        
        events_from_a = chart.get_events_for_state("StateA")
        assert set(events_from_a) == {"EVENT1", "EVENT2"}
        
        events_from_b = chart.get_events_for_state("StateB")
        assert events_from_b == ["EVENT1"]
        
        events_from_unknown = chart.get_events_for_state("Unknown")
        assert events_from_unknown == []
    
    def test_get_transitions_for_event(self):
        """Test getting transitions triggered by an event."""
        root = State("Root", StateType.BASIC)
        
        trans1 = Transition("Trans1", ["StateA"], ["StateB"], "EVENT1")
        trans2 = Transition("Trans2", ["StateA"], ["StateC"], "EVENT2")
        trans3 = Transition("Trans3", ["StateB"], ["StateC"], "EVENT1")
        
        chart = Statechart(root, [trans1, trans2, trans3])
        
        event1_transitions = chart.get_transitions_for_event("EVENT1")
        assert len(event1_transitions) == 2
        assert trans1 in event1_transitions
        assert trans3 in event1_transitions
        
        event2_transitions = chart.get_transitions_for_event("EVENT2")
        assert len(event2_transitions) == 1
        assert trans2 in event2_transitions
        
        unknown_transitions = chart.get_transitions_for_event("UNKNOWN")
        assert unknown_transitions == []


class TestMachine:
    """Test Machine class."""
    
    def test_machine_creation(self):
        """Test creating a Machine."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        context = {"user": "test", "count": 0}
        
        m = Machine("test-machine", chart, context)
        
        assert m.id == "test-machine"
        assert m.state == MachineState.STOPPED
        assert m.context == context
        assert m.statechart == chart
        assert len(m.step_history) == 0
    
    def test_machine_state_checks(self):
        """Test machine state checking methods."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        m = Machine("test-machine", chart)
        
        # Initially stopped
        assert m.is_stopped()
        assert not m.is_running()
        
        # Change to running
        m.state = MachineState.RUNNING
        assert m.is_running()
        assert not m.is_stopped()
    
    def test_machine_to_dict(self):
        """Test machine to dictionary conversion."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        context = {"user": "test"}
        
        m = Machine("test-machine", chart, context)
        
        machine_dict = m.to_dict()
        
        assert machine_dict["id"] == "test-machine"
        assert machine_dict["state"] == "STOPPED"
        assert machine_dict["context"] == context
        assert machine_dict["active_states"] == []
        assert machine_dict["step_count"] == 0
    
    def test_machine_to_json(self):
        """Test machine to JSON conversion."""
        root = State("Root", StateType.BASIC)
        chart = Statechart(root)
        
        m = Machine("test-machine", chart)
        json_str = m.to_json()
        
        assert isinstance(json_str, str)
        assert "test-machine" in json_str
        assert "STOPPED" in json_str


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_basic_state_factory(self):
        """Test basic_state factory function."""
        state = basic_state("TestState", is_initial=True)
        
        assert state.label == "TestState"
        assert state.type == StateType.BASIC
        assert state.is_initial == True
        assert state.is_final == False
        assert state.children == []
    
    def test_normal_state_factory(self):
        """Test normal_state factory function."""
        child1 = basic_state("Child1")
        child2 = basic_state("Child2")
        
        state = normal_state("Parent", [child1, child2], is_final=True)
        
        assert state.label == "Parent"
        assert state.type == StateType.NORMAL
        assert state.is_final == True
        assert len(state.children) == 2
    
    def test_parallel_state_factory(self):
        """Test parallel_state factory function."""
        child1 = basic_state("Child1")
        child2 = basic_state("Child2")
        
        state = parallel_state("Parent", [child1, child2])
        
        assert state.label == "Parent"
        assert state.type == StateType.PARALLEL
        assert len(state.children) == 2
    
    def test_transition_factory(self):
        """Test transition factory function."""
        trans = transition(
            "TestTrans",
            from_states="StateA",
            to_states="StateB", 
            event="EVENT",
            guard="x > 0",
            actions=["action1", "action2"]
        )
        
        assert trans.label == "TestTrans"
        assert trans.from_states == ["StateA"]
        assert trans.to_states == ["StateB"]
        assert trans.event == "EVENT"
        assert trans.guard.expression == "x > 0"
        assert len(trans.actions) == 2
        assert trans.actions[0].label == "action1"
        assert trans.actions[1].label == "action2"
    
    def test_event_factory(self):
        """Test event factory function."""
        evt = event("TEST_EVENT")
        
        assert evt.label == "TEST_EVENT"
    
    def test_guard_factory(self):
        """Test guard factory function."""
        g = guard("enabled == true")
        
        assert g.expression == "enabled == true"
    
    def test_action_factory(self):
        """Test action factory function."""
        a = action("log_message")
        
        assert a.label == "log_message"
    
    def test_statechart_factory(self):
        """Test statechart factory function."""
        root = basic_state("Root")
        trans = transition("Trans", "A", "B", "EVENT")
        evt = event("EVENT")
        
        chart = statechart(root, [trans], [evt])
        
        assert chart.root_state == root
        assert chart.transitions == [trans]
        assert chart.events == [evt]
    
    def test_machine_factory(self):
        """Test machine factory function."""
        root = basic_state("Root")
        chart = statechart(root)
        context = {"user": "test"}
        
        m = machine("test-machine", chart, context)
        
        assert m.id == "test-machine"
        assert m.statechart == chart
        assert m.context == context


if __name__ == "__main__":
    pytest.main([__file__])