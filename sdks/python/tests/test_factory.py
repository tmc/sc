"""
Unit tests for factory functions and builder patterns.

This module tests the factory functions and builder patterns for creating
statechart components with convenient, Pythonic APIs.
"""

import pytest

from statecharts.factory import (
    basic_state,
    normal_state,
    parallel_state,
    orthogonal_state,
    transition,
    event,
    guard,
    action,
    statechart,
    machine,
    configuration,
    StatechartBuilder,
    simple_toggle_statechart,
    hierarchical_statechart_example,
)
from statecharts.core import StateType, MachineState


class TestBasicFactoryFunctions:
    """Test basic factory functions."""
    
    def test_basic_state_factory(self):
        """Test basic_state factory function."""
        state = basic_state("TestState")
        
        assert state.label == "TestState"
        assert state.type == StateType.BASIC
        assert not state.is_initial
        assert not state.is_final
        assert state.children == []
    
    def test_basic_state_with_flags(self):
        """Test basic_state with initial and final flags."""
        state = basic_state("TestState", is_initial=True, is_final=True)
        
        assert state.is_initial
        assert state.is_final
    
    def test_normal_state_factory(self):
        """Test normal_state factory function."""
        child1 = basic_state("Child1")
        child2 = basic_state("Child2")
        
        state = normal_state("Parent", [child1, child2])
        
        assert state.label == "Parent"
        assert state.type == StateType.NORMAL
        assert len(state.children) == 2
        assert state.children[0] == child1
        assert state.children[1] == child2
    
    def test_parallel_state_factory(self):
        """Test parallel_state factory function."""
        child1 = basic_state("Child1")
        child2 = basic_state("Child2")
        
        state = parallel_state("Parent", [child1, child2])
        
        assert state.label == "Parent"
        assert state.type == StateType.PARALLEL
        assert len(state.children) == 2
    
    def test_orthogonal_state_alias(self):
        """Test that orthogonal_state is an alias for parallel_state."""
        child1 = basic_state("Child1")
        child2 = basic_state("Child2")
        
        parallel = parallel_state("Parent", [child1, child2])
        orthogonal = orthogonal_state("Parent", [child1, child2])
        
        assert parallel.type == orthogonal.type
        assert parallel.type == StateType.PARALLEL


class TestTransitionFactory:
    """Test transition factory function."""
    
    def test_simple_transition(self):
        """Test creating a simple transition."""
        trans = transition("TestTrans", "StateA", "StateB", "EVENT")
        
        assert trans.label == "TestTrans"
        assert trans.from_states == ["StateA"]
        assert trans.to_states == ["StateB"]
        assert trans.event == "EVENT"
        assert trans.guard is None
        assert trans.actions == []
    
    def test_transition_with_multiple_states(self):
        """Test transition with multiple from/to states."""
        trans = transition(
            "MultiTrans",
            from_states=["StateA", "StateB"],
            to_states=["StateC", "StateD"],
            event="EVENT"
        )
        
        assert trans.from_states == ["StateA", "StateB"]
        assert trans.to_states == ["StateC", "StateD"]
    
    def test_transition_with_guard_string(self):
        """Test transition with guard as string."""
        trans = transition(
            "GuardedTrans",
            "StateA",
            "StateB",
            "EVENT",
            guard="x > 5"
        )
        
        assert trans.guard is not None
        assert trans.guard.expression == "x > 5"
    
    def test_transition_with_guard_object(self):
        """Test transition with guard as Guard object."""
        guard_obj = guard("enabled == true")
        trans = transition(
            "GuardedTrans",
            "StateA",
            "StateB",
            "EVENT",
            guard=guard_obj
        )
        
        assert trans.guard == guard_obj
    
    def test_transition_with_single_action_string(self):
        """Test transition with single action as string."""
        trans = transition(
            "ActionTrans",
            "StateA",
            "StateB",
            "EVENT",
            actions="log_message"
        )
        
        assert len(trans.actions) == 1
        assert trans.actions[0].label == "log_message"
    
    def test_transition_with_multiple_action_strings(self):
        """Test transition with multiple actions as strings."""
        trans = transition(
            "ActionTrans",
            "StateA",
            "StateB",
            "EVENT",
            actions=["action1", "action2", "action3"]
        )
        
        assert len(trans.actions) == 3
        assert trans.actions[0].label == "action1"
        assert trans.actions[1].label == "action2"
        assert trans.actions[2].label == "action3"
    
    def test_transition_with_action_objects(self):
        """Test transition with Action objects."""
        action1 = action("action1")
        action2 = action("action2")
        
        trans = transition(
            "ActionTrans",
            "StateA",
            "StateB",
            "EVENT",
            actions=[action1, action2]
        )
        
        assert len(trans.actions) == 2
        assert trans.actions[0] == action1
        assert trans.actions[1] == action2


class TestUtilityFactories:
    """Test utility factory functions."""
    
    def test_event_factory(self):
        """Test event factory function."""
        evt = event("TEST_EVENT")
        
        assert evt.label == "TEST_EVENT"
    
    def test_guard_factory(self):
        """Test guard factory function."""
        g = guard("x > 10")
        
        assert g.expression == "x > 10"
    
    def test_action_factory(self):
        """Test action factory function."""
        a = action("log_action")
        
        assert a.label == "log_action"
    
    def test_configuration_factory(self):
        """Test configuration factory function."""
        config = configuration(["State1", "State2", "State3"])
        
        assert len(config.states) == 3
        assert config.states[0].label == "State1"
        assert config.states[1].label == "State2"
        assert config.states[2].label == "State3"


class TestStatechartFactory:
    """Test statechart factory function."""
    
    def test_simple_statechart(self):
        """Test creating a simple statechart."""
        root = basic_state("Root")
        chart = statechart(root)
        
        assert chart.root_state == root
        assert chart.transitions == []
        assert chart.events == []
    
    def test_statechart_with_transitions_and_events(self):
        """Test creating statechart with transitions and events."""
        root = basic_state("Root")
        trans = transition("Trans", "A", "B", "EVENT")
        evt = event("EVENT")
        
        chart = statechart(root, [trans], [evt])
        
        assert chart.root_state == root
        assert len(chart.transitions) == 1
        assert chart.transitions[0] == trans
        assert len(chart.events) == 1
        assert chart.events[0] == evt


class TestMachineFactory:
    """Test machine factory function."""
    
    def test_simple_machine(self):
        """Test creating a simple machine."""
        root = basic_state("Root")
        chart = statechart(root)
        
        m = machine("test-machine", chart)
        
        assert m.id == "test-machine"
        assert m.statechart == chart
        assert m.context == {}
        assert m.state == MachineState.STOPPED
    
    def test_machine_with_context(self):
        """Test creating machine with context."""
        root = basic_state("Root")
        chart = statechart(root)
        context = {"user": "test", "count": 42}
        
        m = machine("test-machine", chart, context)
        
        assert m.context == context


class TestStatechartBuilder:
    """Test StatechartBuilder class."""
    
    def test_simple_builder(self):
        """Test building a simple statechart."""
        chart = (StatechartBuilder()
                .root_state("Root")
                .build())
        
        assert chart.root_state.label == "Root"
        assert chart.root_state.type == StateType.NORMAL  # Default for root
        assert chart.transitions == []
        assert chart.events == []
    
    def test_builder_with_states(self):
        """Test builder with multiple states."""
        chart = (StatechartBuilder()
                .root_state("Root")
                .add_state("Child1", parent="Root", is_initial=True)
                .add_state("Child2", parent="Root")
                .build())
        
        assert chart.root_state.label == "Root"
        assert len(chart.root_state.children) == 2
        assert chart.root_state.children[0].label == "Child1"
        assert chart.root_state.children[0].is_initial
        assert chart.root_state.children[1].label == "Child2"
    
    def test_builder_with_transitions(self):
        """Test builder with transitions."""
        chart = (StatechartBuilder()
                .root_state("Root")
                .add_state("StateA", parent="Root", is_initial=True)
                .add_state("StateB", parent="Root")
                .add_transition("Trans", "StateA", "StateB", "EVENT")
                .build())
        
        assert len(chart.transitions) == 1
        trans = chart.transitions[0]
        assert trans.label == "Trans"
        assert trans.from_states == ["StateA"]
        assert trans.to_states == ["StateB"]
        assert trans.event == "EVENT"
    
    def test_builder_with_events(self):
        """Test builder with events."""
        chart = (StatechartBuilder()
                .root_state("Root")
                .add_event("EVENT1")
                .add_event("EVENT2")
                .build())
        
        assert len(chart.events) == 2
        assert chart.events[0].label == "EVENT1"
        assert chart.events[1].label == "EVENT2"
    
    def test_builder_complex_statechart(self):
        """Test building a complex statechart."""
        chart = (StatechartBuilder()
                .root_state("System")
                .add_state("Off", parent="System", is_initial=True)
                .add_state("On", parent="System", state_type=StateType.NORMAL)
                .add_state("Idle", parent="On", is_initial=True)
                .add_state("Active", parent="On")
                .add_transition("PowerOn", "Off", "On", "POWER")
                .add_transition("PowerOff", "On", "Off", "POWER")
                .add_transition("Activate", "Idle", "Active", "START")
                .add_transition("Deactivate", "Active", "Idle", "STOP")
                .add_event("POWER")
                .add_event("START")
                .add_event("STOP")
                .build())
        
        assert chart.root_state.label == "System"
        assert len(chart.root_state.children) == 2  # Off, On
        
        on_state = chart.find_state("On")
        assert on_state is not None
        assert len(on_state.children) == 2  # Idle, Active
        
        assert len(chart.transitions) == 4
        assert len(chart.events) == 3
    
    def test_builder_missing_root(self):
        """Test builder fails without root state."""
        builder = StatechartBuilder()
        builder.add_state("SomeState")
        
        with pytest.raises(ValueError, match="Root state must be set"):
            builder.build()


class TestConveniencePatterns:
    """Test convenience patterns and example statecharts."""
    
    def test_simple_toggle_statechart(self):
        """Test simple toggle statechart pattern."""
        chart = simple_toggle_statechart()
        
        assert chart.root_state.label == "Root"
        assert len(chart.root_state.children) == 2
        
        # Check states
        off_state = chart.find_state("Off")
        on_state = chart.find_state("On")
        assert off_state is not None
        assert on_state is not None
        assert off_state.is_initial
        
        # Check transitions
        assert len(chart.transitions) == 2
        toggle_transitions = chart.get_transitions_for_event("TOGGLE")
        assert len(toggle_transitions) == 2
        
        # Check events
        assert len(chart.events) == 1
        assert chart.events[0].label == "TOGGLE"
    
    def test_simple_toggle_with_custom_labels(self):
        """Test simple toggle with custom labels."""
        chart = simple_toggle_statechart("Active", "Inactive", "SWITCH")
        
        active_state = chart.find_state("Active")
        inactive_state = chart.find_state("Inactive")
        assert active_state is not None
        assert inactive_state is not None
        assert inactive_state.is_initial
        
        switch_transitions = chart.get_transitions_for_event("SWITCH")
        assert len(switch_transitions) == 2
        
        assert chart.events[0].label == "SWITCH"
    
    def test_hierarchical_statechart_example(self):
        """Test hierarchical statechart example."""
        chart = hierarchical_statechart_example()
        
        # Check root structure
        system_state = chart.root_state
        assert system_state.label == "System"
        
        # Check hierarchy
        off_state = chart.find_state("Off")
        on_state = chart.find_state("On")
        idle_state = chart.find_state("Idle")
        active_state = chart.find_state("Active")
        
        assert off_state is not None
        assert on_state is not None
        assert idle_state is not None
        assert active_state is not None
        
        assert off_state.is_initial
        assert idle_state.is_initial
        
        # Check transitions
        assert len(chart.transitions) == 4
        power_transitions = chart.get_transitions_for_event("POWER")
        assert len(power_transitions) == 2  # PowerOn and PowerOff
        
        # Check events
        assert len(chart.events) == 3
        event_labels = [evt.label for evt in chart.events]
        assert "POWER" in event_labels
        assert "START" in event_labels
        assert "STOP" in event_labels


class TestFactoryEdgeCases:
    """Test edge cases and error conditions in factory functions."""
    
    def test_empty_children_lists(self):
        """Test factory functions with empty children lists."""
        # This should work but might trigger validation errors later
        state = normal_state("Empty", [])
        
        assert state.label == "Empty"
        assert state.type == StateType.NORMAL
        assert state.children == []
    
    def test_transition_with_empty_actions(self):
        """Test transition with empty actions list."""
        trans = transition("Trans", "A", "B", "EVENT", actions=[])
        
        assert trans.actions == []
    
    def test_statechart_with_empty_lists(self):
        """Test statechart with empty transition and event lists."""
        root = basic_state("Root")
        chart = statechart(root, [], [])
        
        assert chart.transitions == []
        assert chart.events == []
    
    def test_machine_with_empty_context(self):
        """Test machine with explicitly empty context."""
        root = basic_state("Root")
        chart = statechart(root)
        m = machine("test", chart, {})
        
        assert m.context == {}


if __name__ == "__main__":
    pytest.main([__file__])