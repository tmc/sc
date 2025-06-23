#!/usr/bin/env python3
"""
Simple Statechart Example

This example demonstrates how to create a simple two-state statechart
using the Python statecharts library.
"""

from statecharts import (
    basic_state,
    normal_state,
    transition,
    event,
    statechart,
    machine,
    validate_statechart,
)


def create_simple_toggle_statechart():
    """
    Create a simple toggle statechart that switches between Off and On states.
    
    The statechart has two states:
    - Off (initial state)
    - On
    
    It responds to a TOGGLE event that switches between the states.
    """
    # Create the basic states
    off_state = basic_state("Off", is_initial=True)
    on_state = basic_state("On")
    
    # Create the root composite state
    root_state = normal_state("ToggleSwitch", [off_state, on_state])
    
    # Create transitions
    turn_on = transition(
        label="TurnOn",
        from_states="Off",
        to_states="On", 
        event="TOGGLE"
    )
    
    turn_off = transition(
        label="TurnOff",
        from_states="On",
        to_states="Off",
        event="TOGGLE"
    )
    
    # Create events
    toggle_event = event("TOGGLE")
    
    # Create the statechart
    chart = statechart(
        root_state=root_state,
        transitions=[turn_on, turn_off],
        events=[toggle_event]
    )
    
    return chart


def demonstrate_basic_usage():
    """Demonstrate basic usage of the statechart."""
    print("=== Simple Statechart Example ===\n")
    
    # Create the statechart
    chart = create_simple_toggle_statechart()
    print(f"Created statechart with root state: {chart.root_state.label}")
    print(f"Number of states: {len(chart.get_all_states())}")
    print(f"Number of transitions: {len(chart.transitions)}")
    print(f"Number of events: {len(chart.events)}")
    
    # List all states
    print("\nAll states:")
    for state in chart.get_all_states():
        state_type = state.type.name
        initial = " (initial)" if state.is_initial else ""
        print(f"  - {state.label} [{state_type}]{initial}")
    
    # List all transitions
    print("\nTransitions:")
    for trans in chart.transitions:
        from_states = ", ".join(trans.from_states)
        to_states = ", ".join(trans.to_states)
        print(f"  - {trans.label}: {from_states} --{trans.event}--> {to_states}")
    
    # Validate the statechart
    print("\nValidation:")
    validation_result = validate_statechart(chart)
    if validation_result.is_valid:
        print("  ✓ Statechart is valid")
        if validation_result.has_warnings:
            print(f"  ⚠ {len(validation_result.get_warnings())} warning(s)")
            for warning in validation_result.get_warnings():
                print(f"    - {warning}")
    else:
        print("  ✗ Statechart has errors:")
        for error in validation_result.get_errors():
            print(f"    - {error}")
    
    # Create a machine instance
    print("\nMachine Instance:")
    my_machine = machine("toggle-machine-1", chart, context={"user": "demo"})
    print(f"  Machine ID: {my_machine.id}")
    print(f"  State: {my_machine.state.name}")
    print(f"  Context: {my_machine.context}")
    
    print("\n" + "="*50)


def demonstrate_statechart_querying():
    """Demonstrate querying capabilities of the statechart."""
    print("=== Statechart Querying Example ===\n")
    
    chart = create_simple_toggle_statechart()
    
    # Find states
    print("Finding states:")
    off_state = chart.find_state("Off")
    if off_state:
        print(f"  Found state: {off_state.label} (type: {off_state.type.name})")
    
    unknown_state = chart.find_state("Unknown")
    print(f"  Unknown state found: {unknown_state is not None}")
    
    # Get events for a state
    print(f"\nEvents that can trigger from 'Off' state:")
    events_from_off = chart.get_events_for_state("Off")
    for event_name in events_from_off:
        print(f"  - {event_name}")
    
    # Get transitions for an event
    print(f"\nTransitions triggered by 'TOGGLE' event:")
    toggle_transitions = chart.get_transitions_for_event("TOGGLE")
    for trans in toggle_transitions:
        from_states = ", ".join(trans.from_states)
        to_states = ", ".join(trans.to_states)
        print(f"  - {trans.label}: {from_states} → {to_states}")
    
    print("\n" + "="*50)


def main():
    """Run the simple statechart examples."""
    demonstrate_basic_usage()
    demonstrate_statechart_querying()


if __name__ == "__main__":
    main()