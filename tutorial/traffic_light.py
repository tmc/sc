#!/usr/bin/env python3
"""
Smart Traffic Light Tutorial - Python Version

This tutorial demonstrates how to build a smart traffic light system using
the Python statecharts library. The traffic light showcases:

1. Hierarchical state composition (Normal vs Emergency modes)
2. State transitions with events  
3. Emergency interrupt behavior
4. Machine instances with context

This is a "Hello World" example that shows the practical power of statecharts
for modeling complex behavior in a simple, understandable way.
"""

import sys
import os

# Add the SDK to the path for this tutorial
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

from statecharts import (
    basic_state,
    normal_state,
    transition,
    event,
    statechart,
    machine,
    validate_statechart,
)


def create_smart_traffic_light():
    """
    Create a smart traffic light statechart.
    
    The traffic light has two main operational modes:
    1. Normal: Standard RED -> GREEN -> YELLOW -> RED cycle
    2. Emergency: Flashing red for all directions
    
    Plus a maintenance mode for servicing.
    """
    
    # Create normal operation states
    red_state = basic_state("Red", is_initial=True)
    green_state = basic_state("Green")
    yellow_state = basic_state("Yellow")
    normal_mode = normal_state("Normal", [red_state, green_state, yellow_state], is_initial=True)
    
    # Create emergency operation states  
    flashing_red = basic_state("FlashingRed", is_initial=True)
    flashing_off = basic_state("FlashingOff")
    emergency_mode = normal_state("Emergency", [flashing_red, flashing_off])
    
    # Create maintenance state
    maintenance_state = basic_state("Maintenance")
    
    # Create root state combining all modes
    root_state = normal_state("SmartTrafficLight", [
        normal_mode,
        emergency_mode, 
        maintenance_state
    ])
    
    # Define transitions
    transitions = [
        # Normal operation cycle
        transition("ToGreen", "Red", "Green", "TIMER_EXPIRED"),
        transition("ToYellow", "Green", "Yellow", "TIMER_EXPIRED"), 
        transition("ToRed", "Yellow", "Red", "TIMER_EXPIRED"),
        
        # Emergency mode transitions
        transition("EmergencyActivate", "Normal", "Emergency", "EMERGENCY"),
        transition("EmergencyDeactivate", "Emergency", "Normal", "EMERGENCY_CLEAR"),
        
        # Emergency flashing cycle
        transition("FlashOn", "FlashingOff", "FlashingRed", "FLASH_TIMER"),
        transition("FlashOff", "FlashingRed", "FlashingOff", "FLASH_TIMER"),
        
        # Maintenance mode
        transition("EnterMaintenance", ["Normal", "Emergency"], "Maintenance", "MAINTENANCE"),
        transition("ExitMaintenance", "Maintenance", "Normal", "MAINTENANCE_COMPLETE"),
    ]
    
    # Define events
    events = [
        event("TIMER_EXPIRED"),
        event("EMERGENCY"),
        event("EMERGENCY_CLEAR"),
        event("FLASH_TIMER"),
        event("MAINTENANCE"),
        event("MAINTENANCE_COMPLETE"),
    ]
    
    # Create the complete statechart
    chart = statechart(
        root_state=root_state,
        transitions=transitions,
        events=events
    )
    
    return chart


def demonstrate_traffic_light():
    """Demonstrate the smart traffic light functionality."""
    
    print("=== Smart Traffic Light Tutorial - Python ===")
    print("Building a traffic light with statecharts...")
    print()
    
    # Step 1: Create the statechart
    chart = create_smart_traffic_light()
    print("✓ Created traffic light statechart")
    
    # Step 2: Validate the statechart
    validation_result = validate_statechart(chart)
    if validation_result.is_valid:
        print("✓ Statechart is valid")
        if validation_result.has_warnings:
            print(f"  ⚠ {len(validation_result.get_warnings())} warning(s)")
    else:
        print("✗ Statechart has validation errors:")
        for error in validation_result.get_errors():
            print(f"    - {error}")
        return
    
    # Step 3: Show the structure
    print("\n--- Traffic Light Structure ---")
    print(f"Root state: {chart.root_state.label}")
    print(f"Number of states: {len(chart.get_all_states())}")
    print(f"Number of transitions: {len(chart.transitions)}")
    print(f"Number of events: {len(chart.events)}")
    
    # Show all states
    print("\nAll states:")
    for state in chart.get_all_states():
        state_type = state.type.name
        initial = " (initial)" if state.is_initial else ""
        final = " (final)" if state.is_final else ""
        indent = "  " * (state.label.count('.') if hasattr(state, 'path') else 0)
        print(f"  {indent}- {state.label} [{state_type}]{initial}{final}")
    
    # Show transitions  
    print("\nTransitions:")
    for trans in chart.transitions:
        from_states = ", ".join(trans.from_states) if isinstance(trans.from_states, list) else trans.from_states
        to_states = ", ".join(trans.to_states) if isinstance(trans.to_states, list) else trans.to_states
        print(f"  - {trans.label}: {from_states} --{trans.event}--> {to_states}")
    
    # Step 4: Create a machine instance
    print("\n--- Creating Machine Instance ---")
    traffic_machine = machine(
        machine_id="traffic-light-001",
        chart=chart,
        context={
            "timer": 0,
            "emergency_mode": False,
            "last_maintenance": "2024-01-01"
        }
    )
    print(f"✓ Created machine: {traffic_machine.id}")
    print(f"Machine state: {traffic_machine.state.name}")
    print(f"Context: {traffic_machine.context}")
    
    # Step 5: Demonstrate the power of statecharts
    print("\n--- Why Statecharts Are Powerful ---")
    print("Traditional if/else approach for traffic lights:")
    print("""
    if current_state == "red":
        if event == "timer_expired":
            if emergency_mode:
                current_state = "flashing_red"
            else:
                current_state = "green" 
        elif event == "emergency":
            emergency_mode = True
            current_state = "flashing_red"
    # ... many more nested conditions
    """)
    
    print("With statecharts:")
    print("- Clear visual representation of all possible states")
    print("- Hierarchical organization (Normal/Emergency modes)")
    print("- Explicit transitions with events")  
    print("- Impossible states are impossible to reach")
    print("- Easy to test, debug, and extend")
    
    # Step 6: Show querying capabilities
    print("\n--- Statechart Querying ---")
    
    # Find specific states
    red_state = chart.find_state("Red")
    if red_state:
        print(f"Found state: {red_state.label} (type: {red_state.type.name})")
    
    # Get events for a state
    events_from_red = chart.get_events_for_state("Red")
    print(f"Events from Red state: {list(events_from_red)}")
    
    # Get transitions for an event
    emergency_transitions = chart.get_transitions_for_event("EMERGENCY")
    print(f"Emergency transitions: {len(emergency_transitions)}")
    for trans in emergency_transitions:
        from_states = ", ".join(trans.from_states) if isinstance(trans.from_states, list) else trans.from_states
        to_states = ", ".join(trans.to_states) if isinstance(trans.to_states, list) else trans.to_states
        print(f"  - {trans.label}: {from_states} → {to_states}")
    
    print("\n--- Tutorial Complete! ---")
    print("You've successfully:")
    print("✓ Created a hierarchical statechart in Python")
    print("✓ Modeled complex behavior with clear states")
    print("✓ Demonstrated emergency interrupt patterns")
    print("✓ Explored statechart querying capabilities")
    print("✓ Seen why statecharts beat traditional state management")
    
    print("\nNext steps:")
    print("- Try the web visualizer to see your statechart graphically")
    print("- Explore parallel states for concurrent behavior")
    print("- Add guards and actions for complex logic")
    print("- Integrate with FastAPI or Flask (see integrations/)")
    print("- Check out the Go and JavaScript versions")


def main():
    """Run the smart traffic light tutorial."""
    try:
        demonstrate_traffic_light()
    except ImportError as e:
        print("Error: Could not import statecharts library")
        print("Make sure you're running from the correct directory")
        print(f"Import error: {e}")
        print("\nTo set up the Python SDK:")
        print("cd sdks/python && make install-dev")
    except Exception as e:
        print(f"Tutorial error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()