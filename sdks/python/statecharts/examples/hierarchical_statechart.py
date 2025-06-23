#!/usr/bin/env python3
"""
Hierarchical Statechart Example

This example demonstrates how to create hierarchical (nested) statecharts
with multiple levels of states and complex transitions.
"""

from statecharts import (
    basic_state,
    normal_state,
    transition,
    event,
    statechart,
    machine,
    guard,
    action,
    validate_statechart,
    StatechartBuilder,
)


def create_alarm_system_statechart():
    """
    Create a hierarchical alarm system statechart.
    
    Structure:
    - AlarmSystem (root)
      ├── Off (initial)
      └── On
          ├── Disarmed (initial)
          ├── Armed
          │   ├── Home (initial)
          │   └── Away
          └── Triggered
    """
    # Create leaf states
    off = basic_state("Off", is_initial=True)
    disarmed = basic_state("Disarmed", is_initial=True)
    home = basic_state("Home", is_initial=True)
    away = basic_state("Away")
    triggered = basic_state("Triggered")
    
    # Create composite states
    armed = normal_state("Armed", [home, away])
    on = normal_state("On", [disarmed, armed, triggered])
    
    # Create root state
    root = normal_state("AlarmSystem", [off, on])
    
    # Create transitions with guards and actions
    transitions = [
        # Power on/off transitions
        transition(
            "PowerOn",
            from_states="Off",
            to_states="On",
            event="POWER",
            actions=["initialize_system", "log_power_on"]
        ),
        transition(
            "PowerOff", 
            from_states="On",
            to_states="Off",
            event="POWER",
            actions=["shutdown_system", "log_power_off"]
        ),
        
        # Arming/disarming transitions
        transition(
            "ArmHome",
            from_states="Disarmed",
            to_states="Home",
            event="ARM",
            guard="mode == 'home'",
            actions=["arm_sensors", "log_arm_home"]
        ),
        transition(
            "ArmAway",
            from_states="Disarmed", 
            to_states="Away",
            event="ARM",
            guard="mode == 'away'",
            actions=["arm_all_sensors", "log_arm_away"]
        ),
        transition(
            "Disarm",
            from_states="Armed",
            to_states="Disarmed",
            event="DISARM",
            guard="valid_code == true",
            actions=["disarm_sensors", "log_disarm"]
        ),
        
        # Mode switching within Armed state
        transition(
            "SwitchToHome",
            from_states="Away",
            to_states="Home", 
            event="MODE_CHANGE",
            guard="mode == 'home'",
            actions=["adjust_sensors_for_home"]
        ),
        transition(
            "SwitchToAway",
            from_states="Home",
            to_states="Away",
            event="MODE_CHANGE", 
            guard="mode == 'away'",
            actions=["adjust_sensors_for_away"]
        ),
        
        # Triggering transitions
        transition(
            "TriggerAlarm",
            from_states="Armed",
            to_states="Triggered",
            event="SENSOR_BREACH",
            actions=["sound_alarm", "notify_security", "record_breach"]
        ),
        transition(
            "ResetAlarm",
            from_states="Triggered",
            to_states="Disarmed",
            event="RESET",
            guard="valid_code == true",
            actions=["stop_alarm", "log_reset"]
        ),
    ]
    
    # Create events
    events = [
        event("POWER"),
        event("ARM"),
        event("DISARM"),
        event("MODE_CHANGE"),
        event("SENSOR_BREACH"),
        event("RESET"),
    ]
    
    return statechart(root, transitions, events)


def create_media_player_statechart():
    """
    Create a hierarchical media player statechart using the builder pattern.
    
    Structure:
    - MediaPlayer (root)
      ├── Stopped (initial)
      └── Playing
          ├── NormalSpeed (initial)
          ├── FastForward
          └── Rewind
    """
    chart = (StatechartBuilder()
        .root_state("MediaPlayer")
        .add_state("Stopped", parent="MediaPlayer", is_initial=True)
        .add_state("Playing", parent="MediaPlayer", state_type=StateType.NORMAL)
        .add_state("NormalSpeed", parent="Playing", is_initial=True)
        .add_state("FastForward", parent="Playing")
        .add_state("Rewind", parent="Playing")
        
        # Transitions
        .add_transition("Play", "Stopped", "Playing", "PLAY", actions=["start_playback"])
        .add_transition("Stop", "Playing", "Stopped", "STOP", actions=["stop_playback"])
        .add_transition("FastFwd", "NormalSpeed", "FastForward", "FF", actions=["increase_speed"])
        .add_transition("SlowDown", "FastForward", "NormalSpeed", "NORMAL", actions=["reset_speed"])
        .add_transition("Reverse", "NormalSpeed", "Rewind", "REW", actions=["reverse_playback"])
        .add_transition("Forward", "Rewind", "NormalSpeed", "NORMAL", actions=["forward_playback"])
        
        # Events
        .add_event("PLAY")
        .add_event("STOP")
        .add_event("FF")
        .add_event("REW") 
        .add_event("NORMAL")
        
        .build())
    
    return chart


def demonstrate_hierarchical_navigation():
    """Demonstrate navigation in hierarchical statecharts."""
    print("=== Hierarchical Navigation Example ===\n")
    
    chart = create_alarm_system_statechart()
    
    # Show the hierarchy
    print("Statechart hierarchy:")
    def print_state_hierarchy(state, indent=0):
        prefix = "  " * indent
        state_type = state.type.name
        initial = " (initial)" if state.is_initial else ""
        final = " (final)" if state.is_final else ""
        print(f"{prefix}- {state.label} [{state_type}]{initial}{final}")
        
        for child in state.children:
            print_state_hierarchy(child, indent + 1)
    
    print_state_hierarchy(chart.root_state)
    
    # Find states at different levels
    print(f"\nFinding states:")
    states_to_find = ["AlarmSystem", "On", "Armed", "Home", "NonExistent"]
    for state_name in states_to_find:
        found = chart.find_state(state_name)
        status = "✓" if found else "✗"
        print(f"  {status} {state_name}")
    
    # Show transitions by level
    print(f"\nTransitions organized by source state:")
    transitions_by_source = {}
    for trans in chart.transitions:
        for from_state in trans.from_states:
            if from_state not in transitions_by_source:
                transitions_by_source[from_state] = []
            transitions_by_source[from_state].append(trans)
    
    for state_name in sorted(transitions_by_source.keys()):
        print(f"  From '{state_name}':")
        for trans in transitions_by_source[state_name]:
            to_states = ", ".join(trans.to_states)
            guard_info = f" [{trans.guard.expression}]" if trans.guard else ""
            actions_info = f" -> {[a.label for a in trans.actions]}" if trans.actions else ""
            print(f"    - {trans.event}{guard_info} → {to_states}{actions_info}")
    
    print("\n" + "="*50)


def demonstrate_builder_pattern():
    """Demonstrate the builder pattern for creating statecharts."""
    print("=== Builder Pattern Example ===\n")
    
    chart = create_media_player_statechart()
    
    print("Media Player Statechart (created with builder pattern):")
    print(f"  Root state: {chart.root_state.label}")
    print(f"  Total states: {len(chart.get_all_states())}")
    print(f"  Total transitions: {len(chart.transitions)}")
    
    # Validate the built statechart
    result = validate_statechart(chart)
    print(f"\nValidation: {'✓ Valid' if result.is_valid else '✗ Invalid'}")
    
    if not result.is_valid:
        for error in result.get_errors():
            print(f"  Error: {error}")
    
    if result.has_warnings:
        for warning in result.get_warnings():
            print(f"  Warning: {warning}")
    
    print("\n" + "="*50)


def demonstrate_complex_transitions():
    """Demonstrate transitions with guards and actions."""
    print("=== Complex Transitions Example ===\n")
    
    chart = create_alarm_system_statechart()
    
    print("Transitions with guards and actions:")
    for trans in chart.transitions:
        if trans.guard or trans.actions:
            from_states = ", ".join(trans.from_states)
            to_states = ", ".join(trans.to_states)
            
            print(f"\n  {trans.label}: {from_states} → {to_states}")
            print(f"    Event: {trans.event}")
            
            if trans.guard:
                print(f"    Guard: {trans.guard.expression}")
            
            if trans.actions:
                action_labels = [action.label for action in trans.actions]
                print(f"    Actions: {', '.join(action_labels)}")
    
    print("\n" + "="*50)


def main():
    """Run the hierarchical statechart examples."""
    demonstrate_hierarchical_navigation()
    demonstrate_builder_pattern()
    demonstrate_complex_transitions()


if __name__ == "__main__":
    main()