#!/usr/bin/env python3
"""
Orthogonal (Parallel) Statechart Example

This example demonstrates how to create orthogonal statecharts where
multiple states are active simultaneously (AND semantics).
"""

from statecharts import (
    basic_state,
    normal_state,
    parallel_state,
    orthogonal_state,  # alias for parallel_state
    transition,
    event,
    statechart,
    machine,
    validate_statechart,
    StateType,
)


def create_smartphone_statechart():
    """
    Create an orthogonal statechart for a smartphone.
    
    The smartphone has multiple independent subsystems that operate concurrently:
    - Display (On/Off)
    - Audio (Muted/Unmuted)  
    - Network (Connected/Disconnected)
    - Battery (Charging/Discharging)
    
    Structure:
    - Smartphone (root, parallel)
      ├── Display
      │   ├── Off (initial)
      │   └── On
      ├── Audio
      │   ├── Unmuted (initial)
      │   └── Muted
      ├── Network
      │   ├── Disconnected (initial)
      │   └── Connected
      └── Battery
          ├── Discharging (initial)
          └── Charging
    """
    # Display subsystem
    display_off = basic_state("DisplayOff", is_initial=True)
    display_on = basic_state("DisplayOn")
    display_system = normal_state("Display", [display_off, display_on])
    
    # Audio subsystem
    audio_unmuted = basic_state("Unmuted", is_initial=True)
    audio_muted = basic_state("Muted")
    audio_system = normal_state("Audio", [audio_unmuted, audio_muted])
    
    # Network subsystem
    network_disconnected = basic_state("Disconnected", is_initial=True)
    network_connected = basic_state("Connected")
    network_system = normal_state("Network", [network_disconnected, network_connected])
    
    # Battery subsystem
    battery_discharging = basic_state("Discharging", is_initial=True)
    battery_charging = basic_state("Charging")
    battery_system = normal_state("Battery", [battery_discharging, battery_charging])
    
    # Root parallel state (all subsystems active simultaneously)
    root = parallel_state("Smartphone", [
        display_system,
        audio_system, 
        network_system,
        battery_system
    ])
    
    # Create transitions
    transitions = [
        # Display transitions
        transition("TurnDisplayOn", "DisplayOff", "DisplayOn", "DISPLAY_BUTTON"),
        transition("TurnDisplayOff", "DisplayOn", "DisplayOff", "DISPLAY_BUTTON"),
        transition("AutoDisplayOff", "DisplayOn", "DisplayOff", "DISPLAY_TIMEOUT"),
        
        # Audio transitions
        transition("MuteAudio", "Unmuted", "Muted", "MUTE_BUTTON"),
        transition("UnmuteAudio", "Muted", "Unmuted", "MUTE_BUTTON"),
        
        # Network transitions
        transition("ConnectNetwork", "Disconnected", "Connected", "NETWORK_AVAILABLE"),
        transition("DisconnectNetwork", "Connected", "Disconnected", "NETWORK_LOST"),
        
        # Battery transitions
        transition("StartCharging", "Discharging", "Charging", "POWER_CONNECTED"),
        transition("StopCharging", "Charging", "Discharging", "POWER_DISCONNECTED"),
    ]
    
    # Create events
    events = [
        event("DISPLAY_BUTTON"),
        event("DISPLAY_TIMEOUT"),
        event("MUTE_BUTTON"),
        event("NETWORK_AVAILABLE"),
        event("NETWORK_LOST"),
        event("POWER_CONNECTED"),
        event("POWER_DISCONNECTED"),
    ]
    
    return statechart(root, transitions, events)


def create_car_entertainment_statechart():
    """
    Create an orthogonal statechart for a car entertainment system.
    
    The system has independent components:
    - Radio (Off/AM/FM)
    - Climate (Auto/Manual)
    - Navigation (Idle/Navigating)
    """
    # Radio subsystem
    radio_off = basic_state("RadioOff", is_initial=True)
    radio_am = basic_state("AM")  
    radio_fm = basic_state("FM")
    radio_system = normal_state("Radio", [radio_off, radio_am, radio_fm])
    
    # Climate control subsystem
    climate_auto = basic_state("Auto", is_initial=True)
    climate_manual = basic_state("Manual")
    climate_system = normal_state("Climate", [climate_auto, climate_manual])
    
    # Navigation subsystem  
    nav_idle = basic_state("Idle", is_initial=True)
    nav_navigating = basic_state("Navigating")
    nav_system = normal_state("Navigation", [nav_idle, nav_navigating])
    
    # Root orthogonal state
    root = orthogonal_state("CarEntertainment", [
        radio_system,
        climate_system,
        nav_system
    ])
    
    # Transitions
    transitions = [
        # Radio transitions
        transition("RadioOn", "RadioOff", "AM", "RADIO_POWER"),
        transition("RadioOff", ["AM", "FM"], "RadioOff", "RADIO_POWER"),
        transition("SwitchToAM", "FM", "AM", "BAND_SWITCH"),
        transition("SwitchToFM", "AM", "FM", "BAND_SWITCH"),
        
        # Climate transitions
        transition("SetManual", "Auto", "Manual", "CLIMATE_MODE"),
        transition("SetAuto", "Manual", "Auto", "CLIMATE_MODE"),
        
        # Navigation transitions
        transition("StartNavigation", "Idle", "Navigating", "SET_DESTINATION"),
        transition("StopNavigation", "Navigating", "Idle", "CANCEL_NAVIGATION"),
    ]
    
    events = [
        event("RADIO_POWER"),
        event("BAND_SWITCH"), 
        event("CLIMATE_MODE"),
        event("SET_DESTINATION"),
        event("CANCEL_NAVIGATION"),
    ]
    
    return statechart(root, transitions, events)


def demonstrate_orthogonal_behavior():
    """Demonstrate the behavior of orthogonal statecharts."""
    print("=== Orthogonal Statechart Behavior ===\n")
    
    chart = create_smartphone_statechart()
    
    print(f"Smartphone statechart structure:")
    print(f"  Root state: {chart.root_state.label} (type: {chart.root_state.type.name})")
    print(f"  Root has {len(chart.root_state.children)} parallel subsystems:")
    
    for i, subsystem in enumerate(chart.root_state.children, 1):
        print(f"    {i}. {subsystem.label} (type: {subsystem.type.name})")
        for j, child in enumerate(subsystem.children, 1):
            initial = " (initial)" if child.is_initial else ""
            print(f"       {j}.{j} {child.label}{initial}")
    
    # Show that all subsystems are active in a parallel state
    print(f"\nIn a parallel/orthogonal state, ALL child states are conceptually active:")
    print(f"  - When the smartphone is on, all 4 subsystems (Display, Audio, Network, Battery) are running")
    print(f"  - Each subsystem maintains its own current state independently")
    print(f"  - Events can affect one or more subsystems simultaneously")
    
    # Show independent transitions
    print(f"\nIndependent subsystem transitions:")
    transitions_by_subsystem = {}
    
    for trans in chart.transitions:
        # Determine which subsystem this transition affects
        affected_subsystem = None
        for from_state in trans.from_states:
            for subsystem in chart.root_state.children:
                if subsystem.find_state(from_state):
                    affected_subsystem = subsystem.label
                    break
        
        if affected_subsystem:
            if affected_subsystem not in transitions_by_subsystem:
                transitions_by_subsystem[affected_subsystem] = []
            transitions_by_subsystem[affected_subsystem].append(trans)
    
    for subsystem_name in sorted(transitions_by_subsystem.keys()):
        print(f"\n  {subsystem_name} subsystem:")
        for trans in transitions_by_subsystem[subsystem_name]:
            from_states = ", ".join(trans.from_states)
            to_states = ", ".join(trans.to_states)
            print(f"    - {trans.event}: {from_states} → {to_states}")
    
    print("\n" + "="*50)


def demonstrate_configuration_combinations():
    """Demonstrate the combinatorial nature of orthogonal states."""
    print("=== Configuration Combinations ===\n")
    
    chart = create_car_entertainment_statechart()
    
    print("Car Entertainment System - Possible Configurations:")
    print("Since this is an orthogonal statechart, the total number of possible")
    print("configurations is the product of states in each subsystem.\n")
    
    subsystem_states = {}
    for subsystem in chart.root_state.children:
        states = [child.label for child in subsystem.children]
        subsystem_states[subsystem.label] = states
        print(f"{subsystem.label}: {len(states)} states - {', '.join(states)}")
    
    # Calculate total combinations
    total_combinations = 1
    for states in subsystem_states.values():
        total_combinations *= len(states)
    
    print(f"\nTotal possible configurations: {total_combinations}")
    
    # Show some example configurations
    print(f"\nExample valid configurations:")
    configurations = [
        ["RadioOff", "Auto", "Idle"],
        ["AM", "Auto", "Navigating"],
        ["FM", "Manual", "Idle"],
        ["RadioOff", "Manual", "Navigating"],
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"  {i}. {', '.join(config)}")
    
    print("\n" + "="*50)


def demonstrate_parallel_vs_normal():
    """Demonstrate the difference between parallel and normal states."""
    print("=== Parallel vs Normal States ===\n")
    
    # Create two similar structures - one normal, one parallel
    child1 = basic_state("Child1", is_initial=True)
    child2 = basic_state("Child2")
    child3 = basic_state("Child3")
    
    normal_parent = normal_state("NormalParent", [child1, child2, child3])
    parallel_parent = parallel_state("ParallelParent", [child1, child2, child3])
    
    print("Comparison of state types:")
    print(f"\nNormal (XOR) State - {normal_parent.label}:")
    print(f"  - Type: {normal_parent.type.name}")
    print(f"  - Semantics: Exactly ONE child is active at any time")
    print(f"  - Initial configuration: [Child1] (only the initial child)")
    print(f"  - Transitions move between children (Child1 → Child2 → Child3)")
    
    print(f"\nParallel (AND) State - {parallel_parent.label}:")
    print(f"  - Type: {parallel_parent.type.name}")
    print(f"  - Semantics: ALL children are active simultaneously")
    print(f"  - Initial configuration: [Child1, Child2, Child3] (all children)")
    print(f"  - Each child can transition independently")
    
    print(f"\nKey Differences:")
    print(f"  1. Active states: Normal = 1, Parallel = ALL")
    print(f"  2. Transitions: Normal = mutually exclusive, Parallel = independent")
    print(f"  3. Complexity: Normal = linear, Parallel = combinatorial")
    print(f"  4. Use cases: Normal = sequential logic, Parallel = concurrent systems")
    
    print("\n" + "="*50)


def main():
    """Run the orthogonal statechart examples."""
    demonstrate_orthogonal_behavior()
    demonstrate_configuration_combinations()
    demonstrate_parallel_vs_normal()


if __name__ == "__main__":
    main()