use statecharts::factory::*;
use statecharts::v1::{StateType};

fn main() {
    // Create a simple hierarchical statechart similar to the one in the examples directory
    let monitoring = basic_state("Monitoring", true);
    let triggered = basic_state("Triggered", false);
    
    let armed = normal_state("Armed", false, vec![monitoring, triggered]);
    let idle = basic_state("Idle", true);
    
    let on = normal_state("On", false, vec![idle, armed]);
    let off = basic_state("Off", true);
    
    let root = normal_state("AlarmSystem", true, vec![off, on]);
    
    // Create the transitions
    let power_on = transition("PowerOn", vec!["Off"], vec!["On"], "POWER_ON");
    let power_off = transition("PowerOff", vec!["On"], vec!["Off"], "POWER_OFF");
    let arm = transition("Arm", vec!["Idle"], vec!["Armed"], "ARM");
    let disarm = transition("Disarm", vec!["Armed"], vec!["Idle"], "DISARM");
    let trigger = transition("Trigger", vec!["Monitoring"], vec!["Triggered"], "MOTION_DETECTED");
    let reset = transition("Reset", vec!["Triggered"], vec!["Monitoring"], "RESET");
    
    // Create the events
    let power_on_event = event("POWER_ON");
    let power_off_event = event("POWER_OFF");
    let arm_event = event("ARM");
    let disarm_event = event("DISARM");
    let motion_detected_event = event("MOTION_DETECTED");
    let reset_event = event("RESET");
    
    // Create the statechart
    let mut statechart = statechart(root);
    
    // Add transitions
    statechart.transitions.extend(vec![power_on, power_off, arm, disarm, trigger, reset]);
    
    // Add events
    statechart.events.extend(vec![
        power_on_event, power_off_event, arm_event, 
        disarm_event, motion_detected_event, reset_event
    ]);
    
    // Print the statechart structure
    println!("Created hierarchical statechart:");
    println!("Root state: {}", statechart.root_state.as_ref().unwrap().label);
    println!("Number of transitions: {}", statechart.transitions.len());
    println!("Number of events: {}", statechart.events.len());
    
    // Print the states hierarchy
    print_state_hierarchy(statechart.root_state.as_ref().unwrap(), 0);
}

fn print_state_hierarchy(state: &statecharts::v1::State, indent: usize) {
    let indent_str = " ".repeat(indent * 2);
    let state_type = match state.r#type {
        x if x == StateType::StateTypeBasic as i32 => "Basic",
        x if x == StateType::StateTypeNormal as i32 => "Normal (OR)",
        x if x == StateType::StateTypeParallel as i32 => "Parallel/Orthogonal (AND)",
        _ => "Unknown",
    };
    
    println!("{}State: {} (Type: {}, Initial: {})", 
             indent_str, state.label, state_type, state.is_initial);
    
    for child in &state.children {
        print_state_hierarchy(child, indent + 1);
    }
}