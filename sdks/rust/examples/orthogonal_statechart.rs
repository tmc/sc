use statecharts::factory::*;
use statecharts::v1::{StateType};

// This example demonstrates creating an orthogonal statechart in Rust, similar to
// the orthogonal_statechart.go example in the main repo
fn main() {
    println!("Creating an orthogonal statechart example (media player)");
    
    // First, create the states for the PlaybackState region
    let playing = basic_state("Playing", false);
    let paused = basic_state("Paused", true);  // initial state
    let stopped = basic_state("Stopped", false);
    
    // Create the PlaybackState region
    let playback_state = normal_state("PlaybackState", false, vec![playing, paused, stopped]);
    
    // Create the states for the VolumeControl region
    let normal_vol = basic_state("Normal", true);  // initial state
    let muted = basic_state("Muted", false);
    
    // Create the VolumeControl region
    let volume_control = normal_state("VolumeControl", false, vec![normal_vol, muted]);
    
    // Create the PlaybackControl orthogonal state containing both regions
    let playback_control = parallel_state(
        "PlaybackControl", 
        true, 
        vec![playback_state, volume_control]
    );
    
    // Create the root MediaPlayer state
    let media_player = normal_state("MediaPlayer", true, vec![playback_control]);
    
    // Create the transitions
    let transitions = vec![
        // Playback state transitions
        transition("Play", vec!["Paused"], vec!["Playing"], "PLAY"),
        transition("Pause", vec!["Playing"], vec!["Paused"], "PAUSE"),
        transition("Stop", vec!["Playing", "Paused"], vec!["Stopped"], "STOP"),
        transition("Resume", vec!["Stopped"], vec!["Playing"], "PLAY"),
        
        // Volume control transitions
        transition("Mute", vec!["Normal"], vec!["Muted"], "MUTE"),
        transition("Unmute", vec!["Muted"], vec!["Normal"], "UNMUTE"),
    ];
    
    // Create the events
    let events = vec![
        event("PLAY"),
        event("PAUSE"),
        event("STOP"),
        event("MUTE"),
        event("UNMUTE"),
    ];
    
    // Create the statechart
    let mut statechart = statechart(media_player);
    
    // Add transitions and events
    statechart.transitions = transitions;
    statechart.events = events;
    
    // Print statechart information
    println!("\nStatechart structure:");
    print_state_hierarchy(statechart.root_state.as_ref().unwrap(), 0);
    
    println!("\nTransitions:");
    for (i, transition) in statechart.transitions.iter().enumerate() {
        println!("{}. {} (Event: {}): {:?} → {:?}", 
            i+1, 
            transition.label, 
            transition.event,
            transition.from,
            transition.to
        );
    }
    
    println!("\nThis statechart demonstrates orthogonal regions where:");
    println!("- PlaybackState and VolumeControl operate independently");
    println!("- When PlaybackControl is active, both regions are active simultaneously");
    println!("- Each region follows its own transitions based on events");
}

fn print_state_hierarchy(state: &statecharts::v1::State, indent: usize) {
    let indent_str = " ".repeat(indent * 2);
    let state_type = match state.r#type {
        x if x == StateType::StateTypeBasic as i32 => "Basic",
        x if x == StateType::StateTypeNormal as i32 => "Normal (OR)",
        x if x == StateType::StateTypeParallel as i32 => "Parallel/Orthogonal (AND)",
        _ => "Unknown",
    };
    
    let initial_marker = if state.is_initial { "★ " } else { "" };
    
    println!("{}{}State: {} (Type: {})", 
             indent_str, initial_marker, state.label, state_type);
    
    for child in &state.children {
        print_state_hierarchy(child, indent + 1);
    }
}