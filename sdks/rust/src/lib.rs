// Include the generated Protocol Buffer code
pub mod generated {
    // Re-export the generated modules
    pub mod statecharts {
        pub mod v1 {
            include!("generated/statecharts.v1.rs");
            
            // Re-export core types for convenience
            pub use self::statechart::*;
            pub use self::state::*;
            pub use self::transition::*;
            pub use self::event::*;
            pub use self::guard::*;
            pub use self::action::*;
            pub use self::state_type::*;
            pub use self::machine_state::*;
        }
    }
    
    pub mod validation {
        pub mod v1 {
            include!("generated/validation.v1.rs");
        }
    }
}

// Re-export key modules for easier access
pub mod v1 {
    pub use crate::generated::statecharts::v1::*;
}

pub mod validation {
    pub use crate::generated::validation::v1::*;
}

// Factory functions
pub mod factory {
    use crate::v1::{State, StateType, Statechart, Transition, Event};

    /// Create a new basic state
    pub fn basic_state(label: &str, is_initial: bool) -> State {
        let mut state = State::default();
        state.label = label.to_string();
        state.r#type = StateType::StateTypeBasic as i32;
        state.is_initial = is_initial;
        state
    }

    /// Create a new normal (OR) state
    pub fn normal_state(label: &str, is_initial: bool, children: Vec<State>) -> State {
        let mut state = State::default();
        state.label = label.to_string();
        state.r#type = StateType::StateTypeNormal as i32;
        state.is_initial = is_initial;
        state.children = children;
        state
    }

    /// Create a new parallel/orthogonal (AND) state
    pub fn parallel_state(label: &str, is_initial: bool, children: Vec<State>) -> State {
        let mut state = State::default();
        state.label = label.to_string();
        state.r#type = StateType::StateTypeParallel as i32;
        state.is_initial = is_initial;
        state.children = children;
        state
    }

    /// Create a transition between states
    pub fn transition(label: &str, from: Vec<&str>, to: Vec<&str>, event_name: &str) -> Transition {
        let mut transition = Transition::default();
        transition.label = label.to_string();
        transition.from = from.into_iter().map(|s| s.to_string()).collect();
        transition.to = to.into_iter().map(|s| s.to_string()).collect();
        transition.event = event_name.to_string();
        transition
    }

    /// Create a new event
    pub fn event(label: &str) -> Event {
        let mut event = Event::default();
        event.label = label.to_string();
        event
    }

    /// Create a new statechart with a given root state
    pub fn statechart(root_state: State) -> Statechart {
        let mut statechart = Statechart::default();
        statechart.root_state = Some(root_state);
        statechart
    }
}

/// Example usage of the statecharts library
#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::*;

    #[test]
    fn test_create_simple_statechart() {
        // Create a simple hierarchical statechart
        let child1 = basic_state("Off", true);
        let child2 = basic_state("On", false);
        
        let root = normal_state("AlarmSystem", true, vec![child1, child2]);
        
        let transition = transition("PowerOn", vec!["Off"], vec!["On"], "POWER_ON");
        
        let mut statechart = statechart(root);
        statechart.transitions.push(transition);
        statechart.events.push(event("POWER_ON"));
        
        // Verify the structure
        assert!(statechart.root_state.is_some());
        let root = statechart.root_state.as_ref().unwrap();
        assert_eq!(root.label, "AlarmSystem");
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].label, "Off");
        assert_eq!(root.children[1].label, "On");
        assert_eq!(statechart.transitions.len(), 1);
        assert_eq!(statechart.events.len(), 1);
    }
}