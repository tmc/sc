# Statecharts Rust SDK Usage Guide

The Statecharts Rust SDK provides idiomatic Rust bindings for working with statecharts, representing hierarchical state machines based on Harel's Statecharts formalism.

## Installation

Add the Statecharts SDK to your `Cargo.toml`:

```toml
[dependencies]
statecharts = { git = "https://github.com/tmc/sc", version = "0.1.0" }
```

## Basic Usage

```rust
use statecharts::v1::{Statechart, State, StateType};
use statecharts::factory::*; // Import factory functions for easier creation

fn main() {
    // Create states
    let off_state = basic_state("Off", true);
    let on_state = basic_state("On", false);
    
    // Create root state with children
    let root = normal_state("Root", true, vec![off_state, on_state]);
    
    // Create a transition
    let power_on = transition("PowerOn", vec!["Off"], vec!["On"], "POWER_ON");
    
    // Create the statechart
    let mut statechart = statechart(root);
    statechart.transitions.push(power_on);
    statechart.events.push(event("POWER_ON"));
    
    println!("Created statechart with root: {}", 
             statechart.root_state.as_ref().unwrap().label);
}
```

## Factory Functions

The SDK provides factory functions to make statechart creation easier:

```rust
// Create states
let basic = basic_state("Basic", true); // Basic state (leaf state)
let normal = normal_state("Normal", false, vec![child1, child2]); // OR-state (XOR semantics)
let parallel = parallel_state("Parallel", true, vec![region1, region2]); // AND-state (parallel regions)

// Create transitions
let trans = transition("MyTransition", vec!["Source"], vec!["Target"], "EVENT");

// Create events
let evt = event("MY_EVENT");

// Create a complete statechart
let sc = statechart(root_state);
```

## Working with Orthogonal Regions

Orthogonal (parallel) regions allow multiple states to be active simultaneously:

```rust
// Create first region with states
let region1_states = vec![
    basic_state("R1_State1", true),
    basic_state("R1_State2", false),
];
let region1 = normal_state("Region1", false, region1_states);

// Create second region with states
let region2_states = vec![
    basic_state("R2_State1", true),
    basic_state("R2_State2", false),
];
let region2 = normal_state("Region2", false, region2_states);

// Create parallel parent state containing both regions
let parallel_state = parallel_state("Parallel", false, vec![region1, region2]);

// When the parallel state is active, both regions are active concurrently
```

## Examples

See the `examples` directory for complete examples:

- `simple_statechart.rs`: A basic hierarchical statechart
- `orthogonal_statechart.rs`: A statechart with orthogonal regions

Run examples with:

```bash
cargo run --example simple_statechart
cargo run --example orthogonal_statechart
```

## Aliases for Academic Terminology

The SDK supports the academic term "orthogonal" as an alias for "parallel" states:

```rust
// These are equivalent:
state.set_type(StateType::StateTypeParallel);
state.set_type(StateType::StateTypeOrthogonal);
```