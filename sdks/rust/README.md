# Statecharts Rust SDK

This directory contains the Rust SDK for working with statecharts. The SDK is generated from Protocol Buffer definitions and provides a type-safe way to interact with statecharts in Rust applications.

## Features

- Type-safe representations of statecharts
- Generated from Protocol Buffer definitions
- Compatibility with the statechart semantics from the main library

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
statecharts = { git = "https://github.com/tmc/sc", version = "0.1.0" }
```

Basic example:

```rust
use statecharts::v1::{Statechart, State, StateType, Transition, Event};

fn main() {
    // Create a simple statechart
    let mut statechart = Statechart::new();
    
    // Set up the root state
    let mut root_state = State::new();
    root_state.set_label("root".to_string());
    
    // Add a basic child state
    let mut child_state = State::new();
    child_state.set_label("child".to_string());
    child_state.set_type(StateType::StateTypeBasic);
    child_state.set_is_initial(true);
    
    // Add the child to the root
    root_state.mut_children().push(child_state);
    
    // Set the root state on the statechart
    statechart.set_root_state(root_state);
    
    println!("Created statechart: {:?}", statechart);
}
```

## Building from Source

To generate the Rust SDK from Protocol Buffer definitions:

```bash
cd proto
make generate-rust
```

## Dependencies

- [prost](https://github.com/tokio-rs/prost) - Protocol Buffers implementation for Rust
- [tonic](https://github.com/hyperium/tonic) - gRPC implementation for Rust