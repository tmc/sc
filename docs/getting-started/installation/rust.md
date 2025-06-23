# Rust Installation Guide

This guide will help you install and set up the Statecharts library for Rust development.

## Prerequisites

- **Rust 1.60 or later** - [Install Rust](https://rustup.rs/)
- **Protocol Buffers compiler** (for development) - [Install protoc](https://grpc.io/docs/protoc-installation/)

## Quick Installation

### Option 1: Git Dependency (Current)

Add to your `Cargo.toml`:

```toml
[dependencies]
statecharts = { git = "https://github.com/tmc/sc", subdirectory = "sdks/rust" }
```

### Option 2: Local Development

Clone and reference locally:

```bash
git clone https://github.com/tmc/sc.git
```

Then in your `Cargo.toml`:

```toml
[dependencies]
statecharts = { path = "../path/to/sc/sdks/rust" }
```

## Verify Installation

Create a simple test to verify everything works:

**src/main.rs**
```rust
use statecharts::factory::*;
use statecharts::v1::StateType;

fn main() {
    // Create a simple statechart using factory functions
    let off_state = basic_state("off", true);
    let on_state = basic_state("on", false);
    
    let root = normal_state("__root__", true, vec![off_state, on_state]);
    
    // Create transitions
    let power_on = transition("PowerOn", vec!["off"], vec!["on"], "POWER_ON");
    let power_off = transition("PowerOff", vec!["on"], vec!["off"], "POWER_OFF");
    
    // Create events
    let power_on_event = event("POWER_ON");
    let power_off_event = event("POWER_OFF");
    
    // Create the complete statechart
    let mut statechart = statechart(root);
    statechart.transitions.extend(vec![power_on, power_off]);
    statechart.events.extend(vec![power_on_event, power_off_event]);
    
    println!("Created statechart with {} child states", 
             statechart.root_state.as_ref().unwrap().children.len());
    println!("Transitions: {}", statechart.transitions.len());
    println!("Events: {}", statechart.events.len());
}
```

Run the test:

```bash
cargo run
```

Expected output:
```
Created statechart with 2 child states
Transitions: 2
Events: 2
```

## Crate Structure

The Rust SDK provides several modules:

```rust
use statecharts::{
    v1::*,           // Protocol Buffer generated types
    factory::*,      // Factory functions for easy creation
};
```

### Core Types

```rust
use statecharts::v1::{
    Statechart,      // Main statechart definition
    State,           // State definitions
    StateType,       // State type enumeration
    Transition,      // Transition definitions
    Event,           // Event definitions
    Configuration,   // Active state configuration
    Machine,         // Runtime machine instance
};
```

### Factory Functions

The factory module provides convenient constructors:

```rust
use statecharts::factory::{
    statechart,      // Create statechart
    basic_state,     // Create basic state
    normal_state,    // Create compound state
    parallel_state,  // Create orthogonal state
    transition,      // Create transition
    event,           // Create event
};
```

## Development Setup

If you plan to contribute or work with the source code:

### 1. Clone the Repository

```bash
git clone https://github.com/tmc/sc.git
cd sc/sdks/rust
```

### 2. Install Development Tools

```bash
# Install Protocol Buffer compiler
# macOS
brew install protobuf

# Linux (Ubuntu/Debian)  
sudo apt-get install protobuf-compiler

# Windows
# Download from https://github.com/protocolbuffers/protobuf/releases

# Install Rust tools
rustup component add rustfmt clippy
```

### 3. Build the SDK

```bash
cargo build
```

### 4. Run Tests

```bash
cargo test
```

### 5. Run Examples

```bash
cargo run --example simple_statechart
cargo run --example orthogonal_statechart
```

## IDE Setup

### VS Code

Recommended extensions:
- **rust-analyzer** - Rust language server
- **CodeLLDB** - Debugging support
- **Protocol Buffer** - Syntax highlighting for .proto files
- **Statecharts** - (if available) Visual editing support

Settings for optimal experience:

**.vscode/settings.json**
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

### CLion/IntelliJ Rust

Built-in Rust support with:
- IntelliJ Rust plugin
- Protocol Buffer plugin
- Toml plugin

## Feature Flags

The Rust SDK supports optional features:

```toml
[dependencies]
statecharts = { 
    git = "https://github.com/tmc/sc", 
    subdirectory = "sdks/rust",
    features = ["serde", "validation"]
}
```

Available features:
- **`serde`** - Serialization support
- **`validation`** - Built-in validation
- **`async`** - Async runtime support (future)

## Examples

The SDK includes several examples:

### Simple Statechart

```rust
use statecharts::factory::*;

fn main() {
    let idle = basic_state("Idle", true);
    let active = basic_state("Active", false);
    let root = normal_state("Machine", true, vec![idle, active]);
    
    let start = transition("Start", vec!["Idle"], vec!["Active"], "START");
    let stop = transition("Stop", vec!["Active"], vec!["Idle"], "STOP");
    
    let mut chart = statechart(root);
    chart.transitions.extend(vec![start, stop]);
    
    println!("Simple statechart created!");
}
```

### Hierarchical States

```rust
use statecharts::factory::*;

fn main() {
    // Create nested states
    let monitoring = basic_state("Monitoring", true);
    let triggered = basic_state("Triggered", false);
    let armed = normal_state("Armed", false, vec![monitoring, triggered]);
    
    let idle = basic_state("Idle", true);
    let on = normal_state("On", false, vec![idle, armed]);
    let off = basic_state("Off", true);
    
    let root = normal_state("AlarmSystem", true, vec![off, on]);
    
    // Add transitions between hierarchical states
    let arm = transition("Arm", vec!["Idle"], vec!["Armed"], "ARM");
    let trigger = transition("Trigger", vec!["Monitoring"], vec!["Triggered"], "MOTION");
    
    let mut chart = statechart(root);
    chart.transitions.extend(vec![arm, trigger]);
}
```

### Orthogonal (Parallel) States

```rust
use statecharts::factory::*;

fn main() {
    // Create orthogonal regions
    let bright = basic_state("Bright", true);
    let dim = basic_state("Dim", false);
    let display = normal_state("Display", true, vec![bright, dim]);
    
    let connected = basic_state("Connected", true);
    let disconnected = basic_state("Disconnected", false);
    let network = normal_state("Network", true, vec![connected, disconnected]);
    
    // Combine in parallel state
    let device = parallel_state("SmartWatch", true, vec![display, network]);
    let root = normal_state("__root__", true, vec![device]);
    
    let chart = statechart(root);
    println!("Orthogonal statechart created!");
}
```

## Common Issues

### Issue: "cannot find crate"

**Solution**: Make sure you're using the correct Git URL and subdirectory:

```toml
[dependencies]
statecharts = { git = "https://github.com/tmc/sc", subdirectory = "sdks/rust" }
```

### Issue: Protocol Buffer compilation errors

**Solution**: Ensure protoc is installed and accessible:

```bash
protoc --version
# Should show version 3.15+ or later
```

If using a development build, regenerate:

```bash
cd proto
make generate-rust
```

### Issue: Factory function not found

**Solution**: Import the factory module:

```rust
use statecharts::factory::*;
```

### Issue: Type conversion errors

**Solution**: The generated types use i32 for enums. Use proper conversion:

```rust
use statecharts::v1::StateType;

let state_type = StateType::StateTypeBasic as i32;
```

## Performance Considerations

### Memory Usage

- States and transitions are heap-allocated
- Use `Rc<State>` for shared state references
- Consider arena allocation for large statecharts

### Build Time

- Protocol Buffer compilation can be slow
- Use `--release` for production builds
- Consider pre-compiled artifacts for CI

### Runtime Performance

- Factory functions allocate new objects
- Cache compiled statecharts when possible
- Use profiling tools (`cargo flamegraph`) for analysis

## Testing

The SDK includes comprehensive tests:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_basic_state

# Run with features
cargo test --features serde
```

## Next Steps

Now that you have Rust set up:

1. **[Build your first statechart](../first-statechart.md)** - Hands-on tutorial
2. **[Learn basic concepts](../basic-concepts.md)** - Understand the fundamentals
3. **[Explore examples](../../examples/simple/)** - See practical applications
4. **[Read API documentation](../../api/rust/)** - Detailed reference

## Documentation

Generate local documentation:

```bash
cargo doc --open
```

This will build and open the API documentation in your browser.

---

Ready to build your first statechart? → [First Statechart Tutorial](../first-statechart.md)