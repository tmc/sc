# Getting Started with Statecharts

Welcome to the Statecharts library! This guide will help you understand statecharts and get started with building reactive systems using formal statechart semantics.

## 📚 What Are Statecharts?

Statecharts are a visual formalism for describing the behavior of reactive systems. Originally introduced by David Harel in 1987, statecharts extend traditional finite state machines with:

- **Hierarchy**: States can contain other states, enabling abstraction and refinement
- **Concurrency**: Multiple orthogonal regions can operate simultaneously 
- **Communication**: Events can trigger transitions and coordinate between components

## 🚀 Quick Start

### 1. Choose Your Language

We provide SDKs for multiple programming languages:

- **[Go](installation/go.md)** - Primary implementation with full semantics engine
- **[Rust](installation/rust.md)** - High-performance SDK with factory functions
- **[Python](installation/python.md)** - Pythonic API with web framework integration
- **[JavaScript](installation/javascript.md)** - Browser and Node.js support

### 2. Install the Library

Follow the installation guide for your chosen language:

```bash
# Go
go get github.com/tmc/sc

# Rust  
cargo add statecharts

# Python
pip install statecharts

# JavaScript/TypeScript
npm install statecharts
```

### 3. Create Your First Statechart

Here's a simple example in each language:

#### Go
```go
package main

import (
    "github.com/tmc/sc"
    "github.com/tmc/sc/semantics/v1"
)

func main() {
    // Create a simple on/off statechart
    statechart := &sc.Statechart{
        RootState: &sc.State{
            Label: "__root__",
            Type:  sc.StateTypeNormal,
            Children: []*sc.State{
                {Label: "Off", Type: sc.StateTypeBasic, IsInitial: true},
                {Label: "On", Type: sc.StateTypeBasic},
            },
        },
        Transitions: []*sc.Transition{
            {Label: "PowerOn", From: []string{"Off"}, To: []string{"On"}, Event: "POWER_ON"},
            {Label: "PowerOff", From: []string{"On"}, To: []string{"Off"}, Event: "POWER_OFF"},
        },
        Events: []*sc.Event{
            {Label: "POWER_ON"},
            {Label: "POWER_OFF"},
        },
    }
    
    // Create and run a machine
    wrapper := semantics.NewStatechart(statechart)
    machine, _ := semantics.NewMachine(wrapper, "my-machine", nil)
    
    machine.Start()
    machine.Step("POWER_ON")  // Off -> On
    machine.Step("POWER_OFF") // On -> Off
}
```

#### Rust
```rust
use statecharts::factory::*;

fn main() {
    // Create states
    let off_state = basic_state("Off", true);
    let on_state = basic_state("On", false);
    
    // Create root state
    let root = normal_state("__root__", true, vec![off_state, on_state]);
    
    // Create transitions
    let power_on = transition("PowerOn", vec!["Off"], vec!["On"], "POWER_ON");
    let power_off = transition("PowerOff", vec!["On"], vec!["Off"], "POWER_OFF");
    
    // Build statechart
    let mut statechart = statechart(root);
    statechart.transitions.extend(vec![power_on, power_off]);
    statechart.events.extend(vec![event("POWER_ON"), event("POWER_OFF")]);
    
    println!("Statechart created with {} states", 
             count_states(statechart.root_state.as_ref().unwrap()));
}
```

#### Python
```python
from statecharts import StatechartBuilder, basic_state, normal_state

# Using the builder pattern
builder = StatechartBuilder()
statechart = (builder
    .root_state("__root__")
    .add_child(basic_state("Off", is_initial=True))
    .add_child(basic_state("On"))
    .add_transition("PowerOn", ["Off"], ["On"], "POWER_ON")
    .add_transition("PowerOff", ["On"], ["Off"], "POWER_OFF")
    .add_event("POWER_ON")
    .add_event("POWER_OFF")
    .build())

print(f"Created statechart with {len(statechart.get_all_states())} states")
```

## 📖 Learning Path

### Beginner
1. **[What Are Statecharts?](what-are-statecharts.md)** - Understand the core concepts
2. **[Your First Statechart](first-statechart.md)** - Build a simple state machine
3. **[Basic Concepts](basic-concepts.md)** - Learn states, transitions, and events

### Intermediate  
4. **[Hierarchical States](../tutorials/intermediate/hierarchical-states.md)** - Nested state structures
5. **[Parallel States](../tutorials/intermediate/parallel-states.md)** - Concurrent execution
6. **[Guards and Actions](../tutorials/intermediate/guards-and-actions.md)** - Conditional behavior

### Advanced
7. **[Complex Patterns](../tutorials/advanced/complex-patterns.md)** - Sophisticated designs
8. **[Performance Optimization](../tutorials/advanced/performance-optimization.md)** - Scale your statecharts
9. **[Custom Semantics](../tutorials/advanced/custom-semantics.md)** - Extend the framework

## 🛠️ Tools and Ecosystem

- **Visual Editor**: Web-based statechart designer
- **CLI Tools**: Code generation and validation utilities  
- **Framework Integrations**: React, Vue, Spring Boot, Django, and more
- **Debugging Tools**: Interactive debugger and state visualization

## 📚 Core Concepts

### States
States represent the various conditions or situations your system can be in:

- **Basic States**: Atomic states with no sub-states
- **Compound States**: States containing other states
- **Parallel States**: States with concurrent sub-regions

### Transitions
Transitions define how your system moves between states:

```
Source State --[Event / Guard]-> Target State / Actions
```

### Events
Events trigger transitions and drive your statechart's behavior:

- **External Events**: Triggered by user actions or system inputs
- **Internal Events**: Generated automatically by the statechart
- **Timeout Events**: Time-based triggers

### Configuration
A configuration represents the active states at any point in time. Valid configurations must satisfy:

- Exactly one child is active in each OR-state
- All children are active in each AND-state  
- If a state is active, its parent must be active

## 🎯 Example Use Cases

Statecharts excel in many domains:

- **User Interfaces**: Complex UI workflows and navigation
- **Game Development**: Game states, character behavior, AI
- **Protocol Implementation**: Network protocols, communication
- **Workflow Management**: Business processes, approval flows
- **Device Control**: Embedded systems, IoT devices
- **Process Automation**: Manufacturing, robotics

## 🤝 Getting Help

- **[Documentation](../README.md)**: Comprehensive guides and references
- **[Examples](../examples/README.md)**: Real-world implementations
- **[Community](../community/README.md)**: Forums, chat, and support
- **[Issues](https://github.com/tmc/sc/issues)**: Bug reports and feature requests

## ➡️ Next Steps

Ready to dive deeper? Choose your path:

- **[Installation Guide](installation/)** - Set up your development environment
- **[First Statechart](first-statechart.md)** - Build your first working example
- **[Tutorials](../tutorials/)** - Structured learning modules
- **[Examples](../examples/)** - Real-world implementations
- **[API Reference](../api/)** - Detailed technical documentation

---

*Welcome to the world of formal reactive systems! Statecharts will help you build more predictable, maintainable, and robust applications.*