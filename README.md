# Statecharts: A Formal Model for Reactive Systems

[![Go Reference](https://pkg.go.dev/badge/github.com/tmc/sc.svg)](https://pkg.go.dev/github.com/tmc/sc)

## Abstract

This repository presents an implementation of the Statecharts formalism, originally introduced by David Harel (1987). Statecharts provide a visual language for the specification and design of complex reactive systems, extending conventional state-transition diagrams with well-defined semantics for hierarchy, concurrency, and communication.

The implementation provides a language-neutral formal model that supports the rigorous development, analysis, and execution of statechart-based systems. By offering standardized type definitions and operational semantics, this library facilitates the construction of provably correct reactive systems across different programming languages.

## Theoretical Foundation

Statecharts extend classical finite state machines with three key concepts:

1. **Hierarchy** - States can be nested within other states, creating a hierarchical structure that enables abstraction and refinement.
2. **Orthogonality** - System components can operate concurrently through orthogonal (parallel) states, allowing the decomposition of complex behaviors.
3. **Communication** - Events can trigger transitions and broadcast to other parts of the system, enabling coordination between components.

The formal semantics of Statecharts in this implementation follow the reconciled definitions presented in academic literature, particularly von der Beeck's comparison of statechart variants (1994), Harel and Naamad's operational semantics (1996), and Eshuis' "Reconciling statechart semantics" (2009).

## Features

- Formal type definitions for statecharts, states, events, transitions, and configurations
- Rigorous implementation of operational semantics for state transitions and event processing
- Precise handling of state configurations and hierarchical state relationships
- Validation rules ensuring well-formed statechart models
- Explicit execution models for the paper semantics:
  - fixpoint
  - Statemate
  - single-event Statemate
  - UML
- Extensible architecture supporting theoretical extensions and domain-specific adaptations

## Paper Semantics

The `semantics/v1` package now includes an explicit engine for the four
execution models compared in Eshuis' paper. Use `Statechart.Reactions(...)` or
`Statechart.ReactionsWithOptions(...)` to evaluate a chart under those
semantics and enumerate the possible reactions.

This is distinct from the package's older `MachineWrapper` and `EventProcessor`
helpers, which remain useful as runtime utilities but do not by themselves
encode the paper's full operational distinctions.

For the paper semantics engine, internally generated events are represented by
transition actions whose labels begin with `raise:`, `emit:`, or `send:`.

## Documentation

Comprehensive documentation is available in the [docs/statecharts/v1/statecharts.md](./docs/statecharts/v1/statecharts.md) file, providing a formal specification of the Statecharts model, its components, and their semantics.

## Formal Specification

The formal specification of the Statecharts model is defined using Protocol Buffers. The canonical definitions can be found in:

- [proto/statecharts/v1/statecharts.proto](./proto/statecharts/v1/statecharts.proto) - Core type definitions
- [proto/statecharts/v1/statechart_service.proto](./proto/statecharts/v1/statechart_service.proto) - Service interface definitions
- [proto/validation/v1/validator.proto](./proto/validation/v1/validator.proto) - Formal validation rules

## Usage

To utilize this Statecharts implementation in research or application development, clone the repository or include it as a dependency in your project. The library provides a foundation for formal verification, model checking, and execution of reactive system specifications.

### Go

The primary implementation is available in Go:

```go
import "github.com/tmc/sc"

// Create a statechart definition
statechart := &sc.Statechart{
    RootState: &sc.State{
        Label: "root",
        Children: []*sc.State{
            {
                Label:     "off",
                Type:      sc.StateTypeBasic,
                IsInitial: true,
            },
            {
                Label: "on",
                Type:  sc.StateTypeBasic,
            },
        },
    },
}
```

### Rust

A fully-featured Rust SDK is available in the `sdks/rust` directory:

```rust
use statecharts::factory::*;
use statecharts::v1::*;

// Create states
let off_state = basic_state("off", true);
let on_state = basic_state("on", false);

// Create root state with children
let root = normal_state("root", true, vec![off_state, on_state]);

// Add transition between states
let transition = transition("PowerOn", vec!["off"], vec!["on"], "POWER_ON");

// Create the statechart
let mut statechart = statechart(root);
statechart.transitions.push(transition);
```

The Rust SDK includes:
- Generated Protocol Buffer bindings for all statechart types
- Factory functions for easier statechart creation
- Support for hierarchical, orthogonal, and history states
- Examples demonstrating different statechart patterns

To generate and build the Rust SDK:

```bash
cd proto
make build-rust
```

To run examples:

```bash
cd sdks/rust
cargo run --example simple_statechart
cargo run --example orthogonal_statechart
```

## Contributing

Contributions to the theoretical foundation or implementation of Statecharts are welcomed. Please adhere to rigorous academic standards when proposing modifications or extensions to the model.

## Citations

When referencing this implementation in academic work, please cite:

```bibtex
@misc{tmc2023statecharts,
  author       = {TMC},
  title        = {Statecharts: A Formal Implementation of Harel Statecharts},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/tmc/sc}}
}
```

## References

- Harel, D. (1987). Statecharts: A visual formalism for complex systems. *Science of Computer Programming, 8(3)*, 231-274.
- von der Beeck, M. (1994). A comparison of statecharts variants. In *Formal Techniques in Real-Time and Fault-Tolerant Systems*, 128-148.
- Harel, D., & Naamad, A. (1996). The STATEMATE semantics of statecharts. *ACM Transactions on Software Engineering and Methodology, 5(4)*, 293-333.
- Harel, D., & Politi, M. (1998). *Modeling Reactive Systems with Statecharts: The STATEMATE Approach*. McGraw-Hill.

## License

This implementation of the Statecharts formalism is available under the [MIT License](LICENSE).
