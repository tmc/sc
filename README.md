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

The formal semantics of Statecharts in this implementation follow the reconciled definitions presented in academic literature, particularly von der Beeck's comparison of statechart variants (1994) and Harel and Naamad's operational semantics (1996).

## Features

- Formal type definitions for statecharts, states, events, transitions, and configurations
- Rigorous implementation of operational semantics for state transitions and event processing
- Precise handling of state configurations and hierarchical state relationships
- Validation rules ensuring well-formed statechart models
- Extensible architecture supporting theoretical extensions and domain-specific adaptations

## Documentation

Comprehensive documentation is available in the [docs/statecharts/v1/statecharts.md](./docs/statecharts/v1/statecharts.md) file, providing a formal specification of the Statecharts model, its components, and their semantics.

## Formal Specification

The formal specification of the Statecharts model is defined using Protocol Buffers. The canonical definitions can be found in:

- [proto/statecharts/v1/statecharts.proto](./proto/statecharts/v1/statecharts.proto) - Core type definitions
- [proto/statecharts/v1/statechart_service.proto](./proto/statecharts/v1/statechart_service.proto) - Service interface definitions
- [proto/validation/v1/validator.proto](./proto/validation/v1/validator.proto) - Formal validation rules

## Usage

To utilize this Statecharts implementation in research or application development, clone the repository or include it as a dependency in your project. The library provides a foundation for formal verification, model checking, and execution of reactive system specifications.

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