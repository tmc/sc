# Statecharts

[![Go Reference](https://pkg.go.dev/badge/github.com/tmc/sc.svg)](https://pkg.go.dev/github.com/tmc/sc)

Statecharts is a language-neutral library that provides a standardized set of types and tools to support the development, testing, and execution of statechart-based machines. It aims to simplify the process of creating statechart-based systems across different programming languages, making it easier for developers to understand, implement, and maintain statechart-driven applications.

## Features

- Language-neutral type definitions for statecharts, states, events, transitions, and more.
- Support for managing statechart instances and their configurations.
- Standardized APIs for creating and interacting with statechart-based machines.
- Extensible design allowing for the integration of custom extensions or modifications.

## Documentation

Detailed documentation is available in the [docs/statecharts/v1/statecharts.md](./docs/statecharts/v1/statecharts.md) file, providing a comprehensive overview of Statecharts, its components, and usage guidelines.

## Protocol Buffer Definitions

The core type definitions for Statecharts are defined using Protocol Buffers. You can find the `.proto` files in the following locations:

- [proto/statecharts/v1/statecharts.proto](./proto/statecharts/v1/statecharts.proto) for statechart type definitions.
- [proto/statecharts/v1/statechart_service.proto](./proto/statecharts/v1/statechart_service.proto) for statechart service definitions.

## Getting Started

To start using Statecharts, clone the repository or integrate it as a dependency in your project. Explore the provided documentation and type definitions to gain a deeper understanding of the library's capabilities and how to utilize it in your application.

## Contributing

We welcome contributions to Statecharts! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

Statecharts is released under the [MIT License](LICENSE).