# Statecharts Language SDKs

This directory contains language-specific SDKs for working with the Statecharts library. Each SDK provides a native interface to the Statechart protocol buffer definitions.

## Available SDKs

### Rust

The Rust SDK provides a type-safe interface for working with statecharts in Rust applications. It includes:

- Generated Protocol Buffer types (using `prost` and `tonic`)
- Factory functions for easier statechart creation
- Comprehensive examples

[Learn more about the Rust SDK](./rust/README.md)

## Building the SDKs

### Building the Rust SDK

```bash
# From the repository root
cd proto
make build-rust
```

## Adding New SDKs

When adding a new language SDK:

1. Create a subdirectory named after the language (e.g., `sdks/python`)
2. Add the necessary Protocol Buffer code generation configuration to the protobuf generation process
3. Implement a clean API that follows the language's idioms
4. Add tests and examples
5. Update this README to include the new SDK