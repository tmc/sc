# Integration Test Suite

This directory contains comprehensive end-to-end integration tests for the statechart library. These tests ensure that all components work together correctly and provide confidence in the library's behavior in real-world usage scenarios.

## Test Organization

### Core Integration Tests

- **`machine_lifecycle_test.go`** - Complete machine lifecycle integration
  - Tests machine creation, validation, startup, event processing, and shutdown
  - Tests context-modifying actions and error recovery
  - Validates cross-component integration

- **`validation_workflow_test.go`** - Validation → Semantics → Execution workflows
  - Tests validation service integration with semantic execution
  - Tests selective rule application and error propagation
  - Validates that validation catches errors before execution

- **`bridge_integration_test.go`** - Bridge conversion integration
  - Tests XState ↔ Harel format conversion workflows
  - Tests roundtrip conversion fidelity
  - Tests bridge error handling and recovery

- **`concurrent_operations_test.go`** - Concurrent machine operations
  - Tests thread safety of machine operations
  - Tests concurrent event processing
  - Tests race condition detection

- **`complex_patterns_test.go`** - Real-world statechart patterns
  - Tests complex business workflows (e-commerce, authentication, media player, etc.)
  - Tests hierarchical and orthogonal statechart patterns
  - Tests integration with example statecharts

- **`error_handling_test.go`** - Cross-package error handling
  - Tests error propagation across package boundaries
  - Tests error recovery mechanisms
  - Tests graceful degradation under error conditions

- **`performance_integration_test.go`** - Performance validation
  - Tests performance characteristics of integrated workflows
  - Tests scalability and resource usage
  - Includes benchmark tests and regression detection

## Running Tests

### Run All Integration Tests
```bash
go test ./integration/...
```

### Run Specific Test Categories
```bash
# Machine lifecycle tests
go test ./integration/ -run TestMachineLifecycle

# Validation workflow tests
go test ./integration/ -run TestValidationToExecution

# Bridge integration tests
go test ./integration/ -run TestBridgeToExecution

# Concurrent operations tests
go test ./integration/ -run TestConcurrentMachine

# Complex patterns tests
go test ./integration/ -run TestComplexStatechart

# Error handling tests
go test ./integration/ -run TestCrossPackageError

# Performance tests (skip in short mode)
go test ./integration/ -run TestPerformanceIntegration
```

### Run with Verbose Output
```bash
go test -v ./integration/...
```

### Run with Race Detection
```bash
go test -race ./integration/...
```

### Run Performance Tests
```bash
# Run all performance tests
go test ./integration/ -run TestPerformanceIntegration

# Run benchmarks
go test -bench=. ./integration/

# Run with memory profiling
go test -bench=. -memprofile=mem.prof ./integration/

# Skip performance tests (for faster CI)
go test -short ./integration/...
```

## Test Coverage Areas

### 1. Full Workflow Integration
- **Creation → Validation → Execution → Cleanup**
- Tests complete statechart lifecycle from creation to disposal
- Ensures all components integrate correctly at each phase

### 2. Cross-Package Integration
- **Validation ↔ Semantics ↔ Machine**
- Tests that package boundaries don't introduce errors
- Validates error propagation and recovery across packages

### 3. Bridge Conversion Workflows
- **External Format → Validation → Execution**
- Tests XState import/export with semantic execution
- Validates roundtrip conversion fidelity

### 4. Concurrent Operations
- **Thread Safety and Race Conditions**
- Tests concurrent machine operations
- Tests concurrent event processing
- Validates thread safety under load

### 5. Complex Real-World Patterns
- **Business Workflows**
  - E-commerce checkout flow
  - User authentication system
  - Media player controls
  - Game state machine
  - Workflow engine

- **Statechart Patterns**
  - Hierarchical state composition
  - Orthogonal (parallel) regions
  - History states and transitions
  - Guard conditions and actions

### 6. Error Handling and Recovery
- **Error Boundaries**
- Tests error containment and isolation
- Tests graceful degradation
- Validates error recovery mechanisms

### 7. Performance and Scalability
- **Performance Characteristics**
- Tests validation performance across statechart sizes
- Tests machine creation and event processing throughput
- Tests memory usage and garbage collection
- Tests concurrent load handling

## Test Data and Fixtures

The tests use a combination of:

1. **Built-in Test Utilities** (`github.com/tmc/sc/testing`)
   - Statechart builders and helpers
   - Common test statechart patterns

2. **Example Statecharts** (`github.com/tmc/sc/semantics/v1/examples`)
   - Academic examples (hierarchical, orthogonal, compound)
   - Real-world examples (Hotel Evanstonian)

3. **Custom Test Statecharts**
   - Business workflow patterns
   - Error condition scenarios
   - Performance test patterns

## Performance Baselines

The performance tests establish baselines for:

- **Validation Time**: < 10ms for simple charts, < 200ms for large charts
- **Machine Creation**: < 1ms for simple charts, < 3ms for complex charts
- **Event Processing**: > 10,000 events/sec for simple transitions
- **Memory Usage**: < 10KB per machine instance
- **Concurrent Load**: > 5,000 events/sec across multiple machines

## Continuous Integration

These tests are designed to run in CI environments:

- **Fast Tests**: Core integration tests run in < 30 seconds
- **Performance Tests**: Skipped in short mode (`-short` flag)
- **Race Detection**: Run with `-race` flag in CI
- **Memory Validation**: Tests include memory leak detection

## Test Maintenance

### Adding New Integration Tests

1. **Identify Integration Points**: Focus on cross-package interactions
2. **Follow Naming Conventions**: Use descriptive test names
3. **Include Error Cases**: Test both success and failure scenarios
4. **Add Performance Validation**: Include timing assertions where appropriate
5. **Update Documentation**: Add new test descriptions to this README

### Performance Test Guidelines

1. **Use Realistic Scenarios**: Base tests on real-world usage patterns
2. **Establish Baselines**: Set reasonable performance expectations
3. **Test Scalability**: Include tests that vary input size
4. **Monitor Regressions**: Use benchmark tests to catch performance regressions

### Error Handling Guidelines

1. **Test Error Propagation**: Ensure errors flow correctly between components
2. **Test Recovery**: Verify systems can recover from error conditions
3. **Test Isolation**: Ensure errors in one component don't corrupt others
4. **Test Graceful Degradation**: Verify partial failure handling

## Debugging Integration Issues

### Common Integration Problems

1. **State Synchronization**: Machine state becomes inconsistent
2. **Memory Leaks**: Resources not properly cleaned up
3. **Race Conditions**: Concurrent access issues
4. **Error Propagation**: Errors not handled across package boundaries

### Debugging Tools

```bash
# Run with race detection
go test -race ./integration/

# Run with memory profiling
go test -memprofile=mem.prof ./integration/

# Run with CPU profiling
go test -cpuprofile=cpu.prof ./integration/

# Run with trace
go test -trace=trace.out ./integration/

# Analyze profiles
go tool pprof mem.prof
go tool pprof cpu.prof
go tool trace trace.out
```

### Test Environment Variables

Set these environment variables for additional debugging:

```bash
# Enable verbose logging
export SC_DEBUG=1

# Enable trace logging
export SC_TRACE=1

# Disable performance tests
export SC_SKIP_PERF=1
```

## Contributing

When contributing to the integration test suite:

1. **Focus on Integration**: Test cross-component interactions, not individual units
2. **Include Documentation**: Update this README with new test descriptions
3. **Follow Patterns**: Use existing test patterns and utilities
4. **Test Error Cases**: Include both positive and negative test scenarios
5. **Consider Performance**: Add timing assertions for critical paths
6. **Validate Concurrency**: Test thread safety where applicable

The integration test suite is a critical component for ensuring the reliability and performance of the statechart library across all its components and usage scenarios.