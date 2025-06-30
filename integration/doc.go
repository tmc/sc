// Package integration provides comprehensive end-to-end integration tests for the statechart library.
// These tests exercise complete workflows across multiple packages and ensure proper integration
// between validation, semantics, bridges, and machine components.
//
// The integration tests cover:
//   - Full machine lifecycle workflows (creation → validation → execution → cleanup)
//   - Cross-package integration (validation + semantics + execution)
//   - Bridge conversion workflows (external format → validation → execution)
//   - Concurrent machine operations and thread safety
//   - Complex statechart patterns (hierarchical, orthogonal, history states)
//   - Error propagation across package boundaries
//   - Performance validation for realistic scenarios
//
// Test Organization:
//   - machine_lifecycle_test.go: Complete machine lifecycle integration
//   - validation_workflow_test.go: Validation → Semantics → Execution workflows
//   - bridge_integration_test.go: Bridge conversion integration
//   - concurrent_operations_test.go: Concurrent machine operations
//   - complex_patterns_test.go: Real-world statechart patterns
//   - error_handling_test.go: Cross-package error handling
//   - performance_integration_test.go: Performance validation
//
// These tests ensure that all components work together correctly and provide
// confidence in the library's behavior in real-world usage scenarios.
package integration