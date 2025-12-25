# Standard Extensions for Statecharts Proto

## Status: PROPOSED
## Author: Claude Code / tmc
## Date: 2025-12-25

---

## Abstract

This document specifies a set of **standard extension types** for the sc
statecharts protocol buffer schema. Extensions use the `google.protobuf.Any`
mechanism in `State.extensions` and `Transition.extensions` fields, enabling
tool-specific data without polluting the core Harel formalism.

## Design Principles

### 1. Core Purity
The core `statecharts.proto` remains pure Harel formalism [H87, HN96]. Extensions
are strictly additive and optional.

### 2. Tool Interoperability
Standard extensions provide a common vocabulary for tools to exchange data.
A Mermaid renderer and a D3 visualizer can both consume `Layout` extensions.

### 3. Namespace Organization
Extensions are organized by concern, not by tool:
```
proto/extensions/v1/
  layout.proto        # Visual positioning
  annotations.proto   # Documentation/metadata
  testing.proto       # Test scenarios
  simulation.proto    # Timing/delays
  provenance.proto    # Origin tracking
```

### 4. Graceful Degradation
Tools SHOULD ignore extensions they don't understand. A CLI validator can
function without layout data; a visualizer can function without provenance.

---

## Extension Specifications

### 1. Layout Extension (`layout.proto`)

Tool-agnostic visual positioning for states, transitions, and connection points.

```protobuf
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

// ===========================================================================
// LAYOUT EXTENSION - Tool-Agnostic Visual Positioning
//
// Provides standardized positioning data consumable by multiple renderers:
// Mermaid, D3.js, Graphviz, web editors, desktop applications.
//
// COORDINATE SYSTEM:
// - Origin (0,0) at top-left of canvas
// - X increases rightward
// - Y increases downward
// - Units are logical pixels (renderer-specific scaling)
//
// DESIGN RATIONALE:
// Unlike XState's layout which is editor-specific, this extension provides
// a minimal, portable representation that any visualization tool can use.
// ===========================================================================

// StateLayout provides visual positioning for a state node.
// Stored in State.extensions.
message StateLayout {
  // Position of state center on canvas.
  Point position = 1;

  // Bounding box dimensions.
  Dimensions size = 2;

  // Visual styling hints (tool may override).
  VisualStyle style = 3;

  // Z-order for overlapping states (higher = on top).
  int32 z_index = 4;

  // Anchor points for incoming/outgoing transitions.
  repeated Anchor anchors = 5;

  // Layout algorithm hint: "manual", "auto", "force-directed", "hierarchical".
  string layout_hint = 6;

  // Whether position is user-specified vs auto-computed.
  bool position_locked = 7;
}

// TransitionLayout provides visual routing for transition edges.
// Stored in Transition.extensions.
message TransitionLayout {
  // Ordered waypoints for edge routing (excluding endpoints).
  repeated Point waypoints = 1;

  // Edge style: "straight", "curved", "orthogonal", "bezier".
  string edge_style = 2;

  // Arrow style: "filled", "open", "none".
  string arrow_style = 3;

  // Label position along edge: 0.0 = source, 0.5 = midpoint, 1.0 = target.
  double label_position = 4;

  // Label offset perpendicular to edge (positive = right side).
  double label_offset = 5;

  // Visual styling for the edge.
  VisualStyle style = 6;

  // Source anchor index (references StateLayout.anchors).
  int32 source_anchor = 7;

  // Target anchor index (references StateLayout.anchors).
  int32 target_anchor = 8;
}

// Point represents a 2D coordinate.
message Point {
  double x = 1;
  double y = 2;
}

// Dimensions represents width and height.
message Dimensions {
  double width = 1;
  double height = 2;
}

// Anchor defines a connection point on a state boundary.
message Anchor {
  // Anchor ID for reference by transitions.
  string id = 1;

  // Position as fraction of state bounds: (0,0)=top-left, (1,1)=bottom-right.
  Point relative_position = 2;

  // Direction hint for edge routing: "n", "s", "e", "w", "ne", etc.
  string direction = 3;
}

// VisualStyle provides portable styling hints.
// Tools SHOULD interpret these but MAY apply their own defaults.
message VisualStyle {
  // Fill color as CSS color string: "#RRGGBB", "rgba(...)", "red".
  string fill_color = 1;

  // Stroke color as CSS color string.
  string stroke_color = 2;

  // Stroke width in pixels.
  double stroke_width = 3;

  // Stroke pattern: "solid", "dashed", "dotted".
  string stroke_pattern = 4;

  // Corner radius for rounded rectangles.
  double corner_radius = 5;

  // Opacity: 0.0 (transparent) to 1.0 (opaque).
  double opacity = 6;

  // Font family for labels.
  string font_family = 7;

  // Font size in points.
  double font_size = 8;

  // Font weight: "normal", "bold", "100"-"900".
  string font_weight = 9;

  // Text color for labels.
  string text_color = 10;
}

// CanvasLayout provides global canvas settings.
// Can be stored in Statechart-level metadata or a dedicated extension.
message CanvasLayout {
  // Canvas dimensions.
  Dimensions size = 1;

  // Background color.
  string background_color = 2;

  // Grid spacing for snap-to-grid (0 = no grid).
  double grid_spacing = 3;

  // Default layout direction: "TB", "LR", "BT", "RL".
  string direction = 4;

  // Inter-state spacing hints.
  double horizontal_spacing = 5;
  double vertical_spacing = 6;

  // Padding around state groups.
  double group_padding = 7;
}
```

**Use Cases:**
- Mermaid diagram generation with consistent positioning
- D3.js force-directed layout with stable anchor points
- Graphviz export with preserved manual positioning
- Web editor state persistence
- Multi-tool round-trip fidelity

---

### 2. Annotations Extension (`annotations.proto`)

Documentation, tagging, and metadata for search, filtering, and generation.

```protobuf
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

import "google/protobuf/timestamp.proto";

// ===========================================================================
// ANNOTATIONS EXTENSION - Documentation and Metadata
//
// Provides structured documentation and categorization for statechart elements.
// Enables: documentation generation, semantic search, filtering, auditing.
//
// DESIGN RATIONALE:
// While State.metadata provides ad-hoc key-value storage, this extension
// defines a structured schema for common documentation patterns.
// ===========================================================================

// StateAnnotations provides documentation for a state.
// Stored in State.extensions.
message StateAnnotations {
  // Human-readable description (Markdown supported).
  string description = 1;

  // Short summary (for tooltips, listings).
  string summary = 2;

  // Semantic tags for categorization and filtering.
  repeated string tags = 3;

  // Category path: ["ui", "modal", "confirmation"].
  repeated string category = 4;

  // Author attribution.
  Author author = 5;

  // Version when this state was added.
  string since_version = 6;

  // Deprecation notice (if deprecated).
  Deprecation deprecation = 7;

  // Related states (for documentation cross-references).
  repeated string related_states = 8;

  // External documentation links.
  repeated Link links = 9;

  // Requirement traceability: ["REQ-001", "US-123"].
  repeated string requirements = 10;

  // Risk assessment for critical states.
  RiskAssessment risk = 11;
}

// TransitionAnnotations provides documentation for a transition.
// Stored in Transition.extensions.
message TransitionAnnotations {
  // Human-readable description (Markdown supported).
  string description = 1;

  // Short summary (for tooltips, listings).
  string summary = 2;

  // Semantic tags for categorization.
  repeated string tags = 3;

  // Author attribution.
  Author author = 4;

  // Version when this transition was added.
  string since_version = 5;

  // Deprecation notice (if deprecated).
  Deprecation deprecation = 6;

  // Requirement traceability.
  repeated string requirements = 7;

  // Preconditions (human-readable, for documentation).
  repeated string preconditions = 8;

  // Postconditions (human-readable, for documentation).
  repeated string postconditions = 9;

  // Side effects (external systems affected).
  repeated string side_effects = 10;
}

// EventAnnotations provides documentation for an event.
// Can be linked via event label lookup.
message EventAnnotations {
  // Human-readable description.
  string description = 1;

  // Event payload schema (JSON Schema format).
  string payload_schema = 2;

  // Example payloads.
  repeated string examples = 3;

  // Producer systems (what emits this event).
  repeated string producers = 4;

  // Consumer states (what reacts to this event).
  repeated string consumers = 5;
}

// Author provides attribution information.
message Author {
  // Display name.
  string name = 1;

  // Email address.
  string email = 2;

  // Organization.
  string organization = 3;

  // ORCID or similar identifier.
  string identifier = 4;
}

// Deprecation marks an element as deprecated.
message Deprecation {
  // Whether this element is deprecated.
  bool deprecated = 1;

  // Version when deprecated.
  string since_version = 2;

  // Reason for deprecation.
  string reason = 3;

  // Replacement element (label).
  string replacement = 4;

  // Target removal version.
  string removal_version = 5;
}

// Link provides an external reference.
message Link {
  // Link title.
  string title = 1;

  // URL.
  string url = 2;

  // Link type: "documentation", "design", "issue", "figma", "confluence".
  string type = 3;
}

// RiskAssessment captures safety-critical information.
message RiskAssessment {
  // Risk level: "low", "medium", "high", "critical".
  string level = 1;

  // Risk category: "safety", "security", "compliance", "performance".
  string category = 2;

  // Detailed risk description.
  string description = 3;

  // Mitigation measures.
  repeated string mitigations = 4;

  // Regulatory references: ["ISO-26262", "DO-178C"].
  repeated string regulatory_refs = 5;
}
```

**Use Cases:**
- Documentation generation (Markdown, HTML, PDF)
- Semantic search across statecharts
- Filtering states by tag/category in editors
- Requirement traceability for compliance
- Risk assessment for safety-critical systems

---

### 3. Testing Extension (`testing.proto`)

Test scenarios, expected traces, and coverage tracking.

```protobuf
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

// ===========================================================================
// TESTING EXTENSION - Test Scenarios and Coverage
//
// Provides test case definitions, expected execution traces, and coverage
// markers for statechart testing tools.
//
// DESIGN RATIONALE:
// Testing statecharts requires defining event sequences and expected
// configurations. This extension standardizes test case representation
// for interoperability between test runners.
//
// REFERENCES:
// - Model-based testing [UTP2]
// - Property-based testing [QuickCheck]
// ===========================================================================

// TestSuite groups related test cases.
// Can be attached to Statechart-level extensions.
message TestSuite {
  // Suite name.
  string name = 1;

  // Suite description.
  string description = 2;

  // Test cases in this suite.
  repeated TestCase test_cases = 3;

  // Setup actions (run before each test).
  repeated TestAction setup = 4;

  // Teardown actions (run after each test).
  repeated TestAction teardown = 5;

  // Tags for filtering.
  repeated string tags = 6;
}

// TestCase defines a single test scenario.
message TestCase {
  // Unique test identifier.
  string id = 1;

  // Human-readable test name.
  string name = 2;

  // Test description.
  string description = 3;

  // Initial context/variables override.
  map<string, string> initial_context = 4;

  // Event sequence to execute.
  repeated TestEvent events = 5;

  // Expected outcomes to verify.
  repeated TestAssertion assertions = 6;

  // Whether test is enabled.
  bool enabled = 7;

  // Tags for filtering: ["smoke", "regression", "edge-case"].
  repeated string tags = 8;

  // Timeout for test execution (milliseconds).
  int64 timeout_ms = 9;

  // Priority: higher = run first.
  int32 priority = 10;

  // Skip reason (if disabled).
  string skip_reason = 11;
}

// TestEvent represents an event to send during test execution.
message TestEvent {
  // Event name.
  string event = 1;

  // Event payload as JSON string.
  string payload = 2;

  // Delay before sending (milliseconds).
  int64 delay_ms = 3;

  // Description of what this event tests.
  string comment = 4;
}

// TestAction represents a setup/teardown action.
message TestAction {
  // Action type: "set_context", "reset", "invoke".
  string type = 1;

  // Action-specific parameters.
  map<string, string> params = 2;
}

// TestAssertion defines an expected outcome.
message TestAssertion {
  // Assertion type.
  AssertionType type = 1;

  // Expected value (interpretation depends on type).
  string expected = 2;

  // Message on failure.
  string message = 3;

  // When to check: "after_event", "final", "any_time".
  string timing = 4;

  // Event index after which to check (for "after_event").
  int32 after_event_index = 5;
}

// AssertionType classifies test assertions.
enum AssertionType {
  ASSERTION_TYPE_UNSPECIFIED = 0;

  // Assert specific state is active.
  ASSERTION_TYPE_STATE_ACTIVE = 1;

  // Assert specific state is NOT active.
  ASSERTION_TYPE_STATE_NOT_ACTIVE = 2;

  // Assert exact configuration (comma-separated state labels).
  ASSERTION_TYPE_CONFIGURATION_EQUALS = 3;

  // Assert configuration contains states (subset check).
  ASSERTION_TYPE_CONFIGURATION_CONTAINS = 4;

  // Assert context variable value.
  ASSERTION_TYPE_CONTEXT_EQUALS = 5;

  // Assert action was executed.
  ASSERTION_TYPE_ACTION_EXECUTED = 6;

  // Assert event was emitted.
  ASSERTION_TYPE_EVENT_EMITTED = 7;

  // Assert transition was taken.
  ASSERTION_TYPE_TRANSITION_TAKEN = 8;

  // Assert invariant expression holds.
  ASSERTION_TYPE_INVARIANT = 9;
}

// StateCoverageMarker tracks coverage for a state.
// Stored in State.extensions.
message StateCoverageMarker {
  // Coverage requirement: "must_enter", "must_exit", "must_dwell".
  repeated string requirements = 1;

  // Minimum dwell time for coverage (milliseconds).
  int64 min_dwell_ms = 2;

  // Whether state was covered in last run.
  bool covered = 3;

  // Coverage count across runs.
  int64 coverage_count = 4;

  // Test cases that cover this state.
  repeated string covered_by = 5;
}

// TransitionCoverageMarker tracks coverage for a transition.
// Stored in Transition.extensions.
message TransitionCoverageMarker {
  // Coverage requirement: "must_fire", "must_fire_with_guard_true".
  repeated string requirements = 1;

  // Whether transition was covered in last run.
  bool covered = 2;

  // Coverage count across runs.
  int64 coverage_count = 3;

  // Test cases that cover this transition.
  repeated string covered_by = 4;

  // Guard evaluation coverage.
  bool guard_true_covered = 5;
  bool guard_false_covered = 6;
}

// ExpectedTrace defines an expected execution sequence.
// Used for trace-based testing.
message ExpectedTrace {
  // Trace identifier.
  string id = 1;

  // Trace name.
  string name = 2;

  // Ordered sequence of expected steps.
  repeated TraceStep steps = 3;

  // Whether order is strict or just occurrence required.
  bool strict_order = 4;
}

// TraceStep defines a single expected step in a trace.
message TraceStep {
  // Step type: "entry", "exit", "action", "transition".
  string type = 1;

  // Target element label.
  string target = 2;

  // Expected event (for transition steps).
  string event = 3;

  // Optional context assertions at this step.
  map<string, string> context = 4;
}
```

**Use Cases:**
- Automated test runners
- Coverage reporting tools
- Regression test suites
- Property-based testing integration
- Mutation testing frameworks

---

### 4. Simulation Extension (`simulation.proto`)

Timing, delays, probabilities, and mock configurations for simulation.

```protobuf
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

import "google/protobuf/duration.proto";

// ===========================================================================
// SIMULATION EXTENSION - Execution Simulation Configuration
//
// Provides timing, delays, probabilities, and mock configurations for
// statechart simulation and model checking.
//
// DESIGN RATIONALE:
// Simulation requires timing semantics beyond the core Harel formalism.
// This extension adds:
// - Temporal constraints (timeouts, delays, deadlines)
// - Stochastic behavior (probabilities, distributions)
// - Mock configurations (simulated external systems)
//
// REFERENCES:
// - Timed Automata [AD94]
// - Stochastic Petri Nets
// - PRISM Model Checker
// ===========================================================================

// StateSimConfig provides simulation settings for a state.
// Stored in State.extensions.
message StateSimConfig {
  // Timing constraints.
  TimingConstraints timing = 1;

  // Resource consumption while in state.
  ResourceProfile resources = 2;

  // Mock/stub configuration for state entry.
  MockConfig entry_mock = 3;

  // Failure injection settings.
  FailureConfig failure = 4;

  // Logging/tracing verbosity: 0=none, 1=entry/exit, 2=actions, 3=all.
  int32 trace_level = 5;
}

// TransitionSimConfig provides simulation settings for a transition.
// Stored in Transition.extensions.
message TransitionSimConfig {
  // Transition probability weight (for non-deterministic choice).
  double probability = 1;

  // Execution delay range.
  DelayRange delay = 2;

  // Mock configuration for transition execution.
  MockConfig mock = 3;

  // Failure injection settings.
  FailureConfig failure = 4;

  // Cost/weight for optimization (e.g., shortest path).
  double cost = 5;
}

// TimingConstraints defines temporal requirements.
message TimingConstraints {
  // Minimum time before exiting state (dwell time).
  google.protobuf.Duration min_dwell = 1;

  // Maximum time before forced exit (timeout).
  google.protobuf.Duration max_dwell = 2;

  // Expected/typical dwell time (for statistics).
  google.protobuf.Duration expected_dwell = 3;

  // Deadline: state must be exited by this time.
  google.protobuf.Duration deadline = 4;

  // Entry delay (simulated processing time).
  DelayRange entry_delay = 5;

  // Exit delay (cleanup time).
  DelayRange exit_delay = 6;

  // Timeout event to generate on max_dwell expiry.
  string timeout_event = 7;
}

// DelayRange specifies a delay with optional variance.
message DelayRange {
  // Minimum delay.
  google.protobuf.Duration min = 1;

  // Maximum delay.
  google.protobuf.Duration max = 2;

  // Distribution: "uniform", "normal", "exponential", "constant".
  string distribution = 3;

  // Distribution parameters (e.g., mean, stddev for normal).
  map<string, double> params = 4;
}

// ResourceProfile defines resource consumption.
message ResourceProfile {
  // CPU utilization: 0.0 to 1.0.
  double cpu_utilization = 1;

  // Memory usage in bytes.
  int64 memory_bytes = 2;

  // Network bandwidth in bytes/sec.
  int64 network_bps = 3;

  // Power consumption in watts.
  double power_watts = 4;

  // Custom resource metrics.
  map<string, double> custom = 5;
}

// MockConfig defines mock/stub behavior for simulation.
message MockConfig {
  // Whether to use mock (true) or real implementation (false).
  bool enabled = 1;

  // Mock response mode: "success", "failure", "random", "sequence".
  string mode = 2;

  // Response delay.
  DelayRange response_delay = 3;

  // Mock responses for sequence mode.
  repeated MockResponse responses = 4;

  // Random failure probability.
  double failure_rate = 5;

  // Default mock return value (JSON).
  string default_value = 6;
}

// MockResponse defines a single mock response.
message MockResponse {
  // Response value (JSON).
  string value = 1;

  // Whether this response is an error.
  bool is_error = 2;

  // Error type/code.
  string error_code = 3;

  // Response delay override.
  google.protobuf.Duration delay = 4;
}

// FailureConfig defines failure injection for testing.
message FailureConfig {
  // Failure injection enabled.
  bool enabled = 1;

  // Failure probability per entry/execution.
  double failure_rate = 2;

  // Failure type: "crash", "timeout", "exception", "corrupt".
  string failure_type = 3;

  // Failure trigger: "random", "after_n", "condition".
  string trigger = 4;

  // Trigger parameter (e.g., N for "after_n").
  int64 trigger_param = 5;

  // Recovery behavior: "retry", "skip", "escalate".
  string recovery = 6;

  // Maximum retry attempts.
  int32 max_retries = 7;
}

// SimulationScenario defines a complete simulation configuration.
// Can be attached to Statechart-level extensions.
message SimulationScenario {
  // Scenario name.
  string name = 1;

  // Scenario description.
  string description = 2;

  // Simulation time limit.
  google.protobuf.Duration time_limit = 3;

  // Event generation schedule.
  repeated ScheduledEvent scheduled_events = 4;

  // Random event generation config.
  RandomEventConfig random_events = 5;

  // Global resource constraints.
  ResourceConstraints constraints = 6;

  // Seed for reproducible random simulation.
  int64 random_seed = 7;

  // Number of simulation runs.
  int32 run_count = 8;

  // Metrics to collect.
  repeated string metrics = 9;
}

// ScheduledEvent defines an event at a specific simulation time.
message ScheduledEvent {
  // Time offset from simulation start.
  google.protobuf.Duration time = 1;

  // Event name.
  string event = 2;

  // Event payload (JSON).
  string payload = 3;

  // Whether to repeat.
  bool repeat = 4;

  // Repeat interval.
  google.protobuf.Duration repeat_interval = 5;
}

// RandomEventConfig defines random event generation.
message RandomEventConfig {
  // Events that can be randomly generated.
  repeated RandomEvent events = 1;

  // Global event rate (events per second).
  double rate = 2;
}

// RandomEvent defines a randomly generated event.
message RandomEvent {
  // Event name.
  string event = 1;

  // Selection weight (higher = more likely).
  double weight = 2;

  // Payload generator: "constant", "random_int", "random_string".
  string payload_generator = 3;

  // Generator parameters.
  map<string, string> generator_params = 4;
}

// ResourceConstraints defines global simulation limits.
message ResourceConstraints {
  // Maximum concurrent states.
  int32 max_concurrent_states = 1;

  // Maximum events per second.
  double max_event_rate = 2;

  // Maximum memory.
  int64 max_memory_bytes = 3;

  // Maximum CPU.
  double max_cpu = 4;
}
```

**Use Cases:**
- Performance simulation
- Stress testing with failure injection
- Model checking with timing constraints
- Stochastic analysis
- Resource consumption estimation

---

### 5. Provenance Extension (`provenance.proto`)

Origin tracking for extracted or generated statecharts.

```protobuf
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

import "google/protobuf/timestamp.proto";

// ===========================================================================
// PROVENANCE EXTENSION - Origin and Extraction Information
//
// Tracks the source of statechart elements for:
// - Extracted statecharts (from source code, binaries, documentation)
// - Generated statecharts (from specifications, models)
// - Imported statecharts (format conversions)
//
// DESIGN RATIONALE:
// When statecharts are extracted from existing systems (e.g., Zelda3 ROM,
// SCXML files, source code), provenance enables:
// - Debugging (trace back to original source)
// - Re-extraction (detect drift)
// - Auditing (compliance, security)
// - Attribution (license, copyright)
//
// REFERENCES:
// - W3C PROV-O (Provenance Ontology)
// - SPDX (Software Package Data Exchange)
// ===========================================================================

// StateProvenance tracks the origin of a state.
// Stored in State.extensions.
message StateProvenance {
  // Source location where state was extracted from.
  SourceLocation source = 1;

  // Extraction tool information.
  Tool extractor = 2;

  // Extraction timestamp.
  google.protobuf.Timestamp extracted_at = 3;

  // Confidence score: 0.0 (guess) to 1.0 (certain).
  double confidence = 4;

  // Extraction method: "static_analysis", "dynamic_trace", "manual", "llm".
  string method = 5;

  // Original identifier in source system.
  string original_id = 6;

  // Original name/label in source system.
  string original_name = 7;

  // Notes from extraction process.
  string notes = 8;

  // Human reviewer (if manually verified).
  string reviewed_by = 9;

  // Review timestamp.
  google.protobuf.Timestamp reviewed_at = 10;

  // Review status: "pending", "approved", "rejected", "needs_revision".
  string review_status = 11;
}

// TransitionProvenance tracks the origin of a transition.
// Stored in Transition.extensions.
message TransitionProvenance {
  // Source location where transition was extracted from.
  SourceLocation source = 1;

  // Extraction tool information.
  Tool extractor = 2;

  // Extraction timestamp.
  google.protobuf.Timestamp extracted_at = 3;

  // Confidence score.
  double confidence = 4;

  // Extraction method.
  string method = 5;

  // Original representation in source.
  string original_expression = 6;

  // Whether guard was inferred vs explicit.
  bool guard_inferred = 7;

  // Notes from extraction.
  string notes = 8;
}

// SourceLocation identifies a location in source material.
message SourceLocation {
  // Source file path or URL.
  string path = 1;

  // Line number (1-indexed).
  int32 line = 2;

  // Column number (1-indexed).
  int32 column = 3;

  // End line (for ranges).
  int32 end_line = 4;

  // End column.
  int32 end_column = 5;

  // Byte offset in file.
  int64 offset = 6;

  // Byte length of source span.
  int64 length = 7;

  // Git commit hash.
  string commit = 8;

  // Branch or tag name.
  string branch = 9;

  // Repository URL.
  string repository = 10;

  // Source type: "c", "asm", "scxml", "xstate", "binary", "documentation".
  string source_type = 11;
}

// Tool identifies a tool that processed the statechart.
message Tool {
  // Tool name.
  string name = 1;

  // Tool version.
  string version = 2;

  // Tool URL/homepage.
  string url = 3;

  // Tool configuration used.
  map<string, string> config = 4;

  // Tool invocation command.
  string command = 5;
}

// ChartProvenance tracks statechart-level provenance.
// Can be attached to Statechart-level metadata or extensions.
message ChartProvenance {
  // Primary source system.
  SourceSystem source_system = 1;

  // Extraction pipeline (ordered tools applied).
  repeated Tool pipeline = 2;

  // Overall extraction timestamp.
  google.protobuf.Timestamp created_at = 3;

  // License information.
  License license = 4;

  // Attribution/copyright.
  string attribution = 5;

  // Extraction report URL.
  string report_url = 6;

  // Derived from (for transformations).
  string derived_from = 7;

  // Generation/version number.
  int32 generation = 8;
}

// SourceSystem identifies the original system.
message SourceSystem {
  // System name: "zelda3", "linux_kernel", "aws_step_functions".
  string name = 1;

  // System version.
  string version = 2;

  // System type: "game_rom", "source_code", "scxml", "documentation".
  string type = 3;

  // Platform: "snes", "linux", "aws".
  string platform = 4;

  // Checksum of source (e.g., ROM hash).
  string checksum = 5;

  // Checksum algorithm: "sha256", "md5", "crc32".
  string checksum_algorithm = 6;

  // Source URL or identifier.
  string url = 7;
}

// License captures licensing information.
message License {
  // SPDX license identifier: "MIT", "Apache-2.0", "GPL-3.0".
  string spdx_id = 1;

  // Full license name.
  string name = 2;

  // License URL.
  string url = 3;

  // Copyright holder.
  string copyright = 4;

  // Year(s).
  string year = 5;

  // Additional terms or notices.
  string notice = 6;
}

// ExtractionIssue records problems encountered during extraction.
message ExtractionIssue {
  // Issue severity: "error", "warning", "info".
  string severity = 1;

  // Issue code/type.
  string code = 2;

  // Human-readable message.
  string message = 3;

  // Source location where issue occurred.
  SourceLocation location = 4;

  // Suggested fix.
  string suggestion = 5;

  // Whether issue was resolved.
  bool resolved = 6;

  // Resolution notes.
  string resolution = 7;
}

// ExtractionReport summarizes the extraction process.
message ExtractionReport {
  // Overall status: "success", "partial", "failed".
  string status = 1;

  // States extracted.
  int32 states_extracted = 2;

  // Transitions extracted.
  int32 transitions_extracted = 3;

  // Events discovered.
  int32 events_discovered = 4;

  // Issues encountered.
  repeated ExtractionIssue issues = 5;

  // Extraction duration.
  int64 duration_ms = 6;

  // Coverage of source analyzed.
  double source_coverage = 7;

  // Confidence metrics.
  map<string, double> confidence_metrics = 8;
}
```

**Use Cases:**
- Zelda3 statechart extraction (ROM to proto)
- Source code analysis tools
- SCXML/XState import with full lineage
- Audit trails for compliance
- License/attribution tracking
- Re-extraction drift detection

---

## Extension Registry

To enable discovery and validation, extensions are registered:

```protobuf
// extension_registry.proto
syntax = "proto3";
package extensions.v1;

option go_package = "github.com/tmc/sc/gen/extensions/v1;extensionspb";

// ExtensionDescriptor documents a registered extension type.
message ExtensionDescriptor {
  // Full proto message type URL.
  string type_url = 1;

  // Human-readable name.
  string name = 2;

  // Description.
  string description = 3;

  // Applicable to: "state", "transition", "event", "statechart".
  repeated string applies_to = 4;

  // Extension category: "layout", "annotations", "testing", "simulation", "provenance".
  string category = 5;

  // Specification version.
  string version = 6;

  // Stability: "stable", "beta", "experimental".
  string stability = 7;
}

// ExtensionRegistry lists all known extensions.
message ExtensionRegistry {
  repeated ExtensionDescriptor extensions = 1;
}
```

## Tool Compatibility Matrix

| Tool | Layout | Annotations | Testing | Simulation | Provenance |
|------|--------|-------------|---------|------------|------------|
| Mermaid Generator | Read | - | - | - | - |
| D3 Visualizer | Read/Write | Read | - | - | - |
| Web Editor | Read/Write | Read/Write | - | - | Read |
| sc CLI | - | Read | - | - | - |
| Test Runner | - | Read | Read/Write | - | - |
| Model Checker | - | - | Read | Read | - |
| Zelda3 Extractor | Write | Write | - | - | Write |
| Doc Generator | Read | Read | - | - | Read |
| Coverage Tool | - | - | Read/Write | - | - |

## Migration Path

1. **Phase 1**: Define extension protos, generate code
2. **Phase 2**: Update extractors to emit provenance
3. **Phase 3**: Update visualizers to consume layout
4. **Phase 4**: Integrate testing extensions with sc test runner
5. **Phase 5**: Add simulation support to model checker

## Cross-References with Core Protos

### testing.proto ↔ execution.proto Alignment

| testing.proto | execution.proto | Purpose |
|---------------|-----------------|---------|
| `ExpectedTrace` | `ExecutionTrace` | Expected vs actual |
| `TraceStep` | `TransitionLogEntry` | Lightweight spec vs full record |
| `TestAssertion` | - | Verification predicates |

**Design Decision:** Keep separate because:
- ExpectedTrace is a *specification* (what should happen)
- ExecutionTrace is a *record* (what did happen)
- Different verbosity levels (TraceStep is minimal, TransitionLogEntry is maximal)

**Interop Functions (to be implemented in Go):**
- `ExecutionTraceToExpected(trace) → ExpectedTrace` - Generate golden test from execution
- `CompareTrace(expected, actual) → []Difference` - Verify execution matches specification

### provenance.proto ↔ execution.proto Alignment

Provenance may reference execution traces for dynamic extraction:
```protobuf
message StateProvenance {
  ...
  // Execution trace where this state was observed (dynamic extraction)
  string discovery_trace_id = 16;
  // Specific entry in trace where state was entered
  uint64 discovery_sequence = 17;
}
```

**Status:** Deferred - add when dynamic extraction is implemented.

---

## Future Extensions

### visualization.proto (PLANNED)

Dynamic visualization beyond static layout:
- **Semantic zoom levels**: detail vs overview
- **Live trace overlay**: highlight active states/transitions
- **Metrics overlay**: state dwell time, transition frequency
- **Animation keyframes**: step-through visualization

**Status:** Await CA16 visualization design doc before specifying.

### integration.proto (PROPOSED)

Tool-specific adapter configurations:
- XState import/export hints
- SCXML element ID mappings
- Tool-specific feature flags

```protobuf
message XStateAdapterConfig {
  bool preserve_invoke_actors = 1;
  bool convert_always_to_completion = 2;
  map<string, string> action_mappings = 3;
}

message SCXMLAdapterConfig {
  string scxml_version = 1;
  bool preserve_datamodel = 2;
  map<string, string> namespace_mappings = 3;
}
```

**Status:** Proposed - implement when import/export stabilizes.

---

## Open Questions

1. **Namespace**: Should extensions be `extensions.v1` or `statecharts.v1.extensions`?
2. **Backward Compatibility**: How to handle old tools ignoring new extensions?
3. **Validation**: Should `sc validate` check extension well-formedness?
4. **Compression**: Should large extensions (layout) support compression?
5. **Trace Conversion**: Should trace interop be in testing.proto or separate?
6. **Visualization Scope**: Static (layout.proto) vs dynamic (visualization.proto) boundary?

---

## References

- [H87] D. Harel, "Statecharts: A visual formalism for complex systems," 1987
- [HN96] D. Harel and A. Naamad, "The STATEMATE semantics of statecharts," 1996
- [AD94] R. Alur and D. Dill, "A theory of timed automata," 1994
- [UTP2] UML Testing Profile 2.0, OMG
- [PROV-O] W3C PROV-O: The PROV Ontology
- [SPDX] Software Package Data Exchange Specification

