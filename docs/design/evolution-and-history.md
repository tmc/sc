# Statechart Evolution and Execution History

**Status:** Draft
**Author:** sc maintainers
**Created:** 2024-12-25

## Problem Statement

Statecharts in production systems need to:

1. **Track execution history** for debugging, auditing, and replay
2. **Evolve over time** as requirements change
3. **Migrate running instances** when chart definitions change
4. **Recover from failures** using checkpoints and logs

Currently, the sc library lacks standardized support for these concerns, leading to ad-hoc implementations that are difficult to maintain and interoperate.

## Goals

- Define standard formats for execution traces and history
- Support chart versioning with clear compatibility semantics
- Enable safe migrations between chart versions
- Provide recovery mechanisms for fault tolerance
- Keep the core statechart runtime simple (history is optional)

## Non-Goals

- Distributed consensus (use external systems like etcd/Raft)
- Real-time streaming (use external message queues)
- Long-term analytics (export to data warehouses)

---

## 1. Execution History (Runtime)

### Transition Log Format

Every transition produces a log entry:

```protobuf
message TransitionLogEntry {
  // Unique identifier for this entry
  string id = 1;

  // When the transition occurred
  google.protobuf.Timestamp timestamp = 2;

  // Sequence number within this machine instance
  uint64 sequence = 3;

  // The event that triggered this transition
  Event trigger_event = 4;

  // Configuration before transition
  Configuration source_config = 5;

  // Configuration after transition
  Configuration target_config = 6;

  // Which transition(s) fired
  repeated TransitionRef transitions_fired = 7;

  // Guard evaluation results (for debugging)
  repeated GuardEvaluation guard_results = 8;

  // Actions executed
  repeated ActionExecution actions_executed = 9;

  // Context/variables before
  google.protobuf.Struct context_before = 10;

  // Context/variables after
  google.protobuf.Struct context_after = 11;

  // Duration of transition processing
  google.protobuf.Duration processing_time = 12;

  // Error if transition failed
  string error = 13;

  // Metadata (trace IDs, request IDs, etc.)
  map<string, string> metadata = 14;

  // Distributed causality tracking
  CausalityInfo causality = 15;
}

message CausalityInfo {
  // Lamport timestamp for ordering
  uint64 lamport_clock = 1;

  // Vector clock for distributed causality (node_id -> clock)
  map<string, uint64> vector_clock = 2;

  // Parent event that caused this transition
  string caused_by = 3;

  // Node/partition that processed this event
  string node_id = 4;
}

message TransitionRef {
  string label = 1;
  repeated string from = 2;
  repeated string to = 3;
  string event = 4;
}

message GuardEvaluation {
  string guard_expression = 1;
  bool result = 2;
  google.protobuf.Struct bound_values = 3;  // Variable values used
  string error = 4;  // If evaluation failed
}

message ActionExecution {
  string action_name = 1;
  google.protobuf.Struct parameters = 2;
  google.protobuf.Duration duration = 3;
  string error = 4;
}
```

### Execution Trace

A complete execution trace aggregates log entries:

```protobuf
message ExecutionTrace {
  // Trace identifier
  string trace_id = 1;

  // Machine instance this trace belongs to
  string machine_id = 2;

  // Chart version being executed
  ChartVersion chart_version = 3;

  // Initial configuration
  Configuration initial_config = 4;

  // Initial context
  google.protobuf.Struct initial_context = 5;

  // Ordered list of transition entries
  repeated TransitionLogEntry entries = 6;

  // Final configuration (current state)
  Configuration final_config = 7;

  // Final context
  google.protobuf.Struct final_context = 8;

  // Trace metadata
  TraceMetadata metadata = 9;
}

message TraceMetadata {
  google.protobuf.Timestamp started_at = 1;
  google.protobuf.Timestamp ended_at = 2;
  uint64 total_transitions = 3;
  uint64 total_events_processed = 4;
  uint64 total_events_ignored = 5;
  repeated string errors = 6;

  // For aggregate traces across multiple machines
  uint64 machine_count = 7;

  // Sampling info (if trace was sampled)
  SamplingInfo sampling = 8;
}

// Trace sampling configuration for high-volume systems
message TraceConfig {
  // Sample rate (0.0 = none, 1.0 = all)
  double sample_rate = 1;

  // Always trace events matching these patterns
  repeated string always_trace_events = 2;

  // Always trace when in these states
  repeated string always_trace_states = 3;

  // Always trace on errors
  bool always_trace_errors = 4;

  // Trace all events for first N transitions (warm-up)
  uint32 warmup_transitions = 5;

  // Head-based vs tail-based sampling
  SamplingStrategy strategy = 6;
}

enum SamplingStrategy {
  SAMPLING_STRATEGY_UNSPECIFIED = 0;
  SAMPLING_STRATEGY_HEAD = 1;       // Decide at trace start
  SAMPLING_STRATEGY_TAIL = 2;       // Decide at trace end (keep interesting)
  SAMPLING_STRATEGY_ADAPTIVE = 3;   // Adjust based on load
}

message SamplingInfo {
  bool was_sampled = 1;
  double sample_rate_at_capture = 2;
  string sampling_decision_reason = 3;  // "always_trace_event:ERROR", "random", etc.
}
```

### History State Snapshots

For history pseudostates (H, H*), we need to track what configuration to restore:

```protobuf
message HistorySnapshot {
  // The composite state this history belongs to
  string composite_state = 1;

  // Type of history (shallow or deep)
  HistoryType type = 2;

  // The saved configuration
  Configuration saved_config = 3;

  // When this snapshot was taken
  google.protobuf.Timestamp captured_at = 4;

  // Which transition caused exit (for debugging)
  string exit_transition = 5;
}

message MachineHistoryState {
  // All history snapshots keyed by composite state label
  map<string, HistorySnapshot> snapshots = 1;
}
```

### Replay from Checkpoint

Support deterministic replay for debugging and testing:

```protobuf
message Checkpoint {
  // Checkpoint identifier
  string checkpoint_id = 1;

  // Machine state at checkpoint
  Machine machine_state = 2;

  // History snapshots at checkpoint
  MachineHistoryState history_state = 3;

  // Sequence number of last applied log entry
  uint64 last_sequence = 4;

  // Timestamp of checkpoint
  google.protobuf.Timestamp created_at = 5;

  // Chart version at checkpoint
  ChartVersion chart_version = 6;

  // Checksum for integrity verification
  string checksum = 7;
}

message ReplayRequest {
  // Start from this checkpoint
  Checkpoint checkpoint = 1;

  // Events to replay
  repeated Event events = 2;

  // Replay options
  ReplayOptions options = 3;
}

message ReplayOptions {
  // Stop at specific sequence number
  uint64 stop_at_sequence = 1;

  // Stop at specific timestamp
  google.protobuf.Timestamp stop_at_time = 2;

  // Stop when reaching specific configuration
  Configuration stop_at_config = 3;

  // Enable detailed logging during replay
  bool verbose = 4;

  // Fail on first divergence from expected trace
  bool strict_mode = 5;
}

message ReplayResult {
  // Final machine state after replay
  Machine final_state = 1;

  // Trace of replay execution
  ExecutionTrace trace = 2;

  // Divergences from expected behavior (if comparing)
  repeated TraceDivergence divergences = 3;
}
```

### Trace Comparison

Diff two execution traces to find behavioral differences:

```protobuf
message TraceDiff {
  // The two traces being compared
  string trace_a_id = 1;
  string trace_b_id = 2;

  // Summary
  TraceDiffSummary summary = 3;

  // Detailed divergences
  repeated TraceDivergence divergences = 4;

  // Comparison mode used
  TraceDiffMode mode = 5;
}

enum TraceDiffMode {
  TRACE_DIFF_MODE_UNSPECIFIED = 0;
  TRACE_DIFF_MODE_SYNTACTIC = 1;   // Exact match (default)
  TRACE_DIFF_MODE_SEMANTIC = 2;    // Same final state, different paths OK
  TRACE_DIFF_MODE_BEHAVIORAL = 3;  // Same observable behavior (events/outputs)
}

message TraceDiffSummary {
  bool identical = 1;
  uint64 common_prefix_length = 2;  // Entries that match
  uint64 divergence_count = 3;
  string first_divergence_event = 4;

  // Semantic equivalence (for SEMANTIC/BEHAVIORAL modes)
  bool semantically_equivalent = 5;
  bool same_final_state = 6;
  bool same_observable_outputs = 7;
}

message TraceDivergence {
  // Where in the trace this divergence occurs
  uint64 sequence = 1;

  // Type of divergence
  DivergenceType type = 2;

  // Details
  string description = 3;

  // Trace A value
  google.protobuf.Any value_a = 4;

  // Trace B value
  google.protobuf.Any value_b = 5;
}

enum DivergenceType {
  DIVERGENCE_TYPE_UNSPECIFIED = 0;
  DIVERGENCE_TYPE_DIFFERENT_TARGET = 1;      // Same event, different result config
  DIVERGENCE_TYPE_DIFFERENT_TRANSITIONS = 2; // Different transitions fired
  DIVERGENCE_TYPE_GUARD_DIFFERENCE = 3;      // Guard evaluated differently
  DIVERGENCE_TYPE_ACTION_DIFFERENCE = 4;     // Action had different effect
  DIVERGENCE_TYPE_MISSING_ENTRY = 5;         // Entry in one trace but not other
  DIVERGENCE_TYPE_EXTRA_ENTRY = 6;           // Extra entry in one trace
  DIVERGENCE_TYPE_CONTEXT_DIFFERENCE = 7;    // Context/variables differ
}
```

### Event Sourcing Pattern

Derive current state entirely from the event log:

```go
// EventSourcedMachine derives state from event log
type EventSourcedMachine struct {
    chartVersion ChartVersion
    log          []TransitionLogEntry

    // Cached derived state (invalidated on append)
    cachedConfig  *Configuration
    cachedContext *structpb.Struct
}

// Append adds an event and computes new state
func (m *EventSourcedMachine) Append(entry TransitionLogEntry) error {
    // Verify entry follows from current state
    if entry.Sequence != uint64(len(m.log)) {
        return fmt.Errorf("sequence mismatch: expected %d, got %d", len(m.log), entry.Sequence)
    }

    // Verify source config matches our current state
    if !configEqual(entry.SourceConfig, m.CurrentConfig()) {
        return fmt.Errorf("source config mismatch at sequence %d", entry.Sequence)
    }

    m.log = append(m.log, entry)
    m.cachedConfig = entry.TargetConfig
    m.cachedContext = entry.ContextAfter
    return nil
}

// Rebuild recomputes state from log (after recovery)
func (m *EventSourcedMachine) Rebuild() error {
    if len(m.log) == 0 {
        return nil
    }

    // Verify log consistency
    for i := 1; i < len(m.log); i++ {
        prev := m.log[i-1]
        curr := m.log[i]

        if curr.Sequence != prev.Sequence+1 {
            return fmt.Errorf("sequence gap at %d", i)
        }
        if !configEqual(curr.SourceConfig, prev.TargetConfig) {
            return fmt.Errorf("config discontinuity at %d", i)
        }
    }

    last := m.log[len(m.log)-1]
    m.cachedConfig = last.TargetConfig
    m.cachedContext = last.ContextAfter
    return nil
}

// CurrentConfig returns the current configuration
func (m *EventSourcedMachine) CurrentConfig() *Configuration {
    return m.cachedConfig
}

// ReplayFrom replays events starting from a sequence number
func (m *EventSourcedMachine) ReplayFrom(seq uint64, events []Event) (*ExecutionTrace, error) {
    // Create temporary machine at checkpoint
    // Apply events and return trace
}
```

---

## 2. Schema Evolution (Definition Changes)

### Chart Versioning Strategy

Use both semantic versioning AND content hash:

```protobuf
message ChartVersion {
  // Human-readable version (semver)
  string version = 1;  // e.g., "2.1.0"

  // Content-addressable hash of chart definition
  string content_hash = 2;  // e.g., "sha256:abc123..."

  // When this version was created
  google.protobuf.Timestamp created_at = 3;

  // Previous version (for lineage)
  string previous_version = 4;

  // Changelog
  string changelog = 5;

  // Compatibility classification
  CompatibilityLevel compatibility = 6;
}

enum CompatibilityLevel {
  COMPATIBILITY_LEVEL_UNSPECIFIED = 0;
  COMPATIBILITY_LEVEL_PATCH = 1;        // Bug fixes, no behavioral change
  COMPATIBILITY_LEVEL_MINOR = 2;        // New features, backward compatible
  COMPATIBILITY_LEVEL_MAJOR = 3;        // Breaking changes
}
```

### Content Hash Computation

```go
func ComputeChartHash(chart *Statechart) string {
    // Normalize chart (sort states, transitions, etc.)
    normalized := normalizeChart(chart)

    // Serialize to canonical form
    data, _ := proto.MarshalOptions{Deterministic: true}.Marshal(normalized)

    // Hash
    hash := sha256.Sum256(data)
    return "sha256:" + hex.EncodeToString(hash[:])
}

func normalizeChart(chart *Statechart) *Statechart {
    // Deep copy
    normalized := proto.Clone(chart).(*Statechart)

    // Sort states by label
    sortStates(normalized.RootState)

    // Sort transitions by (from, to, event)
    sort.Slice(normalized.Transitions, func(i, j int) bool {
        return transitionKey(normalized.Transitions[i]) <
               transitionKey(normalized.Transitions[j])
    })

    // Sort events by name
    sort.Slice(normalized.Events, func(i, j int) bool {
        return normalized.Events[i].Name < normalized.Events[j].Name
    })

    return normalized
}
```

### Chart Diff Format

```protobuf
message ChartDiff {
  // Versions being compared
  ChartVersion from_version = 1;
  ChartVersion to_version = 2;

  // State changes
  repeated StateDiff state_changes = 3;

  // Transition changes
  repeated TransitionDiff transition_changes = 4;

  // Event changes
  repeated EventDiff event_changes = 5;

  // Overall compatibility assessment
  CompatibilityLevel compatibility = 6;

  // Breaking change details
  repeated BreakingChange breaking_changes = 7;
}

message StateDiff {
  ChangeType change_type = 1;
  string state_label = 2;
  State old_state = 3;  // null for ADDED
  State new_state = 4;  // null for REMOVED
  repeated string modified_fields = 5;  // for MODIFIED
}

message TransitionDiff {
  ChangeType change_type = 1;
  string transition_label = 2;
  Transition old_transition = 3;
  Transition new_transition = 4;
  repeated string modified_fields = 5;
}

message EventDiff {
  ChangeType change_type = 1;
  string event_name = 2;
  Event old_event = 3;
  Event new_event = 4;
}

enum ChangeType {
  CHANGE_TYPE_UNSPECIFIED = 0;
  CHANGE_TYPE_ADDED = 1;
  CHANGE_TYPE_REMOVED = 2;
  CHANGE_TYPE_MODIFIED = 3;
  CHANGE_TYPE_RENAMED = 4;
  CHANGE_TYPE_MOVED = 5;  // State moved to different parent
}
```

### Breaking vs Non-Breaking Changes

```protobuf
message BreakingChange {
  BreakingChangeType type = 1;
  string description = 2;
  string affected_element = 3;
  MigrationRequirement migration_required = 4;
}

enum BreakingChangeType {
  BREAKING_CHANGE_TYPE_UNSPECIFIED = 0;

  // State changes
  BREAKING_CHANGE_TYPE_STATE_REMOVED = 1;
  BREAKING_CHANGE_TYPE_STATE_TYPE_CHANGED = 2;
  BREAKING_CHANGE_TYPE_INITIAL_STATE_CHANGED = 3;

  // Transition changes
  BREAKING_CHANGE_TYPE_TRANSITION_REMOVED = 4;
  BREAKING_CHANGE_TYPE_TRANSITION_SOURCE_CHANGED = 5;
  BREAKING_CHANGE_TYPE_GUARD_ADDED = 6;  // May block previously valid paths

  // Event changes
  BREAKING_CHANGE_TYPE_EVENT_REMOVED = 7;
  BREAKING_CHANGE_TYPE_EVENT_PAYLOAD_CHANGED = 8;
}

enum MigrationRequirement {
  MIGRATION_REQUIREMENT_UNSPECIFIED = 0;
  MIGRATION_REQUIREMENT_NONE = 1;           // Auto-migratable
  MIGRATION_REQUIREMENT_STATE_MAPPING = 2;  // Need explicit mapping
  MIGRATION_REQUIREMENT_MANUAL = 3;         // Requires custom code
}
```

### Classification Rules

```go
func ClassifyChange(diff *ChartDiff) CompatibilityLevel {
    // Check for breaking changes
    for _, bc := range diff.BreakingChanges {
        if bc.MigrationRequired == MIGRATION_REQUIREMENT_MANUAL {
            return COMPATIBILITY_LEVEL_MAJOR
        }
    }

    // Check state changes
    for _, sd := range diff.StateChanges {
        switch sd.ChangeType {
        case CHANGE_TYPE_REMOVED:
            // Removing a state is breaking
            return COMPATIBILITY_LEVEL_MAJOR
        case CHANGE_TYPE_MODIFIED:
            if containsAny(sd.ModifiedFields, "type", "is_initial") {
                return COMPATIBILITY_LEVEL_MAJOR
            }
        }
    }

    // Check transition changes
    for _, td := range diff.TransitionChanges {
        switch td.ChangeType {
        case CHANGE_TYPE_REMOVED:
            // Removing a transition may be breaking
            return COMPATIBILITY_LEVEL_MAJOR
        case CHANGE_TYPE_MODIFIED:
            if containsAny(td.ModifiedFields, "from", "to", "event") {
                return COMPATIBILITY_LEVEL_MAJOR
            }
            // Guard changes are minor (more restrictive) or patch (less restrictive)
            if contains(td.ModifiedFields, "guard") {
                // TODO: analyze guard change direction
                return COMPATIBILITY_LEVEL_MINOR
            }
        }
    }

    // Adding states/transitions is minor
    hasAdditions := false
    for _, sd := range diff.StateChanges {
        if sd.ChangeType == CHANGE_TYPE_ADDED {
            hasAdditions = true
        }
    }
    for _, td := range diff.TransitionChanges {
        if td.ChangeType == CHANGE_TYPE_ADDED {
            hasAdditions = true
        }
    }

    if hasAdditions {
        return COMPATIBILITY_LEVEL_MINOR
    }

    // Only metadata/comment changes
    return COMPATIBILITY_LEVEL_PATCH
}
```

### Deprecation Annotations

```protobuf
message DeprecationInfo {
  // When this element was deprecated
  string deprecated_in_version = 1;

  // When it will be removed
  string removal_version = 2;

  // Migration guidance
  string migration_guide = 3;

  // Replacement element (if any)
  string replacement = 4;
}

// Extended State message
message State {
  // ... existing fields ...

  // Deprecation info (if deprecated)
  DeprecationInfo deprecation = 20;
}

// Extended Transition message
message Transition {
  // ... existing fields ...

  // Deprecation info (if deprecated)
  DeprecationInfo deprecation = 20;
}
```

---

## 3. Migration Strategies

### Migration Plan

```protobuf
message MigrationPlan {
  // Plan identifier
  string id = 1;

  // Source and target versions
  ChartVersion from_version = 2;
  ChartVersion to_version = 3;

  // Migration strategy
  MigrationStrategy strategy = 4;

  // State mappings for the migration
  repeated StateMapping state_mappings = 5;

  // Transition mappings (for renamed transitions)
  repeated TransitionMapping transition_mappings = 6;

  // Context/variable transformations
  ContextTransformation context_transform = 7;

  // Validation rules
  repeated MigrationValidation validations = 8;

  // Estimated impact
  MigrationImpact impact = 9;

  // Dry-run mode (validate without applying)
  bool dry_run = 10;
}

// Result of dry-run migration
message DryRunResult {
  // Preview of each machine's migration
  repeated MachineMigrationPreview previews = 1;

  // Warnings (non-blocking issues)
  repeated string warnings = 2;

  // Errors (would cause migration to fail)
  repeated string errors = 3;

  // Would the migration succeed?
  bool would_succeed = 4;

  // Estimated duration
  google.protobuf.Duration estimated_duration = 5;
}

message MachineMigrationPreview {
  string machine_id = 1;
  Configuration current_config = 2;
  Configuration target_config = 3;
  StateMappingType mapping_used = 4;
  repeated string warnings = 5;
}

// Append-only migration history for audit
message MigrationHistory {
  repeated MigrationRecord records = 1;
}

message MigrationRecord {
  string migration_id = 1;
  ChartVersion from_version = 2;
  ChartVersion to_version = 3;
  google.protobuf.Timestamp started_at = 4;
  google.protobuf.Timestamp completed_at = 5;
  MigrationResult result = 6;
  uint64 machines_migrated = 7;
  uint64 machines_failed = 8;
  string initiated_by = 9;  // User/system that triggered
  string rollback_of = 10;  // If this was a rollback, reference original
}

enum MigrationResult {
  MIGRATION_RESULT_UNSPECIFIED = 0;
  MIGRATION_RESULT_SUCCESS = 1;
  MIGRATION_RESULT_PARTIAL = 2;      // Some machines failed
  MIGRATION_RESULT_FAILED = 3;
  MIGRATION_RESULT_ROLLED_BACK = 4;
  MIGRATION_RESULT_CANCELLED = 5;
}

enum MigrationStrategy {
  MIGRATION_STRATEGY_UNSPECIFIED = 0;
  MIGRATION_STRATEGY_STOP_THE_WORLD = 1;  // Pause, migrate all, resume
  MIGRATION_STRATEGY_ROLLING = 2;          // Migrate instances gradually
  MIGRATION_STRATEGY_BLUE_GREEN = 3;       // Run both versions, switch traffic
  MIGRATION_STRATEGY_CANARY = 4;           // Migrate subset first
}

message MigrationImpact {
  // Estimated number of machines affected
  uint64 machines_affected = 1;

  // Machines that cannot be auto-migrated
  uint64 machines_requiring_manual = 2;

  // Estimated downtime (for stop-the-world)
  google.protobuf.Duration estimated_downtime = 3;

  // Risk assessment
  RiskLevel risk = 4;
}

enum RiskLevel {
  RISK_LEVEL_UNSPECIFIED = 0;
  RISK_LEVEL_LOW = 1;       // Auto-migratable, reversible
  RISK_LEVEL_MEDIUM = 2;    // Some manual intervention
  RISK_LEVEL_HIGH = 3;      // Significant manual work, irreversible
}
```

### State Mapping Rules

```protobuf
message StateMapping {
  // Type of mapping
  StateMappingType type = 1;

  // Source state(s) in old version
  repeated string from_states = 2;

  // Target state(s) in new version
  repeated string to_states = 3;

  // Condition for this mapping (optional)
  Expression condition = 4;

  // Context transformation when mapping
  ContextTransformation transform = 5;

  // Priority (higher wins if multiple mappings match)
  int32 priority = 6;
}

enum StateMappingType {
  STATE_MAPPING_TYPE_UNSPECIFIED = 0;
  STATE_MAPPING_TYPE_IDENTITY = 1;      // Same state, no change
  STATE_MAPPING_TYPE_RENAME = 2;        // State renamed
  STATE_MAPPING_TYPE_TO_PARENT = 3;     // State removed, map to parent
  STATE_MAPPING_TYPE_TO_SIBLING = 4;    // State removed, map to sibling
  STATE_MAPPING_TYPE_TO_INITIAL = 5;    // State removed, map to initial state
  STATE_MAPPING_TYPE_SPLIT = 6;         // One state -> multiple states
  STATE_MAPPING_TYPE_MERGE = 7;         // Multiple states -> one state
  STATE_MAPPING_TYPE_CUSTOM = 8;        // Custom mapping logic
  STATE_MAPPING_TYPE_ERROR = 9;         // Cannot migrate, error out
}
```

### Migration Scenarios

#### State Renamed

```yaml
# Old chart
states:
  - label: Processing

# New chart
states:
  - label: InProgress  # Renamed

# Mapping
state_mappings:
  - type: RENAME
    from_states: [Processing]
    to_states: [InProgress]
```

#### State Removed (Map to Parent)

```yaml
# Old chart
states:
  - label: Active
    children:
      - label: Active.SubState1
      - label: Active.SubState2
      - label: Active.Legacy  # Being removed

# New chart
states:
  - label: Active
    children:
      - label: Active.SubState1
      - label: Active.SubState2
      # Active.Legacy removed

# Mapping - instances in Active.Legacy go to Active's initial child
state_mappings:
  - type: TO_INITIAL
    from_states: [Active.Legacy]
    to_states: [Active]  # Will resolve to Active's initial child
```

#### State Removed (Map to Sibling)

```yaml
# Mapping - instances in Active.Legacy go to specific sibling
state_mappings:
  - type: TO_SIBLING
    from_states: [Active.Legacy]
    to_states: [Active.SubState1]
```

#### State Split (One-to-Many)

```yaml
# Old chart
states:
  - label: Processing

# New chart - Processing split into two states
states:
  - label: Validating
  - label: Executing

# Mapping with condition
state_mappings:
  - type: SPLIT
    from_states: [Processing]
    to_states: [Validating]
    condition:
      source: "context.stage == 'validate'"
  - type: SPLIT
    from_states: [Processing]
    to_states: [Executing]
    condition:
      source: "context.stage != 'validate'"
```

#### State Merge (Many-to-One)

```yaml
# Old chart
states:
  - label: Pending
  - label: Queued
  - label: Scheduled

# New chart - merged into one
states:
  - label: Waiting

# Mapping
state_mappings:
  - type: MERGE
    from_states: [Pending, Queued, Scheduled]
    to_states: [Waiting]
    transform:
      # Preserve original state in context
      set:
        original_state: "$source_state"
```

### Handling In-Flight Transitions

```protobuf
message InFlightTransitionPolicy {
  InFlightAction action = 1;
  google.protobuf.Duration grace_period = 2;
}

enum InFlightAction {
  IN_FLIGHT_ACTION_UNSPECIFIED = 0;
  IN_FLIGHT_ACTION_COMPLETE = 1;    // Let transition complete before migrating
  IN_FLIGHT_ACTION_ABORT = 2;       // Abort transition, migrate immediately
  IN_FLIGHT_ACTION_RETRY = 3;       // Complete, then retry with new chart
  IN_FLIGHT_ACTION_ERROR = 4;       // Fail the transition
}
```

```go
// MigrationCoordinator handles live migration
type MigrationCoordinator struct {
    oldChart    *Statechart
    newChart    *Statechart
    plan        *MigrationPlan
    machines    map[string]*MachineWrapper
}

func (mc *MigrationCoordinator) MigrateMachine(machineID string) error {
    machine := mc.machines[machineID]

    // 1. Acquire lock on machine
    machine.mu.Lock()
    defer machine.mu.Unlock()

    // 2. Check for in-flight transition
    if machine.InTransition() {
        switch mc.plan.InFlightPolicy.Action {
        case IN_FLIGHT_ACTION_COMPLETE:
            // Wait for completion with timeout
            if err := machine.WaitForTransition(mc.plan.InFlightPolicy.GracePeriod); err != nil {
                return fmt.Errorf("transition timeout: %w", err)
            }
        case IN_FLIGHT_ACTION_ABORT:
            machine.AbortTransition()
        case IN_FLIGHT_ACTION_ERROR:
            return fmt.Errorf("machine %s has in-flight transition", machineID)
        }
    }

    // 3. Map current configuration to new chart
    oldConfig := machine.CurrentConfig()
    newConfig, err := mc.MapConfiguration(oldConfig)
    if err != nil {
        return fmt.Errorf("config mapping failed: %w", err)
    }

    // 4. Transform context
    oldContext := machine.Context()
    newContext, err := mc.TransformContext(oldContext)
    if err != nil {
        return fmt.Errorf("context transform failed: %w", err)
    }

    // 5. Create migration checkpoint (for rollback)
    checkpoint := &Checkpoint{
        CheckpointID:  fmt.Sprintf("migration-%s-%d", machineID, time.Now().Unix()),
        MachineState:  machine.Machine,
        ChartVersion:  mc.oldChart.Version,
        CreatedAt:     timestamppb.Now(),
    }
    if err := mc.saveCheckpoint(checkpoint); err != nil {
        return fmt.Errorf("checkpoint save failed: %w", err)
    }

    // 6. Apply migration
    machine.SetChart(mc.newChart)
    machine.SetConfig(newConfig)
    machine.SetContext(newContext)

    // 7. Validate new state
    if err := machine.Validate(); err != nil {
        // Rollback
        mc.rollback(machine, checkpoint)
        return fmt.Errorf("validation failed, rolled back: %w", err)
    }

    // 8. Log migration event
    machine.LogEvent(&TransitionLogEntry{
        Timestamp:    timestamppb.Now(),
        TriggerEvent: &Event{Name: "__MIGRATION__"},
        SourceConfig: oldConfig,
        TargetConfig: newConfig,
        Metadata: map[string]string{
            "migration_id":  mc.plan.ID,
            "from_version":  mc.oldChart.Version.Version,
            "to_version":    mc.newChart.Version.Version,
        },
    })

    return nil
}
```

### Rollback Strategies

```protobuf
message RollbackPlan {
  // Trigger conditions for automatic rollback
  repeated RollbackTrigger triggers = 1;

  // Rollback strategy
  RollbackStrategy strategy = 2;

  // State mappings for rollback (inverse of migration)
  repeated StateMapping rollback_mappings = 3;

  // Maximum rollback window
  google.protobuf.Duration rollback_window = 4;
}

message RollbackTrigger {
  RollbackTriggerType type = 1;
  string threshold = 2;  // e.g., "error_rate > 0.05"
}

enum RollbackTriggerType {
  ROLLBACK_TRIGGER_TYPE_UNSPECIFIED = 0;
  ROLLBACK_TRIGGER_TYPE_ERROR_RATE = 1;       // Too many errors
  ROLLBACK_TRIGGER_TYPE_LATENCY = 2;          // Performance degradation
  ROLLBACK_TRIGGER_TYPE_MANUAL = 3;           // Operator triggered
  ROLLBACK_TRIGGER_TYPE_HEALTH_CHECK = 4;     // Health check failures
}

enum RollbackStrategy {
  ROLLBACK_STRATEGY_UNSPECIFIED = 0;
  ROLLBACK_STRATEGY_CHECKPOINT = 1;   // Restore from checkpoint
  ROLLBACK_STRATEGY_REVERSE = 2;      // Apply inverse mappings
  ROLLBACK_STRATEGY_RECREATE = 3;     // Recreate from event log
}
```

---

## 4. Storage and Recovery

### Snapshot + Log Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     Storage Timeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Checkpoint 1        Checkpoint 2        Checkpoint 3       │
│      │                   │                   │              │
│      ▼                   ▼                   ▼              │
│  ┌───────┐           ┌───────┐           ┌───────┐         │
│  │ Snap  │           │ Snap  │           │ Snap  │         │
│  │  @10  │           │  @50  │           │  @100 │         │
│  └───────┘           └───────┘           └───────┘         │
│      │                   │                   │              │
│      ▼                   ▼                   ▼              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Log: [1] [2] ... [10] [11] ... [50] [51] ... [N]│      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  Recovery: Load Checkpoint 3, replay entries [101..N]       │
└─────────────────────────────────────────────────────────────┘
```

```protobuf
message StorageConfig {
  // Storage backend
  StorageBackend backend = 1;

  // Checkpoint configuration
  CheckpointConfig checkpoint = 2;

  // Log configuration
  LogConfig log = 3;

  // Compaction configuration
  CompactionConfig compaction = 4;

  // Replication configuration
  ReplicationConfig replication = 5;
}

enum StorageBackend {
  STORAGE_BACKEND_UNSPECIFIED = 0;
  STORAGE_BACKEND_MEMORY = 1;       // In-memory (testing)
  STORAGE_BACKEND_FILE = 2;         // Local filesystem
  STORAGE_BACKEND_SQLITE = 3;       // SQLite database
  STORAGE_BACKEND_POSTGRES = 4;     // PostgreSQL
  STORAGE_BACKEND_S3 = 5;           // Object storage
  STORAGE_BACKEND_CUSTOM = 6;       // Custom implementation
}

message CheckpointConfig {
  // When to create checkpoints
  CheckpointTrigger trigger = 1;

  // Maximum checkpoints to retain
  uint32 max_checkpoints = 2;

  // Compression
  CompressionType compression = 3;
}

message CheckpointTrigger {
  // Create checkpoint every N log entries
  uint64 every_n_entries = 1;

  // Create checkpoint every duration
  google.protobuf.Duration every_duration = 2;

  // Create checkpoint when log size exceeds threshold
  uint64 log_size_threshold_bytes = 3;

  // Adaptive checkpointing based on activity
  AdaptiveCheckpointConfig adaptive = 4;
}

message AdaptiveCheckpointConfig {
  // Enable adaptive checkpointing
  bool enabled = 1;

  // Target recovery time (checkpoint more during high activity)
  google.protobuf.Duration target_recovery_time = 2;

  // Minimum interval between checkpoints
  google.protobuf.Duration min_interval = 3;

  // Maximum interval between checkpoints
  google.protobuf.Duration max_interval = 4;

  // Activity threshold to trigger more frequent checkpoints
  double high_activity_threshold = 5;  // events/second
}

message LogConfig {
  // Maximum log entries before compaction
  uint64 max_entries = 1;

  // Maximum log size before compaction
  uint64 max_size_bytes = 2;

  // Sync mode
  SyncMode sync_mode = 3;

  // Compression for log segments
  CompressionType compression = 4;
}

enum SyncMode {
  SYNC_MODE_UNSPECIFIED = 0;
  SYNC_MODE_NONE = 1;           // No sync (fastest, may lose data)
  SYNC_MODE_BATCH = 2;          // Sync every batch
  SYNC_MODE_EVERY_ENTRY = 3;    // Sync every entry (slowest, safest)
}
```

### Checkpoint Frequency Strategies

```go
type CheckpointStrategy interface {
    ShouldCheckpoint(state *CheckpointState) bool
}

// Time-based checkpointing
type TimeBasedCheckpoint struct {
    Interval time.Duration
    lastCheckpoint time.Time
}

func (t *TimeBasedCheckpoint) ShouldCheckpoint(state *CheckpointState) bool {
    return time.Since(t.lastCheckpoint) >= t.Interval
}

// Count-based checkpointing
type CountBasedCheckpoint struct {
    EntryThreshold uint64
    lastSequence   uint64
}

func (c *CountBasedCheckpoint) ShouldCheckpoint(state *CheckpointState) bool {
    return state.CurrentSequence - c.lastSequence >= c.EntryThreshold
}

// Size-based checkpointing
type SizeBasedCheckpoint struct {
    SizeThreshold uint64
}

func (s *SizeBasedCheckpoint) ShouldCheckpoint(state *CheckpointState) bool {
    return state.LogSizeBytes >= s.SizeThreshold
}

// Adaptive checkpointing (based on recovery time target)
type AdaptiveCheckpoint struct {
    TargetRecoveryTime time.Duration
    avgEntryReplayTime time.Duration
}

func (a *AdaptiveCheckpoint) ShouldCheckpoint(state *CheckpointState) bool {
    estimatedRecovery := time.Duration(state.EntriesSinceCheckpoint) * a.avgEntryReplayTime
    return estimatedRecovery >= a.TargetRecoveryTime
}
```

### Compaction

```protobuf
message CompactionConfig {
  // Compaction strategy
  CompactionStrategy strategy = 1;

  // Retention policy
  RetentionPolicy retention = 2;

  // Schedule
  CompactionSchedule schedule = 3;
}

enum CompactionStrategy {
  COMPACTION_STRATEGY_UNSPECIFIED = 0;
  COMPACTION_STRATEGY_SNAPSHOT = 1;     // Keep only latest snapshot
  COMPACTION_STRATEGY_TIERED = 2;       // Keep recent detailed, older summarized
  COMPACTION_STRATEGY_WINDOW = 3;       // Keep fixed time window
}

message RetentionPolicy {
  // Minimum entries to retain
  uint64 min_entries = 1;

  // Minimum checkpoints to retain
  uint32 min_checkpoints = 2;

  // Maximum age of entries
  google.protobuf.Duration max_age = 3;

  // Keep all entries matching these criteria (for audit)
  repeated RetentionRule keep_rules = 4;

  // GDPR/Compliance settings
  ComplianceConfig compliance = 5;
}

message ComplianceConfig {
  // Legal minimum retention (e.g., 7 years for financial)
  google.protobuf.Duration min_retention = 1;

  // GDPR right to deletion maximum
  google.protobuf.Duration max_retention = 2;

  // Fields containing PII to scrub after retention
  repeated string pii_fields = 3;

  // Anonymization strategy for PII
  AnonymizationStrategy anonymization = 4;

  // Audit log requirements
  bool require_deletion_audit = 5;
}

enum AnonymizationStrategy {
  ANONYMIZATION_STRATEGY_UNSPECIFIED = 0;
  ANONYMIZATION_STRATEGY_DELETE = 1;        // Remove field entirely
  ANONYMIZATION_STRATEGY_HASH = 2;          // Replace with hash
  ANONYMIZATION_STRATEGY_PSEUDONYMIZE = 3;  // Replace with pseudonym
  ANONYMIZATION_STRATEGY_GENERALIZE = 4;    // e.g., "NYC" -> "US East"
}

message RetentionRule {
  string name = 1;
  string event_filter = 2;  // e.g., "event.name == 'PAYMENT_*'"
  google.protobuf.Duration retention = 3;
}
```

```go
// Compactor manages log compaction
type Compactor struct {
    config *CompactionConfig
    store  Storage
}

func (c *Compactor) Compact() error {
    // 1. Identify entries to remove
    entries, err := c.store.ListLogEntries()
    if err != nil {
        return err
    }

    checkpoints, err := c.store.ListCheckpoints()
    if err != nil {
        return err
    }

    // 2. Find safe compaction point
    // (oldest checkpoint we want to keep)
    safePoint := c.findSafeCompactionPoint(checkpoints)

    // 3. Apply retention rules
    toKeep := make(map[uint64]bool)
    for _, entry := range entries {
        if c.shouldKeep(entry) {
            toKeep[entry.Sequence] = true
        }
    }

    // 4. Delete old entries and checkpoints
    for _, entry := range entries {
        if entry.Sequence < safePoint && !toKeep[entry.Sequence] {
            if err := c.store.DeleteLogEntry(entry.Sequence); err != nil {
                return err
            }
        }
    }

    // 5. Delete old checkpoints
    for _, cp := range checkpoints {
        if cp.LastSequence < safePoint {
            if c.canDeleteCheckpoint(cp, checkpoints) {
                if err := c.store.DeleteCheckpoint(cp.CheckpointID); err != nil {
                    return err
                }
            }
        }
    }

    return nil
}
```

### Distributed/Replicated State

```protobuf
message ReplicationConfig {
  // Replication mode
  ReplicationMode mode = 1;

  // Number of replicas
  uint32 replica_count = 2;

  // Consistency level for reads
  ConsistencyLevel read_consistency = 3;

  // Consistency level for writes
  ConsistencyLevel write_consistency = 4;

  // Replica endpoints
  repeated string replicas = 5;
}

enum ReplicationMode {
  REPLICATION_MODE_UNSPECIFIED = 0;
  REPLICATION_MODE_NONE = 1;              // Single node
  REPLICATION_MODE_PRIMARY_BACKUP = 2;    // Primary with hot standbys
  REPLICATION_MODE_MULTI_PRIMARY = 3;     // Multiple writers (conflict resolution needed)
  REPLICATION_MODE_RAFT = 4;              // Raft consensus
}

enum ConsistencyLevel {
  CONSISTENCY_LEVEL_UNSPECIFIED = 0;
  CONSISTENCY_LEVEL_ONE = 1;          // Read/write from any replica
  CONSISTENCY_LEVEL_QUORUM = 2;       // Majority of replicas
  CONSISTENCY_LEVEL_ALL = 3;          // All replicas
  CONSISTENCY_LEVEL_LOCAL = 4;        // Prefer local replica
}
```

```go
// ReplicatedStore wraps storage with replication
type ReplicatedStore struct {
    local    Storage
    replicas []Storage
    config   *ReplicationConfig
}

func (rs *ReplicatedStore) AppendLogEntry(entry *TransitionLogEntry) error {
    switch rs.config.WriteConsistency {
    case CONSISTENCY_LEVEL_ONE:
        // Write to local only, async replicate
        if err := rs.local.AppendLogEntry(entry); err != nil {
            return err
        }
        go rs.replicateAsync(entry)
        return nil

    case CONSISTENCY_LEVEL_QUORUM:
        // Write to quorum of replicas
        return rs.writeQuorum(entry)

    case CONSISTENCY_LEVEL_ALL:
        // Write to all replicas
        return rs.writeAll(entry)
    }
    return nil
}

func (rs *ReplicatedStore) writeQuorum(entry *TransitionLogEntry) error {
    results := make(chan error, len(rs.replicas)+1)

    // Write to local
    go func() {
        results <- rs.local.AppendLogEntry(entry)
    }()

    // Write to replicas
    for _, replica := range rs.replicas {
        go func(r Storage) {
            results <- r.AppendLogEntry(entry)
        }(replica)
    }

    // Wait for quorum
    required := (len(rs.replicas) + 2) / 2  // Majority
    successes := 0
    var lastErr error

    for i := 0; i < len(rs.replicas)+1; i++ {
        if err := <-results; err != nil {
            lastErr = err
        } else {
            successes++
            if successes >= required {
                return nil
            }
        }
    }

    return fmt.Errorf("quorum not reached: %w", lastErr)
}
```

---

## 5. Complete Proto Schema

```protobuf
syntax = "proto3";

package statecharts.v1;

import "google/protobuf/any.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";

// ============================================
// Execution History
// ============================================

message TransitionLogEntry {
  string id = 1;
  google.protobuf.Timestamp timestamp = 2;
  uint64 sequence = 3;
  Event trigger_event = 4;
  Configuration source_config = 5;
  Configuration target_config = 6;
  repeated TransitionRef transitions_fired = 7;
  repeated GuardEvaluation guard_results = 8;
  repeated ActionExecution actions_executed = 9;
  google.protobuf.Struct context_before = 10;
  google.protobuf.Struct context_after = 11;
  google.protobuf.Duration processing_time = 12;
  string error = 13;
  map<string, string> metadata = 14;
}

message TransitionRef {
  string label = 1;
  repeated string from = 2;
  repeated string to = 3;
  string event = 4;
}

message GuardEvaluation {
  string guard_expression = 1;
  bool result = 2;
  google.protobuf.Struct bound_values = 3;
  string error = 4;
}

message ActionExecution {
  string action_name = 1;
  google.protobuf.Struct parameters = 2;
  google.protobuf.Duration duration = 3;
  string error = 4;
}

message ExecutionTrace {
  string trace_id = 1;
  string machine_id = 2;
  ChartVersion chart_version = 3;
  Configuration initial_config = 4;
  google.protobuf.Struct initial_context = 5;
  repeated TransitionLogEntry entries = 6;
  Configuration final_config = 7;
  google.protobuf.Struct final_context = 8;
  TraceMetadata metadata = 9;
}

message TraceMetadata {
  google.protobuf.Timestamp started_at = 1;
  google.protobuf.Timestamp ended_at = 2;
  uint64 total_transitions = 3;
  uint64 total_events_processed = 4;
  uint64 total_events_ignored = 5;
  repeated string errors = 6;
}

message HistorySnapshot {
  string composite_state = 1;
  HistoryType type = 2;
  Configuration saved_config = 3;
  google.protobuf.Timestamp captured_at = 4;
  string exit_transition = 5;
}

message MachineHistoryState {
  map<string, HistorySnapshot> snapshots = 1;
}

message Checkpoint {
  string checkpoint_id = 1;
  Machine machine_state = 2;
  MachineHistoryState history_state = 3;
  uint64 last_sequence = 4;
  google.protobuf.Timestamp created_at = 5;
  ChartVersion chart_version = 6;
  string checksum = 7;
}

message ReplayRequest {
  Checkpoint checkpoint = 1;
  repeated Event events = 2;
  ReplayOptions options = 3;
}

message ReplayOptions {
  uint64 stop_at_sequence = 1;
  google.protobuf.Timestamp stop_at_time = 2;
  Configuration stop_at_config = 3;
  bool verbose = 4;
  bool strict_mode = 5;

  // Replay speed multiplier (1.0 = realtime, 0 = max speed)
  double speed = 6;

  // Pause before each transition for debugging
  bool step_mode = 7;
}

message ReplayResult {
  Machine final_state = 1;
  ExecutionTrace trace = 2;
  repeated TraceDivergence divergences = 3;
}

message TraceDiff {
  string trace_a_id = 1;
  string trace_b_id = 2;
  TraceDiffSummary summary = 3;
  repeated TraceDivergence divergences = 4;
}

message TraceDiffSummary {
  bool identical = 1;
  uint64 common_prefix_length = 2;
  uint64 divergence_count = 3;
  string first_divergence_event = 4;
}

message TraceDivergence {
  uint64 sequence = 1;
  DivergenceType type = 2;
  string description = 3;
  google.protobuf.Any value_a = 4;
  google.protobuf.Any value_b = 5;
}

enum DivergenceType {
  DIVERGENCE_TYPE_UNSPECIFIED = 0;
  DIVERGENCE_TYPE_DIFFERENT_TARGET = 1;
  DIVERGENCE_TYPE_DIFFERENT_TRANSITIONS = 2;
  DIVERGENCE_TYPE_GUARD_DIFFERENCE = 3;
  DIVERGENCE_TYPE_ACTION_DIFFERENCE = 4;
  DIVERGENCE_TYPE_MISSING_ENTRY = 5;
  DIVERGENCE_TYPE_EXTRA_ENTRY = 6;
  DIVERGENCE_TYPE_CONTEXT_DIFFERENCE = 7;
}

// ============================================
// Schema Evolution
// ============================================

message ChartVersion {
  string version = 1;
  string content_hash = 2;
  google.protobuf.Timestamp created_at = 3;
  string previous_version = 4;
  string changelog = 5;
  CompatibilityLevel compatibility = 6;
}

enum CompatibilityLevel {
  COMPATIBILITY_LEVEL_UNSPECIFIED = 0;
  COMPATIBILITY_LEVEL_PATCH = 1;
  COMPATIBILITY_LEVEL_MINOR = 2;
  COMPATIBILITY_LEVEL_MAJOR = 3;
}

message ChartDiff {
  ChartVersion from_version = 1;
  ChartVersion to_version = 2;
  repeated StateDiff state_changes = 3;
  repeated TransitionDiff transition_changes = 4;
  repeated EventDiff event_changes = 5;
  CompatibilityLevel compatibility = 6;
  repeated BreakingChange breaking_changes = 7;
}

message StateDiff {
  ChangeType change_type = 1;
  string state_label = 2;
  State old_state = 3;
  State new_state = 4;
  repeated string modified_fields = 5;
}

message TransitionDiff {
  ChangeType change_type = 1;
  string transition_label = 2;
  Transition old_transition = 3;
  Transition new_transition = 4;
  repeated string modified_fields = 5;
}

message EventDiff {
  ChangeType change_type = 1;
  string event_name = 2;
  Event old_event = 3;
  Event new_event = 4;
}

enum ChangeType {
  CHANGE_TYPE_UNSPECIFIED = 0;
  CHANGE_TYPE_ADDED = 1;
  CHANGE_TYPE_REMOVED = 2;
  CHANGE_TYPE_MODIFIED = 3;
  CHANGE_TYPE_RENAMED = 4;
  CHANGE_TYPE_MOVED = 5;
}

message BreakingChange {
  BreakingChangeType type = 1;
  string description = 2;
  string affected_element = 3;
  MigrationRequirement migration_required = 4;
}

enum BreakingChangeType {
  BREAKING_CHANGE_TYPE_UNSPECIFIED = 0;
  BREAKING_CHANGE_TYPE_STATE_REMOVED = 1;
  BREAKING_CHANGE_TYPE_STATE_TYPE_CHANGED = 2;
  BREAKING_CHANGE_TYPE_INITIAL_STATE_CHANGED = 3;
  BREAKING_CHANGE_TYPE_TRANSITION_REMOVED = 4;
  BREAKING_CHANGE_TYPE_TRANSITION_SOURCE_CHANGED = 5;
  BREAKING_CHANGE_TYPE_GUARD_ADDED = 6;
  BREAKING_CHANGE_TYPE_EVENT_REMOVED = 7;
  BREAKING_CHANGE_TYPE_EVENT_PAYLOAD_CHANGED = 8;
}

enum MigrationRequirement {
  MIGRATION_REQUIREMENT_UNSPECIFIED = 0;
  MIGRATION_REQUIREMENT_NONE = 1;
  MIGRATION_REQUIREMENT_STATE_MAPPING = 2;
  MIGRATION_REQUIREMENT_MANUAL = 3;
}

message DeprecationInfo {
  string deprecated_in_version = 1;
  string removal_version = 2;
  string migration_guide = 3;
  string replacement = 4;
}

// ============================================
// Migration
// ============================================

message MigrationPlan {
  string id = 1;
  ChartVersion from_version = 2;
  ChartVersion to_version = 3;
  MigrationStrategy strategy = 4;
  repeated StateMapping state_mappings = 5;
  repeated TransitionMapping transition_mappings = 6;
  ContextTransformation context_transform = 7;
  repeated MigrationValidation validations = 8;
  MigrationImpact impact = 9;
  InFlightTransitionPolicy in_flight_policy = 10;
  RollbackPlan rollback_plan = 11;
}

enum MigrationStrategy {
  MIGRATION_STRATEGY_UNSPECIFIED = 0;
  MIGRATION_STRATEGY_STOP_THE_WORLD = 1;
  MIGRATION_STRATEGY_ROLLING = 2;
  MIGRATION_STRATEGY_BLUE_GREEN = 3;
  MIGRATION_STRATEGY_CANARY = 4;
}

message StateMapping {
  StateMappingType type = 1;
  repeated string from_states = 2;
  repeated string to_states = 3;
  Expression condition = 4;
  ContextTransformation transform = 5;
  int32 priority = 6;
}

enum StateMappingType {
  STATE_MAPPING_TYPE_UNSPECIFIED = 0;
  STATE_MAPPING_TYPE_IDENTITY = 1;
  STATE_MAPPING_TYPE_RENAME = 2;
  STATE_MAPPING_TYPE_TO_PARENT = 3;
  STATE_MAPPING_TYPE_TO_SIBLING = 4;
  STATE_MAPPING_TYPE_TO_INITIAL = 5;
  STATE_MAPPING_TYPE_SPLIT = 6;
  STATE_MAPPING_TYPE_MERGE = 7;
  STATE_MAPPING_TYPE_CUSTOM = 8;
  STATE_MAPPING_TYPE_ERROR = 9;
}

message TransitionMapping {
  string from_transition = 1;
  string to_transition = 2;
}

message ContextTransformation {
  // CEL expressions for transforming context
  map<string, string> set = 1;      // key -> expression
  repeated string remove = 2;        // keys to remove
  map<string, string> rename = 3;    // old_key -> new_key
}

message MigrationValidation {
  string name = 1;
  string expression = 2;  // CEL expression that must be true
  string error_message = 3;
}

message MigrationImpact {
  uint64 machines_affected = 1;
  uint64 machines_requiring_manual = 2;
  google.protobuf.Duration estimated_downtime = 3;
  RiskLevel risk = 4;
}

enum RiskLevel {
  RISK_LEVEL_UNSPECIFIED = 0;
  RISK_LEVEL_LOW = 1;
  RISK_LEVEL_MEDIUM = 2;
  RISK_LEVEL_HIGH = 3;
}

message InFlightTransitionPolicy {
  InFlightAction action = 1;
  google.protobuf.Duration grace_period = 2;
}

enum InFlightAction {
  IN_FLIGHT_ACTION_UNSPECIFIED = 0;
  IN_FLIGHT_ACTION_COMPLETE = 1;
  IN_FLIGHT_ACTION_ABORT = 2;
  IN_FLIGHT_ACTION_RETRY = 3;
  IN_FLIGHT_ACTION_ERROR = 4;
}

message RollbackPlan {
  repeated RollbackTrigger triggers = 1;
  RollbackStrategy strategy = 2;
  repeated StateMapping rollback_mappings = 3;
  google.protobuf.Duration rollback_window = 4;
}

message RollbackTrigger {
  RollbackTriggerType type = 1;
  string threshold = 2;
}

enum RollbackTriggerType {
  ROLLBACK_TRIGGER_TYPE_UNSPECIFIED = 0;
  ROLLBACK_TRIGGER_TYPE_ERROR_RATE = 1;
  ROLLBACK_TRIGGER_TYPE_LATENCY = 2;
  ROLLBACK_TRIGGER_TYPE_MANUAL = 3;
  ROLLBACK_TRIGGER_TYPE_HEALTH_CHECK = 4;
}

enum RollbackStrategy {
  ROLLBACK_STRATEGY_UNSPECIFIED = 0;
  ROLLBACK_STRATEGY_CHECKPOINT = 1;
  ROLLBACK_STRATEGY_REVERSE = 2;
  ROLLBACK_STRATEGY_RECREATE = 3;
}

// ============================================
// Storage
// ============================================

message StorageConfig {
  StorageBackend backend = 1;
  CheckpointConfig checkpoint = 2;
  LogConfig log = 3;
  CompactionConfig compaction = 4;
  ReplicationConfig replication = 5;
}

enum StorageBackend {
  STORAGE_BACKEND_UNSPECIFIED = 0;
  STORAGE_BACKEND_MEMORY = 1;
  STORAGE_BACKEND_FILE = 2;
  STORAGE_BACKEND_SQLITE = 3;
  STORAGE_BACKEND_POSTGRES = 4;
  STORAGE_BACKEND_S3 = 5;
  STORAGE_BACKEND_CUSTOM = 6;
}

message CheckpointConfig {
  CheckpointTrigger trigger = 1;
  uint32 max_checkpoints = 2;
  CompressionType compression = 3;
}

message CheckpointTrigger {
  uint64 every_n_entries = 1;
  google.protobuf.Duration every_duration = 2;
  uint64 log_size_threshold_bytes = 3;
}

message LogConfig {
  uint64 max_entries = 1;
  uint64 max_size_bytes = 2;
  SyncMode sync_mode = 3;
  CompressionType compression = 4;
}

enum SyncMode {
  SYNC_MODE_UNSPECIFIED = 0;
  SYNC_MODE_NONE = 1;
  SYNC_MODE_BATCH = 2;
  SYNC_MODE_EVERY_ENTRY = 3;
}

enum CompressionType {
  COMPRESSION_TYPE_UNSPECIFIED = 0;
  COMPRESSION_TYPE_NONE = 1;
  COMPRESSION_TYPE_GZIP = 2;
  COMPRESSION_TYPE_ZSTD = 3;
  COMPRESSION_TYPE_LZ4 = 4;
}

message CompactionConfig {
  CompactionStrategy strategy = 1;
  RetentionPolicy retention = 2;
  CompactionSchedule schedule = 3;
}

enum CompactionStrategy {
  COMPACTION_STRATEGY_UNSPECIFIED = 0;
  COMPACTION_STRATEGY_SNAPSHOT = 1;
  COMPACTION_STRATEGY_TIERED = 2;
  COMPACTION_STRATEGY_WINDOW = 3;
}

message RetentionPolicy {
  uint64 min_entries = 1;
  uint32 min_checkpoints = 2;
  google.protobuf.Duration max_age = 3;
  repeated RetentionRule keep_rules = 4;
}

message RetentionRule {
  string name = 1;
  string event_filter = 2;
  google.protobuf.Duration retention = 3;
}

message CompactionSchedule {
  google.protobuf.Duration interval = 1;
  string cron = 2;  // Alternative: cron expression
}

message ReplicationConfig {
  ReplicationMode mode = 1;
  uint32 replica_count = 2;
  ConsistencyLevel read_consistency = 3;
  ConsistencyLevel write_consistency = 4;
  repeated string replicas = 5;
}

enum ReplicationMode {
  REPLICATION_MODE_UNSPECIFIED = 0;
  REPLICATION_MODE_NONE = 1;
  REPLICATION_MODE_PRIMARY_BACKUP = 2;
  REPLICATION_MODE_MULTI_PRIMARY = 3;
  REPLICATION_MODE_RAFT = 4;
}

enum ConsistencyLevel {
  CONSISTENCY_LEVEL_UNSPECIFIED = 0;
  CONSISTENCY_LEVEL_ONE = 1;
  CONSISTENCY_LEVEL_QUORUM = 2;
  CONSISTENCY_LEVEL_ALL = 3;
  CONSISTENCY_LEVEL_LOCAL = 4;
}
```

---

## Implementation Plan

### Phase 1: Execution History (Week 1-2)
- [ ] Add TransitionLogEntry, ExecutionTrace to proto
- [ ] Implement in-memory trace collector
- [ ] Add trace export (JSON, proto binary)
- [ ] Implement trace comparison (TraceDiff)

### Phase 2: Checkpoints & Replay (Week 3-4)
- [ ] Add Checkpoint message to proto
- [ ] Implement checkpoint creation/restoration
- [ ] Implement deterministic replay
- [ ] Add replay CLI commands

### Phase 3: Schema Evolution (Week 5-6)
- [ ] Add ChartVersion, ChartDiff to proto
- [ ] Implement content hash computation
- [ ] Implement diff generation
- [ ] Add compatibility classification

### Phase 4: Migration (Week 7-8)
- [ ] Add MigrationPlan, StateMapping to proto
- [ ] Implement state mapping engine
- [ ] Implement migration coordinator
- [ ] Add rollback support

### Phase 5: Storage Backends (Week 9-10)
- [ ] Define Storage interface
- [ ] Implement file-based storage
- [ ] Implement SQLite storage
- [ ] Add compaction

### Phase 6: Advanced Features (Week 11-12)
- [ ] Implement replication (primary-backup)
- [ ] Add migration CLI commands
- [ ] Integration tests
- [ ] Documentation

---

## Open Questions

1. **Should history be opt-in or default?**
   - Opt-in: Less overhead for simple use cases
   - Default: Easier debugging, audit compliance

2. **How to handle clock skew in distributed traces?**
   - Logical clocks (Lamport/vector)?
   - Hybrid logical clocks?

3. **Should migration plans be stored with the chart?**
   - Embedded: Self-contained, versioned together
   - Separate: More flexible, can update independently

4. **Maximum trace size before auto-compaction?**
   - Configurable per deployment
   - Default: 10,000 entries or 100MB

5. **How to handle non-deterministic actions during replay?**
   - Record action results in log
   - Mock external calls during replay

---

## References

- [Event Sourcing Pattern](https://martinfowler.com/eaaDev/EventSourcing.html)
- [CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Raft Consensus](https://raft.github.io/)
- [Write-Ahead Logging](https://www.sqlite.org/wal.html)
- [Schema Evolution in Protobuf](https://protobuf.dev/programming-guides/proto3/#updating)
- [Blue-Green Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html)
