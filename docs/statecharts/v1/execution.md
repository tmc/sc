---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="execution-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="statecharts-v1-TransitionLogEntry"></a>

### TransitionLogEntry

TransitionLogEntry records a single step in statechart execution.

FORMAL DEFINITION [HN96, Section 4]:
A log entry L = (t, seq, e, σ, σ', T, G, A, Γ, Γ') captures:
  - t: timestamp of step execution
  - seq: monotonic sequence number
  - e: triggering event
  - σ, σ': source and target configurations
  - T: set of transitions that fired
  - G: guard evaluation results
  - A: actions executed
  - Γ, Γ': context before and after

SEMANTIC INVARIANTS:
1. seq(Lᵢ) < seq(Lᵢ₊₁) for all consecutive entries (monotonicity)
2. σ'(Lᵢ) = σ(Lᵢ₊₁) (configuration continuity)
3. Γ'(Lᵢ) = Γ(Lᵢ₊₁) (context continuity)

CAUSALITY [Lam78]:
For distributed statecharts, entries include causality information
to establish happened-before ordering across nodes.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Unique identifier for this log entry   |
| timestamp |Timestamp| Timestamp: t ∈ Time Wall-clock time when this step was executed   |
| sequence |uint64| Sequence number: seq ∈ ℕ Monotonically increasing within a machine instance. Used for ordering and gap detection.   |
| trigger_event |[Event](./statecharts.md#statecharts-v1-Event)| Triggering event: e ∈ E ∪ {τ} The event that initiated this step (τ for completion events)   |
| source_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)| Source configuration: σ ∈ P(S) Active states before transition   |
| target_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)| Target configuration: σ' ∈ P(S) Active states after transition   |
| transitions_fired[] |[TransitionRef](#statecharts-v1-TransitionRef)| Fired transitions: T ⊆ δ Set of transitions that executed in this step   |
| guard_results[] |[GuardEvaluation](#statecharts-v1-GuardEvaluation)| Guard evaluations: G = {(g, result, bindings)} Results of guard condition evaluation for debugging   |
| actions_executed[] |[ActionExecution](#statecharts-v1-ActionExecution)| Executed actions: A = α₁ · α₂ · ... · αₙ Sequence of actions executed during this step   |
| context_before |Struct| Context before: Γ Variable bindings before transition   |
| context_after |Struct| Context after: Γ' Variable bindings after transition   |
| processing_time |Duration| Processing duration: Δt Time spent executing this step   |
| error |string| Error message if step failed   |
| metadata |[TransitionLogEntry.MetadataEntry](#statecharts-v1-TransitionLogEntry-MetadataEntry)| Arbitrary metadata (trace IDs, request IDs, etc.)   |
| causality |[CausalityInfo](#statecharts-v1-CausalityInfo)| Causality information for distributed ordering [Lam78]   |






<a name="statecharts-v1-TransitionLogEntry-MetadataEntry"></a>

### MetadataEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TransitionRef"></a>

### TransitionRef

TransitionRef is a lightweight reference to a transition.

FORMAL DEFINITION:
A reference r = (label, from, to, event) identifies a transition t ∈ δ
without including the full transition definition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|   |
| from[] |string|   |
| to[] |string|   |
| event |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-GuardEvaluation"></a>

### GuardEvaluation

GuardEvaluation records the result of evaluating a guard condition.

FORMAL DEFINITION [HN96, Section 4.2]:
Guard evaluation Geval = (expr, result, bindings, error) where:
  - expr: the guard expression g ∈ G
  - result: evaluation outcome ∈ {true, false}
  - bindings: variable values used during evaluation
  - error: evaluation error if any

DETERMINISM REQUIREMENT:
For reproducible replay, guard evaluation must be deterministic:
  eval(g, Γ₁) = eval(g, Γ₂) when Γ₁ = Γ₂




| Field | Type | Description |
| ----- | ---- | ----------- |
| guard_expression |string| Guard expression that was evaluated   |
| result |bool| Evaluation result: true if guard passed   |
| bound_values |Struct| Variable bindings used during evaluation Enables debugging of why guard passed/failed   |
| error |string| Error message if evaluation failed   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ActionExecution"></a>

### ActionExecution

ActionExecution records the execution of a single action.

FORMAL DEFINITION [HN96, Section 6]:
Action execution Aexec = (name, params, duration, error) where:
  - name: action identifier
  - params: action parameters
  - duration: execution time
  - error: execution error if any




| Field | Type | Description |
| ----- | ---- | ----------- |
| action_name |string|   |
| parameters |Struct|   |
| duration |Duration|   |
| error |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CausalityInfo"></a>

### CausalityInfo

CausalityInfo establishes ordering for distributed statechart execution.

FORMAL DEFINITION [Lam78]:
Causality tracking uses logical clocks to establish happened-before (→):
  - Lamport clock: L(e) ∈ ℕ, monotonic counter
  - Vector clock: V(e) = (c₁, c₂, ..., cₙ) for n nodes

ORDERING PROPERTIES:
1. If a → b within same process, then L(a) < L(b)
2. If a is send and b is corresponding receive, then L(a) < L(b)
3. Vector clocks provide exact causality: V(a) < V(b) ⟺ a → b




| Field | Type | Description |
| ----- | ---- | ----------- |
| lamport_clock |uint64| Lamport timestamp: L ∈ ℕ Simple scalar clock for total ordering   |
| vector_clock |[CausalityInfo.VectorClockEntry](#statecharts-v1-CausalityInfo-VectorClockEntry)| Vector clock: V = {node_id → counter} Provides exact causality detection for distributed systems   |
| caused_by |string| Parent event ID: caused_by Links this entry to the event that caused it   |
| node_id |string| Processing node identifier Which node/partition processed this event   |






<a name="statecharts-v1-CausalityInfo-VectorClockEntry"></a>

### VectorClockEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |uint64|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ExecutionTrace"></a>

### ExecutionTrace

ExecutionTrace represents a complete execution history of a machine instance.

FORMAL DEFINITION [HN96, Section 4]:
A trace τ = (id, σ₀, Γ₀, L*, σₙ, Γₙ) where:
  - id: unique trace identifier
  - σ₀, Γ₀: initial configuration and context
  - L* = L₁ · L₂ · ... · Lₙ: sequence of log entries
  - σₙ, Γₙ: final configuration and context

WELL-FORMEDNESS CONSTRAINTS:
1. |L*| = n ⟹ seq(Lᵢ) = i for all i ∈ [1,n]
2. σ'(Lᵢ) = σ(Lᵢ₊₁) (configuration chain)
3. σ(L₁) = σ₀ and σ'(Lₙ) = σₙ (boundary conditions)

REPLAY PROPERTY:
Given (σ₀, Γ₀) and events(τ), deterministic replay yields identical trace.




| Field | Type | Description |
| ----- | ---- | ----------- |
| trace_id |string| Trace identifier   |
| machine_id |string| Machine instance this trace belongs to   |
| chart_version |[ChartVersion](#statecharts-v1-ChartVersion)| Chart version being executed   |
| initial_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)| Initial configuration: σ₀   |
| initial_context |Struct| Initial context: Γ₀   |
| entries[] |[TransitionLogEntry](#statecharts-v1-TransitionLogEntry)| Log entries: L* = L₁ · L₂ · ... · Lₙ   |
| final_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)| Final configuration: σₙ   |
| final_context |Struct| Final context: Γₙ   |
| metadata |[TraceMetadata](#statecharts-v1-TraceMetadata)| Trace metadata   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ChartVersion"></a>

### ChartVersion

ChartVersion identifies a specific version of a statechart definition.

FORMAL DEFINITION:
Version V = (semver, hash, parent) where:
  - semver: human-readable version string
  - hash: content-addressable hash of chart definition
  - parent: previous version for lineage

IDENTITY SEMANTICS:
Two charts are identical iff their content hashes match:
  SC₁ ≡ SC₂ ⟺ hash(SC₁) = hash(SC₂)




| Field | Type | Description |
| ----- | ---- | ----------- |
| version |string| Human-readable version (semver recommended)   |
| content_hash |string| Content-addressable hash: sha256(canonical(SC))   |
| created_at |Timestamp| Creation timestamp   |
| previous_version |string| Previous version for lineage tracking   |
| changelog |string| Changelog description   |
| compatibility |[CompatibilityLevel](#statecharts-v1-CompatibilityLevel)| Compatibility classification   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TraceMetadata"></a>

### TraceMetadata

TraceMetadata provides summary statistics and sampling information.




| Field | Type | Description |
| ----- | ---- | ----------- |
| started_at |Timestamp|   |
| ended_at |Timestamp|   |
| total_transitions |uint64|   |
| total_events_processed |uint64|   |
| total_events_ignored |uint64|   |
| errors[] |string|   |
| machine_count |uint64| Number of machines in aggregate trace   |
| sampling |[SamplingInfo](#statecharts-v1-SamplingInfo)| Sampling information   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TraceConfig"></a>

### TraceConfig

TraceConfig configures trace sampling for high-volume systems.

FORMAL DEFINITION:
Sampling config S = (rate, filters, strategy) where:
  - rate ∈ [0,1]: probability of sampling a trace
  - filters: conditions that force sampling
  - strategy: when sampling decision is made

SAMPLING SEMANTICS:
For rate r, approximately r×N traces are sampled from N total.
Filters override rate for important events (errors, specific states).




| Field | Type | Description |
| ----- | ---- | ----------- |
| sample_rate |double| Sample rate: P(sample) ∈ [0,1]   |
| always_trace_events[] |string| Force sampling for these events   |
| always_trace_states[] |string| Force sampling when in these states   |
| always_trace_errors |bool| Force sampling on errors   |
| warmup_transitions |uint32| Warmup: trace first N transitions unconditionally   |
| strategy |[SamplingStrategy](#statecharts-v1-SamplingStrategy)| Sampling strategy   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-SamplingInfo"></a>

### SamplingInfo

SamplingInfo records how a trace was sampled.




| Field | Type | Description |
| ----- | ---- | ----------- |
| was_sampled |bool|   |
| sample_rate_at_capture |double|   |
| sampling_decision_reason |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-HistorySnapshot"></a>

### HistorySnapshot

HistorySnapshot captures configuration for history pseudostate restoration.

FORMAL DEFINITION [UML 2.5, Section 14.5.5]:
History snapshot H = (composite, type, σ_saved, t) where:
  - composite: the composite state owning this history
  - type ∈ {SHALLOW, DEEP}: history variant
  - σ_saved: saved configuration
  - t: capture timestamp

RESTORATION SEMANTICS:
- SHALLOW: restore(H) = {s ∈ σ_saved | parent(s) = composite}
- DEEP: restore(H*) = σ_saved ∩ descendants*(composite)




| Field | Type | Description |
| ----- | ---- | ----------- |
| composite_state |string| Composite state this history belongs to   |
| type |[HistoryType](./statecharts.md#statecharts-v1-HistoryType)| History type: H (shallow) or H* (deep)   |
| saved_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)| Saved configuration: σ_saved ⊆ P(S)   |
| captured_at |Timestamp| When this snapshot was captured   |
| exit_transition |string| Transition that caused exit (for debugging)   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MachineHistoryState"></a>

### MachineHistoryState

MachineHistoryState aggregates all history snapshots for a machine.




| Field | Type | Description |
| ----- | ---- | ----------- |
| snapshots |[MachineHistoryState.SnapshotsEntry](#statecharts-v1-MachineHistoryState-SnapshotsEntry)| History snapshots keyed by composite state label   |






<a name="statecharts-v1-MachineHistoryState-SnapshotsEntry"></a>

### SnapshotsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[HistorySnapshot](#statecharts-v1-HistorySnapshot)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Checkpoint"></a>

### Checkpoint

Checkpoint captures complete machine state for recovery or replay.

FORMAL DEFINITION:
Checkpoint C = (id, M, H, seq, t, V, checksum) where:
  - id: unique checkpoint identifier
  - M: complete machine state
  - H: history snapshots
  - seq: sequence number of last applied log entry
  - t: checkpoint creation time
  - V: chart version
  - checksum: integrity verification hash

RECOVERY SEMANTICS:
To recover to sequence n from checkpoint C with seq(C) < n:
  1. Load machine state M from C
  2. Replay log entries [seq(C)+1, n]
  3. Verify final state matches expected




| Field | Type | Description |
| ----- | ---- | ----------- |
| checkpoint_id |string|   |
| machine_state |[Machine](./statecharts.md#statecharts-v1-Machine)|   |
| history_state |[MachineHistoryState](#statecharts-v1-MachineHistoryState)|   |
| last_sequence |uint64|   |
| created_at |Timestamp|   |
| chart_version |[ChartVersion](#statecharts-v1-ChartVersion)|   |
| checksum |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ReplayRequest"></a>

### ReplayRequest

ReplayRequest specifies parameters for deterministic replay.




| Field | Type | Description |
| ----- | ---- | ----------- |
| checkpoint |[Checkpoint](#statecharts-v1-Checkpoint)| Starting checkpoint   |
| events[] |[Event](./statecharts.md#statecharts-v1-Event)| Events to replay   |
| options |[ReplayOptions](#statecharts-v1-ReplayOptions)| Replay options   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ReplayOptions"></a>

### ReplayOptions

ReplayOptions controls replay behavior.




| Field | Type | Description |
| ----- | ---- | ----------- |
| stop_at_sequence |uint64| Stop conditions   |
| stop_at_time |Timestamp|   |
| stop_at_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)|   |
| verbose |bool| Debugging options   |
| strict_mode |bool|   |
| speed |double| Replay speed: 0 = max, 1.0 = realtime   |
| step_mode |bool| Step mode: pause before each transition   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ReplayResult"></a>

### ReplayResult

ReplayResult contains the outcome of a replay operation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| final_state |[Machine](./statecharts.md#statecharts-v1-Machine)|   |
| trace |[ExecutionTrace](#statecharts-v1-ExecutionTrace)|   |
| divergences[] |[TraceDivergence](#statecharts-v1-TraceDivergence)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TraceDiff"></a>

### TraceDiff

TraceDiff compares two execution traces to find behavioral differences.

FORMAL DEFINITION:
Trace diff D(τ₁, τ₂) = (mode, summary, divergences) where:
  - mode: comparison semantics
  - summary: high-level comparison result
  - divergences: detailed list of differences

COMPARISON MODES:
  - SYNTACTIC: exact entry-by-entry match
  - SEMANTIC: same final state, different paths OK
  - BEHAVIORAL: same observable outputs




| Field | Type | Description |
| ----- | ---- | ----------- |
| trace_a_id |string|   |
| trace_b_id |string|   |
| summary |[TraceDiffSummary](#statecharts-v1-TraceDiffSummary)|   |
| divergences[] |[TraceDivergence](#statecharts-v1-TraceDivergence)|   |
| mode |[TraceDiffMode](#statecharts-v1-TraceDiffMode)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TraceDiffSummary"></a>

### TraceDiffSummary

TraceDiffSummary provides high-level comparison results.




| Field | Type | Description |
| ----- | ---- | ----------- |
| identical |bool|   |
| common_prefix_length |uint64|   |
| divergence_count |uint64|   |
| first_divergence_event |string|   |
| semantically_equivalent |bool| Semantic equivalence fields   |
| same_final_state |bool|   |
| same_observable_outputs |bool|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TraceDivergence"></a>

### TraceDivergence

TraceDivergence describes a specific difference between traces.




| Field | Type | Description |
| ----- | ---- | ----------- |
| sequence |uint64|   |
| type |[DivergenceType](#statecharts-v1-DivergenceType)|   |
| description |string|   |
| value_a |Any|   |
| value_b |Any|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StorageConfig"></a>

### StorageConfig

StorageConfig defines persistence and replication settings.

FORMAL DEFINITION:
Storage config S = (backend, checkpoint, log, compaction, replication)
specifies the durable storage strategy for execution history.

DURABILITY GUARANTEES:
The combination of checkpoints and logs provides:
  - Atomicity: each entry is fully written or not at all
  - Consistency: log entries maintain causal ordering
  - Durability: sync mode determines persistence guarantees




| Field | Type | Description |
| ----- | ---- | ----------- |
| backend |[StorageBackend](#statecharts-v1-StorageBackend)|   |
| checkpoint |[CheckpointConfig](#statecharts-v1-CheckpointConfig)|   |
| log |[LogConfig](#statecharts-v1-LogConfig)|   |
| compaction |[CompactionConfig](#statecharts-v1-CompactionConfig)|   |
| replication |[ReplicationConfig](#statecharts-v1-ReplicationConfig)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CheckpointConfig"></a>

### CheckpointConfig

CheckpointConfig controls when checkpoints are created.




| Field | Type | Description |
| ----- | ---- | ----------- |
| trigger |[CheckpointTrigger](#statecharts-v1-CheckpointTrigger)|   |
| max_checkpoints |uint32|   |
| compression |[CompressionType](#statecharts-v1-CompressionType)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CheckpointTrigger"></a>

### CheckpointTrigger

CheckpointTrigger specifies conditions for checkpoint creation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| every_n_entries |uint64|   |
| every_duration |Duration|   |
| log_size_threshold_bytes |uint64|   |
| adaptive |[AdaptiveCheckpointConfig](#statecharts-v1-AdaptiveCheckpointConfig)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-AdaptiveCheckpointConfig"></a>

### AdaptiveCheckpointConfig

AdaptiveCheckpointConfig enables SLO-driven checkpoint frequency.

FORMAL DEFINITION:
Adaptive checkpointing targets recovery time objective (RTO):
  checkpoint_interval = RTO / avg_replay_rate

Under high load, checkpoints become more frequent to maintain RTO.




| Field | Type | Description |
| ----- | ---- | ----------- |
| enabled |bool|   |
| target_recovery_time |Duration|   |
| min_interval |Duration|   |
| max_interval |Duration|   |
| high_activity_threshold |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-LogConfig"></a>

### LogConfig

LogConfig controls write-ahead log behavior.




| Field | Type | Description |
| ----- | ---- | ----------- |
| max_entries |uint64|   |
| max_size_bytes |uint64|   |
| sync_mode |[SyncMode](#statecharts-v1-SyncMode)|   |
| compression |[CompressionType](#statecharts-v1-CompressionType)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CompactionConfig"></a>

### CompactionConfig

CompactionConfig controls log compaction and retention.




| Field | Type | Description |
| ----- | ---- | ----------- |
| strategy |[CompactionStrategy](#statecharts-v1-CompactionStrategy)|   |
| retention |[RetentionPolicy](#statecharts-v1-RetentionPolicy)|   |
| schedule |[CompactionSchedule](#statecharts-v1-CompactionSchedule)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-RetentionPolicy"></a>

### RetentionPolicy

RetentionPolicy defines what data to keep and for how long.




| Field | Type | Description |
| ----- | ---- | ----------- |
| min_entries |uint64|   |
| min_checkpoints |uint32|   |
| max_age |Duration|   |
| keep_rules[] |[RetentionRule](#statecharts-v1-RetentionRule)|   |
| compliance |[ComplianceConfig](#statecharts-v1-ComplianceConfig)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-RetentionRule"></a>

### RetentionRule

RetentionRule keeps specific entries beyond normal retention.




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string|   |
| event_filter |string|   |
| retention |Duration|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ComplianceConfig"></a>

### ComplianceConfig

ComplianceConfig addresses regulatory requirements (GDPR, etc.).




| Field | Type | Description |
| ----- | ---- | ----------- |
| min_retention |Duration|   |
| max_retention |Duration|   |
| pii_fields[] |string|   |
| anonymization |[AnonymizationStrategy](#statecharts-v1-AnonymizationStrategy)|   |
| require_deletion_audit |bool|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CompactionSchedule"></a>

### CompactionSchedule

CompactionSchedule controls when compaction runs.




| Field | Type | Description |
| ----- | ---- | ----------- |
| interval |Duration|   |
| cron |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ReplicationConfig"></a>

### ReplicationConfig

ReplicationConfig for distributed deployments.




| Field | Type | Description |
| ----- | ---- | ----------- |
| mode |[ReplicationMode](#statecharts-v1-ReplicationMode)|   |
| replica_count |uint32|   |
| read_consistency |[ConsistencyLevel](#statecharts-v1-ConsistencyLevel)|   |
| write_consistency |[ConsistencyLevel](#statecharts-v1-ConsistencyLevel)|   |
| replicas[] |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-v1-CompatibilityLevel"></a>

### CompatibilityLevel
CompatibilityLevel classifies changes between chart versions.

FORMAL DEFINITION:
Compatibility C ∈ {PATCH, MINOR, MAJOR} follows semver semantics:
  - PATCH: bug fixes, no behavioral change
  - MINOR: new features, backward compatible
  - MAJOR: breaking changes, migration required



| Name | Number | Description |
| ---- | ------ | ----------- |
| COMPATIBILITY_LEVEL_UNSPECIFIED | 0 |   |
| COMPATIBILITY_LEVEL_PATCH | 1 |   |
| COMPATIBILITY_LEVEL_MINOR | 2 |   |
| COMPATIBILITY_LEVEL_MAJOR | 3 |   |




<a name="statecharts-v1-SamplingStrategy"></a>

### SamplingStrategy
SamplingStrategy determines when sampling decisions are made.

FORMAL DEFINITION:
  - HEAD: decide at trace start (efficient, may miss interesting traces)
  - TAIL: decide at trace end (captures all interesting, more overhead)
  - ADAPTIVE: adjust rate based on system load



| Name | Number | Description |
| ---- | ------ | ----------- |
| SAMPLING_STRATEGY_UNSPECIFIED | 0 |   |
| SAMPLING_STRATEGY_HEAD | 1 |   |
| SAMPLING_STRATEGY_TAIL | 2 |   |
| SAMPLING_STRATEGY_ADAPTIVE | 3 |   |




<a name="statecharts-v1-TraceDiffMode"></a>

### TraceDiffMode
TraceDiffMode specifies comparison semantics.



| Name | Number | Description |
| ---- | ------ | ----------- |
| TRACE_DIFF_MODE_UNSPECIFIED | 0 |   |
| TRACE_DIFF_MODE_SYNTACTIC | 1 | SYNTACTIC: Exact match of all entries τ₁ ≡ τ₂ ⟺ ∀i: L₁ᵢ = L₂ᵢ   |
| TRACE_DIFF_MODE_SEMANTIC | 2 | SEMANTIC: Same final state, different paths allowed τ₁ ≈ τ₂ ⟺ σₙ(τ₁) = σₙ(τ₂) ∧ Γₙ(τ₁) = Γₙ(τ₂)   |
| TRACE_DIFF_MODE_BEHAVIORAL | 3 | BEHAVIORAL: Same observable behavior (events, outputs) τ₁ ∼ τ₂ ⟺ outputs(τ₁) = outputs(τ₂)   |




<a name="statecharts-v1-DivergenceType"></a>

### DivergenceType
DivergenceType classifies the nature of trace differences.



| Name | Number | Description |
| ---- | ------ | ----------- |
| DIVERGENCE_TYPE_UNSPECIFIED | 0 |   |
| DIVERGENCE_TYPE_DIFFERENT_TARGET | 1 |   |
| DIVERGENCE_TYPE_DIFFERENT_TRANSITIONS | 2 |   |
| DIVERGENCE_TYPE_GUARD_DIFFERENCE | 3 |   |
| DIVERGENCE_TYPE_ACTION_DIFFERENCE | 4 |   |
| DIVERGENCE_TYPE_MISSING_ENTRY | 5 |   |
| DIVERGENCE_TYPE_EXTRA_ENTRY | 6 |   |
| DIVERGENCE_TYPE_CONTEXT_DIFFERENCE | 7 |   |




<a name="statecharts-v1-StorageBackend"></a>

### StorageBackend
StorageBackend enumerates supported storage implementations.



| Name | Number | Description |
| ---- | ------ | ----------- |
| STORAGE_BACKEND_UNSPECIFIED | 0 |   |
| STORAGE_BACKEND_MEMORY | 1 |   |
| STORAGE_BACKEND_FILE | 2 |   |
| STORAGE_BACKEND_SQLITE | 3 |   |
| STORAGE_BACKEND_POSTGRES | 4 |   |
| STORAGE_BACKEND_S3 | 5 |   |
| STORAGE_BACKEND_CUSTOM | 6 |   |




<a name="statecharts-v1-SyncMode"></a>

### SyncMode
SyncMode determines durability vs performance tradeoff.



| Name | Number | Description |
| ---- | ------ | ----------- |
| SYNC_MODE_UNSPECIFIED | 0 |   |
| SYNC_MODE_NONE | 1 |   |
| SYNC_MODE_BATCH | 2 |   |
| SYNC_MODE_EVERY_ENTRY | 3 |   |




<a name="statecharts-v1-CompressionType"></a>

### CompressionType
CompressionType for log and checkpoint data.



| Name | Number | Description |
| ---- | ------ | ----------- |
| COMPRESSION_TYPE_UNSPECIFIED | 0 |   |
| COMPRESSION_TYPE_NONE | 1 |   |
| COMPRESSION_TYPE_GZIP | 2 |   |
| COMPRESSION_TYPE_ZSTD | 3 |   |
| COMPRESSION_TYPE_LZ4 | 4 |   |




<a name="statecharts-v1-CompactionStrategy"></a>

### CompactionStrategy
CompactionStrategy determines how old entries are consolidated.



| Name | Number | Description |
| ---- | ------ | ----------- |
| COMPACTION_STRATEGY_UNSPECIFIED | 0 |   |
| COMPACTION_STRATEGY_SNAPSHOT | 1 |   |
| COMPACTION_STRATEGY_TIERED | 2 |   |
| COMPACTION_STRATEGY_WINDOW | 3 |   |




<a name="statecharts-v1-AnonymizationStrategy"></a>

### AnonymizationStrategy
AnonymizationStrategy for PII data handling.



| Name | Number | Description |
| ---- | ------ | ----------- |
| ANONYMIZATION_STRATEGY_UNSPECIFIED | 0 |   |
| ANONYMIZATION_STRATEGY_DELETE | 1 |   |
| ANONYMIZATION_STRATEGY_HASH | 2 |   |
| ANONYMIZATION_STRATEGY_PSEUDONYMIZE | 3 |   |
| ANONYMIZATION_STRATEGY_GENERALIZE | 4 |   |




<a name="statecharts-v1-ReplicationMode"></a>

### ReplicationMode
ReplicationMode determines replication topology.



| Name | Number | Description |
| ---- | ------ | ----------- |
| REPLICATION_MODE_UNSPECIFIED | 0 |   |
| REPLICATION_MODE_NONE | 1 |   |
| REPLICATION_MODE_PRIMARY_BACKUP | 2 |   |
| REPLICATION_MODE_MULTI_PRIMARY | 3 |   |
| REPLICATION_MODE_RAFT | 4 |   |




<a name="statecharts-v1-ConsistencyLevel"></a>

### ConsistencyLevel
ConsistencyLevel for distributed reads/writes.



| Name | Number | Description |
| ---- | ------ | ----------- |
| CONSISTENCY_LEVEL_UNSPECIFIED | 0 |   |
| CONSISTENCY_LEVEL_ONE | 1 |   |
| CONSISTENCY_LEVEL_QUORUM | 2 |   |
| CONSISTENCY_LEVEL_ALL | 3 |   |
| CONSISTENCY_LEVEL_LOCAL | 4 |   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

