import datetime

from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from statecharts.v1 import statecharts_pb2 as _statecharts_pb2
from statecharts.v1 import expressions_pb2 as _expressions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CompatibilityLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPATIBILITY_LEVEL_UNSPECIFIED: _ClassVar[CompatibilityLevel]
    COMPATIBILITY_LEVEL_PATCH: _ClassVar[CompatibilityLevel]
    COMPATIBILITY_LEVEL_MINOR: _ClassVar[CompatibilityLevel]
    COMPATIBILITY_LEVEL_MAJOR: _ClassVar[CompatibilityLevel]

class SamplingStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SAMPLING_STRATEGY_UNSPECIFIED: _ClassVar[SamplingStrategy]
    SAMPLING_STRATEGY_HEAD: _ClassVar[SamplingStrategy]
    SAMPLING_STRATEGY_TAIL: _ClassVar[SamplingStrategy]
    SAMPLING_STRATEGY_ADAPTIVE: _ClassVar[SamplingStrategy]

class TraceDiffMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACE_DIFF_MODE_UNSPECIFIED: _ClassVar[TraceDiffMode]
    TRACE_DIFF_MODE_SYNTACTIC: _ClassVar[TraceDiffMode]
    TRACE_DIFF_MODE_SEMANTIC: _ClassVar[TraceDiffMode]
    TRACE_DIFF_MODE_BEHAVIORAL: _ClassVar[TraceDiffMode]

class DivergenceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DIVERGENCE_TYPE_UNSPECIFIED: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_DIFFERENT_TARGET: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_DIFFERENT_TRANSITIONS: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_GUARD_DIFFERENCE: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_ACTION_DIFFERENCE: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_MISSING_ENTRY: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_EXTRA_ENTRY: _ClassVar[DivergenceType]
    DIVERGENCE_TYPE_CONTEXT_DIFFERENCE: _ClassVar[DivergenceType]

class StorageBackend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STORAGE_BACKEND_UNSPECIFIED: _ClassVar[StorageBackend]
    STORAGE_BACKEND_MEMORY: _ClassVar[StorageBackend]
    STORAGE_BACKEND_FILE: _ClassVar[StorageBackend]
    STORAGE_BACKEND_SQLITE: _ClassVar[StorageBackend]
    STORAGE_BACKEND_POSTGRES: _ClassVar[StorageBackend]
    STORAGE_BACKEND_S3: _ClassVar[StorageBackend]
    STORAGE_BACKEND_CUSTOM: _ClassVar[StorageBackend]

class SyncMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SYNC_MODE_UNSPECIFIED: _ClassVar[SyncMode]
    SYNC_MODE_NONE: _ClassVar[SyncMode]
    SYNC_MODE_BATCH: _ClassVar[SyncMode]
    SYNC_MODE_EVERY_ENTRY: _ClassVar[SyncMode]

class CompressionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPRESSION_TYPE_UNSPECIFIED: _ClassVar[CompressionType]
    COMPRESSION_TYPE_NONE: _ClassVar[CompressionType]
    COMPRESSION_TYPE_GZIP: _ClassVar[CompressionType]
    COMPRESSION_TYPE_ZSTD: _ClassVar[CompressionType]
    COMPRESSION_TYPE_LZ4: _ClassVar[CompressionType]

class CompactionStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPACTION_STRATEGY_UNSPECIFIED: _ClassVar[CompactionStrategy]
    COMPACTION_STRATEGY_SNAPSHOT: _ClassVar[CompactionStrategy]
    COMPACTION_STRATEGY_TIERED: _ClassVar[CompactionStrategy]
    COMPACTION_STRATEGY_WINDOW: _ClassVar[CompactionStrategy]

class AnonymizationStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANONYMIZATION_STRATEGY_UNSPECIFIED: _ClassVar[AnonymizationStrategy]
    ANONYMIZATION_STRATEGY_DELETE: _ClassVar[AnonymizationStrategy]
    ANONYMIZATION_STRATEGY_HASH: _ClassVar[AnonymizationStrategy]
    ANONYMIZATION_STRATEGY_PSEUDONYMIZE: _ClassVar[AnonymizationStrategy]
    ANONYMIZATION_STRATEGY_GENERALIZE: _ClassVar[AnonymizationStrategy]

class ReplicationMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REPLICATION_MODE_UNSPECIFIED: _ClassVar[ReplicationMode]
    REPLICATION_MODE_NONE: _ClassVar[ReplicationMode]
    REPLICATION_MODE_PRIMARY_BACKUP: _ClassVar[ReplicationMode]
    REPLICATION_MODE_MULTI_PRIMARY: _ClassVar[ReplicationMode]
    REPLICATION_MODE_RAFT: _ClassVar[ReplicationMode]

class ConsistencyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONSISTENCY_LEVEL_UNSPECIFIED: _ClassVar[ConsistencyLevel]
    CONSISTENCY_LEVEL_ONE: _ClassVar[ConsistencyLevel]
    CONSISTENCY_LEVEL_QUORUM: _ClassVar[ConsistencyLevel]
    CONSISTENCY_LEVEL_ALL: _ClassVar[ConsistencyLevel]
    CONSISTENCY_LEVEL_LOCAL: _ClassVar[ConsistencyLevel]
COMPATIBILITY_LEVEL_UNSPECIFIED: CompatibilityLevel
COMPATIBILITY_LEVEL_PATCH: CompatibilityLevel
COMPATIBILITY_LEVEL_MINOR: CompatibilityLevel
COMPATIBILITY_LEVEL_MAJOR: CompatibilityLevel
SAMPLING_STRATEGY_UNSPECIFIED: SamplingStrategy
SAMPLING_STRATEGY_HEAD: SamplingStrategy
SAMPLING_STRATEGY_TAIL: SamplingStrategy
SAMPLING_STRATEGY_ADAPTIVE: SamplingStrategy
TRACE_DIFF_MODE_UNSPECIFIED: TraceDiffMode
TRACE_DIFF_MODE_SYNTACTIC: TraceDiffMode
TRACE_DIFF_MODE_SEMANTIC: TraceDiffMode
TRACE_DIFF_MODE_BEHAVIORAL: TraceDiffMode
DIVERGENCE_TYPE_UNSPECIFIED: DivergenceType
DIVERGENCE_TYPE_DIFFERENT_TARGET: DivergenceType
DIVERGENCE_TYPE_DIFFERENT_TRANSITIONS: DivergenceType
DIVERGENCE_TYPE_GUARD_DIFFERENCE: DivergenceType
DIVERGENCE_TYPE_ACTION_DIFFERENCE: DivergenceType
DIVERGENCE_TYPE_MISSING_ENTRY: DivergenceType
DIVERGENCE_TYPE_EXTRA_ENTRY: DivergenceType
DIVERGENCE_TYPE_CONTEXT_DIFFERENCE: DivergenceType
STORAGE_BACKEND_UNSPECIFIED: StorageBackend
STORAGE_BACKEND_MEMORY: StorageBackend
STORAGE_BACKEND_FILE: StorageBackend
STORAGE_BACKEND_SQLITE: StorageBackend
STORAGE_BACKEND_POSTGRES: StorageBackend
STORAGE_BACKEND_S3: StorageBackend
STORAGE_BACKEND_CUSTOM: StorageBackend
SYNC_MODE_UNSPECIFIED: SyncMode
SYNC_MODE_NONE: SyncMode
SYNC_MODE_BATCH: SyncMode
SYNC_MODE_EVERY_ENTRY: SyncMode
COMPRESSION_TYPE_UNSPECIFIED: CompressionType
COMPRESSION_TYPE_NONE: CompressionType
COMPRESSION_TYPE_GZIP: CompressionType
COMPRESSION_TYPE_ZSTD: CompressionType
COMPRESSION_TYPE_LZ4: CompressionType
COMPACTION_STRATEGY_UNSPECIFIED: CompactionStrategy
COMPACTION_STRATEGY_SNAPSHOT: CompactionStrategy
COMPACTION_STRATEGY_TIERED: CompactionStrategy
COMPACTION_STRATEGY_WINDOW: CompactionStrategy
ANONYMIZATION_STRATEGY_UNSPECIFIED: AnonymizationStrategy
ANONYMIZATION_STRATEGY_DELETE: AnonymizationStrategy
ANONYMIZATION_STRATEGY_HASH: AnonymizationStrategy
ANONYMIZATION_STRATEGY_PSEUDONYMIZE: AnonymizationStrategy
ANONYMIZATION_STRATEGY_GENERALIZE: AnonymizationStrategy
REPLICATION_MODE_UNSPECIFIED: ReplicationMode
REPLICATION_MODE_NONE: ReplicationMode
REPLICATION_MODE_PRIMARY_BACKUP: ReplicationMode
REPLICATION_MODE_MULTI_PRIMARY: ReplicationMode
REPLICATION_MODE_RAFT: ReplicationMode
CONSISTENCY_LEVEL_UNSPECIFIED: ConsistencyLevel
CONSISTENCY_LEVEL_ONE: ConsistencyLevel
CONSISTENCY_LEVEL_QUORUM: ConsistencyLevel
CONSISTENCY_LEVEL_ALL: ConsistencyLevel
CONSISTENCY_LEVEL_LOCAL: ConsistencyLevel

class TransitionLogEntry(_message.Message):
    __slots__ = ()
    class MetadataEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_EVENT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TARGET_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TRANSITIONS_FIRED_FIELD_NUMBER: _ClassVar[int]
    GUARD_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_EXECUTED_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_BEFORE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_AFTER_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CAUSALITY_FIELD_NUMBER: _ClassVar[int]
    id: str
    timestamp: _timestamp_pb2.Timestamp
    sequence: int
    trigger_event: _statecharts_pb2.Event
    source_config: _statecharts_pb2.Configuration
    target_config: _statecharts_pb2.Configuration
    transitions_fired: _containers.RepeatedCompositeFieldContainer[TransitionRef]
    guard_results: _containers.RepeatedCompositeFieldContainer[GuardEvaluation]
    actions_executed: _containers.RepeatedCompositeFieldContainer[ActionExecution]
    context_before: _struct_pb2.Struct
    context_after: _struct_pb2.Struct
    processing_time: _duration_pb2.Duration
    error: str
    metadata: _containers.ScalarMap[str, str]
    causality: CausalityInfo
    def __init__(self, id: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., sequence: _Optional[int] = ..., trigger_event: _Optional[_Union[_statecharts_pb2.Event, _Mapping]] = ..., source_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., target_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., transitions_fired: _Optional[_Iterable[_Union[TransitionRef, _Mapping]]] = ..., guard_results: _Optional[_Iterable[_Union[GuardEvaluation, _Mapping]]] = ..., actions_executed: _Optional[_Iterable[_Union[ActionExecution, _Mapping]]] = ..., context_before: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., context_after: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., processing_time: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., error: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., causality: _Optional[_Union[CausalityInfo, _Mapping]] = ...) -> None: ...

class TransitionRef(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    label: str
    to: _containers.RepeatedScalarFieldContainer[str]
    event: str
    def __init__(self, label: _Optional[str] = ..., to: _Optional[_Iterable[str]] = ..., event: _Optional[str] = ..., **kwargs) -> None: ...

class GuardEvaluation(_message.Message):
    __slots__ = ()
    GUARD_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    BOUND_VALUES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    guard_expression: str
    result: bool
    bound_values: _struct_pb2.Struct
    error: str
    def __init__(self, guard_expression: _Optional[str] = ..., result: _Optional[bool] = ..., bound_values: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class ActionExecution(_message.Message):
    __slots__ = ()
    ACTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    action_name: str
    parameters: _struct_pb2.Struct
    duration: _duration_pb2.Duration
    error: str
    def __init__(self, action_name: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class CausalityInfo(_message.Message):
    __slots__ = ()
    class VectorClockEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    LAMPORT_CLOCK_FIELD_NUMBER: _ClassVar[int]
    VECTOR_CLOCK_FIELD_NUMBER: _ClassVar[int]
    CAUSED_BY_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    lamport_clock: int
    vector_clock: _containers.ScalarMap[str, int]
    caused_by: str
    node_id: str
    def __init__(self, lamport_clock: _Optional[int] = ..., vector_clock: _Optional[_Mapping[str, int]] = ..., caused_by: _Optional[str] = ..., node_id: _Optional[str] = ...) -> None: ...

class ExecutionTrace(_message.Message):
    __slots__ = ()
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    MACHINE_ID_FIELD_NUMBER: _ClassVar[int]
    CHART_VERSION_FIELD_NUMBER: _ClassVar[int]
    INITIAL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    INITIAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    machine_id: str
    chart_version: ChartVersion
    initial_config: _statecharts_pb2.Configuration
    initial_context: _struct_pb2.Struct
    entries: _containers.RepeatedCompositeFieldContainer[TransitionLogEntry]
    final_config: _statecharts_pb2.Configuration
    final_context: _struct_pb2.Struct
    metadata: TraceMetadata
    def __init__(self, trace_id: _Optional[str] = ..., machine_id: _Optional[str] = ..., chart_version: _Optional[_Union[ChartVersion, _Mapping]] = ..., initial_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., initial_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., entries: _Optional[_Iterable[_Union[TransitionLogEntry, _Mapping]]] = ..., final_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., final_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., metadata: _Optional[_Union[TraceMetadata, _Mapping]] = ...) -> None: ...

class ChartVersion(_message.Message):
    __slots__ = ()
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    CHANGELOG_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    version: str
    content_hash: str
    created_at: _timestamp_pb2.Timestamp
    previous_version: str
    changelog: str
    compatibility: CompatibilityLevel
    def __init__(self, version: _Optional[str] = ..., content_hash: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., previous_version: _Optional[str] = ..., changelog: _Optional[str] = ..., compatibility: _Optional[_Union[CompatibilityLevel, str]] = ...) -> None: ...

class TraceMetadata(_message.Message):
    __slots__ = ()
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENDED_AT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EVENTS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EVENTS_IGNORED_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    MACHINE_COUNT_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_FIELD_NUMBER: _ClassVar[int]
    started_at: _timestamp_pb2.Timestamp
    ended_at: _timestamp_pb2.Timestamp
    total_transitions: int
    total_events_processed: int
    total_events_ignored: int
    errors: _containers.RepeatedScalarFieldContainer[str]
    machine_count: int
    sampling: SamplingInfo
    def __init__(self, started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., ended_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., total_transitions: _Optional[int] = ..., total_events_processed: _Optional[int] = ..., total_events_ignored: _Optional[int] = ..., errors: _Optional[_Iterable[str]] = ..., machine_count: _Optional[int] = ..., sampling: _Optional[_Union[SamplingInfo, _Mapping]] = ...) -> None: ...

class TraceConfig(_message.Message):
    __slots__ = ()
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    ALWAYS_TRACE_EVENTS_FIELD_NUMBER: _ClassVar[int]
    ALWAYS_TRACE_STATES_FIELD_NUMBER: _ClassVar[int]
    ALWAYS_TRACE_ERRORS_FIELD_NUMBER: _ClassVar[int]
    WARMUP_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    sample_rate: float
    always_trace_events: _containers.RepeatedScalarFieldContainer[str]
    always_trace_states: _containers.RepeatedScalarFieldContainer[str]
    always_trace_errors: bool
    warmup_transitions: int
    strategy: SamplingStrategy
    def __init__(self, sample_rate: _Optional[float] = ..., always_trace_events: _Optional[_Iterable[str]] = ..., always_trace_states: _Optional[_Iterable[str]] = ..., always_trace_errors: _Optional[bool] = ..., warmup_transitions: _Optional[int] = ..., strategy: _Optional[_Union[SamplingStrategy, str]] = ...) -> None: ...

class SamplingInfo(_message.Message):
    __slots__ = ()
    WAS_SAMPLED_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_AT_CAPTURE_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_DECISION_REASON_FIELD_NUMBER: _ClassVar[int]
    was_sampled: bool
    sample_rate_at_capture: float
    sampling_decision_reason: str
    def __init__(self, was_sampled: _Optional[bool] = ..., sample_rate_at_capture: _Optional[float] = ..., sampling_decision_reason: _Optional[str] = ...) -> None: ...

class HistorySnapshot(_message.Message):
    __slots__ = ()
    COMPOSITE_STATE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SAVED_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CAPTURED_AT_FIELD_NUMBER: _ClassVar[int]
    EXIT_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    composite_state: str
    type: _statecharts_pb2.HistoryType
    saved_config: _statecharts_pb2.Configuration
    captured_at: _timestamp_pb2.Timestamp
    exit_transition: str
    def __init__(self, composite_state: _Optional[str] = ..., type: _Optional[_Union[_statecharts_pb2.HistoryType, str]] = ..., saved_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., captured_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., exit_transition: _Optional[str] = ...) -> None: ...

class MachineHistoryState(_message.Message):
    __slots__ = ()
    class SnapshotsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: HistorySnapshot
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[HistorySnapshot, _Mapping]] = ...) -> None: ...
    SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    snapshots: _containers.MessageMap[str, HistorySnapshot]
    def __init__(self, snapshots: _Optional[_Mapping[str, HistorySnapshot]] = ...) -> None: ...

class Checkpoint(_message.Message):
    __slots__ = ()
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    MACHINE_STATE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_STATE_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CHART_VERSION_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    checkpoint_id: str
    machine_state: _statecharts_pb2.Machine
    history_state: MachineHistoryState
    last_sequence: int
    created_at: _timestamp_pb2.Timestamp
    chart_version: ChartVersion
    checksum: str
    def __init__(self, checkpoint_id: _Optional[str] = ..., machine_state: _Optional[_Union[_statecharts_pb2.Machine, _Mapping]] = ..., history_state: _Optional[_Union[MachineHistoryState, _Mapping]] = ..., last_sequence: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., chart_version: _Optional[_Union[ChartVersion, _Mapping]] = ..., checksum: _Optional[str] = ...) -> None: ...

class ReplayRequest(_message.Message):
    __slots__ = ()
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    checkpoint: Checkpoint
    events: _containers.RepeatedCompositeFieldContainer[_statecharts_pb2.Event]
    options: ReplayOptions
    def __init__(self, checkpoint: _Optional[_Union[Checkpoint, _Mapping]] = ..., events: _Optional[_Iterable[_Union[_statecharts_pb2.Event, _Mapping]]] = ..., options: _Optional[_Union[ReplayOptions, _Mapping]] = ...) -> None: ...

class ReplayOptions(_message.Message):
    __slots__ = ()
    STOP_AT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    STOP_AT_TIME_FIELD_NUMBER: _ClassVar[int]
    STOP_AT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    VERBOSE_FIELD_NUMBER: _ClassVar[int]
    STRICT_MODE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    STEP_MODE_FIELD_NUMBER: _ClassVar[int]
    stop_at_sequence: int
    stop_at_time: _timestamp_pb2.Timestamp
    stop_at_config: _statecharts_pb2.Configuration
    verbose: bool
    strict_mode: bool
    speed: float
    step_mode: bool
    def __init__(self, stop_at_sequence: _Optional[int] = ..., stop_at_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., stop_at_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., verbose: _Optional[bool] = ..., strict_mode: _Optional[bool] = ..., speed: _Optional[float] = ..., step_mode: _Optional[bool] = ...) -> None: ...

class ReplayResult(_message.Message):
    __slots__ = ()
    FINAL_STATE_FIELD_NUMBER: _ClassVar[int]
    TRACE_FIELD_NUMBER: _ClassVar[int]
    DIVERGENCES_FIELD_NUMBER: _ClassVar[int]
    final_state: _statecharts_pb2.Machine
    trace: ExecutionTrace
    divergences: _containers.RepeatedCompositeFieldContainer[TraceDivergence]
    def __init__(self, final_state: _Optional[_Union[_statecharts_pb2.Machine, _Mapping]] = ..., trace: _Optional[_Union[ExecutionTrace, _Mapping]] = ..., divergences: _Optional[_Iterable[_Union[TraceDivergence, _Mapping]]] = ...) -> None: ...

class TraceDiff(_message.Message):
    __slots__ = ()
    TRACE_A_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_B_ID_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    DIVERGENCES_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    trace_a_id: str
    trace_b_id: str
    summary: TraceDiffSummary
    divergences: _containers.RepeatedCompositeFieldContainer[TraceDivergence]
    mode: TraceDiffMode
    def __init__(self, trace_a_id: _Optional[str] = ..., trace_b_id: _Optional[str] = ..., summary: _Optional[_Union[TraceDiffSummary, _Mapping]] = ..., divergences: _Optional[_Iterable[_Union[TraceDivergence, _Mapping]]] = ..., mode: _Optional[_Union[TraceDiffMode, str]] = ...) -> None: ...

class TraceDiffSummary(_message.Message):
    __slots__ = ()
    IDENTICAL_FIELD_NUMBER: _ClassVar[int]
    COMMON_PREFIX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    DIVERGENCE_COUNT_FIELD_NUMBER: _ClassVar[int]
    FIRST_DIVERGENCE_EVENT_FIELD_NUMBER: _ClassVar[int]
    SEMANTICALLY_EQUIVALENT_FIELD_NUMBER: _ClassVar[int]
    SAME_FINAL_STATE_FIELD_NUMBER: _ClassVar[int]
    SAME_OBSERVABLE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    identical: bool
    common_prefix_length: int
    divergence_count: int
    first_divergence_event: str
    semantically_equivalent: bool
    same_final_state: bool
    same_observable_outputs: bool
    def __init__(self, identical: _Optional[bool] = ..., common_prefix_length: _Optional[int] = ..., divergence_count: _Optional[int] = ..., first_divergence_event: _Optional[str] = ..., semantically_equivalent: _Optional[bool] = ..., same_final_state: _Optional[bool] = ..., same_observable_outputs: _Optional[bool] = ...) -> None: ...

class TraceDivergence(_message.Message):
    __slots__ = ()
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    VALUE_A_FIELD_NUMBER: _ClassVar[int]
    VALUE_B_FIELD_NUMBER: _ClassVar[int]
    sequence: int
    type: DivergenceType
    description: str
    value_a: _any_pb2.Any
    value_b: _any_pb2.Any
    def __init__(self, sequence: _Optional[int] = ..., type: _Optional[_Union[DivergenceType, str]] = ..., description: _Optional[str] = ..., value_a: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., value_b: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class StorageConfig(_message.Message):
    __slots__ = ()
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    COMPACTION_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FIELD_NUMBER: _ClassVar[int]
    backend: StorageBackend
    checkpoint: CheckpointConfig
    log: LogConfig
    compaction: CompactionConfig
    replication: ReplicationConfig
    def __init__(self, backend: _Optional[_Union[StorageBackend, str]] = ..., checkpoint: _Optional[_Union[CheckpointConfig, _Mapping]] = ..., log: _Optional[_Union[LogConfig, _Mapping]] = ..., compaction: _Optional[_Union[CompactionConfig, _Mapping]] = ..., replication: _Optional[_Union[ReplicationConfig, _Mapping]] = ...) -> None: ...

class CheckpointConfig(_message.Message):
    __slots__ = ()
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    MAX_CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    trigger: CheckpointTrigger
    max_checkpoints: int
    compression: CompressionType
    def __init__(self, trigger: _Optional[_Union[CheckpointTrigger, _Mapping]] = ..., max_checkpoints: _Optional[int] = ..., compression: _Optional[_Union[CompressionType, str]] = ...) -> None: ...

class CheckpointTrigger(_message.Message):
    __slots__ = ()
    EVERY_N_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    EVERY_DURATION_FIELD_NUMBER: _ClassVar[int]
    LOG_SIZE_THRESHOLD_BYTES_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_FIELD_NUMBER: _ClassVar[int]
    every_n_entries: int
    every_duration: _duration_pb2.Duration
    log_size_threshold_bytes: int
    adaptive: AdaptiveCheckpointConfig
    def __init__(self, every_n_entries: _Optional[int] = ..., every_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., log_size_threshold_bytes: _Optional[int] = ..., adaptive: _Optional[_Union[AdaptiveCheckpointConfig, _Mapping]] = ...) -> None: ...

class AdaptiveCheckpointConfig(_message.Message):
    __slots__ = ()
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    TARGET_RECOVERY_TIME_FIELD_NUMBER: _ClassVar[int]
    MIN_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    MAX_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    HIGH_ACTIVITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    target_recovery_time: _duration_pb2.Duration
    min_interval: _duration_pb2.Duration
    max_interval: _duration_pb2.Duration
    high_activity_threshold: float
    def __init__(self, enabled: _Optional[bool] = ..., target_recovery_time: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., min_interval: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max_interval: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., high_activity_threshold: _Optional[float] = ...) -> None: ...

class LogConfig(_message.Message):
    __slots__ = ()
    MAX_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    MAX_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SYNC_MODE_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    max_entries: int
    max_size_bytes: int
    sync_mode: SyncMode
    compression: CompressionType
    def __init__(self, max_entries: _Optional[int] = ..., max_size_bytes: _Optional[int] = ..., sync_mode: _Optional[_Union[SyncMode, str]] = ..., compression: _Optional[_Union[CompressionType, str]] = ...) -> None: ...

class CompactionConfig(_message.Message):
    __slots__ = ()
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    RETENTION_FIELD_NUMBER: _ClassVar[int]
    SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    strategy: CompactionStrategy
    retention: RetentionPolicy
    schedule: CompactionSchedule
    def __init__(self, strategy: _Optional[_Union[CompactionStrategy, str]] = ..., retention: _Optional[_Union[RetentionPolicy, _Mapping]] = ..., schedule: _Optional[_Union[CompactionSchedule, _Mapping]] = ...) -> None: ...

class RetentionPolicy(_message.Message):
    __slots__ = ()
    MIN_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    MIN_CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_FIELD_NUMBER: _ClassVar[int]
    KEEP_RULES_FIELD_NUMBER: _ClassVar[int]
    COMPLIANCE_FIELD_NUMBER: _ClassVar[int]
    min_entries: int
    min_checkpoints: int
    max_age: _duration_pb2.Duration
    keep_rules: _containers.RepeatedCompositeFieldContainer[RetentionRule]
    compliance: ComplianceConfig
    def __init__(self, min_entries: _Optional[int] = ..., min_checkpoints: _Optional[int] = ..., max_age: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., keep_rules: _Optional[_Iterable[_Union[RetentionRule, _Mapping]]] = ..., compliance: _Optional[_Union[ComplianceConfig, _Mapping]] = ...) -> None: ...

class RetentionRule(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    EVENT_FILTER_FIELD_NUMBER: _ClassVar[int]
    RETENTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    event_filter: str
    retention: _duration_pb2.Duration
    def __init__(self, name: _Optional[str] = ..., event_filter: _Optional[str] = ..., retention: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class ComplianceConfig(_message.Message):
    __slots__ = ()
    MIN_RETENTION_FIELD_NUMBER: _ClassVar[int]
    MAX_RETENTION_FIELD_NUMBER: _ClassVar[int]
    PII_FIELDS_FIELD_NUMBER: _ClassVar[int]
    ANONYMIZATION_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_DELETION_AUDIT_FIELD_NUMBER: _ClassVar[int]
    min_retention: _duration_pb2.Duration
    max_retention: _duration_pb2.Duration
    pii_fields: _containers.RepeatedScalarFieldContainer[str]
    anonymization: AnonymizationStrategy
    require_deletion_audit: bool
    def __init__(self, min_retention: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max_retention: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., pii_fields: _Optional[_Iterable[str]] = ..., anonymization: _Optional[_Union[AnonymizationStrategy, str]] = ..., require_deletion_audit: _Optional[bool] = ...) -> None: ...

class CompactionSchedule(_message.Message):
    __slots__ = ()
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    CRON_FIELD_NUMBER: _ClassVar[int]
    interval: _duration_pb2.Duration
    cron: str
    def __init__(self, interval: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., cron: _Optional[str] = ...) -> None: ...

class ReplicationConfig(_message.Message):
    __slots__ = ()
    MODE_FIELD_NUMBER: _ClassVar[int]
    REPLICA_COUNT_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    WRITE_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    mode: ReplicationMode
    replica_count: int
    read_consistency: ConsistencyLevel
    write_consistency: ConsistencyLevel
    replicas: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, mode: _Optional[_Union[ReplicationMode, str]] = ..., replica_count: _Optional[int] = ..., read_consistency: _Optional[_Union[ConsistencyLevel, str]] = ..., write_consistency: _Optional[_Union[ConsistencyLevel, str]] = ..., replicas: _Optional[_Iterable[str]] = ...) -> None: ...
