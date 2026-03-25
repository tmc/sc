import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StateSimConfig(_message.Message):
    __slots__ = ()
    TIMING_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    ENTRY_MOCK_FIELD_NUMBER: _ClassVar[int]
    EXIT_MOCK_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    TRACE_LEVEL_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    timing: TimingConstraints
    resources: ResourceProfile
    entry_mock: MockConfig
    exit_mock: MockConfig
    failure: FailureConfig
    trace_level: int
    cost: float
    reward: float
    priority: int
    def __init__(self, timing: _Optional[_Union[TimingConstraints, _Mapping]] = ..., resources: _Optional[_Union[ResourceProfile, _Mapping]] = ..., entry_mock: _Optional[_Union[MockConfig, _Mapping]] = ..., exit_mock: _Optional[_Union[MockConfig, _Mapping]] = ..., failure: _Optional[_Union[FailureConfig, _Mapping]] = ..., trace_level: _Optional[int] = ..., cost: _Optional[float] = ..., reward: _Optional[float] = ..., priority: _Optional[int] = ...) -> None: ...

class TimingConstraints(_message.Message):
    __slots__ = ()
    MIN_DWELL_FIELD_NUMBER: _ClassVar[int]
    MAX_DWELL_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_DWELL_FIELD_NUMBER: _ClassVar[int]
    DWELL_DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    ENTRY_DELAY_FIELD_NUMBER: _ClassVar[int]
    EXIT_DELAY_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_EVENT_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_EVENT_FIELD_NUMBER: _ClassVar[int]
    min_dwell: _duration_pb2.Duration
    max_dwell: _duration_pb2.Duration
    expected_dwell: _duration_pb2.Duration
    dwell_distribution: Distribution
    deadline: _duration_pb2.Duration
    entry_delay: DelayRange
    exit_delay: DelayRange
    timeout_event: str
    deadline_event: str
    def __init__(self, min_dwell: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max_dwell: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., expected_dwell: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., dwell_distribution: _Optional[_Union[Distribution, _Mapping]] = ..., deadline: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., entry_delay: _Optional[_Union[DelayRange, _Mapping]] = ..., exit_delay: _Optional[_Union[DelayRange, _Mapping]] = ..., timeout_event: _Optional[str] = ..., deadline_event: _Optional[str] = ...) -> None: ...

class TransitionSimConfig(_message.Message):
    __slots__ = ()
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DELAY_FIELD_NUMBER: _ClassVar[int]
    MOCK_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DELAY_FIELD_NUMBER: _ClassVar[int]
    weight: float
    delay: DelayRange
    mock: MockConfig
    failure: FailureConfig
    cost: float
    reward: float
    rate: float
    enable_delay: _duration_pb2.Duration
    def __init__(self, weight: _Optional[float] = ..., delay: _Optional[_Union[DelayRange, _Mapping]] = ..., mock: _Optional[_Union[MockConfig, _Mapping]] = ..., failure: _Optional[_Union[FailureConfig, _Mapping]] = ..., cost: _Optional[float] = ..., reward: _Optional[float] = ..., rate: _Optional[float] = ..., enable_delay: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class DelayRange(_message.Message):
    __slots__ = ()
    class ParamsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    min: _duration_pb2.Duration
    max: _duration_pb2.Duration
    distribution: str
    params: _containers.ScalarMap[str, float]
    def __init__(self, min: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., distribution: _Optional[str] = ..., params: _Optional[_Mapping[str, float]] = ...) -> None: ...

class Distribution(_message.Message):
    __slots__ = ()
    class ParamsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    type: str
    params: _containers.ScalarMap[str, float]
    min: float
    max: float
    def __init__(self, type: _Optional[str] = ..., params: _Optional[_Mapping[str, float]] = ..., min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class ResourceProfile(_message.Message):
    __slots__ = ()
    class CustomEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    CPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    NETWORK_BPS_FIELD_NUMBER: _ClassVar[int]
    DISK_IOPS_FIELD_NUMBER: _ClassVar[int]
    POWER_WATTS_FIELD_NUMBER: _ClassVar[int]
    COST_PER_HOUR_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_NUMBER: _ClassVar[int]
    CONSTANT_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    cpu_utilization: float
    memory_bytes: int
    network_bps: int
    disk_iops: int
    power_watts: float
    cost_per_hour: float
    custom: _containers.ScalarMap[str, float]
    constant: bool
    samples: _containers.RepeatedCompositeFieldContainer[ResourceSample]
    def __init__(self, cpu_utilization: _Optional[float] = ..., memory_bytes: _Optional[int] = ..., network_bps: _Optional[int] = ..., disk_iops: _Optional[int] = ..., power_watts: _Optional[float] = ..., cost_per_hour: _Optional[float] = ..., custom: _Optional[_Mapping[str, float]] = ..., constant: _Optional[bool] = ..., samples: _Optional[_Iterable[_Union[ResourceSample, _Mapping]]] = ...) -> None: ...

class ResourceSample(_message.Message):
    __slots__ = ()
    class ValuesEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    offset: _duration_pb2.Duration
    values: _containers.ScalarMap[str, float]
    def __init__(self, offset: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., values: _Optional[_Mapping[str, float]] = ...) -> None: ...

class MockConfig(_message.Message):
    __slots__ = ()
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DELAY_FIELD_NUMBER: _ClassVar[int]
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    FAILURE_RATE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RECORD_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    mode: str
    response_delay: DelayRange
    responses: _containers.RepeatedCompositeFieldContainer[MockResponse]
    failure_rate: float
    default_value: str
    default_error_code: str
    default_error_message: str
    record: bool
    def __init__(self, enabled: _Optional[bool] = ..., mode: _Optional[str] = ..., response_delay: _Optional[_Union[DelayRange, _Mapping]] = ..., responses: _Optional[_Iterable[_Union[MockResponse, _Mapping]]] = ..., failure_rate: _Optional[float] = ..., default_value: _Optional[str] = ..., default_error_code: _Optional[str] = ..., default_error_message: _Optional[str] = ..., record: _Optional[bool] = ...) -> None: ...

class MockResponse(_message.Message):
    __slots__ = ()
    VALUE_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DELAY_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    value: str
    is_error: bool
    error_code: str
    error_message: str
    delay: _duration_pb2.Duration
    condition: str
    def __init__(self, value: _Optional[str] = ..., is_error: _Optional[bool] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., delay: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., condition: _Optional[str] = ...) -> None: ...

class FailureConfig(_message.Message):
    __slots__ = ()
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    FAILURE_TYPE_FIELD_NUMBER: _ClassVar[int]
    FAILURE_RATE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_AFTER_N_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_CONDITION_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_AT_FIELD_NUMBER: _ClassVar[int]
    RECOVERY_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    RETRY_DELAY_FIELD_NUMBER: _ClassVar[int]
    FALLBACK_VALUE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CORRUPTION_FIELD_NUMBER: _ClassVar[int]
    ONLY_IN_STATES_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_STATES_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    failure_type: str
    failure_rate: float
    trigger: str
    trigger_after_n: int
    trigger_condition: str
    trigger_at: _duration_pb2.Duration
    recovery: str
    max_retries: int
    retry_delay: DelayRange
    fallback_value: str
    error_code: str
    error_message: str
    corruption: CorruptionConfig
    only_in_states: _containers.RepeatedScalarFieldContainer[str]
    exclude_states: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, enabled: _Optional[bool] = ..., failure_type: _Optional[str] = ..., failure_rate: _Optional[float] = ..., trigger: _Optional[str] = ..., trigger_after_n: _Optional[int] = ..., trigger_condition: _Optional[str] = ..., trigger_at: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., recovery: _Optional[str] = ..., max_retries: _Optional[int] = ..., retry_delay: _Optional[_Union[DelayRange, _Mapping]] = ..., fallback_value: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., corruption: _Optional[_Union[CorruptionConfig, _Mapping]] = ..., only_in_states: _Optional[_Iterable[str]] = ..., exclude_states: _Optional[_Iterable[str]] = ...) -> None: ...

class CorruptionConfig(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    type: str
    probability: float
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., probability: _Optional[float] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class SimulationScenario(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TIME_LIMIT_FIELD_NUMBER: _ClassVar[int]
    WALL_TIME_LIMIT_FIELD_NUMBER: _ClassVar[int]
    MAX_EVENTS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULED_EVENTS_FIELD_NUMBER: _ClassVar[int]
    RANDOM_EVENTS_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    RUN_COUNT_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    WARMUP_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_FAILURE_FIELD_NUMBER: _ClassVar[int]
    TIME_SCALE_FIELD_NUMBER: _ClassVar[int]
    INITIAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    time_limit: _duration_pb2.Duration
    wall_time_limit: _duration_pb2.Duration
    max_events: int
    scheduled_events: _containers.RepeatedCompositeFieldContainer[ScheduledEvent]
    random_events: RandomEventConfig
    constraints: ResourceConstraints
    random_seed: int
    run_count: int
    metrics: _containers.RepeatedScalarFieldContainer[str]
    warmup: _duration_pb2.Duration
    global_failure: FailureConfig
    time_scale: float
    initial_context: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., time_limit: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., wall_time_limit: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max_events: _Optional[int] = ..., scheduled_events: _Optional[_Iterable[_Union[ScheduledEvent, _Mapping]]] = ..., random_events: _Optional[_Union[RandomEventConfig, _Mapping]] = ..., constraints: _Optional[_Union[ResourceConstraints, _Mapping]] = ..., random_seed: _Optional[int] = ..., run_count: _Optional[int] = ..., metrics: _Optional[_Iterable[str]] = ..., warmup: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., global_failure: _Optional[_Union[FailureConfig, _Mapping]] = ..., time_scale: _Optional[float] = ..., initial_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ScheduledEvent(_message.Message):
    __slots__ = ()
    TIME_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    REPEAT_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    MAX_REPETITIONS_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    time: _duration_pb2.Duration
    event: str
    payload: str
    repeat: bool
    interval: _duration_pb2.Duration
    max_repetitions: int
    condition: str
    def __init__(self, time: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., event: _Optional[str] = ..., payload: _Optional[str] = ..., repeat: _Optional[bool] = ..., interval: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., max_repetitions: _Optional[int] = ..., condition: _Optional[str] = ...) -> None: ...

class RandomEventConfig(_message.Message):
    __slots__ = ()
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_PROCESS_FIELD_NUMBER: _ClassVar[int]
    BURST_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[RandomEvent]
    rate: float
    arrival_process: str
    burst: BurstConfig
    def __init__(self, events: _Optional[_Iterable[_Union[RandomEvent, _Mapping]]] = ..., rate: _Optional[float] = ..., arrival_process: _Optional[str] = ..., burst: _Optional[_Union[BurstConfig, _Mapping]] = ...) -> None: ...

class RandomEvent(_message.Message):
    __slots__ = ()
    EVENT_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_GENERATOR_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_PARAMS_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    event: str
    weight: float
    payload_generator: str
    generator_params: _struct_pb2.Struct
    condition: str
    def __init__(self, event: _Optional[str] = ..., weight: _Optional[float] = ..., payload_generator: _Optional[str] = ..., generator_params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., condition: _Optional[str] = ...) -> None: ...

class BurstConfig(_message.Message):
    __slots__ = ()
    BURST_SIZE_FIELD_NUMBER: _ClassVar[int]
    BURST_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    INTRA_BURST_DELAY_FIELD_NUMBER: _ClassVar[int]
    burst_size: int
    burst_interval: _duration_pb2.Duration
    intra_burst_delay: _duration_pb2.Duration
    def __init__(self, burst_size: _Optional[int] = ..., burst_interval: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., intra_burst_delay: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class ResourceConstraints(_message.Message):
    __slots__ = ()
    MAX_CONCURRENT_STATES_FIELD_NUMBER: _ClassVar[int]
    MAX_EVENT_RATE_FIELD_NUMBER: _ClassVar[int]
    MAX_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_CPU_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    OVERFLOW_ACTION_FIELD_NUMBER: _ClassVar[int]
    max_concurrent_states: int
    max_event_rate: float
    max_memory_bytes: int
    max_cpu: float
    max_queue_depth: int
    overflow_action: str
    def __init__(self, max_concurrent_states: _Optional[int] = ..., max_event_rate: _Optional[float] = ..., max_memory_bytes: _Optional[int] = ..., max_cpu: _Optional[float] = ..., max_queue_depth: _Optional[int] = ..., overflow_action: _Optional[str] = ...) -> None: ...

class SimulationResult(_message.Message):
    __slots__ = ()
    class MetricsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class ResourceUtilizationEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SCENARIO_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    WALL_DURATION_FIELD_NUMBER: _ClassVar[int]
    EVENTS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    FAILURES_INJECTED_FIELD_NUMBER: _ClassVar[int]
    FAILURES_RECOVERED_FIELD_NUMBER: _ClassVar[int]
    VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_REASON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    scenario_id: str
    run_number: int
    random_seed: int
    duration: _duration_pb2.Duration
    wall_duration: _duration_pb2.Duration
    events_processed: int
    final_configuration: _containers.RepeatedScalarFieldContainer[str]
    final_context: _struct_pb2.Struct
    metrics: _containers.ScalarMap[str, float]
    resource_utilization: _containers.ScalarMap[str, float]
    failures_injected: int
    failures_recovered: int
    violations: _containers.RepeatedScalarFieldContainer[str]
    termination_reason: str
    error: str
    def __init__(self, scenario_id: _Optional[str] = ..., run_number: _Optional[int] = ..., random_seed: _Optional[int] = ..., duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., wall_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., events_processed: _Optional[int] = ..., final_configuration: _Optional[_Iterable[str]] = ..., final_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., metrics: _Optional[_Mapping[str, float]] = ..., resource_utilization: _Optional[_Mapping[str, float]] = ..., failures_injected: _Optional[int] = ..., failures_recovered: _Optional[int] = ..., violations: _Optional[_Iterable[str]] = ..., termination_reason: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
