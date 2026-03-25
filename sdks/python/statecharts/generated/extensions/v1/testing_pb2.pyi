import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AssertionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ASSERTION_TYPE_UNSPECIFIED: _ClassVar[AssertionType]
    ASSERTION_TYPE_STATE_ACTIVE: _ClassVar[AssertionType]
    ASSERTION_TYPE_STATE_NOT_ACTIVE: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONFIGURATION_EQUALS: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONFIGURATION_CONTAINS: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONFIGURATION_SUBSET: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONTEXT_EQUALS: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONTEXT_MATCHES: _ClassVar[AssertionType]
    ASSERTION_TYPE_CONTEXT_EXISTS: _ClassVar[AssertionType]
    ASSERTION_TYPE_ACTION_EXECUTED: _ClassVar[AssertionType]
    ASSERTION_TYPE_ACTION_NOT_EXECUTED: _ClassVar[AssertionType]
    ASSERTION_TYPE_EVENT_EMITTED: _ClassVar[AssertionType]
    ASSERTION_TYPE_TRANSITION_TAKEN: _ClassVar[AssertionType]
    ASSERTION_TYPE_EXPRESSION: _ClassVar[AssertionType]
ASSERTION_TYPE_UNSPECIFIED: AssertionType
ASSERTION_TYPE_STATE_ACTIVE: AssertionType
ASSERTION_TYPE_STATE_NOT_ACTIVE: AssertionType
ASSERTION_TYPE_CONFIGURATION_EQUALS: AssertionType
ASSERTION_TYPE_CONFIGURATION_CONTAINS: AssertionType
ASSERTION_TYPE_CONFIGURATION_SUBSET: AssertionType
ASSERTION_TYPE_CONTEXT_EQUALS: AssertionType
ASSERTION_TYPE_CONTEXT_MATCHES: AssertionType
ASSERTION_TYPE_CONTEXT_EXISTS: AssertionType
ASSERTION_TYPE_ACTION_EXECUTED: AssertionType
ASSERTION_TYPE_ACTION_NOT_EXECUTED: AssertionType
ASSERTION_TYPE_EVENT_EMITTED: AssertionType
ASSERTION_TYPE_TRANSITION_TAKEN: AssertionType
ASSERTION_TYPE_EXPRESSION: AssertionType

class TestSuite(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TEST_CASES_FIELD_NUMBER: _ClassVar[int]
    SETUP_FIELD_NUMBER: _ClassVar[int]
    TEARDOWN_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_FIELD_NUMBER: _ClassVar[int]
    MAX_PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    FAIL_FAST_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    test_cases: _containers.RepeatedCompositeFieldContainer[TestCase]
    setup: _containers.RepeatedCompositeFieldContainer[TestAction]
    teardown: _containers.RepeatedCompositeFieldContainer[TestAction]
    tags: _containers.RepeatedScalarFieldContainer[str]
    timeout: _duration_pb2.Duration
    parallel: bool
    max_parallelism: int
    fail_fast: bool
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., test_cases: _Optional[_Iterable[_Union[TestCase, _Mapping]]] = ..., setup: _Optional[_Iterable[_Union[TestAction, _Mapping]]] = ..., teardown: _Optional[_Iterable[_Union[TestAction, _Mapping]]] = ..., tags: _Optional[_Iterable[str]] = ..., timeout: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., parallel: _Optional[bool] = ..., max_parallelism: _Optional[int] = ..., fail_fast: _Optional[bool] = ...) -> None: ...

class TestCase(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    INITIAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    INITIAL_STATES_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    ASSERTIONS_FIELD_NUMBER: _ClassVar[int]
    INVARIANTS_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    SKIP_REASON_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    KNOWN_ISSUES_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    initial_context: _struct_pb2.Struct
    initial_states: _containers.RepeatedScalarFieldContainer[str]
    events: _containers.RepeatedCompositeFieldContainer[TestEvent]
    assertions: _containers.RepeatedCompositeFieldContainer[TestAssertion]
    invariants: _containers.RepeatedCompositeFieldContainer[TestInvariant]
    enabled: bool
    skip_reason: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    timeout: _duration_pb2.Duration
    priority: int
    expected_outcome: str
    known_issues: _containers.RepeatedScalarFieldContainer[str]
    author: str
    requirements: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., initial_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., initial_states: _Optional[_Iterable[str]] = ..., events: _Optional[_Iterable[_Union[TestEvent, _Mapping]]] = ..., assertions: _Optional[_Iterable[_Union[TestAssertion, _Mapping]]] = ..., invariants: _Optional[_Iterable[_Union[TestInvariant, _Mapping]]] = ..., enabled: _Optional[bool] = ..., skip_reason: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., timeout: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., priority: _Optional[int] = ..., expected_outcome: _Optional[str] = ..., known_issues: _Optional[_Iterable[str]] = ..., author: _Optional[str] = ..., requirements: _Optional[_Iterable[str]] = ...) -> None: ...

class TestEvent(_message.Message):
    __slots__ = ()
    EVENT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DELAY_BEFORE_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    ASSERTIONS_FIELD_NUMBER: _ClassVar[int]
    event: str
    payload: _struct_pb2.Struct
    delay_before: _duration_pb2.Duration
    comment: str
    assertions: _containers.RepeatedCompositeFieldContainer[TestAssertion]
    def __init__(self, event: _Optional[str] = ..., payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., delay_before: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., comment: _Optional[str] = ..., assertions: _Optional[_Iterable[_Union[TestAssertion, _Mapping]]] = ...) -> None: ...

class TestAction(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    type: str
    params: _struct_pb2.Struct
    comment: str
    def __init__(self, type: _Optional[str] = ..., params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., comment: _Optional[str] = ...) -> None: ...

class TestAssertion(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    AFTER_EVENT_INDEX_FIELD_NUMBER: _ClassVar[int]
    NEGATED_FIELD_NUMBER: _ClassVar[int]
    type: AssertionType
    expected: str
    field: str
    message: str
    timing: str
    after_event_index: int
    negated: bool
    def __init__(self, type: _Optional[_Union[AssertionType, str]] = ..., expected: _Optional[str] = ..., field: _Optional[str] = ..., message: _Optional[str] = ..., timing: _Optional[str] = ..., after_event_index: _Optional[int] = ..., negated: _Optional[bool] = ...) -> None: ...

class TestInvariant(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    VIOLATION_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    expression: str
    violation_message: str
    severity: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., expression: _Optional[str] = ..., violation_message: _Optional[str] = ..., severity: _Optional[str] = ...) -> None: ...

class StateCoverageMarker(_message.Message):
    __slots__ = ()
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    MIN_DWELL_FIELD_NUMBER: _ClassVar[int]
    COVERED_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXIT_COUNT_FIELD_NUMBER: _ClassVar[int]
    COVERED_BY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DWELL_TIME_FIELD_NUMBER: _ClassVar[int]
    requirements: _containers.RepeatedScalarFieldContainer[str]
    min_dwell: _duration_pb2.Duration
    covered: bool
    coverage_count: int
    entry_count: int
    exit_count: int
    covered_by: _containers.RepeatedScalarFieldContainer[str]
    total_dwell_time: _duration_pb2.Duration
    def __init__(self, requirements: _Optional[_Iterable[str]] = ..., min_dwell: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., covered: _Optional[bool] = ..., coverage_count: _Optional[int] = ..., entry_count: _Optional[int] = ..., exit_count: _Optional[int] = ..., covered_by: _Optional[_Iterable[str]] = ..., total_dwell_time: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class TransitionCoverageMarker(_message.Message):
    __slots__ = ()
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    COVERED_FIELD_NUMBER: _ClassVar[int]
    FIRE_COUNT_FIELD_NUMBER: _ClassVar[int]
    COVERED_BY_FIELD_NUMBER: _ClassVar[int]
    GUARD_TRUE_COVERED_FIELD_NUMBER: _ClassVar[int]
    GUARD_FALSE_COVERED_FIELD_NUMBER: _ClassVar[int]
    GUARD_TRUE_COUNT_FIELD_NUMBER: _ClassVar[int]
    GUARD_FALSE_COUNT_FIELD_NUMBER: _ClassVar[int]
    requirements: _containers.RepeatedScalarFieldContainer[str]
    covered: bool
    fire_count: int
    covered_by: _containers.RepeatedScalarFieldContainer[str]
    guard_true_covered: bool
    guard_false_covered: bool
    guard_true_count: int
    guard_false_count: int
    def __init__(self, requirements: _Optional[_Iterable[str]] = ..., covered: _Optional[bool] = ..., fire_count: _Optional[int] = ..., covered_by: _Optional[_Iterable[str]] = ..., guard_true_covered: _Optional[bool] = ..., guard_false_covered: _Optional[bool] = ..., guard_true_count: _Optional[int] = ..., guard_false_count: _Optional[int] = ...) -> None: ...

class CoverageReport(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    GENERATED_AT_FIELD_NUMBER: _ClassVar[int]
    STATECHART_ID_FIELD_NUMBER: _ClassVar[int]
    TEST_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    EVENT_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    GUARD_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    UNCOVERED_STATES_FIELD_NUMBER: _ClassVar[int]
    UNCOVERED_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    generated_at: _timestamp_pb2.Timestamp
    statechart_id: str
    test_run_id: str
    state_coverage: CoverageStats
    transition_coverage: CoverageStats
    event_coverage: CoverageStats
    guard_coverage: CoverageStats
    uncovered_states: _containers.RepeatedScalarFieldContainer[str]
    uncovered_transitions: _containers.RepeatedScalarFieldContainer[str]
    suggestions: _containers.RepeatedCompositeFieldContainer[SuggestedTestCase]
    def __init__(self, id: _Optional[str] = ..., generated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., statechart_id: _Optional[str] = ..., test_run_id: _Optional[str] = ..., state_coverage: _Optional[_Union[CoverageStats, _Mapping]] = ..., transition_coverage: _Optional[_Union[CoverageStats, _Mapping]] = ..., event_coverage: _Optional[_Union[CoverageStats, _Mapping]] = ..., guard_coverage: _Optional[_Union[CoverageStats, _Mapping]] = ..., uncovered_states: _Optional[_Iterable[str]] = ..., uncovered_transitions: _Optional[_Iterable[str]] = ..., suggestions: _Optional[_Iterable[_Union[SuggestedTestCase, _Mapping]]] = ...) -> None: ...

class CoverageStats(_message.Message):
    __slots__ = ()
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    COVERED_FIELD_NUMBER: _ClassVar[int]
    PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENT_MET_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    total: int
    covered: int
    percentage: float
    requirement_met: bool
    required_percentage: float
    def __init__(self, total: _Optional[int] = ..., covered: _Optional[int] = ..., percentage: _Optional[float] = ..., requirement_met: _Optional[bool] = ..., required_percentage: _Optional[float] = ...) -> None: ...

class SuggestedTestCase(_message.Message):
    __slots__ = ()
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    EVENT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    VIA_STATES_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    target: str
    target_type: str
    event_sequence: _containers.RepeatedScalarFieldContainer[str]
    via_states: _containers.RepeatedScalarFieldContainer[str]
    confidence: float
    def __init__(self, target: _Optional[str] = ..., target_type: _Optional[str] = ..., event_sequence: _Optional[_Iterable[str]] = ..., via_states: _Optional[_Iterable[str]] = ..., confidence: _Optional[float] = ...) -> None: ...

class ExpectedTrace(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    STRICT_ORDER_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    steps: _containers.RepeatedCompositeFieldContainer[TraceStep]
    strict_order: bool
    complete: bool
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., steps: _Optional[_Iterable[_Union[TraceStep, _Mapping]]] = ..., strict_order: _Optional[bool] = ..., complete: _Optional[bool] = ...) -> None: ...

class TraceStep(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MAX_TIME_SINCE_START_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    type: str
    target: str
    event: str
    context: _struct_pb2.Struct
    max_time_since_start: _duration_pb2.Duration
    required: bool
    def __init__(self, type: _Optional[str] = ..., target: _Optional[str] = ..., event: _Optional[str] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., max_time_since_start: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., required: _Optional[bool] = ...) -> None: ...

class TestResult(_message.Message):
    __slots__ = ()
    TEST_CASE_ID_FIELD_NUMBER: _ClassVar[int]
    TEST_CASE_NAME_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    EXECUTED_AT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_TRACE_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    FINAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    STATES_COVERED_FIELD_NUMBER: _ClassVar[int]
    TRANSITIONS_COVERED_FIELD_NUMBER: _ClassVar[int]
    test_case_id: str
    test_case_name: str
    outcome: str
    duration: _duration_pb2.Duration
    executed_at: _timestamp_pb2.Timestamp
    failure: TestFailure
    actual_trace: _containers.RepeatedCompositeFieldContainer[TraceStep]
    final_configuration: _containers.RepeatedScalarFieldContainer[str]
    final_context: _struct_pb2.Struct
    states_covered: _containers.RepeatedScalarFieldContainer[str]
    transitions_covered: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, test_case_id: _Optional[str] = ..., test_case_name: _Optional[str] = ..., outcome: _Optional[str] = ..., duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., executed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., failure: _Optional[_Union[TestFailure, _Mapping]] = ..., actual_trace: _Optional[_Iterable[_Union[TraceStep, _Mapping]]] = ..., final_configuration: _Optional[_Iterable[str]] = ..., final_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., states_covered: _Optional[_Iterable[str]] = ..., transitions_covered: _Optional[_Iterable[str]] = ...) -> None: ...

class TestFailure(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FAILED_ASSERTION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STACK_TRACE_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    AT_STEP_FIELD_NUMBER: _ClassVar[int]
    AT_EVENT_FIELD_NUMBER: _ClassVar[int]
    type: str
    failed_assertion: TestAssertion
    message: str
    stack_trace: str
    expected: str
    actual: str
    at_step: int
    at_event: str
    def __init__(self, type: _Optional[str] = ..., failed_assertion: _Optional[_Union[TestAssertion, _Mapping]] = ..., message: _Optional[str] = ..., stack_trace: _Optional[str] = ..., expected: _Optional[str] = ..., actual: _Optional[str] = ..., at_step: _Optional[int] = ..., at_event: _Optional[str] = ...) -> None: ...
