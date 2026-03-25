from google.protobuf import struct_pb2 as _struct_pb2
from google.rpc import status_pb2 as _status_pb2
from statecharts.v1 import statecharts_pb2 as _statecharts_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Severity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEVERITY_UNSPECIFIED: _ClassVar[Severity]
    INFO: _ClassVar[Severity]
    WARNING: _ClassVar[Severity]
    ERROR: _ClassVar[Severity]

class RuleId(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RULE_UNSPECIFIED: _ClassVar[RuleId]
    UNIQUE_STATE_LABELS: _ClassVar[RuleId]
    SINGLE_DEFAULT_CHILD: _ClassVar[RuleId]
    BASIC_HAS_NO_CHILDREN: _ClassVar[RuleId]
    COMPOUND_HAS_CHILDREN: _ClassVar[RuleId]
    DETERMINISTIC_TRANSITION_SELECTION: _ClassVar[RuleId]
    NO_EVENT_BROADCAST_CYCLES: _ClassVar[RuleId]
    HISTORY_STATES_WELL_FORMED: _ClassVar[RuleId]
    PSEUDO_STATES_WELL_FORMED: _ClassVar[RuleId]
    FORK_JOIN_BALANCED: _ClassVar[RuleId]
    CHOICE_GUARDS_COMPLETE: _ClassVar[RuleId]
    TIMEOUT_EVENTS_UNIQUE: _ClassVar[RuleId]
    ACTION_EXPRESSIONS_VALID: _ClassVar[RuleId]
    GUARD_EXPRESSIONS_VALID: _ClassVar[RuleId]
    EVENT_PARAMETERS_CONSISTENT: _ClassVar[RuleId]
    INTERNAL_TRANSITIONS_VALID: _ClassVar[RuleId]
    COMPLETION_TRANSITIONS_VALID: _ClassVar[RuleId]
    INVARIANTS_SATISFIABLE: _ClassVar[RuleId]
    HISTORY_DEFAULTS_VALID: _ClassVar[RuleId]
SEVERITY_UNSPECIFIED: Severity
INFO: Severity
WARNING: Severity
ERROR: Severity
RULE_UNSPECIFIED: RuleId
UNIQUE_STATE_LABELS: RuleId
SINGLE_DEFAULT_CHILD: RuleId
BASIC_HAS_NO_CHILDREN: RuleId
COMPOUND_HAS_CHILDREN: RuleId
DETERMINISTIC_TRANSITION_SELECTION: RuleId
NO_EVENT_BROADCAST_CYCLES: RuleId
HISTORY_STATES_WELL_FORMED: RuleId
PSEUDO_STATES_WELL_FORMED: RuleId
FORK_JOIN_BALANCED: RuleId
CHOICE_GUARDS_COMPLETE: RuleId
TIMEOUT_EVENTS_UNIQUE: RuleId
ACTION_EXPRESSIONS_VALID: RuleId
GUARD_EXPRESSIONS_VALID: RuleId
EVENT_PARAMETERS_CONSISTENT: RuleId
INTERNAL_TRANSITIONS_VALID: RuleId
COMPLETION_TRANSITIONS_VALID: RuleId
INVARIANTS_SATISFIABLE: RuleId
HISTORY_DEFAULTS_VALID: RuleId

class ValidateChartRequest(_message.Message):
    __slots__ = ()
    CHART_FIELD_NUMBER: _ClassVar[int]
    IGNORE_RULES_FIELD_NUMBER: _ClassVar[int]
    chart: _statecharts_pb2.Statechart
    ignore_rules: _containers.RepeatedScalarFieldContainer[RuleId]
    def __init__(self, chart: _Optional[_Union[_statecharts_pb2.Statechart, _Mapping]] = ..., ignore_rules: _Optional[_Iterable[_Union[RuleId, str]]] = ...) -> None: ...

class ValidateTraceRequest(_message.Message):
    __slots__ = ()
    CHART_FIELD_NUMBER: _ClassVar[int]
    TRACE_FIELD_NUMBER: _ClassVar[int]
    IGNORE_RULES_FIELD_NUMBER: _ClassVar[int]
    chart: _statecharts_pb2.Statechart
    trace: _containers.RepeatedCompositeFieldContainer[_statecharts_pb2.Machine]
    ignore_rules: _containers.RepeatedScalarFieldContainer[RuleId]
    def __init__(self, chart: _Optional[_Union[_statecharts_pb2.Statechart, _Mapping]] = ..., trace: _Optional[_Iterable[_Union[_statecharts_pb2.Machine, _Mapping]]] = ..., ignore_rules: _Optional[_Iterable[_Union[RuleId, str]]] = ...) -> None: ...

class ValidateChartResponse(_message.Message):
    __slots__ = ()
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    status: _status_pb2.Status
    violations: _containers.RepeatedCompositeFieldContainer[Violation]
    def __init__(self, status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ..., violations: _Optional[_Iterable[_Union[Violation, _Mapping]]] = ...) -> None: ...

class ValidateTraceResponse(_message.Message):
    __slots__ = ()
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    status: _status_pb2.Status
    violations: _containers.RepeatedCompositeFieldContainer[Violation]
    def __init__(self, status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ..., violations: _Optional[_Iterable[_Union[Violation, _Mapping]]] = ...) -> None: ...

class Violation(_message.Message):
    __slots__ = ()
    RULE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    XPATH_FIELD_NUMBER: _ClassVar[int]
    rule: RuleId
    severity: Severity
    message: str
    xpath: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, rule: _Optional[_Union[RuleId, str]] = ..., severity: _Optional[_Union[Severity, str]] = ..., message: _Optional[str] = ..., xpath: _Optional[_Iterable[str]] = ...) -> None: ...
