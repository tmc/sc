import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from statecharts.v1 import statecharts_pb2 as _statecharts_pb2
from statecharts.v1 import expressions_pb2 as _expressions_pb2
from statecharts.v1 import execution_pb2 as _execution_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChangeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHANGE_TYPE_UNSPECIFIED: _ClassVar[ChangeType]
    CHANGE_TYPE_ADDED: _ClassVar[ChangeType]
    CHANGE_TYPE_REMOVED: _ClassVar[ChangeType]
    CHANGE_TYPE_MODIFIED: _ClassVar[ChangeType]
    CHANGE_TYPE_RENAMED: _ClassVar[ChangeType]
    CHANGE_TYPE_MOVED: _ClassVar[ChangeType]

class BreakingChangeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BREAKING_CHANGE_TYPE_UNSPECIFIED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_STATE_REMOVED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_STATE_TYPE_CHANGED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_INITIAL_STATE_CHANGED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_TRANSITION_REMOVED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_TRANSITION_SOURCE_CHANGED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_GUARD_ADDED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_EVENT_REMOVED: _ClassVar[BreakingChangeType]
    BREAKING_CHANGE_TYPE_EVENT_PAYLOAD_CHANGED: _ClassVar[BreakingChangeType]

class MigrationRequirement(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MIGRATION_REQUIREMENT_UNSPECIFIED: _ClassVar[MigrationRequirement]
    MIGRATION_REQUIREMENT_NONE: _ClassVar[MigrationRequirement]
    MIGRATION_REQUIREMENT_STATE_MAPPING: _ClassVar[MigrationRequirement]
    MIGRATION_REQUIREMENT_MANUAL: _ClassVar[MigrationRequirement]

class MigrationStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MIGRATION_STRATEGY_UNSPECIFIED: _ClassVar[MigrationStrategy]
    MIGRATION_STRATEGY_STOP_THE_WORLD: _ClassVar[MigrationStrategy]
    MIGRATION_STRATEGY_ROLLING: _ClassVar[MigrationStrategy]
    MIGRATION_STRATEGY_BLUE_GREEN: _ClassVar[MigrationStrategy]
    MIGRATION_STRATEGY_CANARY: _ClassVar[MigrationStrategy]

class StateMappingType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATE_MAPPING_TYPE_UNSPECIFIED: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_IDENTITY: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_RENAME: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_TO_PARENT: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_TO_SIBLING: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_TO_INITIAL: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_SPLIT: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_MERGE: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_CUSTOM: _ClassVar[StateMappingType]
    STATE_MAPPING_TYPE_ERROR: _ClassVar[StateMappingType]

class RiskLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RISK_LEVEL_UNSPECIFIED: _ClassVar[RiskLevel]
    RISK_LEVEL_LOW: _ClassVar[RiskLevel]
    RISK_LEVEL_MEDIUM: _ClassVar[RiskLevel]
    RISK_LEVEL_HIGH: _ClassVar[RiskLevel]

class MigrationResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MIGRATION_RESULT_UNSPECIFIED: _ClassVar[MigrationResult]
    MIGRATION_RESULT_SUCCESS: _ClassVar[MigrationResult]
    MIGRATION_RESULT_PARTIAL: _ClassVar[MigrationResult]
    MIGRATION_RESULT_FAILED: _ClassVar[MigrationResult]
    MIGRATION_RESULT_ROLLED_BACK: _ClassVar[MigrationResult]
    MIGRATION_RESULT_CANCELLED: _ClassVar[MigrationResult]

class InFlightAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    IN_FLIGHT_ACTION_UNSPECIFIED: _ClassVar[InFlightAction]
    IN_FLIGHT_ACTION_COMPLETE: _ClassVar[InFlightAction]
    IN_FLIGHT_ACTION_ABORT: _ClassVar[InFlightAction]
    IN_FLIGHT_ACTION_RETRY: _ClassVar[InFlightAction]
    IN_FLIGHT_ACTION_ERROR: _ClassVar[InFlightAction]

class RollbackTriggerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLLBACK_TRIGGER_TYPE_UNSPECIFIED: _ClassVar[RollbackTriggerType]
    ROLLBACK_TRIGGER_TYPE_ERROR_RATE: _ClassVar[RollbackTriggerType]
    ROLLBACK_TRIGGER_TYPE_LATENCY: _ClassVar[RollbackTriggerType]
    ROLLBACK_TRIGGER_TYPE_MANUAL: _ClassVar[RollbackTriggerType]
    ROLLBACK_TRIGGER_TYPE_HEALTH_CHECK: _ClassVar[RollbackTriggerType]

class RollbackStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLLBACK_STRATEGY_UNSPECIFIED: _ClassVar[RollbackStrategy]
    ROLLBACK_STRATEGY_CHECKPOINT: _ClassVar[RollbackStrategy]
    ROLLBACK_STRATEGY_REVERSE: _ClassVar[RollbackStrategy]
    ROLLBACK_STRATEGY_RECREATE: _ClassVar[RollbackStrategy]
CHANGE_TYPE_UNSPECIFIED: ChangeType
CHANGE_TYPE_ADDED: ChangeType
CHANGE_TYPE_REMOVED: ChangeType
CHANGE_TYPE_MODIFIED: ChangeType
CHANGE_TYPE_RENAMED: ChangeType
CHANGE_TYPE_MOVED: ChangeType
BREAKING_CHANGE_TYPE_UNSPECIFIED: BreakingChangeType
BREAKING_CHANGE_TYPE_STATE_REMOVED: BreakingChangeType
BREAKING_CHANGE_TYPE_STATE_TYPE_CHANGED: BreakingChangeType
BREAKING_CHANGE_TYPE_INITIAL_STATE_CHANGED: BreakingChangeType
BREAKING_CHANGE_TYPE_TRANSITION_REMOVED: BreakingChangeType
BREAKING_CHANGE_TYPE_TRANSITION_SOURCE_CHANGED: BreakingChangeType
BREAKING_CHANGE_TYPE_GUARD_ADDED: BreakingChangeType
BREAKING_CHANGE_TYPE_EVENT_REMOVED: BreakingChangeType
BREAKING_CHANGE_TYPE_EVENT_PAYLOAD_CHANGED: BreakingChangeType
MIGRATION_REQUIREMENT_UNSPECIFIED: MigrationRequirement
MIGRATION_REQUIREMENT_NONE: MigrationRequirement
MIGRATION_REQUIREMENT_STATE_MAPPING: MigrationRequirement
MIGRATION_REQUIREMENT_MANUAL: MigrationRequirement
MIGRATION_STRATEGY_UNSPECIFIED: MigrationStrategy
MIGRATION_STRATEGY_STOP_THE_WORLD: MigrationStrategy
MIGRATION_STRATEGY_ROLLING: MigrationStrategy
MIGRATION_STRATEGY_BLUE_GREEN: MigrationStrategy
MIGRATION_STRATEGY_CANARY: MigrationStrategy
STATE_MAPPING_TYPE_UNSPECIFIED: StateMappingType
STATE_MAPPING_TYPE_IDENTITY: StateMappingType
STATE_MAPPING_TYPE_RENAME: StateMappingType
STATE_MAPPING_TYPE_TO_PARENT: StateMappingType
STATE_MAPPING_TYPE_TO_SIBLING: StateMappingType
STATE_MAPPING_TYPE_TO_INITIAL: StateMappingType
STATE_MAPPING_TYPE_SPLIT: StateMappingType
STATE_MAPPING_TYPE_MERGE: StateMappingType
STATE_MAPPING_TYPE_CUSTOM: StateMappingType
STATE_MAPPING_TYPE_ERROR: StateMappingType
RISK_LEVEL_UNSPECIFIED: RiskLevel
RISK_LEVEL_LOW: RiskLevel
RISK_LEVEL_MEDIUM: RiskLevel
RISK_LEVEL_HIGH: RiskLevel
MIGRATION_RESULT_UNSPECIFIED: MigrationResult
MIGRATION_RESULT_SUCCESS: MigrationResult
MIGRATION_RESULT_PARTIAL: MigrationResult
MIGRATION_RESULT_FAILED: MigrationResult
MIGRATION_RESULT_ROLLED_BACK: MigrationResult
MIGRATION_RESULT_CANCELLED: MigrationResult
IN_FLIGHT_ACTION_UNSPECIFIED: InFlightAction
IN_FLIGHT_ACTION_COMPLETE: InFlightAction
IN_FLIGHT_ACTION_ABORT: InFlightAction
IN_FLIGHT_ACTION_RETRY: InFlightAction
IN_FLIGHT_ACTION_ERROR: InFlightAction
ROLLBACK_TRIGGER_TYPE_UNSPECIFIED: RollbackTriggerType
ROLLBACK_TRIGGER_TYPE_ERROR_RATE: RollbackTriggerType
ROLLBACK_TRIGGER_TYPE_LATENCY: RollbackTriggerType
ROLLBACK_TRIGGER_TYPE_MANUAL: RollbackTriggerType
ROLLBACK_TRIGGER_TYPE_HEALTH_CHECK: RollbackTriggerType
ROLLBACK_STRATEGY_UNSPECIFIED: RollbackStrategy
ROLLBACK_STRATEGY_CHECKPOINT: RollbackStrategy
ROLLBACK_STRATEGY_REVERSE: RollbackStrategy
ROLLBACK_STRATEGY_RECREATE: RollbackStrategy

class ChartDiff(_message.Message):
    __slots__ = ()
    FROM_VERSION_FIELD_NUMBER: _ClassVar[int]
    TO_VERSION_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGES_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_CHANGES_FIELD_NUMBER: _ClassVar[int]
    EVENT_CHANGES_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_FIELD_NUMBER: _ClassVar[int]
    BREAKING_CHANGES_FIELD_NUMBER: _ClassVar[int]
    from_version: _execution_pb2.ChartVersion
    to_version: _execution_pb2.ChartVersion
    state_changes: _containers.RepeatedCompositeFieldContainer[StateDiff]
    transition_changes: _containers.RepeatedCompositeFieldContainer[TransitionDiff]
    event_changes: _containers.RepeatedCompositeFieldContainer[EventDiff]
    compatibility: _execution_pb2.CompatibilityLevel
    breaking_changes: _containers.RepeatedCompositeFieldContainer[BreakingChange]
    def __init__(self, from_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., to_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., state_changes: _Optional[_Iterable[_Union[StateDiff, _Mapping]]] = ..., transition_changes: _Optional[_Iterable[_Union[TransitionDiff, _Mapping]]] = ..., event_changes: _Optional[_Iterable[_Union[EventDiff, _Mapping]]] = ..., compatibility: _Optional[_Union[_execution_pb2.CompatibilityLevel, str]] = ..., breaking_changes: _Optional[_Iterable[_Union[BreakingChange, _Mapping]]] = ...) -> None: ...

class StateDiff(_message.Message):
    __slots__ = ()
    CHANGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATE_LABEL_FIELD_NUMBER: _ClassVar[int]
    OLD_STATE_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    change_type: ChangeType
    state_label: str
    old_state: _statecharts_pb2.State
    new_state: _statecharts_pb2.State
    modified_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, change_type: _Optional[_Union[ChangeType, str]] = ..., state_label: _Optional[str] = ..., old_state: _Optional[_Union[_statecharts_pb2.State, _Mapping]] = ..., new_state: _Optional[_Union[_statecharts_pb2.State, _Mapping]] = ..., modified_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class TransitionDiff(_message.Message):
    __slots__ = ()
    CHANGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_LABEL_FIELD_NUMBER: _ClassVar[int]
    OLD_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    NEW_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    change_type: ChangeType
    transition_label: str
    old_transition: _statecharts_pb2.Transition
    new_transition: _statecharts_pb2.Transition
    modified_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, change_type: _Optional[_Union[ChangeType, str]] = ..., transition_label: _Optional[str] = ..., old_transition: _Optional[_Union[_statecharts_pb2.Transition, _Mapping]] = ..., new_transition: _Optional[_Union[_statecharts_pb2.Transition, _Mapping]] = ..., modified_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class EventDiff(_message.Message):
    __slots__ = ()
    CHANGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    EVENT_NAME_FIELD_NUMBER: _ClassVar[int]
    OLD_EVENT_FIELD_NUMBER: _ClassVar[int]
    NEW_EVENT_FIELD_NUMBER: _ClassVar[int]
    change_type: ChangeType
    event_name: str
    old_event: _statecharts_pb2.Event
    new_event: _statecharts_pb2.Event
    def __init__(self, change_type: _Optional[_Union[ChangeType, str]] = ..., event_name: _Optional[str] = ..., old_event: _Optional[_Union[_statecharts_pb2.Event, _Mapping]] = ..., new_event: _Optional[_Union[_statecharts_pb2.Event, _Mapping]] = ...) -> None: ...

class BreakingChange(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_ELEMENT_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    type: BreakingChangeType
    description: str
    affected_element: str
    migration_required: MigrationRequirement
    def __init__(self, type: _Optional[_Union[BreakingChangeType, str]] = ..., description: _Optional[str] = ..., affected_element: _Optional[str] = ..., migration_required: _Optional[_Union[MigrationRequirement, str]] = ...) -> None: ...

class DeprecationInfo(_message.Message):
    __slots__ = ()
    DEPRECATED_IN_VERSION_FIELD_NUMBER: _ClassVar[int]
    REMOVAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_GUIDE_FIELD_NUMBER: _ClassVar[int]
    REPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    deprecated_in_version: str
    removal_version: str
    migration_guide: str
    replacement: str
    def __init__(self, deprecated_in_version: _Optional[str] = ..., removal_version: _Optional[str] = ..., migration_guide: _Optional[str] = ..., replacement: _Optional[str] = ...) -> None: ...

class MigrationPlan(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    FROM_VERSION_FIELD_NUMBER: _ClassVar[int]
    TO_VERSION_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    STATE_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    VALIDATIONS_FIELD_NUMBER: _ClassVar[int]
    IMPACT_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    IN_FLIGHT_POLICY_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_PLAN_FIELD_NUMBER: _ClassVar[int]
    id: str
    from_version: _execution_pb2.ChartVersion
    to_version: _execution_pb2.ChartVersion
    strategy: MigrationStrategy
    state_mappings: _containers.RepeatedCompositeFieldContainer[StateMapping]
    transition_mappings: _containers.RepeatedCompositeFieldContainer[TransitionMapping]
    context_transform: ContextTransformation
    validations: _containers.RepeatedCompositeFieldContainer[MigrationValidation]
    impact: MigrationImpact
    dry_run: bool
    in_flight_policy: InFlightTransitionPolicy
    rollback_plan: RollbackPlan
    def __init__(self, id: _Optional[str] = ..., from_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., to_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., strategy: _Optional[_Union[MigrationStrategy, str]] = ..., state_mappings: _Optional[_Iterable[_Union[StateMapping, _Mapping]]] = ..., transition_mappings: _Optional[_Iterable[_Union[TransitionMapping, _Mapping]]] = ..., context_transform: _Optional[_Union[ContextTransformation, _Mapping]] = ..., validations: _Optional[_Iterable[_Union[MigrationValidation, _Mapping]]] = ..., impact: _Optional[_Union[MigrationImpact, _Mapping]] = ..., dry_run: _Optional[bool] = ..., in_flight_policy: _Optional[_Union[InFlightTransitionPolicy, _Mapping]] = ..., rollback_plan: _Optional[_Union[RollbackPlan, _Mapping]] = ...) -> None: ...

class StateMapping(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FROM_STATES_FIELD_NUMBER: _ClassVar[int]
    TO_STATES_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    type: StateMappingType
    from_states: _containers.RepeatedScalarFieldContainer[str]
    to_states: _containers.RepeatedScalarFieldContainer[str]
    condition: _expressions_pb2.Expression
    transform: ContextTransformation
    priority: int
    def __init__(self, type: _Optional[_Union[StateMappingType, str]] = ..., from_states: _Optional[_Iterable[str]] = ..., to_states: _Optional[_Iterable[str]] = ..., condition: _Optional[_Union[_expressions_pb2.Expression, _Mapping]] = ..., transform: _Optional[_Union[ContextTransformation, _Mapping]] = ..., priority: _Optional[int] = ...) -> None: ...

class TransitionMapping(_message.Message):
    __slots__ = ()
    FROM_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    TO_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    from_transition: str
    to_transition: str
    def __init__(self, from_transition: _Optional[str] = ..., to_transition: _Optional[str] = ...) -> None: ...

class ContextTransformation(_message.Message):
    __slots__ = ()
    class SetEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class RenameEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SET_FIELD_NUMBER: _ClassVar[int]
    REMOVE_FIELD_NUMBER: _ClassVar[int]
    RENAME_FIELD_NUMBER: _ClassVar[int]
    set: _containers.ScalarMap[str, str]
    remove: _containers.RepeatedScalarFieldContainer[str]
    rename: _containers.ScalarMap[str, str]
    def __init__(self, set: _Optional[_Mapping[str, str]] = ..., remove: _Optional[_Iterable[str]] = ..., rename: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MigrationValidation(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    expression: str
    error_message: str
    def __init__(self, name: _Optional[str] = ..., expression: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class MigrationImpact(_message.Message):
    __slots__ = ()
    MACHINES_AFFECTED_FIELD_NUMBER: _ClassVar[int]
    MACHINES_REQUIRING_MANUAL_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_DOWNTIME_FIELD_NUMBER: _ClassVar[int]
    RISK_FIELD_NUMBER: _ClassVar[int]
    machines_affected: int
    machines_requiring_manual: int
    estimated_downtime: _duration_pb2.Duration
    risk: RiskLevel
    def __init__(self, machines_affected: _Optional[int] = ..., machines_requiring_manual: _Optional[int] = ..., estimated_downtime: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., risk: _Optional[_Union[RiskLevel, str]] = ...) -> None: ...

class DryRunResult(_message.Message):
    __slots__ = ()
    PREVIEWS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    WOULD_SUCCEED_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_DURATION_FIELD_NUMBER: _ClassVar[int]
    previews: _containers.RepeatedCompositeFieldContainer[MachineMigrationPreview]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    errors: _containers.RepeatedScalarFieldContainer[str]
    would_succeed: bool
    estimated_duration: _duration_pb2.Duration
    def __init__(self, previews: _Optional[_Iterable[_Union[MachineMigrationPreview, _Mapping]]] = ..., warnings: _Optional[_Iterable[str]] = ..., errors: _Optional[_Iterable[str]] = ..., would_succeed: _Optional[bool] = ..., estimated_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class MachineMigrationPreview(_message.Message):
    __slots__ = ()
    MACHINE_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TARGET_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MAPPING_USED_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    machine_id: str
    current_config: _statecharts_pb2.Configuration
    target_config: _statecharts_pb2.Configuration
    mapping_used: StateMappingType
    warnings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, machine_id: _Optional[str] = ..., current_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., target_config: _Optional[_Union[_statecharts_pb2.Configuration, _Mapping]] = ..., mapping_used: _Optional[_Union[StateMappingType, str]] = ..., warnings: _Optional[_Iterable[str]] = ...) -> None: ...

class MigrationHistory(_message.Message):
    __slots__ = ()
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[MigrationRecord]
    def __init__(self, records: _Optional[_Iterable[_Union[MigrationRecord, _Mapping]]] = ...) -> None: ...

class MigrationRecord(_message.Message):
    __slots__ = ()
    MIGRATION_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_VERSION_FIELD_NUMBER: _ClassVar[int]
    TO_VERSION_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    MACHINES_MIGRATED_FIELD_NUMBER: _ClassVar[int]
    MACHINES_FAILED_FIELD_NUMBER: _ClassVar[int]
    INITIATED_BY_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_OF_FIELD_NUMBER: _ClassVar[int]
    migration_id: str
    from_version: _execution_pb2.ChartVersion
    to_version: _execution_pb2.ChartVersion
    started_at: _timestamp_pb2.Timestamp
    completed_at: _timestamp_pb2.Timestamp
    result: MigrationResult
    machines_migrated: int
    machines_failed: int
    initiated_by: str
    rollback_of: str
    def __init__(self, migration_id: _Optional[str] = ..., from_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., to_version: _Optional[_Union[_execution_pb2.ChartVersion, _Mapping]] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., result: _Optional[_Union[MigrationResult, str]] = ..., machines_migrated: _Optional[int] = ..., machines_failed: _Optional[int] = ..., initiated_by: _Optional[str] = ..., rollback_of: _Optional[str] = ...) -> None: ...

class InFlightTransitionPolicy(_message.Message):
    __slots__ = ()
    ACTION_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    action: InFlightAction
    grace_period: _duration_pb2.Duration
    def __init__(self, action: _Optional[_Union[InFlightAction, str]] = ..., grace_period: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class RollbackPlan(_message.Message):
    __slots__ = ()
    TRIGGERS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    ROLLBACK_WINDOW_FIELD_NUMBER: _ClassVar[int]
    triggers: _containers.RepeatedCompositeFieldContainer[RollbackTrigger]
    strategy: RollbackStrategy
    rollback_mappings: _containers.RepeatedCompositeFieldContainer[StateMapping]
    rollback_window: _duration_pb2.Duration
    def __init__(self, triggers: _Optional[_Iterable[_Union[RollbackTrigger, _Mapping]]] = ..., strategy: _Optional[_Union[RollbackStrategy, str]] = ..., rollback_mappings: _Optional[_Iterable[_Union[StateMapping, _Mapping]]] = ..., rollback_window: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class RollbackTrigger(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    type: RollbackTriggerType
    threshold: str
    def __init__(self, type: _Optional[_Union[RollbackTriggerType, str]] = ..., threshold: _Optional[str] = ...) -> None: ...
