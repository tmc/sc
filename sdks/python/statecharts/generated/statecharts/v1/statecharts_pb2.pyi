from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from statecharts.v1 import expressions_pb2 as _expressions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StateType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATE_TYPE_UNSPECIFIED: _ClassVar[StateType]
    STATE_TYPE_BASIC: _ClassVar[StateType]
    STATE_TYPE_OR: _ClassVar[StateType]
    STATE_TYPE_AND: _ClassVar[StateType]
    STATE_TYPE_NORMAL: _ClassVar[StateType]
    STATE_TYPE_PARALLEL: _ClassVar[StateType]
    STATE_TYPE_ORTHOGONAL: _ClassVar[StateType]

class HistoryType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HISTORY_TYPE_UNSPECIFIED: _ClassVar[HistoryType]
    HISTORY_TYPE_SHALLOW: _ClassVar[HistoryType]
    HISTORY_TYPE_DEEP: _ClassVar[HistoryType]

class MachineState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MACHINE_STATE_UNSPECIFIED: _ClassVar[MachineState]
    MACHINE_STATE_RUNNING: _ClassVar[MachineState]
    MACHINE_STATE_STOPPED: _ClassVar[MachineState]
STATE_TYPE_UNSPECIFIED: StateType
STATE_TYPE_BASIC: StateType
STATE_TYPE_OR: StateType
STATE_TYPE_AND: StateType
STATE_TYPE_NORMAL: StateType
STATE_TYPE_PARALLEL: StateType
STATE_TYPE_ORTHOGONAL: StateType
HISTORY_TYPE_UNSPECIFIED: HistoryType
HISTORY_TYPE_SHALLOW: HistoryType
HISTORY_TYPE_DEEP: HistoryType
MACHINE_STATE_UNSPECIFIED: MachineState
MACHINE_STATE_RUNNING: MachineState
MACHINE_STATE_STOPPED: MachineState

class Statechart(_message.Message):
    __slots__ = ()
    ROOT_STATE_FIELD_NUMBER: _ClassVar[int]
    TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    root_state: State
    transitions: _containers.RepeatedCompositeFieldContainer[Transition]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    variables: _struct_pb2.Struct
    name: str
    description: str
    def __init__(self, root_state: _Optional[_Union[State, _Mapping]] = ..., transitions: _Optional[_Iterable[_Union[Transition, _Mapping]]] = ..., events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., variables: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., name: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CHILDREN_FIELD_NUMBER: _ClassVar[int]
    IS_INITIAL_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    ENTRY_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    EXIT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    IS_HISTORY_FIELD_NUMBER: _ClassVar[int]
    HISTORY_TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    label: str
    type: StateType
    children: _containers.RepeatedCompositeFieldContainer[State]
    is_initial: bool
    is_final: bool
    entry_actions: _containers.RepeatedCompositeFieldContainer[Action]
    exit_actions: _containers.RepeatedCompositeFieldContainer[Action]
    is_history: bool
    history_type: HistoryType
    metadata: _struct_pb2.Struct
    extensions: _containers.RepeatedCompositeFieldContainer[_any_pb2.Any]
    def __init__(self, label: _Optional[str] = ..., type: _Optional[_Union[StateType, str]] = ..., children: _Optional[_Iterable[_Union[State, _Mapping]]] = ..., is_initial: _Optional[bool] = ..., is_final: _Optional[bool] = ..., entry_actions: _Optional[_Iterable[_Union[Action, _Mapping]]] = ..., exit_actions: _Optional[_Iterable[_Union[Action, _Mapping]]] = ..., is_history: _Optional[bool] = ..., history_type: _Optional[_Union[HistoryType, str]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., extensions: _Optional[_Iterable[_Union[_any_pb2.Any, _Mapping]]] = ...) -> None: ...

class Transition(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    GUARD_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    label: str
    to: _containers.RepeatedScalarFieldContainer[str]
    event: str
    guard: Guard
    actions: _containers.RepeatedCompositeFieldContainer[Action]
    priority: int
    metadata: _struct_pb2.Struct
    extensions: _containers.RepeatedCompositeFieldContainer[_any_pb2.Any]
    def __init__(self, label: _Optional[str] = ..., to: _Optional[_Iterable[str]] = ..., event: _Optional[str] = ..., guard: _Optional[_Union[Guard, _Mapping]] = ..., actions: _Optional[_Iterable[_Union[Action, _Mapping]]] = ..., priority: _Optional[int] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., extensions: _Optional[_Iterable[_Union[_any_pb2.Any, _Mapping]]] = ..., **kwargs) -> None: ...

class Event(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    label: str
    parameters: _struct_pb2.Struct
    def __init__(self, label: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Guard(_message.Message):
    __slots__ = ()
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    expression: str
    language: str
    condition: _expressions_pb2.Expression
    def __init__(self, expression: _Optional[str] = ..., language: _Optional[str] = ..., condition: _Optional[_Union[_expressions_pb2.Expression, _Mapping]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    label: str
    expression: str
    language: str
    parameters: _struct_pb2.Struct
    body: _expressions_pb2.Expression
    def __init__(self, label: _Optional[str] = ..., expression: _Optional[str] = ..., language: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., body: _Optional[_Union[_expressions_pb2.Expression, _Mapping]] = ...) -> None: ...

class StateRef(_message.Message):
    __slots__ = ()
    LABEL_FIELD_NUMBER: _ClassVar[int]
    label: str
    def __init__(self, label: _Optional[str] = ...) -> None: ...

class Configuration(_message.Message):
    __slots__ = ()
    class HistoryEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Configuration
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Configuration, _Mapping]] = ...) -> None: ...
    STATES_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[StateRef]
    history: _containers.MessageMap[str, Configuration]
    def __init__(self, states: _Optional[_Iterable[_Union[StateRef, _Mapping]]] = ..., history: _Optional[_Mapping[str, Configuration]] = ...) -> None: ...

class Machine(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    STATECHART_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    STEP_HISTORY_FIELD_NUMBER: _ClassVar[int]
    id: str
    state: MachineState
    context: _struct_pb2.Struct
    statechart: Statechart
    configuration: Configuration
    step_history: _containers.RepeatedCompositeFieldContainer[Step]
    def __init__(self, id: _Optional[str] = ..., state: _Optional[_Union[MachineState, str]] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., statechart: _Optional[_Union[Statechart, _Mapping]] = ..., configuration: _Optional[_Union[Configuration, _Mapping]] = ..., step_history: _Optional[_Iterable[_Union[Step, _Mapping]]] = ...) -> None: ...

class Step(_message.Message):
    __slots__ = ()
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    STARTING_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    RESULTING_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    STATES_ENTERED_FIELD_NUMBER: _ClassVar[int]
    STATES_EXITED_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_EXECUTED_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    transitions: _containers.RepeatedCompositeFieldContainer[Transition]
    starting_configuration: Configuration
    resulting_configuration: Configuration
    context: _struct_pb2.Struct
    states_entered: _containers.RepeatedScalarFieldContainer[str]
    states_exited: _containers.RepeatedScalarFieldContainer[str]
    actions_executed: _containers.RepeatedCompositeFieldContainer[Action]
    def __init__(self, events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., transitions: _Optional[_Iterable[_Union[Transition, _Mapping]]] = ..., starting_configuration: _Optional[_Union[Configuration, _Mapping]] = ..., resulting_configuration: _Optional[_Union[Configuration, _Mapping]] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., states_entered: _Optional[_Iterable[str]] = ..., states_exited: _Optional[_Iterable[str]] = ..., actions_executed: _Optional[_Iterable[_Union[Action, _Mapping]]] = ...) -> None: ...
