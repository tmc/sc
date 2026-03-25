import datetime

from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TransitionTriggerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSITION_TRIGGER_TYPE_UNSPECIFIED: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_EVENT: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_ALWAYS: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_AFTER: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_INVOKE_DONE: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_INVOKE_ERROR: _ClassVar[TransitionTriggerType]
    TRANSITION_TRIGGER_TYPE_STATE_DONE: _ClassVar[TransitionTriggerType]
TRANSITION_TRIGGER_TYPE_UNSPECIFIED: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_EVENT: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_ALWAYS: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_AFTER: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_INVOKE_DONE: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_INVOKE_ERROR: TransitionTriggerType
TRANSITION_TRIGGER_TYPE_STATE_DONE: TransitionTriggerType

class XStateLayout(_message.Message):
    __slots__ = ()
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_ID_FIELD_NUMBER: _ClassVar[int]
    position: Position
    size: Size
    color: str
    unique_id: str
    def __init__(self, position: _Optional[_Union[Position, _Mapping]] = ..., size: _Optional[_Union[Size, _Mapping]] = ..., color: _Optional[str] = ..., unique_id: _Optional[str] = ...) -> None: ...

class Position(_message.Message):
    __slots__ = ()
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Size(_message.Message):
    __slots__ = ()
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    width: float
    height: float
    def __init__(self, width: _Optional[float] = ..., height: _Optional[float] = ...) -> None: ...

class XStateStateData(_message.Message):
    __slots__ = ()
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    ASSETS_FIELD_NUMBER: _ClassVar[int]
    INVOKES_FIELD_NUMBER: _ClassVar[int]
    META_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    description: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    assets: _containers.RepeatedCompositeFieldContainer[XStateAsset]
    invokes: _containers.RepeatedCompositeFieldContainer[XStateInvoke]
    meta_entries: _containers.RepeatedCompositeFieldContainer[MetaEntry]
    def __init__(self, description: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., assets: _Optional[_Iterable[_Union[XStateAsset, _Mapping]]] = ..., invokes: _Optional[_Iterable[_Union[XStateInvoke, _Mapping]]] = ..., meta_entries: _Optional[_Iterable[_Union[MetaEntry, _Mapping]]] = ...) -> None: ...

class XStateTransitionData(_message.Message):
    __slots__ = ()
    INTERNAL_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    internal: bool
    trigger_type: TransitionTriggerType
    description: str
    def __init__(self, internal: _Optional[bool] = ..., trigger_type: _Optional[_Union[TransitionTriggerType, str]] = ..., description: _Optional[str] = ...) -> None: ...

class XStateInvoke(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    SRC_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    src: str
    kind: str
    input: _struct_pb2.Struct
    settings: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., src: _Optional[str] = ..., kind: _Optional[str] = ..., input: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., settings: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class XStateAsset(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    name: str
    template: str
    properties: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., template: _Optional[str] = ..., properties: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class MetaEntry(_message.Message):
    __slots__ = ()
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class XStateMachineData(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    FORK_PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    LAST_EDITED_BY_ID_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_CODE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    SCHEMAS_FIELD_NUMBER: _ClassVar[int]
    IMPLEMENTATIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    project_version_id: str
    fork_parent_id: str
    last_edited_by_id: str
    original_code: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    schemas: XStateSchemas
    implementations: XStateImplementations
    def __init__(self, id: _Optional[str] = ..., project_version_id: _Optional[str] = ..., fork_parent_id: _Optional[str] = ..., last_edited_by_id: _Optional[str] = ..., original_code: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., schemas: _Optional[_Union[XStateSchemas, _Mapping]] = ..., implementations: _Optional[_Union[XStateImplementations, _Mapping]] = ...) -> None: ...

class XStateSchemas(_message.Message):
    __slots__ = ()
    TAGS_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    DELAYS_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    GUARDS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    tags: _struct_pb2.Struct
    input: _struct_pb2.Struct
    output: _struct_pb2.Struct
    actors: _struct_pb2.Struct
    delays: _struct_pb2.Struct
    events: _struct_pb2.Struct
    guards: _struct_pb2.Struct
    actions: _struct_pb2.Struct
    context: _struct_pb2.Struct
    def __init__(self, tags: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., input: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., output: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., actors: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., delays: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., events: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., guards: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., actions: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class XStateImplementations(_message.Message):
    __slots__ = ()
    class ActionsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: XStateActionImpl
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[XStateActionImpl, _Mapping]] = ...) -> None: ...
    class GuardsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: XStateGuardImpl
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[XStateGuardImpl, _Mapping]] = ...) -> None: ...
    class ActorsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: XStateActorImpl
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[XStateActorImpl, _Mapping]] = ...) -> None: ...
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    GUARDS_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.MessageMap[str, XStateActionImpl]
    guards: _containers.MessageMap[str, XStateGuardImpl]
    actors: _containers.MessageMap[str, XStateActorImpl]
    def __init__(self, actions: _Optional[_Mapping[str, XStateActionImpl]] = ..., guards: _Optional[_Mapping[str, XStateGuardImpl]] = ..., actors: _Optional[_Mapping[str, XStateActorImpl]] = ...) -> None: ...

class XStateActionImpl(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    IMPORTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    code: str
    schema: _struct_pb2.Struct
    imports: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., code: _Optional[str] = ..., schema: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., imports: _Optional[_Iterable[str]] = ...) -> None: ...

class XStateGuardImpl(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    IMPORTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    params: _struct_pb2.Struct
    imports: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., imports: _Optional[_Iterable[str]] = ...) -> None: ...

class XStateActorImpl(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    IMPORTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    kind: str
    input: _struct_pb2.Struct
    output: _struct_pb2.Struct
    imports: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., input: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., output: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., imports: _Optional[_Iterable[str]] = ...) -> None: ...
