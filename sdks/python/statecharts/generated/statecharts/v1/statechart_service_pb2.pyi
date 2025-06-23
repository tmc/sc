from google.protobuf import struct_pb2 as _struct_pb2
from google.rpc import status_pb2 as _status_pb2
from statecharts.v1 import statecharts_pb2 as _statecharts_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StatechartRegistry(_message.Message):
    __slots__ = ("statecharts",)
    class StatechartsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _statecharts_pb2.Statechart
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_statecharts_pb2.Statechart, _Mapping]] = ...) -> None: ...
    STATECHARTS_FIELD_NUMBER: _ClassVar[int]
    statecharts: _containers.MessageMap[str, _statecharts_pb2.Statechart]
    def __init__(self, statecharts: _Optional[_Mapping[str, _statecharts_pb2.Statechart]] = ...) -> None: ...

class CreateMachineRequest(_message.Message):
    __slots__ = ("statechart_id", "context")
    STATECHART_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    statechart_id: str
    context: _struct_pb2.Struct
    def __init__(self, statechart_id: _Optional[str] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CreateMachineResponse(_message.Message):
    __slots__ = ("machine",)
    MACHINE_FIELD_NUMBER: _ClassVar[int]
    machine: _statecharts_pb2.Machine
    def __init__(self, machine: _Optional[_Union[_statecharts_pb2.Machine, _Mapping]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("statechart_id", "event", "context")
    STATECHART_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    statechart_id: str
    event: str
    context: _struct_pb2.Struct
    def __init__(self, statechart_id: _Optional[str] = ..., event: _Optional[str] = ..., context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("machine", "result")
    MACHINE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    machine: _statecharts_pb2.Machine
    result: _status_pb2.Status
    def __init__(self, machine: _Optional[_Union[_statecharts_pb2.Machine, _Mapping]] = ..., result: _Optional[_Union[_status_pb2.Status, _Mapping]] = ...) -> None: ...
