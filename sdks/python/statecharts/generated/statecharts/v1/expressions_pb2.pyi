from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExpressionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXPRESSION_TYPE_UNSPECIFIED: _ClassVar[ExpressionType]
    EXPRESSION_TYPE_RAW: _ClassVar[ExpressionType]
    EXPRESSION_TYPE_CEL: _ClassVar[ExpressionType]
    EXPRESSION_TYPE_STARLARK: _ClassVar[ExpressionType]
EXPRESSION_TYPE_UNSPECIFIED: ExpressionType
EXPRESSION_TYPE_RAW: ExpressionType
EXPRESSION_TYPE_CEL: ExpressionType
EXPRESSION_TYPE_STARLARK: ExpressionType

class Expression(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CEL_AST_FIELD_NUMBER: _ClassVar[int]
    CEL_CHECKED_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    type: ExpressionType
    source: str
    cel_ast: bytes
    cel_checked: bytes
    metadata: _struct_pb2.Struct
    def __init__(self, type: _Optional[_Union[ExpressionType, str]] = ..., source: _Optional[str] = ..., cel_ast: _Optional[bytes] = ..., cel_checked: _Optional[bytes] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExpressionSecurityConfig(_message.Message):
    __slots__ = ()
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_EXECUTION_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_STARLARK_FIELD_NUMBER: _ClassVar[int]
    ALLOW_STARLARK_LOOPS_FIELD_NUMBER: _ClassVar[int]
    max_depth: int
    max_execution_ms: int
    max_memory_bytes: int
    allowed_functions: _containers.RepeatedScalarFieldContainer[str]
    blocked_functions: _containers.RepeatedScalarFieldContainer[str]
    allow_starlark: bool
    allow_starlark_loops: bool
    def __init__(self, max_depth: _Optional[int] = ..., max_execution_ms: _Optional[int] = ..., max_memory_bytes: _Optional[int] = ..., allowed_functions: _Optional[_Iterable[str]] = ..., blocked_functions: _Optional[_Iterable[str]] = ..., allow_starlark: _Optional[bool] = ..., allow_starlark_loops: _Optional[bool] = ...) -> None: ...
