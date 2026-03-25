import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Author(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    ORGANIZATION_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    name: str
    email: str
    organization: str
    identifier: str
    role: str
    def __init__(self, name: _Optional[str] = ..., email: _Optional[str] = ..., organization: _Optional[str] = ..., identifier: _Optional[str] = ..., role: _Optional[str] = ...) -> None: ...

class License(_message.Message):
    __slots__ = ()
    SPDX_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    COPYRIGHT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    NOTICE_FIELD_NUMBER: _ClassVar[int]
    DETECTED_FIELD_NUMBER: _ClassVar[int]
    spdx_id: str
    name: str
    url: str
    copyright: str
    year: str
    notice: str
    detected: bool
    def __init__(self, spdx_id: _Optional[str] = ..., name: _Optional[str] = ..., url: _Optional[str] = ..., copyright: _Optional[str] = ..., year: _Optional[str] = ..., notice: _Optional[str] = ..., detected: _Optional[bool] = ...) -> None: ...

class Link(_message.Message):
    __slots__ = ()
    TITLE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    title: str
    url: str
    type: str
    description: str
    def __init__(self, title: _Optional[str] = ..., url: _Optional[str] = ..., type: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class ChangeEntry(_message.Message):
    __slots__ = ()
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    CHANGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TICKET_FIELD_NUMBER: _ClassVar[int]
    version: str
    timestamp: _timestamp_pb2.Timestamp
    author: Author
    change_type: str
    description: str
    ticket: str
    def __init__(self, version: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., author: _Optional[_Union[Author, _Mapping]] = ..., change_type: _Optional[str] = ..., description: _Optional[str] = ..., ticket: _Optional[str] = ...) -> None: ...
