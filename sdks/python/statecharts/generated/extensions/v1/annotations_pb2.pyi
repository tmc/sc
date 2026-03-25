from google.protobuf import timestamp_pb2 as _timestamp_pb2
from extensions.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StateAnnotations(_message.Message):
    __slots__ = ()
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    SINCE_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEPRECATION_FIELD_NUMBER: _ClassVar[int]
    RELATED_STATES_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    RISK_FIELD_NUMBER: _ClassVar[int]
    CHANGE_HISTORY_FIELD_NUMBER: _ClassVar[int]
    INVARIANTS_FIELD_NUMBER: _ClassVar[int]
    PRECONDITIONS_FIELD_NUMBER: _ClassVar[int]
    description: str
    summary: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    category: _containers.RepeatedScalarFieldContainer[str]
    author: _common_pb2.Author
    since_version: str
    deprecation: Deprecation
    related_states: _containers.RepeatedScalarFieldContainer[str]
    links: _containers.RepeatedCompositeFieldContainer[_common_pb2.Link]
    requirements: _containers.RepeatedScalarFieldContainer[str]
    risk: RiskAssessment
    change_history: _containers.RepeatedCompositeFieldContainer[_common_pb2.ChangeEntry]
    invariants: _containers.RepeatedScalarFieldContainer[str]
    preconditions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, description: _Optional[str] = ..., summary: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., category: _Optional[_Iterable[str]] = ..., author: _Optional[_Union[_common_pb2.Author, _Mapping]] = ..., since_version: _Optional[str] = ..., deprecation: _Optional[_Union[Deprecation, _Mapping]] = ..., related_states: _Optional[_Iterable[str]] = ..., links: _Optional[_Iterable[_Union[_common_pb2.Link, _Mapping]]] = ..., requirements: _Optional[_Iterable[str]] = ..., risk: _Optional[_Union[RiskAssessment, _Mapping]] = ..., change_history: _Optional[_Iterable[_Union[_common_pb2.ChangeEntry, _Mapping]]] = ..., invariants: _Optional[_Iterable[str]] = ..., preconditions: _Optional[_Iterable[str]] = ...) -> None: ...

class TransitionAnnotations(_message.Message):
    __slots__ = ()
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    SINCE_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEPRECATION_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    PRECONDITIONS_FIELD_NUMBER: _ClassVar[int]
    POSTCONDITIONS_FIELD_NUMBER: _ClassVar[int]
    SIDE_EFFECTS_FIELD_NUMBER: _ClassVar[int]
    CHANGE_HISTORY_FIELD_NUMBER: _ClassVar[int]
    PERFORMANCE_NOTES_FIELD_NUMBER: _ClassVar[int]
    description: str
    summary: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    author: _common_pb2.Author
    since_version: str
    deprecation: Deprecation
    requirements: _containers.RepeatedScalarFieldContainer[str]
    preconditions: _containers.RepeatedScalarFieldContainer[str]
    postconditions: _containers.RepeatedScalarFieldContainer[str]
    side_effects: _containers.RepeatedScalarFieldContainer[str]
    change_history: _containers.RepeatedCompositeFieldContainer[_common_pb2.ChangeEntry]
    performance_notes: str
    def __init__(self, description: _Optional[str] = ..., summary: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., author: _Optional[_Union[_common_pb2.Author, _Mapping]] = ..., since_version: _Optional[str] = ..., deprecation: _Optional[_Union[Deprecation, _Mapping]] = ..., requirements: _Optional[_Iterable[str]] = ..., preconditions: _Optional[_Iterable[str]] = ..., postconditions: _Optional[_Iterable[str]] = ..., side_effects: _Optional[_Iterable[str]] = ..., change_history: _Optional[_Iterable[_Union[_common_pb2.ChangeEntry, _Mapping]]] = ..., performance_notes: _Optional[str] = ...) -> None: ...

class EventAnnotations(_message.Message):
    __slots__ = ()
    EVENT_LABEL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    PRODUCERS_FIELD_NUMBER: _ClassVar[int]
    CONSUMERS_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    event_label: str
    description: str
    payload_schema: str
    examples: _containers.RepeatedScalarFieldContainer[str]
    producers: _containers.RepeatedScalarFieldContainer[str]
    consumers: _containers.RepeatedScalarFieldContainer[str]
    internal: bool
    frequency: str
    priority: str
    def __init__(self, event_label: _Optional[str] = ..., description: _Optional[str] = ..., payload_schema: _Optional[str] = ..., examples: _Optional[_Iterable[str]] = ..., producers: _Optional[_Iterable[str]] = ..., consumers: _Optional[_Iterable[str]] = ..., internal: _Optional[bool] = ..., frequency: _Optional[str] = ..., priority: _Optional[str] = ...) -> None: ...

class Deprecation(_message.Message):
    __slots__ = ()
    DEPRECATED_FIELD_NUMBER: _ClassVar[int]
    SINCE_VERSION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    REPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    REMOVAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_GUIDE_FIELD_NUMBER: _ClassVar[int]
    deprecated: bool
    since_version: str
    reason: str
    replacement: str
    removal_version: str
    migration_guide: str
    def __init__(self, deprecated: _Optional[bool] = ..., since_version: _Optional[str] = ..., reason: _Optional[str] = ..., replacement: _Optional[str] = ..., removal_version: _Optional[str] = ..., migration_guide: _Optional[str] = ...) -> None: ...

class RiskAssessment(_message.Message):
    __slots__ = ()
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MITIGATIONS_FIELD_NUMBER: _ClassVar[int]
    REGULATORY_REFS_FIELD_NUMBER: _ClassVar[int]
    FAILURE_MODES_FIELD_NUMBER: _ClassVar[int]
    REVIEW_LEVEL_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    level: str
    category: str
    description: str
    mitigations: _containers.RepeatedScalarFieldContainer[str]
    regulatory_refs: _containers.RepeatedScalarFieldContainer[str]
    failure_modes: _containers.RepeatedCompositeFieldContainer[FailureMode]
    review_level: str
    analysis_complete: bool
    def __init__(self, level: _Optional[str] = ..., category: _Optional[str] = ..., description: _Optional[str] = ..., mitigations: _Optional[_Iterable[str]] = ..., regulatory_refs: _Optional[_Iterable[str]] = ..., failure_modes: _Optional[_Iterable[_Union[FailureMode, _Mapping]]] = ..., review_level: _Optional[str] = ..., analysis_complete: _Optional[bool] = ...) -> None: ...

class FailureMode(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    MITIGATION_FIELD_NUMBER: _ClassVar[int]
    id: str
    description: str
    probability: str
    severity: str
    detection: str
    mitigation: str
    def __init__(self, id: _Optional[str] = ..., description: _Optional[str] = ..., probability: _Optional[str] = ..., severity: _Optional[str] = ..., detection: _Optional[str] = ..., mitigation: _Optional[str] = ...) -> None: ...

class StatechartAnnotations(_message.Message):
    __slots__ = ()
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    CONTRIBUTORS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LICENSE_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    CHANGE_HISTORY_FIELD_NUMBER: _ClassVar[int]
    RISK_FIELD_NUMBER: _ClassVar[int]
    GLOSSARY_FIELD_NUMBER: _ClassVar[int]
    description: str
    summary: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    category: _containers.RepeatedScalarFieldContainer[str]
    owner: _common_pb2.Author
    contributors: _containers.RepeatedCompositeFieldContainer[_common_pb2.Author]
    version: str
    license: _common_pb2.License
    links: _containers.RepeatedCompositeFieldContainer[_common_pb2.Link]
    requirements: _containers.RepeatedScalarFieldContainer[str]
    change_history: _containers.RepeatedCompositeFieldContainer[_common_pb2.ChangeEntry]
    risk: RiskAssessment
    glossary: _containers.RepeatedCompositeFieldContainer[GlossaryEntry]
    def __init__(self, description: _Optional[str] = ..., summary: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., category: _Optional[_Iterable[str]] = ..., owner: _Optional[_Union[_common_pb2.Author, _Mapping]] = ..., contributors: _Optional[_Iterable[_Union[_common_pb2.Author, _Mapping]]] = ..., version: _Optional[str] = ..., license: _Optional[_Union[_common_pb2.License, _Mapping]] = ..., links: _Optional[_Iterable[_Union[_common_pb2.Link, _Mapping]]] = ..., requirements: _Optional[_Iterable[str]] = ..., change_history: _Optional[_Iterable[_Union[_common_pb2.ChangeEntry, _Mapping]]] = ..., risk: _Optional[_Union[RiskAssessment, _Mapping]] = ..., glossary: _Optional[_Iterable[_Union[GlossaryEntry, _Mapping]]] = ...) -> None: ...

class GlossaryEntry(_message.Message):
    __slots__ = ()
    TERM_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    RELATED_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    term: str
    definition: str
    related: _containers.RepeatedScalarFieldContainer[str]
    reference: str
    def __init__(self, term: _Optional[str] = ..., definition: _Optional[str] = ..., related: _Optional[_Iterable[str]] = ..., reference: _Optional[str] = ...) -> None: ...
