import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from extensions.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExtractionMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXTRACTION_METHOD_UNSPECIFIED: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_PARSED: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_STATIC_ANALYSIS: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_DYNAMIC_TRACE: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_BINARY_ANALYSIS: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_DOCUMENTATION: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_ML_INFERENCE: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_MANUAL: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_HEURISTIC: _ClassVar[ExtractionMethod]
    EXTRACTION_METHOD_UNKNOWN: _ClassVar[ExtractionMethod]

class SourceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_TYPE_UNSPECIFIED: _ClassVar[SourceType]
    SOURCE_TYPE_C: _ClassVar[SourceType]
    SOURCE_TYPE_ASSEMBLY: _ClassVar[SourceType]
    SOURCE_TYPE_SCXML: _ClassVar[SourceType]
    SOURCE_TYPE_XSTATE: _ClassVar[SourceType]
    SOURCE_TYPE_BINARY: _ClassVar[SourceType]
    SOURCE_TYPE_DOCUMENTATION: _ClassVar[SourceType]
    SOURCE_TYPE_UML: _ClassVar[SourceType]
    SOURCE_TYPE_OTHER: _ClassVar[SourceType]
    SOURCE_TYPE_RUST: _ClassVar[SourceType]
    SOURCE_TYPE_GO: _ClassVar[SourceType]
    SOURCE_TYPE_PYTHON: _ClassVar[SourceType]
    SOURCE_TYPE_TYPESCRIPT: _ClassVar[SourceType]
    SOURCE_TYPE_PROTOBUF: _ClassVar[SourceType]

class ToolType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TOOL_TYPE_UNSPECIFIED: _ClassVar[ToolType]
    TOOL_TYPE_EXTRACTOR: _ClassVar[ToolType]
    TOOL_TYPE_IMPORTER: _ClassVar[ToolType]
    TOOL_TYPE_VALIDATOR: _ClassVar[ToolType]
    TOOL_TYPE_GENERATOR: _ClassVar[ToolType]
    TOOL_TYPE_AI: _ClassVar[ToolType]
    TOOL_TYPE_EDITOR: _ClassVar[ToolType]
    TOOL_TYPE_TRANSFORMER: _ClassVar[ToolType]

class TransformationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSFORMATION_TYPE_UNSPECIFIED: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_EXTRACTION: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_CONVERSION: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_NORMALIZATION: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_OPTIMIZATION: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_MERGE: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_SPLIT: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_MANUAL_EDIT: _ClassVar[TransformationType]
    TRANSFORMATION_TYPE_AI_EDIT: _ClassVar[TransformationType]

class SystemType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SYSTEM_TYPE_UNSPECIFIED: _ClassVar[SystemType]
    SYSTEM_TYPE_GAME_ROM: _ClassVar[SystemType]
    SYSTEM_TYPE_SOURCE_CODE: _ClassVar[SystemType]
    SYSTEM_TYPE_SCXML: _ClassVar[SystemType]
    SYSTEM_TYPE_XSTATE: _ClassVar[SystemType]
    SYSTEM_TYPE_DOCUMENTATION: _ClassVar[SystemType]
    SYSTEM_TYPE_API: _ClassVar[SystemType]
    SYSTEM_TYPE_BINARY: _ClassVar[SystemType]
    SYSTEM_TYPE_OTHER: _ClassVar[SystemType]
EXTRACTION_METHOD_UNSPECIFIED: ExtractionMethod
EXTRACTION_METHOD_PARSED: ExtractionMethod
EXTRACTION_METHOD_STATIC_ANALYSIS: ExtractionMethod
EXTRACTION_METHOD_DYNAMIC_TRACE: ExtractionMethod
EXTRACTION_METHOD_BINARY_ANALYSIS: ExtractionMethod
EXTRACTION_METHOD_DOCUMENTATION: ExtractionMethod
EXTRACTION_METHOD_ML_INFERENCE: ExtractionMethod
EXTRACTION_METHOD_MANUAL: ExtractionMethod
EXTRACTION_METHOD_HEURISTIC: ExtractionMethod
EXTRACTION_METHOD_UNKNOWN: ExtractionMethod
SOURCE_TYPE_UNSPECIFIED: SourceType
SOURCE_TYPE_C: SourceType
SOURCE_TYPE_ASSEMBLY: SourceType
SOURCE_TYPE_SCXML: SourceType
SOURCE_TYPE_XSTATE: SourceType
SOURCE_TYPE_BINARY: SourceType
SOURCE_TYPE_DOCUMENTATION: SourceType
SOURCE_TYPE_UML: SourceType
SOURCE_TYPE_OTHER: SourceType
SOURCE_TYPE_RUST: SourceType
SOURCE_TYPE_GO: SourceType
SOURCE_TYPE_PYTHON: SourceType
SOURCE_TYPE_TYPESCRIPT: SourceType
SOURCE_TYPE_PROTOBUF: SourceType
TOOL_TYPE_UNSPECIFIED: ToolType
TOOL_TYPE_EXTRACTOR: ToolType
TOOL_TYPE_IMPORTER: ToolType
TOOL_TYPE_VALIDATOR: ToolType
TOOL_TYPE_GENERATOR: ToolType
TOOL_TYPE_AI: ToolType
TOOL_TYPE_EDITOR: ToolType
TOOL_TYPE_TRANSFORMER: ToolType
TRANSFORMATION_TYPE_UNSPECIFIED: TransformationType
TRANSFORMATION_TYPE_EXTRACTION: TransformationType
TRANSFORMATION_TYPE_CONVERSION: TransformationType
TRANSFORMATION_TYPE_NORMALIZATION: TransformationType
TRANSFORMATION_TYPE_OPTIMIZATION: TransformationType
TRANSFORMATION_TYPE_MERGE: TransformationType
TRANSFORMATION_TYPE_SPLIT: TransformationType
TRANSFORMATION_TYPE_MANUAL_EDIT: TransformationType
TRANSFORMATION_TYPE_AI_EDIT: TransformationType
SYSTEM_TYPE_UNSPECIFIED: SystemType
SYSTEM_TYPE_GAME_ROM: SystemType
SYSTEM_TYPE_SOURCE_CODE: SystemType
SYSTEM_TYPE_SCXML: SystemType
SYSTEM_TYPE_XSTATE: SystemType
SYSTEM_TYPE_DOCUMENTATION: SystemType
SYSTEM_TYPE_API: SystemType
SYSTEM_TYPE_BINARY: SystemType
SYSTEM_TYPE_OTHER: SystemType

class StateProvenance(_message.Message):
    __slots__ = ()
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    EXTRACTOR_FIELD_NUMBER: _ClassVar[int]
    EXTRACTED_AT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_ID_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_NAME_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_TYPE_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    DERIVATION_FIELD_NUMBER: _ClassVar[int]
    REVIEWED_BY_FIELD_NUMBER: _ClassVar[int]
    REVIEWED_AT_FIELD_NUMBER: _ClassVar[int]
    REVIEW_STATUS_FIELD_NUMBER: _ClassVar[int]
    REVIEW_NOTES_FIELD_NUMBER: _ClassVar[int]
    ISSUES_FIELD_NUMBER: _ClassVar[int]
    source: SourceLocation
    extractor: Tool
    extracted_at: _timestamp_pb2.Timestamp
    confidence: float
    method: ExtractionMethod
    original_id: str
    original_name: str
    original_type: str
    notes: str
    derivation: _containers.RepeatedCompositeFieldContainer[DerivationStep]
    reviewed_by: str
    reviewed_at: _timestamp_pb2.Timestamp
    review_status: str
    review_notes: str
    issues: _containers.RepeatedCompositeFieldContainer[ExtractionIssue]
    def __init__(self, source: _Optional[_Union[SourceLocation, _Mapping]] = ..., extractor: _Optional[_Union[Tool, _Mapping]] = ..., extracted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., confidence: _Optional[float] = ..., method: _Optional[_Union[ExtractionMethod, str]] = ..., original_id: _Optional[str] = ..., original_name: _Optional[str] = ..., original_type: _Optional[str] = ..., notes: _Optional[str] = ..., derivation: _Optional[_Iterable[_Union[DerivationStep, _Mapping]]] = ..., reviewed_by: _Optional[str] = ..., reviewed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., review_status: _Optional[str] = ..., review_notes: _Optional[str] = ..., issues: _Optional[_Iterable[_Union[ExtractionIssue, _Mapping]]] = ...) -> None: ...

class TransitionProvenance(_message.Message):
    __slots__ = ()
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    GUARD_SOURCE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SOURCES_FIELD_NUMBER: _ClassVar[int]
    EXTRACTOR_FIELD_NUMBER: _ClassVar[int]
    EXTRACTED_AT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    GUARD_INFERRED_FIELD_NUMBER: _ClassVar[int]
    TARGET_INFERRED_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    DERIVATION_FIELD_NUMBER: _ClassVar[int]
    REVIEWED_BY_FIELD_NUMBER: _ClassVar[int]
    REVIEWED_AT_FIELD_NUMBER: _ClassVar[int]
    REVIEW_STATUS_FIELD_NUMBER: _ClassVar[int]
    ISSUES_FIELD_NUMBER: _ClassVar[int]
    source: SourceLocation
    guard_source: SourceLocation
    action_sources: _containers.RepeatedCompositeFieldContainer[SourceLocation]
    extractor: Tool
    extracted_at: _timestamp_pb2.Timestamp
    confidence: float
    method: ExtractionMethod
    original_expression: str
    guard_inferred: bool
    target_inferred: bool
    notes: str
    derivation: _containers.RepeatedCompositeFieldContainer[DerivationStep]
    reviewed_by: str
    reviewed_at: _timestamp_pb2.Timestamp
    review_status: str
    issues: _containers.RepeatedCompositeFieldContainer[ExtractionIssue]
    def __init__(self, source: _Optional[_Union[SourceLocation, _Mapping]] = ..., guard_source: _Optional[_Union[SourceLocation, _Mapping]] = ..., action_sources: _Optional[_Iterable[_Union[SourceLocation, _Mapping]]] = ..., extractor: _Optional[_Union[Tool, _Mapping]] = ..., extracted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., confidence: _Optional[float] = ..., method: _Optional[_Union[ExtractionMethod, str]] = ..., original_expression: _Optional[str] = ..., guard_inferred: _Optional[bool] = ..., target_inferred: _Optional[bool] = ..., notes: _Optional[str] = ..., derivation: _Optional[_Iterable[_Union[DerivationStep, _Mapping]]] = ..., reviewed_by: _Optional[str] = ..., reviewed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., review_status: _Optional[str] = ..., issues: _Optional[_Iterable[_Union[ExtractionIssue, _Mapping]]] = ...) -> None: ...

class SourceLocation(_message.Message):
    __slots__ = ()
    PATH_FIELD_NUMBER: _ClassVar[int]
    LINE_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    END_LINE_FIELD_NUMBER: _ClassVar[int]
    END_COLUMN_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_FIELD_NUMBER: _ClassVar[int]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    BRANCH_FIELD_NUMBER: _ClassVar[int]
    REPOSITORY_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_FIELD_NUMBER: _ClassVar[int]
    path: str
    line: int
    column: int
    end_line: int
    end_column: int
    offset: int
    length: int
    section: str
    segment: str
    commit: str
    branch: str
    repository: str
    source_type: SourceType
    checksum: str
    checksum_algorithm: str
    snippet: str
    def __init__(self, path: _Optional[str] = ..., line: _Optional[int] = ..., column: _Optional[int] = ..., end_line: _Optional[int] = ..., end_column: _Optional[int] = ..., offset: _Optional[int] = ..., length: _Optional[int] = ..., section: _Optional[str] = ..., segment: _Optional[str] = ..., commit: _Optional[str] = ..., branch: _Optional[str] = ..., repository: _Optional[str] = ..., source_type: _Optional[_Union[SourceType, str]] = ..., checksum: _Optional[str] = ..., checksum_algorithm: _Optional[str] = ..., snippet: _Optional[str] = ...) -> None: ...

class Tool(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VENDOR_FIELD_NUMBER: _ClassVar[int]
    LICENSE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    url: str
    config: _struct_pb2.Struct
    command: str
    type: ToolType
    vendor: str
    license: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., url: _Optional[str] = ..., config: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., command: _Optional[str] = ..., type: _Optional[_Union[ToolType, str]] = ..., vendor: _Optional[str] = ..., license: _Optional[str] = ...) -> None: ...

class DerivationStep(_message.Message):
    __slots__ = ()
    STEP_FIELD_NUMBER: _ClassVar[int]
    TRANSFORMATION_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TOOL_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    step: int
    transformation: TransformationType
    description: str
    tool: Tool
    timestamp: _timestamp_pb2.Timestamp
    input_id: str
    output_id: str
    params: _struct_pb2.Struct
    def __init__(self, step: _Optional[int] = ..., transformation: _Optional[_Union[TransformationType, str]] = ..., description: _Optional[str] = ..., tool: _Optional[_Union[Tool, _Mapping]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., input_id: _Optional[str] = ..., output_id: _Optional[str] = ..., params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExtractionIssue(_message.Message):
    __slots__ = ()
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    SUGGESTION_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    RELATED_FIELD_NUMBER: _ClassVar[int]
    severity: str
    code: str
    message: str
    location: SourceLocation
    suggestion: str
    resolved: bool
    resolution: str
    related: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, severity: _Optional[str] = ..., code: _Optional[str] = ..., message: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ..., suggestion: _Optional[str] = ..., resolved: _Optional[bool] = ..., resolution: _Optional[str] = ..., related: _Optional[_Iterable[str]] = ...) -> None: ...

class ChartProvenance(_message.Message):
    __slots__ = ()
    SOURCE_SYSTEM_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_AT_FIELD_NUMBER: _ClassVar[int]
    LICENSE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    REPORT_URL_FIELD_NUMBER: _ClassVar[int]
    DERIVED_FROM_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    EXTRACTION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    REVIEW_STATUS_FIELD_NUMBER: _ClassVar[int]
    source_system: SourceSystem
    pipeline: _containers.RepeatedCompositeFieldContainer[Tool]
    created_at: _timestamp_pb2.Timestamp
    modified_at: _timestamp_pb2.Timestamp
    license: _common_pb2.License
    attribution: str
    report_url: str
    derived_from: str
    generation: int
    extraction_config: _struct_pb2.Struct
    stats: ExtractionStats
    confidence: float
    review_status: str
    def __init__(self, source_system: _Optional[_Union[SourceSystem, _Mapping]] = ..., pipeline: _Optional[_Iterable[_Union[Tool, _Mapping]]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., modified_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., license: _Optional[_Union[_common_pb2.License, _Mapping]] = ..., attribution: _Optional[str] = ..., report_url: _Optional[str] = ..., derived_from: _Optional[str] = ..., generation: _Optional[int] = ..., extraction_config: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., stats: _Optional[_Union[ExtractionStats, _Mapping]] = ..., confidence: _Optional[float] = ..., review_status: _Optional[str] = ...) -> None: ...

class SourceSystem(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    type: SystemType
    platform: str
    checksum: str
    checksum_algorithm: str
    url: str
    metadata: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., type: _Optional[_Union[SystemType, str]] = ..., platform: _Optional[str] = ..., checksum: _Optional[str] = ..., checksum_algorithm: _Optional[str] = ..., url: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExtractionStats(_message.Message):
    __slots__ = ()
    class IssuesBySeverityEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class ConfidenceDistributionEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    STATES_EXTRACTED_FIELD_NUMBER: _ClassVar[int]
    TRANSITIONS_EXTRACTED_FIELD_NUMBER: _ClassVar[int]
    EVENTS_DISCOVERED_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FILES_FIELD_NUMBER: _ClassVar[int]
    LINES_OF_CODE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    ISSUES_BY_SEVERITY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    NEEDS_REVIEW_FIELD_NUMBER: _ClassVar[int]
    states_extracted: int
    transitions_extracted: int
    events_discovered: int
    source_files: int
    lines_of_code: int
    duration_ms: int
    source_coverage: float
    issues_by_severity: _containers.ScalarMap[str, int]
    confidence_distribution: _containers.ScalarMap[str, int]
    needs_review: int
    def __init__(self, states_extracted: _Optional[int] = ..., transitions_extracted: _Optional[int] = ..., events_discovered: _Optional[int] = ..., source_files: _Optional[int] = ..., lines_of_code: _Optional[int] = ..., duration_ms: _Optional[int] = ..., source_coverage: _Optional[float] = ..., issues_by_severity: _Optional[_Mapping[str, int]] = ..., confidence_distribution: _Optional[_Mapping[str, int]] = ..., needs_review: _Optional[int] = ...) -> None: ...

class ExtractionReport(_message.Message):
    __slots__ = ()
    ID_FIELD_NUMBER: _ClassVar[int]
    GENERATED_AT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    ISSUES_FIELD_NUMBER: _ClassVar[int]
    LOW_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    NEEDS_REVIEW_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDATIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    generated_at: _timestamp_pb2.Timestamp
    status: str
    provenance: ChartProvenance
    stats: ExtractionStats
    issues: _containers.RepeatedCompositeFieldContainer[ExtractionIssue]
    low_confidence: _containers.RepeatedCompositeFieldContainer[LowConfidenceElement]
    needs_review: _containers.RepeatedScalarFieldContainer[str]
    recommendations: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., generated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., status: _Optional[str] = ..., provenance: _Optional[_Union[ChartProvenance, _Mapping]] = ..., stats: _Optional[_Union[ExtractionStats, _Mapping]] = ..., issues: _Optional[_Iterable[_Union[ExtractionIssue, _Mapping]]] = ..., low_confidence: _Optional[_Iterable[_Union[LowConfidenceElement, _Mapping]]] = ..., needs_review: _Optional[_Iterable[str]] = ..., recommendations: _Optional[_Iterable[str]] = ...) -> None: ...

class LowConfidenceElement(_message.Message):
    __slots__ = ()
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    VERIFICATION_STEPS_FIELD_NUMBER: _ClassVar[int]
    element_type: str
    label: str
    confidence: float
    reason: str
    verification_steps: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, element_type: _Optional[str] = ..., label: _Optional[str] = ..., confidence: _Optional[float] = ..., reason: _Optional[str] = ..., verification_steps: _Optional[_Iterable[str]] = ...) -> None: ...
