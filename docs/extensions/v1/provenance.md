---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="provenance-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-StateProvenance"></a>

### StateProvenance

StateProvenance tracks the origin of a state.

STORAGE:
Pack into State.extensions using google.protobuf.Any.

CONFIDENCE SEMANTICS:
The confidence score represents extraction certainty:
  - 1.0: Explicit state declaration found (e.g., `state "Idle"`)
  - 0.8: High-confidence inference (e.g., clear switch case)
  - 0.5: Moderate inference (e.g., inferred from control flow)
  - 0.2: Low-confidence guess (e.g., naming heuristic)
  - 0.0: Placeholder or unknown




| Field | Type | Description |
| ----- | ---- | ----------- |
| source |[SourceLocation](#extensions-v1-SourceLocation)| Source location where state was extracted from.   |
| extractor |[Tool](#extensions-v1-Tool)| Extraction tool information.   |
| extracted_at |Timestamp| Extraction timestamp.   |
| confidence |double| Confidence score: 0.0 (guess) to 1.0 (certain).   |
| method |[ExtractionMethod](#extensions-v1-ExtractionMethod)| Extraction method.   |
| original_id |string| Original identifier in source system. May differ from current label due to normalization.   |
| original_name |string| Original name/label in source system.   |
| original_type |string| Original type in source system (e.g., SCXML state type).   |
| notes |string| Free-form notes from extraction process.   |
| derivation[] |[DerivationStep](#extensions-v1-DerivationStep)| Derivation chain (if derived from another state).   |
| reviewed_by |string| Human reviewer (if manually verified).   |
| reviewed_at |Timestamp| Review timestamp.   |
| review_status |string| Review status: "pending", "approved", "rejected", "needs_revision".   |
| review_notes |string| Review notes.   |
| issues[] |[ExtractionIssue](#extensions-v1-ExtractionIssue)| Issues/warnings from extraction.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TransitionProvenance"></a>

### TransitionProvenance

TransitionProvenance tracks the origin of a transition.

STORAGE:
Pack into Transition.extensions using google.protobuf.Any.




| Field | Type | Description |
| ----- | ---- | ----------- |
| source |[SourceLocation](#extensions-v1-SourceLocation)| Source location where transition was extracted from.   |
| guard_source |[SourceLocation](#extensions-v1-SourceLocation)| Guard expression source location (may differ from transition).   |
| action_sources[] |[SourceLocation](#extensions-v1-SourceLocation)| Action source locations.   |
| extractor |[Tool](#extensions-v1-Tool)| Extraction tool information.   |
| extracted_at |Timestamp| Extraction timestamp.   |
| confidence |double| Confidence score.   |
| method |[ExtractionMethod](#extensions-v1-ExtractionMethod)| Extraction method.   |
| original_expression |string| Original representation in source. Example: C conditional, assembly branch.   |
| guard_inferred |bool| Whether guard was inferred (vs explicit).   |
| target_inferred |bool| Whether target was inferred (vs explicit).   |
| notes |string| Notes from extraction.   |
| derivation[] |[DerivationStep](#extensions-v1-DerivationStep)| Derivation chain.   |
| reviewed_by |string| Review information.   |
| reviewed_at |Timestamp|   |
| review_status |string|   |
| issues[] |[ExtractionIssue](#extensions-v1-ExtractionIssue)| Issues from extraction.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SourceLocation"></a>

### SourceLocation

SourceLocation identifies a location in source material.

COORDINATE SYSTEMS:
For text sources:
  - line/column: 1-indexed, human-readable
  - offset/length: 0-indexed bytes for machine processing

For binary sources:
  - offset: byte offset in binary
  - length: span length in bytes
  - section: binary section name

VERSION CONTROL:
When available, include commit hash and branch for precise reference.
This enables:
  - Reproducible extraction
  - Drift detection on re-extraction
  - Blame/attribution




| Field | Type | Description |
| ----- | ---- | ----------- |
| path |string| File path or URL (relative or absolute).   |
| line |int32| Line number (1-indexed, for text sources).   |
| column |int32| Column number (1-indexed, for text sources).   |
| end_line |int32| End line (for multi-line spans).   |
| end_column |int32| End column.   |
| offset |int64| Byte offset from file start (0-indexed).   |
| length |int64| Span length in bytes.   |
| section |string| Binary section name (for binaries). Example: ".text", ".data", ".rodata".   |
| segment |string| Binary segment (for segmented formats). Example: SNES bank number, PE section.   |
| commit |string| Git commit hash.   |
| branch |string| Branch or tag name.   |
| repository |string| Repository URL.   |
| source_type |[SourceType](#extensions-v1-SourceType)| Source type classifier.   |
| checksum |string| Checksum of source file.   |
| checksum_algorithm |string| Checksum algorithm: "sha256", "md5", "crc32".   |
| snippet |string| Human-readable snippet of source (for context).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Tool"></a>

### Tool

Tool identifies a tool that processed the statechart.

CHAIN OF CUSTODY:
For multi-stage extraction pipelines, each tool is recorded
in the derivation chain, preserving full provenance.




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string| Tool name. Example: "zelda3-extractor", "scxml-importer", "claude-code".   |
| version |string| Tool version (SemVer preferred).   |
| url |string| Tool URL/homepage.   |
| config |Struct| Tool configuration used.   |
| command |string| Exact invocation command.   |
| type |[ToolType](#extensions-v1-ToolType)| Tool type classifier.   |
| vendor |string| Tool vendor/author.   |
| license |string| Tool license.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-DerivationStep"></a>

### DerivationStep

DerivationStep records a transformation in the derivation chain.

PROVENANCE MODEL [PROV-O]:
Each step represents a prov:wasDerivedFrom relationship:
  entity₂ prov:wasDerivedFrom entity₁
with additional activity and agent information.




| Field | Type | Description |
| ----- | ---- | ----------- |
| step |int32| Step index in derivation chain (0 = original).   |
| transformation |[TransformationType](#extensions-v1-TransformationType)| Transformation type.   |
| description |string| Description of transformation.   |
| tool |[Tool](#extensions-v1-Tool)| Tool that performed transformation.   |
| timestamp |Timestamp| Timestamp of transformation.   |
| input_id |string| Input state/entity (before transformation).   |
| output_id |string| Output state/entity (after transformation).   |
| params |Struct| Transformation parameters.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ExtractionIssue"></a>

### ExtractionIssue

ExtractionIssue records a problem encountered during extraction.

ISSUE TRACKING:
Issues are informational - they don't prevent extraction but
flag areas requiring attention.




| Field | Type | Description |
| ----- | ---- | ----------- |
| severity |string| Issue severity: "error", "warning", "info", "hint".   |
| code |string| Issue code/type for categorization.   |
| message |string| Human-readable message.   |
| location |[SourceLocation](#extensions-v1-SourceLocation)| Source location where issue occurred.   |
| suggestion |string| Suggested fix.   |
| resolved |bool| Whether issue has been resolved.   |
| resolution |string| Resolution notes.   |
| related[] |string| Related elements (labels).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ChartProvenance"></a>

### ChartProvenance

ChartProvenance tracks statechart-level provenance.

STORAGE:
Store in Statechart-level metadata or as a dedicated extension.




| Field | Type | Description |
| ----- | ---- | ----------- |
| source_system |[SourceSystem](#extensions-v1-SourceSystem)| Primary source system.   |
| pipeline[] |[Tool](#extensions-v1-Tool)| Extraction pipeline (ordered tools applied).   |
| created_at |Timestamp| Overall extraction timestamp.   |
| modified_at |Timestamp| Last modification timestamp.   |
| license |[License](./common.md#extensions-v1-License)| License information.   |
| attribution |string| Attribution/copyright notice.   |
| report_url |string| Link to extraction report.   |
| derived_from |string| Parent chart ID (if derived/forked).   |
| generation |int32| Generation number (increments on each derivation).   |
| extraction_config |Struct| Extraction configuration.   |
| stats |[ExtractionStats](#extensions-v1-ExtractionStats)| Aggregate extraction statistics.   |
| confidence |double| Overall confidence score.   |
| review_status |string| Review status.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SourceSystem"></a>

### SourceSystem

SourceSystem identifies the original system.

EXAMPLES:
- Game ROM: {name: "zelda3", version: "1.0", type: "game_rom", platform: "snes"}
- Linux kernel: {name: "linux", version: "6.1", type: "source_code", platform: "linux"}
- AWS: {name: "step_functions", version: "2023-01", type: "api", platform: "aws"}




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string| System name.   |
| version |string| System version.   |
| type |[SystemType](#extensions-v1-SystemType)| System type.   |
| platform |string| Platform/runtime.   |
| checksum |string| File/artifact checksum.   |
| checksum_algorithm |string| Checksum algorithm.   |
| url |string| Source URL or identifier.   |
| metadata |Struct| Additional system metadata.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ExtractionStats"></a>

### ExtractionStats

ExtractionStats aggregates extraction statistics.




| Field | Type | Description |
| ----- | ---- | ----------- |
| states_extracted |int32| Total states extracted.   |
| transitions_extracted |int32| Total transitions extracted.   |
| events_discovered |int32| Total events discovered.   |
| source_files |int32| Source files analyzed.   |
| lines_of_code |int64| Lines of code analyzed.   |
| duration_ms |int64| Extraction duration.   |
| source_coverage |double| Coverage of source analyzed.   |
| issues_by_severity |[ExtractionStats.IssuesBySeverityEntry](#extensions-v1-ExtractionStats-IssuesBySeverityEntry)| Issues by severity.   |
| confidence_distribution |[ExtractionStats.ConfidenceDistributionEntry](#extensions-v1-ExtractionStats-ConfidenceDistributionEntry)| Confidence distribution.   |
| needs_review |int32| Elements requiring review.   |






<a name="extensions-v1-ExtractionStats-IssuesBySeverityEntry"></a>

### IssuesBySeverityEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |int32|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ExtractionStats-ConfidenceDistributionEntry"></a>

### ConfidenceDistributionEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |int32|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ExtractionReport"></a>

### ExtractionReport

ExtractionReport provides a detailed extraction summary.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Report identifier.   |
| generated_at |Timestamp| Report generation timestamp.   |
| status |string| Overall status: "success", "partial", "failed".   |
| provenance |[ChartProvenance](#extensions-v1-ChartProvenance)| Chart provenance summary.   |
| stats |[ExtractionStats](#extensions-v1-ExtractionStats)| Extraction statistics.   |
| issues[] |[ExtractionIssue](#extensions-v1-ExtractionIssue)| All issues encountered.   |
| low_confidence[] |[LowConfidenceElement](#extensions-v1-LowConfidenceElement)| Elements with low confidence.   |
| needs_review[] |string| Elements needing review.   |
| recommendations[] |string| Recommendations for improving extraction.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-LowConfidenceElement"></a>

### LowConfidenceElement

LowConfidenceElement flags an element with low extraction confidence.




| Field | Type | Description |
| ----- | ---- | ----------- |
| element_type |string| Element type: "state", "transition", "guard", "action".   |
| label |string| Element label.   |
| confidence |double| Confidence score.   |
| reason |string| Reason for low confidence.   |
| verification_steps[] |string| Suggested verification steps.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="extensions-v1-ExtractionMethod"></a>

### ExtractionMethod
ExtractionMethod classifies how an element was extracted.



| Name | Number | Description |
| ---- | ------ | ----------- |
| EXTRACTION_METHOD_UNSPECIFIED | 0 |   |
| EXTRACTION_METHOD_PARSED | 1 | Parsed from explicit declaration (highest confidence). Example: SCXML <state id="...">, XState machine definition.   |
| EXTRACTION_METHOD_STATIC_ANALYSIS | 2 | Static analysis of source code. Example: switch statement analysis, enum extraction.   |
| EXTRACTION_METHOD_DYNAMIC_TRACE | 3 | Dynamic tracing/recording. Example: Runtime instrumentation, debugger traces.   |
| EXTRACTION_METHOD_BINARY_ANALYSIS | 4 | Binary/ROM analysis. Example: Disassembly, memory pattern matching.   |
| EXTRACTION_METHOD_DOCUMENTATION | 5 | Documentation analysis. Example: Extracted from design docs, diagrams.   |
| EXTRACTION_METHOD_ML_INFERENCE | 6 | Machine learning/AI inference. Example: LLM analysis, pattern recognition.   |
| EXTRACTION_METHOD_MANUAL | 7 | Manual specification. Example: Hand-authored by human.   |
| EXTRACTION_METHOD_HEURISTIC | 8 | Heuristic-based inference. Example: Naming conventions, structural patterns.   |
| EXTRACTION_METHOD_UNKNOWN | 9 | Unknown or mixed methods.   |




<a name="extensions-v1-SourceType"></a>

### SourceType
SourceType classifies the type of source material.



| Name | Number | Description |
| ---- | ------ | ----------- |
| SOURCE_TYPE_UNSPECIFIED | 0 |   |
| SOURCE_TYPE_C | 1 | C/C++ source code.   |
| SOURCE_TYPE_ASSEMBLY | 2 | Assembly language.   |
| SOURCE_TYPE_SCXML | 3 | SCXML state machine definition.   |
| SOURCE_TYPE_XSTATE | 4 | XState/Stately machine definition.   |
| SOURCE_TYPE_BINARY | 5 | Binary/ROM file.   |
| SOURCE_TYPE_DOCUMENTATION | 6 | Documentation (Markdown, text, PDF).   |
| SOURCE_TYPE_UML | 7 | UML diagram.   |
| SOURCE_TYPE_OTHER | 8 | Other structured format.   |
| SOURCE_TYPE_RUST | 9 | Rust source code.   |
| SOURCE_TYPE_GO | 10 | Go source code.   |
| SOURCE_TYPE_PYTHON | 11 | Python source code.   |
| SOURCE_TYPE_TYPESCRIPT | 12 | TypeScript/JavaScript source code.   |
| SOURCE_TYPE_PROTOBUF | 13 | Protocol buffer definition.   |




<a name="extensions-v1-ToolType"></a>

### ToolType
ToolType classifies the type of extraction/processing tool.



| Name | Number | Description |
| ---- | ------ | ----------- |
| TOOL_TYPE_UNSPECIFIED | 0 |   |
| TOOL_TYPE_EXTRACTOR | 1 | Source code extractor.   |
| TOOL_TYPE_IMPORTER | 2 | Format converter/importer.   |
| TOOL_TYPE_VALIDATOR | 3 | Validator/verifier.   |
| TOOL_TYPE_GENERATOR | 4 | Code generator.   |
| TOOL_TYPE_AI | 5 | AI/ML model.   |
| TOOL_TYPE_EDITOR | 6 | Human editor.   |
| TOOL_TYPE_TRANSFORMER | 7 | Transformer/optimizer.   |




<a name="extensions-v1-TransformationType"></a>

### TransformationType
TransformationType classifies derivation transformations.



| Name | Number | Description |
| ---- | ------ | ----------- |
| TRANSFORMATION_TYPE_UNSPECIFIED | 0 |   |
| TRANSFORMATION_TYPE_EXTRACTION | 1 | Direct extraction from source.   |
| TRANSFORMATION_TYPE_CONVERSION | 2 | Format conversion (e.g., SCXML → proto).   |
| TRANSFORMATION_TYPE_NORMALIZATION | 3 | Normalization (e.g., label cleanup).   |
| TRANSFORMATION_TYPE_OPTIMIZATION | 4 | Optimization (e.g., state minimization).   |
| TRANSFORMATION_TYPE_MERGE | 5 | Merge from multiple sources.   |
| TRANSFORMATION_TYPE_SPLIT | 6 | Split into sub-machines.   |
| TRANSFORMATION_TYPE_MANUAL_EDIT | 7 | Manual edit by human.   |
| TRANSFORMATION_TYPE_AI_EDIT | 8 | AI-assisted modification.   |




<a name="extensions-v1-SystemType"></a>

### SystemType
SystemType classifies source system types.



| Name | Number | Description |
| ---- | ------ | ----------- |
| SYSTEM_TYPE_UNSPECIFIED | 0 |   |
| SYSTEM_TYPE_GAME_ROM | 1 |   |
| SYSTEM_TYPE_SOURCE_CODE | 2 |   |
| SYSTEM_TYPE_SCXML | 3 |   |
| SYSTEM_TYPE_XSTATE | 4 |   |
| SYSTEM_TYPE_DOCUMENTATION | 5 |   |
| SYSTEM_TYPE_API | 6 |   |
| SYSTEM_TYPE_BINARY | 7 |   |
| SYSTEM_TYPE_OTHER | 8 |   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

