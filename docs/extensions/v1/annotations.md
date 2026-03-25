---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="annotations-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-StateAnnotations"></a>

### StateAnnotations

StateAnnotations provides documentation for a state.

STORAGE:
Pack into State.extensions using google.protobuf.Any.

SEMANTIC PROPERTIES:
- Annotations are ADVISORY; they don't affect execution semantics
- Tools SHOULD display annotations but MAY ignore unknown fields
- Empty annotations are equivalent to no annotation




| Field | Type | Description |
| ----- | ---- | ----------- |
| description |string| Full description of the state (Markdown supported). Explain purpose, invariants, and relationships to other states.   |
| summary |string| Brief summary for tooltips and listings (plain text, ~80 chars).   |
| tags[] |string| Semantic tags for categorization and filtering. Examples: ["error-handling", "user-input", "background-task"]. Tags SHOULD be lowercase, hyphenated, and hierarchical-capable.   |
| category[] |string| Category path for hierarchical organization. Example: ["ui", "modal", "confirmation"] represents ui.modal.confirmation.   |
| author |[Author](./common.md#extensions-v1-Author)| Author attribution.   |
| since_version |string| Semantic version when this state was introduced. Follows SemVer: "1.2.0", "2.0.0-beta.1".   |
| deprecation |[Deprecation](#extensions-v1-Deprecation)| Deprecation notice (if state is deprecated).   |
| related_states[] |string| Related states for documentation cross-references. Contains state labels for linking.   |
| links[] |[Link](./common.md#extensions-v1-Link)| External documentation links.   |
| requirements[] |string| Requirement traceability identifiers. Examples: ["REQ-001", "US-123", "JIRA-456"].   |
| risk |[RiskAssessment](#extensions-v1-RiskAssessment)| Risk assessment for safety-critical states.   |
| change_history[] |[ChangeEntry](./common.md#extensions-v1-ChangeEntry)| Change history entries.   |
| invariants[] |string| Invariants that hold while in this state (human-readable).   |
| preconditions[] |string| Preconditions for entering this state (human-readable).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TransitionAnnotations"></a>

### TransitionAnnotations

TransitionAnnotations provides documentation for a transition.

STORAGE:
Pack into Transition.extensions using google.protobuf.Any.




| Field | Type | Description |
| ----- | ---- | ----------- |
| description |string| Full description of the transition (Markdown supported).   |
| summary |string| Brief summary for tooltips (plain text, ~80 chars).   |
| tags[] |string| Semantic tags for categorization.   |
| author |[Author](./common.md#extensions-v1-Author)| Author attribution.   |
| since_version |string| Version when this transition was introduced.   |
| deprecation |[Deprecation](#extensions-v1-Deprecation)| Deprecation notice.   |
| requirements[] |string| Requirement traceability identifiers.   |
| preconditions[] |string| Preconditions beyond the guard (human-readable documentation). These complement the formal guard expression.   |
| postconditions[] |string| Postconditions established by this transition (human-readable).   |
| side_effects[] |string| Side effects of executing this transition. Examples: ["sends email notification", "updates database", "logs event"].   |
| change_history[] |[ChangeEntry](./common.md#extensions-v1-ChangeEntry)| Change history entries.   |
| performance_notes |string| Performance notes (e.g., "may be slow on large inputs").   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-EventAnnotations"></a>

### EventAnnotations

EventAnnotations provides documentation for an event.

STORAGE:
Can be linked via event label lookup or stored in Statechart-level metadata.
Since Event messages are lightweight in core schema, annotations provide
additional documentation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| event_label |string| Event label (for linking).   |
| description |string| Full description of the event (Markdown supported).   |
| payload_schema |string| Event payload schema in JSON Schema format. Allows tooling to validate event payloads.   |
| examples[] |string| Example payloads as JSON strings.   |
| producers[] |string| Systems/components that produce this event.   |
| consumers[] |string| States that react to this event.   |
| internal |bool| Whether this is an internal or external event.   |
| frequency |string| Event frequency: "rare", "occasional", "frequent", "continuous".   |
| priority |string| Priority/urgency: "low", "normal", "high", "critical".   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Deprecation"></a>

### Deprecation

Deprecation marks an element as deprecated.

SEMANTICS:
Deprecated elements are still functional but SHOULD be avoided.
Tools MAY emit warnings when deprecated elements are used.




| Field | Type | Description |
| ----- | ---- | ----------- |
| deprecated |bool| Whether this element is deprecated.   |
| since_version |string| Version when deprecation was announced.   |
| reason |string| Reason for deprecation (Markdown supported).   |
| replacement |string| Label of replacement element (if any).   |
| removal_version |string| Version when element will be removed.   |
| migration_guide |string| Migration instructions (Markdown supported).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-RiskAssessment"></a>

### RiskAssessment

RiskAssessment captures safety and compliance information.

USE CASES:
- Safety-critical systems (automotive, aerospace, medical)
- Security-sensitive states (authentication, authorization)
- Compliance tracking (GDPR, HIPAA, SOX)

REFERENCES:
[ISO26262] Automotive Safety Integrity Levels (ASIL)
[DO-178C] Design Assurance Levels (DAL)




| Field | Type | Description |
| ----- | ---- | ----------- |
| level |string| Risk level: "negligible", "low", "medium", "high", "critical".   |
| category |string| Risk category: "safety", "security", "compliance", "performance", "availability".   |
| description |string| Detailed risk description (Markdown supported).   |
| mitigations[] |string| Mitigation measures in place.   |
| regulatory_refs[] |string| Regulatory references: ["ISO-26262-ASIL-B", "DO-178C-DAL-C", "GDPR-Art-32"].   |
| failure_modes[] |[FailureMode](#extensions-v1-FailureMode)| Failure modes and effects.   |
| review_level |string| Required review level: "none", "peer", "expert", "formal".   |
| analysis_complete |bool| Whether safety analysis is complete.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-FailureMode"></a>

### FailureMode

FailureMode documents a potential failure scenario.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Failure mode identifier.   |
| description |string| Description of the failure.   |
| probability |string| Probability: "improbable", "remote", "occasional", "probable", "frequent".   |
| severity |string| Severity: "negligible", "minor", "major", "hazardous", "catastrophic".   |
| detection |string| Detection method.   |
| mitigation |string| Mitigation strategy.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-StatechartAnnotations"></a>

### StatechartAnnotations

StatechartAnnotations provides documentation for the entire statechart.

STORAGE:
Store in Statechart-level metadata or as a dedicated extension type.




| Field | Type | Description |
| ----- | ---- | ----------- |
| description |string| Full description of the statechart's purpose (Markdown supported).   |
| summary |string| Brief summary.   |
| tags[] |string| Semantic tags.   |
| category[] |string| Category path.   |
| owner |[Author](./common.md#extensions-v1-Author)| Primary author/owner.   |
| contributors[] |[Author](./common.md#extensions-v1-Author)| Contributors.   |
| version |string| Current version.   |
| license |[License](./common.md#extensions-v1-License)| License information.   |
| links[] |[Link](./common.md#extensions-v1-Link)| External links.   |
| requirements[] |string| Requirement traceability.   |
| change_history[] |[ChangeEntry](./common.md#extensions-v1-ChangeEntry)| Change history.   |
| risk |[RiskAssessment](#extensions-v1-RiskAssessment)| Overall risk assessment.   |
| glossary[] |[GlossaryEntry](#extensions-v1-GlossaryEntry)| Glossary of domain terms.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-GlossaryEntry"></a>

### GlossaryEntry

GlossaryEntry defines a domain term.




| Field | Type | Description |
| ----- | ---- | ----------- |
| term |string| The term being defined.   |
| definition |string| Definition (Markdown supported).   |
| related[] |string| Related terms.   |
| reference |string| External reference for the term.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

