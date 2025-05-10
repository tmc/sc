---
title: statecharts.validation.v1
description: API Specification for the statecharts.validation.v1 package.
---

<a name="validator-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->


<a name="statecharts-validation-v1-SemanticValidator"></a>

### SemanticValidator

SemanticValidator service provides methods to validate statecharts and traces.
It applies semantic validation rules to ensure correctness of statechart definitions.



| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| ValidateChart | [ValidateChartRequest](#statecharts-validation-v1-ValidateChartRequest) | [ValidateChartResponse](#statecharts-validation-v1-ValidateChartResponse) | ValidateChart validates a statechart definition against semantic rules.   |
| ValidateTrace | [ValidateTraceRequest](#statecharts-validation-v1-ValidateTraceRequest) | [ValidateTraceResponse](#statecharts-validation-v1-ValidateTraceResponse) | ValidateTrace validates a statechart and a trace of machine states.   |



<!-- begin services -->



<a name="statecharts-validation-v1-ValidateChartRequest"></a>

### ValidateChartRequest

ValidateChartRequest is the request message for validating a statechart.
It contains the statechart to validate and an optional list of rules to ignore.




| Field | Type | Description |
| ----- | ---- | ----------- |
| chart |[Statechart](./statecharts.md#statecharts-v1-Statechart)|  The statechart to validate.  |
| ignore_rules[] |[RuleId](#statecharts-validation-v1-RuleId)|  Optional list of rules to ignore during validation.  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-validation-v1-ValidateTraceRequest"></a>

### ValidateTraceRequest

ValidateTraceRequest is the request message for validating a trace.
It contains the statechart and trace to validate, and an optional list of rules to ignore.




| Field | Type | Description |
| ----- | ---- | ----------- |
| chart |[Statechart](./statecharts.md#statecharts-v1-Statechart)|  The statechart definition.  |
| trace[] |[Machine](./statecharts.md#statecharts-v1-Machine)|  The trace of machine states to validate.  |
| ignore_rules[] |[RuleId](#statecharts-validation-v1-RuleId)|  Optional list of rules to ignore during validation.  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-validation-v1-ValidateChartResponse"></a>

### ValidateChartResponse

ValidateChartResponse is the response message for chart validation.
It contains a status and a list of violations found during validation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| status |Status|  The overall validation status.  |
| violations[] |[Violation](#statecharts-validation-v1-Violation)|  List of violations found, if any.  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-validation-v1-ValidateTraceResponse"></a>

### ValidateTraceResponse

ValidateTraceResponse is the response message for trace validation.
It contains a status and a list of violations found during validation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| status |Status|  The overall validation status.  |
| violations[] |[Violation](#statecharts-validation-v1-Violation)|  List of violations found, if any.  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-validation-v1-Violation"></a>

### Violation

Violation represents a rule violation found during validation.
It includes the rule that was violated, the severity, a message, and optional location hints.




| Field | Type | Description |
| ----- | ---- | ----------- |
| rule |[RuleId](#statecharts-validation-v1-RuleId)|  The rule that was violated.  |
| severity |[Severity](#statecharts-validation-v1-Severity)|  The severity of the violation.  |
| message |string|  A human-readable message describing the violation.  |
| xpath[] |string|  Location hints (optional).  |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-validation-v1-Severity"></a>

### Severity
Severity defines the severity level of a validation violation.



| Name | Number | Description |
| ---- | ------ | ----------- |
| SEVERITY_UNSPECIFIED | 0 |  Unspecified severity.  |
| INFO | 1 |  Informational message, not a violation.  |
| WARNING | 2 |  Warning, potentially problematic but not critical.  |
| ERROR | 3 |  Error, severe violation that must be fixed.  |




<a name="statecharts-validation-v1-RuleId"></a>

### RuleId
RuleId identifies specific validation rules for statecharts.



| Name | Number | Description |
| ---- | ------ | ----------- |
| RULE_UNSPECIFIED | 0 |  Unspecified rule.  |
| UNIQUE_STATE_LABELS | 1 |  All state labels must be unique.  |
| SINGLE_DEFAULT_CHILD | 2 |  XOR composite states must have exactly one default child.  |
| BASIC_HAS_NO_CHILDREN | 3 |  Basic states cannot have children.  |
| COMPOUND_HAS_CHILDREN | 4 |  Compound states must have children.  |
| DETERMINISTIC_TRANSITION_SELECTION | 5 |  Transition selection must be deterministic.  |
| NO_EVENT_BROADCAST_CYCLES | 6 |  Event broadcast must not create cycles.  |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

