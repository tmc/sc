---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="testing-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-TestSuite"></a>

### TestSuite

TestSuite groups related test cases.

STORAGE:
Can be attached to Statechart-level extensions or stored separately.

EXECUTION SEMANTICS:
1. Run setup actions before each test
2. Execute test cases (order may vary based on priority)
3. Run teardown actions after each test
4. Aggregate results




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Unique identifier for the suite.   |
| name |string| Human-readable suite name.   |
| description |string| Suite description (Markdown supported).   |
| test_cases[] |[TestCase](#extensions-v1-TestCase)| Test cases in this suite.   |
| setup[] |[TestAction](#extensions-v1-TestAction)| Setup actions run before EACH test case.   |
| teardown[] |[TestAction](#extensions-v1-TestAction)| Teardown actions run after EACH test case.   |
| tags[] |string| Tags for filtering: ["smoke", "integration", "slow"].   |
| timeout |Duration| Suite-level timeout (applies to all tests).   |
| parallel |bool| Whether to run tests in parallel.   |
| max_parallelism |int32| Maximum parallel test count.   |
| fail_fast |bool| Fail-fast: stop suite on first failure.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestCase"></a>

### TestCase

TestCase defines a single test scenario.

TEST STRUCTURE:
A test case specifies:
  1. Initial context (variables, configuration)
  2. Event sequence to send
  3. Assertions to verify after execution

EXECUTION SEMANTICS [MBT]:
1. Reset machine to initial configuration
2. Apply initial_context overrides
3. For each event in sequence:
   a. Wait delay_ms (if specified)
   b. Send event with payload
   c. Process run-to-completion
   d. Check timing-based assertions
4. Verify final assertions




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Unique test identifier.   |
| name |string| Human-readable test name.   |
| description |string| Test description (Markdown supported).   |
| initial_context |Struct| Initial context/variables override. Applied after machine reset, before event sequence.   |
| initial_states[] |string| Initial configuration override. If set, machine starts in this configuration instead of default.   |
| events[] |[TestEvent](#extensions-v1-TestEvent)| Event sequence to execute.   |
| assertions[] |[TestAssertion](#extensions-v1-TestAssertion)| Assertions to verify.   |
| invariants[] |[TestInvariant](#extensions-v1-TestInvariant)| Invariants that must hold throughout execution.   |
| enabled |bool| Whether test is enabled.   |
| skip_reason |string| Skip reason (if disabled).   |
| tags[] |string| Tags for filtering: ["smoke", "regression", "edge-case"].   |
| timeout |Duration| Test timeout.   |
| priority |int32| Priority: higher = run first.   |
| expected_outcome |string| Expected outcome: "pass", "fail", "skip", "flaky".   |
| known_issues[] |string| Known issues that cause expected failures.   |
| author |string| Author attribution.   |
| requirements[] |string| Requirement traceability.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestEvent"></a>

### TestEvent

TestEvent represents an event to send during test execution.

TIMING:
- delay_before: wait before sending this event
- Events are sent in order, one at a time
- Each event triggers run-to-completion before next




| Field | Type | Description |
| ----- | ---- | ----------- |
| event |string| Event name (must match defined events).   |
| payload |Struct| Event payload as structured data.   |
| delay_before |Duration| Delay before sending this event.   |
| comment |string| Description of what this event tests.   |
| assertions[] |[TestAssertion](#extensions-v1-TestAssertion)| Assertions to check AFTER this specific event.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestAction"></a>

### TestAction

TestAction represents a setup/teardown action.

ACTION TYPES:
- "set_context": Set context variable(s)
- "reset": Reset machine to initial state
- "invoke": Call external function/mock
- "wait": Delay for specified duration
- "log": Log a message




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |string| Action type.   |
| params |Struct| Action-specific parameters.   |
| comment |string| Description of action purpose.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestAssertion"></a>

### TestAssertion

TestAssertion defines an expected outcome to verify.

ASSERTION SEMANTICS:
Assertions are evaluated at specific timing points:
- "after_event": After specific event in sequence
- "final": After all events complete
- "any_time": Must hold at some point during execution
- "always": Must hold at every step (use TestInvariant instead)




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |[AssertionType](#extensions-v1-AssertionType)| Assertion type.   |
| expected |string| Expected value (interpretation depends on type). For state assertions: state label For configuration: comma-separated labels or JSON array For context: JSON value   |
| field |string| Field name for context assertions.   |
| message |string| Custom failure message.   |
| timing |string| When to check: "after_event", "final", "any_time".   |
| after_event_index |int32| Event index after which to check (for "after_event" timing). 0 = after first event, -1 = after last event.   |
| negated |bool| Negation: assert the opposite.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestInvariant"></a>

### TestInvariant

TestInvariant defines a property that must hold throughout execution.

SEMANTICS:
Invariants are checked at every step of execution, not just at end.
Violation at any step causes test failure.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Invariant identifier.   |
| name |string| Human-readable name.   |
| description |string| Description of what this invariant ensures.   |
| expression |string| Invariant expression (CEL). Variables available: configuration, context, event, step   |
| violation_message |string| Error message on violation.   |
| severity |string| Severity: "error", "warning".   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-StateCoverageMarker"></a>

### StateCoverageMarker

StateCoverageMarker tracks coverage for a state.

STORAGE:
Pack into State.extensions.

COVERAGE REQUIREMENTS:
- "must_enter": State must be entered at least once
- "must_exit": State must be exited (not stuck)
- "must_dwell": State must be occupied for min_dwell_ms
- "all_exits": All outgoing transitions must be taken




| Field | Type | Description |
| ----- | ---- | ----------- |
| requirements[] |string| Coverage requirements for this state.   |
| min_dwell |Duration| Minimum dwell time for "must_dwell" requirement.   |
| covered |bool| Whether state was covered in last run.   |
| coverage_count |int64| Total coverage count across runs.   |
| entry_count |int64| Entry count (times state was entered).   |
| exit_count |int64| Exit count (times state was exited).   |
| covered_by[] |string| Test case IDs that cover this state.   |
| total_dwell_time |Duration| Total time spent in state across all runs.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TransitionCoverageMarker"></a>

### TransitionCoverageMarker

TransitionCoverageMarker tracks coverage for a transition.

STORAGE:
Pack into Transition.extensions.

COVERAGE REQUIREMENTS:
- "must_fire": Transition must fire at least once
- "guard_true": Transition must fire with guard evaluating true
- "guard_false": Guard must evaluate false at least once




| Field | Type | Description |
| ----- | ---- | ----------- |
| requirements[] |string| Coverage requirements for this transition.   |
| covered |bool| Whether transition was covered in last run.   |
| fire_count |int64| Total fire count across runs.   |
| covered_by[] |string| Test case IDs that cover this transition.   |
| guard_true_covered |bool| Guard evaluation coverage.   |
| guard_false_covered |bool|   |
| guard_true_count |int64| Count of guard true evaluations.   |
| guard_false_count |int64| Count of guard false evaluations.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-CoverageReport"></a>

### CoverageReport

CoverageReport summarizes test coverage results.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Report identifier.   |
| generated_at |Timestamp| Report generation timestamp.   |
| statechart_id |string| Statechart identifier.   |
| test_run_id |string| Test suite/run identifier.   |
| state_coverage |[CoverageStats](#extensions-v1-CoverageStats)| State coverage statistics.   |
| transition_coverage |[CoverageStats](#extensions-v1-CoverageStats)| Transition coverage statistics.   |
| event_coverage |[CoverageStats](#extensions-v1-CoverageStats)| Event coverage statistics.   |
| guard_coverage |[CoverageStats](#extensions-v1-CoverageStats)| Guard expression coverage.   |
| uncovered_states[] |string| Uncovered states.   |
| uncovered_transitions[] |string| Uncovered transitions.   |
| suggestions[] |[SuggestedTestCase](#extensions-v1-SuggestedTestCase)| Suggested test cases to improve coverage.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-CoverageStats"></a>

### CoverageStats

CoverageStats provides coverage statistics.




| Field | Type | Description |
| ----- | ---- | ----------- |
| total |int32| Total items.   |
| covered |int32| Covered items.   |
| percentage |double| Coverage percentage: covered/total * 100.   |
| requirement_met |bool| Coverage requirement met.   |
| required_percentage |double| Required coverage percentage.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SuggestedTestCase"></a>

### SuggestedTestCase

SuggestedTestCase suggests a test to improve coverage.




| Field | Type | Description |
| ----- | ---- | ----------- |
| target |string| Target to cover (state or transition label).   |
| target_type |string| Target type: "state", "transition", "guard_branch".   |
| event_sequence[] |string| Suggested event sequence.   |
| via_states[] |string| Expected states to reach target.   |
| confidence |double| Confidence score: 0.0 to 1.0.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ExpectedTrace"></a>

### ExpectedTrace

ExpectedTrace defines an expected execution sequence.

USE CASES:
- Trace-based testing: verify exact execution path
- Regression testing: ensure behavior doesn't change
- Documentation: show expected execution flow




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Trace identifier.   |
| name |string| Trace name.   |
| description |string| Trace description.   |
| steps[] |[TraceStep](#extensions-v1-TraceStep)| Ordered sequence of expected steps.   |
| strict_order |bool| Whether order is strict. true: steps must occur in exact order false: all steps must occur but order may vary   |
| complete |bool| Whether trace must be complete. true: no additional steps allowed false: additional steps are okay   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TraceStep"></a>

### TraceStep

TraceStep defines a single expected step in a trace.

STEP TYPES:
- "entry": State entry (target = state label)
- "exit": State exit (target = state label)
- "transition": Transition taken (target = transition label, event optional)
- "action": Action executed (target = action label)
- "event": Event processed (target = event label)




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |string| Step type.   |
| target |string| Target element label.   |
| event |string| Expected event (for transition steps).   |
| context |Struct| Expected context values at this step.   |
| max_time_since_start |Duration| Optional timing constraint.   |
| required |bool| Step must occur (vs optional).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestResult"></a>

### TestResult

TestResult captures the outcome of a test case execution.




| Field | Type | Description |
| ----- | ---- | ----------- |
| test_case_id |string| Test case ID.   |
| test_case_name |string| Test case name.   |
| outcome |string| Outcome: "passed", "failed", "skipped", "error".   |
| duration |Duration| Execution duration.   |
| executed_at |Timestamp| Execution timestamp.   |
| failure |[TestFailure](#extensions-v1-TestFailure)| Failure details (if failed).   |
| actual_trace[] |[TraceStep](#extensions-v1-TraceStep)| Execution trace (actual steps taken).   |
| final_configuration[] |string| Final configuration.   |
| final_context |Struct| Final context.   |
| states_covered[] |string| Coverage contributed by this test.   |
| transitions_covered[] |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TestFailure"></a>

### TestFailure

TestFailure captures failure details.




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |string| Failure type: "assertion", "invariant", "timeout", "error".   |
| failed_assertion |[TestAssertion](#extensions-v1-TestAssertion)| Failed assertion/invariant.   |
| message |string| Error message.   |
| stack_trace |string| Stack trace (if applicable).   |
| expected |string| Expected value.   |
| actual |string| Actual value.   |
| at_step |int32| Step index where failure occurred.   |
| at_event |string| Event that triggered failure.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="extensions-v1-AssertionType"></a>

### AssertionType
AssertionType classifies test assertions.



| Name | Number | Description |
| ---- | ------ | ----------- |
| ASSERTION_TYPE_UNSPECIFIED | 0 |   |
| ASSERTION_TYPE_STATE_ACTIVE | 1 | Assert specific state is active in current configuration. expected = state label   |
| ASSERTION_TYPE_STATE_NOT_ACTIVE | 2 | Assert specific state is NOT active. expected = state label   |
| ASSERTION_TYPE_CONFIGURATION_EQUALS | 3 | Assert exact configuration match. expected = comma-separated state labels or JSON array   |
| ASSERTION_TYPE_CONFIGURATION_CONTAINS | 4 | Assert configuration contains these states (subset). expected = comma-separated state labels or JSON array   |
| ASSERTION_TYPE_CONFIGURATION_SUBSET | 5 | Assert configuration is a subset of these states. expected = comma-separated state labels or JSON array   |
| ASSERTION_TYPE_CONTEXT_EQUALS | 6 | Assert context variable equals value. field = variable name, expected = JSON value   |
| ASSERTION_TYPE_CONTEXT_MATCHES | 7 | Assert context variable matches pattern. field = variable name, expected = regex pattern   |
| ASSERTION_TYPE_CONTEXT_EXISTS | 8 | Assert context variable exists. field = variable name   |
| ASSERTION_TYPE_ACTION_EXECUTED | 9 | Assert action was executed. expected = action label   |
| ASSERTION_TYPE_ACTION_NOT_EXECUTED | 10 | Assert action was NOT executed. expected = action label   |
| ASSERTION_TYPE_EVENT_EMITTED | 11 | Assert event was emitted/raised. expected = event label   |
| ASSERTION_TYPE_TRANSITION_TAKEN | 12 | Assert specific transition was taken. expected = transition label   |
| ASSERTION_TYPE_EXPRESSION | 13 | Assert CEL expression evaluates to true. expected = CEL expression   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

