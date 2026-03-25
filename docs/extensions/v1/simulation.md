---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="simulation-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-StateSimConfig"></a>

### StateSimConfig

StateSimConfig provides simulation settings for a state.

STORAGE:
Pack into State.extensions using google.protobuf.Any.

EXECUTION SEMANTICS:
- Timing constraints are enforced during simulation
- Resource consumption is tracked for capacity analysis
- Mocks replace real implementations for testing
- Failures are injected according to configuration




| Field | Type | Description |
| ----- | ---- | ----------- |
| timing |[TimingConstraints](#extensions-v1-TimingConstraints)| Timing constraints for this state.   |
| resources |[ResourceProfile](#extensions-v1-ResourceProfile)| Resource consumption profile while in state.   |
| entry_mock |[MockConfig](#extensions-v1-MockConfig)| Mock configuration for entry actions.   |
| exit_mock |[MockConfig](#extensions-v1-MockConfig)| Mock configuration for exit actions.   |
| failure |[FailureConfig](#extensions-v1-FailureConfig)| Failure injection settings.   |
| trace_level |int32| Logging/tracing verbosity. 0 = none, 1 = entry/exit, 2 = +actions, 3 = +context changes   |
| cost |double| Cost/weight for optimization algorithms.   |
| reward |double| Reward for reinforcement learning scenarios.   |
| priority |int32| Priority for resource allocation.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TimingConstraints"></a>

### TimingConstraints

TimingConstraints defines temporal requirements for a state.

TIMED SEMANTICS [AD94]:
A state s with timing constraints defines clock invariants:
  - min_dwell: x ≥ min_dwell before exit enabled
  - max_dwell: x ≤ max_dwell or timeout event generated
  - deadline: absolute deadline from system start

where x is a clock reset upon state entry.




| Field | Type | Description |
| ----- | ---- | ----------- |
| min_dwell |Duration| Minimum time before exiting state (dwell time). Exit transitions are disabled until min_dwell elapsed.   |
| max_dwell |Duration| Maximum time before forced exit (timeout). If reached, timeout_event is generated.   |
| expected_dwell |Duration| Expected/typical dwell time (for statistics/analysis).   |
| dwell_distribution |[Distribution](#extensions-v1-Distribution)| Dwell time distribution for stochastic simulation.   |
| deadline |Duration| Absolute deadline from simulation start.   |
| entry_delay |[DelayRange](#extensions-v1-DelayRange)| Entry delay (simulated processing time before becoming active).   |
| exit_delay |[DelayRange](#extensions-v1-DelayRange)| Exit delay (cleanup time after exit decision).   |
| timeout_event |string| Event generated on max_dwell expiry. If empty, state is force-exited to parent's default.   |
| deadline_event |string| Event generated on deadline violation.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-TransitionSimConfig"></a>

### TransitionSimConfig

TransitionSimConfig provides simulation settings for a transition.

STORAGE:
Pack into Transition.extensions using google.protobuf.Any.

STOCHASTIC SEMANTICS [SPN]:
For non-deterministic choice between enabled transitions:
  P(t_i selected) = weight_i / Σ weight_j
where the sum is over all enabled transitions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| weight |double| Probability weight for non-deterministic choice. Higher weight = more likely to be selected. Default 1.0 for uniform distribution.   |
| delay |[DelayRange](#extensions-v1-DelayRange)| Execution delay range.   |
| mock |[MockConfig](#extensions-v1-MockConfig)| Mock configuration for transition actions.   |
| failure |[FailureConfig](#extensions-v1-FailureConfig)| Failure injection for this transition.   |
| cost |double| Cost/weight for shortest path algorithms.   |
| reward |double| Reward for reinforcement learning.   |
| rate |double| Rate parameter for continuous-time Markov chains. Firing rate λ: inter-arrival time ~ Exp(λ).   |
| enable_delay |Duration| Enabling condition delay (time after enabled before fireable).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-DelayRange"></a>

### DelayRange

DelayRange specifies a delay with optional variance.

DISTRIBUTIONS:
The delay is sampled from the specified distribution:
  - "constant": always returns (min + max) / 2
  - "uniform": uniform distribution over [min, max]
  - "normal": normal distribution, params = {mean, stddev}
  - "exponential": exponential distribution, params = {rate}
  - "lognormal": log-normal distribution, params = {mu, sigma}
  - "weibull": Weibull distribution, params = {scale, shape}




| Field | Type | Description |
| ----- | ---- | ----------- |
| min |Duration| Minimum delay.   |
| max |Duration| Maximum delay.   |
| distribution |string| Distribution type.   |
| params |[DelayRange.ParamsEntry](#extensions-v1-DelayRange-ParamsEntry)| Distribution-specific parameters.   |






<a name="extensions-v1-DelayRange-ParamsEntry"></a>

### ParamsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Distribution"></a>

### Distribution

Distribution specifies a probability distribution.




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |string| Distribution type: "constant", "uniform", "normal", "exponential", etc.   |
| params |[Distribution.ParamsEntry](#extensions-v1-Distribution-ParamsEntry)| Distribution parameters.   |
| min |double| Minimum value (truncation).   |
| max |double| Maximum value (truncation).   |






<a name="extensions-v1-Distribution-ParamsEntry"></a>

### ParamsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ResourceProfile"></a>

### ResourceProfile

ResourceProfile defines resource consumption while in a state.

USE CASES:
- Capacity planning
- Performance modeling
- Cost estimation
- Energy-aware scheduling




| Field | Type | Description |
| ----- | ---- | ----------- |
| cpu_utilization |double| CPU utilization: 0.0 to 1.0 (fraction of one core).   |
| memory_bytes |int64| Memory usage in bytes.   |
| network_bps |int64| Network bandwidth consumption in bytes/second.   |
| disk_iops |int64| Disk I/O in bytes/second.   |
| power_watts |double| Power consumption in watts.   |
| cost_per_hour |double| Monetary cost per time unit (e.g., $/hour).   |
| custom |[ResourceProfile.CustomEntry](#extensions-v1-ResourceProfile-CustomEntry)| Custom resource metrics.   |
| constant |bool| Resource consumption is constant vs varies over time.   |
| samples[] |[ResourceSample](#extensions-v1-ResourceSample)| Time-varying resource profile.   |






<a name="extensions-v1-ResourceProfile-CustomEntry"></a>

### CustomEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ResourceSample"></a>

### ResourceSample

ResourceSample defines resource consumption at a point in time.




| Field | Type | Description |
| ----- | ---- | ----------- |
| offset |Duration| Time offset from state entry.   |
| values |[ResourceSample.ValuesEntry](#extensions-v1-ResourceSample-ValuesEntry)| Resource values at this time.   |






<a name="extensions-v1-ResourceSample-ValuesEntry"></a>

### ValuesEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-MockConfig"></a>

### MockConfig

MockConfig defines mock/stub behavior for simulation.

MODES:
- "passthrough": Use real implementation (no mocking)
- "success": Always return success with default_value
- "failure": Always return error with error_code
- "random": Random success/failure based on failure_rate
- "sequence": Return responses in order
- "replay": Replay recorded responses




| Field | Type | Description |
| ----- | ---- | ----------- |
| enabled |bool| Whether mocking is enabled.   |
| mode |string| Mock behavior mode.   |
| response_delay |[DelayRange](#extensions-v1-DelayRange)| Response delay.   |
| responses[] |[MockResponse](#extensions-v1-MockResponse)| Ordered mock responses (for sequence mode).   |
| failure_rate |double| Random failure probability (for random mode).   |
| default_value |string| Default return value (JSON).   |
| default_error_code |string| Default error code.   |
| default_error_message |string| Default error message.   |
| record |bool| Whether to record actual responses for later replay.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-MockResponse"></a>

### MockResponse

MockResponse defines a single mock response.




| Field | Type | Description |
| ----- | ---- | ----------- |
| value |string| Response value (JSON).   |
| is_error |bool| Whether this response is an error.   |
| error_code |string| Error code.   |
| error_message |string| Error message.   |
| delay |Duration| Response delay.   |
| condition |string| Condition for this response (CEL expression).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-FailureConfig"></a>

### FailureConfig

FailureConfig defines failure injection for testing resilience.

FAILURE TYPES:
- "crash": Immediate termination, no cleanup
- "timeout": Operation hangs indefinitely
- "exception": Throws specified error
- "corrupt": Returns corrupted/invalid data
- "slow": Adds extreme delay
- "partial": Partial completion

TRIGGERS:
- "random": Based on failure_rate probability
- "after_n": After N successful executions
- "condition": When CEL condition is true
- "scheduled": At specific simulation time




| Field | Type | Description |
| ----- | ---- | ----------- |
| enabled |bool| Failure injection enabled.   |
| failure_type |string| Failure type.   |
| failure_rate |double| Failure probability per execution (for random trigger).   |
| trigger |string| Trigger type.   |
| trigger_after_n |int64| N for "after_n" trigger.   |
| trigger_condition |string| CEL condition for "condition" trigger.   |
| trigger_at |Duration| Scheduled failure time for "scheduled" trigger.   |
| recovery |string| Recovery behavior: "none", "retry", "skip", "escalate", "fallback".   |
| max_retries |int32| Maximum retry attempts.   |
| retry_delay |[DelayRange](#extensions-v1-DelayRange)| Delay between retries.   |
| fallback_value |string| Fallback value (JSON) if recovery = "fallback".   |
| error_code |string| Error details for exception type.   |
| error_message |string|   |
| corruption |[CorruptionConfig](#extensions-v1-CorruptionConfig)| Corruption parameters for corrupt type.   |
| only_in_states[] |string| Only inject in these states (empty = all).   |
| exclude_states[] |string| Never inject in these states.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-CorruptionConfig"></a>

### CorruptionConfig

CorruptionConfig defines data corruption parameters.




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |string| Corruption type: "bit_flip", "truncate", "null", "garbage", "swap".   |
| probability |double| Corruption probability for each field/byte.   |
| fields[] |string| Fields to corrupt (empty = random).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SimulationScenario"></a>

### SimulationScenario

SimulationScenario defines a complete simulation configuration.

USE CASES:
- Performance benchmarking
- Load testing
- Chaos engineering
- What-if analysis




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Scenario identifier.   |
| name |string| Scenario name.   |
| description |string| Description.   |
| time_limit |Duration| Simulation time limit (virtual time).   |
| wall_time_limit |Duration| Wall-clock time limit.   |
| max_events |int64| Maximum events to process.   |
| scheduled_events[] |[ScheduledEvent](#extensions-v1-ScheduledEvent)| Scheduled event injections.   |
| random_events |[RandomEventConfig](#extensions-v1-RandomEventConfig)| Random event generation.   |
| constraints |[ResourceConstraints](#extensions-v1-ResourceConstraints)| Global resource constraints.   |
| random_seed |int64| Random seed for reproducibility. 0 = use current time.   |
| run_count |int32| Number of simulation runs.   |
| metrics[] |string| Metrics to collect.   |
| warmup |Duration| Warmup period (metrics not collected).   |
| global_failure |[FailureConfig](#extensions-v1-FailureConfig)| Global failure injection.   |
| time_scale |double| Time speedup factor: 1.0 = real-time, 10.0 = 10x faster.   |
| initial_context |Struct| Initial context for all runs.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ScheduledEvent"></a>

### ScheduledEvent

ScheduledEvent defines an event at a specific simulation time.




| Field | Type | Description |
| ----- | ---- | ----------- |
| time |Duration| Time offset from simulation start.   |
| event |string| Event name.   |
| payload |string| Event payload (JSON).   |
| repeat |bool| Whether to repeat.   |
| interval |Duration| Repeat interval.   |
| max_repetitions |int32| Maximum repetitions (0 = unlimited).   |
| condition |string| Condition for injection (CEL).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-RandomEventConfig"></a>

### RandomEventConfig

RandomEventConfig defines random event generation.

ARRIVAL PROCESS:
Events arrive according to a Poisson process with rate λ.
Event type is selected based on weights.




| Field | Type | Description |
| ----- | ---- | ----------- |
| events[] |[RandomEvent](#extensions-v1-RandomEvent)| Event specifications.   |
| rate |double| Global event arrival rate (events per second).   |
| arrival_process |string| Arrival process: "poisson", "uniform", "burst".   |
| burst |[BurstConfig](#extensions-v1-BurstConfig)| Burst configuration (for burst arrival).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-RandomEvent"></a>

### RandomEvent

RandomEvent defines a randomly generated event type.




| Field | Type | Description |
| ----- | ---- | ----------- |
| event |string| Event name.   |
| weight |double| Selection weight (relative probability).   |
| payload_generator |string| Payload generator: "constant", "random_int", "random_string", "random_json".   |
| generator_params |Struct| Generator parameters.   |
| condition |string| Condition for allowing this event (CEL).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-BurstConfig"></a>

### BurstConfig

BurstConfig defines burst event generation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| burst_size |int32| Events per burst.   |
| burst_interval |Duration| Time between bursts.   |
| intra_burst_delay |Duration| Delay between events within burst.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ResourceConstraints"></a>

### ResourceConstraints

ResourceConstraints defines global simulation limits.




| Field | Type | Description |
| ----- | ---- | ----------- |
| max_concurrent_states |int32| Maximum concurrent active states.   |
| max_event_rate |double| Maximum events per second.   |
| max_memory_bytes |int64| Maximum memory usage.   |
| max_cpu |double| Maximum CPU utilization.   |
| max_queue_depth |int32| Maximum queue depth.   |
| overflow_action |string| Action on constraint violation: "reject", "queue", "drop", "error".   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SimulationResult"></a>

### SimulationResult

SimulationResult captures the outcome of a simulation run.




| Field | Type | Description |
| ----- | ---- | ----------- |
| scenario_id |string| Scenario identifier.   |
| run_number |int32| Run number (for multi-run scenarios).   |
| random_seed |int64| Random seed used.   |
| duration |Duration| Simulation duration (virtual time).   |
| wall_duration |Duration| Wall-clock duration.   |
| events_processed |int64| Total events processed.   |
| final_configuration[] |string| Final configuration.   |
| final_context |Struct| Final context.   |
| metrics |[SimulationResult.MetricsEntry](#extensions-v1-SimulationResult-MetricsEntry)| Collected metrics.   |
| resource_utilization |[SimulationResult.ResourceUtilizationEntry](#extensions-v1-SimulationResult-ResourceUtilizationEntry)| Resource utilization summary.   |
| failures_injected |int64| Failures injected.   |
| failures_recovered |int64| Failures recovered.   |
| violations[] |string| Constraint violations.   |
| termination_reason |string| Termination reason: "time_limit", "event_limit", "final_state", "error".   |
| error |string| Error details (if terminated with error).   |






<a name="extensions-v1-SimulationResult-MetricsEntry"></a>

### MetricsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-SimulationResult-ResourceUtilizationEntry"></a>

### ResourceUtilizationEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |double|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

