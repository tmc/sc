---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="statechart_service-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->


<a name="statecharts-v1-StatechartService"></a>

### StatechartService

Statechart service.



| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| CreateMachine | [CreateMachineRequest](#statecharts-v1-CreateMachineRequest) | [CreateMachineResponse](#statecharts-v1-CreateMachineResponse) | Create a new machine.   |
| Step | [StepRequest](#statecharts-v1-StepRequest) | [StepResponse](#statecharts-v1-StepResponse) | Step a statechart through a single iteration.   |



<!-- begin services -->



<a name="statecharts-v1-StateChartRegistry"></a>

### StateChartRegistry

A registry of statecharts.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statecharts |[StateChartRegistry.StatechartsEntry](#statecharts-v1-StateChartRegistry-StatechartsEntry)| The registry of statecharts.   |






<a name="statecharts-v1-StateChartRegistry-StatechartsEntry"></a>

### StatechartsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[StateChart](./statecharts.md#statecharts-v1-StateChart)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CreateMachineRequest"></a>

### CreateMachineRequest

A request to create a new machine.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statechart_id |string| The ID of the statechart to create an instance from.   |
| context |Struct| The initial context of the machine.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CreateMachineResponse"></a>

### CreateMachineResponse

A response to a create machine request.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machine |[Machine](./statecharts.md#statecharts-v1-Machine)| The created machine.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StepRequest"></a>

### StepRequest

StepRequest is the request message for the Step method.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statechart_id |string| The id of the statechart to step.   |
| event |string| The event to step the statechart with.  The context attached to the Event.  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StepResponse"></a>

### StepResponse

StepResponse is the response message for the Step method.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machine |[Machine](./statecharts.md#statecharts-v1-Machine)| The statechart's current state.   |
| result |Status| The result of the step operation.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

