---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="statechart_service-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->


<a name="statecharts-v1-StatechartService"></a>

### StatechartService

StatechartService defines the main service for interacting with statecharts.
It allows creating a new machine and stepping a statechart through a single iteration.



| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| CreateMachine | [CreateMachineRequest](#statecharts-v1-CreateMachineRequest) | [CreateMachineResponse](#statecharts-v1-CreateMachineResponse) | Create a new machine.   |
| Step | [StepRequest](#statecharts-v1-StepRequest) | [StepResponse](#statecharts-v1-StepResponse) | Step a statechart through a single iteration.   |



<!-- begin services -->



<a name="statecharts-v1-StatechartRegistry"></a>

### StatechartRegistry

StatechartRegistry maintains a collection of Statecharts.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statecharts |[StatechartRegistry.StatechartsEntry](#statecharts-v1-StatechartRegistry-StatechartsEntry)| The registry of Statecharts.   |






<a name="statecharts-v1-StatechartRegistry-StatechartsEntry"></a>

### StatechartsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[Statechart](./statecharts.md#statecharts-v1-Statechart)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CreateMachineRequest"></a>

### CreateMachineRequest

CreateMachineRequest is the request message for creating a new machine.
It requires a statechart ID.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statechart_id |string| The ID of the statechart to create an instance from.   |
| context |Struct| The initial context of the machine.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-CreateMachineResponse"></a>

### CreateMachineResponse

CreateMachineResponse is the response message for creating a new machine.
It returns the created machine.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machine |[Machine](./statecharts.md#statecharts-v1-Machine)| The created machine.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StepRequest"></a>

### StepRequest

StepRequest is the request message for the Step method.
It is defined a statechart ID, an event, and an optional context.




| Field | Type | Description |
| ----- | ---- | ----------- |
| statechart_id |string| The id of the statechart to step.   |
| event |string| The event to step the statechart with.   |
| context |Struct| The context attached to the Event.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StepResponse"></a>

### StepResponse

StepResponse is the response message for the Step method.
It returns the current state of the statechart and the result of the step operation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machine |[Machine](./statecharts.md#statecharts-v1-Machine)| The statechart's current state (machine).   |
| result |Status| The result of the step operation.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

