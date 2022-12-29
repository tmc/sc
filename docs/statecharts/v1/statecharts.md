---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="statecharts-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="statecharts-v1-Statechart"></a>

### Statechart

Statechart definition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| states[] |[State](#statecharts-v1-State)| The top-level states in the statechart.   |
| transitions[] |[Transition](#statecharts-v1-Transition)| Transitions is the set of transitions that connect the states.   |
| events[] |[Event](#statecharts-v1-Event)| Events is the set of events that transitions are triggered by.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-State"></a>

### State

State is a state in a statechart.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string| The label of the state.   |
| type |[StateType](#statecharts-v1-StateType)| The type of the state.   |
| children[] |[State](#statecharts-v1-State)| The sub-states. If a state has no sub-states, it is considered a BASIC state.   |
| is_initial |bool| The state is the initial state of its parent state.   |
| is_final |bool| The state is the final state of its parent state.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Transition"></a>

### Transition

Transition is a transition between states in a statechart.
A transition connects source to target states.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string| The label of the transition.   |
| from[] |string| The from State reference.   |
| to[] |string| The to State   |
| event |string| The label of the event that triggers the transition.   |
| guard |[Guard](#statecharts-v1-Guard)| The guard of the transition.   |
| actions[] |[Action](#statecharts-v1-Action)| The action(s) of the transition.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Event"></a>

### Event

Event is an event in a statechart.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string| The label of the event.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Guard"></a>

### Guard

Guard is a guard for a transition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| expression |string| The guard expression.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Action"></a>

### Action

Action is an action for a transition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string| The label of the action.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StateRef"></a>

### StateRef

StateRef is a reference to a state.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string| The label of the state.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Configuration"></a>

### Configuration

Configuration is a status for a statechart which is defined by a subset of the states that are active.




| Field | Type | Description |
| ----- | ---- | ----------- |
| states[] |[StateRef](#statecharts-v1-StateRef)| The set of states in the status.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Machine"></a>

### Machine

Machine is an instance of a statechart.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| The id of the machine.   |
| state |[MachineState](#statecharts-v1-MachineState)| The overall state of the machine.   |
| context |Struct| The context of the machine.   |
| statechart |[Statechart](#statecharts-v1-Statechart)| The statechart definition.   |
| configuration |[Configuration](#statecharts-v1-Configuration)| The current configuration of the machine.   |
| step_history[] |[Step](#statecharts-v1-Step)| The history of steps that have been carried out by the machine.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Step"></a>

### Step

Step is a step in the execution of a statechart.




| Field | Type | Description |
| ----- | ---- | ----------- |
| events[] |[Event](#statecharts-v1-Event)| The events that occurred.   |
| transitions[] |[Transition](#statecharts-v1-Transition)| The transitions that occurred.   |
| starting_configuration |[Configuration](#statecharts-v1-Configuration)| The starting configuration.   |
| resulting_configuration |[Configuration](#statecharts-v1-Configuration)| The resulting configuration.   |
| context |Struct| The context of the event.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-v1-StateType"></a>

### StateType
StateType describes the type of a state.



| Name | Number | Description |
| ---- | ------ | ----------- |
| STATE_TYPE_UNSPECIFIED | 0 | Unspecified state type.   |
| STATE_TYPE_BASIC | 1 | A basic state (has no sub-states).   |
| STATE_TYPE_NORMAL | 2 | A normal state (has sub-states related by XOR semantics).   |
| STATE_TYPE_PARALLEL | 3 | A parallel state (has sub-states related by AND semantics). Also known in the literature as an "orthogonal state".   |




<a name="statecharts-v1-MachineState"></a>

### MachineState
MachineState encodes the high-level state of a statechart.



| Name | Number | Description |
| ---- | ------ | ----------- |
| MACHINE_STATE_UNSPECIFIED | 0 | The machine is in an unspecified state.   |
| MACHINE_STATE_RUNNING | 1 | The machine is in a running state.   |
| MACHINE_STATE_STOPPED | 2 | The machine is in a stopped state.   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

