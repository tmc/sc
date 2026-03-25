---
title: xstate.v1
description: API Specification for the xstate.v1 package.
---

<a name="xstate-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="xstate-v1-XStateLayout"></a>

### XStateLayout

XStateLayout captures visual positioning for editor display.
Stored in State.extensions or Transition.extensions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| position |[Position](#xstate-v1-Position)| x,y coordinates on the editor canvas.   |
| size |[Size](#xstate-v1-Size)| Width and height dimensions.   |
| color |string| Visual color hint: blue, green, orange, purple, red, yellow.   |
| unique_id |string| Internal editor-assigned unique identifier.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-Position"></a>

### Position

Position represents x,y coordinates in the editor canvas.




| Field | Type | Description |
| ----- | ---- | ----------- |
| x |double| Horizontal position.   |
| y |double| Vertical position.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-Size"></a>

### Size

Size represents width and height dimensions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| width |double| Width in pixels.   |
| height |double| Height in pixels.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateStateData"></a>

### XStateStateData

XStateStateData captures XState-specific state properties.
Stored in State.extensions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| description |string| Natural language documentation for the state.   |
| tags[] |string| State tags for matching and categorization.   |
| assets[] |[XStateAsset](#xstate-v1-XStateAsset)| Visual assets attached to the state (Figma links, images).   |
| invokes[] |[XStateInvoke](#xstate-v1-XStateInvoke)| Actor invocations that run while the state is active.   |
| meta_entries[] |[MetaEntry](#xstate-v1-MetaEntry)| Arbitrary key-value metadata pairs.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateTransitionData"></a>

### XStateTransitionData

XStateTransitionData captures XState-specific transition properties.
Stored in Transition.extensions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| internal |bool| Internal transition flag (skips entry/exit actions when true).   |
| trigger_type |[TransitionTriggerType](#xstate-v1-TransitionTriggerType)| Classification of how the transition is triggered.   |
| description |string| Documentation for the transition.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateInvoke"></a>

### XStateInvoke

XStateInvoke represents an actor invocation on a state.
In XState, invokes spawn actors that run while the state is active.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Unique identifier for this invocation.   |
| src |string| Reference to the actor source/implementation.   |
| kind |string| Kind of invocation: "named", "inline", etc.   |
| input |Struct| Input parameters passed to the actor.   |
| settings |Struct| Additional settings for the invocation.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateAsset"></a>

### XStateAsset

XStateAsset represents visual assets attached to states.
Supports Figma links and uploaded images.




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string| Display name for the asset.   |
| template |string| Asset template: "@figma.link", "@stately.image".   |
| properties |Struct| Template-specific properties (URL, fileId, nodeId, etc.).   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-MetaEntry"></a>

### MetaEntry

MetaEntry represents an arbitrary key-value pair.
Maps to XState metaEntries: [[key, value], ...]




| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string| Metadata key.   |
| value |string| Metadata value.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateMachineData"></a>

### XStateMachineData

XStateMachineData captures machine-level XState metadata.
Can be stored in Statechart.metadata as JSON or in a dedicated extension.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Machine UUID.   |
| project_version_id |string| Project version UUID.   |
| fork_parent_id |string| Fork parent UUID if this machine was forked.   |
| last_edited_by_id |string| UUID of the last user to edit this machine.   |
| original_code |string| Original XState JavaScript code.   |
| created_at |Timestamp| Timestamp when the machine was created.   |
| updated_at |Timestamp| Timestamp when the machine was last updated.   |
| schemas |[XStateSchemas](#xstate-v1-XStateSchemas)| Type schemas for machine elements.   |
| implementations |[XStateImplementations](#xstate-v1-XStateImplementations)| Action, guard, and actor implementations.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateSchemas"></a>

### XStateSchemas

XStateSchemas captures type schemas for machine elements.




| Field | Type | Description |
| ----- | ---- | ----------- |
| tags |Struct| Tag schemas.   |
| input |Struct| Input schema.   |
| output |Struct| Output schema.   |
| actors |Struct| Actor schemas.   |
| delays |Struct| Delay schemas.   |
| events |Struct| Event payload schemas.   |
| guards |Struct| Guard schemas.   |
| actions |Struct| Action schemas.   |
| context |Struct| Context variable schemas.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateImplementations"></a>

### XStateImplementations

XStateImplementations maps implementation details for actions, guards, actors.




| Field | Type | Description |
| ----- | ---- | ----------- |
| actions |[XStateImplementations.ActionsEntry](#xstate-v1-XStateImplementations-ActionsEntry)| Action implementations keyed by action name.   |
| guards |[XStateImplementations.GuardsEntry](#xstate-v1-XStateImplementations-GuardsEntry)| Guard implementations keyed by guard name.   |
| actors |[XStateImplementations.ActorsEntry](#xstate-v1-XStateImplementations-ActorsEntry)| Actor implementations keyed by actor name.   |






<a name="xstate-v1-XStateImplementations-ActionsEntry"></a>

### ActionsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[XStateActionImpl](#xstate-v1-XStateActionImpl)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateImplementations-GuardsEntry"></a>

### GuardsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[XStateGuardImpl](#xstate-v1-XStateGuardImpl)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateImplementations-ActorsEntry"></a>

### ActorsEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[XStateActorImpl](#xstate-v1-XStateActorImpl)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateActionImpl"></a>

### XStateActionImpl

XStateActionImpl describes an action implementation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Action identifier.   |
| name |string| Display name for the action.   |
| code |string| JavaScript implementation code.   |
| schema |Struct| Parameter schema.   |
| imports[] |string| Import statements required by this action.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateGuardImpl"></a>

### XStateGuardImpl

XStateGuardImpl describes a guard implementation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Guard identifier.   |
| name |string| Display name for the guard.   |
| params |Struct| Parameter definitions.   |
| imports[] |string| Import statements required by this guard.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="xstate-v1-XStateActorImpl"></a>

### XStateActorImpl

XStateActorImpl describes an actor implementation.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Actor identifier.   |
| name |string| Display name for the actor.   |
| kind |string| Actor kind.   |
| input |Struct| Input schema.   |
| output |Struct| Output schema.   |
| imports[] |string| Import statements required by this actor.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="xstate-v1-TransitionTriggerType"></a>

### TransitionTriggerType
TransitionTriggerType classifies how a transition is triggered.
Maps to XState eventTypeData.type values.



| Name | Number | Description |
| ---- | ------ | ----------- |
| TRANSITION_TRIGGER_TYPE_UNSPECIFIED | 0 | Unspecified trigger type.   |
| TRANSITION_TRIGGER_TYPE_EVENT | 1 | Regular named event trigger.   |
| TRANSITION_TRIGGER_TYPE_ALWAYS | 2 | Eventless/completion transition (always).   |
| TRANSITION_TRIGGER_TYPE_AFTER | 3 | Delayed transition (after).   |
| TRANSITION_TRIGGER_TYPE_INVOKE_DONE | 4 | Actor completed successfully (invocation.done).   |
| TRANSITION_TRIGGER_TYPE_INVOKE_ERROR | 5 | Actor errored (invocation.error).   |
| TRANSITION_TRIGGER_TYPE_STATE_DONE | 6 | Child state completed (state.done).   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

