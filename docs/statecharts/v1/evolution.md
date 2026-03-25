---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="evolution-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="statecharts-v1-ChartDiff"></a>

### ChartDiff

ChartDiff captures differences between two statechart versions.

FORMAL DEFINITION:
A diff D(SC₁, SC₂) = (V₁, V₂, ΔS, Δδ, ΔE, C, B) where:
  - V₁, V₂: source and target versions
  - ΔS: state changes (added, removed, modified)
  - Δδ: transition changes
  - ΔE: event changes
  - C: overall compatibility classification
  - B: list of breaking changes

DIFF COMPUTATION:
States are matched by label, transitions by (from, to, event) tuple.
Changes are classified by their impact on migration requirements.




| Field | Type | Description |
| ----- | ---- | ----------- |
| from_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)| Source version: V₁   |
| to_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)| Target version: V₂   |
| state_changes[] |[StateDiff](#statecharts-v1-StateDiff)| State changes: ΔS = S₂ \ S₁ ∪ S₁ \ S₂ ∪ modified(S₁ ∩ S₂)   |
| transition_changes[] |[TransitionDiff](#statecharts-v1-TransitionDiff)| Transition changes: Δδ   |
| event_changes[] |[EventDiff](#statecharts-v1-EventDiff)| Event changes: ΔE   |
| compatibility |[CompatibilityLevel](./execution.md#statecharts-v1-CompatibilityLevel)| Overall compatibility: C = max(severity(B))   |
| breaking_changes[] |[BreakingChange](#statecharts-v1-BreakingChange)| Breaking changes requiring attention   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StateDiff"></a>

### StateDiff

StateDiff describes a change to a single state.

CHANGE SEMANTICS:
  - ADDED: s ∈ S₂ ∧ s ∉ S₁
  - REMOVED: s ∈ S₁ ∧ s ∉ S₂
  - MODIFIED: s ∈ S₁ ∩ S₂ ∧ attrs(s, SC₁) ≠ attrs(s, SC₂)
  - RENAMED: ∃s₁ ∈ S₁, s₂ ∈ S₂: structure(s₁) ≈ structure(s₂) ∧ label(s₁) ≠ label(s₂)
  - MOVED: parent(s, SC₁) ≠ parent(s, SC₂)




| Field | Type | Description |
| ----- | ---- | ----------- |
| change_type |[ChangeType](#statecharts-v1-ChangeType)|   |
| state_label |string|   |
| old_state |[State](./statecharts.md#statecharts-v1-State)|   |
| new_state |[State](./statecharts.md#statecharts-v1-State)|   |
| modified_fields[] |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TransitionDiff"></a>

### TransitionDiff

TransitionDiff describes a change to a single transition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| change_type |[ChangeType](#statecharts-v1-ChangeType)|   |
| transition_label |string|   |
| old_transition |[Transition](./statecharts.md#statecharts-v1-Transition)|   |
| new_transition |[Transition](./statecharts.md#statecharts-v1-Transition)|   |
| modified_fields[] |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-EventDiff"></a>

### EventDiff

EventDiff describes a change to an event definition.




| Field | Type | Description |
| ----- | ---- | ----------- |
| change_type |[ChangeType](#statecharts-v1-ChangeType)|   |
| event_name |string|   |
| old_event |[Event](./statecharts.md#statecharts-v1-Event)|   |
| new_event |[Event](./statecharts.md#statecharts-v1-Event)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-BreakingChange"></a>

### BreakingChange

BreakingChange identifies a change requiring migration attention.

BREAKING CHANGE CRITERIA:
A change is breaking if it can cause:
  1. Invalid configuration: σ valid in SC₁ but invalid in SC₂
  2. Behavioral change: same inputs produce different outputs
  3. Lost functionality: capabilities in SC₁ unavailable in SC₂




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |[BreakingChangeType](#statecharts-v1-BreakingChangeType)|   |
| description |string|   |
| affected_element |string|   |
| migration_required |[MigrationRequirement](#statecharts-v1-MigrationRequirement)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-DeprecationInfo"></a>

### DeprecationInfo

DeprecationInfo marks elements scheduled for removal.




| Field | Type | Description |
| ----- | ---- | ----------- |
| deprecated_in_version |string|   |
| removal_version |string|   |
| migration_guide |string|   |
| replacement |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MigrationPlan"></a>

### MigrationPlan

MigrationPlan specifies how to migrate machines between chart versions.

FORMAL DEFINITION:
A migration plan M = (V₁, V₂, strategy, mappings, transform, validations) where:
  - V₁, V₂: source and target versions
  - strategy: migration approach (stop-the-world, rolling, etc.)
  - mappings: state mapping rules
  - transform: context transformation
  - validations: post-migration checks

MIGRATION INVARIANT:
For valid migration: ∀ machine m with config σ in V₁:
  migrate(σ, M) produces valid config σ' in V₂




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string| Plan identifier   |
| from_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)| Source version: V₁   |
| to_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)| Target version: V₂   |
| strategy |[MigrationStrategy](#statecharts-v1-MigrationStrategy)| Migration strategy   |
| state_mappings[] |[StateMapping](#statecharts-v1-StateMapping)| State mappings: how to transform configurations   |
| transition_mappings[] |[TransitionMapping](#statecharts-v1-TransitionMapping)| Transition mappings: for renamed transitions   |
| context_transform |[ContextTransformation](#statecharts-v1-ContextTransformation)| Context transformation: how to transform variables   |
| validations[] |[MigrationValidation](#statecharts-v1-MigrationValidation)| Post-migration validations   |
| impact |[MigrationImpact](#statecharts-v1-MigrationImpact)| Impact assessment   |
| dry_run |bool| Dry-run mode: validate without applying   |
| in_flight_policy |[InFlightTransitionPolicy](#statecharts-v1-InFlightTransitionPolicy)| In-flight transition handling   |
| rollback_plan |[RollbackPlan](#statecharts-v1-RollbackPlan)| Rollback configuration   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StateMapping"></a>

### StateMapping

StateMapping defines how to map configurations between versions.

FORMAL DEFINITION:
A state mapping m = (type, from, to, condition, transform) where:
  - type: mapping strategy
  - from ⊆ S₁: source states
  - to ⊆ S₂: target states
  - condition: optional guard for conditional mapping
  - transform: context transformation during mapping

MAPPING SEMANTICS BY TYPE:
  - IDENTITY: s ∈ σ₁ ⟹ s ∈ σ₂ (label unchanged)
  - RENAME: s₁ ∈ σ₁ ⟹ s₂ ∈ σ₂ (label changed)
  - TO_PARENT: s ∈ σ₁ ∧ s removed ⟹ default(parent(s)) ∈ σ₂
  - SPLIT: s ∈ σ₁ ⟹ f(context) ∈ σ₂ (one-to-many based on condition)
  - MERGE: {s₁, s₂, ...} ∩ σ₁ ≠ ∅ ⟹ t ∈ σ₂ (many-to-one)




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |[StateMappingType](#statecharts-v1-StateMappingType)|   |
| from_states[] |string|   |
| to_states[] |string|   |
| condition |[Expression](./expressions.md#statecharts-v1-Expression)|   |
| transform |[ContextTransformation](#statecharts-v1-ContextTransformation)|   |
| priority |int32|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-TransitionMapping"></a>

### TransitionMapping

TransitionMapping maps transitions between versions.




| Field | Type | Description |
| ----- | ---- | ----------- |
| from_transition |string|   |
| to_transition |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ContextTransformation"></a>

### ContextTransformation

ContextTransformation specifies variable changes during migration.

TRANSFORM SEMANTICS:
Given context Γ, produce Γ' = transform(Γ) by:
  1. Apply set: Γ'[k] = eval(expr, Γ) for each (k, expr) in set
  2. Apply remove: delete Γ'[k] for each k in remove
  3. Apply rename: Γ'[new] = Γ[old], delete Γ'[old]




| Field | Type | Description |
| ----- | ---- | ----------- |
| set |[ContextTransformation.SetEntry](#statecharts-v1-ContextTransformation-SetEntry)| Set variable values: key → CEL expression   |
| remove[] |string| Remove variables   |
| rename |[ContextTransformation.RenameEntry](#statecharts-v1-ContextTransformation-RenameEntry)| Rename variables: old_key → new_key   |






<a name="statecharts-v1-ContextTransformation-SetEntry"></a>

### SetEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ContextTransformation-RenameEntry"></a>

### RenameEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MigrationValidation"></a>

### MigrationValidation

MigrationValidation defines post-migration checks.




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string|   |
| expression |string|   |
| error_message |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MigrationImpact"></a>

### MigrationImpact

MigrationImpact assesses migration risk and scope.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machines_affected |uint64|   |
| machines_requiring_manual |uint64|   |
| estimated_downtime |Duration|   |
| risk |[RiskLevel](#statecharts-v1-RiskLevel)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-DryRunResult"></a>

### DryRunResult

DryRunResult contains the outcome of a migration dry-run.




| Field | Type | Description |
| ----- | ---- | ----------- |
| previews[] |[MachineMigrationPreview](#statecharts-v1-MachineMigrationPreview)|   |
| warnings[] |string|   |
| errors[] |string|   |
| would_succeed |bool|   |
| estimated_duration |Duration|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MachineMigrationPreview"></a>

### MachineMigrationPreview

MachineMigrationPreview shows expected migration outcome for one machine.




| Field | Type | Description |
| ----- | ---- | ----------- |
| machine_id |string|   |
| current_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)|   |
| target_config |[Configuration](./statecharts.md#statecharts-v1-Configuration)|   |
| mapping_used |[StateMappingType](#statecharts-v1-StateMappingType)|   |
| warnings[] |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MigrationHistory"></a>

### MigrationHistory

MigrationHistory provides an audit trail of all migrations.

AUDIT PROPERTIES:
  - Append-only: records never modified or deleted
  - Complete: every migration attempt is recorded
  - Traceable: each record links to initiator and result




| Field | Type | Description |
| ----- | ---- | ----------- |
| records[] |[MigrationRecord](#statecharts-v1-MigrationRecord)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-MigrationRecord"></a>

### MigrationRecord

MigrationRecord captures details of a single migration execution.




| Field | Type | Description |
| ----- | ---- | ----------- |
| migration_id |string|   |
| from_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)|   |
| to_version |[ChartVersion](./execution.md#statecharts-v1-ChartVersion)|   |
| started_at |Timestamp|   |
| completed_at |Timestamp|   |
| result |[MigrationResult](#statecharts-v1-MigrationResult)|   |
| machines_migrated |uint64|   |
| machines_failed |uint64|   |
| initiated_by |string|   |
| rollback_of |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-InFlightTransitionPolicy"></a>

### InFlightTransitionPolicy

InFlightTransitionPolicy handles machines with active transitions.

FORMAL DEFINITION:
For machine m with in-flight transition t:
  - COMPLETE: wait for t to finish, then migrate
  - ABORT: cancel t, migrate immediately
  - RETRY: complete t, migrate, re-evaluate pending events
  - ERROR: fail migration for this machine




| Field | Type | Description |
| ----- | ---- | ----------- |
| action |[InFlightAction](#statecharts-v1-InFlightAction)|   |
| grace_period |Duration|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-RollbackPlan"></a>

### RollbackPlan

RollbackPlan defines conditions and procedures for migration rollback.

ROLLBACK SEMANTICS:
Rollback reverts machines to pre-migration state using:
  1. Checkpoint restoration (if available)
  2. Reverse state mappings (inverse of forward mappings)
  3. Log replay from checkpoint




| Field | Type | Description |
| ----- | ---- | ----------- |
| triggers[] |[RollbackTrigger](#statecharts-v1-RollbackTrigger)|   |
| strategy |[RollbackStrategy](#statecharts-v1-RollbackStrategy)|   |
| rollback_mappings[] |[StateMapping](#statecharts-v1-StateMapping)|   |
| rollback_window |Duration|   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-RollbackTrigger"></a>

### RollbackTrigger

RollbackTrigger conditions that initiate automatic rollback.




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |[RollbackTriggerType](#statecharts-v1-RollbackTriggerType)|   |
| threshold |string|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-v1-ChangeType"></a>

### ChangeType
ChangeType classifies the nature of a schema change.

FORMAL CLASSIFICATION:
  - ADDED: element exists in new but not old
  - REMOVED: element exists in old but not new
  - MODIFIED: element exists in both with different attributes
  - RENAMED: identity preserved, label changed
  - MOVED: position in hierarchy changed



| Name | Number | Description |
| ---- | ------ | ----------- |
| CHANGE_TYPE_UNSPECIFIED | 0 |   |
| CHANGE_TYPE_ADDED | 1 |   |
| CHANGE_TYPE_REMOVED | 2 |   |
| CHANGE_TYPE_MODIFIED | 3 |   |
| CHANGE_TYPE_RENAMED | 4 |   |
| CHANGE_TYPE_MOVED | 5 |   |




<a name="statecharts-v1-BreakingChangeType"></a>

### BreakingChangeType
BreakingChangeType enumerates specific breaking change patterns.



| Name | Number | Description |
| ---- | ------ | ----------- |
| BREAKING_CHANGE_TYPE_UNSPECIFIED | 0 |   |
| BREAKING_CHANGE_TYPE_STATE_REMOVED | 1 | State changes   |
| BREAKING_CHANGE_TYPE_STATE_TYPE_CHANGED | 2 |   |
| BREAKING_CHANGE_TYPE_INITIAL_STATE_CHANGED | 3 |   |
| BREAKING_CHANGE_TYPE_TRANSITION_REMOVED | 4 | Transition changes   |
| BREAKING_CHANGE_TYPE_TRANSITION_SOURCE_CHANGED | 5 |   |
| BREAKING_CHANGE_TYPE_GUARD_ADDED | 6 |   |
| BREAKING_CHANGE_TYPE_EVENT_REMOVED | 7 | Event changes   |
| BREAKING_CHANGE_TYPE_EVENT_PAYLOAD_CHANGED | 8 |   |




<a name="statecharts-v1-MigrationRequirement"></a>

### MigrationRequirement
MigrationRequirement indicates complexity of migration.



| Name | Number | Description |
| ---- | ------ | ----------- |
| MIGRATION_REQUIREMENT_UNSPECIFIED | 0 |   |
| MIGRATION_REQUIREMENT_NONE | 1 |   |
| MIGRATION_REQUIREMENT_STATE_MAPPING | 2 |   |
| MIGRATION_REQUIREMENT_MANUAL | 3 |   |




<a name="statecharts-v1-MigrationStrategy"></a>

### MigrationStrategy
MigrationStrategy determines migration execution approach.

STRATEGY SEMANTICS:
  - STOP_THE_WORLD: halt all machines, migrate, resume
  - ROLLING: migrate machines one at a time
  - BLUE_GREEN: run both versions, switch traffic
  - CANARY: migrate subset, validate, proceed



| Name | Number | Description |
| ---- | ------ | ----------- |
| MIGRATION_STRATEGY_UNSPECIFIED | 0 |   |
| MIGRATION_STRATEGY_STOP_THE_WORLD | 1 |   |
| MIGRATION_STRATEGY_ROLLING | 2 |   |
| MIGRATION_STRATEGY_BLUE_GREEN | 3 |   |
| MIGRATION_STRATEGY_CANARY | 4 |   |




<a name="statecharts-v1-StateMappingType"></a>

### StateMappingType
StateMappingType enumerates state mapping strategies.



| Name | Number | Description |
| ---- | ------ | ----------- |
| STATE_MAPPING_TYPE_UNSPECIFIED | 0 |   |
| STATE_MAPPING_TYPE_IDENTITY | 1 |   |
| STATE_MAPPING_TYPE_RENAME | 2 |   |
| STATE_MAPPING_TYPE_TO_PARENT | 3 |   |
| STATE_MAPPING_TYPE_TO_SIBLING | 4 |   |
| STATE_MAPPING_TYPE_TO_INITIAL | 5 |   |
| STATE_MAPPING_TYPE_SPLIT | 6 |   |
| STATE_MAPPING_TYPE_MERGE | 7 |   |
| STATE_MAPPING_TYPE_CUSTOM | 8 |   |
| STATE_MAPPING_TYPE_ERROR | 9 |   |




<a name="statecharts-v1-RiskLevel"></a>

### RiskLevel
RiskLevel classifies migration risk.



| Name | Number | Description |
| ---- | ------ | ----------- |
| RISK_LEVEL_UNSPECIFIED | 0 |   |
| RISK_LEVEL_LOW | 1 |   |
| RISK_LEVEL_MEDIUM | 2 |   |
| RISK_LEVEL_HIGH | 3 |   |




<a name="statecharts-v1-MigrationResult"></a>

### MigrationResult
MigrationResult indicates migration outcome.



| Name | Number | Description |
| ---- | ------ | ----------- |
| MIGRATION_RESULT_UNSPECIFIED | 0 |   |
| MIGRATION_RESULT_SUCCESS | 1 |   |
| MIGRATION_RESULT_PARTIAL | 2 |   |
| MIGRATION_RESULT_FAILED | 3 |   |
| MIGRATION_RESULT_ROLLED_BACK | 4 |   |
| MIGRATION_RESULT_CANCELLED | 5 |   |




<a name="statecharts-v1-InFlightAction"></a>

### InFlightAction
InFlightAction specifies how to handle in-flight transitions.



| Name | Number | Description |
| ---- | ------ | ----------- |
| IN_FLIGHT_ACTION_UNSPECIFIED | 0 |   |
| IN_FLIGHT_ACTION_COMPLETE | 1 |   |
| IN_FLIGHT_ACTION_ABORT | 2 |   |
| IN_FLIGHT_ACTION_RETRY | 3 |   |
| IN_FLIGHT_ACTION_ERROR | 4 |   |




<a name="statecharts-v1-RollbackTriggerType"></a>

### RollbackTriggerType
RollbackTriggerType enumerates rollback trigger conditions.



| Name | Number | Description |
| ---- | ------ | ----------- |
| ROLLBACK_TRIGGER_TYPE_UNSPECIFIED | 0 |   |
| ROLLBACK_TRIGGER_TYPE_ERROR_RATE | 1 |   |
| ROLLBACK_TRIGGER_TYPE_LATENCY | 2 |   |
| ROLLBACK_TRIGGER_TYPE_MANUAL | 3 |   |
| ROLLBACK_TRIGGER_TYPE_HEALTH_CHECK | 4 |   |




<a name="statecharts-v1-RollbackStrategy"></a>

### RollbackStrategy
RollbackStrategy determines rollback execution method.



| Name | Number | Description |
| ---- | ------ | ----------- |
| ROLLBACK_STRATEGY_UNSPECIFIED | 0 |   |
| ROLLBACK_STRATEGY_CHECKPOINT | 1 |   |
| ROLLBACK_STRATEGY_REVERSE | 2 |   |
| ROLLBACK_STRATEGY_RECREATE | 3 |   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

