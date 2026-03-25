---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="statecharts-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="statecharts-v1-Statechart"></a>

### Statechart

Complete, static description of a statechart.

FORMAL DEFINITION [H87, Section 2]:
A statechart SC is a 7-tuple SC = (S, ρ, ψ, δ, γ, λ, σ₀) where this message
represents the concrete encoding of such a tuple.

WELL-FORMEDNESS CONSTRAINTS [H87, HN96]:
1. The hierarchy relation ρ must form a tree rooted at root_state
2. Each OR-state must have exactly one default child (λ function)
3. All state labels must be unique within the statechart
4. Transitions must reference valid states from S
5. Events in transitions must be from the event alphabet E

SEMANTIC INVARIANTS:
- ∀s ∈ S: s ≠ root ⟹ ∃!p ∈ S: (p,s) ∈ ρ (unique parent except root)
- ∀s ∈ S: ψ(s) = NORMAL ∧ children(s) ≠ ∅ ⟹ |λ(s)| = 1 (single default)
- ∀s ∈ S: ψ(s) = PARALLEL ⟹ ∀c ∈ children(s): c ∈ σ₀ (all children active)




| Field | Type | Description |
| ----- | ---- | ----------- |
| root_state |[State](#statecharts-v1-State)|  Root node ∈ S, must be labeled "__root__"  |
| transitions[] |[Transition](#statecharts-v1-Transition)|  Transition relation δ ⊆ S×E×G×A×S  |
| events[] |[Event](#statecharts-v1-Event)|  Event alphabet E (superset allowed)  |
| variables |Struct| Global context and metadata  Global context Γ: Var → Val  |
| name |string|  Human-readable identifier  |
| description |string|  Natural language specification  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-State"></a>

### State

State represents a node in the statechart hierarchy.

FORMAL DEFINITION [H87, Section 2.2]:
A state s ∈ S is characterized by:
- label: unique identifier within statechart
- type: ψ(s) ∈ {BASIC, NORMAL, PARALLEL, ...}
- parent: unique p where (p,s) ∈ ρ (except root)
- children: {c ∈ S | (s,c) ∈ ρ}

WELL-FORMEDNESS CONSTRAINTS [H87, HN96]:
1. Unique labeling: ∀s₁,s₂ ∈ S: s₁ ≠ s₂ ⟹ label(s₁) ≠ label(s₂)
2. Type consistency: ψ(s) = BASIC ⟺ children(s) = ∅
3. Default existence: ψ(s) = NORMAL ∧ children(s) ≠ ∅ ⟹ ∃!c ∈ children(s): is_initial(c)
4. Parallel semantics: ψ(s) = PARALLEL ⟹ ∀c ∈ children(s): ¬is_initial(c)
5. Tree structure: ∀s ∈ S\{root}: ∃!p ∈ S: (p,s) ∈ ρ

ACTION SEMANTICS [HN96, Section 6]:
Entry/exit actions implement the mapping γ: S → A* for state-based actions.
Do-activities provide continuous behavior while state is active.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|  Unique identifier: s.label ∈ String  |
| type |[StateType](#statecharts-v1-StateType)|  Type function: ψ(s) ∈ StateType  |
| children[] |[State](#statecharts-v1-State)|  Hierarchy: {c | (s,c) ∈ ρ}  |
| is_initial |bool|  Default child: λ(parent) = {s} when true  |
| is_final |bool|  Terminal state: no outgoing transitions  |
| entry_actions[] |[Action](#statecharts-v1-Action)| Action mappings γ: S → A* [H87, HN96, Section 6]  γₑₙₜᵣᵧ(s): executed on state entry  |
| exit_actions[] |[Action](#statecharts-v1-Action)|  γₑₓᵢₜ(s): executed on state exit  |
| is_history |bool| History pseudostate fields [UML 2.5, Section 14.5.5] Optional - only set when state is a history pseudostate  True if this is a history pseudostate  |
| history_type |[HistoryType](#statecharts-v1-HistoryType)|  SHALLOW (H) or DEEP (H*) history  |
| metadata |Struct| Extension points for tool-specific data These fields allow extractors and importers to preserve provenance and domain-specific metadata without modifying the core schema.  Ad-hoc key-value annotations  |
| extensions[] |Any|  Typed extension messages  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Transition"></a>

### Transition

Transition represents an edge in the statechart's transition relation.

FORMAL DEFINITION [H87, Section 3]:
A transition t ∈ δ is a 5-tuple t = (src, event, guard, action, tgt) where:
- src ∈ S: source state(s) 
- event ∈ E ∪ {τ}: triggering event (τ for completion)
- guard ∈ G: boolean condition
- action ∈ A*: sequence of actions to execute
- tgt ∈ S: target state(s)

ENABLEMENT SEMANTICS [HN96, Section 4]:
Transition t is enabled in configuration σ iff:
1. src ∩ σ ≠ ∅ (source active)
2. event occurred in current step
3. guard evaluates to true
4. No higher priority transition enabled

CONFLICT RESOLUTION [HN96, Section 4.3]:
Priority ordering resolves multiple enabled transitions:
- Explicit priority values (higher = more priority)
- Hierarchical ordering (deeper states win)
- Textual ordering (deterministic fallback)

FIRING SEMANTICS [HN96, Section 5]:
When fired, transition execution follows precise sequence:
1. Exit states (src to LCA)
2. Execute transition actions
3. Enter states (LCA to tgt)




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|  Human-readable identifier  |
| from[] |string|  Source states: src ∈ P(S)  |
| to[] |string|  Target states: tgt ∈ P(S)  |
| event |string|  Trigger event: e ∈ E ∪ {τ}  |
| guard |[Guard](#statecharts-v1-Guard)|  Guard condition: g ∈ G → Bool  |
| actions[] |[Action](#statecharts-v1-Action)|  Action sequence: α ∈ A*  |
| priority |int32| Priority for conflict resolution [HN96, Section 4.3]  Explicit priority (higher = more priority)  |
| metadata |Struct| Extension points for tool-specific data  Ad-hoc key-value annotations  |
| extensions[] |Any|  Typed extension messages  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Event"></a>

### Event

Event represents an element of the communication alphabet.

FORMAL DEFINITION [H87, Section 3]:
Events E constitute the communication alphabet for statechart execution.
An event e ∈ E may carry parameters and trigger state transitions.

CORE SEMANTICS [H87, HN96]:
Events are atomic communication signals between the statechart and its environment.
The original formalism treats events uniformly without complex type taxonomies.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|  Event identifier: e.label ∈ String  |
| parameters |Struct|  Event payload: param(e) → Value  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Guard"></a>

### Guard

Guard represents a boolean condition for transition enablement.

FORMAL DEFINITION [H87, Section 3.1]:
A guard g ∈ G is a boolean expression over variables and event parameters:
g: Context × Event → Bool

EVALUATION SEMANTICS [HN96, Section 4.2]:
Guards are evaluated atomically during transition enablement testing.
Implementation languages should provide deterministic evaluation semantics.




| Field | Type | Description |
| ----- | ---- | ----------- |
| expression |string| Legacy: raw expression string (deprecated, use condition field)  Boolean expression: g ∈ G  |
| language |string|  Expression language hint (for legacy expressions)  |
| condition |[Expression](./expressions.md#statecharts-v1-Expression)| Structured expression: evaluatable guard condition When present, takes precedence over the deprecated expression field.

EVALUATION SEMANTICS: eval(condition, Γ, e) → Bool where Γ is the current context and e is the triggering event.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Action"></a>

### Action

Action represents executable code associated with statechart elements.

FORMAL DEFINITION [H87, Section 3.3; HN96, Section 6]:
Actions α ∈ A are side-effecting computations executed during statechart
operation. The action mapping γ: S ∪ δ → A* assigns action sequences.

EXECUTION SEMANTICS [HN96, Section 6]:
Actions execute atomically and may modify the statechart context.
The core formalism distinguishes actions by their execution context
(entry/exit for states, effect for transitions) rather than complex taxonomies.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|  Action identifier  |
| expression |string|  Legacy: executable code string  |
| language |string|  Implementation language hint (for legacy)  |
| parameters |Struct|  Action parameters  |
| body |[Expression](./expressions.md#statecharts-v1-Expression)| Structured expression: evaluatable action body When present, takes precedence over the deprecated expression field.

EXECUTION SEMANTICS: exec(body, Γ) → Γ' where Γ is the current context and Γ' is the updated context.

For CEL expressions, actions typically use assignment syntax: "context.health = context.health - damage"

For Starlark expressions, actions can include control flow: "def on_enter(ctx): ctx.emit('SOUND', {'name': 'beep'})"   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-StateRef"></a>

### StateRef

StateRef represents a reference to a state within a statechart.

FORMAL DEFINITION [H87, Section 2]:
A state reference r is simply a label mapping r: Label → S where
Label is the set of all state identifiers and S is the state set.




| Field | Type | Description |
| ----- | ---- | ----------- |
| label |string|  State identifier: r(label) ∈ S  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Configuration"></a>

### Configuration

Configuration represents the active state set of a statechart at any point in time.

FORMAL DEFINITION [H87, Section 4; HN96, Section 3]:
A configuration σ ∈ P(S) is a subset of states that are simultaneously active.
The configuration must satisfy consistency constraints for hierarchical states.

WELL-FORMEDNESS CONSTRAINTS [HN96, Section 3.2]:
1. Ancestry closure: ∀s ∈ σ, ∀p ∈ ancestors(s): p ∈ σ (active states imply active ancestors)
2. OR-consistency: ∀s ∈ σ: ψ(s) = NORMAL ⟹ |{c ∈ children(s) ∩ σ}| ≤ 1 (at most one child active)
3. AND-consistency: ∀s ∈ σ: ψ(s) = PARALLEL ∧ s ∈ σ ⟹ children(s) ⊆ σ (all children active)
4. Basic leaves: ∀s ∈ σ: ψ(s) = BASIC ⟹ children(s) ∩ σ = ∅ (basic states have no active children)

SEMANTIC INVARIANTS [H87, HN96]:
- Consistency: isConsistent(σ) ≡ satisfies all well-formedness constraints
- Maximality: isMaximal(σ) ≡ cannot add more states without violating consistency
- Reachability: isReachable(σ) ≡ ∃ event sequence from initial configuration

HISTORY SEMANTICS [UML 2.5, SCXML]:
History tracking enables restoration of previous configurations when re-entering
composite states, supporting both shallow and deep history variants.




| Field | Type | Description |
| ----- | ---- | ----------- |
| states[] |[StateRef](#statecharts-v1-StateRef)|  Active state set: σ ⊆ S  |
| history |[Configuration.HistoryEntry](#statecharts-v1-Configuration-HistoryEntry)| History mechanism [UML 2.5] for configuration restoration  H: Label → P(S) history mapping  |






<a name="statecharts-v1-Configuration-HistoryEntry"></a>

### HistoryEntry





| Field | Type | Description |
| ----- | ---- | ----------- |
| key |string|   |
| value |[Configuration](#statecharts-v1-Configuration)|   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Machine"></a>

### Machine

Machine represents a statechart instance with execution state.

OPERATIONAL SEMANTICS [HN96, Section 4]:
A machine M = (SC, σ, Γ) consists of:
- SC: statechart definition (static structure)
- σ: current configuration (active states)
- Γ: variable context (dynamic data)

EXECUTION MODEL [HN96, Section 4.1]:
Machine execution proceeds through discrete steps triggered by events,
maintaining the current configuration and context state.




| Field | Type | Description |
| ----- | ---- | ----------- |
| id |string|  Machine identifier  |
| state |[MachineState](#statecharts-v1-MachineState)|  Execution state (running/stopped)  |
| context |Struct|  Variable context: Γ: Var → Val  |
| statechart |[Statechart](#statecharts-v1-Statechart)|  Static definition: SC  |
| configuration |[Configuration](#statecharts-v1-Configuration)|  Current configuration: σ ∈ P(S)  |
| step_history[] |[Step](#statecharts-v1-Step)|  Execution trace for analysis  |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-Step"></a>

### Step

Step represents a single execution step in statechart operation.

STEP SEMANTICS [HN96, Section 4.2]:
A step σ →ᵉ σ' represents the transition from configuration σ to σ'
triggered by event set E', executing the enabled transitions.

STEP COMPONENTS [HN96, Section 5]:
Each step records:
- Input events that triggered the step
- Enabled and fired transitions
- Configuration change (σ → σ')
- Executed actions during the transition sequence




| Field | Type | Description |
| ----- | ---- | ----------- |
| events[] |[Event](#statecharts-v1-Event)|  Triggering events: E' ⊆ E  |
| transitions[] |[Transition](#statecharts-v1-Transition)|  Fired transitions: T' ⊆ δ  |
| starting_configuration |[Configuration](#statecharts-v1-Configuration)|  Initial configuration: σ  |
| resulting_configuration |[Configuration](#statecharts-v1-Configuration)|  Final configuration: σ'  |
| context |Struct|  Updated context: Γ'  |
| states_entered[] |string| Execution trace information  States entered: enter(σ')  |
| states_exited[] |string|  States exited: exit(σ)  |
| actions_executed[] |[Action](#statecharts-v1-Action)|  Actions fired: α*  |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-v1-StateType"></a>

### StateType
StateType describes the fundamental classification of states in Harel's statechart formalism.

THEORETICAL FOUNDATION [H87, Section 2.1]:
The type function ψ: S → {BASIC, OR, AND} categorizes states by their decomposition:
- BASIC: atomic states with no substates (ψ(s) = BASIC ⟹ children(s) = ∅)
- OR: exclusive decomposition with XOR substate semantics (normal states)
- AND: concurrent decomposition with parallel substate semantics (orthogonal states)

SEMANTIC PROPERTIES [HN96, Section 3]:
- OR-states: ∀σ,s: ψ(s)=OR ⟹ |{c ∈ children(s) : c ∈ σ}| ≤ 1
- AND-states: ∀σ,s: ψ(s)=AND ∧ s ∈ σ ⟹ children(s) ⊆ σ

Note: This enum contains ONLY the core types from Harel's original formalism.
Modern extensions (history, pseudo-states) are intentionally excluded to maintain
theoretical purity and alignment with the cited academic papers.



| Name | Number | Description |
| ---- | ------ | ----------- |
| STATE_TYPE_UNSPECIFIED | 0 |  Undefined type (invalid in well-formed statecharts)  |
| STATE_TYPE_BASIC | 1 | Core Harel state types [H87, Section 2.1]  Atomic state: ψ(s) = BASIC ⟹ children(s) = ∅  |
| STATE_TYPE_OR | 2 |  OR-decomposition: exclusive substate semantics  |
| STATE_TYPE_AND | 3 |  AND-decomposition: concurrent substate semantics  |
| STATE_TYPE_NORMAL | 2 | Academic terminology aliases [H87] for clarity  Alias for OR (common in literature)  |
| STATE_TYPE_PARALLEL | 3 |  Alias for AND (UML terminology)  |
| STATE_TYPE_ORTHOGONAL | 3 |  Alias for AND (Harel's original terminology)  |




<a name="statecharts-v1-HistoryType"></a>

### HistoryType
HistoryType distinguishes shallow vs deep history pseudostates.

FORMAL DEFINITION [UML 2.5, Section 14.5.5]:
History pseudostates enable restoration of prior configurations when
re-entering composite states, supporting two semantic variants:
- SHALLOW: Restores only the immediate child state of the composite
- DEEP: Restores the full nested configuration recursively

SEMANTIC PROPERTIES:
- H (shallow): restore(H, σ) = {s ∈ σ_prev | parent(s) = parent(H)}
- H* (deep): restore(H*, σ) = σ_prev ∩ descendants(parent(H*))



| Name | Number | Description |
| ---- | ------ | ----------- |
| HISTORY_TYPE_UNSPECIFIED | 0 |  Default: treated as SHALLOW  |
| HISTORY_TYPE_SHALLOW | 1 |  H: restore immediate child only  |
| HISTORY_TYPE_DEEP | 2 |  H*: restore full nested configuration  |




<a name="statecharts-v1-MachineState"></a>

### MachineState
MachineState encodes the execution status of a statechart interpreter.

OPERATIONAL SEMANTICS [HN96, Section 4]:
Machine state tracks the interpreter's lifecycle, distinct from the 
statechart's logical configuration σ ∈ P(S).



| Name | Number | Description |
| ---- | ------ | ----------- |
| MACHINE_STATE_UNSPECIFIED | 0 |  Undefined interpreter state  |
| MACHINE_STATE_RUNNING | 1 |  Interpreter active, processing events  |
| MACHINE_STATE_STOPPED | 2 |  Interpreter halted, no event processing  |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

