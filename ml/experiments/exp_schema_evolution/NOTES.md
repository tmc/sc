# Schema Evolution Experiment

## Overview

This experiment learns safe statechart migrations from example pairs.
Instead of manually writing migration rules, we synthesize them from
(old_config, new_config) examples.

## Theoretical Foundation

From proto/statecharts/v1/evolution.proto:

```
Schema evolution follows principles from database theory [BCH+07]:
- Compatibility: new version accepts inputs valid for old version
- Substitutability: new version preserves observable behavior
- Migrability: running instances can be transformed to new version

Migration formalism:
  migrate: Configuration_old × MappingRules → Configuration_new
```

## Schema Diff

### Change Types

| Type | Semantics |
|------|-----------|
| ADDED | s ∈ S₂ ∧ s ∉ S₁ |
| REMOVED | s ∈ S₁ ∧ s ∉ S₂ |
| MODIFIED | s ∈ S₁ ∩ S₂ ∧ attrs differ |
| RENAMED | structure match, label changed |
| MOVED | parent changed |

### Breaking Changes

- STATE_REMOVED: Requires migration mapping
- STATE_TYPE_CHANGED: May require manual intervention
- INITIAL_STATE_CHANGED: Affects new machine creation
- TRANSITION_REMOVED: May affect reachability
- GUARD_ADDED: May block previously valid paths

## State Mapping Types

From proto StateMappingType:

| Type | Semantics | Use Case |
|------|-----------|----------|
| IDENTITY | Same label | Unchanged states |
| RENAME | Label changed | Refactoring |
| TO_PARENT | → default(parent) | State removed, parent exists |
| TO_SIBLING | → sibling state | State removed, sibling exists |
| TO_INITIAL | → chart initial | State removed, no parent |
| SPLIT | → f(context) | Conditional mapping |
| MERGE | many → one | Consolidation |

## Evolution Approach

### Algorithm

1. Compute schema diff
2. Generate candidate mappings for each change
3. Evolve mapping selection on example pairs
4. Verify safety properties
5. Generate rollback plan

### Fitness Function

```python
fitness = correct_migrations / total_examples
```

Where a migration is correct if:
- All old states are removed
- All target states are added
- Target states exist in new chart

### Mutation Operators

- Change target state
- Change mapping type
- Add/remove context transformation

## Verification Properties

### Completeness
Every removed state must have a mapping.

### Determinism
Each old state maps to exactly one new state (unless conditional).

### Reachability
Target states must exist in new chart.

### Reversibility
Rollback mappings should restore original state.

## Benchmark Scenarios

### 1. State Removal
Old: Idle → Active → Done
New: Idle → Active → Complete

Migration: Done → Complete

### 2. State Rename
Old: Login → Settings
New: SignIn → Preferences

Migration: Login → SignIn, Settings → Preferences

### 3. Hierarchy Change
Old: A, B, C (flat)
New: Parent{A, B}, C

Migration: A → A+Parent, B → B+Parent

### 4. Feature Addition
Old: Idle → Active
New: Idle → Active → Paused

Migration: Identity (backward compatible)

### 5. Combined Changes
Multiple renames, moves, and additions together.

## Implementation

### schema_diff.py
- `SchemaDiffer`: Computes ChartDiff
- `StateDiff`, `TransitionDiff`, `EventDiff`
- Rename detection via structure similarity

### migration_synthesizer.py
- `MigrationSynthesizer`: Evolves migration plans
- `StateMapping`: Individual mapping rules
- `MigrationPlan`: Complete migration specification

### migration_verifier.py
- `MigrationVerifier`: Checks safety properties
- `dry_run()`: Preview migration results
- Configurable verification rules

### benchmark.py
- 5 migration scenarios
- Example pair generation
- Accuracy and verification metrics

## Connections to Other Experiments

### exp_unified_statechart_evolution
- Uses same Configuration representation
- Could evolve migrations alongside chart structure

### exp_deep_history
- History affects migration (preserve history state?)
- Migration must respect history semantics

### exp_temporal_guards
- Temporal guards may need migration
- Timeout values could change between versions

## Future Directions

1. **Semantic Preservation**: Verify behavioral equivalence
2. **Incremental Migration**: Online migration without downtime
3. **Version Chains**: A → B → C migration composition
4. **Automatic Rollback**: Learn when to trigger rollback
5. **Context Migration**: Transform context variables

## References

- proto/statecharts/v1/evolution.proto
- [BCH+07] Buneman et al., "Principles of Database Schema Evolution"
- [H87] Harel, "Statecharts: A Visual Formalism"
