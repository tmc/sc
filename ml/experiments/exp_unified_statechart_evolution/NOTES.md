# Unified Statechart Evolution - FLAGSHIP Experiment

## Overview

This is the ENDGAME unification experiment. Instead of evolving individual statechart
components in isolation, we co-evolve ALL components together in a single `UnifiedGenome`.

## Gene Types

The UnifiedGenome contains six interdependent gene categories:

### 1. Topology Genes (from exp_topology_evolution)
- `n_states`: Number of states in the machine
- `parent[]`: Hierarchy structure (parent[i] = parent state of i, -1 for root)
- `state_type[]`: BASIC, OR (XOR children), AND (parallel regions)

### 2. History Genes (from exp_deep_history)
- `history_type[]`: Per-composite-state history setting
  - NONE: No history, enter via initial child
  - SHALLOW (H): Restore direct child only
  - DEEP (H*): Restore full nested configuration

### 3. Transition Genes
Each transition has:
- `src, tgt`: Source and target states
- `event`: Triggering event ID
- `guard`: Condition (op, var, val) - e.g., EQ("x", 5)
- `priority`: Explicit priority for conflict resolution
- `actions[]`: Side effects to execute

### 4. Guard Genes (from exp_transition_priorities)
- `GuardOp`: TRUE, FALSE, EQ, NE, LT, LE, GT, GE
- Guards reference context variables
- Specificity contributes to effective priority

### 5. Action Genes (from exp_action_side_effects)
- `ActionType`: SET, INCREMENT, DECREMENT, TOGGLE, APPEND, CLEAR
- Actions modify context variables
- Enable guard conditions for subsequent transitions

### 6. Event/Context Genes
- `n_events`: Number of distinct events
- `context_variables[]`: Named variables for guards/actions
- `initial_context{}`: Starting values

## Multi-Objective Optimization (NSGA-II)

Three competing objectives form the Pareto front:

### 1. Correctness (Primary)
- Accuracy on test scenarios
- Weighted most heavily (1.0 default)
- Target: Maximize correct state predictions

### 2. Minimality
- State parsimony: Fewer states is better
- Transition parsimony: Fewer transitions is better
- Action parsimony: Fewer actions is better
- Combined: 0.4 * states + 0.4 * transitions + 0.2 * actions

### 3. Interpretability
- Hierarchy score: Deeper structure = more organized
- History usage: Appropriate use of history features
- Guard clarity: ~30-50% of transitions should have guards
- Combined: 0.3 * hierarchy + 0.3 * history + 0.4 * guards

## NSGA-II Selection Algorithm

1. **Non-dominated sorting**: Compute Pareto fronts
2. **Crowding distance**: Maintain diversity within fronts
3. **Selection**: Fill population front-by-front
4. **Tournament**: Compare by rank, then crowding

## Mutation Operators

### Topology Mutations
- `mutate_add_state`: Add new state to hierarchy
- `mutate_remove_state`: Remove leaf state
- `mutate_reparent_state`: Move state to different parent

### History Mutations
- `mutate_history_type`: Change NONE/SHALLOW/DEEP for composite states

### Transition Mutations
- `mutate_add_transition`: Add new transition
- `mutate_remove_transition`: Remove existing transition
- `mutate_retarget_transition`: Change src or tgt
- `mutate_transition_event`: Change triggering event
- `mutate_guard`: Modify guard condition
- `mutate_priority`: Change explicit priority
- `mutate_action`: Add/remove/modify actions

## Key Insight: Component Interactions

Why co-evolution matters:

1. **Topology affects guards**: Different state structures need different guard patterns
2. **History affects reachability**: History changes which states can be reached
3. **Actions enable guards**: Actions modify variables that guards test
4. **Priorities resolve conflicts**: When topology creates multiple enabled transitions

These interactions mean components CANNOT be optimized independently.

## Connections to Other Experiments

### exp_topology_evolution
- Foundation for hierarchical state structure
- Parent/child relationships
- State type classification

### exp_deep_history
- History type per state
- Shallow vs deep restoration semantics
- History machine execution

### exp_transition_priorities
- Guard specificity calculation
- Priority-based conflict resolution
- Transition selection algorithm

### exp_action_side_effects
- Action effect types
- Context modification
- Guard enablement through actions

### exp_inter_level_transitions
- Transitions between hierarchy levels
- Entry/exit state resolution
- Configuration updates

### exp_differentiable_statecharts
- Future: Could add differentiable fitness
- Gradient-based optimization of continuous parameters

## Implementation Details

### UnifiedMachine Execution
1. Receive event
2. Find enabled transitions (guard evaluation)
3. Resolve conflicts (priority + specificity)
4. Save history for exited composites
5. Execute actions
6. Resolve target (history if composite)
7. Update configuration

### Scenario Format
```python
scenarios = [
    (events=[0, 1, 2], expected_state=3),
    (events=[0, 0, 1], expected_state=5),
]
```

### Fitness Computation
```python
components = compute_unified_fitness(
    genome,
    scenarios,
    max_states=20,
    max_transitions=50
)
# Returns: correctness, minimality, interpretability
```

## Future Directions

1. **Symbolic Regression**: Evolve guard expressions, not just operators
2. **Hierarchical Crossover**: Exchange entire subtrees between genomes
3. **Transfer Learning**: Pre-train on simple scenarios, fine-tune on complex
4. **Neural Fitness**: Learn fitness function from expert preferences
5. **Parallel Regions**: Extend to AND-decomposition with synchronization

## References

- `semantics/v1/machine.go`: Execution semantics reference
- `proto/statecharts/v1/statecharts.proto`: Full statechart model
- Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
- Harel, "Statecharts: A Visual Formalism for Complex Systems"
