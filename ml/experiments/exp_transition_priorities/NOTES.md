# exp_transition_priorities: Research Notes

## Key Insight

**Specificity beats explicit priority.** When multiple transitions are enabled, prioritizing by guard specificity (how restrictive the condition is) outperforms arbitrary numeric priorities.

```
SpecificityPriorityStrategy: 0.564 mean fitness
LearnedPriorityStrategy:     0.405 mean fitness
ExplicitPriorityStrategy:    0.363 mean fitness
```

This makes intuitive sense: a guard like `state_eq(x, 5)` should fire before `has_nonzero` because it's more specific. The "most specific applicable rule" principle from expert systems applies here.

## What Worked

1. **Auto-computed specificity scores** - Assigning specificity based on predicate type:
   - `true` → 0.0 (always matches = least specific)
   - `has_nonzero` → 0.3 (broad condition)
   - `has_color(c)` → 0.6 (checks specific value)
   - `state_eq(k,v)` → 0.8 (exact match = very specific)

2. **Two-level evolution** - Evolving both:
   - Transition priority values (inner loop)
   - Priority strategy itself (outer loop)

3. **Strategy pool** - Starting with diverse strategies lets evolution discover what works without bias.

4. **Conflict tracking** - Recording wins/losses per transition enables post-hoc analysis of which transitions dominate.

## What Didn't Work

1. **Learned weighted combination** - The `LearnedPriorityStrategy` that combines multiple factors (explicit, specificity, fire_count, recency) performed worse than pure specificity. Simpler is better here.

2. **Selection noise** - Adding Gaussian noise to priorities for exploration hurt more than helped. Deterministic selection works better once good priorities are found.

3. **Fire count as priority factor** - Preferring frequently-fired transitions creates positive feedback loops that hurt exploration.

## Future Directions

1. **Hierarchical specificity** - Compute specificity from guard AST structure, not just predicate type. Conjunctions of conditions should be more specific than single conditions.

2. **Context-dependent priorities** - Priority could depend on execution history (what states we came from). This connects to history states in statecharts.

3. **Learning specificity scores** - Instead of hardcoding specificity by predicate type, learn it from data. Which guards empirically fire less often?

4. **Priority gradients** - Could we backprop through priority selection? Soft argmax over priorities would enable gradient-based learning.

5. **Multi-objective** - Balance specificity (correctness) vs coverage (all transitions get used).

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_guard_synthesis` | Guards created there need priorities assigned here |
| `exp_rule_discovery` | Discovered rules have implicit priorities from specificity |
| `exp_soar_statechart` | REX uses priority for parent selection - same principle |
| `exp_learnable_policies` | Policy could learn to set priorities dynamically |
| `exp_ko_evolution` | Ko rules are highly specific - should have high priority |

## Implementation Notes

- Pure MLX, no numpy
- `Transition.priority` field now actively used (was dormant)
- Specificity computed lazily and cached on `Guard._specificity`
- Strategies are clonable for evolution (`.clone()` method)

## Open Questions

1. Should specificity be normalized across all guards in a statechart?
2. How do nested/hierarchical states affect priority? Inner states more specific?
3. Can we learn a "priority network" that takes guard embedding → priority?
4. What's the relationship between priority and transition probability in soft/differentiable statecharts?

## Code Patterns

```python
# Conflict resolution in one line
winner = max(enabled, key=lambda t: strategy.compute_priority(t))

# Specificity from predicate type
specificity_map = {
    "true": 0.0,      # Least specific
    "state_eq": 0.8,  # Most specific
}
```

## Date
2024-01-04
