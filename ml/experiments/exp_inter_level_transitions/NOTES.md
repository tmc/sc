# Experiment: Inter-Level Transitions

## Research Notes

### Key Insight

**LCA paths are LEARNABLE via evolution** - No hardcoded algorithms needed. Evolution discovers:
- Correct LCA (Least Common Ancestor) for any state pair
- Exit sequences (inside-out order per Harel semantics)
- Entry sequences (outside-in order per Harel semantics)

### Core Implementation

#### HierarchicalStatechart

Implements LCA computation matching `semantics/v1/transitions.go:453-474`:

```python
def least_common_ancestor(self, label_a: str, label_b: str) -> Optional[str]:
    ancestors_a = set([label_a] + self.get_ancestors(label_a))
    ancestors_b = set([label_b] + self.get_ancestors(label_b))
    common = ancestors_a & ancestors_b
    # Return deepest common ancestor
    return max(common, key=lambda x: self.get_depth(x))
```

#### LCAPathEvolver

Evolves path predictions for transitions:
- `predicted_exit_path`: States to exit (inside-out)
- `predicted_entry_path`: States to enter (outside-in)
- `predicted_lca`: Predicted scope of transition

Fitness = 0.4×LCA_correct + 0.3×exit_accuracy + 0.3×entry_accuracy

### Results

| Transition | Depth Δ | Exit Path | Entry Path | Fitness |
|------------|---------|-----------|------------|---------|
| PlayerTurn → MainMenu | 4→2 | 4 states | 2 states | 1.000 |
| EnemyTurn → Exploring | 4→3 | 3 states | 2 states | 1.000 |
| Audio → PlayerTurn | 3→4 | 3 states | 4 states | 1.000 |
| PlayerTurn → EnemyTurn | 4→4 | 1 (sibling) | 1 | 1.000 |
| Exploring → Defeat | 3→2 | 3 states | 2 states | 1.000 |

**10/10 transitions evolved to perfect fitness (1.000)**

### What Worked

1. **State hierarchy with proper depth tracking** - Recursive `_update_children_depth()` ensures correct depths
2. **Fitness combining LCA + paths** - All three components needed for full accuracy
3. **Tournament selection** - Works better than roulette for discrete path problems
4. **Constrained mutation** - Only mutate with valid ancestors/descendants

### What Didn't Work (Initially)

1. **Flat depth initialization** - All states had depth=1
   - **Fix**: Recursive depth update in `__post_init__`

2. **Random path generation** - Generated invalid paths
   - **Fix**: Constrain to actual ancestors/descendants

### Connections to Other Experiments

| Experiment | Connection | Integration Opportunity |
|------------|------------|------------------------|
| `exp_deep_history` | LCA determines history save scope | Use LCA to optimize when to save history |
| `exp_action_side_effects` | Exit/entry order determines action order | Predict action sequences from LCA path |
| `exp_transition_priorities` | Depth determines priority | Priority = f(source_depth, lca_depth) |
| `exp_temporal_guards` | Deep transitions take time | Add `transition_time(src, tgt)` temporal |
| `exp_topology_evolution` | Topology defines hierarchy | Unified genome: topology + lca_strategy |
| `exp_internal_vs_external` | Exit actions raise events | LCA recalculation for cascade transitions |
| `exp_ko_evolution` | Ko scope = repetition boundary | LCA determines what "same position" means |

### Future Directions

1. **Differentiable LCA**
   - Neural network predicts LCA from (source, target)
   - Soft attention over ancestor paths
   - Gradient-based learning

2. **LCA-Aware Guards**
   ```python
   depth() > 3                    # Current state depth
   in_ancestor("Playing")         # Is X ancestor of current
   lca_depth("A", "B") <= 2       # LCA depth constraint
   exit_path_length() < 4         # Limit transition scope
   ```

3. **Hierarchy Compression**
   - Learn minimal hierarchy with same behavior
   - Remove unnecessary intermediate states
   - Flatten non-semantic nesting

4. **Unified Hierarchy Genome**
   ```python
   @dataclass
   class UnifiedGenome:
       topology_genes: List[int]      # Parent pointers
       history_genes: List[HistoryType]  # Per-state history
       lca_strategy_genes: Dict        # Path preferences
       priority_genes: List[int]       # Transition priorities
   ```

### Reference Implementation

Go semantics in `semantics/v1/transitions.go`:

```go
// calculateTransitionScope finds LCA of all involved states
func (s *Statechart) calculateTransitionScope(
    transitions []*sc.Transition,
    config *sc.Configuration,
) (StateLabel, error) {
    var allStates []StateLabel
    for _, transition := range transitions {
        allStates = append(allStates, transition.From...)
        allStates = append(allStates, transition.To...)
    }
    return s.LeastCommonAncestor(allStates...)
}

// calculateStateChanges determines exit/enter states
func (s *Statechart) calculateStateChanges(...) ([]StateLabel, []StateLabel, error) {
    // Exit: inside-out (deepest first)
    // Enter: outside-in (shallowest first)
}
```

### Files

```
exp_inter_level_transitions/
├── __init__.py           # Package exports
├── hierarchy_evolver.py  # Core: State, HierarchicalStatechart, LCAPathEvolver
└── NOTES.md              # This file
```

### Metrics

- **LCA Accuracy**: 100% (all 10 transitions)
- **Exit Path Accuracy**: 100%
- **Entry Path Accuracy**: 100%
- **Evolution Generations**: ~40 to converge
- **Population Size**: 40

### Status

- [x] State hierarchy with depth tracking
- [x] LCA computation matching Go semantics
- [x] Exit/entry path calculation
- [x] Evolution of path predictions
- [x] Game hierarchy test (Menu/Playing/GameOver)
- [x] 10/10 perfect fitness on all transitions
- [ ] Integration with exp_deep_history
- [ ] Neural LCA predictor
- [ ] Unified genome with topology

### Created

Date: 2026-01-04
