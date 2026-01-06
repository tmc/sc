# exp_active_regex_learning: Research Notes

## Goal

Minimize examples needed for regex/statechart synthesis via active learning.
Target: **50% fewer examples** than passive (random) learning.

## Approach

### Active Learning Loop

```
1. Initialize with few random examples
2. Build hypothesis from labeled examples
3. Select next query using strategy:
   - Uncertainty: query where model is uncertain
   - Boundary: query near decision boundary
   - Diversity: query strings different from existing
4. Query oracle for label
5. Update hypothesis
6. Repeat until convergence or budget exhausted
```

### Query Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Random | Baseline - random selection | Control |
| Uncertainty | Query where prob ≈ 0.5 | General active learning |
| Boundary | Query strings near decision boundary | When boundary is complex |
| Committee | Query where ensemble disagrees | Multiple hypotheses |
| Diversity | Maximize coverage | Avoid redundant queries |
| Hybrid | Uncertainty + Diversity | Best of both worlds |

## Key Insight

**Uncertainty sampling works best for regex learning** because:
1. DFA boundaries are sharp (string either matches or doesn't)
2. Strings near boundary are most informative
3. N-gram features capture local structure well

## Implementation

### oracle_interface.py

Provides ground truth for benchmark patterns:
- `(ab)*` - Simple alternation
- `even_zeros` - DFA with state tracking
- `no_consecutive` - Negative lookahead equivalent
- `binary_div3` - Classic DFA example
- `identifier` - Programming identifiers
- `email_prefix` - Complex real-world pattern

### query_strategy.py

Uncertainty sampling implementation:
```python
entropy = -p*log(p) - (1-p)*log(1-p)
# High entropy = high uncertainty = good query
```

Uses n-gram features for probability estimation.

### active_learner.py

Main loop with:
- Candidate pool management
- Batch querying
- Holdout evaluation
- Convergence detection

### sample_efficiency.py

Comparison framework:
- Queries to reach X% accuracy
- Learning curves
- Statistical comparison across runs

## Results (Preliminary)

Pattern-dependent efficiency gains:

| Pattern | Passive to 90% | Active to 90% | Reduction |
|---------|---------------|---------------|-----------|
| ab_star | ~60 queries | ~35 queries | ~42% |
| even_zeros | ~40 queries | ~25 queries | ~38% |
| no_consecutive | ~50 queries | ~30 queries | ~40% |

**Note**: Simple DFA hypothesis limits accuracy. With better hypothesis learning (e.g., state merging), expect higher gains.

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_regex_statechart` | Target patterns for synthesis |
| `exp_grammar_induction` | Similar inductive learning |
| `exp_coverage_prediction` | Coverage guides queries |
| `exp_priority_attention` | Could attend to informative strings |

## Future Directions

1. **Better hypothesis models**: State merging, RPNI algorithm
2. **Membership + equivalence queries**: L* style learning
3. **Transfer across patterns**: Pre-trained uncertainty model
4. **Negative mining**: Generate hard negative examples
5. **Human-in-the-loop**: Interactive labeling interface

## Files

```
exp_active_regex_learning/
├── __init__.py           # Package exports
├── oracle_interface.py   # Ground truth oracles
├── query_strategy.py     # Query selection strategies
├── active_learner.py     # Active learning loop
├── sample_efficiency.py  # Comparison framework
└── NOTES.md              # This file
```

## Status

- [x] Oracle interface with benchmark patterns
- [x] Query strategies (uncertainty, boundary, diversity, hybrid)
- [x] Active learning loop
- [x] Comparison framework
- [x] Basic testing
- [ ] Achieve consistent 50% reduction
- [ ] Better hypothesis model (state merging)

## Date

2026-01-04
