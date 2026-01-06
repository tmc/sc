# exp_sc_equivalence: Behavioral Equivalence Checking

## Goal

Determine if two statecharts are behaviorally equivalent - do they
accept the same traces (event sequences leading to equivalent outcomes)?

## Methods

### 1. Trace Comparison
- Enumerate all traces up to depth N
- Compare trace sets between SCs
- High accuracy but limited by depth

### 2. Structural Isomorphism
- Compare state counts, transitions, events
- Fast but may miss semantic equivalence
- Different names can be isomorphic

### 3. Bisimulation
- Formal equivalence relation
- Two states bisimilar if:
  - Same outgoing events
  - Corresponding targets also bisimilar
- Strongest guarantee

### 4. Combined (default)
- Use trace + structural for robust checking
- Higher confidence when methods agree

## Architecture

```
trace_generator.py      - Enumerate execution traces (BFS)
equivalence_checker.py  - Main equivalence analysis
benchmark.py            - Accuracy/FP/FN evaluation
```

## Benchmark Cases

| Category | Cases | Description |
|----------|-------|-------------|
| Equivalent | 5 | Identical, renamed, reordered |
| Non-equivalent | 7 | Missing state, different events, etc. |

**Total: 12 test cases**

## Evaluation Metrics

- **Accuracy**: % correct verdicts
- **False Positive Rate**: % incorrectly marked equivalent
- **False Negative Rate**: % incorrectly marked non-equivalent

## Key Insights

1. **Trace comparison is most reliable** for finite depths
2. **Structural similarity != behavioral equivalence**
3. **Unreachable states** don't affect behavioral equivalence
4. **Bisimulation** is computationally harder but more precise

## Challenges

- State name differences (isomorphism needed)
- Infinite traces (depth limiting)
- Guards make equivalence undecidable in general
- Hierarchical states add complexity

## Dependencies

- Pure Python implementation
- No LLM needed for core equivalence checking
- LLM could help explain non-equivalence
