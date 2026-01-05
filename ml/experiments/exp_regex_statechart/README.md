# exp_regex_statechart: Evolve Statecharts from Regex Behavior

## Goal

Given positive and negative examples, evolve a statechart that matches
RE2 regular expression behavior.

## Why This Is Simpler

Regular expressions ARE finite automata:
- DFA/NFA = statechart (direct mathematical equivalence)
- RE2 guarantees linear-time matching (no backtracking)
- Clear ground truth from RE2 engine

## Approach

### Input
```
Positive examples: ["abd", "abbd", "acd", "abcbd"]
Negative examples: ["ab", "d", "aad", "abdc"]
```

### Output
```
Statechart matching regex: a(b|c)*d

States: START, SAW_A, IN_BC, ACCEPT
Transitions:
  START  --'a'--> SAW_A
  SAW_A  --'b'--> IN_BC
  SAW_A  --'c'--> IN_BC
  SAW_A  --'d'--> ACCEPT
  IN_BC  --'b'--> IN_BC
  IN_BC  --'c'--> IN_BC
  IN_BC  --'d'--> ACCEPT
```

## Evolution Strategy

Use topology evolution (exp_topology_evolution) adapted for regex:

1. **State Mutations**:
   - Add/remove states
   - Mark states as accepting

2. **Transition Mutations**:
   - Add/remove character transitions
   - Change transition targets
   - Widen/narrow character classes

3. **Fitness**:
   - F1 score on positive/negative examples
   - Bonus for smaller statecharts (Occam's razor)

## Benchmark Patterns

| Pattern | Complexity | Description |
|---------|------------|-------------|
| `a*` | Simple | Zero or more 'a' |
| `(ab)+` | Concatenation | One or more 'ab' |
| `[a-z]+` | Character class | Lowercase words |
| `\d{3}-\d{4}` | Quantifiers | Phone numbers |
| `(foo|bar)baz` | Alternation | Choice pattern |
| `a.*b` | Wildcards | Anything between |

## Files to Implement

1. `regex_statechart.py` - Statechart for character-level matching
2. `re2_oracle.py` - Ground truth from RE2 engine
3. `evolver.py` - Evolution with regex-specific mutations
4. `benchmark.py` - Test suite of patterns
5. `synthesis.py` - End-to-end regex synthesis

## Evaluation

| Metric | Target |
|--------|--------|
| F1 on test examples | 99%+ |
| Pattern reconstruction | 90%+ |
| Evolution generations | <100 |
| Statechart minimality | ≤ DFA size |

## Connection to Other Experiments

| Experiment | Contribution |
|------------|--------------|
| exp_topology_evolution | Evolution framework |
| exp_guard_synthesis | Character guards |
| exp_transfer_learning | Pattern structure transfer |

## Research Questions

1. Can evolution discover minimal DFAs?
2. How many examples needed per pattern complexity?
3. Can learned patterns generalize to new inputs?
4. Transfer: train on simple patterns, synthesize complex ones?
