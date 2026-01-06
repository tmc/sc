# exp_formal_language_suite - Research Notes

## Core Goal

**Unify DFA/NFA/CFG induction into common statechart framework.**

Key insight: All formal languages map to statecharts:
- **DFA** = flat statechart (OR-state with basic children)
- **NFA** = statechart with ε-transitions (completion events)
- **CFG** = hierarchical statechart (recursive composite states)
- **PDA** = statechart with stack context variable

## Chomsky Hierarchy Mapping

| Type | Language Class | Automaton | Statechart Equivalent |
|------|----------------|-----------|----------------------|
| 3 | Regular | DFA/NFA | Flat OR-state |
| 2 | Context-Free | PDA | Hierarchical with recursion |
| 1 | Context-Sensitive | LBA | Bounded context variables |
| 0 | Recursively Enumerable | TM | Unbounded context |

## Algorithms Implemented

### RPNI (Regular Positive and Negative Inference)

**Reference**: Oncina & García, 1992

Algorithm:
1. Build prefix tree acceptor (PTA) from positive examples
2. Order states canonically (BFS order)
3. Try merging states in order
4. Accept merge if consistent with negative examples

**Complexity**: O(n² × |Σ|) where n = total length of examples

**Strengths**:
- Polynomial time
- Identifies in the limit
- Works with both positive and negative examples

**Weaknesses**:
- Sensitive to example order
- May not find minimal DFA

### L* (Angluin's Algorithm)

**Reference**: Angluin, 1987

Algorithm:
1. Initialize observation table (S, E, T)
2. Make table closed and consistent
3. Build hypothesis DFA from table
4. Query equivalence oracle
5. Process counterexample
6. Repeat until equivalent

**Complexity**: O(n² × m × |Σ|) queries where n = states, m = max counterexample

**Strengths**:
- Learns minimal DFA
- Active learning (can query oracle)
- Polynomial in queries

**Weaknesses**:
- Requires equivalence oracle
- May make many membership queries

### Sequitur (Grammar Compression)

**Reference**: Nevill-Manning & Witten, 1997

Algorithm:
1. Read input left to right
2. Replace repeated digrams with rules
3. Enforce digram uniqueness
4. Enforce rule utility (each rule used ≥2 times)

**Complexity**: O(n) in input length

**Strengths**:
- Linear time
- Produces hierarchical structure
- Lossless compression

**Weaknesses**:
- Grammar may not be minimal
- Specific to compression, not language learning

### ADIOS (Automatic Distillation of Structure)

**Reference**: Solan et al., 2005

Algorithm:
1. Build graph from example sequences
2. Find significant patterns via random walks
3. Generalize patterns to rules

**Strengths**:
- Motivated by language acquisition
- Discovers hierarchical structure

**Weaknesses**:
- Heuristic approach
- May not converge to target grammar

## Statechart Conversion

### DFA → Statechart

Direct mapping:
- Each DFA state → basic state in OR-state
- Each transition → statechart transition with event = symbol
- Initial/final states → is_initial/is_final flags

```python
def dfa_to_statechart(dfa: DFA) -> Dict:
    return {
        "root_state": {
            "label": "__root__",
            "type": 2,  # NORMAL (OR-state)
            "children": [
                {"label": s.name, "type": 1, "is_initial": s.is_initial}
                for s in dfa.states.values()
            ]
        },
        "transitions": [
            {"from": [src], "to": [tgt], "event": symbol}
            for src, symbol, tgt in dfa.transitions
        ]
    }
```

### NFA → Statechart

Same as DFA, plus:
- ε-transitions → completion events (empty event string)
- Non-determinism → multiple enabled transitions

### CFG → Hierarchical Statechart

Each non-terminal becomes composite state:
- Alternative productions → OR-state children
- Sequence in production → sequence of states
- Recursion → recursive composite states

```python
S → AB | C     # S is OR-state with two children
A → aA | a     # A is OR-state (recursive)
B → b          # B is basic state
```

Maps to:
```
S (OR-state)
├── prod1 (sequence)
│   ├── A (OR-state, recursive)
│   └── B (basic)
└── prod2 (sequence)
    └── C (basic)
```

## Key Insights

### 1. Hierarchy Emergence

CFG induction naturally discovers hierarchical structure:
- Repeated patterns → rules (composite states)
- Nesting → recursive substates
- Alternatives → OR-state children

This matches how statecharts organize complexity.

### 2. Epsilon Transitions

NFA ε-transitions = statechart completion events:
- Fire without consuming input
- Enable internal state changes
- Model "default" transitions

### 3. Context = Extended State

PDA stack = statechart context variable:
- Stack push/pop = context update actions
- Stack top = guard condition
- Bounded stack = finite context

### 4. Active vs Passive Learning

| Aspect | Active (L*) | Passive (RPNI) |
|--------|-------------|----------------|
| Oracle | Required | Not required |
| Samples | On-demand | Fixed |
| Minimal | Guaranteed | Not guaranteed |
| Use case | Interactive | Batch |

For statechart learning:
- Active: User provides examples interactively
- Passive: Learn from trace logs

## Benchmarking Results (Expected)

### Algorithm Comparison

| Language | RPNI States | L* States | RPNI Time | L* Time |
|----------|-------------|-----------|-----------|---------|
| (ab)* | 2 | 2 | ~1ms | ~5ms |
| ends_ab | 3 | 3 | ~2ms | ~10ms |
| even_a | 2 | 2 | ~1ms | ~5ms |
| div_by_3 | 3 | 3 | ~3ms | ~15ms |

### Scaling Behavior

| n_examples | RPNI Time | L* Queries |
|------------|-----------|------------|
| 10 | ~1ms | ~20 |
| 50 | ~5ms | ~100 |
| 100 | ~20ms | ~400 |
| 500 | ~500ms | ~2000 |

### Detection Accuracy

| Language Type | Detection Rate |
|---------------|----------------|
| Regular | 95%+ |
| Context-Free (nested) | 90%+ |
| Context-Free (non-nested) | 70% |

## Future Directions

### 1. Neural Induction

Combine with neural language models:
- Use LLM to generate positive examples
- Learn statechart from LLM outputs
- Verify LLM behavior via statechart

### 2. Incremental Learning

Online adaptation:
- Start with small grammar
- Refine as new examples arrive
- Maintain consistency

### 3. Transfer Learning

Use learned statecharts across domains:
- Extract common patterns
- Fine-tune for specific applications
- Compose learned automata

### 4. Probabilistic Extension

Add probabilities to transitions:
- PDFA (Probabilistic DFA)
- PCFG (Probabilistic CFG)
- Model likelihood of sequences

## Integration with Other Experiments

| Experiment | Integration |
|------------|-------------|
| exp_regex_statechart | Base for regex → statechart |
| exp_grammar_induction | CFG learning algorithms |
| exp_active_regex_learning | L* active learning |
| exp_code_completion | Starlark syntax as CFG |
| exp_sae_coverage | Discover language from activations |

## References

- Angluin, D. "Learning Regular Sets from Queries and Counterexamples" (1987)
- Oncina, J. & García, P. "Inferring Regular Languages in Polynomial Time" (1992)
- Nevill-Manning, C. & Witten, I. "Identifying Hierarchical Structure in Sequences" (1997)
- Solan, Z. et al. "Unsupervised Learning of Natural Languages" (2005)
- de la Higuera, C. "Grammatical Inference: Learning Automata and Grammars" (2010)
