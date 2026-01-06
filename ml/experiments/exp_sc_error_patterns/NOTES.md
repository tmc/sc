# exp_sc_error_patterns - Research Notes

## Core Goal

**Analyze and detect common statechart generation errors.**

Categories:
1. Missing states - referenced but not defined
2. Dangling transitions - invalid source/target
3. Invalid guards - syntax errors, undefined variables
4. Hierarchy violations - nesting errors, parallel conflicts

## Error Taxonomy

### By Category

| Category | Severity | Frequency | Detectability |
|----------|----------|-----------|---------------|
| Missing State | ERROR | 25% | High |
| Dangling Transition | ERROR | 15% | High |
| Invalid Guard | ERROR/WARN | 20% | Medium |
| Hierarchy Violation | ERROR/WARN | 15% | High |
| Unreachable State | WARNING | 10% | Medium |
| Non-determinism | WARNING | 10% | Medium |
| Duplicate Element | ERROR | 5% | High |

### By Source

| Source | Common Errors |
|--------|---------------|
| LLM Generation | Missing initial, invalid guards, incomplete transitions |
| Manual Authoring | Typos, dangling refs after refactor |
| Format Conversion | Missing attributes, type mismatches |
| Refactoring | Orphaned transitions, broken references |

## Detection Strategies

### Structural Validation

1. **State Existence Check**
   - Build set of defined states
   - Check all transition refs against set
   - O(n) where n = states + transitions

2. **Hierarchy Validation**
   - Recursive tree traversal
   - Check type constraints (basic no children, composite has children)
   - Verify initial state per region

3. **Guard Syntax Check**
   - Pattern matching for common errors
   - Balanced parentheses
   - Valid operators

### Semantic Validation

1. **Reachability Analysis**
   - BFS from initial states
   - Mark reachable states
   - Report unreachable

2. **Non-determinism Detection**
   - Group transitions by (source, event)
   - Check guard mutual exclusivity
   - Heuristic: assume guards exclusive if all present

## Common Error Patterns

### LLM-Generated Errors

1. **State Name Hallucination**
   - LLM invents state names not in original
   - Detection: unknown state in transition

2. **Incomplete Transitions**
   - Missing source or target
   - Detection: empty from/to arrays

3. **Guard Syntax Errors**
   - Using = instead of ==
   - Using >> instead of >
   - Detection: regex patterns

4. **Missing Initial State**
   - Composite without is_initial child
   - Detection: tree traversal check

### Human Authoring Errors

1. **Typos**
   - "Procesing" instead of "Processing"
   - Detection: edit distance < 2

2. **Refactoring Orphans**
   - State deleted, transitions remain
   - Detection: missing state check

3. **Copy-Paste Duplicates**
   - Same state defined twice
   - Detection: count occurrences

## Fix Generation

### Automatic Fixes

| Error Type | Fix Strategy | Confidence |
|------------|--------------|------------|
| Typo | Suggest closest match | HIGH |
| Missing state | Add state OR fix reference | MEDIUM |
| Bad guard syntax | Pattern replacement | HIGH |
| No initial | Set first child | MEDIUM |
| Dangling transition | Remove OR fix reference | LOW |

### Fix Confidence Levels

- **HIGH**: Safe to auto-apply
- **MEDIUM**: Suggest with confirmation
- **LOW**: Show options, require selection

## Benchmark Results (Expected)

### Detection Accuracy

| Error Category | Precision | Recall | F1 |
|----------------|-----------|--------|-----|
| Missing State | 0.95 | 0.98 | 0.96 |
| Dangling Trans | 0.98 | 0.95 | 0.96 |
| Invalid Guard | 0.85 | 0.80 | 0.82 |
| Hierarchy | 0.92 | 0.90 | 0.91 |
| Overall | 0.92 | 0.91 | 0.91 |

### Performance

| Statechart Size | Detection Time |
|-----------------|----------------|
| 10 states | < 1ms |
| 100 states | < 5ms |
| 1000 states | < 50ms |

## Integration Points

### With LLM Generation

```python
# Post-generation validation
sc = llm_generate_statechart(prompt)
errors = detect_errors(sc)

if errors:
    # Auto-fix what we can
    fixes = suggest_fixes(errors, sc)
    sc = apply_fixes(sc, fixes, min_confidence=HIGH)

    # Regenerate if still broken
    if detect_errors(sc).errors:
        sc = llm_regenerate_with_feedback(prompt, errors)
```

### With Evolution

```python
# Validate evolved statecharts
def fitness(sc):
    base_fitness = evaluate(sc)

    # Penalize errors
    errors = detect_errors(sc)
    penalty = len(errors.errors) * 0.1

    return base_fitness - penalty
```

### With Training Data

```python
# Filter training data
def filter_valid(statecharts):
    return [
        sc for sc in statecharts
        if detect_errors(sc).is_valid
    ]
```

## Future Improvements

### 1. ML-Based Detection

Train classifier on error patterns:
- Input: statechart features
- Output: error probability per category

### 2. Context-Aware Fixes

Use LLM to generate contextual fixes:
- Consider domain (UI, network, game)
- Learn from past fixes

### 3. Incremental Validation

Validate as user edits:
- Only re-check affected regions
- Real-time feedback

### 4. Error Explanation

Generate natural language explanations:
- Why is this an error?
- What could go wrong at runtime?

## Related Experiments

| Experiment | Relation |
|------------|----------|
| exp_code_completion | Validate generated code |
| exp_qwen_repair | Use errors for repair prompts |
| exp_coverage_prediction | Error-free coverage |
| exp_formal_verification | Formal error properties |
