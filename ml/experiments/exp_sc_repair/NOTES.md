# exp_sc_repair: Statechart Repair Experiment

## Goal

Automatically repair invalid statecharts using LLM-guided strategies.

## Error Categories Handled

1. **Duplicate Labels** - States with same name at same level
   - Strategy: Rename with suffix (A_1, A_2) or merge duplicates

2. **Unreachable States** - States with no path from initial
   - Strategy: Add transition from initial, remove state, or make initial

3. **Missing Initial** - Composite state without initial child
   - Strategy: Set first child as initial, or clear multiple initials

4. **Invalid Hierarchy** - Empty composites, malformed nesting
   - Strategy: Add placeholder child, flatten, or convert to basic

5. **Dangling Transitions** - References to non-existent states
   - Strategy: Remove transition or create missing state

## Architecture

```
repair_engine.py      - Main LLM repair interface (SCRepairEngine)
repair_strategies.py  - Per-error type strategies (rule-based)
repair_validator.py   - Validates repairs don't introduce new errors
benchmark.py          - Accuracy evaluation with 10 test cases
```

## LLM Integration

Uses Qwen2.5-Coder-0.5B-Instruct via mlx_lm for:
- Intelligent repair selection when multiple strategies apply
- Handling edge cases not covered by rule-based strategies
- Generating meaningful names for renamed states

## Benchmark Cases

| Category | Cases | Description |
|----------|-------|-------------|
| duplicate_element | 2 | Simple and nested duplicates |
| unreachable_state | 2 | Orphan states and islands |
| hierarchy_violation | 4 | Missing initial, empty composite, multiple initial |
| dangling_reference | 2 | Bad source and target references |

## Evaluation Metrics

- **Success**: All errors fixed, no new errors, semantics preserved
- **Partial**: Some errors fixed, no new errors introduced
- **Failed**: New errors introduced or repair failed

## Key Insights

1. **Rule-based first**: Many repairs can be done with simple rules
2. **LLM for selection**: Use LLM to choose between valid strategies
3. **Validation critical**: Always validate repairs before returning
4. **Semantic preservation**: Don't lose states/transitions unnecessarily

## Dependencies

- mlx_lm (REAL inference required)
- Qwen2.5-Coder-0.5B-Instruct model
- exp_sc_error_patterns.error_detector for validation
