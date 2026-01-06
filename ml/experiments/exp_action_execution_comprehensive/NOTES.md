# exp_action_execution_comprehensive

**Session**: DDB5
**Model**: Qwen2.5-Coder-1.5B-Instruct-4bit
**Date**: 2025-01-05

## Hypothesis

LLMs can correctly execute statechart actions (entry/exit/transition) with proper Harel semantics and context mutations.

## Task

Test comprehensive action execution across:
1. **Action Placement**: Entry, exit, transition, combined, nested
2. **Context Mutations**: Assignment, increment, conditional, multi-var, nested access
3. **Execution Order**: Harel semantics (exit→trans→entry, bottom-up exit, top-down entry)
4. **Guard-Action Interaction**: Actions affecting subsequent guards

## Results

```
[DDB5]: ACTION_EXEC entry=67%, exit=50%, trans=50%, order=50%, context_mut=75%
```

### Overall Metrics

| Metric | Value |
|--------|-------|
| **State** | **88%** |
| **Context** | **65%** |
| **Order** | **76%** |
| **All Correct** | **47%** |

### By Category

| Category | State | Context | Order |
|----------|-------|---------|-------|
| entry | 100% | 67% | 100% |
| exit | 100% | 50% | 100% |
| transition | 100% | 50% | 100% |
| order | 100% | 75% | 50% |
| context_mutation | 75% | 75% | 100% |
| combined | 50% | 50% | 0% |

## Test Cases

| Test | State | Context | Order | Issue |
|------|-------|---------|-------|-------|
| entry_simple | ✅ | ✅ | ✅ | - |
| entry_multiple | ✅ | ❌ | ✅ | Missed second assignment |
| entry_increment | ✅ | ✅ | ✅ | - |
| exit_simple | ✅ | ✅ | ✅ | - |
| exit_save_state | ✅ | ❌ | ✅ | String copy failed |
| trans_simple | ✅ | ✅ | ✅ | - |
| trans_set_target | ✅ | ❌ | ✅ | Large number handling |
| order_exit_entry | ✅ | ✅ | ✅ | - |
| order_exit_trans_entry | ✅ | ✅ | ❌ | exit,entry,trans instead of exit,trans,entry |
| order_nested_exit | ✅ | ✅ | ✅ | - |
| order_nested_entry | ✅ | ❌ | ❌ | Didn't track Parent.entry |
| context_conditional | ✅ | ✅ | ✅ | - |
| context_chain | ✅ | ✅ | ✅ | - |
| context_nested_access | ✅ | ❌ | ✅ | Nested object access failed |
| context_multi_var | ❌ | ✅ | ✅ | State extraction issue |
| combined_all | ❌ | ❌ | ❌ | Multi-event processing failed |
| self_transition | ✅ | ✅ | ❌ | Order tracking across iterations |

## Key Findings

1. **State prediction excellent** (88%) - Model correctly identifies target states
2. **Simple context mutations work** - Basic assignments and increments succeed
3. **Order tracking mostly works** - Subsequence matching helps (76%)
4. **Complex patterns struggle**:
   - Multiple assignments in single action
   - Nested object access
   - Multi-event sequences
   - Top-down entry order for nested states

## Error Analysis

### Pattern 1: Multiple Assignments
```
Entry action: ["power = true", "brightness = 100"]
Expected: {power: true, brightness: 100}
Got: {power: true}  # Missed second
```

### Pattern 2: Nested Object Access
```
Entry action: ["player.score = player.score + 100"]
Initial: {player: {name: "Alice", score: 500}}
Expected: {player: {name: "Alice", score: 600}}
Got: {score: 600}  # Lost nesting
```

### Pattern 3: Multi-Event Processing
```
Events: [START, FINISH]
Expected: Both events processed
Got: Only first event processed
```

## Prompt Approach

Uses scratchpad reasoning with explicit Harel semantics:

```
HAREL SEMANTICS:
- On transition from A to B:
  1. Execute A's exit actions (bottom-up for nested states)
  2. Execute transition action (if any)
  3. Execute B's entry actions (top-down for nested states)
```

## Files

- `__init__.py` - Module exports
- `test_cases.py` - 17 comprehensive test cases
- `action_executor.py` - LLM-based action execution
- `benchmark.py` - Benchmark runner
- `results/` - Benchmark results

## Reproduction

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_action_execution_comprehensive.benchmark

# Quick mode (6 tests):
.venv/bin/python3 -m experiments.exp_action_execution_comprehensive.benchmark --quick
```

## Future Improvements

1. **Multi-assignment parsing**: Split action list before execution
2. **Nested context tracking**: Explicit JSON path handling
3. **Multi-event loop**: Process events iteratively with state updates
4. **Order verification**: Add explicit order tracking in scratchpad

## Conclusion

Action execution shows strong state prediction (88%) and reasonable context handling (65%). The model understands Harel semantics conceptually but struggles with:
- Complex context mutations (nested access, multiple vars)
- Multi-event sequences
- Top-down entry order in hierarchies

For production use, recommend:
- Single assignments per action
- Flat context structures
- Single-event transitions
