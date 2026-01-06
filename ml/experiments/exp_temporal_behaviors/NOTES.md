# exp_temporal_behaviors

## Hypothesis

Temporal statechart behaviors can be accurately modeled and predicted
using algorithmic simulation, while LLMs will struggle with precise timing.

## Approach

1. **Temporal Semantics**: Model AFTER events, timeouts, delayed transitions
2. **Simulation**: Step-based simulation with timer management
3. **Prediction**: Predict final state and state sequence for scenarios
4. **Evaluation**: Compare algorithmic vs LLM prediction accuracy

## Temporal Constructs

### AFTER(duration) Events
- Trigger after specified time in state
- Reset when state is exited
- Example: `Idle --after(30s)--> Screensaver`

### Timeout Patterns
- Idle timeout: return to initial after inactivity
- Retry with backoff: wait longer between retries
- Deadline: must complete before time limit

### Time Guards
- Conditions based on elapsed time
- Example: `[elapsed > 5000]` - only fire after 5s in state

## Test Cases

| TC | Pattern | Description |
|----|---------|-------------|
| TC1 | simple_timeout | Basic AFTER transition |
| TC2 | activity_reset | Timer reset on activity |
| TC3 | exponential_backoff | Retry with increasing delays |
| TC4 | deadline_race | Complete vs timeout race |
| TC5 | debounce | Wait for input stabilization |
| TC6 | nested_timeouts | Parent/child timeout interaction |
| TC7 | parallel_timeouts | Independent regional timeouts |
| TC8 | time_guards | Guards based on elapsed time |

## Results

**Algorithmic simulation significantly outperforms LLM!**

| Method | Accuracy | Avg Time | Notes |
|--------|----------|----------|-------|
| Algorithmic | **77%** | 0.2ms | Step-based simulation |
| LLM | 18% | 2191ms | Qwen2.5-Coder-1.5B |

### By Pattern (Algorithmic vs LLM)

| Pattern | Algorithmic | LLM |
|---------|-------------|-----|
| simple_timeout | 50% | 50% |
| activity_reset | **100%** | 33% |
| exponential_backoff | 67% | 0% |
| deadline_race | 67% | 67% |
| debounce | **100%** | 0% |
| nested_timeouts | 50% | 0% |
| parallel_timeouts | **100%** | 0% |
| time_guards | 67% | 0% |

## Key Findings

### Algorithmic Strengths

1. **Timer management works** - 100% on activity_reset, debounce, parallel_timeouts
2. **Event ordering handled** - Most scenarios correctly sequenced
3. **Very fast** - 0.2ms average vs 2191ms for LLM

### Algorithmic Failures (Improvement Needed)

1. **Event-cancels-timeout**: activity_prevents failed - need to block AFTER when event arrives
2. **Guard evaluation timing**: Time guards not continuously evaluated
3. **Tie-breaking**: Event at exact timeout moment should win
4. **Hierarchical states**: Nested timeout interaction incomplete

### LLM Weaknesses (Confirmed)

1. **Complex timing fails** - 0% on backoff, debounce, time_guards
2. **Defaults to simple answer** - Often picks initial or most common state
3. **No step-through reasoning** - Can't simulate multi-step timing

### Observed Challenges

1. **Timer reset semantics**: Self-transitions that reset timers
2. **Backoff calculation**: Variable delays based on attempt count
3. **Guard evaluation**: Time-based conditions
4. **Parallel regions**: Independent timer management
5. **Nested timeouts**: Parent vs child timeout priority

### LLM Predicted Weaknesses

1. **Precise timing arithmetic**: Calculating exactly when timeouts fire
2. **Event/timeout ordering**: Determining which fires first
3. **Timer reset logic**: Tracking when timers restart
4. **Multi-step simulation**: Following complex timing sequences

## Files Created

- `temporal_behavior.py` - Core temporal modeling and simulation
- `test_cases.py` - 8 test cases with multiple scenarios
- `benchmark.py` - Evaluation framework
- `__init__.py` - Module exports
- `NOTES.md` - This file

## Report Format

```
[SID]: TEMPORAL simple_after=X%, reset=Y%, backoff=Z%, deadline=W%, debounce=V%
```
