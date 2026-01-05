# Temporal Guards Experiment

## Overview

This experiment extends the guard DSL with TIME-BASED predicates for statechart transitions.
Instead of hardcoding timeout values, we EVOLVE temporal thresholds from timestamped examples.

## Key Insight

**Temporal guards are LEARNABLE.** Given positive examples (contexts where transition should fire)
and negative examples (where it shouldn't), evolution discovers optimal time thresholds.

## Temporal DSL

### Core Predicates

| Predicate | Semantics | Use Case |
|-----------|-----------|----------|
| `after(duration)` | True if elapsed >= duration | Timeouts |
| `within(duration)` | True if elapsed < duration | Deadlines |
| `elapsed` | Returns time in current state | Comparisons |
| `since(state)` | Time since leaving state | Re-auth |
| `timeout(state, d)` | Named state timeout | Parallel regions |

### Rate Limiting Predicates

| Predicate | Semantics | Use Case |
|-----------|-----------|----------|
| `rate_limit(n, w)` | True if < n events in window w | API throttling |
| `cooldown(duration)` | True if >= duration since last transition | Action spacing |
| `debounce(duration)` | True if stable for duration | UI stability |

### Schedule Predicates

| Predicate | Semantics | Use Case |
|-----------|-----------|----------|
| `time_of_day()` | Returns hour (0-23) | Business hours |
| `day_of_week()` | Returns day (0=Mon, 6=Sun) | Weekly schedules |
| `business_hours(s, e)` | True if hour in [s, e) | Access control |

## Evolution Strategy

### Duration Candidates

Evolution explores common duration values:
```
0.1, 0.25, 0.5,           # Sub-second (debounce)
1.0, 2.0, 3.0, 5.0,       # Short (cooldown)
10.0, 15.0, 30.0,         # Medium (game timers)
60.0, 120.0, 300.0,       # Minutes (session)
600.0, 1800.0, 3600.0,    # Long (idle timeout)
7200.0, 14400.0, 86400.0  # Hours/day (expiration)
```

### Mutation Operators

1. **Duration mutation**: Gradual refinement (0.7x-1.4x) or jump to candidate
2. **Rate limit mutation**: Adjust count and window independently
3. **Operator mutation**: Flip comparison operators (>, <, >=, <=)
4. **Structure mutation**: Replace subtrees with new random expressions

### Fitness Function

- F1 score on positive/negative examples
- Precision: Avoid false positives (don't timeout too early)
- Recall: Catch all true timeouts

## Benchmark Scenarios

### 1. Session Timeout (30s)
Pattern: Session expires after 30s of inactivity
Expected guard: `after(30.0s)` or `elapsed >= 30`

### 2. Game Timer (30s)
Pattern: Player forfeits turn after 30s
Expected guard: `after(30.0s)`

### 3. Rate Limiting (5 req / 60s)
Pattern: Block if too many requests
Expected guard: `rate_limit(5, 60.0s)` (negated for blocking)

### 4. Ability Cooldown (5s)
Pattern: Allow ability after 5s cooldown
Expected guard: `cooldown(5.0s)`

### 5. UI Debounce (0.5s)
Pattern: Process input after 0.5s stability
Expected guard: `debounce(0.5s)` or `after(0.5s)`

### 6. Business Hours (9-17)
Pattern: Allow access during work hours
Expected guard: `business_hours(9, 17)` or `time_of_day() >= 9 and time_of_day() < 17`

## Context Variables

Temporal guards use special context variables:

| Variable | Type | Description |
|----------|------|-------------|
| `__timestamp__` | float | Current time (seconds since epoch) |
| `__state_entry_time__` | float | When current state was entered |
| `__state_exit_times__` | Dict[str, float] | When each state was last exited |
| `__state_entry_times__` | Dict[str, float] | When each state was last entered |
| `__active_states__` | Set[str] | Currently active state names |
| `__event_times__` | List[float] | Timestamps of recent events |
| `__last_transition_time__` | float | When last transition fired |

## Connections to Other Experiments

### exp_guard_synthesis
- Base expression DSL (Expr, BinOp, UnaryOp, etc.)
- Evolution framework (mutation, crossover, fitness)
- Adapted for temporal extensions

### exp_transition_priorities
- Temporal guards contribute to guard specificity
- More specific temporal constraints = higher priority

### exp_unified_statechart_evolution
- Temporal guards can be incorporated into UnifiedGuard
- NSGA-II can optimize temporal thresholds

### exp_action_side_effects
- Actions can update temporal context (reset timers)
- Guards can check action-modified timestamps

## Implementation Notes

### No MLX Dependency
Unlike some experiments, temporal_guard.py uses pure Python.
This makes it portable and easy to test.

### Time Precision
Durations use float seconds. For sub-second precision (debounce),
use values like 0.1, 0.25, 0.5.

### Testing
Run individual tests:
```python
python -m exp_temporal_guards.temporal_guard  # DSL tests
python -m exp_temporal_guards.temporal_evolver  # Evolution tests
python -m exp_temporal_guards.temporal_benchmark  # Full benchmark
```

## Future Directions

1. **Relative time references**: "5 seconds after event X"
2. **Periodic guards**: "Every 10 seconds while in state"
3. **Duration variables**: Learn duration from context (user tier -> timeout)
4. **Temporal action effects**: Actions that modify time context
5. **Neural duration prediction**: Learn duration from features

## References

- `semantics/v1/machine.go`: State entry/exit timing
- `proto/statecharts/v1/statecharts.proto`: Transition guard model
- Harel, "Statecharts: A Visual Formalism" - timeout transitions
- SCXML specification - delay/timeout semantics
