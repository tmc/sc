# exp_internal_vs_external: Research Notes

## Overview

This experiment explores the distinction between **internal** (synchronous) and **external** (queued) events in Harel statechart semantics, and **evolves** optimal priority assignments without hardcoding.

---

## Key Insights

### 1. Internal vs External is a Semantic Distinction

From [HN96] (Harel-Naamad STATEMATE semantics):
- **Internal events** are generated during a step (by actions, entry/exit handlers)
- They are processed **synchronously** within the same RTC (run-to-completion) step
- This enables **micro-step chains**: one transition triggers another in the same macro-step

**Example cascade:**
```
USER_INPUT → COLLISION (internal) → DAMAGE (internal) → GAME_OVER
     ^          micro-step 1          micro-step 2       micro-step 3
     |
macro-step boundary
```

### 2. Evolution Discovers Semantic Categories

Without any hardcoding, evolution found:
- **INTERNAL**: TICK, COLLISION, SCORE_UPDATE
- **EXTERNAL**: USER_INPUT, SPAWN_ENEMY, DAMAGE, etc.

This makes semantic sense:
- COLLISION naturally causes cascading effects (damage, state changes)
- TICK is the frame clock, needs synchronous processing
- USER_INPUT starts new interaction sequences (external trigger)

### 3. Priority Affects Latency-Throughput Tradeoff

The fitness function balances:
- **Latency**: High-priority events processed faster
- **Throughput**: Overall events processed per unit time
- **Correctness**: Causal ordering preserved
- **Parsimony**: Avoid over-prioritizing (too many CRITICAL = no prioritization)

---

## What Worked

1. **Evolutionary approach**: Simple genome (event → kind/priority) evolves well
2. **Multi-objective fitness**: Balancing latency/throughput/correctness gives nuanced results
3. **Trace-based seeding**: Analyzing execution traces provides good initial priorities
4. **PriorityEventQueue**: Clean separation of internal buffer vs external queue

## What Didn't Work (Initially)

1. **Pure random initialization**: Converged slowly without trace-based seeding
2. **Single-objective fitness**: Just optimizing latency led to everything being CRITICAL
3. **Fixed internal/external ratio**: Better to let evolution discover the right balance

---

## Future Directions

### 1. Dynamic Priority Adjustment
- Priority could change based on context (e.g., DAMAGE is CRITICAL when health < 10%)
- Evolve **guard conditions** for priority, not just static assignments

### 2. Event Coalescing
- Multiple TICK events in queue → coalesce to single TICK
- Evolve coalescing rules per event type

### 3. Priority Inversion Detection
- High-priority event blocked by low-priority processing
- Learn to detect and prevent priority inversion patterns

### 4. Real-Time Constraints
- Add deadline constraints: "COLLISION must be processed within 16ms"
- Fitness includes deadline miss rate

### 5. Distributed Event Ordering
- Extend to multi-machine scenarios (Lamport clocks from execution.proto)
- Evolve consensus protocols for event ordering

---

## Connections to Other Experiments

### exp_execution_replay
- Shares trace loading infrastructure
- Both use `TraceEntry` with trigger_event
- This experiment adds priority learning on top

### exp_topology_evolution
- Uses same evolutionary patterns: genome, fitness, mutation, crossover
- This is "event-level evolution" vs "state-level evolution"
- Could combine: evolve topology AND event priorities together

### exp_guard_synthesis
- Guards control transition enablement
- Priority controls processing order
- Combined: `guard → enabled` AND `priority → when`

### exp_temporal_guards
- Temporal guards (timeouts, rate limits) relate to priority
- High-priority events might bypass rate limits
- Cooldowns are a form of priority demotion

### exp_learnable_policies
- SAE discovers states from activations
- This experiment discovers event semantics from traces
- Combined: learn BOTH states AND event handling from data

---

## Implementation Notes

### EventKind Enum
```python
class EventKind(IntEnum):
    EXTERNAL = 0   # Queued
    INTERNAL = 1   # Synchronous
```

### Priority Queue Design
- Internal events go to separate buffer (always processed first)
- External events sorted by priority (CRITICAL > HIGH > NORMAL > LOW)
- FIFO within same priority (sequence number tiebreaker)

### Fitness Components
```python
@dataclass
class FitnessComponents:
    latency_score: float      # Lower latency for high-priority
    throughput_score: float   # Overall event throughput
    correctness_score: float  # Ordering correctness
    parsimony_score: float    # Penalty for over-prioritization
```

### Trace-Based Learning
Heuristic: Events with high `caused_internal` ratio should be INTERNAL
```python
internal_ratio = stats["caused_internal"] / stats["count"]
if internal_ratio > 0.3:
    kind = EventKind.INTERNAL
```

---

## References

- [HN96] Harel & Naamad, "The STATEMATE semantics of statecharts", ACM TOSEM 1996
- [vdB94] von der Beeck, "A comparison of statecharts variants", FTRTFT 1994
- execution.proto: TransitionLogEntry, CausalityInfo
- exp_topology_evolution/framework.py: EvolutionConfig, TopologyEvolver patterns

---

## Metrics Summary

| Run | Events | Generations | Best Fitness | Internal | External |
|-----|--------|-------------|--------------|----------|----------|
| Demo | 10 | 50 | 3.16 | 3 | 7 |

Evolution time: ~1.5s for 50 generations with 30 population.
