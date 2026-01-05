# exp_execution_replay: Research Notes

## Key Insight

**Offline learning from traces is more sample-efficient than online exploration.**

Why:
- Traces provide complete (source, event, target) tuples
- No exploration overhead (random actions, invalid transitions)
- Can replay same trace multiple times
- Parallel processing possible

## Execution Trace Format

From `proto/statecharts/v1/execution.proto`:

```
ExecutionTrace τ = (id, σ₀, Γ₀, L*, σₙ, Γₙ)
  - σ₀, Γ₀: initial configuration and context
  - L* = L₁ · L₂ · ... · Lₙ: sequence of log entries
  - σₙ, Γₙ: final configuration and context

TransitionLogEntry L = (t, seq, e, σ, σ', T, G, A, Γ, Γ')
  - t: timestamp
  - seq: sequence number
  - e: triggering event
  - σ, σ': source and target configurations
  - T: transitions that fired
  - G: guard evaluation results
  - A: actions executed
  - Γ, Γ': context before and after
```

## Approaches

### 1. Direct Extraction (Baseline)

Maximum likelihood estimation:
```
P(target | source, event) = count(source, event, target) / count(source, event)
```

Pros:
- Simple, fast
- Exact recovery with enough data
- No hyperparameters

Cons:
- Needs explicit state labels
- No generalization

### 2. State Inference

When traces have observations but no state labels:
```
1. Extract context features
2. K-means clustering → infer states
3. Count transitions between clusters
```

Pros:
- Works without explicit states
- Discovers hidden structure

Cons:
- Cluster count is hyperparameter
- May not match true states

### 3. Neural Transition Model

```
Input: (state_embedding, event_embedding, context)
Output: P(next_state)
```

Pros:
- Learns from context
- Generalizes to unseen combinations

Cons:
- Needs more data
- Slower training

### 4. Hybrid Learning

Mix offline traces with online exploration:
```
1. Load offline traces into replay buffer
2. Online: observe, store in buffer
3. Learn from mixed samples (prioritized)
```

Pros:
- Best of both worlds
- Adapts to distribution shift

Cons:
- More complex
- Needs environment access

## Sample Efficiency Comparison

```
Online Learning:
- Must explore (random actions)
- Many invalid transitions
- Sequential, slow

Offline Learning:
- All transitions valid
- Parallel processing
- Can filter/augment
```

Expected: Offline reaches 90% accuracy with 2-3x fewer samples.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION TRACES                         │
│  trace₁ = [L₁, L₂, ..., Lₙ]                                │
│  trace₂ = [L₁, L₂, ..., Lₘ]                                │
│  ...                                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRACE PARSER                             │
│  - Parse proto/JSON                                         │
│  - Extract (source, event, target) tuples                   │
│  - Build vocabularies                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │  Direct  │    │  State   │    │  Neural  │
     │Extraction│    │Inference │    │  Model   │
     └────┬─────┘    └────┬─────┘    └────┬─────┘
          │               │               │
          ▼               ▼               ▼
     ┌─────────────────────────────────────────┐
     │           LEARNED STATECHART            │
     │  - States with visit counts             │
     │  - Transitions with probabilities       │
     │  - Guards (optional)                    │
     └─────────────────────────────────────────┘
```

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_rule_discovery` | Uses TraceCollector, similar transition counting |
| `exp_go_9x9` | GameTrace format similar to ExecutionTrace |
| `exp_transition_priorities` | Learned priorities from trace frequencies |
| `exp_guard_synthesis` | Guard learning from trace context |
| `exp_soar_statechart` | Hindsight relabeling uses trace replay |

## What Worked

1. **Direct extraction** - Simple and effective when states are labeled

2. **Replay buffer** - Mixing online/offline data improves both

3. **Context features** - Even simple numeric features help neural model

4. **Priority sampling** - Recent samples weighted higher in hybrid

## What Didn't Work

1. **Pure neural without structure** - Needs explicit state/event embeddings

2. **Too few clusters** - State inference needs k ≥ true states

3. **Ignoring sequence** - RNN helps but adds complexity

## Guard Learning

Future direction: Learn guard conditions from context:

```python
# For transition (A → B on event e):
# Collect contexts where A → B fired
# vs. contexts where A stayed or → C

# Train binary classifier:
guard_AB(context) → P(fire A→B)
```

## Performance

With 5 states, 4 events, 8 transitions:

| Method | 50 traces | 100 traces | Time |
|--------|-----------|------------|------|
| DirectExtraction | 75% | 100% | 0.01s |
| StateInference | 50% | 65% | 0.05s |
| NeuralModel | 60% | 85% | 0.3s |
| Hybrid | 80% | 95% | 0.1s |

## Pure MLX

No protobuf, no numpy dependencies:
- Python dataclasses mirror proto messages
- MLX arrays for neural network
- JSON parsing for trace loading

## Future Directions

1. **Hierarchical traces** - Learn nested statecharts from traces with parent refs

2. **Temporal patterns** - Learn timing from timestamps (dwell time, rate limits)

3. **Causal inference** - Use vector clocks for distributed trace ordering

4. **Online fine-tuning** - Start with offline, fine-tune with online

5. **Trace compression** - Learn compact representations for long traces

## Code Patterns

```python
# Parse trace from dict
parser = TraceParser()
trace = parser.parse_dict({
    "trace_id": "t1",
    "initial_config": {"active_states": ["IDLE"]},
    "entries": [
        {
            "source_config": {"active_states": ["IDLE"]},
            "target_config": {"active_states": ["RUNNING"]},
            "trigger_event": {"event_type": "start"},
        }
    ]
})

# Learn from traces
learner = DirectExtractionLearner()
learner.add_trace(trace)
chart = learner.learn()

# Get transition probability
probs = chart.get_transition_probs("IDLE", "start")
# → {"RUNNING": 1.0}
```

## Date
2024-01-04
