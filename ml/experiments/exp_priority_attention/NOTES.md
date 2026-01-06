# exp_priority_attention: Research Notes

## Core Hypothesis

**Transformer attention can learn context-dependent transition priorities better than hand-coded specificity heuristics.**

When multiple transitions are enabled in a statechart, the "correct" choice depends on the current context. The specificity heuristic from exp_transition_priorities works well on average (0.56 fitness), but misses context-dependent patterns that a learned model could capture.

## Architecture

```
                    ┌─────────────────┐
                    │   Context       │
                    │ (grid, state)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ ContextEncoder  │
                    │ (CNN + MLP)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
     Transitions ──►│ PriorityAttention│──► Selection
     (K, V)         │  (Cross-attn)   │    Probabilities
                    └─────────────────┘
```

### TransitionEncoder

Embeds each transition into a dense vector:
- Predicate type embedding (10 types → 32 dims)
- Source state embedding (20 states → 16 dims)
- Target state embedding (20 states → 16 dims)
- Guard parameters (8 dims for threshold, color, value, etc.)
- Final projection to 64 dims

### ContextEncoder

Embeds the execution context:
- Grid: 2-layer CNN (1→16→32 channels) + global avg pool
- State variables: MLP projection to 32 dims
- Combine: Concatenate and project to 64 dims

### PriorityAttention

Multi-head cross-attention:
- Query: Context embedding (what am I looking for?)
- Key/Value: Transition embeddings (what's available?)
- Output: Attention weights over transitions = selection probabilities

Key design choice: **Context attends to transitions**, not vice versa. This mirrors the decision process: "given this context, which transition should I choose?"

## Training

### Evolution-Based

Evolution works better than gradient descent here because:
1. The objective (sequence matching) is non-differentiable
2. Small population sizes work fine (15-20)
3. No hyperparameter tuning for learning rate, momentum, etc.

```python
evolver = AttentionEvolver(
    population_size=15,
    mutation_rate=0.2,
    mutation_std=0.15,
)
best = evolver.evolve(test_cases, n_generations=30)
```

### Gradient-Based (Future)

Could use gradient descent if we:
1. Use soft selection (Gumbel-softmax) instead of hard argmax
2. Define differentiable sequence loss
3. Use MLX optimizers (Adam, SGD)

## Experiments

### Baseline Comparison

| Strategy | Mean Fitness | Notes |
|----------|--------------|-------|
| Specificity (heuristic) | 0.564 | From exp_transition_priorities |
| Explicit priority | 0.363 | Numeric priorities without context |
| **Attention (learned)** | **TBD** | This experiment |

### What to Measure

1. **Fitness vs specificity baseline**: Does attention beat 0.56?
2. **Attention interpretability**: Do attention weights correlate with guard specificity?
3. **Context sensitivity**: Does the model use context features?
4. **Generalization**: Does it work on unseen state/transition combinations?

## Key Insights

### 1. Attention Encodes Soft Specificity

The attention mechanism implicitly learns a "soft" version of specificity. Instead of hard-coded specificity scores per predicate type, it learns context-dependent importance weights.

### 2. Cross-Attention > Self-Attention

Cross-attention (context → transitions) works better than self-attention over transitions alone. The context provides the "query" that selects the right transition.

### 3. Temperature Matters

The softmax temperature controls exploration vs exploitation:
- Low temp (0.1): Nearly deterministic, picks highest score
- High temp (1.0): More uniform, explores alternatives

We make temperature learnable so the model can tune this tradeoff.

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_transition_priorities` | Direct evolution, this uses attention |
| `exp_differentiable_statecharts` | Both enable gradient-based learning |
| `exp_learnable_policies` | Policy attention could use this architecture |
| `exp_coverage_prediction` | Attention weights could predict coverage |
| `exp_guard_synthesis` | Synthesized guards need learned priorities |

## Implementation Details

### Pure MLX

All operations use MLX tensors. No numpy conversion except for final output.

### Batched Inference

Model supports batched inputs for efficient evaluation:
```python
scores, attn = model(
    pred_indices,    # [B, N]
    source_indices,  # [B, N]
    target_indices,  # [B, N]
    guard_params,    # [B, N, 8]
    grid,            # [B, H, W]
    state_vars,      # [B, n_vars]
    enabled_mask,    # [B, N]
)
```

### Tracking

Each transition tracks:
- `fire_count`: How often it was selected
- `attention_scores`: History of attention weights when selected

## Future Directions

1. **Multi-step lookahead**: Attention over transition *sequences*, not just single steps
2. **Hierarchical attention**: Separate attention for composite vs atomic states
3. **Meta-learning**: Learn to adapt priorities based on task description
4. **Interpretable attention**: Visualize what context features drive selection
5. **Adversarial training**: Generate hard test cases to improve robustness

## Code Patterns

```python
# Complete priority selection in one line
selected, score = strategy.select_transition(enabled, context)

# Attention weights for analysis
scores, attn = strategy.compute_priorities(transitions, context)
winner_idx = max(range(len(scores)), key=lambda i: scores[i])

# Interpretability: which transitions get high attention?
for i, (t, a) in enumerate(zip(transitions, attn)):
    print(f"{t.source}->{t.target}: attn={a:.3f}, guard={t.guard.predicate}")
```

## Date

2026-01-04
