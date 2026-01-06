# exp_sae_coverage_synthesis - Research Notes

## Core Insight

**SAE features ARE states. No clustering needed.**

The traditional approach:
1. Encode with SAE
2. Cluster activations into states
3. Build statechart from clusters

Our approach:
1. Encode with SAE
2. Active features = state configuration
3. Transitions = changes in active feature sets
4. Guards = patterns over feature activations

This is simpler and more interpretable.

## Architecture

```
Hidden States (seq_len, dim)
        ↓
   TopK SAE Encoder
        ↓
Feature Activations (seq_len, k active features)
        ↓
State Configurations (sequence of frozen sets)
        ↓
Transition Extraction (observe src→tgt pairs)
        ↓
Guard Synthesis (evolve guards for each transition)
        ↓
Coverage Simulation (simulate statechart)
        ↓
Predicted Coverage (which features/states visited)
```

## What Worked

### 1. Features as States (No Clustering)
- Simpler than clustering approaches
- Each feature has clear semantics
- Configuration changes are explicit transitions
- Much easier to debug and interpret

### 2. Guard Synthesis from Patterns
- Guards are boolean expressions over features
- Example: `F5_active AND NOT F10_active`
- Evolved from positive/negative transition examples
- Achieves high F1 on transition prediction

### 3. Simulation for Coverage
- Treat feature configs as statechart states
- Simulate execution from initial state
- Visited features = predicted coverage
- Works well for structured sequences

## What Didn't Work

### 1. Dense Activation Thresholds
- Initially tried continuous activation levels
- Too noisy for guard synthesis
- TopK with binary active/inactive works better

### 2. All-Pairs Guard Synthesis
- Synthesizing guards for every possible transition
- Too expensive (O(n²) transitions)
- Better to focus on observed transitions

### 3. Deep Guard Trees
- max_depth > 3 led to overfitting
- Simpler guards generalize better
- AND/OR of 2-3 feature checks is optimal

## Connections to Other Experiments

### exp_sae_statechart
- Source of TopK SAE implementation
- We extend: SAE features ARE states directly
- They cluster; we don't

### exp_coverage_prediction
- Source of simulation approach
- We extend: use SAE features as CFG states
- They use explicit CFG; we discover structure

### exp_guard_synthesis
- Source of guard evolution
- We extend: guards over features (not game state)
- Same evolutionary approach

### exp_action_side_effects
- Actions modify feature activations
- Guards check feature activations
- Could combine: learn what actions do to features

## Key Metrics

| Metric | Baseline | This Approach |
|--------|----------|---------------|
| State Discovery | Clustering-based | Direct from TopK |
| Guard Complexity | Hand-crafted | Evolved |
| Coverage Prediction | CFG-based | Feature-based |
| Interpretability | Low (clusters) | High (features) |

## Future Directions

### 1. Real Model Integration
- Apply to actual LLM hidden states
- Map features to code semantics
- Validate coverage predictions

### 2. Hierarchical Features
- Some features form superstates
- Co-occurrence analysis identifies hierarchy
- Build multi-level statechart

### 3. Online Learning
- Update guards as new transitions observed
- Adapt to distribution shift
- Continuous refinement

### 4. Code-Specific Features
- Features that correspond to syntax constructs
- Features that correspond to control flow
- Features that correspond to data flow

## Implementation Notes

### TopK SAE
```python
# Always exactly k features active
sorted_indices = mx.argsort(-pre_acts, axis=-1)
top_indices = sorted_indices[:, :k]
```

### State Configuration
```python
# Frozenset of active feature indices
config = StateConfiguration(
    active_features=frozenset(active_set),
    activations=activation_dict,
)
```

### Guard Synthesis
```python
# Evolve guards from examples
guard = synthesizer.synthesize(
    positive=transitions_that_fire,
    negative=transitions_that_dont,
)
```

### Coverage Simulation
```python
# BFS over statechart
prediction = simulator.predict_coverage(
    initial_pattern,
    use_all_paths=True,
)
```

## Open Questions

1. How many features are needed for good coverage prediction?
2. Do features have stable semantics across inputs?
3. Can we transfer learned guards to new programs?
4. How does k (active features) affect accuracy?

## References

- Anthropic SAE work: Features as atomic concepts
- exp_sae_statechart: SAE for state discovery
- exp_coverage_prediction: Coverage via simulation
- exp_guard_synthesis: Evolutionary guard learning
