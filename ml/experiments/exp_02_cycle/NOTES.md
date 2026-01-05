# Experiment 02: 3-State Cycle Learning

## Goal
Train a model to predict the 3-state cycle: A → B → C → A

## Results

**PASS** - Converged in 2 epochs!

| Metric | Value |
|--------|-------|
| Initial accuracy | 66.67% |
| Final accuracy | 100% |
| Final loss | 1.0045 |
| Epochs to converge | 2 |
| Baseline | 33% |

### Predictions
```
A → pred=B, expected=B ✓
B → pred=C, expected=C ✓
C → pred=A, expected=A ✓
```

## Key Observations

1. **Trivial for the model** - The cycle pattern is deterministic and simple
2. **Fast convergence** - Only 2 epochs needed (vs 100 for toggle)
3. **Initial accuracy 66.67%** - Model randomly got 2/3 correct initially

## Architecture

```python
CyclePredictor:
  - encoder: Linear(3→32) + ReLU + Linear(32→32) + ReLU
  - predictor: Linear(32→3) + softmax
```

## Why This Was Easy

- Each state has exactly one outgoing transition
- No conflicting events (all use implicit NEXT event)
- State encoding is unambiguous (one-hot)
- Pattern is strictly deterministic

## Comparison to Level 1 (Toggle)

| Aspect | Toggle | Cycle |
|--------|--------|-------|
| States | 2 | 3 |
| Transitions | 2 (bidirectional) | 3 (unidirectional) |
| Events | 1 (TOGGLE) | 1 (implicit NEXT) |
| Epochs to converge | ~100 | 2 |
| Complexity | Medium | Low |

Toggle required more epochs because the same event (TOGGLE) leads to different states depending on current state. The cycle has unambiguous transitions.

## Commands to Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_02_cycle/train.py
```

## Next Steps

- Level 3: More complex patterns (guards, hierarchies)
- Test with noisy data
- Try reverse cycle (A→C→B→A)
