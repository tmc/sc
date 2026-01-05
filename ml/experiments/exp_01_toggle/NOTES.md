# Experiment 01: 2-State Toggle Learning

## Objective
THE BLOCKING QUESTION: Can we learn ANYTHING?

Learn to predict next state given (current_state, event) for a simple 2-state toggle.

## Task
- States: Off (0), On (1)
- Events: TOGGLE (0)
- Dynamics: Off + TOGGLE -> On, On + TOGGLE -> Off
- Pass criterion: >99% accuracy after 1000 steps
- Baseline: random = 50%

## Model
ToggleLearner: Simple embedding + MLP
- State embedding: 2 states -> 8 dims
- Event embedding: 1 event -> 8 dims
- Hidden layer: 16 -> 8 (tanh)
- Output: 8 -> 2 (logits)
- Total params: ~180

## Data
- 4 samples from Go semantics traces (testdata/traces/toggle_trace.json)
- 1000 synthetic samples (deterministic toggle dynamics)
- Total: 1004 samples

## Results

| Metric | Value |
|--------|-------|
| Initial accuracy | 0.4990 (expected ~0.50 random) |
| Final accuracy | 1.0000 |
| Steps to converge | ~100 |
| Pass criterion | >0.99 |
| Status | **PASS** |

### Training Log
```
Step    1: loss=0.7871, acc=0.4990
Step  100: loss=0.0000, acc=1.0000
Step  200: loss=0.0000, acc=1.0000
...
Step 1000: loss=0.0000, acc=1.0000
```

### Verification
```
Off + TOGGLE -> On: pred=1, expected=1 [OK]
  P(Off)=0.0000, P(On)=1.0000
On + TOGGLE -> Off: pred=0, expected=0 [OK]
  P(Off)=1.0000, P(On)=0.0000
```

## Reproduce
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_01_toggle/train.py
```

## Notes
- Learning is extremely fast (100 steps to perfect accuracy)
- This is expected: the task is deterministic and simple
- Model perfectly learns the toggle dynamics
- Loss goes to 0.0000 (numerical precision limit)

## Conclusion
**YES, we can learn the simplest statechart dynamics.**

Ready for Level 2: more states, more events.
