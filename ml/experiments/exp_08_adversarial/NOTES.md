# Experiment 08: Statechart Advantage Tests

## Executive Summary

Three adversarial tests demonstrate statechart advantages over transformers:

| Test | Statechart | Transformer | Winner |
|------|------------|-------------|--------|
| **Constraint (Safety)** | 0% illegal | 100% illegal | Statechart |
| **Infinite (10K steps)** | 100% valid | 100% valid | Tie* |
| **Merge (Composition)** | 100% zero-shot | FAILS | Statechart |

*Statechart wins on guarantees - transformer happens to work but isn't guaranteed.

---

## Test 1: Constraint Violation (Safety)

### Hypothesis
Statecharts cannot produce illegal outputs; transformers can be poisoned.

### Setup
- **Statechart**: Traffic light with valid transitions only (Red→Green→Yellow→Red)
- **Poison**: 50% of training data has illegal Red→Yellow transitions
- **Metric**: % of test outputs that are illegal

### Results

| Training Data | Transformer | Statechart |
|---------------|-------------|------------|
| Clean | 0.0% illegal | 0.0% illegal |
| **Poisoned (50%)** | **100% illegal** | **0.0% illegal** |

### Conclusion
✓ **Statechart is SAFE BY CONSTRUCTION** - topology prevents illegal outputs
✗ **Transformer is VULNERABLE** - learns any pattern in training data

---

## Test 2: Infinite Steps (Robustness)

### Hypothesis
Statecharts maintain perfect validity forever; transformers may drift.

### Setup
- **Task**: 3-state cycle (A→B→C→A), 10,000 consecutive steps
- **Metrics**: % valid states, % correct predictions

### Results

| Metric | Statechart | Transformer |
|--------|------------|-------------|
| Valid states (10K) | 100.0% | 100.0% |
| Correct predictions | 100.0% | 100.0% |
| Soft prediction drift | N/A | 0.0 entropy |
| Execution time | 0.00s | 6.14s |

### Analysis
Both achieved 100% on this simple cycle. However:
- **Statechart**: Guaranteed correct by topology (O(1) per step)
- **Transformer**: Learned correctly but not guaranteed (O(n²) per step)

For complex systems, statecharts provide **formal verification** that transformers cannot.

---

## Test 3: Zero-Shot Composition

### Hypothesis
Statecharts compose via parallel regions; transformers need retraining.

### Setup
- **Components**: Door (Closed↔Open), Light (Off↔On)
- **Composed**: Parallel(Door, Light) = 4 combined states
- **Test**: Zero-shot prediction on combined state space

### Results

| Scenario | Statechart | Transformer |
|----------|------------|-------------|
| Manual decomposition | 100% | 100% |
| **True zero-shot** | **100%** | **FAILS** |

When transformer sees unseen combined state indices (2, 3):
```
Open+Off + OPEN → pred=1 (WRONG - predicts 0-1 space, not 0-3)
```

### Conclusion
✓ **Statechart topology ENCODES composition** - no retraining needed
✗ **Transformer vocabulary is fixed** - cannot generalize to unseen state indices

---

## Publication-Ready Summary Table

| Property | Statechart | Transformer |
|----------|------------|-------------|
| **Safety** | Guaranteed (by topology) | Vulnerable (learns any data) |
| **Robustness** | Guaranteed (deterministic) | Empirical (may drift) |
| **Composition** | Zero-shot (parallel regions) | Requires retraining |
| **Interpretability** | High (states = meaning) | Low (embeddings) |
| **Verification** | Formal (state invariants) | None |

---

## When to Use Each

### Use Statecharts When:
- Safety-critical (medical, automotive, aerospace)
- Modular/compositional systems
- Formal verification required
- States have semantic meaning
- Long-running processes

### Use Transformers When:
- Raw performance is priority
- Training data is clean and complete
- Single, fixed state space
- No composition needed
- Interpretability not required

---

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml

# Constraint test (safety)
.venv/bin/python3 experiments/exp_08_adversarial/constraint_test.py

# Infinite test (robustness)
.venv/bin/python3 experiments/exp_08_adversarial/infinite_test.py

# Merge test (composition)
.venv/bin/python3 experiments/exp_08_adversarial/merge_test.py
```
