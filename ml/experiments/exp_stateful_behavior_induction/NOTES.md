# exp_stateful_behavior_induction

## Hypothesis

Can an LLM (Qwen2.5-Coder-1.5B) induce stateful machine rules from I/O sequences where order matters?

Core insight: Output depends on HISTORY, not just current input.

## Results

| Metric | Result |
|--------|--------|
| Rule induction | **88%** |
| Predict (seen) | 25% |
| Predict (novel) | 10% |

**By Machine:**
| Machine | Rule | Seen | Novel | Overall |
|---------|------|------|-------|---------|
| Counter | 100% | 25% | 50% | **58%** |
| Queue | 100% | 50% | 0% | 50% |
| Stack | 62% | 50% | 0% | 38% |
| Accumulator | 100% | 0% | 0% | 33% |
| ToggleMemory | 75% | 0% | 0% | 25% |

## Key Findings

**What Worked:**
1. Rule induction is strong (88%) - LLM correctly identifies:
   - State type (list vs int)
   - Event names
   - Rule descriptions (e.g., "append to list", "decrement by 1")
2. Counter machine works best (58%) - simple int state, clear rules
3. Queue/Stack rules induced correctly but predictions fail

**What Didn't Work:**
1. **Prediction accuracy is poor** (25% seen, 10% novel)
   - Model induces correct rules but can't APPLY them
   - List operations particularly error-prone
   - Output parsing struggles with nested structures
2. Accumulator fails completely - multiplication rules confuse model
3. ToggleMemory (tuple state) too complex for current approach

## Analysis

The gap between rule induction (88%) and prediction (25%) reveals:
- **Understanding ≠ Execution**: Model can describe rules but struggles to execute them
- **List manipulation is hard**: Appending/removing from lists generates errors
- **Simple state types work better**: Counter (int) >> Stack/Queue (list) >> ToggleMemory (tuple)

## Recommendations

1. **Code execution**: Generate Python code and execute instead of LLM prediction
2. **Simpler state representations**: Use string-based states instead of lists
3. **Chain-of-thought for prediction**: Force step-by-step arithmetic
4. **Larger model**: 7B+ may have better state tracking

## Report to Orchestrator

```
[9D1B]: STATEFUL_INDUCTION rule_acc=88%, pred_seen=25%, pred_novel=10%, by_machine=[stack:38%, queue:50%, counter:58%, accum:33%]

Key insight: Rule induction strong (88%) but prediction weak (25%)
- Model understands rules but can't execute them
- Counter (int state) works best: 58%
- List-based machines fail at prediction
- Recommendation: Generate code, don't predict directly
```

## Files

- `__init__.py` - Machine implementations, trace generation
- `behavior_inducer.py` - LLM-based rule induction
- `state_predictor.py` - Apply rules for prediction
- `benchmark.py` - Full evaluation

## Usage

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_stateful_behavior_induction.benchmark
```
