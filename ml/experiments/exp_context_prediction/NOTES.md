# exp_context_prediction

**Session**: DDB5
**Model**: Qwen2.5-Coder-1.5B-Instruct-4bit
**Date**: 2025-01-05

## Hypothesis

Can an LLM predict final context values after statechart execution using few-shot prompting?

Given:
- Statechart definition (states, transitions, guards, actions)
- Initial context values
- Event sequence

Predict: Final context after execution

## Results

**Report to B90CCCD4:**
```
[DDB5]: CONTEXT_PREDICTION exact=75%, partial=92%, mae=0.08, by_level=[L1:67%, L2:67%, L3:67%, L4:100%]
```

### By Complexity Level

| Level | Description | Exact% | Partial% | MAE |
|-------|-------------|--------|----------|-----|
| L1 | Single increment (count++) | 67% | 83% | 0.33 |
| L2 | Arithmetic (score + bonus) | 67% | 92% | 0.00 |
| L3 | Conditionals (if score >= threshold) | 67% | 92% | 0.00 |
| L4 | Accumulation (sum over loop) | 100% | 100% | 0.00 |

### Overall
- **Exact Match**: 75% (9/12 test cases)
- **Partial Match**: 92% (variables correctly predicted)
- **MAE**: 0.08 (numeric prediction error)

## Key Findings

### What Worked
1. **Accumulation (L4)**: 100% accuracy on summing arrays
   - Model correctly tracks `sum = sum + values[index]` over multiple iterations
   - Array indexing understood correctly

2. **Arithmetic propagation**: When guards evaluate True, model correctly computes downstream effects
   - `score=75, threshold=50` → `result='pass', bonus=10, final_score=85` ✓

3. **Few-shot format**: Compact transition notation worked well
   - `Start --(BEGIN, count=0)--> Counting` easily parsed

### What Didn't Work
1. **Guard evaluation to False**: Model defaults to "pass" path
   - `score=30 < threshold=50` → predicted `result='pass'` (should be 'fail')
   - 3/12 errors were this exact pattern

2. **Longer sequences**: counter_5 predicted count=4 (off-by-one)
   - Model may lose track over 5+ TICK events

## Gaps Identified

| Gap | Evidence | Severity |
|-----|----------|----------|
| **False-branch blindness** | All 3 errors predicted 'pass' when guard was False | High |
| **Sequence length limits** | counter_5 wrong, counter_3 and counter_7 correct | Medium |
| **Guard semantics** | Model ignores `score < threshold` meaning | High |

### Specific Failures
```
counter_5:      Expected count=5, Predicted count=4
branch_score30: Expected result='fail', Predicted result='pass'
conditional_25: Expected result='fail', Predicted result='pass'
```

## Recommendations

### Short-term
1. **Add explicit guard evaluation examples** in few-shot prompt
   - Include examples where guard evaluates False
   - Show both branches: `score >= 50 → pass` AND `score < 50 → fail`

2. **Chain-of-thought for guards**: Ask model to evaluate guards step-by-step
   ```
   Guard: score >= 50
   score = 30
   30 >= 50? No → take FailPath
   ```

3. **Reduce sequence length**: Break long sequences into checkpoints

### Long-term
1. **Fine-tune on guard evaluation**: Create dataset of (guard_expr, context, result) triples
2. **Hybrid approach**: Use symbolic execution for guards, LLM for action effects
3. **Verify predictions**: Cross-check LLM output against actual SC executor

## Files

- `__init__.py` - Module exports
- `context_predictor.py` - Few-shot LLM prediction
- `benchmark.py` - Test harness with L1-L4 cases
- `results/REAL_RESULTS.json` - Raw results

## Reproduction

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_context_prediction.benchmark
```
