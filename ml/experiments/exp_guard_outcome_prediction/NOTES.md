# Guard Outcome Prediction Experiment

## Hypothesis

Can LLMs predict which guard expression fires given a set of competing guards and context values?

This is a core capability for statechart execution: when multiple transitions have guards, the runtime must evaluate each guard against the current context to determine which transition is enabled.

## Task Definition

**Input:** List of guard expressions + context values
**Output:** Index of first guard that evaluates to TRUE (0..N-1) or -1 if none

**Guard Complexity Levels:**
- L1: Simple comparison (`x > 5`)
- L2: Compound comparison (`x > 5 && y < 10`)
- L3: Arithmetic comparison (`(x + y) > threshold`)
- L4: Complex/list access (`len(items) > 0 && items[0]['valid']`)

## Results

**Model:** `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`
**Samples:** 10 per level (40 total)

| Level | Description | Accuracy |
|-------|-------------|----------|
| L1 | Simple comparison | 70% (7/10) |
| L2 | Compound comparison | 50% (5/10) |
| L3 | Arithmetic comparison | 40% (4/10) |
| L4 | Complex/list access | 70% (7/10) |
| **Overall** | All levels | **57.5% (23/40)** |

**Report sent to B90CCCD4:**
```
[91C6]: GUARD_OUTCOME accuracy=57%, by_level=[L1:70%, L2:50%, L3:40%, L4:70%]
```

## Key Findings

### What Worked
1. **Few-shot prompting without chat template** - Direct completion format outperformed chat format
2. **Simple guards (L1)** - Model reliably evaluates single comparisons
3. **Complex guards (L4)** - Surprisingly good at list/property access patterns
4. **Guard index 0** - Model consistently correct when answer is 0

### What Didn't Work
1. **Chat template** - Caused model to generate Python code instead of answers
2. **Middle-range answers** - Model biased toward extremes (0 or 2), struggles with index 1
3. **Arithmetic evaluation (L3)** - Weakest performance, model doesn't reliably compute sums
4. **Compound boolean logic (L2)** - Struggles with AND/OR short-circuit evaluation

## Gaps Identified

1. **Arithmetic reasoning gap**: Model can't reliably compute `(x + y)` and compare to threshold
2. **Boolean composition gap**: Struggles to evaluate `A && B` when A is true but B is false
3. **Order sensitivity**: Doesn't consistently evaluate guards in order (first TRUE wins)
4. **Default bias**: When uncertain, model defaults to 0 instead of reasoning through

## Recommendations

### Short-term Improvements
1. **Chain-of-thought prompting**: Add explicit "Step 1: Evaluate guard 0..." format
2. **More balanced examples**: Current examples may bias toward certain answers
3. **Separate arithmetic**: Pre-compute arithmetic expressions in prompt

### Training Improvements
1. **GRPO on guard evaluation**: Fine-tune specifically for boolean/arithmetic reasoning
2. **Curriculum learning**: Train on L1 first, then L2, etc.
3. **Synthetic data**: Generate thousands of guard/context/answer triples

### Architecture Considerations
1. **Hybrid approach**: Use symbolic evaluation for arithmetic, LLM for semantic understanding
2. **Verifier model**: Second pass to check if predicted guard actually fires
3. **Constrained decoding**: Force output to be valid index (0, 1, 2, or -1)

## Files

| File | Purpose |
|------|---------|
| `guard_predictor.py` | Guard generation and evaluation |
| `benchmark.py` | LLM benchmark harness |
| `__init__.py` | Module exports |

## Running the Experiment

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_guard_outcome_prediction.benchmark
```

## Next Steps

1. Try larger model (3B+) for arithmetic reasoning
2. Implement chain-of-thought variant
3. Compare with symbolic baseline (direct eval)
4. Integrate with execution prediction pipeline
