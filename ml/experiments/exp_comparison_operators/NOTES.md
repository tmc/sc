# exp_comparison_operators_improved

**Session**: DDB5
**Model**: Qwen2.5-Coder-1.5B-Instruct-4bit
**Date**: 2025-01-05

## Hypothesis

L3 guard synthesis (comparison operators) was reported at 20% accuracy in prior experiments. We hypothesized that improved prompting strategies could achieve 80%+ accuracy.

Strategies tested:
1. **Baseline** - Simple direct prompt
2. **Chain-of-Thought (CoT)** - Step-by-step reasoning
3. **Template** - Explicit structure with placeholders
4. **Augmented** - 10+ few-shot examples

## Results

**Report to B90CCCD4:**
```
[DDB5]: L3_GUARD_IMPROVED baseline=100%, cot=93%, template=87%, augmented=100%, best=100%
```

### Strategy Comparison

| Strategy | Accuracy | Has Comparison | Avg Time (ms) |
|----------|----------|----------------|---------------|
| Baseline | **100%** | 100% | 4538 |
| CoT | 93% | 100% | 7147 |
| Template | 87% | 93% | 8154 |
| Augmented | **100%** | 100% | 10501 |

**Best**: Baseline and Augmented tied at 100%
**Fastest**: Baseline (4.5s per test)

## Key Findings

### Surprising Result: Baseline Achieves 100%

The original 20% figure may have been from:
- Different test cases (more ambiguous inputs)
- Different model version
- Different prompt format
- Integration with full SC generation (not isolated guard task)

### What Worked

1. **Direct prompting** - Simple "Convert to guard expression" works perfectly
2. **All operators covered** - <, >, <=, >=, ==, != all handled correctly
3. **Variable extraction** - Model correctly identifies variable names
4. **Literal vs reference** - Handles both numeric literals and variable references

### What Didn't Work

1. **CoT introduced errors** - Added reasoning caused "time over limit" → "time > 10" (hallucinated literal)
2. **Template format confusion** - "value exactly 42" returned verbatim, "==" operator missed
3. **Longer prompts = slower** - Augmented took 2.3x longer than baseline for same accuracy

## Gaps Identified

| Gap | Evidence | Severity |
|-----|----------|----------|
| CoT can hallucinate values | "limit" → "10" | Medium |
| Template format leakage | Verbatim input returned | Low |
| Isolated vs integrated task | May differ in full SC context | Unknown |

### Error Analysis

**CoT (1 error)**:
- Input: "time over limit"
- Expected: `time > limit`
- Got: `time > 10`
- Cause: CoT reasoning introduced literal substitution

**Template (2 errors)**:
- "retries exceed maximum" → `retries > 0` (wrong value)
- "value exactly 42" → `value exactly 42` (no operator)

## Recommendations

### For L3 Guard Synthesis

1. **Use simple baseline prompts** - Direct prompting outperforms complex strategies
2. **Avoid CoT for simple transformations** - Adds overhead and can introduce errors
3. **Few-shot helps with edge cases** - Augmented matches baseline, useful for harder cases

### For Further Testing

1. **Test in integrated SC generation** - May behave differently when generating full statecharts
2. **Add ambiguous cases** - "X is more or less Y" type inputs
3. **Test with boolean conditions** - Ensure no regression on L1/L2

## Files

- `comparison_dataset.py` - 100+ L3 examples (17 per operator)
- `improved_generator.py` - 4 prompting strategies
- `benchmark_improved.py` - Comparison benchmark
- `results/IMPROVED_RESULTS.json` - Raw results

## Reproduction

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_comparison_operators.benchmark_improved
```

## Conclusion

The L3 guard synthesis "problem" appears to be context-dependent. When isolated as a simple NL→expression task, Qwen2.5-Coder-1.5B achieves **100% accuracy** with baseline prompting.

The original 20% figure likely came from a more complex integrated task where the model had to:
- Generate full statechart structure
- Identify where guards should be placed
- Synthesize guards in context

**Recommendation**: Decompose SC generation into subtasks (structure → guards → actions) rather than end-to-end generation.
