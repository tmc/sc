# exp_trace_continuation

## Hypothesis

LLMs can predict the next N states in a statechart execution trace given:
- The statechart structure (states, transitions, guards)
- A partial execution trace (sequence of visited states)

This tests pattern recognition, SC structure understanding, and implicit event/guard inference.

## Results

| Metric | Value |
|--------|-------|
| Next 1 state accuracy | 20% (3/15) |
| Next 3 states accuracy | 0% (6/6) |
| Next 5 states accuracy | 0% (2/2) |
| Overall first-correct | 13% (3/23) |

**Report sent to B90CCCD4:**
```
[9D1B]: TRACE_CONTINUATION next_1=20%, next_3=0%, next_5=0%,
notes=[Model struggles with self-loops and guard-dependent transitions,
predicts next distinct state rather than current state repeating]
```

## Key Findings

### What Worked
- Simple linear transitions (e.g., `PassPath -> Merge -> Final`)
- Cases where only one outgoing transition exists
- State name extraction from LLM output

### What Didn't Work
- **Self-loops**: Model predicts next *distinct* state instead of same state
  - Example: `Counting -> Counting` predicted as `Counting -> Done`
- **Guard-dependent branching**: No understanding of context values
  - Example: Doesn't track `count < target` to know loop continues
- **Multi-step prediction**: Accuracy drops to 0% for n>1

## Gaps Identified

1. **No context tracking**: Model sees guard expressions but can't evaluate them
   - `count < 5` means nothing without knowing current `count` value

2. **Self-loop blindness**: Training data likely has more distinct state sequences
   - Model biased toward predicting state *changes* not state *persistence*

3. **Hierarchical state confusion**: Jumps between composite states prematurely
   - `P1_Process` -> predicts `Phase2` instead of continuing in Phase1

4. **No event inference**: Model doesn't reason about which events would fire
   - Prompt shows transitions but model doesn't simulate execution

## Recommendations

### Short-term Improvements
1. **Include context values in prompt**: Show `{count: 2, target: 5}` explicitly
2. **Add self-loop examples**: Few-shot with `A -> A -> A -> B` patterns
3. **Simplify to event prediction**: Predict next *event* not next *state*

### Medium-term Experiments
1. **Chain-of-thought**: Have model reason through guard evaluation
2. **Two-stage prediction**: First predict event, then compute resulting state
3. **Fine-tuning**: Train on synthetic trace datasets with self-loops

### Alternative Approaches
1. **Hybrid**: Use LLM for event selection, deterministic executor for state
2. **Constrained decoding**: Only allow valid next states as output tokens
3. **Retrieval-augmented**: Find similar traces in training data

## Files

- `__init__.py` - Module exports
- `trace_continuer.py` - LLM prediction logic, dataset generation
- `benchmark.py` - Evaluation harness and metrics

## Model

- `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`
- Max tokens: 100
- Samples: 23 continuation tasks across 3 SC templates
