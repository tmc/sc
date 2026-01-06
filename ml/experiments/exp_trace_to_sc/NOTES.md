# exp_trace_to_sc / exp_trace_induction_cot

## Hypothesis

Chain-of-thought (CoT) prompting can improve trace-to-statechart induction
accuracy by forcing explicit reasoning about states and transitions rather
than copying example templates.

Baseline problem: Direct prompting achieves ~20% accuracy because models
copy the example structure instead of reasoning about the actual traces.

## Approach

1. **Trace Analysis**: Extract patterns algorithmically before prompting
   - Unique events (potential states)
   - Transition pairs (src -> tgt)
   - Initial state (most common first event)

2. **CoT Prompting**: Include analysis in prompt to guide reasoning
   - Show extracted states, transitions, initial state
   - Let model use this structure to generate JSON

3. **Schema Normalization**: Handle alternate output formats
   - `{states: [...], transitions: [...]}` format
   - `{root_state: {children: [...]}}` format

## Results

**Model:** Qwen2.5-Coder family (0.5B, 1.5B, 3B)

| Model | Direct | CoT | Delta |
|-------|--------|-----|-------|
| 0.5B  | 25%    | 50% | +25%  |
| 1.5B  | 33%    | 50% | +17%  |
| 3B    | 0%     | ?   | slow  |

**Transitions accuracy:** 0% for all (semantic mismatch)

## Key Findings

### What Worked

1. **CoT improves state identification** - +17-25% improvement
2. **Analysis pre-processing** - Helps model identify structure
3. **Schema normalization** - Handles model output variations
4. **Smaller models perform better** - 0.5B/1.5B > 3B for this task

### What Didn't Work

1. **Transition semantics** - Models misinterpret event/state relationship
   - Traces `[ON, OFF]` are EVENTS
   - Model names states after events, confusing transitions

2. **Template copying** - Direct prompting still copies examples

3. **3B model** - Slower and less accurate than smaller models

## Gaps Identified

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Event vs State confusion | 0% trans acc | Better naming conventions |
| Transition direction | Wrong from/to | Explicit state naming hints |
| Output format variance | Parsing errors | Schema normalization |
| Generation speed | Timeout | Use smaller models |

## Recommendations

### Short-term

1. **State naming hints** - Tell model to use descriptive state names
2. **Transition examples** - Show correct event->transition mapping
3. **Post-processing** - Fix common semantic errors

### Longer-term

1. **Fine-tuning** - Train on (trace, SC) pairs
2. **Two-stage** - LLM identifies structure, symbolic generates JSON
3. **Constrained decoding** - Force valid SC JSON structure

## Files Created

- `trace_analyzer.py` - Algorithmic trace analysis
- `cot_inducer.py` - CoT and Direct prompting inducers
- `benchmark_cot.py` - Comparison benchmark

## Report

Sent to orchestrator B90CCCD4:

```
[12FF]: TRACE_INDUCTION_COT direct_0.5B=25%, cot_0.5B=50%, direct_1.5B=33%, cot_1.5B=50%, delta=+17-25% states
Notes: Trans=0% (semantic mismatch), 3B underperforms smaller models
```

## Conclusion

CoT improves state identification by 17-25% but doesn't reach the 60% target.
The main bottleneck is semantic mismatch between event traces and state
naming conventions. Further work should focus on state naming guidance
and transition direction disambiguation.
