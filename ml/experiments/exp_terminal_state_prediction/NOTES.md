# exp_terminal_state_prediction

## Hypothesis

Can an LLM (Qwen2.5-Coder-1.5B) predict the final configuration of a statechart
after executing a sequence of events, given only the SC definition and event trace?

This tests the model's ability to:
1. Parse state transition rules
2. Execute transitions step-by-step
3. Handle statechart-specific semantics (hierarchy, parallel regions)

## Results

**Overall Metrics:**
- Exact match: 39% (27/70)
- Jaccard similarity: 40%
- Inference time: 245s for 70 cases (~3.5s/case)

**By Category:**
| Category   | Exact Match | Jaccard | Notes |
|------------|-------------|---------|-------|
| Linear     | 45% (9/20)  | 45%     | Sequential A→B→C→D chains |
| Cycle      | 65% (13/20) | 65%     | Best performance |
| Hierarchy  | 15% (3/20)  | 15%     | Worst performance |
| Parallel   | 20% (2/10)  | 30%     | Multiple active states |

## Key Findings

**What Worked:**
- Chain-of-thought prompting essential (200 tokens for step-by-step reasoning)
- Few-shot example significantly improved accuracy
- Simple transition rule format (`A + EVENT -> B`) parsed reliably
- Cyclic patterns predicted well (model recognizes repetition)

**What Didn't Work:**
- Short max_tokens (50) caused truncated reasoning, wrong answers
- Composite state semantics (entering `Active` should yield `Running`)
- Parallel region semantics (multiple simultaneous active states)
- Off-by-one errors in longer traces

## Gaps Identified

1. **Hierarchy Entry Semantics**: Model predicts composite state name (`Active`)
   instead of entering its initial substate (`Running`). Lacks understanding
   that configurations only contain leaf states.

2. **Parallel Region Handling**: Model outputs single state for parallel SCs
   instead of multiple active states. Doesn't understand AND-decomposition.

3. **Trace Length Sensitivity**: Accuracy degrades with longer event sequences.
   More steps = more opportunities for cumulative errors.

4. **Rule Application Consistency**: Sometimes skips transitions or applies
   wrong rule when multiple transitions share same event name.

## Recommendations

1. **Fine-tuning**: Train on SC execution traces with explicit step-by-step
   annotations showing state before/after each transition.

2. **Structured Output**: Use constrained generation to output only valid
   state names, preventing hallucination.

3. **Hierarchy Encoding**: Add explicit markers for composite vs leaf states
   in the prompt (e.g., `Active(composite, initial=Running)`).

4. **Parallel State Examples**: Include few-shot examples specifically for
   parallel regions showing comma-separated output.

5. **Verification Loop**: Generate trace, then verify each step against
   transition rules before outputting final answer.

## Report Sent to Orchestrator

```
[F624]: TERMINAL_STATE exact=39%, jaccard=40%, by_type=[linear:45%, cycle:65%, hier:15%, parallel:20%]

Test summary:
- Total: 70 test cases (10 per machine)
- Linear chains: 45% exact match - model traces correctly but sometimes off-by-one
- Cycles: 65% exact match - best performance, cyclic patterns predictable
- Hierarchy: 15% exact match - struggles with composite->leaf state mapping
- Parallel: 20% exact match - doesn't handle multiple active regions well

Key findings:
- Chain-of-thought prompting essential (200 tokens needed for reasoning)
- Model good at simple sequential transitions
- Struggles with statechart-specific semantics (hierarchy, parallel)
- Would benefit from fine-tuning on SC execution traces

Files created:
- experiments/exp_terminal_state_prediction/__init__.py
- experiments/exp_terminal_state_prediction/state_predictor.py
- experiments/exp_terminal_state_prediction/benchmark.py
```

## Files

- `__init__.py` - Test machines, TraceExecutor, evaluation metrics
- `state_predictor.py` - LLM predictor with chain-of-thought prompting
- `benchmark.py` - Benchmark runner with category breakdown
