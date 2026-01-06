# exp_counterfactual_sc: Counterfactual State Prediction

## Goal

Given a statechart, current state, and event - predict the next state
and explain why that transition occurs.

## Architecture

```
predictor.py   - State transition prediction (rule-based + LLM)
explainer.py   - Natural language explanation generation
benchmark.py   - Accuracy and explanation quality evaluation
```

## Prediction Approach

1. **Deterministic Analysis** (primary):
   - Find transitions matching (current_state, event)
   - If single match: predict target state with high confidence
   - If no match: predict same state (no transition)

2. **LLM Resolution** (for ambiguity):
   - Multiple transitions with guards
   - LLM selects based on guard semantics

## Explanation Generation

- Template-based for simple cases
- LLM-enhanced for fluent natural language
- Includes: preconditions, postconditions, reasoning steps

## Benchmark Cases

| Domain | Cases | Description |
|--------|-------|-------------|
| Traffic Light | 4 | Cyclic transitions + unknown event |
| Door Lock | 4 | Lock/unlock with self-loops |
| Elevator | 3 | Multi-state movement |
| Vending Machine | 3 | Transaction flow |

**Total: 14 test cases**

## Evaluation Metrics

1. **Prediction Accuracy**: % correct next state predictions
2. **Explanation Quality** (1-5 scale):
   - Mentions event (+1)
   - Mentions target state (+1)
   - Grammatically coherent (+1)
   - Provides reasoning (+1)
   - Concise and clear (+1)

## Key Insights

1. **Rule-based is sufficient** for deterministic statecharts
2. **LLM helps** with guarded transitions and ambiguity
3. **Explanations** need both correctness and fluency
4. **Counterfactual reasoning** = "what if this event fired?"

## Dependencies

- mlx_lm (REAL inference)
- Qwen2.5-Coder-0.5B-Instruct model
