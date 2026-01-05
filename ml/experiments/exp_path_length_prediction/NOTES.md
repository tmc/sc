# exp_path_length_prediction

## Hypothesis
Given a statechart and initial context, a small LLM (1.5B) can predict the number of state transitions needed to reach a terminal (final) state.

## Approach
- Extract statechart structure (states, transitions, guards)
- Use few-shot prompting with examples of different path types
- Test three categories: fixed path, variable path, unbounded

## Test Categories

| Category | Description | Example |
|----------|-------------|---------|
| Fixed | Deterministic path length | Counter machine with target=N |
| Variable | Path depends on guards/context | Branching based on score threshold |
| Unbounded | No terminal state reachable | Cyclic graph with no final states |

## Results

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Fixed | 0 | 3 | 0% |
| Variable | 0 | 3 | 0% |
| Unbounded | 1 | 2 | 50% |
| **Overall** | 1 | 8 | **12%** |

- Within ±1: 12%
- MAE: 2.0

## Key Findings

1. **Example Copying**: Model tends to copy numbers from few-shot examples rather than reasoning about the specific statechart structure.

2. **Default to Infinite**: When uncertain, model defaults to "infinite" regardless of actual structure.

3. **No Multi-Step Reasoning**: The 1.5B model cannot perform the multi-step reasoning required:
   - Parse transition structure
   - Identify loops and bounds
   - Calculate path length through guards

4. **Prompt Sensitivity**: Results vary wildly based on prompt format:
   - With calculation hints: 100% (but that's cheating)
   - Without hints: ~12%
   - Model follows example patterns, not logical reasoning

## Technical Details

- Model: Qwen2.5-Coder-1.5B-Instruct-4bit
- Prompt Strategy: Few-shot with varied examples (2, 5, infinite)
- Path types tested: fixed (counter), variable (branching), unbounded (cycles)

## Conclusion

Path length prediction requires sophisticated reasoning capabilities beyond what 1.5B models can perform:
- Graph traversal understanding
- Loop bound calculation
- Guard condition evaluation

This is a **hard task** that may require:
- Larger models (7B+)
- Chain-of-thought prompting
- External reasoning tools (symbolic execution)
- Fine-tuning on path analysis tasks

## Usage

```python
from experiments.exp_path_length_prediction import run_benchmark, format_report

result = run_benchmark(samples_per_type=3)
print(format_report(result))
```
