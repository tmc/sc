# Constrained Decoding Approaches Comparison

## Summary of Approaches

| Approach | Description | Mask Time | Valid JSON | Key Trade-off |
|----------|-------------|-----------|------------|---------------|
| **Naive** | Map tokens to first event | 2.7ms | ✅ (with ForceClose) | Fast but imprecise |
| **TokenSim-TopK** | Simulate each top-k candidate | 36ms | ❌ (long strings) | Precise but unbounded strings |
| **Retok-TopK** | Validate via simulation + retokenize | 35ms | ✅ | Handles boundaries well |
| **JumpForward** | Force tokens on singular paths | 967ms | ✅ | Correct but slow |

## Key Insights

### 1. Token Simulation (llguidance-style)
The most important finding is that **proper token simulation** is required to correctly validate multi-character tokens like `"},`. The naive approach of checking only the first event fails to catch tokens that would cause invalid transitions mid-token.

```python
# Naive: Only checks first event
if '}' in token_str:
    event = 'RBRACE'  # Misses COMMA in `},`

# Correct: Simulate all characters
for char in token_str:
    event = tokenizer.process_char(char)
    if event not in enabled_events:
        return False  # Reject token
```

### 2. Top-K Optimization
Checking all 150k+ tokens per step is infeasible. The **top-k optimization** (checking only the top 500 candidates by logit score) reduces mask computation from O(V) to O(k) while rarely missing valid tokens.

### 3. ForceClose Mechanism
The statechart's **ForceClose state** is essential for guaranteed termination. Without it, the model can generate arbitrarily long valid content.

Triggers:
- `depth >= max_depth` when opening new structure
- `element_count >= max_elements` when adding array elements

### 4. Trailing Comma Problem
When ForceClose triggers, there may be a trailing comma that makes JSON invalid:
```json
{"children": [{"label": "A"},]}  // Invalid: comma before ]
```

**Solution**: Strip trailing comma before adding closing tokens.

### 5. String Length Limitation
TokenSim-TopK failed because the statechart allows **arbitrarily long strings**. The model generated:
```
"Sleeping in a computer or other device."
```

**Potential fixes**:
1. Add `string_length` variable and `max_string_length` guard
2. Token-level limit in generation loop
3. Bias against STRING tokens when string is long

## Architecture Comparison

### Pre-computed Masks (Outlines-style)
- **Pros**: O(1) lookup at inference time
- **Cons**: High startup cost, memory usage, doesn't scale with grammar complexity

### Compute-per-Token (llguidance-style)
- **Pros**: No startup cost, handles complex grammars
- **Cons**: O(k) per token where k = candidates checked
- **Optimization**: Only check top-k candidates by logit score

### Re-tokenization (SGLang-style)
- **Pros**: Handles tokenization boundaries perfectly
- **Cons**: Re-encoding overhead (~4%)
- **Best for**: When token boundaries are critical

### Jump-Forward
- **Pros**: Skips deterministic portions, reduces sampling overhead
- **Cons**: High mask time due to lookahead exploration
- **Best for**: Highly constrained grammars with long deterministic paths

## Recommendation

For statechart-guided JSON generation:

1. **Use Token Simulation with Top-K** for validation accuracy
2. **Implement ForceClose** for guaranteed termination
3. **Add string length limits** to prevent runaway generation
4. **Consider Jump-Forward** for boilerplate portions

## Code Structure

```
approaches.py
├── NaiveApproach          # Baseline: first-event mapping
├── TokenSimulationApproach # llguidance-style compute-per-token
├── RetokenizationApproach # SGLang-style string-level validation
├── JumpForwardApproach    # Singular transition path detection
└── benchmark_approaches() # Comparison runner
```

## References

- [Fast JSON Decoding with Compressed FSM - LMSYS](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
- [Guiding LLMs The Right Way - arXiv](https://arxiv.org/html/2403.06988v1)
- [llguidance - Guidance AI](https://github.com/guidance-ai/llguidance)
- [Structured Decoding in vLLM](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
