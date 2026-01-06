# exp_steered_decoding: Custom Autoregressive Decoding with Steering

## Goal

Implement token-by-token generation with steering vectors applied at each
forward pass step, rather than just the initial encoding.

**Target: +10% validity improvement on edge cases**

## Problem

Standard generation (mlx_lm.generate) doesn't support hooks during autoregressive
decoding. The steering experiment (exp_mlux_sc_steering) computes valid steering
vectors but can't apply them during generation.

## Solution

Custom autoregressive decoding loop:

```python
for step in range(max_tokens):
    # Forward pass WITH hooks
    logits = model.run_with_hooks(tokens, hooks=[(layer, steering_hook)])

    # Sample next token
    next_token = sampler(logits[:, -1, :])

    # Append and continue
    tokens = concat(tokens, next_token)
```

## Key Components

### 1. steered_sampler.py

Custom sampler that applies hooks at each generation step:
- `SteeredSampler.generate()`: Main generation with optional steering
- `SteeredSampler.generate_comparison()`: Baseline vs steered comparison
- Uses mlux `run_with_hooks` for hook injection

### 2. hook_injection.py

Hook creation and management:
- `create_steering_hook()`: Basic steering hook
- `create_multi_layer_hooks()`: Apply to multiple layers with decay
- `create_contrastive_hook()`: Push towards positive, away from negative
- `AdaptiveSteeringHook`: Decay steering strength during generation

Hook signature for mlux:
```python
def hook(args, output, wrapper) -> modified_output:
    # output shape: (batch, seq_len, hidden_dim)
    # Add steering to last position
    return output[:, -1:, :] + steering_vector
```

### 3. benchmark.py

Compare steered vs baseline:
- Edge case prompts (minimal, complex, nested, parallel, ambiguous)
- Standard prompts (simple, well-specified)
- Grid search over layers and alphas

## Integration

Uses steering vectors from exp_mlux_sc_steering:

```python
from experiments.exp_mlux_sc_steering import SteeringVectorComputer

computer = SteeringVectorComputer(model=sampler.model)
vec = computer.compute_averaged(layer=12)
```

## Steering Strategies

1. **Last-position steering**: Apply only to the last token position (default)
2. **All-position steering**: Apply to entire sequence (stronger)
3. **Multi-layer steering**: Apply to multiple layers with decay
4. **Adaptive steering**: Reduce strength as generation progresses

## Edge Cases Tested

- Minimal: Might forget initial state
- Complex: Might produce syntax errors
- Nested: Might break hierarchy
- Parallel: Might confuse AND/OR states
- Ambiguous: Might produce incomplete output

## Usage

```python
from exp_steered_decoding import SteeredDecodingBenchmark

benchmark = SteeredDecodingBenchmark()
summary = benchmark.run_edge_cases()

print(f"Baseline: {summary.baseline_rate:.1%}")
print(f"Steered: {summary.steered_rate:.1%}")
print(f"Improvement: {summary.improvement:+.1%}")
```

## Performance Considerations

1. **No KV cache**: Current implementation recomputes full sequence at each step
   - Future: Implement KV caching for faster generation

2. **Single sample**: Sequential generation, no batching
   - Future: Batch parallel generations

3. **Hook overhead**: Small overhead per forward pass
   - Negligible compared to model compute

## Results

| Test | Status | Notes |
|------|--------|-------|
| Hook creation | PASS | Correctly modifies hidden states at last position |
| Steered sampler | PASS | Token-by-token generation with hooks |
| Steering applied | PASS | `steering_applied: True` in results |
| Output difference | PASS | Baseline vs steered produce different outputs |

Current limitation: The model uses a different JSON schema than our validator expects.
Need properly computed steering vectors from exp_mlux_sc_steering to see validity
improvement.

## Future Work

1. **KV caching**: Store key/value pairs for faster generation
2. **Batched generation**: Generate multiple samples in parallel
3. **Token-specific steering**: Vary steering based on token type
4. **Confidence-based steering**: Stronger steering when model is uncertain
