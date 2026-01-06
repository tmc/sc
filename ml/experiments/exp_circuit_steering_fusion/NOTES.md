# exp_circuit_steering_fusion: Circuit-Specific Steered Decoding

## Goal

Apply circuit-specific steering during token-by-token generation, comparing
against global (single-layer) steering.

**Hypothesis**: Targeting specific circuits (TRANSITION L8-14, HIERARCHY L0-6)
with steering should outperform global steering on a single layer.

## Circuits

From exp_mlux_sc_circuits and exp_circuit_intervention:

| Circuit | Layers | Function |
|---------|--------|----------|
| TRANSITION | L8-14 | Transition validity, source/target matching |
| HIERARCHY | L0-6 | Hierarchical structure, nested states |
| STRUCTURAL | L0-6 | JSON structure, syntax |

## Approach

### 1. Circuit Hooks (circuit_hooks.py)

Define circuits and create hooks for layer ranges:

```python
CIRCUITS = {
    CircuitType.TRANSITION: Circuit(layers=range(8, 15)),  # L8-14
    CircuitType.HIERARCHY: Circuit(layers=range(0, 7)),    # L0-6
}

# Create hooks for a circuit
hooks = create_transition_hooks(steering_vector, alpha=1.0)
# Returns: [(model.layers.8, hook), (model.layers.9, hook), ...]
```

### 2. Fused Steering (fused_steering.py)

FusedSteeredSampler with multi-circuit support:

```python
sampler = FusedSteeredSampler()
sampler.set_circuit_vectors(
    transition_vector=t_vec,
    hierarchy_vector=h_vec,
)

# Generate with fused steering (both circuits)
config = FusedSteeringConfig(mode="fused")
result = sampler.generate(prompt, config)
```

Modes:
- `baseline`: No steering
- `global`: Single layer (L12) steering
- `transition`: TRANSITION circuit only (L8-14)
- `hierarchy`: HIERARCHY circuit only (L0-6)
- `fused`: Both circuits

### 3. Benchmark (benchmark.py)

Compare modes on prompt categories:

- **TRANSITION_PROMPTS**: Focus on transitions
- **HIERARCHY_PROMPTS**: Focus on nested states
- **MIXED_PROMPTS**: General statechart generation

## Integration

Uses components from:
- `exp_steered_decoding`: Token-by-token generation with hooks
- `exp_circuit_intervention`: Circuit definitions
- `exp_mlux_sc_steering`: Steering vector computation

## Usage

```python
from exp_circuit_steering_fusion import (
    CircuitFusionBenchmark,
    FusedSteeredSampler,
)

# Quick comparison
benchmark = CircuitFusionBenchmark()
result = benchmark.run_quick()

print(f"Best mode: {result.best_overall_mode}")
print(f"Fused wins: {result.fused_wins}")
```

## Expected Results

| Mode | Expected Performance | Rationale |
|------|---------------------|-----------|
| baseline | Low | No steering applied |
| global | Medium | Steering at one layer |
| transition | Medium-High | Good for transition-focused prompts |
| hierarchy | Medium-High | Good for hierarchy-focused prompts |
| fused | High | Best of both circuits |

## Key Insight

Circuit-specific steering allows targeted intervention:
- HIERARCHY circuit handles early processing (structure)
- TRANSITION circuit handles later processing (validity)

Global steering at a single layer misses one or the other.

## Results

| Test | Status | Notes |
|------|--------|-------|
| Circuit hooks | PASS | 7 hooks for TRANSITION (L8-14), 7 for HIERARCHY (L0-6) |
| Fused generation | PASS | 14 combined hooks, steering at each step |
| Mode comparison | PASS | All modes produce different outputs |

Mode test results:
- `global`: Single layer (L12), 1 hook
- `transition`: L8-14, 7 hooks
- `hierarchy`: L0-6, 7 hooks
- `fused`: Both circuits, 14 hooks

## Future Work

1. **Adaptive circuit selection**: Choose circuits based on prompt type
2. **Per-circuit vectors**: Different steering vectors per circuit
3. **Dynamic alpha**: Adjust steering strength based on generation progress
4. **Attention steering**: Add attention-specific hooks to circuits
