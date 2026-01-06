# exp_sc_lora_finetuning - Research Notes

## Core Goal

**Improve statechart generation validity by +15% using LoRA fine-tuning.**

Base model: Qwen2.5-Coder-0.5B-Instruct
Framework: mlx_lm with LoRA adapters
Data: exp_synthetic_sc_dataset output

## LoRA Overview

### What is LoRA?

Low-Rank Adaptation (LoRA) adds trainable low-rank matrices to frozen pretrained weights:

```
W' = W + BA
```

Where:
- W: Original frozen weights (d × k)
- B: Low-rank matrix (d × r)
- A: Low-rank matrix (r × k)
- r << min(d, k): Rank hyperparameter

### Why LoRA for Statecharts?

1. **Efficient**: Only ~0.1-1% parameters trainable
2. **Fast**: Minutes on Apple Silicon vs hours for full fine-tune
3. **Composable**: Can merge/swap adapters
4. **Reversible**: Base model unchanged

## Hyperparameter Selection

### LoRA Rank

| Rank | Trainable Params | Capacity | Use Case |
|------|-----------------|----------|----------|
| 4 | ~100K | Low | Quick experiments |
| 8 | ~200K | Medium | Balanced |
| 16 | ~400K | High | Standard fine-tuning |
| 64 | ~1.6M | Very High | Complex tasks |

**Recommendation for SC generation**: rank=16

Rationale: Statechart JSON has consistent structure but varied content.
Medium-high rank captures structural patterns without overfitting.

### Alpha (Scaling)

Rule of thumb: `alpha = 2 × rank`

Alpha controls the magnitude of LoRA updates:
```
ΔW = (alpha / rank) × BA
```

Higher alpha = larger updates = faster learning but potential instability.

### Target Modules

For Qwen models:
- **Minimal**: `["q_proj", "v_proj"]` - 2 modules
- **Standard**: `["q_proj", "k_proj", "v_proj", "o_proj"]` - 4 modules
- **Full**: + `["gate_proj", "up_proj", "down_proj"]` - 7 modules

**Recommendation**: Standard (4 modules)

Attention layers capture structural patterns in JSON.
MLP layers less critical for format learning.

### Learning Rate

| Base Model Size | Recommended LR |
|-----------------|----------------|
| 0.5B | 2e-4 to 5e-4 |
| 1.5B | 1e-4 to 2e-4 |
| 7B | 5e-5 to 1e-4 |

**Recommendation for 0.5B**: 2e-4 with cosine schedule

## Training Strategy

### Dataset Requirements

Minimum for statechart fine-tuning:
- ~1000 examples for basic improvement
- ~5000 examples for robust generalization
- ~10000+ examples for complex hierarchical statecharts

### Data Format

```json
{
  "prompt": "Generate a statechart JSON for: <description>",
  "completion": "<valid statechart JSON>"
}
```

### Training Schedule

```
Epochs: 3
Warmup: 100 steps
Batch size: 4
Gradient accumulation: 4
Effective batch: 16
```

### Early Stopping

Monitor validation loss with patience=3.
Stop if no improvement for 3 consecutive evaluations.

## Validity Metrics

### Metric Definitions

1. **JSON Validity**: Output parses as valid JSON
2. **Schema Validity**: Has required fields (root_state, transitions)
3. **Structural Validity**: States reachable, types valid
4. **Overall Validity**: All checks pass

### Expected Baseline (Qwen 0.5B)

| Metric | Expected Rate |
|--------|--------------|
| JSON | ~60-70% |
| Schema | ~50-60% |
| Structural | ~40-50% |
| Overall | ~35-45% |

### Target After Fine-tuning

| Metric | Target Rate | Improvement |
|--------|-------------|-------------|
| JSON | ~85-90% | +20-25pp |
| Schema | ~75-85% | +20-25pp |
| Structural | ~60-70% | +15-20pp |
| Overall | ~55-65% | +15-20pp |

## Ablation Studies

### Rank Ablation

Expected results:
| Rank | Validity Δ | Time |
|------|-----------|------|
| 4 | +8% | 10min |
| 8 | +12% | 12min |
| 16 | +15% | 15min |
| 32 | +16% | 20min |
| 64 | +15% | 30min |

Diminishing returns above rank=16.

### Module Ablation

| Modules | Validity Δ |
|---------|-----------|
| q,v only | +10% |
| q,k,v | +13% |
| q,k,v,o | +15% |
| all 7 | +16% |

Attention output (o_proj) important for JSON structure.

## Common Issues

### 1. Mode Collapse

Symptom: Model generates same statechart regardless of prompt.
Cause: Overfitting on small dataset.
Fix: More data, lower LR, higher dropout.

### 2. Invalid JSON

Symptom: Missing brackets, quotes, commas.
Cause: Tokenization issues, truncation.
Fix: Ensure max_seq_length covers full outputs.

### 3. Schema Errors

Symptom: Missing root_state, wrong type values.
Cause: Insufficient schema examples in training.
Fix: Data augmentation, schema-focused prompts.

### 4. Speed Degradation

Symptom: Fine-tuned slower than base.
Cause: Larger adapter, no fusion.
Fix: Merge adapter into base weights for inference.

## Integration

### With exp_synthetic_sc_dataset

```python
from exp_synthetic_sc_dataset import load_dataset
from exp_sc_lora_finetuning import train_lora

dataset = load_dataset("path/to/synthetic_data")
result = train_lora(dataset)
```

### With exp_qwen_sc_to_code

Use fine-tuned model for higher-quality statecharts:

```python
from exp_sc_lora_finetuning import load_finetuned_model
from exp_qwen_sc_to_code import generate_code

model = load_finetuned_model("path/to/adapter")
sc = model.generate("Generate statechart for: ...")
code = generate_code(sc)
```

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- mlx_lm documentation
- Qwen technical report
