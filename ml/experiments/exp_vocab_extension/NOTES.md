# exp_vocab_extension: Single-Token SC Commands

**Status**: PLANNED (Phase 6)
**Priority**: Future work
**Logged**: 2026-01-05

## Motivation

Current `meta_constrained_sampler` uses multi-token sequences for special commands:
- `<SC:LOAD>` → 5 tokens: [27, 3540, 25, 12988, 29]
- Constraint logic must handle partial token sequences

Single-token approach would simplify constraint enforcement.

## Findings from MODEL_SIZE Analysis

Tested across 0.5B, 1.5B, 3B Qwen2.5-Coder models:
- All use identical tokenizer (same token IDs)
- Special tokens consistently encode to 5 sub-tokens
- Roundtrip encode/decode: 100% accurate
- All models can generate special tokens in output

## Proposed Approach

1. **Tokenizer Extension**
   - Add 7 special tokens via `add_special_tokens()`
   - New IDs: 151936-151942

2. **Embedding Extension**
   - Resize: (151936, 112) → (151943, 112)
   - Initialize: mean of existing embeddings

3. **Output Layer Extension**
   - Resize lm_head to match

4. **Fine-tuning**
   - Generate training data with SC commands
   - LoRA fine-tune (~100-1000 examples)
   - Target: model learns when to emit SC commands

5. **Evaluation**
   - Compare constraint efficiency: single vs multi-token
   - Measure generation quality
   - Benchmark on SC generation tasks

## Dependencies

- `meta_constrained_sampler.py` (grammars/)
- MLX LoRA fine-tuning infrastructure
- Training data generation pipeline

## Related Work

- Current multi-token approach: 100% hierarchical, 83% GRPO (works well)
- Vocab extension may improve: constraint masking speed, generation coherence
