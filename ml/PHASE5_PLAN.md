# Phase 5: Scaling, Fine-tuning & Real-World Applications

## Context (Phase 4 Achievements)
- Head amplification: ~1.5x sweet spot, o_proj pre-hooks working
- Circuit mapping: STRUCTURAL L0-6, TRANSITION L8-14, HIERARCHY L0-6+L23
- TRM + steering hybrid: 99% validity in 1.2 iterations
- Key insight: Model size (0.5B) is limiting factor, not steering mechanism
- mlux SAE module: Complete with 4 architectures

---

## Tier 1: Scaling Experiments (High Priority)

### 1. exp_model_scaling_comparison
**Goal**: Validate that steering improves with model scale.

Test steering on multiple model sizes:
- Qwen2.5-Coder-0.5B (baseline, have results)
- Qwen2.5-Coder-1.5B
- Qwen2.5-Coder-3B
- Qwen2.5-Coder-7B (if memory allows)

Metrics:
- Baseline validity rate per model size
- Steering improvement (Δ validity)
- Optimal amplification scale per model

**Target**: Show steering Δ increases with model size

### 2. exp_sae_feature_steering
**Goal**: Use SAE features for more precise steering.

Leverage mlux SAE module:
- Train SAE on SC generation activations
- Identify features for: validity, hierarchy, transitions
- Create feature-based steering vectors
- Compare: head steering vs feature steering

**Target**: Find 5+ interpretable SC-related features

### 3. exp_cross_model_steering_transfer
**Goal**: Test if steering vectors transfer between models.

- Compute steering vectors on Qwen-0.5B
- Apply to Qwen-1.5B/3B (same family)
- Measure transfer effectiveness

**Target**: >50% effectiveness within model family

---

## Tier 2: Fine-tuning Experiments

### 4. exp_sc_lora_finetuning
**Goal**: Fine-tune model specifically for SC generation.

- LoRA fine-tuning on Qwen-0.5B
- Training data: (prompt, valid_sc) pairs from benchmark
- Compare: base+steering vs fine-tuned vs fine-tuned+steering

**Target**: Fine-tuned achieves 95%+ without steering

### 5. exp_steering_aware_finetuning
**Goal**: Train model to be more responsive to steering.

- Include steering signal in training
- Contrastive loss: steered vs unsteered generations
- Model learns to "listen" to steering vectors

**Target**: 2x steering sensitivity vs base model

### 6. exp_distillation_to_smaller
**Goal**: Distill steering knowledge to smaller model.

- Teacher: Qwen-3B with optimal steering
- Student: Custom 100M parameter model
- Distill SC generation + steering response

**Target**: 100M model matching 0.5B steered performance

---

## Tier 3: Real-World Applications

### 7. exp_react_state_extraction
**Goal**: Extract statecharts from React/XState codebases.

- Parse React components with useReducer/useState
- Extract state transitions from code
- Generate formal statechart representation

Test on XState examples repository.

**Target**: 80%+ extraction accuracy

### 8. exp_game_logic_extraction
**Goal**: Extract statecharts from game source code.

Apply to:
- Zelda3 decompilation (already have access)
- Simple game repos (Tetris, Pac-Man clones)

Extract: Game states, enemy AI, animation FSMs

**Target**: Valid SC from 5+ game systems

### 9. exp_api_state_machine_inference
**Goal**: Infer API state machines from OpenAPI specs + traces.

- Parse OpenAPI spec for endpoints
- Analyze API call traces
- Infer state transitions with guards

**Target**: Infer state machine from 3+ real APIs

---

## Tier 4: Formal Methods Integration

### 10. exp_tla_plus_export
**Goal**: Export statecharts to TLA+ for model checking.

- SC proto → TLA+ spec translation
- Invariant generation from guards
- Integration with TLC model checker

**Target**: Verify generated SCs against safety properties

### 11. exp_alloy_validation
**Goal**: Use Alloy for SC constraint validation.

- Generate Alloy models from SC
- Define well-formedness constraints
- Find counterexamples for invalid SCs

**Target**: Catch 95%+ of structural errors

---

## Tier 5: Benchmarks & Datasets

### 12. exp_sc_generation_benchmark
**Goal**: Create standardized SC generation benchmark.

Categories:
- Simple (3-5 states)
- Hierarchical (nested, 2-3 levels)
- Parallel (orthogonal regions)
- Guards (conditional transitions)
- History (deep/shallow)
- Complex (all features)

**Target**: 100 tasks with gold standard SCs

### 13. exp_synthetic_sc_dataset_10k
**Goal**: Generate 10k training samples.

Methods:
- Template-based generation
- LLM generation + validation + repair
- Mutation from seed SCs

**Target**: 10k samples, 100% validated

---

## Session Assignment Plan

### Wave 1 (Start Immediately)
| Experiment | Session | Effort |
|------------|---------|--------|
| exp_model_scaling_comparison | TBD | 2-3h |
| exp_sae_feature_steering | TBD | 3-4h |
| exp_sc_generation_benchmark | TBD | 2-3h |

### Wave 2 (After Wave 1)
| Experiment | Session | Effort |
|------------|---------|--------|
| exp_sc_lora_finetuning | TBD | 4-6h |
| exp_synthetic_sc_dataset_10k | TBD | 4-6h |
| exp_react_state_extraction | TBD | 3-4h |

### Wave 3 (Research Frontier)
| Experiment | Session | Effort |
|------------|---------|--------|
| exp_tla_plus_export | TBD | 4-6h |
| exp_distillation_to_smaller | TBD | 6-8h |
| exp_cross_model_steering_transfer | TBD | 3-4h |

---

## Success Metrics

| Metric | Current | Phase 5 Target |
|--------|---------|----------------|
| Best validity (0.5B) | 99% (TRM+steering) | Baseline |
| Best validity (3B) | Unknown | 99%+ |
| SAE interpretable features | 0 | 5+ |
| Fine-tuned model validity | N/A | 95%+ |
| Real-world extraction | 0 | 5+ systems |
| Public benchmark tasks | 0 | 100 |
| Training dataset size | 0 | 10k |

---

## Key Research Questions

1. **Does steering scale?** Larger models should show larger Δ
2. **Can we fine-tune for steerability?** Train models to respond to steering
3. **Do SAE features provide better steering?** More interpretable than heads
4. **Can we extract SCs from real code?** Practical validation
5. **Does formal verification help?** Counterexample-guided repair
