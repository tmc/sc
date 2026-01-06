# Experiments Guide

A comprehensive guide to all ML experiments for statechart generation.

---

## Quick Reference

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| [Grammar Ceiling](#grammar-constrained-ceiling) | 100% | Constraint → perfect validity |
| [Model Scaling](#model-scaling) | 100%/0%/100% | Repair scales, complete stable |
| [SC-to-Code](#sc-to-code) | 100% | Few-shot achieves perfect |
| [History States](#history-states) | 100% | H and H* both work |
| [Guard Synthesis](#guard-synthesis) | 80% | L3 comparison is weak (20%) |
| [Entry/Exit Actions](#entryexit-actions) | 100%/50% | Entry >> Exit |
| [Two-Phase Gen](#two-phase-generation) | 100%/67% | Validity perfect, planning harder |
| [Goal Traces](#goal-directed-traces) | 100% | Optimal paths found |
| [Trace Induction](#trace-induction) | 20% | 0.5B copies vs generalizes |
| [Dynamic Loading](#dynamic-sampler-loading) | ~0ms | Near-zero overhead |

---

## Phase 5: Foundation Experiments

### Grammar Infrastructure

**Location:** `grammars/`

Core constrained decoding infrastructure:

| File | Purpose |
|------|---------|
| `sc_json_grammar.statechart.json` | SC JSON syntax as statechart |
| `sc_validity_grammar.statechart.json` | Semantic validity rules |
| `trace_constraint_grammar.statechart.json` | Execution trace constraints |
| `dynamic_constrained_sampler.py` | Retok-TopK implementation |
| `meta_constrained_sampler.py` | Multi-SC switching |

**Run:**
```bash
python -c "from grammars.dynamic_constrained_sampler import test_sampler; test_sampler()"
```

---

### exp_hierarchical_constrained_gen

**Location:** `experiments/exp_hierarchical_constrained_gen/`

Tests push/pop SC stack for nested constraints.

**Key files:**
- `__init__.py` - Test cases for hierarchy
- `real_model_benchmark.py` - MLX inference tests

**Result:** 100% (7/7 tests)

**Run:**
```bash
python -m experiments.exp_hierarchical_constrained_gen
```

---

### exp_introspective_generation

**Location:** `experiments/exp_introspective_generation/`

Tests if introspection tokens (`[SC:STATE=X]`, `[SC:VALID=...]`) help generation.

**Key files:**
- `benchmark.py` - Comparison of introspection modes

**Result:** +20% improvement with VALID tokens

**Run:**
```bash
python -m experiments.exp_introspective_generation.benchmark
```

---

### exp_sc_lora_grpo

**Location:** `experiments/exp_sc_lora_grpo/`

Group Relative Policy Optimization with constrained decoding.

**Key files:**
- `grpo_real.py` - Real gradient training
- `grpo_constrained.py` - Combined GRPO + Retok-TopK

**Result:** 83% → 100% with constraints

**Run:**
```bash
python -m experiments.exp_sc_lora_grpo.grpo_constrained
```

---

## Phase 6: Semantic Understanding

### exp_history_states

**Location:** `experiments/exp_history_states/`

Tests shallow (H) and deep (H*) history state semantics.

**Key files:**
- `history_executor.py` - History-aware execution
- `benchmark.py` - 8 test cases

**Result:** 100% (both H and H*)

**Run:**
```bash
python -m experiments.exp_history_states.benchmark
```

---

### exp_guard_synthesis

**Location:** `experiments/exp_guard_synthesis/`

Tests guard expression generation by complexity level.

**Key files:**
- `guard_generator.py` - Complexity-leveled generation
- `benchmark.py` - L1-L4 evaluation

**Result:** 80% overall (L3 comparison weak at 20%)

**Run:**
```bash
python -m experiments.exp_guard_synthesis.benchmark
```

---

### exp_sc_to_code

**Location:** `experiments/exp_sc_to_code/`

Statechart to Python code generation.

**Key files:**
- `code_generator.py` - Few-shot prompting
- `code_validator.py` - Syntax/runtime/behavior checks
- `benchmark.py` - 5 SC test cases

**Result:** 100% (syntax, runnable, correct)

**Run:**
```bash
python -m experiments.exp_sc_to_code.benchmark
```

---

### exp_3b_model_scaling

**Location:** `experiments/exp_3b_model_scaling/`

Compares 0.5B, 1.5B, 3B models on SC tasks.

**Key files:**
- `scaling_benchmark.py` - Multi-model evaluation
- `results.json` - Detailed results

**Result:** Repair 20%→80%→100%, Debug anomaly at 3B

**Run:**
```bash
python -m experiments.exp_3b_model_scaling.scaling_benchmark
```

---

### exp_entry_exit_actions

**Location:** `experiments/exp_entry_exit_actions/`

Tests entry/exit action generation.

**Key files:**
- `action_grammar.py` - Action grammar definition
- `benchmark.py` - Scenario evaluation

**Result:** Entry 100%, Exit 50%, Both 33%

**Run:**
```bash
python -m experiments.exp_entry_exit_actions.benchmark
```

---

### exp_trace_to_sc

**Location:** `experiments/exp_trace_to_sc/`

Induces statecharts from execution traces.

**Key files:**
- `inducer.py` - Trace-to-SC induction
- `benchmark.py` - Pattern evaluation

**Result:** 20% (0.5B copies vs generalizes)

**Run:**
```bash
python -m experiments.exp_trace_to_sc.benchmark
```

---

## Phase 7: Dynamic Generation

### exp_complete_sc_grammar

**Location:** `experiments/exp_complete_sc_grammar/`

Complete SC JSON grammar covering full proto schema.

**Key files:**
- `full_grammar.json` - Complete grammar SC
- `grammar_stats.py` - Coverage metrics
- `sampler_config.py` - Sampler configuration

**Result:** 12 states, 20 transitions, full coverage

---

### exp_grammar_constrained_ceiling

**Location:** `experiments/exp_grammar_constrained_ceiling/`

Establishes performance ceiling with full grammar.

**Key files:**
- `constrained_sampler.py` - Full grammar sampler
- `benchmark.py` - 100-prompt evaluation

**Result:** No constraint 0%, Full grammar 100%

---

### exp_sc_then_trace

**Location:** `experiments/exp_sc_then_trace/`

Two-phase: generate SC, then generate traces.

**Key files:**
- `two_phase_generator.py` - Phase switching
- `trace_validator.py` - Trace validation
- `benchmark.py` - End-to-end evaluation

**Result:** SC 100%, Trace 100%, Goal 67%

---

### exp_dynamic_sampler_loading

**Location:** `experiments/exp_dynamic_sampler_loading/`

Runtime SC loading for dynamic constraints.

**Key files:**
- `dynamic_sampler.py` - Loading implementation
- `sc_executor.py` - Lightweight execution
- `benchmark.py` - Timing evaluation

**Result:** Load 0.02ms, Switch ~0ms

---

### exp_goal_directed_traces

**Location:** `experiments/exp_goal_directed_traces/`

Generate traces to reach specified goals.

**Key files:**
- `goal_parser.py` - Goal specification parsing
- `path_finder.py` - BFS baseline
- `benchmark.py` - Complexity evaluation

**Result:** 100% reachability, 100% optimality

---

## Running All Experiments

### Quick Validation
```bash
# Run all Phase 5 experiments
for exp in exp_hierarchical_constrained_gen exp_introspective_generation exp_sc_lora_grpo; do
    python -m experiments.$exp 2>&1 | tail -5
done
```

### Full Benchmark Suite
```bash
# Run comprehensive benchmarks
python -m experiments.exp_grammar_constrained_ceiling.benchmark
python -m experiments.exp_3b_model_scaling.scaling_benchmark
python -m experiments.exp_sc_to_code.benchmark
```

### With Real MLX Inference
```bash
# Ensure MLX is available
python -c "import mlx; print('MLX available')"

# Run real model experiments
PYTHONUNBUFFERED=1 python -m experiments.exp_guard_synthesis.benchmark
```

---

## Experiment Structure

Each experiment follows this structure:

```
experiments/exp_<name>/
├── __init__.py          # Module exports, docstring
├── benchmark.py         # Main evaluation script
├── <core_logic>.py      # Implementation
├── NOTES.md            # Design notes, findings
└── results/            # Output artifacts (optional)
    └── *.json
```

### Adding New Experiments

1. Create directory: `experiments/exp_<name>/`
2. Add `__init__.py` with docstring and exports
3. Implement core logic in separate file
4. Create `benchmark.py` with `test_*` functions
5. Document in this guide

---

## Key Insights Summary

### What Works

| Technique | Effectiveness | Use Case |
|-----------|---------------|----------|
| Grammar constraints | 100% | Syntax validity |
| Few-shot prompting | 100% | Code generation |
| GRPO + constraints | 100% | Training + inference |
| Dynamic loading | ~0ms | Runtime switching |

### What's Hard

| Task | Challenge | Mitigation |
|------|-----------|------------|
| Comparison guards | 20% accuracy | Training data augmentation |
| Exit actions | 50% accuracy | Balanced examples |
| Trace induction | 20% accuracy | Larger models |
| Debug at scale | Prompt sensitivity | Prompt engineering |

### Scaling Laws

| Task | 0.5B | 1.5B | 3B | Pattern |
|------|------|------|-----|---------|
| Repair | 20% | 80% | 100% | Strong scaling |
| Complete | 100% | 100% | 100% | Constraint-dominated |
| Induction | 20% | ? | ? | Needs investigation |

---

## References

- Paper drafts: `papers/01_*.md` through `papers/08_*.md`
- Results summary: `RESULTS.md`
- Proto schema: `../proto/statecharts/v1/statecharts.proto`
