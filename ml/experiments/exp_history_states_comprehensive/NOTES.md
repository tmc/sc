# exp_history_states_comprehensive

## Hypothesis

Can an LLM (Qwen2.5-Coder-1.5B) correctly apply shallow (H) vs deep (H*) history semantics?

## Results

| Category | Accuracy | Jaccard |
|----------|----------|---------|
| Deep (H*) | **100%** | 100% |
| Shallow (H) | **0%** | 53% |
| No prior | **0%** | 0% |
| Parallel | **50%** | 62% |
| **Overall** | **50%** | 68% |

## Key Findings

**What Worked:**
1. **Deep history perfect (100%)** - LLM understands "restore exact configuration"
   - TC2: Power.On.High → restore Power.On.High ✓
   - TC5: Multiple cycles still restores exactly ✓
   - TC6: 3-level nesting restored correctly ✓
   - TC10: Nested with deep → exact ✓

2. **Parallel regions work with deep history**
   - TC7: {Movement.Running, Combat.Fighting} restored exactly ✓

**What Didn't Work:**
1. **Shallow history completely failed (0%)**
   - Model doesn't understand "restore immediate child, cascade to defaults"
   - Always returns the full prior configuration (treats H like H*)
   - TC1: Expected Low (default), got High (prior)
   - TC3: Expected C (default), got D (prior)

2. **No prior history failed (0%)**
   - Model outputs "None" when no history exists
   - Should use defaults throughout

3. **Shallow parallel failed**
   - TC8: Should restore {Walking, Idle} (defaults), got {Running, Fighting}

## Analysis

The LLM has a **systematic misconception** about shallow history:

```
Model's (incorrect) understanding:
  Shallow H = restore whatever was active before

Correct semantics:
  Shallow H = restore ONLY immediate child, nested states use DEFAULTS
```

This is a **semantic boundary issue** - the model doesn't understand that shallow history has **limited depth**.

## Shallow vs Deep Comparison

| Scenario | Shallow H (expected) | Deep H* (expected) | Model predicts |
|----------|---------------------|-------------------|----------------|
| Was in A.B.D | A.B.C (C=default) | A.B.D | A.B.D (wrong for shallow) |
| Was in On.High | On.Low (Low=default) | On.High | On.High (wrong for shallow) |

## Recommendations

1. **Explicit cascade prompting**: Show step-by-step "restore B, then enter B's default"
2. **Contrast training**: Show H and H* side-by-side for same scenario
3. **Rule-based post-processing**: If H, force nested states to defaults programmatically
4. **Larger model**: May better distinguish the semantic nuance

## Report to Orchestrator

```
[9D1B]: HISTORY_STATES shallow=0%, deep=100%, no_prior=0%, parallel=50%, overall=50%

Key insight: Deep history PERFECT (100%), Shallow FAILS (0%)
- Model understands "restore exact" but NOT "restore then cascade to defaults"
- Systematic misconception: treats H like H*
- Parallel deep works (100%), parallel shallow fails (0%)

Recommendation: Rule-based post-processing for shallow history
```

## History Semantics Reference

### Shallow History (H)
- Remembers ONLY the immediate child that was active
- Does NOT remember nested substates
- On re-entry via H: restore immediate child, then enter its DEFAULT substate

### Deep History (H*)
- Remembers the FULL configuration (all nested active states)
- On re-entry via H*: restore EXACT configuration

### No History
- Always enter default substate at each level

## Files

- `__init__.py` - Test cases and types
- `history_predictor.py` - LLM-based prediction
- `benchmark.py` - Full evaluation
- `NOTES.md` - Documentation

## Usage

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_history_states_comprehensive.benchmark
```
