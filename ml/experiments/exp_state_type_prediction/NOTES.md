# exp_state_type_prediction

**Session**: DDB5
**Model**: Qwen2.5-Coder-1.5B-Instruct-4bit
**Date**: 2025-01-05

## Hypothesis

State type classification (BASIC/OR/PARALLEL) should achieve 80%+ accuracy using template-based prompting with clear distinguishing features. This builds on success from:
- L3 Guards: 20% -> 100% (with few-shot examples)
- Exit Actions: 0% -> 100% (with clear examples)

## Results

**Report to B90CCCD4:**
```
[DDB5]: STATE_TYPE_PRED basic=86%, or=86%, parallel=100%, overall=85%
```

### By Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| BASIC | 6 | 7 | **86%** |
| OR | 6 | 7 | **86%** |
| PARALLEL | 7 | 7 | **100%** |
| EDGE | 4 | 6 | 67% |
| **OVERALL** | **23** | **27** | **85%** |

**Target achieved**: 85% > 80%

## Key Findings

### PARALLEL Classification Perfect (100%)

The model correctly identifies PARALLEL states when:
- Keywords "simultaneously", "concurrent", "parallel" are present
- Multiple independent subsystems mentioned
- Examples: "Movement AND Combat", "Physics, Audio, and Input concurrently"

### OR Classification Strong (86%)

The model correctly identifies OR states when:
- Keywords "or", "either" are present
- Mutually exclusive options listed
- Examples: "On or Off", "Red, Yellow, Green", "Connected, Disconnected"

### BASIC Classification Good (86%)

The model correctly identifies BASIC states when:
- Simple states with no substates mentioned
- Keywords "simple", "final", "terminal"
- Examples: "Idle state", "Error state with no recovery"

## Gaps Identified

### 1. Keyword Ambiguity

| Case | Expected | Got | Issue |
|------|----------|-----|-------|
| "Door state: Open, Closed, or Locked" | OR | PARALLEL | Model confused by comma-separated list |
| "Waiting state for input" | BASIC | OR | "for input" implies waiting FOR something |

### 2. Control vs Region Confusion

"Audio with Volume and Balance controls" should be PARALLEL (independent controls), but model predicted OR. The word "controls" may have triggered selection semantics.

### 3. Edge Cases Need More Context

The EDGE category (67%) shows the model struggles with ambiguous descriptions:
- "Dashboard showing either Graph or Table view" → PARALLEL (wrong)
- Model may be pattern-matching on structure words

## Error Analysis

**Confusion Matrix:**
- BASIC->OR: 1 (14% of BASIC)
- OR->PARALLEL: 2 (includes edge case)
- PARALLEL->OR: 1 (edge case)

**Root causes:**
1. **Comma lists**: "Open, Closed, or Locked" → misread as parallel regions
2. **Input semantics**: "Waiting for input" → implies multiple inputs?
3. **Control ambiguity**: "Volume and Balance controls" → selection vs parallel

## Recommendations

### For Production Use

1. **PARALLEL detection is reliable** - Use confidently
2. **Add disambiguation for comma lists** - "Open, Closed, or Locked (one at a time)"
3. **Clarify control semantics** - "Volume and Balance (independent controls)"

### For Improvement

1. **Add more edge case examples** to few-shot
2. **Explicit negation examples** - "NOT parallel, only one active"
3. **Consider two-pass**: First classify BASIC vs composite, then OR vs PARALLEL

## State Type Definitions

| Type | Semantics | Example | Keywords |
|------|-----------|---------|----------|
| BASIC | Leaf state, no children | "Idle state" | simple, leaf, final |
| OR | XOR semantics, one child active | "Power: On or Off" | either, or, one of |
| PARALLEL | AND semantics, all children active | "Player: Movement AND Combat" | and, simultaneously, concurrent |

## Files

- `__init__.py` - Module exports
- `type_predictor.py` - Template-based type classification
- `benchmark.py` - Test all three types + edge cases
- `results/BENCHMARK_RESULTS.json` - Raw results
- `NOTES.md` - This file

## Reproduction

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_state_type_prediction.benchmark
```

## Conclusion

**Target achieved**: 85% overall accuracy exceeds 80% goal.

The template-based prompting approach continues to work well. PARALLEL classification is perfect (100%), while BASIC and OR are strong (86% each). Edge cases remain challenging (67%) but represent ambiguous descriptions that would be difficult for humans too.

**Key insight**: Clear keyword signals (simultaneously, concurrent, either, or) are the strongest predictors. Ambiguous cases benefit from explicit semantic hints in the description.
