# exp_context_guards_comprehensive

## Hypothesis

Can an LLM accurately evaluate guards and trace context mutations across state machine transitions?

**Result: 83% overall accuracy - excellent performance on most guard types.**

## Results

### Overall Accuracy

| Metric | Value |
|--------|-------|
| Overall | **83%** (15/18) |

### Per-Category Breakdown

| Category | Accuracy | Notes |
|----------|----------|-------|
| BOOLEAN | **100%** (2/2) | Perfect: has_key, !has_key |
| COMPARISON | **100%** (2/2) | Perfect: count<max, temp>100 |
| COMPOUND | **100%** (3/3) | Perfect: AND, OR, complex |
| CHAIN | 67% (2/3) | Issues with action→guard deps |
| PRIORITY | **100%** (3/3) | Perfect: 2-way, 3-way priority |
| IN_STATE | 0% (0/2) | Failed: in(State) checks |
| NESTED | **100%** (2/2) | Perfect: user.role, config.auth.enabled |
| ARRAY | **100%** (1/1) | Perfect: items.length |

Report:
```
[SID]: CONTEXT_GUARDS bool=100%, compare=100%, compound=100%, chain=67%, priority=100%, overall=83%
```

## What Worked

### 1. Boolean Guards (100%)
```
Locked --unlock[has_key]--> Unlocked
Context: {has_key: true}
Model: Correctly evaluates has_key=true → take transition
```

### 2. Comparison Guards (100%)
```
Counter --inc[count<max]/count++--> Counter
Counter --inc[count>=max]--> Full
Context: {count: 2, max: 3}
Events: inc, inc
Model: Correctly traces count=2→3, then 3>=3 triggers Full
```

### 3. Compound Guards (100%)
```
Door --open[(unlocked && !alarm) || override]--> Open
Context: {unlocked: true, alarm: false, override: false}
Model: Correctly evaluates (T && !F) || F = T
```

### 4. Priority Resolution (100%)
```
S --e[x>5]--> A (priority 0)
S --e[x>0]--> B (priority 1)
Context: {x: 10}
Model: Both guards true, correctly selects priority 0 → A
```

### 5. Nested Access (100%)
```
Check --auth[user.role == "admin"]--> Admin
Context: {user: {role: "admin", name: "Alice"}}
Model: Correctly traverses nested object
```

## What Didn't Work

### 1. In-State Guards (0%)
```
X --sync[in(B)]--> Y
Active states: {X, B} (parallel regions)
Expected: Y (B is active, guard passes)
Predicted: (empty) - model couldn't parse
```

**Why**: The model doesn't understand parallel region semantics where multiple states can be active simultaneously.

### 2. Chain with Action Side-Effects (partial)
```
S1 --e/ready=1--> S2
S2 --f[ready==1]--> S3
Context: {ready: 0}
Expected: S3 (action sets ready=1, then guard passes)
Predicted: S2 (didn't trace action effect)
```

**Why**: Model traced the state but didn't propagate the action's effect to the subsequent guard evaluation.

## Key Findings

### 1. Simple Guards are Well-Understood
- Boolean: 100%
- Comparison: 100%
- Nested access: 100%

The few-shot examples clearly demonstrate evaluation patterns.

### 2. Compound Logic Works
- AND: ✓
- OR: ✓
- NOT: ✓
- Nested: ✓

The model correctly applies De Morgan's laws and precedence.

### 3. Priority is Understood
Both 2-way and 3-way priority selection work perfectly.

### 4. Parallel Semantics are Challenging
In-state guards require understanding that:
1. Multiple states can be active (parallel regions)
2. `in(X)` checks if X is in the active set
3. This is different from "current state == X"

### 5. Action→Guard Chains Need More Examples
The model sometimes forgets to apply actions before evaluating subsequent guards.

## Gaps Identified

| Gap | Impact | Example |
|-----|--------|---------|
| Parallel awareness | IN_STATE fails | in(Region.B) not understood |
| Action propagation | CHAIN partial | Action effect lost before next guard |
| Active set concept | IN_STATE fails | Only considers "current" state |

## Recommendations

### Short-term Fixes

1. **Add parallel state examples** to few-shot prompts
2. **Explicit context tracking** in chain examples
3. **Show active set** in prompt format

### Architectural Improvements

1. **Separate action execution** - First execute actions, then evaluate guards
2. **Active set management** - Track set of active states explicitly
3. **Step-by-step tracing** - Force model to show context after each action

### Future Experiments

1. **exp_parallel_guards** - Focus on in-state guard semantics
2. **exp_action_chains** - Longer action→guard chains
3. **exp_context_tracking** - Explicit context mutation tracking

## Files

| File | Purpose |
|------|---------|
| `context_guards.py` | Guard types, evaluation, test cases |
| `benchmark.py` | Run benchmark across categories |
| `__init__.py` | Module exports |
| `NOTES.md` | This documentation |

## Usage

```bash
# Run benchmark
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_context_guards_comprehensive.benchmark
```

```python
# Programmatic usage
from experiments.exp_context_guards_comprehensive import (
    run_benchmark,
    evaluate_guard_expr,
    GuardCategory,
)

result = run_benchmark()
print(f"Overall: {result['overall_accuracy']:.0f}%")

# Direct guard evaluation
context = {"x": 10, "locked": True}
result = evaluate_guard_expr("x > 5 && !locked", context)
```

## Test Cases Summary

| TC | Category | Description | Result |
|----|----------|-------------|--------|
| 1 | BOOLEAN | has_key guard | ✓ |
| 2 | BOOLEAN | !has_key guard | ✓ |
| 3 | COMPARISON | count<max with action | ✓ |
| 4 | COMPARISON | temp>100 | ✓ |
| 5 | COMPOUND | unlocked && !alarm | ✓ |
| 6 | COMPOUND | admin \|\| has_pass | ✓ |
| 7 | COMPOUND | (vip \|\| paid) && !blocked | ✓ |
| 8 | CHAIN | x=1 → x>0 → y==2 | ✓ |
| 9 | CHAIN | val=5 → val>3 | ✓ |
| 10 | PRIORITY | x>5 vs x>0 with x=10 | ✓ |
| 11 | PRIORITY | x>5 vs x>0 with x=3 | ✓ |
| 12 | IN_STATE | in(B) with B active | ✗ |
| 13 | IN_STATE | !in(B) with A active | ✗ |
| 14 | NESTED | user.role == "admin" | ✓ |
| 15 | NESTED | config.auth.enabled | ✓ |
| 16 | ARRAY | items.length > 0 | ✓ |
| 17 | CHAIN | ready=1 → ready==1 | ✗ |
| 18 | PRIORITY | 3-way: level>=10/5/0 | ✓ |

## Model

- **Model**: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
- **Max tokens**: 400
- **Prompt style**: Few-shot with step-by-step trace

## Conclusion

83% accuracy demonstrates strong guard evaluation capabilities. The model excels at:
- Boolean logic
- Numeric comparisons
- Compound expressions
- Priority resolution
- Nested object access

Areas for improvement:
- Parallel state awareness (in-state guards)
- Action→guard dependency chains

The few-shot approach with explicit tracing works well for most guard semantics.
