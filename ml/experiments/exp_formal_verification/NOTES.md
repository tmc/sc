# exp_formal_verification - Research Notes

## Core Insight

**Formal verification as fitness component ensures evolved statecharts are not just accurate but CORRECT.**

Traditional ML approach:
1. Generate candidate statecharts
2. Evaluate on task accuracy
3. Select best performers
4. Repeat

Our approach:
1. Generate candidate statecharts
2. Verify correctness properties with SMT
3. REJECT candidates that violate properties
4. Combine accuracy + verification in fitness
5. Select best verified performers

This bridges ML and formal methods.

## Architecture

```
Statechart Definition
        ↓
   SMT Encoder
        ↓
   Z3 Formulas
   (states as booleans,
    transitions as implications)
        ↓
   Property Checker
   (check each property)
        ↓
PropertyResult per property
        ↓
 VerificationFitness
        ↓
 Combined Fitness = α*accuracy + β*verification
        ↓
   Evolution Loop
   (reject invalid, select verified)
```

## Properties Verified

### 1. No Illegal Transitions
- Only declared transitions can fire
- Guards prevent invalid state changes
- Encoded: ∀s1,s2: (s1 → s2) ∈ transitions

### 2. Deadlock Freedom
- Every non-final state has enabled transition
- No stuck configurations
- Encoded: ∀s: ¬final(s) ⟹ ∃t: enabled(t) ∧ src(t)=s

### 3. Liveness
- Target states eventually reachable
- System makes progress
- Encoded: ◇target (bounded model checking)

### 4. Determinism
- At most one transition enabled per state
- No ambiguous behavior
- Encoded: ∀s: |{t: enabled(t) ∧ src(t)=s}| ≤ 1

### 5. Reachability
- All states reachable from initial
- No orphan states
- Encoded: BFS from initial covers all states

## What Worked

### 1. SMT Encoding Pattern
- States as boolean variables: `s_active`, `s_active_next`
- Transitions as implications: `src_active ∧ guard ⟹ tgt_active_next`
- OR constraint: exactly one child active
- AND constraint: all children active

### 2. Property-Based Fitness
- Each property contributes to score
- Properties passed / total = verification score
- Combined: `0.7 * accuracy + 0.3 * verification`

### 3. Rejection Strategy
- Invalid statecharts rejected from population
- Forces evolution toward correctness
- More efficient than just penalizing

### 4. Counterexample Generation
- Z3 provides counterexamples when properties fail
- Useful for debugging and understanding failures
- Can potentially guide mutations

## What Didn't Work

### 1. Full Liveness Verification
- Unbounded liveness requires infinite unrolling
- Solution: Bounded model checking (k steps)
- Trade-off: May miss deep liveness violations

### 2. Complex Guard Parsing
- Full expression parsing is complex
- Solution: Simple expression grammar
- Future: Use proper parser

### 3. High Verification Weight
- Weight > 0.5 biases toward trivial valid solutions
- Sweet spot: 0.2-0.4 verification weight
- Balance accuracy and correctness

## Connections to Other Experiments

### exp_guard_synthesis
- We verify guards are sufficient
- They synthesize guards from data
- Combine: Evolve guards, verify correctness

### exp_action_side_effects
- Actions modify context
- Guards check context
- Verify: Actions don't break invariants

### exp_sae_coverage_synthesis
- SAE extracts states
- We verify state machine properties
- Combine: Extract then verify

### exp_soar_statechart
- SOAR generates programs
- We verify program state machines
- Combine: Generate verified programs

### exp_coverage_prediction
- They predict coverage
- We verify coverage is achievable
- Combine: Predict reachable coverage

## Key Metrics

| Metric | Description |
|--------|-------------|
| Properties Passed | Count of verified properties |
| Verification Score | passed / total |
| Is Valid | All properties pass |
| Verification Time | SMT solving time |
| Rejection Rate | Invalid candidates filtered |

## SMT Encoding Details

### State Variables
```python
s_active = Bool(f"{name}_active_t")      # Active at time t
s_active_next = Bool(f"{name}_active_t+1")  # Active at time t+1
```

### Transition Formula
```python
trans = And(src.active, guard)  # Transition enabled
effect = Implies(trans, tgt.active_next)  # Effect on next state
```

### OR Constraint (XOR semantics)
```python
at_least_one = Or(*[c.active for c in children])
at_most_one = And(*[Not(And(c1.active, c2.active))
                    for c1, c2 in pairs(children)])
constraint = Implies(parent.active, And(at_least_one, at_most_one))
```

### AND Constraint (parallel)
```python
all_active = And(*[c.active for c in children])
constraint = Implies(parent.active, all_active)
```

## Future Directions

### 1. Counterexample-Guided Refinement
- Use counterexamples to guide mutations
- Target mutations at violated properties
- More efficient evolution

### 2. Incremental Verification
- Only re-verify changed parts
- Cache verification results
- Faster evolution loop

### 3. Property Templates
- Domain-specific property templates
- e.g., "never in error state for > k steps"
- Customizable correctness criteria

### 4. Synthesis from Properties
- Given properties, synthesize statechart
- Use CEGIS (counterexample-guided inductive synthesis)
- Stronger guarantees

### 5. Real-Time Properties
- Timing constraints (within k steps)
- TCTL/MTL specifications
- Clock variables in SMT

## Open Questions

1. How to handle probabilistic transitions?
2. Can we verify infinite-state systems approximately?
3. How to integrate with learned guards?
4. Can verification guide SAE feature selection?

## Implementation Notes

### Z3 Fallback
```python
try:
    from z3 import Bool, And, Or, ...
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    # Mock implementation for structure
```

### Timeout Handling
```python
solver.set("timeout", timeout_ms)
result = solver.check()
if result == unknown:
    return PropertyResult(status=UNKNOWN)
```

### Counterexample Extraction
```python
if result == sat:
    model = solver.model()
    counterexample = {str(d): str(model[d]) for d in model.decls()}
```

## References

- Harel, D. "Statecharts: A visual formalism for complex systems" (1987)
- Clarke, E.M. "Model Checking" (1999)
- De Moura, L. "Z3: An Efficient SMT Solver" (2008)
- Alur, R. "Principles of Cyber-Physical Systems" (2015)
