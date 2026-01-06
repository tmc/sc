# exp_action_composition: Research Notes

## Key Insight

Actions form an algebra under sequential composition. The algebraic properties (commutativity, idempotence, inverses) can be **discovered through evolution** rather than hardcoded.

## Discovered Properties

From demo with 7 actions on 2 variables:

| Property | Examples |
|----------|----------|
| **Inverse pairs** | `Inc_x ; Dec_x = I` |
| **Self-inverse** | `Toggle_y ; Toggle_y = I` |
| **Commutative** | `Inc_x ; Inc_y = Inc_y ; Inc_x` (different variables) |
| **Idempotent** | `Set_x_0 ; Set_x_0 = Set_x_0` |
| **NOT commutative** | `Set_x_0 ; Inc_x ≠ Inc_x ; Set_x_0` |

## What Worked

1. **Effect-based action model** - Actions as lists of variable effects enables symbolic reasoning about composition.

2. **Test-based property verification** - Running actions on diverse test contexts to verify algebraic claims. 95%+ match threshold for property acceptance.

3. **Evolutionary hypothesis search** - Rather than exhaustively testing all pairs, evolve hypotheses about which pairs have which properties.

4. **Composition rules as rewrite patterns** - `("A", "A") -> "A"` for idempotent, `("A", "B") -> "I"` for inverse pairs.

## What Didn't Work

1. **Symbolic effect composition** - Tried to compute composed effects symbolically but got complicated with conditionals. Empirical testing is simpler.

2. **Deep sequence learning** - Tried to learn rules for sequences > 3, but combinatorial explosion. Focus on pairs/triples.

3. **Gradient-based learning** - Discrete algebraic properties don't lend themselves to gradients. Evolution works better.

## Algebraic Laws

```
Identity:      A;I = I;A = A           (always holds)
Associativity: (A;B);C = A;(B;C)       (always holds for functions)
Commutativity: A;B = B;A               (rare - only when disjoint vars)
Idempotence:   A;A = A                 (SET operations)
Inverse:       A;B = I                 (Inc/Dec pairs)
Self-inverse:  A;A = I                 (Toggle)
Absorption:    A;B;A = A               (SET absorbs everything before)
Nilpotent:     A^n = I                 (cycles)
```

## Future Directions

1. **Conditional composition** - `A;B = C if guard else D`
2. **Partial commutativity** - Actions commute on some contexts but not others
3. **Composition types** - Learn type signatures for actions (what they need/produce)
4. **Monoid structure** - Formally verify that actions form a monoid
5. **Category theory** - Actions as morphisms, states as objects

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_action_side_effects` | Foundation - action effects model |
| `exp_guard_synthesis` | Guards depend on action effects |
| `exp_transition_priorities` | Composition order affects priority |
| `exp_topology_evolution` | Topology determines valid compositions |
| `exp_rule_discovery` | Discovered rules imply composition |

## Applications

1. **State machine minimization** - Replace `A;A` with `A` if idempotent
2. **Dead code elimination** - `A;A^-1` cancels out
3. **Reordering for optimization** - Commutative pairs can be reordered
4. **Invariant discovery** - Find action sequences that preserve state

## Code Patterns

```python
# Compose two actions
composed = compose(action_a, action_b)

# Test if A;B = B;A
comm_score = learner._test_commutativity(a, b)

# Simplify sequence using learned rules
simplified = learner.simplify([a, a, b, b])  # -> [a, b] if idempotent
```

## Pure MLX
No numpy dependencies. All operations use:
- MLX arrays for grid-based contexts
- Python dicts for variable contexts
- Python stdlib for evolution

## Date
2024-01-04
