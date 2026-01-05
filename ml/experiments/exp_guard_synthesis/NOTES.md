# Program Synthesis for Guard and Action Expressions

## The Problem

Statecharts require:
1. **Guards**: Boolean expressions that gate transitions
2. **Actions**: Side effects executed on transition
3. **Entry/Exit**: Behaviors on state lifecycle

Traditionally these are HAND-CODED. We want them LEARNED.

## Our Approach

```
Natural Language Description
         ↓
    LLM Proposes Candidates
         ↓
    Evolution Refines
         ↓
    Verified Expression
```

### 1. LLM Generation

Given: "Block immediate recapture at same position"
LLM proposes:
```python
"(move_x == last_capture_x) and (move_y == last_capture_y)"
"(move == last_capture) and would_recapture_single"
"is_ko_violation(move)"
```

### 2. Evolutionary Refinement

LLM candidates seed the population. Evolution:
- Tests against positive/negative examples
- Mutates operators, constants, structure
- Crossover combines good subtrees
- Selects for F1 score

### 3. Action Inference

From before/after state pairs, infer actions:
```python
before = {king_moved: False, can_castle: True}
after  = {king_moved: True,  can_castle: False}

# Inferred:
actions = [
    "king_moved := True",
    "can_castle := False",
]
```

## Expression DSL

```
Expr = Const(value)
     | Var(name)
     | BinOp(op, left, right)    # and, or, ==, !=, <, >, <=, >=
     | UnaryOp(op, operand)      # not, -
     | FuncCall(name, args)      # domain functions
     | IfThenElse(cond, then, else)

Action = ActionExpr(target, operation, value)
         # operations: set, increment, decrement, toggle, append
```

## Domain Templates

We provide domain-specific templates as strong priors:

### Go (Ko Rule)
```python
"(move_x == last_capture_x) and (move_y == last_capture_y) and would_capture_single"
```

### Chess (En Passant)
```python
"(en_passant_file == target_file) and (piece_type == PAWN)"
```

### Chess (Castling)
```python
"(not king_moved) and (not rook_moved) and (not in_check) and path_clear"
```

## Key Files

| File | Purpose |
|------|---------|
| `guard_synthesizer.py` | Core synthesis with evolution |
| `llm_guard_gen.py` | LLM-based candidate generation |

## Research Contributions

1. **Guards are LEARNABLE**: Given examples, evolve correct expressions
2. **Actions are INFERABLE**: From state diffs, deduce modifications
3. **LLM + Evolution**: Best of both - creativity + verification
4. **Interpretable Output**: AST expressions, not neural black boxes

## Integration with SAE Statecharts

The synthesized guards/actions complete the picture:

```
SAE discovers:     STATES (feature patterns)
Guard synthesis:   TRANSITION CONDITIONS (when to move)
Action inference:  STATE UPDATES (what changes)
```

Together: fully automatic statechart extraction!

## Example: Learning Ko from Self-Play

1. **Self-play generates games** (some have infinite loops)
2. **Label transitions**:
   - Positive: moves that led to loops
   - Negative: moves that didn't
3. **Synthesize guard**: "block if would_recapture"
4. **Extract rule**: `move_pos != last_capture_pos OR stones_captured != 1`

## Running the Experiments

```bash
cd /Users/tmc/go/src/github.com/tmc/sc/ml

# Guard synthesis demo
.venv/bin/python3 experiments/exp_guard_synthesis/guard_synthesizer.py

# LLM-guided synthesis
.venv/bin/python3 experiments/exp_guard_synthesis/llm_guard_gen.py
```

## TODO

- [ ] Integrate with actual LLM (Claude API)
- [ ] Add symbolic regression for numeric guards
- [ ] Support temporal operators (always, eventually)
- [ ] Learn guards from game replays (imitation)
- [ ] Combine with SAE states for full extraction
