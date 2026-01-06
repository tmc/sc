# exp_trm_vs_sc_sudoku: Vanilla TRM vs SC-enabled Sudoku Solvers

## Research Question

**Can explicit statechart structure improve upon TinyRecursiveModels' 87% Sudoku accuracy?**

Hypothesis: SC-augmented models will:
1. Learn faster (fewer samples to accuracy threshold)
2. Achieve higher accuracy (explicit constraints)
3. Generalize better (structure as inductive bias)
4. Produce more interpretable solutions

## Architecture Comparison

### Vanilla TRM (Baseline)
```
Input Puzzle → Cell Embeddings → [H×L Transformer Iterations] → Digit Predictions
                                          │
                                   Learned everything
```

**Characteristics:**
- Pure learned iteration dynamics
- No explicit constraints
- All structure must be learned from data
- Matches Samsung SAIL Montreal's approach

### SC-TRM Hybrid (Proposed)
```
Input Puzzle → Cell Embeddings → [H×L Iterations with Guards] → Masked Predictions
                    │                      │
             Constraint Guards ────────────┘
                    │
            Row/Col/Box checks
```

**Key Additions:**
1. **Constraint Guards**: Differentiable row/col/box constraint checking
2. **Guard-Masked Predictions**: Can't predict invalid digits
3. **Violation Loss**: Auxiliary loss penalizing constraint violations
4. **Statechart States**: Empty → Candidate[d] → Fixed[d] transitions

## Theoretical Advantages of SC Structure

### 1. Reduced Search Space
- Vanilla: 9^81 possible outputs
- SC-TRM: Only constraint-satisfying outputs (much smaller)

### 2. Built-in Inductive Bias
- Vanilla: Must learn Sudoku rules from examples
- SC-TRM: Rules encoded in architecture

### 3. Guaranteed Validity
- Vanilla: May produce invalid solutions
- SC-TRM: Guards prevent invalid predictions

### 4. Interpretability
- Vanilla: Black-box iterations
- SC-TRM: Can inspect guard activations, see which constraints are satisfied

## Experiment Design

### Fair Comparison Protocol

| Aspect | Vanilla TRM | SC-TRM Hybrid |
|--------|-------------|---------------|
| Hidden dim | 128 | 128 |
| Layers | 3 | 3 |
| Heads | 4 | 4 |
| H×L cycles | 3×6 | 3×6 |
| Training data | Same | Same |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 |

Only difference: SC structure (guards, masks, violation loss)

### Metrics

| Metric | Description |
|--------|-------------|
| Exact Accuracy | All 81 cells correct |
| Cell Accuracy | Fraction of correct cells |
| Constraint Satisfaction | Fraction of valid cells |
| Epochs to 50% | Learning speed |
| Epochs to 80% | Learning speed |
| Inference Time | Speed comparison |

### Dataset

For rigorous comparison, use TRM's original dataset:
- sapientinc/sudoku-extreme from HuggingFace
- ~1M puzzles with difficulty ratings
- Augmentation: digit permutation, transpose, shuffle

## Expected Results

### Optimistic Scenario (SC wins)
```
                Vanilla    SC-TRM    Δ
Exact Acc       87%        92%      +5%
Learning Speed  100%       60%      40% faster
Constraint Sat  95%        100%     +5%
```

### Pessimistic Scenario (similar)
```
                Vanilla    SC-TRM    Δ
Exact Acc       87%        88%      +1%
Learning Speed  100%       90%      10% faster
Constraint Sat  95%        99%      +4%
```

### Why SC Might Not Help
- Transformer already learns constraints implicitly
- Guards add computational overhead
- Hard constraints may reduce exploration

## Ablation Studies

1. **Guard ablation**: SC-TRM without guards (just structure)
2. **Soft vs hard guards**: Temperature sweep
3. **Violation loss weight**: 0.0, 0.1, 1.0, 10.0
4. **Iteration ablation**: Effect of H vs L cycles

## Implementation Details

### Constraint Guard
```python
def guard(board_probs, cell_idx):
    # Get cells in same row/col/box
    constraint_cells = get_constraint_cells(cell_idx)

    # Check which digits are "taken"
    taken = max(board_probs[constraint_cells, :], dim=0)

    # Valid = 1 - taken (soft)
    return sigmoid((0.5 - taken) / temperature)
```

### Guard-Masked Prediction
```python
logits = output_head(h)
guards = compute_all_guards(current_probs)
masked_logits = logits + log(guards + epsilon)
probs = softmax(masked_logits)
```

### Violation Loss
```python
violation = mean(probs * (1 - guards))
loss = ce_loss + lambda * violation
```

## Files

| File | Description |
|------|-------------|
| `vanilla_trm.py` | Faithful TRM implementation |
| `sc_trm_hybrid.py` | SC-augmented TRM |
| `comparison_benchmark.py` | Fair comparison framework |
| `NOTES.md` | This document |

## Usage

```bash
# Run full comparison
python -m ml.experiments.exp_trm_vs_sc_sudoku.comparison_benchmark

# Test individual models
python -m ml.experiments.exp_trm_vs_sc_sudoku.vanilla_trm
python -m ml.experiments.exp_trm_vs_sc_sudoku.sc_trm_hybrid
```

## Integration

```python
from ml.experiments.exp_trm_vs_sc_sudoku import (
    VanillaTRM,
    SCTRMHybrid,
    run_comparison,
)

# Run comparison
result = run_comparison()
print(f"SC improvement: {result.accuracy_improvement:+.2%}")
```

## Related Work

- **TinyRecursiveModels**: Samsung SAIL Montreal, 87% on Sudoku-Extreme
- **AlphaZero**: Learned constraints for game playing
- **Neural Constraint Satisfaction**: DL for CSP
- **Differentiable Logic**: Neuro-symbolic approaches

## Next Steps

1. Run on full Sudoku-Extreme dataset
2. Ablation studies on guard temperature
3. Compare interpretability (SAE on both)
4. Test on harder variants (16×16, Killer Sudoku)
5. Apply SC structure to other CSPs (N-Queens, Graph Coloring)
