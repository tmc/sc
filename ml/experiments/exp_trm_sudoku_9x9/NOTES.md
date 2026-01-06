# exp_trm_sudoku_9x9: Statechart Sudoku 9x9 with TRM Iteration

## Overview

Combines statechart-based constraint modeling with TinyRecursiveModels (TRM)
iterative refinement for 9x9 Sudoku solving.

## Architecture

### Statechart Structure
- 81 parallel cell statecharts
- Each cell: 10 states (empty + 1-9)
- Constraint guards: row_valid, col_valid, box_valid
- Soft transitions with guard masking

### TRM Iteration Pattern
- H_cycles = 3 (outer macro-steps)
- L_cycles = 6 (inner micro-steps per macro)
- Total: 18 iterations
- Non-autoregressive: all cells update in parallel

### Key Differences from Pure TRM
1. Explicit constraint guards (not just learned)
2. Interpretable state transitions
3. Position-specific context embeddings
4. Optional learned halting

## Data

Dataset: `sapientinc/sudoku-extreme` from HuggingFace
- ~1M puzzles with difficulty ratings
- Augmentation: digit permutation, transpose, row/col band shuffle

## Training

```bash
# Quick test (100 samples, 10 epochs)
python -m ml.experiments.exp_trm_sudoku_9x9.train --max_samples 100 --epochs 10

# Full training
python -m ml.experiments.exp_trm_sudoku_9x9.train --epochs 100 --batch_size 64

# With augmentation
python -m ml.experiments.exp_trm_sudoku_9x9.train --num_augmentations 10 --epochs 50
```

## Evaluation

```bash
# Evaluate checkpoint
python -m ml.experiments.exp_trm_sudoku_9x9.evaluate --checkpoint best_model.safetensors

# Analyze by difficulty
python -m ml.experiments.exp_trm_sudoku_9x9.evaluate --analyze_difficulty
```

## Target Metrics

| Metric | TRM Baseline | Target |
|--------|--------------|--------|
| Exact Accuracy | 87% | ≥87% |
| Cell Accuracy | ~95% | ≥95% |
| Violations | >0% | 0% |

## Key Innovations

1. **Statechart Guards**: Explicit constraint satisfaction via soft guards
2. **Stablemax Loss**: More stable than softmax for iterative refinement
3. **Two-Level Iteration**: Macro (H) for global context, micro (L) for local refinement
4. **Interpretability**: Can trace which constraints influence each cell

## Results

(To be filled after training)

| Config | Exact Acc | Cell Acc | Violations | Notes |
|--------|-----------|----------|------------|-------|
| baseline | | | | |

## Next Steps

1. Run SAE analysis on trained model (exp_sae_reasoning_states)
2. Compare learned vs explicit guards
3. Analyze iteration dynamics
4. Test adaptive halting
