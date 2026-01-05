# Experiment N: Sudoku as Statechart

## Overview
Models 4x4 Sudoku puzzle solving as a differentiable statechart. Each cell placement is a transition with constraint guards.

## Components
- `sudoku_statechart.py`: Full implementation

## Architecture

### SudokuCell
- States: 0 (empty), 1, 2, 3, 4
- Transitions: PLACE_1, PLACE_2, PLACE_3, PLACE_4
- Transition network scores placements based on context
- Guard-masked transitions (only valid placements enabled)

### SudokuConstraintGuard
- Evaluates row/column/box constraints as soft guards
- Learnable constraint networks (can learn Sudoku rules from data)
- Combined guard: row_guard * col_guard * box_guard

### SudokuStatechart
- Parallel composition of 16 cell statecharts
- Board encoder: flattened states -> context vector
- Position embeddings for cell-specific context
- Iterative solving via repeated step() calls

## Results
```
1. Generated puzzles: OK
2. Soft state shape: [2, 16, 5]
3. Statechart step: Output shape [2, 16, 5]
4. State sums: min=1.000, max=1.000 (normalized)
5. Gradient flow: OK
6. Solve (10 steps): Produces discrete boards
```

## Issues Fixed
- MLX `.at[].set()` doesn't exist (unlike JAX): Rewrote generate_sudoku_4x4 using Python lists instead of in-place array updates

## Reproduction Commands
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_n_sudoku/sudoku_statechart.py
```

## Notes
- 4x4 Sudoku chosen for simplicity (vs 9x9)
- Untrained model produces invalid boards
- Would need training loop to learn constraint satisfaction
- Inspired by TinyRecursiveModels Sudoku benchmark approach
