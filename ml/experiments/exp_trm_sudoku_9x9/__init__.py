"""
Statechart-based 9x9 Sudoku Solver with TRM-style Iterative Refinement

Combines:
- Parallel cell statecharts (81 cells, 10 states each)
- Constraint guards (row/col/box validity)
- TRM-style two-level iteration (H_cycles × L_cycles)
- Stablemax cross-entropy loss

Target: Match or exceed TRM's 87% exact accuracy on Sudoku-Extreme
"""

from .sudoku_statechart_9x9 import (
    SudokuCell9x9,
    SudokuConstraintGuard9x9,
    SudokuStatechart9x9,
)
from .data_loader import SudokuExtremeDataset, load_sudoku_extreme
from .losses import stablemax_cross_entropy

__all__ = [
    "SudokuCell9x9",
    "SudokuConstraintGuard9x9",
    "SudokuStatechart9x9",
    "SudokuExtremeDataset",
    "load_sudoku_extreme",
    "stablemax_cross_entropy",
]
