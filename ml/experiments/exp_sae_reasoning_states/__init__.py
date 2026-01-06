"""
SAE-based Reasoning State Discovery for TRM-style Sudoku Solving

Uses Sparse Autoencoders to discover interpretable reasoning stages
in the iterative refinement process of the Sudoku statechart model.

Key Questions:
1. Do TRM iterations correspond to discrete reasoning stages?
2. Can we identify constraint-type-specific features (row/col/box)?
3. Does the model learn a progression: easy cells → hard cells?

Pipeline:
1. Train Sudoku model (exp_trm_sudoku_9x9)
2. Extract activations at each (H, L) iteration
3. Train TopK SAE on concatenated activations
4. Cluster SAE features into reasoning states
5. Map features to constraint types and cell difficulty
"""

from .extract_activations import (
    extract_iteration_activations,
    ActivationDataset,
)
from .train_sae import (
    ReasoningSAE,
    ReasoningSAEConfig,
    train_reasoning_sae,
)
from .analyze_features import (
    analyze_reasoning_features,
    FeatureAnalysis,
)

__all__ = [
    "extract_iteration_activations",
    "ActivationDataset",
    "ReasoningSAE",
    "ReasoningSAEConfig",
    "train_reasoning_sae",
    "analyze_reasoning_features",
    "FeatureAnalysis",
]
