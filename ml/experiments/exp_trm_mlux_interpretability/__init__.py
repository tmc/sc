"""
TRM-mlux Interpretability: Analyze TRM reasoning iterations with mlux.

Uses mlux hooks and activation caching to understand how the TRM-style
Sudoku statechart model reasons through iterative refinement.

Key Analyses:
1. Iteration Dynamics: How activations evolve across H×L cycles
2. Representation Tracking: What the model "thinks" at each step
3. Solution Correlation: Attention patterns for correct vs incorrect

Integration:
- Uses exp_trm_sudoku_9x9 model
- Applies mlux-style hooks and caching
- Generates interpretability reports

Sources:
- mlux library: https://github.com/batson/mlux
- MLX framework: https://github.com/ml-explore/mlx
"""

from .iteration_analyzer import (
    IterationAnalyzer,
    IterationSnapshot,
    IterationTrajectory,
    analyze_iterations,
)
from .activation_tracker import (
    ActivationTracker,
    ActivationHook,
    track_activations,
)
from .solution_correlator import (
    SolutionCorrelator,
    CorrelationResult,
    correlate_solutions,
)
from .benchmark import (
    run_interpretability_benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    "IterationAnalyzer",
    "IterationSnapshot",
    "IterationTrajectory",
    "analyze_iterations",
    "ActivationTracker",
    "ActivationHook",
    "track_activations",
    "SolutionCorrelator",
    "CorrelationResult",
    "correlate_solutions",
    "run_interpretability_benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
]
