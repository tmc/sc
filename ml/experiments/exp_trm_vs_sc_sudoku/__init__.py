"""
exp_trm_vs_sc_sudoku: Compare vanilla TRM vs SC-enabled Sudoku solvers.

Hypothesis: SC-augmented models learn faster and achieve higher accuracy
by leveraging explicit constraint structure.
"""

from .vanilla_trm import (
    VanillaTRM,
    VanillaTRMConfig,
)
from .sc_trm_hybrid import (
    SCTRMHybrid,
    SCTRMConfig,
    ConstraintGuard,
)
from .comparison_benchmark import (
    ComparisonBenchmark,
    ComparisonResult,
    run_comparison,
)

__all__ = [
    # Vanilla TRM
    "VanillaTRM",
    "VanillaTRMConfig",
    # SC-TRM Hybrid
    "SCTRMHybrid",
    "SCTRMConfig",
    "ConstraintGuard",
    # Benchmark
    "ComparisonBenchmark",
    "ComparisonResult",
    "run_comparison",
]
