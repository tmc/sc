"""
exp_trm_sc_refinement: TRM-style iterative statechart refinement.

Uses TinyRecursiveModels' H×L iteration dynamics to transform
partial/invalid statecharts into valid ones.
"""

from .sc_refiner import (
    SCRefiner,
    SCRefinerConfig,
    refine_statechart,
)
from .refinement_loop import (
    RefinementLoop,
    RefinementStep,
    run_refinement,
)
from .validity_tracker import (
    ValidityTracker,
    ValidityMetrics,
    track_validity,
)
from .benchmark import (
    RefinementBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Core refiner
    "SCRefiner",
    "SCRefinerConfig",
    "refine_statechart",
    # Refinement loop
    "RefinementLoop",
    "RefinementStep",
    "run_refinement",
    # Validity tracking
    "ValidityTracker",
    "ValidityMetrics",
    "track_validity",
    # Benchmark
    "RefinementBenchmark",
    "BenchmarkResult",
    "run_benchmark",
]
