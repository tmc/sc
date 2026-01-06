"""
exp_trm_steering_hybrid: TRM Refinement with Steering

Combines Test-time Refinement Method (TRM) with activation steering
for high-validity statechart generation.

TWO-STAGE APPROACH:
1. First pass: Steered generation with validity-oriented direction
2. Refinement: TRM H×L iteration with steering-guided corrections

H×L PATTERN:
- H (Hierarchy): Root → Compound → Leaf
- L (Layers): Early (structural) → Middle (semantic) → Late (output)

TARGET: 60% → 99% validity

Key Components:
- HybridRefiner: Combined TRM + steering refinement
- SteeringGuidedIterator: H×L iteration with steering
- HybridBenchmark: Compare methods for 60%→99% goal
"""

from .hybrid_refiner import (
    RefinementPhase,
    SteeringConfig,
    TRMConfig,
    HybridConfig,
    RefinementStep,
    RefinementResult,
    ValidityChecker,
    SteeringGenerator,
    TRMRefiner,
    HybridRefiner,
    create_hybrid_refiner,
)

from .steering_guided_iteration import (
    HierarchyLevel,
    LayerGroup,
    SteeringAspect,
    SteeringDirection,
    IterationState,
    IterationResult,
    GuidedIterationConfig,
    SteeringVectorCache,
    AspectAnalyzer,
    SteeringGuidedIterator,
)

from .benchmark import (
    BenchmarkCase,
    MethodResult,
    MethodSummary,
    BaselineMethod,
    TRMOnlyMethod,
    SteeringOnlyMethod,
    HybridMethod,
    HybridPlusMethod,
    HybridBenchmark,
    BENCHMARK_CASES,
    demo,
)

__all__ = [
    # Hybrid refiner
    "RefinementPhase",
    "SteeringConfig",
    "TRMConfig",
    "HybridConfig",
    "RefinementStep",
    "RefinementResult",
    "ValidityChecker",
    "SteeringGenerator",
    "TRMRefiner",
    "HybridRefiner",
    "create_hybrid_refiner",
    # Steering-guided iteration
    "HierarchyLevel",
    "LayerGroup",
    "SteeringAspect",
    "SteeringDirection",
    "IterationState",
    "IterationResult",
    "GuidedIterationConfig",
    "SteeringVectorCache",
    "AspectAnalyzer",
    "SteeringGuidedIterator",
    # Benchmark
    "BenchmarkCase",
    "MethodResult",
    "MethodSummary",
    "BaselineMethod",
    "TRMOnlyMethod",
    "SteeringOnlyMethod",
    "HybridMethod",
    "HybridPlusMethod",
    "HybridBenchmark",
    "BENCHMARK_CASES",
    "demo",
]
