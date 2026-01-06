"""
Statechart Compression Experiment

Compresses statecharts for efficient storage and transfer using:
1. Bisimulation - merge equivalent states
2. Transition sharing - deduplicate patterns
3. Learned compression - autoencoder

Target: 50% size reduction while preserving semantics.
"""

from .bisimulation import (
    Statechart,
    State,
    Transition,
    StateType,
    BisimulationCompressor,
    create_sample_statechart_with_redundancy,
    create_dfa_statechart,
)

from .transition_share import (
    TransitionSharer,
    TransitionPattern,
    TransitionTemplate,
    GuardCanonicalizer,
    ActionFactorizer,
)

from .learned_compress import (
    LearnedCompressor,
    StatechartAutoencoder,
    create_sample_statecharts,
)

from .compression_benchmark import (
    CompressionResult,
    BenchmarkSummary,
    run_benchmark,
    create_benchmark_dataset,
    demo,
)

__all__ = [
    # Core types
    "Statechart",
    "State",
    "Transition",
    "StateType",
    # Bisimulation
    "BisimulationCompressor",
    "create_sample_statechart_with_redundancy",
    "create_dfa_statechart",
    # Transition sharing
    "TransitionSharer",
    "TransitionPattern",
    "TransitionTemplate",
    "GuardCanonicalizer",
    "ActionFactorizer",
    # Learned compression
    "LearnedCompressor",
    "StatechartAutoencoder",
    "create_sample_statecharts",
    # Benchmark
    "CompressionResult",
    "BenchmarkSummary",
    "run_benchmark",
    "create_benchmark_dataset",
    "demo",
]
