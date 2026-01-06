"""
exp_mlux_sc_circuits: Circuit Discovery for Statechart Validity

Discovers which model components (attention heads, MLPs) are responsible
for different aspects of statechart validity.

APPROACH:
1. Ablate components and measure validity degradation
2. Identify critical layers for: state names, transitions, hierarchy
3. Build circuit graph showing information flow

Key Components:
- ValidityMeasurer: Fine-grained SC validity metrics
- AblationRunner: Systematic component ablation
- CircuitFinder: Circuit identification and graphing
- CircuitBenchmark: Evaluation framework

Circuit Types:
- STATE_NAME_MEMORY: Maintains state name consistency
- TRANSITION_VALIDITY: Ensures valid state references
- HIERARCHY: Maintains parent-child relationships
- STRUCTURAL: Produces valid JSON structure
"""

from .validity_measurer import (
    ValidityMetrics,
    ValidityMeasurer,
    ComponentValidityTests,
)

from .ablation_runner import (
    AblationType,
    ComponentSpec,
    AblationResult,
    AblationStudy,
    AblationRunner,
    default_parser,
)

from .circuit_finder import (
    CircuitType,
    CircuitNode,
    Circuit,
    CircuitGraph,
    CircuitFinder,
    summarize_circuits,
)

from .benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    CircuitBenchmark,
    BENCHMARK_PROMPTS,
    demo,
)

__all__ = [
    # Validity measurement
    "ValidityMetrics",
    "ValidityMeasurer",
    "ComponentValidityTests",
    # Ablation
    "AblationType",
    "ComponentSpec",
    "AblationResult",
    "AblationStudy",
    "AblationRunner",
    "default_parser",
    # Circuit finding
    "CircuitType",
    "CircuitNode",
    "Circuit",
    "CircuitGraph",
    "CircuitFinder",
    "summarize_circuits",
    # Benchmark
    "BenchmarkResult",
    "BenchmarkSummary",
    "CircuitBenchmark",
    "BENCHMARK_PROMPTS",
    "demo",
]
