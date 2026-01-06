"""
exp_circuit_intervention: Targeted Circuit Interventions for Statechart Validity

Applies targeted interventions on discovered circuits to improve statechart generation.

CIRCUITS (from exp_mlux_sc_circuits):
- TRANSITION_VALIDITY (L8-14): Fix invalid state references
- HIERARCHY (L0-6): Improve nested state handling
- STRUCTURAL (L0-6): Strengthen JSON structure

INTERVENTION METHODS:
1. Scaling: Multiply activations by factor > 1
2. Steering: Add direction vectors
3. Clamping: Ensure minimum activation magnitude
4. Attention boost: Increase attention weights

Key Components:
- CircuitAmplifier: Apply hooks to amplify circuits
- TargetedFixer: Diagnose errors and apply fixes
- InterventionBenchmark: Compare circuit-specific vs global steering
"""

from .circuit_amplifier import (
    CircuitType,
    CircuitSpec,
    AmplificationType,
    AmplificationConfig,
    InterventionResult,
    CircuitAmplifier,
    transition_fix_config,
    hierarchy_boost_config,
    structural_fix_config,
)

from .targeted_fix import (
    ErrorType,
    ErrorDiagnosis,
    FixResult,
    ErrorDiagnoser,
    TargetedFixer,
    fix_transitions,
    fix_hierarchy,
)

from .benchmark import (
    InterventionMetrics,
    SteeringMethod,
    BenchmarkResult,
    BenchmarkSummary,
    InterventionBenchmark,
    BROKEN_CHARTS,
    TEST_PROMPTS,
    demo,
)

__all__ = [
    # Circuit amplifier
    "CircuitType",
    "CircuitSpec",
    "AmplificationType",
    "AmplificationConfig",
    "InterventionResult",
    "CircuitAmplifier",
    "transition_fix_config",
    "hierarchy_boost_config",
    "structural_fix_config",
    # Targeted fix
    "ErrorType",
    "ErrorDiagnosis",
    "FixResult",
    "ErrorDiagnoser",
    "TargetedFixer",
    "fix_transitions",
    "fix_hierarchy",
    # Benchmark
    "InterventionMetrics",
    "SteeringMethod",
    "BenchmarkResult",
    "BenchmarkSummary",
    "InterventionBenchmark",
    "BROKEN_CHARTS",
    "TEST_PROMPTS",
    "demo",
]
