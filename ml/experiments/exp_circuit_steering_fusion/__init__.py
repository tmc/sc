"""
exp_circuit_steering_fusion: Merge circuit intervention with steered decoding.

GOAL: Apply circuit-specific steering during token-by-token generation
and compare against global steering.

Circuits:
- TRANSITION (L8-14): For transition validity
- HIERARCHY (L0-6): For hierarchical structure

Pipeline:
1. Load steering vectors from exp_mlux_sc_steering
2. Create circuit-specific hooks (L0-6, L8-14)
3. Apply fused steering during autoregressive generation
4. Compare global vs circuit-specific steering

Key Components:
- circuit_hooks.py: Circuit definitions and hook creation
- fused_steering.py: FusedSteeredSampler with multi-circuit support
- benchmark.py: Compare global vs transition vs hierarchy vs fused

Integrates:
- exp_steered_decoding: Token-by-token generation with hooks
- exp_circuit_intervention: Circuit-specific layer targeting
- exp_mlux_sc_steering: Steering vector computation
"""

# Circuit hooks
from .circuit_hooks import (
    # Types
    CircuitType,
    Circuit,
    CircuitHook,
    MultiCircuitConfig,
    # Constants
    CIRCUITS,
    # Functions
    get_circuit,
    create_circuit_hook,
    create_circuit_hooks,
    create_multi_circuit_hooks,
    create_transition_hooks,
    create_hierarchy_hooks,
    create_global_hooks,
    create_combined_hooks,
)

# Fused steering
from .fused_steering import (
    # Config
    FusedSteeringConfig,
    FusedGenerationResult,
    # Sampler
    FusedSteeredSampler,
    # Functions
    load_steering_vectors,
)

# Benchmark
from .benchmark import (
    # Results
    ModeResult,
    BenchmarkResult,
    FullBenchmarkResult,
    # Benchmark class
    CircuitFusionBenchmark,
    # Prompts
    TRANSITION_PROMPTS,
    HIERARCHY_PROMPTS,
    MIXED_PROMPTS,
    ALL_PROMPTS,
    # Validation
    validate_statechart,
)

__all__ = [
    # Circuit hooks
    'CircuitType',
    'Circuit',
    'CircuitHook',
    'MultiCircuitConfig',
    'CIRCUITS',
    'get_circuit',
    'create_circuit_hook',
    'create_circuit_hooks',
    'create_multi_circuit_hooks',
    'create_transition_hooks',
    'create_hierarchy_hooks',
    'create_global_hooks',
    'create_combined_hooks',
    # Fused steering
    'FusedSteeringConfig',
    'FusedGenerationResult',
    'FusedSteeredSampler',
    'load_steering_vectors',
    # Benchmark
    'ModeResult',
    'BenchmarkResult',
    'FullBenchmarkResult',
    'CircuitFusionBenchmark',
    'TRANSITION_PROMPTS',
    'HIERARCHY_PROMPTS',
    'MIXED_PROMPTS',
    'ALL_PROMPTS',
    'validate_statechart',
]
