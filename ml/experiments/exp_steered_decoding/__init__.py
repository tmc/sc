"""
exp_steered_decoding: Custom autoregressive decoding with steering hooks.

GOAL: Apply steering vectors at each token generation step for improved
statechart validity.

Pipeline:
1. Load steering vectors from exp_mlux_sc_steering
2. Create hooks for layer injection
3. Custom autoregressive loop with hook application
4. Compare steered vs baseline on edge cases

Key Components:
- steered_sampler.py: Custom token-by-token generation with hooks
- hook_injection.py: Create and manage steering hooks for mlux
- benchmark.py: Compare steered vs baseline on edge cases

Integrates with exp_mlux_sc_steering for steering vector computation.

Key Insight: Standard mlx_lm generation doesn't support hooks during
autoregressive decoding. This module implements custom decoding that
applies steering at every forward pass.
"""

# Steered sampler
from .steered_sampler import (
    # Config
    SamplerConfig,
    GenerationResult,
    # Sampler
    SteeredSampler,
    # Functions
    batch_generate,
    batch_comparison,
)

# Hook injection
from .hook_injection import (
    # Config
    SteeringConfig,
    # Hook creators
    create_steering_hook,
    create_multi_layer_hooks,
    create_contrastive_hook,
    create_validity_hook,
    create_syntax_hook,
    create_semantic_hook,
    # Adaptive
    AdaptiveSteeringHook,
    # Utilities
    get_available_layers,
    find_layer_hooks,
)

# Benchmark
from .benchmark import (
    # Result types
    BenchmarkResult,
    BenchmarkSummary,
    # Benchmark class
    SteeredDecodingBenchmark,
    # Functions
    grid_search,
    validate_statechart,
    # Test prompts
    EDGE_CASE_PROMPTS,
    STANDARD_PROMPTS,
)

__all__ = [
    # Steered sampler
    'SamplerConfig',
    'GenerationResult',
    'SteeredSampler',
    'batch_generate',
    'batch_comparison',
    # Hook injection
    'SteeringConfig',
    'create_steering_hook',
    'create_multi_layer_hooks',
    'create_contrastive_hook',
    'create_validity_hook',
    'create_syntax_hook',
    'create_semantic_hook',
    'AdaptiveSteeringHook',
    'get_available_layers',
    'find_layer_hooks',
    # Benchmark
    'BenchmarkResult',
    'BenchmarkSummary',
    'SteeredDecodingBenchmark',
    'grid_search',
    'validate_statechart',
    'EDGE_CASE_PROMPTS',
    'STANDARD_PROMPTS',
]
