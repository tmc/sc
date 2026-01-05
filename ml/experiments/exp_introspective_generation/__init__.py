"""
exp_introspective_generation: Test if introspection tokens help SC generation.

Hypothesis: Models that see their constraint state make better choices.

Compares:
- Baseline: No introspection tokens
- STATE_ONLY: Model sees [SC:STATE=X]
- VALID_ONLY: Model sees [SC:VALID=a,b,c]
- FULL: Model sees all introspection info

Usage:
    from ml.experiments.exp_introspective_generation import run_introspection_benchmark
    
    results = run_introspection_benchmark()
"""

from .benchmark import (
    run_introspection_benchmark,
    IntrospectionBenchmarkResult,
)

__all__ = [
    'run_introspection_benchmark',
    'IntrospectionBenchmarkResult',
]
