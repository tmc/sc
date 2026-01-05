"""
exp_guard_outcome_prediction: Predict which guards fire given context.

Given:
- List of guard expressions (competing transitions)
- Context values

Predict:
- Which guard index fires (0..N-1) or -1 if none

Guard Complexity Levels:
- L1: Simple comparison (x > 5)
- L2: Compound comparison (x > 5 && y < 10)
- L3: Arithmetic comparison ((x + y) > threshold)
- L4: List/property access (items.length > 0 && items[0].valid)
"""

from .guard_predictor import (
    GuardLevel,
    generate_guard_set,
    generate_context_for_guards,
    compute_expected_outcome,
    create_guard_prediction_prompt,
    parse_prediction,
)

from .benchmark import (
    run_benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    'GuardLevel',
    'generate_guard_set',
    'generate_context_for_guards',
    'compute_expected_outcome',
    'create_guard_prediction_prompt',
    'parse_prediction',
    'run_benchmark',
    'BenchmarkConfig',
    'BenchmarkResult',
]
