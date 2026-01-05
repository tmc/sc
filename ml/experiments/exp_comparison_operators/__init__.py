"""
exp_comparison_operators: Fix L3 guard synthesis weakness (20% -> 80%+).

Problem: Guard synthesis achieves 100% on L1/L2/L4 but only 20% on L3 (comparisons).
L3 requires operators like <, >, <=, >=, ==, != (e.g., "count < max_retries").

Hypothesis: Training data bias toward boolean guards. Need targeted examples.

Approach:
1. Comparison operator taxonomy: <, >, <=, >=, ==, !=
2. Context-specific training: count, value, score, level, health, time
3. Few-shot examples focused exclusively on comparisons
4. Test each operator x multiple contexts
"""

from .comparison_generator import (
    ComparisonOperator,
    ComparisonContext,
    generate_comparison_guard,
    create_comparison_prompt,
)

from .benchmark import run_comparison_benchmark

__all__ = [
    'ComparisonOperator',
    'ComparisonContext',
    'generate_comparison_guard',
    'create_comparison_prompt',
    'run_comparison_benchmark',
]
