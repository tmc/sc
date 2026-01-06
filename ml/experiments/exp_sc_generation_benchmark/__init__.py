"""
exp_sc_generation_benchmark: Standardized 100-Task SC Generation Benchmark

GOAL: Provide reproducible benchmark for evaluating SC generation models.

Categories (20 tasks each):
1. Simple: 3-5 states, no nesting, basic transitions
2. Hierarchical: Nested states, 2-3 levels deep
3. Parallel: Orthogonal regions, concurrent states
4. Guards: Conditional transitions with guards
5. Complex: All features combined

Each task includes:
- Natural language prompt
- Gold standard statechart (validated)
- Difficulty rating (1-5)
- Category label
"""

from .evaluator import (
    BenchmarkEvaluator,
    EvaluationResult,
    TaskResult,
    load_benchmark,
    evaluate_generation,
    run_benchmark,
)

__all__ = [
    'BenchmarkEvaluator',
    'EvaluationResult',
    'TaskResult',
    'load_benchmark',
    'evaluate_generation',
    'run_benchmark',
]
