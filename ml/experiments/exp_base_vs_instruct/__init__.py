"""
exp_base_vs_instruct: Compare base vs instruct models for SC generation.

Tests if base models (non-Instruct) perform differently for
completion-style statechart generation.

HYPOTHESIS: Base models may be better at raw completion tasks
since we're doing JSON continuation, not instruction-following.
"""

from .compare import (
    MODELS,
    TEST_PROMPTS,
    ModelResult,
    parse_and_analyze,
    test_model,
    run_comparison,
    print_summary,
    quick_test,
    full_test,
)

__all__ = [
    "MODELS",
    "TEST_PROMPTS",
    "ModelResult",
    "parse_and_analyze",
    "test_model",
    "run_comparison",
    "print_summary",
    "quick_test",
    "full_test",
]
