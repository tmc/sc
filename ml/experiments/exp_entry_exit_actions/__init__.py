"""
exp_entry_exit_actions: Generate statecharts with entry and exit actions.

Tests model's ability to generate SC JSON with:
- on_entry: Actions executed when entering a state
- on_exit: Actions executed when leaving a state
- action: Actions executed during transitions
- Chained actions: Multiple actions per state/transition

Uses REAL MLX inference (no mocking).
"""

from .action_grammar import (
    validate_sc_with_actions,
    validate_action_syntax,
    validate_state_actions,
    validate_transition_action,
    get_few_shot_examples,
    ENTRY_ACTION_EXAMPLE,
    EXIT_ACTION_EXAMPLE,
    BOTH_ACTIONS_EXAMPLE,
    TRANSITION_ACTION_EXAMPLE,
    CHAINED_ACTIONS_EXAMPLE,
)

from .benchmark import (
    run_benchmark,
    format_report,
    BenchmarkResult,
    TestCase,
    TestResult,
)


def benchmark():
    """Run benchmark and return formatted report."""
    result = run_benchmark(samples_per_test=2)
    return format_report(result)


__all__ = [
    "run_benchmark",
    "format_report",
    "benchmark",
    "BenchmarkResult",
    "TestCase",
    "TestResult",
    "validate_sc_with_actions",
    "validate_action_syntax",
    "get_few_shot_examples",
    "ENTRY_ACTION_EXAMPLE",
    "EXIT_ACTION_EXAMPLE",
    "BOTH_ACTIONS_EXAMPLE",
    "TRANSITION_ACTION_EXAMPLE",
    "CHAINED_ACTIONS_EXAMPLE",
]
