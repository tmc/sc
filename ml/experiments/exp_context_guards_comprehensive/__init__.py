"""
exp_context_guards_comprehensive: Comprehensive guard and context variable testing.

Covers all guard types from the proto specification:
- Boolean guards (is_locked, has_key)
- Comparison guards (count > 5, temp <= 100)
- Compound guards ((a && b) || c)
- Guard-action chains
- Transition priority with overlapping guards
- In-state guards for parallel regions
- Nested object/array access

Input: (transitions, context, events, active_states)
Output: (final_state, final_context)

Model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from .context_guards import (
    GuardCategory,
    Transition,
    GuardTestCase,
    evaluate_guard_expr,
    execute_action,
    simulate_machine,
    format_transitions,
    create_guard_eval_prompt,
    create_context_mutation_prompt,
    parse_state_response,
    parse_context_response,
    get_test_cases,
)

from .benchmark import (
    run_benchmark,
    BenchmarkConfig,
    PredictionResult,
)

__all__ = [
    # Data types
    'GuardCategory',
    'Transition',
    'GuardTestCase',
    # Evaluation
    'evaluate_guard_expr',
    'execute_action',
    'simulate_machine',
    # Prompts
    'format_transitions',
    'create_guard_eval_prompt',
    'create_context_mutation_prompt',
    # Parsing
    'parse_state_response',
    'parse_context_response',
    # Test cases
    'get_test_cases',
    # Benchmark
    'run_benchmark',
    'BenchmarkConfig',
    'PredictionResult',
]
