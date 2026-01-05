"""
exp_execution_prediction: Shared infrastructure for execution prediction experiments.

Provides:
- SC templates (counter, accumulator, branching, parallel)
- Starlark action evaluator
- Trace executor (runs SC to completion)
- Dataset builder (train/val/test splits)
- Metrics (exact match, MAE, partial match)

Individual experiments:
- exp_context_prediction: Predict final context values
- exp_terminal_state_prediction: Predict final configuration
- exp_guard_outcome_prediction: Predict which guards fire
- exp_path_length_prediction: Predict steps to terminal
- exp_trace_continuation: Predict next N states
- exp_reachability_prediction: Predict if state reachable
"""

from .sc_templates import (
    CounterMachine,
    AccumulatorMachine,
    BranchingMachine,
    ParallelCounterMachine,
    HierarchicalMachine,
    generate_random_sc,
)

from .starlark_eval import (
    StarlarkEvaluator,
    eval_action,
    eval_guard,
)

from .trace_executor import (
    TraceExecutor,
    ExecutionResult,
    execute_to_completion,
)

from .metrics import (
    exact_match_accuracy,
    partial_match_accuracy,
    mean_absolute_error,
    configuration_accuracy,
)

__all__ = [
    # Templates
    'CounterMachine',
    'AccumulatorMachine',
    'BranchingMachine',
    'ParallelCounterMachine',
    'HierarchicalMachine',
    'generate_random_sc',
    # Starlark
    'StarlarkEvaluator',
    'eval_action',
    'eval_guard',
    # Executor
    'TraceExecutor',
    'ExecutionResult',
    'execute_to_completion',
    # Metrics
    'exact_match_accuracy',
    'partial_match_accuracy',
    'mean_absolute_error',
    'configuration_accuracy',
]
