"""
exp_action_execution_comprehensive - Action Execution Coverage

Comprehensive testing of entry/exit/transition actions and context mutations.

Session: DDB5
Model: Qwen2.5-Coder-1.5B-Instruct-4bit

Covers:
- Action placement (entry/exit/transition/combined/nested)
- Context mutations (assignment, increment, conditional, multi-var, nested)
- Execution order (Harel semantics: exit→trans→entry, bottom-up exit, top-down entry)
- Guard-action interaction
"""

from .action_executor import (
    ActionExecutor,
    ExecutionResult,
    ActionType,
    compare_contexts,
    compare_action_order,
)
from .test_cases import (
    TEST_CASES,
    TEST_CASES_BY_CATEGORY,
    TestCase,
    Action,
)
from .benchmark import (
    run_benchmark,
    run_single_test,
    print_summary,
    save_results,
)

__all__ = [
    "ActionExecutor",
    "ExecutionResult",
    "ActionType",
    "compare_contexts",
    "compare_action_order",
    "TEST_CASES",
    "TEST_CASES_BY_CATEGORY",
    "TestCase",
    "Action",
    "run_benchmark",
    "run_single_test",
    "print_summary",
    "save_results",
]
