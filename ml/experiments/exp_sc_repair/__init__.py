"""
exp_sc_repair: Fix Invalid Statecharts Using LLM

GOAL: Given an invalid statechart, automatically repair it.

Error Types Handled:
1. Duplicate labels - rename or merge duplicate states
2. Unreachable states - add transitions or remove states
3. Missing initial - designate initial state
4. Invalid hierarchy - fix nesting, parallel regions

Uses REAL Qwen2.5-Coder-0.5B-Instruct inference via mlx_lm.
"""

from .repair_engine import (
    SCRepairEngine,
    RepairConfig,
    RepairResult,
    repair_statechart,
)
from .repair_strategies import (
    RepairStrategy,
    DuplicateLabelStrategy,
    UnreachableStateStrategy,
    MissingInitialStrategy,
    InvalidHierarchyStrategy,
)
from .repair_validator import (
    RepairValidator,
    ValidationResult,
    validate_repair,
)
from .benchmark import (
    RepairBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Engine
    'SCRepairEngine',
    'RepairConfig',
    'RepairResult',
    'repair_statechart',
    # Strategies
    'RepairStrategy',
    'DuplicateLabelStrategy',
    'UnreachableStateStrategy',
    'MissingInitialStrategy',
    'InvalidHierarchyStrategy',
    # Validator
    'RepairValidator',
    'ValidationResult',
    'validate_repair',
    # Benchmark
    'RepairBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
