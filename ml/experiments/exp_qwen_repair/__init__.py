"""
exp_qwen_repair: Fix Invalid Statecharts using QwenCoder LLM

Goal: Achieve 90%+ repair success rate for invalid statecharts.

Strategy:
1. Parse validation errors into structured form
2. Apply deterministic repairs for common patterns
3. Use LLM for complex/novel errors
4. Iterate until valid or max attempts

Key Components:
- ErrorAnalyzer: Parse `sc validate` output into structured errors
- RepairStrategies: Deterministic fixes for common error types
- SCRepairer: LLM-based repair using QwenCoder
- Benchmark: Measure repair success rate

Error Types Handled:
- DUPLICATE_STATE: Rename duplicates
- INVALID_SOURCE/TARGET: Fix typos via similarity matching
- NO_INITIAL_STATE: Mark first child as initial
- MISSING_EVENT: Generate event name from states
- EMPTY_COMPOUND: Change to basic type
- And more...
"""

from .error_analyzer import (
    ErrorType,
    ValidationError,
    ValidationResult,
    ErrorAnalyzer,
    analyze_errors,
)

from .repair_strategies import (
    RepairResult,
    RepairAction,
    RepairPlan,
    RepairStrategies,
    apply_strategies,
)

from .sc_repairer import (
    RepairConfig,
    RepairAttempt,
    RepairSession,
    SCRepairer,
    repair_chart,
)

from .benchmark import (
    BenchmarkResult,
    InvalidChartGenerator,
    RepairBenchmark,
    run_full_benchmark,
)
