"""
Temporal Behavior Experiment

Models and tests time-based statechart behaviors:
- AFTER(duration) events: trigger after time in state
- Timeout patterns: idle timeout, retry with backoff
- Delayed transitions: wait before firing
- Time guards: conditions based on elapsed time

Follows Harel statecharts and UML state machine temporal semantics.
"""

from .temporal_behavior import (
    TimeEvent,
    TemporalState,
    TemporalTransition,
    TemporalStatechart,
    TemporalSimulator,
    build_temporal_sc,
    predict_temporal_behavior,
    parse_after_duration,
)

from .test_cases import (
    TEST_CASES,
    TC1_SIMPLE_TIMEOUT,
    TC2_ACTIVITY_RESET,
    TC3_EXPONENTIAL_BACKOFF,
    TC4_DEADLINE_RACE,
    TC5_DEBOUNCE,
    TC6_NESTED_TIMEOUTS,
    TC7_PARALLEL_TIMEOUTS,
    TC8_TIME_GUARDS,
    get_test_case,
    get_all_scenarios,
)

from .benchmark import (
    run_benchmark,
    run_algorithmic_benchmark,
    run_llm_benchmark,
    format_report,
    BenchmarkResult,
)

__all__ = [
    # Temporal behavior modeling
    "TimeEvent",
    "TemporalState",
    "TemporalTransition",
    "TemporalStatechart",
    "TemporalSimulator",
    "build_temporal_sc",
    "predict_temporal_behavior",
    "parse_after_duration",
    # Test cases
    "TEST_CASES",
    "TC1_SIMPLE_TIMEOUT",
    "TC2_ACTIVITY_RESET",
    "TC3_EXPONENTIAL_BACKOFF",
    "TC4_DEADLINE_RACE",
    "TC5_DEBOUNCE",
    "TC6_NESTED_TIMEOUTS",
    "TC7_PARALLEL_TIMEOUTS",
    "TC8_TIME_GUARDS",
    "get_test_case",
    "get_all_scenarios",
    # Benchmark
    "run_benchmark",
    "run_algorithmic_benchmark",
    "run_llm_benchmark",
    "format_report",
    "BenchmarkResult",
]
