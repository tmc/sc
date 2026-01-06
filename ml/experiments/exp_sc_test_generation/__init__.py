"""
SC Test Generation Experiment

Given a statechart, generate test traces (event sequences) that achieve
coverage goals: state coverage and transition coverage.

Uses LLM to intelligently generate event sequences that maximize coverage.
"""

from .coverage_analyzer import (
    CoverageAnalyzer,
    CoverageReport,
    compute_state_coverage,
    compute_transition_coverage,
)

from .trace_generator import (
    TraceGenerator,
    GeneratedTrace,
    generate_test_traces,
)

from .benchmark import (
    TestGenBenchmark,
    run_benchmark,
    BenchmarkResult,
)

__all__ = [
    "CoverageAnalyzer",
    "CoverageReport",
    "compute_state_coverage",
    "compute_transition_coverage",
    "TraceGenerator",
    "GeneratedTrace",
    "generate_test_traces",
    "TestGenBenchmark",
    "run_benchmark",
    "BenchmarkResult",
]
