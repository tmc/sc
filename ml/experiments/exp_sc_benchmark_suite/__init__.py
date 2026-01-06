"""
exp_sc_benchmark_suite: Comprehensive Statechart Generation Benchmark

GOAL: Benchmark ALL SC generation methods across diverse test cases.

Components:
1. TEST SUITE: 50+ prompts from simple to complex
2. METHOD RUNNER: Unified interface for all generation methods
3. REPORT GENERATOR: Markdown comparison tables
4. BENCHMARK: Main orchestration and metrics

Methods compared:
- Baseline: Direct LLM generation
- Steering: Activation steering for SC structure
- Circuits: SAE circuit-guided generation
- TRM: Transition Relation Model
- Hybrid: Combined approaches

Metrics:
- Validity: % of valid statecharts
- Latency: Generation time
- Efficiency: Tokens per state/transition

Usage:
    from ml.experiments.exp_sc_benchmark_suite import run_benchmark
    results = run_benchmark()
    # Results saved to benchmark_report.md
"""

from .test_suite import (
    TestCase,
    TestSuite,
    Complexity,
    get_test_suite,
    SIMPLE_TESTS,
    MEDIUM_TESTS,
    COMPLEX_TESTS,
)

from .method_runner import (
    MethodRunner,
    GenerationMethod,
    MethodResult,
    BaselineRunner,
    SteeringRunner,
    CircuitsRunner,
    TRMRunner,
    HybridRunner,
)

from .report_generator import (
    ReportGenerator,
    generate_markdown_report,
    generate_comparison_table,
)

from .benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResults,
    run_benchmark,
)

__all__ = [
    # Test suite
    'TestCase',
    'TestSuite',
    'Complexity',
    'get_test_suite',
    'SIMPLE_TESTS',
    'MEDIUM_TESTS',
    'COMPLEX_TESTS',
    # Method runner
    'MethodRunner',
    'GenerationMethod',
    'MethodResult',
    'BaselineRunner',
    'SteeringRunner',
    'CircuitsRunner',
    'TRMRunner',
    'HybridRunner',
    # Report generator
    'ReportGenerator',
    'generate_markdown_report',
    'generate_comparison_table',
    # Benchmark
    'BenchmarkRunner',
    'BenchmarkConfig',
    'BenchmarkResults',
    'run_benchmark',
]
