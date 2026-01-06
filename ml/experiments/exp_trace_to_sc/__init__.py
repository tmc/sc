"""
Trace to Statechart Induction Experiment

Given event traces, induce the statechart that produced them.
This is grammar/automata induction from observed sequences.

Includes:
- Direct prompting baseline
- Chain-of-thought (CoT) prompting for improved induction
- Trace analysis utilities
"""

from .inducer import (
    SCInducer,
    InductionResult,
    induce_statechart,
    compare_statecharts,
)

from .benchmark import (
    InductionBenchmark,
    run_benchmark,
)

from .trace_analyzer import (
    TraceAnalyzer,
    TraceAnalysis,
    analyze_traces,
    format_trace_analysis,
)

from .cot_inducer import (
    CoTInducer,
    DirectInducer,
    CoTInductionResult,
)

from .benchmark_cot import (
    run_cot_benchmark,
    COT_TEST_CASES,
)

__all__ = [
    # Original
    "SCInducer",
    "InductionResult",
    "induce_statechart",
    "compare_statecharts",
    "InductionBenchmark",
    "run_benchmark",
    # Trace analysis
    "TraceAnalyzer",
    "TraceAnalysis",
    "analyze_traces",
    "format_trace_analysis",
    # CoT
    "CoTInducer",
    "DirectInducer",
    "CoTInductionResult",
    "run_cot_benchmark",
    "COT_TEST_CASES",
]
