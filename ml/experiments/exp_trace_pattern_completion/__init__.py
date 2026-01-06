"""
Trace Pattern Completion Experiment

Learn patterns from example traces and complete partial traces.

Pattern types supported:
- Cycle: [A,B,C,A,B,C,...]
- Alternating: [A,B,A,B,...]
- Growth: [A, A,B, A,B,C,...]
- Nested: patterns with sub-patterns

Uses CoT-style scratchpad reasoning to identify patterns
and predict next events.
"""

from .pattern_learner import (
    PatternLearner,
    Pattern,
    PatternAnalysis,
    learn_patterns,
)

from .trace_completer import (
    TraceCompleter,
    CompletionResult,
    complete_trace,
)

from .benchmark import (
    run_benchmark,
    format_report,
    TEST_CASES,
    BenchmarkResult,
)

from .complex_cases import (
    COMPLEX_TEST_CASES,
    get_challenge_cases,
    get_all_challenge_types,
)

from .differential_learner import (
    DifferentialLearner,
    UnigramLearner,
    NGramLearner,
    StackAwareLearner,
    ModeAwareLearner,
    CounterAwareLearner,
    EnsembleLearner,
    LearnerResult,
)

from .benchmark_extended import (
    run_extended_benchmark,
)

__all__ = [
    # Pattern learning
    "PatternLearner",
    "Pattern",
    "PatternAnalysis",
    "learn_patterns",
    # Trace completion
    "TraceCompleter",
    "CompletionResult",
    "complete_trace",
    # Benchmark
    "run_benchmark",
    "format_report",
    "TEST_CASES",
    "BenchmarkResult",
    # Complex cases
    "COMPLEX_TEST_CASES",
    "get_challenge_cases",
    "get_all_challenge_types",
    # Differential learning
    "DifferentialLearner",
    "UnigramLearner",
    "NGramLearner",
    "StackAwareLearner",
    "ModeAwareLearner",
    "CounterAwareLearner",
    "EnsembleLearner",
    "LearnerResult",
    "run_extended_benchmark",
]
