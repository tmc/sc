"""
exp_sc_guided_sampling: Grammar-Guided Sampling for Statechart JSON

Uses constrained decoding to guarantee valid statechart output.

APPROACH:
1. Define SC JSON grammar as FSM
2. Track parser state during generation
3. Mask invalid tokens before sampling
4. Achieve 100% syntactic validity

Key Components:
- SCGrammar: JSON grammar as finite state machine
- GuidedSampler: Constrained decoding implementation
- UnguidedSampler: Baseline comparison
- GuidedSamplingBenchmark: Compare base vs instruct models
"""

from .sc_grammar import (
    GrammarState,
    TokenClass,
    ParserState,
    SCGrammar,
    SCGrammarValidator,
    get_valid_next_tokens,
)

from .guided_sampler import (
    SamplingConfig,
    SamplingResult,
    TokenMaskBuilder,
    GuidedSampler,
    UnguidedSampler,
)

from .benchmark_1_5b import (
    MODELS,
    TEST_PROMPTS,
    BenchmarkResult,
    BenchmarkSummary,
    GuidedSamplingBenchmark,
    run_experiment,
    demo,
)

__all__ = [
    # Grammar
    "GrammarState",
    "TokenClass",
    "ParserState",
    "SCGrammar",
    "SCGrammarValidator",
    "get_valid_next_tokens",
    # Sampler
    "SamplingConfig",
    "SamplingResult",
    "TokenMaskBuilder",
    "GuidedSampler",
    "UnguidedSampler",
    # Benchmark
    "MODELS",
    "TEST_PROMPTS",
    "BenchmarkResult",
    "BenchmarkSummary",
    "GuidedSamplingBenchmark",
    "run_experiment",
    "demo",
]
