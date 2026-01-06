"""
JSON Guided Decoding Experiment

Uses a JSON grammar statechart to constrain LLM token generation,
guaranteeing syntactically valid JSON output.

Key insight: Model JSON syntax as FSM, mask invalid tokens at each state.
"""

from .json_grammar import (
    JSONGrammar,
    JSONState,
    JSONTransition,
    get_valid_next_tokens,
)

from .json_guided_sampler import (
    JSONGuidedSampler,
    SamplerConfig,
    sample_with_grammar,
)

from .benchmark import (
    JSONGuidedBenchmark,
    run_benchmark,
    BenchmarkResult,
)

__all__ = [
    "JSONGrammar",
    "JSONState",
    "JSONTransition",
    "get_valid_next_tokens",
    "JSONGuidedSampler",
    "SamplerConfig",
    "sample_with_grammar",
    "JSONGuidedBenchmark",
    "run_benchmark",
    "BenchmarkResult",
]
