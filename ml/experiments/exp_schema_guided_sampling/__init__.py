"""
exp_schema_guided_sampling: Use statechart to guide JSON generation.

Implements constrained decoding using a statechart that tracks JSON
parsing state and guides token selection to ensure valid output.

Key insight: Use statecharts to generate statecharts!

Components:
- json_schema_statechart: Statechart for JSON/SC schema validation
- guided_sampler: Token masking based on statechart state
- benchmark: Compare guided vs unguided generation
"""

from .json_schema_statechart import (
    TokenType,
    JSONState,
    ParserState,
    JSONSchemaStatechart,
    SC_REQUIRED_FIELDS,
    SC_OPTIONAL_FIELDS,
)

from .guided_sampler import (
    TokenMask,
    GuidedSampler,
    GuidedGenerator,
)

from .benchmark import (
    BenchmarkResult,
    TEST_PROMPTS,
    check_balanced,
    analyze_output,
    run_benchmark,
    print_summary,
    quick_test,
    full_test,
)

__all__ = [
    # json_schema_statechart
    "TokenType",
    "JSONState",
    "ParserState",
    "JSONSchemaStatechart",
    "SC_REQUIRED_FIELDS",
    "SC_OPTIONAL_FIELDS",
    # guided_sampler
    "TokenMask",
    "GuidedSampler",
    "GuidedGenerator",
    # benchmark
    "BenchmarkResult",
    "TEST_PROMPTS",
    "check_balanced",
    "analyze_output",
    "run_benchmark",
    "print_summary",
    "quick_test",
    "full_test",
]
