"""
exp_code_completion: Statechart-Guided LLM Code Generation

Goal: Achieve 99%+ syntactic validity by masking LLM logits
with statechart-derived syntax constraints.

Key Components:
1. SyntaxStatechart - States for Python/Go grammar (STATEMENT, EXPRESSION, etc.)
2. TokenMasker - Maps syntax states to token validity masks
3. GuidedGeneration - Constrained LLM generation with statechart tracking
4. Benchmark - Measure syntax error rate vs unconstrained LLM (target: 99%+)

Key insight: Programming language grammars ARE statecharts!
States = parser contexts, Events = tokens, Guards = lookahead conditions.

This bridges the SAE statechart work with practical LLM code generation.
"""

from .syntax_statechart import (
    PythonSyntaxState,
    PythonSyntaxStatechart,
    get_valid_tokens,
)

from .constrained_generator import (
    LogitMasker,
    ConstrainedCodeGenerator,
)

from .token_masker import (
    TokenMasker,
    TokenCategory,
    MaskContext,
    SAEGuidedMasker,
)

from .guided_generation import (
    GuidedCodeGenerator,
    GenerationConfig,
    SyntaxStateTracker,
)

from .benchmark import (
    CodeValidator,
    CodeCompletionBenchmark,
    BenchmarkResult,
)
