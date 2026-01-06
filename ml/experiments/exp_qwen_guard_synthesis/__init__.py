"""
exp_qwen_guard_synthesis: NL -> Guard Expressions using Qwen2.5-Coder

GOAL: Generate boolean guard expressions from natural language
descriptions using Qwen2.5-Coder-0.5B-Instruct.

Pipeline:
1. NL description parsed for intent (nl_parser)
2. Few-shot prompting generates candidates (guard_generator)
3. Syntax validation filters invalid expressions (validator)
4. Semantic validation tests against contexts (benchmark)

Targets:
- 95%+ syntax valid expressions
- 80%+ semantic correctness

Key Components:
- guard_generator.py: Qwen-based guard generation with few-shot prompting
- nl_parser.py: Extract intent (comparison, boolean, compound) from NL
- validator.py: AST-based syntax validation
- benchmark.py: Test suites for different guard types

NO HARDCODING: All patterns learned from few-shot examples.

Reference: exp_guard_synthesis/llm_guard_gen.py
"""

# Guard generator
from .guard_generator import (
    # Config
    GeneratorConfig,
    # Generator
    QwenGuardGenerator,
    # Few-shot examples
    FEW_SHOT_EXAMPLES,
    # Prompt builders
    build_generation_prompt,
    # Batch processing
    GenerationRequest,
    GenerationResult,
    batch_generate,
)

# NL parser
from .nl_parser import (
    # Types
    IntentType,
    ComparisonOp,
    BooleanOp,
    # Data classes
    ExtractedIntent,
    ParseResult,
    # Parser
    NLParser,
)

# Validator
from .validator import (
    # Data classes
    ValidationError,
    ValidationResult,
    BatchValidationResult,
    # Validator
    GuardValidator,
    # Functions
    batch_validate,
)

# Benchmark
from .benchmark import (
    # Class
    GuardSynthesisBenchmark,
    # Result
    BenchmarkResult,
    # Functions
    run_category_benchmark,
    run_full_benchmark,
    quick_benchmark,
    # Test suites
    SIMPLE_COMPARISON_TESTS,
    BOOLEAN_TESTS,
    GAME_RULE_TESTS,
    STATE_MACHINE_TESTS,
)

__all__ = [
    # Generator
    'GeneratorConfig',
    'QwenGuardGenerator',
    'FEW_SHOT_EXAMPLES',
    'build_generation_prompt',
    'GenerationRequest',
    'GenerationResult',
    'batch_generate',
    # NL parser
    'IntentType',
    'ComparisonOp',
    'BooleanOp',
    'ExtractedIntent',
    'ParseResult',
    'NLParser',
    # Validator
    'ValidationError',
    'ValidationResult',
    'BatchValidationResult',
    'GuardValidator',
    'batch_validate',
    # Benchmark
    'GuardSynthesisBenchmark',
    'BenchmarkResult',
    'run_category_benchmark',
    'run_full_benchmark',
    'quick_benchmark',
    'SIMPLE_COMPARISON_TESTS',
    'BOOLEAN_TESTS',
    'GAME_RULE_TESTS',
    'STATE_MACHINE_TESTS',
]
