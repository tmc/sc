"""
exp_expression_languages: Cross-Language Guard Expression Testing

Test guard equivalence across expression languages supported by the statechart
proto: RAW, CEL, STARLARK, JAVASCRIPT, GO.

Key Questions:
1. Do same guards in different languages produce same behavior?
2. What security limits apply to each language?
3. Which language is best for different use cases?

Components:
- expression_parser.py: Multi-language parsing (AST extraction)
- guard_equivalence.py: Semantic comparison across languages
- security_tester.py: Test depth, timeout, function limits
- language_benchmark.py: Compare performance and capabilities

Based on proto/statecharts/v1/expressions.proto:
- ExpressionType: RAW, CEL, STARLARK
- ExpressionSecurityConfig: max_depth, max_execution_ms, allowed_functions

Usage:
    from experiments.exp_expression_languages import (
        LanguageBenchmark,
        GuardEquivalenceChecker,
        ExpressionSecurityTester,
        MultiLanguageParser,
    )

    # Quick benchmark
    benchmark = LanguageBenchmark()
    results = benchmark.run()

    # Check equivalence
    checker = GuardEquivalenceChecker()
    result = checker.check_equivalence(
        "x > 0 and y < 10", ExpressionLanguage.RAW,
        "x > 0 && y < 10", ExpressionLanguage.CEL,
    )

    # Test security
    tester = ExpressionSecurityTester()
    result = tester.test_expression("eval('x')", ExpressionLanguage.RAW)
"""

from .expression_parser import (
    ExpressionLanguage,
    ParsedExpression,
    ExpressionParser,
    RawExpressionParser,
    CELExpressionParser,
    StarlarkExpressionParser,
    JavaScriptExpressionParser,
    GoExpressionParser,
    MultiLanguageParser,
)

from .guard_equivalence import (
    EquivalenceLevel,
    EquivalenceResult,
    GuardEquivalenceSet,
    StructuralComparator,
    SemanticComparator,
    GuardEquivalenceChecker,
    EquivalenceTestSuite,
)

from .security_tester import (
    SecurityViolationType,
    SecurityConfig,
    SecurityViolation,
    SecurityTestResult,
    DepthChecker,
    FunctionChecker,
    PatternChecker,
    TimeoutChecker,
    ExpressionSecurityTester,
    SecurityTestSuite,
)

from .language_benchmark import (
    LanguageMetrics,
    BenchmarkResult,
    ExpressionGenerator,
    ParseBenchmark,
    EvalBenchmark,
    EquivalenceBenchmark,
    SecurityBenchmark,
    LanguageBenchmark,
)

__all__ = [
    # Expression Parser
    'ExpressionLanguage',
    'ParsedExpression',
    'ExpressionParser',
    'RawExpressionParser',
    'CELExpressionParser',
    'StarlarkExpressionParser',
    'JavaScriptExpressionParser',
    'GoExpressionParser',
    'MultiLanguageParser',
    # Guard Equivalence
    'EquivalenceLevel',
    'EquivalenceResult',
    'GuardEquivalenceSet',
    'StructuralComparator',
    'SemanticComparator',
    'GuardEquivalenceChecker',
    'EquivalenceTestSuite',
    # Security Tester
    'SecurityViolationType',
    'SecurityConfig',
    'SecurityViolation',
    'SecurityTestResult',
    'DepthChecker',
    'FunctionChecker',
    'PatternChecker',
    'TimeoutChecker',
    'ExpressionSecurityTester',
    'SecurityTestSuite',
    # Language Benchmark
    'LanguageMetrics',
    'BenchmarkResult',
    'ExpressionGenerator',
    'ParseBenchmark',
    'EvalBenchmark',
    'EquivalenceBenchmark',
    'SecurityBenchmark',
    'LanguageBenchmark',
]
