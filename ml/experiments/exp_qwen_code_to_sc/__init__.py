"""
exp_qwen_code_to_sc: Extract statecharts from code using Qwen2.5-Coder.

Goal: Extract FSM structures from switch/case and if/else code patterns.
Target: 90%+ structure recovery (states, transitions, events).

Components:
- code_analyzer.py: AST-based analysis for Python, C-like, and Go code
- pattern_extractor.py: FSM pattern extraction with optional LLM support
- sc_builder.py: Proto-compatible statechart builder
- benchmark.py: Structure recovery benchmarks

Usage:
    from experiments.exp_qwen_code_to_sc import (
        # Code Analysis
        MultiLanguageAnalyzer, CodeLanguage, CodeAnalysis,
        PythonAnalyzer, CLikeAnalyzer, GoAnalyzer,
        # Pattern Extraction
        RuleBasedExtractor, QwenPatternExtractor, HybridExtractor,
        ExtractedPattern, ExtractionResult,
        # Statechart Building
        StatechartBuilder, StatechartValidator, StatechartSerializer,
        StateProto, TransitionProto, StatechartProto, StateType,
        # Benchmarking
        StructureRecoveryBenchmark, run_benchmark,
    )

    # Extract from Python code
    extractor = RuleBasedExtractor()
    result = extractor.extract('''
        state = "idle"
        if state == "idle":
            state = "running"
    ''')

    # Build statechart
    builder = StatechartBuilder()
    sc = builder.build(result)

    # Output formats
    print(sc.to_json())     # JSON
    print(sc.to_textproto()) # TextProto
    print(StatechartSerializer.to_mermaid(sc))  # Mermaid diagram

    # Run benchmarks
    suite = run_benchmark()
    print(f"Overall: {suite.avg_overall*100:.1f}%")
"""

from .code_analyzer import (
    CodeLanguage,
    StateVariable,
    StateTransition,
    StateBlock,
    CodeAnalysis,
    PythonAnalyzer,
    CLikeAnalyzer,
    GoAnalyzer,
    MultiLanguageAnalyzer,
)

from .pattern_extractor import (
    ExtractedPattern,
    ExtractionResult,
    QwenPatternExtractor,
    RuleBasedExtractor,
    HybridExtractor,
)

from .sc_builder import (
    StateType,
    StateProto,
    TransitionProto,
    StatechartProto,
    StatechartBuilder,
    StatechartValidator,
    StatechartSerializer,
)

from .benchmark import (
    TestCase,
    RecoveryMetrics,
    BenchmarkResult,
    BenchmarkSuite,
    StructureRecoveryBenchmark,
    TEST_CASES,
    run_benchmark,
)

__all__ = [
    # Code Analyzer
    'CodeLanguage',
    'StateVariable',
    'StateTransition',
    'StateBlock',
    'CodeAnalysis',
    'PythonAnalyzer',
    'CLikeAnalyzer',
    'GoAnalyzer',
    'MultiLanguageAnalyzer',
    # Pattern Extractor
    'ExtractedPattern',
    'ExtractionResult',
    'QwenPatternExtractor',
    'RuleBasedExtractor',
    'HybridExtractor',
    # Statechart Builder
    'StateType',
    'StateProto',
    'TransitionProto',
    'StatechartProto',
    'StatechartBuilder',
    'StatechartValidator',
    'StatechartSerializer',
    # Benchmark
    'TestCase',
    'RecoveryMetrics',
    'BenchmarkResult',
    'BenchmarkSuite',
    'StructureRecoveryBenchmark',
    'TEST_CASES',
    'run_benchmark',
]
