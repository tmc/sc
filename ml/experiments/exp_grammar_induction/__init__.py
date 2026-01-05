"""
exp_grammar_induction: Learn Go Syntax Specification as Statechart

Goal: Reconstruct Go syntax specification as a statechart from observing
valid Go code only.

Key Insight:
    Programming language syntax IS a statechart:
    - States are parser contexts (e.g., "inside function", "after identifier")
    - Transitions are token sequences that move between contexts
    - Guards are lookahead/context conditions

Approach:
    1. Collect token sequences from valid Go code (go/scanner)
    2. Train next-token prediction model (Transformer)
    3. Apply SAE to hidden states to discover discrete syntax contexts
    4. Build statechart from SAE feature patterns
    5. Validate against go/parser accept/reject behavior

Reference:
    - Go Language Specification: https://go.dev/ref/spec
    - Finite State Automata Inside Transformers (2025)
    - Anthropic's SAE work for interpretability
"""

from .token_collector import (
    GoTokenCollector,
    TokenSequence,
    GoToken,
)

from .sequence_model import (
    GoSyntaxTransformer,
    TransformerConfig,
)

from .sae_syntax import (
    SyntaxSAE,
    SAEConfig,
    SyntaxFeature,
)

from .grammar_inducer import (
    GrammarInducer,
    InducedGrammar,
    SyntaxState,
)

from .go_parser_oracle import (
    GoParserOracle,
    ParseResult,
)

from .benchmark import (
    GrammarBenchmark,
    BenchmarkResult,
)

__all__ = [
    # Token collection
    'GoTokenCollector',
    'TokenSequence',
    'GoToken',
    # Sequence model
    'GoSyntaxTransformer',
    'TransformerConfig',
    # SAE
    'SyntaxSAE',
    'SAEConfig',
    'SyntaxFeature',
    # Grammar induction
    'GrammarInducer',
    'InducedGrammar',
    'SyntaxState',
    # Oracle
    'GoParserOracle',
    'ParseResult',
    # Benchmark
    'GrammarBenchmark',
    'BenchmarkResult',
]
