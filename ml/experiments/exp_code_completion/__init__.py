"""
exp_code_completion: Statechart-Guided LLM Code Generation

Goal: Achieve 99%+ syntactic validity by masking LLM logits
with statechart-derived syntax constraints.

Key Components:
1. SyntaxStatechart - States for Python grammar (STATEMENT, EXPRESSION, etc.)
2. SAE State Discovery - Find monosemantic features for syntax contexts
3. LogitMasker - Block tokens invalid in current syntax state
4. Evaluation - Measure syntax error rate vs unconstrained LLM

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
