"""
exp_guard_synthesis: Program Synthesis for Guard and Action Expressions

Use LLMs to synthesize:
1. Guard expressions (boolean conditions for transitions)
2. Action expressions (side effects on transition)
3. Entry/exit actions (state lifecycle behaviors)

Key insight: Instead of hand-writing guards like `x > 0 && !locked`,
we let the LLM propose candidate expressions and evolve them via self-play.
"""

from .guard_synthesizer import (
    GuardSynthesizer,
    ActionSynthesizer,
    ExpressionEvolver,
)
