"""
exp_guard_synthesis: Program Synthesis for Guard and Action Expressions

Use LLMs to synthesize:
1. Guard expressions (boolean conditions for transitions)
2. Action expressions (side effects on transition)
3. Entry/exit actions (state lifecycle behaviors)

Guard Complexity Levels:
- L1: Single variable (is_ready)
- L2: Boolean operators (a && b, a || b)
- L3: Comparisons (x > 5, count < max)
- L4: Nested expressions ((a || b) && c)

Key insight: Instead of hand-writing guards like `x > 0 && !locked`,
we let the LLM propose candidate expressions and evolve them via self-play.
"""

from .guard_synthesizer import (
    GuardSynthesizer,
    ActionSynthesizer,
    ExpressionEvolver,
)

from .guard_generator import (
    GuardLevel,
    generate_guard,
    generate_statechart_with_guard,
    validate_guard_syntax,
    validate_guard_semantics,
    extract_guard_from_statechart,
    BOOL_VARS,
    COUNT_VARS,
    THRESHOLD_VARS,
)

from .benchmark import (
    run_benchmark,
    BenchmarkConfig,
    GuardResult,
    create_guard_prompt,
    validate_generated_guard,
)
