"""
exp_sc_guided_decoding: SC-Specific Guided Decoding

Build statechart grammar as FSM for constrained SC JSON generation.
Unlike generic JSON guidance, this enforces:
- SC-specific field names (root_state, label, type, children, transitions)
- Required fields at each level
- Type values constrained to 1, 2, 3
- Transition structure (from, to, event, guard, action)

Usage:
    from ml.experiments.exp_sc_guided_decoding import (
        SCGrammar,
        SCGuidedSampler,
        run_benchmark,
    )

    # Build grammar
    grammar = SCGrammar()

    # Create sampler
    sampler = SCGuidedSampler(grammar)

    # Get valid tokens at each step
    valid_tokens = sampler.get_valid_tokens(current_state)
"""

from .sc_grammar import (
    SCGrammar,
    SCGrammarState,
    SCFieldType,
    SCTransition,
)

from .sc_guided_sampler import (
    SCGuidedSampler,
    SamplerConfig,
    GenerationResult,
)

from .benchmark import (
    run_benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    'SCGrammar',
    'SCGrammarState',
    'SCFieldType',
    'SCTransition',
    'SCGuidedSampler',
    'SamplerConfig',
    'GenerationResult',
    'run_benchmark',
    'BenchmarkConfig',
    'BenchmarkResult',
]
