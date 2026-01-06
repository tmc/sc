"""
exp_sae_regex_states: SAE Features as Regex States

GOAL: Use Sparse Autoencoder features directly as DFA states for regex matching.

KEY INSIGHT:
Instead of clustering prefixes to discover states (Myhill-Nerode approximation),
we use SAE monosemantic features AS the states directly:

    Traditional: prefix_embeddings → clustering → discrete states
    SAE Approach: prefix_embeddings → SAE → active features = state

Why SAE features make good states:
1. MONOSEMANTIC: Each feature represents one concept (e.g., "saw an 'a'")
2. SPARSE: Only K features active = clean state representation
3. INTERPRETABLE: Can name states by what activates them
4. LEARNED: Discovered from data, not hand-specified

The DFA-SAE correspondence:
    DFA State     ↔  SAE Feature(s) active
    Transition    ↔  Feature activation change on input
    Accept State  ↔  Features predicting "match"
    Start State   ↔  Features active on empty string

This experiment tests whether SAE-discovered states outperform
clustering-discovered states for regex synthesis.

Approach:
1. Train SAE on character sequence embeddings (prefix representations)
2. Map SAE features to discrete states
3. Learn transitions from feature activation dynamics
4. Compare to exp_regex_statechart's clustering approach

Usage:
    from ml.experiments.exp_sae_regex_states import (
        CharSequenceSAE,
        FeatureStateMapper,
        SAERegexEvolver,
        SAERegexBenchmark,
    )

    # Train SAE on character sequences
    sae = CharSequenceSAE(n_features=32, k_active=4)
    sae.train(positive_strings + negative_strings)

    # Map features to states
    mapper = FeatureStateMapper(sae)
    states = mapper.discover_states(strings)

    # Evolve transitions
    evolver = SAERegexEvolver(sae, mapper)
    statechart = evolver.evolve(positive_strings, negative_strings)
"""

from .sae_regex import (
    CharSequenceSAE,
    CharEncoder,
    PrefixEmbedder,
    SAEConfig,
    train_char_sae,
)

from .feature_to_state import (
    FeatureStateMapper,
    SAEState,
    StateActivation,
    FeatureInterpretation,
    extract_states_from_sae,
    interpret_features,
)

from .sae_regex_evolver import (
    SAERegexEvolver,
    SAEEvolutionConfig,
    SAEStatechart,
    SAETransition,
    evolve_sae_regex,
)

from .benchmark import (
    SAERegexBenchmark,
    SAEBenchmarkResult,
    compare_sae_vs_clustering,
    run_sae_benchmark,
)

__all__ = [
    # SAE for characters
    'CharSequenceSAE',
    'CharEncoder',
    'PrefixEmbedder',
    'SAEConfig',
    'train_char_sae',
    # Feature to state mapping
    'FeatureStateMapper',
    'SAEState',
    'StateActivation',
    'FeatureInterpretation',
    'extract_states_from_sae',
    'interpret_features',
    # Evolution
    'SAERegexEvolver',
    'SAEEvolutionConfig',
    'SAEStatechart',
    'SAETransition',
    'evolve_sae_regex',
    # Benchmark
    'SAERegexBenchmark',
    'SAEBenchmarkResult',
    'compare_sae_vs_clustering',
    'run_sae_benchmark',
]
