"""
exp_active_regex_learning: Minimize Examples for Regex Synthesis

Uses active learning to strategically query examples, reducing the
number of labeled examples needed to learn a regex/statechart.

GOAL: 50% fewer examples than passive (random) learning.

APPROACH:
1. Start with few examples
2. Query oracle for strategically selected strings (uncertainty sampling)
3. Iteratively refine hypothesis
4. Compare to passive baseline

KEY STRATEGIES:
- Uncertainty Sampling: Query strings where model is most uncertain
- Boundary Sampling: Query strings near decision boundary
- Query-by-Committee: Query strings where ensemble disagrees
- Hybrid: Combine uncertainty with diversity

MODULES:
- oracle_interface.py: Ground truth oracles for benchmark patterns
- query_strategy.py: Various query selection strategies
- active_learner.py: Main active learning loop
- sample_efficiency.py: Comparison of active vs passive
"""

from .oracle_interface import (
    # Core types
    Oracle,
    RegexOracle,
    StatechartOracle,

    # Benchmark oracles
    create_ab_star_oracle,
    create_even_zeros_oracle,
    create_no_consecutive_oracle,
    create_binary_divisible_by_3_oracle,
    create_identifier_oracle,
    create_email_prefix_oracle,
    create_phone_number_oracle,

    # Utilities
    get_oracle,
    list_oracles,
    generate_random_string,
    generate_string_pool,
    BENCHMARK_ORACLES,
)

from .query_strategy import (
    # Strategy base
    QueryStrategy,

    # Strategies
    RandomStrategy,
    UncertaintySampling,
    BoundarySampling,
    QueryByCommittee,
    LengthStratifiedSampling,
    DiversitySampling,
    HybridStrategy,

    # Hypothesis
    StatechartHypothesis,

    # Utilities
    get_strategy,
    list_strategies,
)

from .active_learner import (
    # Core classes
    LearningState,
    DFAHypothesis,
    ActiveLearner,

    # Convenience functions
    run_active_learning,
    compare_strategies,
)

from .sample_efficiency import (
    # Result types
    ExperimentResult,
    ComparisonResult,

    # Experiment runners
    run_single_experiment,
    run_comparison,
    run_full_benchmark,

    # Visualization
    generate_learning_curves,
    print_learning_curves,
)

__all__ = [
    # Oracle interface
    "Oracle",
    "RegexOracle",
    "StatechartOracle",
    "create_ab_star_oracle",
    "create_even_zeros_oracle",
    "create_no_consecutive_oracle",
    "create_binary_divisible_by_3_oracle",
    "get_oracle",
    "list_oracles",
    "BENCHMARK_ORACLES",

    # Query strategies
    "QueryStrategy",
    "RandomStrategy",
    "UncertaintySampling",
    "BoundarySampling",
    "QueryByCommittee",
    "HybridStrategy",
    "get_strategy",
    "list_strategies",

    # Active learner
    "LearningState",
    "ActiveLearner",
    "run_active_learning",

    # Sample efficiency
    "ExperimentResult",
    "ComparisonResult",
    "run_comparison",
    "run_full_benchmark",
]
