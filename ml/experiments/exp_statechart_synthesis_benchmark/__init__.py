"""
exp_statechart_synthesis_benchmark

Comprehensive benchmark suite for statechart synthesis methods.
Compares evolution, SAE, neural, and hybrid approaches across
games, regex, dialogue, and code domains.
"""

from .synthesis_methods import (
    SynthesisMethod,
    SynthesisConfig,
    SynthesisResult,
    SynthesizedStatechart,
    MethodCategory,
    EvolutionaryMethod,
    NSGA2Method,
    SAEMethod,
    NeuralMethod,
    HybridMethod,
    RandomMethod,
    get_all_methods,
    get_methods_by_category,
)

from .domain_suite import (
    Domain,
    DomainSuite,
    DomainType,
    DomainExample,
    Difficulty,
    get_all_domains,
    get_domains_by_type,
    get_domains_by_difficulty,
    create_full_suite,
    create_quick_suite,
)

from .metrics import (
    BenchmarkMetrics,
    MetricCategory,
    MetricResult,
    compute_all_metrics,
    compute_accuracy,
    compute_precision_recall_f1,
)

from .leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    MethodRanking,
    DomainRanking,
    aggregate_experiment_results,
)

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkRun,
    MethodDomainResult,
    print_summary,
    export_results,
)

__all__ = [
    # Methods
    'SynthesisMethod',
    'SynthesisConfig',
    'SynthesisResult',
    'SynthesizedStatechart',
    'MethodCategory',
    'EvolutionaryMethod',
    'NSGA2Method',
    'SAEMethod',
    'NeuralMethod',
    'HybridMethod',
    'RandomMethod',
    'get_all_methods',
    'get_methods_by_category',

    # Domains
    'Domain',
    'DomainSuite',
    'DomainType',
    'DomainExample',
    'Difficulty',
    'get_all_domains',
    'get_domains_by_type',
    'get_domains_by_difficulty',
    'create_full_suite',
    'create_quick_suite',

    # Metrics
    'BenchmarkMetrics',
    'MetricCategory',
    'MetricResult',
    'compute_all_metrics',
    'compute_accuracy',
    'compute_precision_recall_f1',

    # Leaderboard
    'Leaderboard',
    'LeaderboardEntry',
    'MethodRanking',
    'DomainRanking',
    'aggregate_experiment_results',

    # Runner
    'BenchmarkRunner',
    'BenchmarkConfig',
    'BenchmarkRun',
    'MethodDomainResult',
    'print_summary',
    'export_results',
]
