"""
Experiment: Regex Transfer Learning Across Domains

Transfer learned patterns from one regex domain to another:
- Email: user@domain.tld
- URL: protocol://domain/path
- Phone: +1-area-prefix-line
- Date: YYYY-MM-DD
- IPv4: 192.168.1.1

Key insight: These domains share STRUCTURAL patterns:
- [content]+ [delimiter] [content]+ [delimiter] [content]+
- The specific characters differ, but structure transfers.

Components:
- DomainEncoder: Extract domain-agnostic structural features
- TransferLearner: Learn on source domain, transfer to target
- DomainBenchmark: Measure sample efficiency improvement

Target: 50% sample efficiency improvement via transfer learning.

Based on E192 suggestion.
"""

from .domain_encoder import (
    # Character types
    CharType,
    get_char_type,
    # Segment representation
    Segment,
    PatternStructure,
    # Analyzers
    PatternAnalyzer,
    DomainEncoder,
    # Pattern generators
    DomainPatterns,
    # Analysis
    compute_domain_similarity,
)

from .transfer_learner import (
    # Networks
    PatternMatcherNetwork,
    # Learners
    TransferLearner,
    BaselineLearner,
    # Stats
    TransferStats,
    # Functions
    run_transfer_experiment,
    compare_transfer_vs_baseline,
)

from .domain_benchmark import (
    # Config
    BenchmarkConfig,
    # Results
    DomainResult,
    BenchmarkResult,
    # Runner
    DomainBenchmark,
    # Analysis
    analyze_transfer_matrix,
    find_optimal_transfer_path,
    compute_sample_efficiency_improvement,
    # Reporting
    print_benchmark_report,
    run_quick_benchmark,
)

__all__ = [
    # Domain encoder
    "CharType",
    "get_char_type",
    "Segment",
    "PatternStructure",
    "PatternAnalyzer",
    "DomainEncoder",
    "DomainPatterns",
    "compute_domain_similarity",
    # Transfer learner
    "PatternMatcherNetwork",
    "TransferLearner",
    "BaselineLearner",
    "TransferStats",
    "run_transfer_experiment",
    "compare_transfer_vs_baseline",
    # Benchmark
    "BenchmarkConfig",
    "DomainResult",
    "BenchmarkResult",
    "DomainBenchmark",
    "analyze_transfer_matrix",
    "find_optimal_transfer_path",
    "compute_sample_efficiency_improvement",
    "print_benchmark_report",
    "run_quick_benchmark",
]
