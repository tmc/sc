"""
exp_scalability_benchmark: Million-State Performance Testing

GOAL: Identify performance limits and bottlenecks at scale.

Test scales: 10 / 100 / 1K / 10K / 100K / 1M states

Operations profiled:
1. Topology evolution (mutation, crossover)
2. SAE training (encoding, reconstruction)
3. Guard synthesis (evolution, evaluation)
4. State traversal (BFS, DFS, reachability)
5. Configuration management
6. Serialization/deserialization

Key metrics:
- Time complexity vs state count
- Memory usage vs state count
- Bottleneck identification
- Scaling factor analysis
"""

from .scale_generator import (
    ScaleConfig,
    StatechartGenerator,
    generate_at_scale,
    SCALE_LEVELS,
)
from .profiler import (
    Profiler,
    ProfileResult,
    MemoryTracker,
    TimeTracker,
    profile_operation,
)
from .bottleneck_analyzer import (
    BottleneckAnalyzer,
    Bottleneck,
    BottleneckReport,
    analyze_bottlenecks,
)
from .benchmark import (
    ScalabilityBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    'ScaleConfig',
    'StatechartGenerator',
    'generate_at_scale',
    'SCALE_LEVELS',
    'Profiler',
    'ProfileResult',
    'MemoryTracker',
    'TimeTracker',
    'profile_operation',
    'BottleneckAnalyzer',
    'Bottleneck',
    'BottleneckReport',
    'analyze_bottlenecks',
    'ScalabilityBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
