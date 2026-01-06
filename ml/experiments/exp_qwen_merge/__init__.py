"""
exp_qwen_merge: Merge Multiple Statecharts via LLM

GOAL: Given SC1 + SC2, produce unified statechart preserving behaviors.
Uses Qwen2.5-Coder for intelligent merging decisions.

CHALLENGES:
1. State name conflicts (both have "Idle" state)
2. Event name conflicts (same event, different semantics)
3. Transition merging (combine transition tables)
4. Initial state selection
5. Semantic preservation (merged SC should behave like both)

APPROACH:
1. CONFLICT DETECTION: Find overlapping state/event names
2. CONFLICT RESOLUTION: Rename, merge, or namespace
3. STRUCTURE MERGE: Combine state hierarchies
4. TRANSITION MERGE: Union of transitions with conflict handling
5. VALIDATION: Verify merged SC preserves original behaviors

TARGET: 85%+ semantic preservation on benchmark.

Usage:
    from ml.experiments.exp_qwen_merge import (
        StatechartMerger,
        ConflictResolver,
        MergeBenchmark,
    )

    # Merge two statecharts
    merger = StatechartMerger()
    merged = merger.merge(sc1, sc2)

    # Check semantic preservation
    benchmark = MergeBenchmark()
    result = benchmark.evaluate(sc1, sc2, merged)
"""

from .sc_merger import (
    StatechartMerger,
    MergeConfig,
    MergeResult,
    MergeStrategy,
    merge_statecharts,
)

from .conflict_resolver import (
    ConflictResolver,
    Conflict,
    ConflictType,
    Resolution,
    detect_conflicts,
    resolve_conflicts,
)

from .benchmark import (
    MergeBenchmark,
    BenchmarkResult,
    SemanticTest,
    run_merge_benchmark,
)

__all__ = [
    # Merger
    'StatechartMerger',
    'MergeConfig',
    'MergeResult',
    'MergeStrategy',
    'merge_statecharts',
    # Conflict resolution
    'ConflictResolver',
    'Conflict',
    'ConflictType',
    'Resolution',
    'detect_conflicts',
    'resolve_conflicts',
    # Benchmark
    'MergeBenchmark',
    'BenchmarkResult',
    'SemanticTest',
    'run_merge_benchmark',
]
