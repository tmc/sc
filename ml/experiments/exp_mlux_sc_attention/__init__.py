"""
exp_mlux_sc_attention: Analyze attention patterns during SC generation.

Goal: Identify attention heads and patterns that correlate with
statechart structure, hierarchy, and validity.

Key Components:
- structure_heads: Find heads that attend to structural tokens
- hierarchy_attention: Analyze parent-child attention patterns
- validity_diff: Compare attention for valid vs invalid SCs

Key Findings (when mlux available):
1. Structure-aware heads: Later layers attend more to braces/brackets
2. Hierarchy heads: Specific heads track parent-child relationships
3. Validity patterns: Attention differs for valid vs invalid structures

Use Cases:
- Early error detection during generation
- Steering toward valid structures
- Understanding model's implicit validation
"""

from .structure_heads import (
    STRUCTURE_TOKENS,
    HeadScore,
    StructureHeadAnalysis,
    StructureHeadDetector,
)

from .hierarchy_attention import (
    HierarchyNode,
    HierarchyAttentionScore,
    HierarchyAnalysis,
    HierarchyAttentionAnalyzer,
)

from .validity_diff import (
    AttentionDiff,
    ValidityPattern,
    ValidityAnalysis,
    ValidityDiffAnalyzer,
)

from .benchmark import (
    BenchmarkResult,
    TEST_STATECHARTS,
    SCAttentionBenchmark,
    demo,
    run_full_benchmark,
)

__all__ = [
    # structure_heads
    "STRUCTURE_TOKENS",
    "HeadScore",
    "StructureHeadAnalysis",
    "StructureHeadDetector",
    # hierarchy_attention
    "HierarchyNode",
    "HierarchyAttentionScore",
    "HierarchyAnalysis",
    "HierarchyAttentionAnalyzer",
    # validity_diff
    "AttentionDiff",
    "ValidityPattern",
    "ValidityAnalysis",
    "ValidityDiffAnalyzer",
    # benchmark
    "BenchmarkResult",
    "TEST_STATECHARTS",
    "SCAttentionBenchmark",
    "demo",
    "run_full_benchmark",
]
