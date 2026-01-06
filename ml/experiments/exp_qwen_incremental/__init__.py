"""
exp_qwen_incremental: Incremental Statechart Updates via Qwen

Goal: Given statechart + NL change request, produce minimal edits.
Target: 90%+ valid edits using diff-based output.

APPROACH:
1. Parse natural language change requests
2. Generate structured diffs (not full replacements)
3. Apply diffs incrementally
4. Validate results

Key Components:
- DiffGenerator: Computes minimal diffs between charts
- DiffApplier: Applies diffs forward/reverse
- ChangeParser: Extracts intent from NL requests
- IncrementalEditor: Combines parsing + LLM for edit generation

Diff Operations:
- add_state, remove_state, modify_state, rename_state
- add_transition, remove_transition, modify_transition
- set_initial, set_property
"""

from .diff_generator import (
    DiffOp,
    DiffOperation,
    StatechartDiff,
    DiffGenerator,
    DiffApplier,
    diff_size,
    chart_size,
    compression_ratio,
)

from .incremental_editor import (
    ChangeType,
    ChangeRequest,
    EditResult,
    ChangeParser,
    IncrementalEditor,
)

from .benchmark import (
    BenchmarkCase,
    CaseResult,
    BenchmarkSummary,
    IncrementalBenchmark,
    create_base_charts,
    create_benchmark_cases,
    demo,
)

__all__ = [
    # Diff generation
    "DiffOp",
    "DiffOperation",
    "StatechartDiff",
    "DiffGenerator",
    "DiffApplier",
    "diff_size",
    "chart_size",
    "compression_ratio",
    # Incremental editing
    "ChangeType",
    "ChangeRequest",
    "EditResult",
    "ChangeParser",
    "IncrementalEditor",
    # Benchmark
    "BenchmarkCase",
    "CaseResult",
    "BenchmarkSummary",
    "IncrementalBenchmark",
    "create_base_charts",
    "create_benchmark_cases",
    "demo",
]
