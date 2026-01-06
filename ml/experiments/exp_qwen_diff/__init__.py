"""
exp_qwen_diff: Generate Statechart Diffs via QwenCoder

Goal: Generate human-readable diff summaries for statechart changes.
Target: 95%+ accuracy in detecting added/removed/modified states and transitions.

Key Components:
- ChangeDetector: Programmatically detect changes between statecharts
- SCDiffer: Generate LLM-enhanced summaries using QwenCoder
- Benchmark: Measure detection accuracy

Change Types Detected:
- State changes: ADDED, REMOVED, MODIFIED, RENAMED
- Transition changes: ADDED, REMOVED
- Compatibility assessment: COMPATIBLE, BREAKING, MIGRATION_REQUIRED
"""

from .change_detector import (
    ChangeType,
    Compatibility,
    StateChange,
    TransitionChange,
    DiffResult,
    ChangeDetector,
    detect_changes,
)

from .sc_differ import (
    DiffConfig,
    DiffSummary,
    SCDiffer,
    generate_diff,
)

from .benchmark import (
    ExpectedChange,
    TestCase,
    BenchmarkResult,
    TestCaseGenerator,
    DiffBenchmark,
    run_full_benchmark,
)
