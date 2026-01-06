"""
exp_qwen_explanation: Generate Natural Language Explanations of Statecharts

GOAL: Transform statechart specifications into human-readable explanations
using Qwen2.5-Coder-0.5B-Instruct via mlx_lm.

Components:
- explainer.py: Main SC->NL conversion interface
- summary_generator.py: High-level statechart summaries
- detail_generator.py: Detailed transition explanations
- benchmark.py: Clarity rating evaluation (target: 4+/5)

Key insight: Statecharts have structured semantics that can be
systematically explained - states represent situations, transitions
represent reactions, guards represent conditions.
"""

from .explainer import (
    StatechartExplainer,
    ExplanationConfig,
    Explanation,
    explain_statechart,
)
from .summary_generator import (
    SummaryGenerator,
    StatechartSummary,
    generate_summary,
)
from .detail_generator import (
    DetailGenerator,
    TransitionExplanation,
    StateExplanation,
    generate_details,
)
from .benchmark import (
    ClarityBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Main explainer
    'StatechartExplainer',
    'ExplanationConfig',
    'Explanation',
    'explain_statechart',
    # Summary
    'SummaryGenerator',
    'StatechartSummary',
    'generate_summary',
    # Details
    'DetailGenerator',
    'TransitionExplanation',
    'StateExplanation',
    'generate_details',
    # Benchmark
    'ClarityBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
