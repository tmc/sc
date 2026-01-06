"""
Topology Completion Experiment

Given incomplete statecharts, identify and fix missing elements.

Uses scratchpad enumeration to analyze gaps:
- Reachability check (unreachable states)
- Terminal check (dead ends)
- Completeness check (orphan states)
- Initial state verification

Includes algorithmic and LLM-based completion methods.
"""

from .topology_completer import (
    TopologyAnalyzer,
    TopologyCompleter,
    TopologyGap,
    TopologyFix,
    TopologyAnalysis,
    CompletionResult,
)

from .benchmark import (
    run_benchmark,
    format_report,
    TEST_CASES,
    BenchmarkResult,
)

__all__ = [
    # Core classes
    "TopologyAnalyzer",
    "TopologyCompleter",
    "TopologyGap",
    "TopologyFix",
    "TopologyAnalysis",
    "CompletionResult",
    # Benchmark
    "run_benchmark",
    "format_report",
    "TEST_CASES",
    "BenchmarkResult",
]
