"""
Trace to Statechart Induction Experiment

Given event traces, induce the statechart that produced them.
This is grammar/automata induction from observed sequences.
"""

from .inducer import (
    SCInducer,
    InductionResult,
    induce_statechart,
)

from .benchmark import (
    InductionBenchmark,
    run_benchmark,
)

__all__ = [
    "SCInducer",
    "InductionResult",
    "induce_statechart",
    "InductionBenchmark",
    "run_benchmark",
]
