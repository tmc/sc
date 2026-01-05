"""
exp_trace_continuation: Predict next N states given partial trace.

Given:
- A statechart SC
- A partial execution trace (sequence of states)
- A number N

Predict:
- The next N states in the trace

Tests:
- Pattern recognition in traces
- SC structure understanding
- Event inference (what event caused each transition)

Usage:
    from experiments.exp_trace_continuation import TraceContinuer, run_continuation_benchmark

    continuer = TraceContinuer()
    predictions = continuer.predict_next_states(sc_json, partial_trace, n=3)
"""

from .trace_continuer import (
    TraceContinuer,
    ContinuationResult,
    generate_continuation_dataset,
)

from .benchmark import (
    run_continuation_benchmark,
)

__all__ = [
    'TraceContinuer',
    'ContinuationResult',
    'generate_continuation_dataset',
    'run_continuation_benchmark',
]
