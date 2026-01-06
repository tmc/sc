"""
exp_differentiable_statecharts: End-to-End Differentiable Statechart Learning

Makes statecharts fully differentiable for gradient-based optimization:
- Soft state selection via Gumbel-Softmax
- Differentiable guard evaluation
- Attention-based transition selection
- Learnable topology (soft adjacency matrix)
"""

from .differentiable_statechart import (
    DiffStatechartConfig,
    DifferentiableStatechart,
    DifferentiableGuard,
    SoftTopology,
    TransitionAttention,
    TopologySearcher,
    gumbel_softmax,
    gumbel_softmax_sample,
)

__all__ = [
    "DiffStatechartConfig",
    "DifferentiableStatechart",
    "DifferentiableGuard",
    "SoftTopology",
    "TransitionAttention",
    "TopologySearcher",
    "gumbel_softmax",
    "gumbel_softmax_sample",
]
