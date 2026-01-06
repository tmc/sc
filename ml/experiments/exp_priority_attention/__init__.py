"""
exp_priority_attention: Transformer Attention for Transition Selection

Uses learned attention to select which transition fires when multiple are enabled.
Replaces the specificity heuristic from exp_transition_priorities with a context-aware
learned attention mechanism.

Key insight: The "right" transition depends on context. A transformer can learn
complex context-dependent priority patterns that hand-coded heuristics cannot.

Architecture:
- TransitionEncoder: Embeds transitions (guard type, source, target, parameters)
- ContextEncoder: Embeds execution context (grid state, state variables)
- PriorityAttention: Cross-attention where context queries transitions
- Output: Attention weights = transition selection probabilities

Training:
- Evolution-based (no gradient computation needed)
- Can also use gradient descent with differentiable softmax
- Supervised from oracle priority decisions (specificity-based ground truth)

Builds on: exp_transition_priorities
"""

from .attention_priority import (
    Guard,
    Transition,
    TransitionEncoder,
    ContextEncoder,
    PriorityAttention,
    PriorityAttentionModel,
    AttentionPriorityStrategy,
    AttentionStatechart,
    AttentionEvolver,
    create_test_statechart,
    generate_test_cases,
)

__all__ = [
    "Guard",
    "Transition",
    "TransitionEncoder",
    "ContextEncoder",
    "PriorityAttention",
    "PriorityAttentionModel",
    "AttentionPriorityStrategy",
    "AttentionStatechart",
    "AttentionEvolver",
    "create_test_statechart",
    "generate_test_cases",
]
