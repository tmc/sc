"""
Differentiable Statecharts - MLX-based differentiable state machine primitives.

This package provides differentiable implementations of statechart concepts,
enabling gradient-based optimization of state machine behavior.

Modules:
- exp_a_soft_config: Core soft state configuration and guards
- exp_b_memory: Differentiable memory mechanisms (Attention, NTM, Recurrent)
- exp_d_integrated: Memory-augmented statecharts and TRM-inspired hierarchical models
"""

from .exp_a_soft_config import (
    SoftStateConfiguration,
    DifferentiableGuard,
    SoftToggle,
)

from .exp_b_memory import (
    SoftAttentionMemory,
    NTMMemory,
    RecurrentMemory,
    UnifiedMemory,
    MemoryType,
)

from .exp_c_transitions import (
    DifferentiableTransitionSelector,
    TransitionEmbedding,
    StatechartMachine,
)

from .exp_d_integrated import (
    MemoryAugmentedGuard,
    MemoryAugmentedStatechart,
    TRMInspiredStatechart,
    FullyIntegratedStatechart,
)

__all__ = [
    # Core (exp_a)
    "SoftStateConfiguration",
    "DifferentiableGuard",
    "SoftToggle",
    # Memory (exp_b)
    "SoftAttentionMemory",
    "NTMMemory",
    "RecurrentMemory",
    "UnifiedMemory",
    "MemoryType",
    # Transitions (exp_c)
    "DifferentiableTransitionSelector",
    "TransitionEmbedding",
    "StatechartMachine",
    # Integrated (exp_d)
    "MemoryAugmentedGuard",
    "MemoryAugmentedStatechart",
    "TRMInspiredStatechart",
    "FullyIntegratedStatechart",
]
