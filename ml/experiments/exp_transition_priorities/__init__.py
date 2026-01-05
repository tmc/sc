"""
exp_transition_priorities: Evolve Conflict Resolution for Transitions

When multiple transitions are enabled from a state, how do we choose?
This experiment evolves priority assignment strategies instead of hardcoding.

Two competing approaches:
1. Explicit Priority: Each transition has numeric priority, highest wins
2. Specificity Priority: More specific guards (tighter conditions) win

Key insight: The "right" priority scheme is domain-dependent.
Evolution discovers what works without hardcoding assumptions.

Components:
- PriorityStatechart: Statechart with evolvable conflict resolution
- PriorityStrategy: Base class for priority assignment strategies
- ExplicitPriorityStrategy: Use transition.priority field directly
- SpecificityPriorityStrategy: Prioritize by guard specificity
- LearnedPriorityStrategy: Weighted combination of factors
- PriorityEvolver: Evolves both priorities and strategies
"""

from .priority_evolver import (
    PriorityStatechart,
    PriorityStrategy,
    ExplicitPriorityStrategy,
    SpecificityPriorityStrategy,
    LearnedPriorityStrategy,
    PriorityEvolver,
    Transition,
    Guard,
)

__all__ = [
    "PriorityStatechart",
    "PriorityStrategy",
    "ExplicitPriorityStrategy",
    "SpecificityPriorityStrategy",
    "LearnedPriorityStrategy",
    "PriorityEvolver",
    "Transition",
    "Guard",
]
