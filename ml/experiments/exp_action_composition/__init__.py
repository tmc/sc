"""
Experiment: Action Composition Algebra

Learn how sequential actions compose: A;B = ?
Discover algebraic properties through evolution:
- Commutativity: A;B = B;A ?
- Associativity: (A;B);C = A;(B;C) ?
- Identity: A;I = I;A = A ?
- Inverses: A;A^-1 = I ?
- Idempotence: A;A = A ?
- Absorption: A;B;A = A ?

Builds on exp_action_side_effects.
NO HARDCODING - all rules evolved from examples.

Components:
- CompositionAlgebraLearner: Discovers properties from observation
- CompositionEvolver: Evolves algebra hypotheses
- Action/Effect: Core action representation
- CompositionRule: Learned simplification rules
"""

from .composition_algebra import (
    Action,
    Effect,
    EffectType,
    IDENTITY,
    compose,
    compose_sequence,
    AlgebraicProperty,
    CompositionRule,
    CompositionAlgebraLearner,
    CompositionEvolver,
    CompositionGenome,
    create_demo_actions,
)

__all__ = [
    "Action",
    "Effect",
    "EffectType",
    "IDENTITY",
    "compose",
    "compose_sequence",
    "AlgebraicProperty",
    "CompositionRule",
    "CompositionAlgebraLearner",
    "CompositionEvolver",
    "CompositionGenome",
    "create_demo_actions",
]
