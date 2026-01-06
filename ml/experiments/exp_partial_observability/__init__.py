"""
exp_partial_observability: Belief Tracking via Parallel AND-States

KEY RESEARCH CONTRIBUTION:
Model partial observability using orthogonal (AND) statechart regions.
Each parallel region tracks beliefs about ONE hidden dimension.

Approach:
1. AND-state regions for belief dimensions (opponent hand, deck state, etc.)
2. Bayesian updates on observations (actions reveal information)
3. Policy conditioned on belief state (optimal play under uncertainty)

Key insight: Statechart parallel regions ARE a natural representation
for factored belief states in POMDPs.

Example: Poker with Hidden Cards
- Region 1: Opponent hand belief (weak/medium/strong distribution)
- Region 2: Deck state belief (remaining high/low cards)
- Region 3: Opponent strategy belief (aggressive/passive/bluffer)

Observations (opponent actions) trigger belief updates across regions.
Policy selects actions based on joint belief configuration.

Builds on: exp_poker_bluff
"""

from .belief_state import (
    BeliefDimension,
    BeliefConfiguration,
    ParallelBeliefState,
    create_poker_belief_state,
)
from .belief_tracker import (
    BeliefTracker,
    BayesianUpdater,
    ObservationModel,
    TransitionModel,
)
from .poker_beliefs import (
    PokerBeliefState,
    OpponentHandBelief,
    OpponentStrategyBelief,
    DeckStateBelief,
    PokerObservation,
)
from .belief_policy import (
    BeliefPolicy,
    BeliefConditionedAction,
    EvolvedBeliefPolicy,
    PolicyGenome,
)
from .benchmark import run_benchmark

__all__ = [
    'BeliefDimension',
    'BeliefConfiguration',
    'ParallelBeliefState',
    'create_poker_belief_state',
    'BeliefTracker',
    'BayesianUpdater',
    'ObservationModel',
    'TransitionModel',
    'PokerBeliefState',
    'OpponentHandBelief',
    'OpponentStrategyBelief',
    'DeckStateBelief',
    'PokerObservation',
    'BeliefPolicy',
    'BeliefConditionedAction',
    'EvolvedBeliefPolicy',
    'PolicyGenome',
    'run_benchmark',
]
