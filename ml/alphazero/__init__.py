"""
StatechartAlphaZero: AlphaZero with Statechart-Guaranteed Legal Moves

This module integrates statecharts with AlphaZero for 9x9 Go using MLX.

Key innovations:
1. Replace flat board encoding with soft state configurations
2. Guards guarantee 100% legal moves by construction
3. Observable, evolvable policy statechart for interpretability

Components:
- StatechartMCTS: Guard-masked MCTS with 0% illegal moves
- PolicyStatechart: Observable policy engine with strategic modes
- ObservableMCTS: Full introspection of policy decisions
- Evolution operators: Mutate topology and guards
"""

from .go9x9_game import Go9x9Game
from .statechart_encoder import StatechartEncoder, encode_soft_config
from .statechart_nnet import StatechartAlphaZeroNet, ResBlock
from .statechart_mcts import StatechartMCTS
from .statechart_coach import StatechartCoach
from .policy_statechart import (
    PolicyStatechart, PolicyGuards, PolicyEvolution,
    StrategicMode, Urgency, Phase, Focus
)
from .observable_mcts import ObservableMCTS, EvolvableMCTSPopulation

__all__ = [
    # Core AlphaZero components
    'Go9x9Game',
    'StatechartEncoder',
    'encode_soft_config',
    'StatechartAlphaZeroNet',
    'ResBlock',
    'StatechartMCTS',
    'StatechartCoach',

    # Observable policy engine
    'PolicyStatechart',
    'PolicyGuards',
    'PolicyEvolution',
    'StrategicMode',
    'Urgency',
    'Phase',
    'Focus',

    # Observable MCTS
    'ObservableMCTS',
    'EvolvableMCTSPopulation',
]
