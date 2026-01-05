"""
Hide-and-Seek Emergence Experiment

Evolves seeker and hider strategies from self-play using extractable statecharts.
Inspired by OpenAI's emergent tool use, but with OBSERVABLE strategy extraction.

Grid world features:
- Partial observability (limited vision cone)
- Objects that block line of sight
- Movable obstacles for emergent tool use

Seeker statechart modes (to emerge):
- Search: Explore unseen areas systematically
- Chase: Pursue spotted hider
- Predict: Move to likely hiding spots

Hider statechart modes (to emerge):
- Hide: Stay behind cover
- Evade: Move when spotted
- Distract: Bait and switch tactics
"""

from .grid_world import GridWorld, Cell, Agent, Observation
from .seeker_statechart import SeekerStatechart, SeekerMode
from .hider_statechart import HiderStatechart, HiderMode
from .evolution import HideSeekEvolution, EvolutionConfig
from .run_experiment import run_hide_seek_experiment

__all__ = [
    'GridWorld', 'Cell', 'Agent', 'Observation',
    'SeekerStatechart', 'SeekerMode',
    'HiderStatechart', 'HiderMode',
    'HideSeekEvolution', 'EvolutionConfig',
    'run_hide_seek_experiment',
]
