"""
Game Rule Discovery Pipeline

Discovers game rules as interpretable statecharts from observation traces.

Pipeline:
1. TraceCollector - observe games, collect (state, action, next_state, outcome) tuples
2. StateDiscoverer - cluster observations into meaningful abstract states
3. GuardSynthesizer - learn symbolic guards from transition examples
4. SelfPlayRefiner - validate and refine discovered rules through play

Key insight: Rules are DISCOVERED, not hand-coded.
Output is a fully interpretable statechart.
"""

from .trace_collector import TraceCollector, GameTrace, Transition
from .state_discoverer import StateDiscoverer, AbstractState
from .guard_synthesizer import GuardSynthesizer, SymbolicGuard
from .rule_statechart import RuleStatechart, DiscoveredRule
from .self_play_refiner import SelfPlayRefiner
from .pipeline import RuleDiscoveryPipeline

__all__ = [
    'TraceCollector', 'GameTrace', 'Transition',
    'StateDiscoverer', 'AbstractState',
    'GuardSynthesizer', 'SymbolicGuard',
    'RuleStatechart', 'DiscoveredRule',
    'SelfPlayRefiner',
    'RuleDiscoveryPipeline',
]
