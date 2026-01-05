"""
Experiment: Action Side Effects

Goal: Actions mutate context, guards check context. EVOLVE the dependency
relationships between actions and guards rather than hardcoding them.

Key insight: If action A enables guard G, there's a causal relationship:
  A modifies variable X
  G checks variable X

We LEARN these relationships through:
1. Observing context mutations during action execution
2. Tracking guard evaluation changes after actions
3. Evolving dependency hypotheses that explain observations

NO HARDCODING of action->guard relationships. Everything is learned.
"""

from .action_evolver import (
    ActionEvolver,
    DependencyGenome,
    ActionEffect,
    GuardDependency,
)
from .context_tracker import ContextTracker, ContextDiff
from .causal_learner import CausalLearner, CausalEdge
from .experiment import run_experiment

__all__ = [
    'ActionEvolver',
    'DependencyGenome',
    'ActionEffect',
    'GuardDependency',
    'ContextTracker',
    'ContextDiff',
    'CausalLearner',
    'CausalEdge',
    'run_experiment',
]
