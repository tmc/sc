"""
SAE + Guard Synthesis Integrated Pipeline

This pipeline combines:
1. SAE (Sparse Autoencoder) to discover interpretable states from neural activations
2. Guard Synthesizer to evolve guard expressions for transitions between states

The goal: Automatically discover both STATES and GUARDS from data.

Architecture:
  Raw Observations → Neural Net → SAE Bottleneck → Discovered States
                                       ↓
  Transition Data → Guard Synthesizer → Learned Guards
                                       ↓
                              Complete Statechart
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_sae_statechart.sae_state_module import TopKSAE, SAEConfig
from exp_guard_synthesis.guard_synthesizer import (
    GuardSynthesizer, GuardGenome, Expr, Var, Const, BinOp, UnaryOp
)


# =============================================================================
# STATE DISCOVERY MODULE (SAE-based)
# =============================================================================

@dataclass
class DiscoveredState:
    """A state discovered from SAE feature patterns."""
    id: int
    active_features: Set[int]
    frequency: int = 0
    label: str = ""

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class TransitionObservation:
    """An observed transition between discovered states."""
    from_state: DiscoveredState
    to_state: DiscoveredState
    context: Dict[str, float]
    step: int


class SAEStateDiscoverer:
    """
    Discovers states from neural network activations using SAE.

    Key idea: SAE features are monosemantic → stable feature patterns = states
    """

    def __init__(self, input_dim: int = 128, expansion_factor: int = 4, k_active: int = 8):
        self.config = SAEConfig(
            input_dim=input_dim,
            expansion_factor=expansion_factor,  # latent_dim = input_dim * expansion_factor
            k_active=k_active,
        )
        self.sae = TopKSAE(self.config)
        self.discovered_states: Dict[frozenset, DiscoveredState] = {}
        self.state_counter = 0

    def encode(self, hidden: mx.array) -> Tuple[mx.array, Set[int]]:
        """Encode hidden state to SAE latent space."""
        acts, indices = self.sae.encode(hidden)

        # Get active feature indices
        if len(hidden.shape) == 1:
            hidden = hidden.reshape(1, -1)

        active_features = set()
        for i in range(indices.shape[1]):
            idx = int(indices[0, i].item())
            if acts[0, idx].item() > 0:
                active_features.add(idx)

        return acts, active_features

    def get_or_create_state(self, active_features: Set[int]) -> DiscoveredState:
        """Get existing state or create new one for this feature pattern."""
        key = frozenset(active_features)

        if key not in self.discovered_states:
            state = DiscoveredState(
                id=self.state_counter,
                active_features=active_features,
                label=f"S{self.state_counter}"
            )
            self.discovered_states[key] = state
            self.state_counter += 1

        state = self.discovered_states[key]
        state.frequency += 1
        return state

    def prune_rare_states(self, min_frequency: int = 2):
        """Remove states that appear too rarely."""
        self.discovered_states = {
            k: v for k, v in self.discovered_states.items()
            if v.frequency >= min_frequency
        }


# =============================================================================
# GUARD SYNTHESIS MODULE
# =============================================================================

class TransitionGuardLearner:
    """
    Learns guard conditions for transitions between discovered states.

    For each (from_state, to_state) pair, evolves a guard expression
    that predicts when this transition should fire.
    """

    def __init__(self, context_vars: List[str], population_size: int = 20):
        self.context_vars = context_vars
        self.constants = [0.0, 0.5, 1.0, -1.0, 10.0, 30.0, 50.0, 100.0]  # Common thresholds
        self.population_size = population_size
        self.learned_guards: Dict[Tuple[int, int], GuardGenome] = {}
        self.transition_data: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

    def record_transition(self, from_state: DiscoveredState,
                          to_state: DiscoveredState, context: Dict[str, float]):
        """Record an observed transition with its context."""
        key = (from_state.id, to_state.id)
        self.transition_data[key].append(context)

    def record_non_transition(self, from_state: DiscoveredState,
                               to_state: DiscoveredState, context: Dict[str, float]):
        """Record when a transition did NOT fire (negative example)."""
        key = (from_state.id, to_state.id)
        # Store as negative example (prefix with 'neg_')
        neg_ctx = {f'neg_{k}': v for k, v in context.items()}
        self.transition_data[key].append(neg_ctx)

    def learn_guards(self, n_generations: int = 30) -> Dict[Tuple[int, int], GuardGenome]:
        """Learn guards for all observed transitions."""
        for key, contexts in self.transition_data.items():
            # Separate positive and negative examples
            positive = [c for c in contexts if not any(k.startswith('neg_') for k in c)]
            negative_raw = [c for c in contexts if any(k.startswith('neg_') for k in c)]
            negative = [{k.replace('neg_', ''): v for k, v in c.items()}
                       for c in negative_raw]

            if len(positive) < 2:
                continue  # Not enough data

            synthesizer = GuardSynthesizer(
                variables=self.context_vars,
                constants=self.constants,
                max_depth=3
            )

            # Need at least some negative examples for evolution
            if not negative:
                # Use empty list if no negatives
                negative = []

            best = synthesizer.evolve(
                positive_examples=positive,
                negative_examples=negative,
                population_size=self.population_size,
                n_generations=n_generations,
                verbose=False
            )

            self.learned_guards[key] = best

        return self.learned_guards


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================

@dataclass
class LearnedStatechart:
    """A statechart learned from data via SAE + Guard Synthesis."""
    states: List[DiscoveredState]
    transitions: List[Dict]
    guards: Dict[Tuple[int, int], str]


class SAEGuardPipeline:
    """
    Complete pipeline: observations → states (SAE) → guards (evolution)

    Usage:
        pipeline = SAEGuardPipeline(context_vars=['x', 'y', 'score'])

        # Process sequence of observations
        for obs, ctx in data:
            pipeline.process(obs, ctx)

        # Learn the statechart
        chart = pipeline.learn()
    """

    def __init__(self, input_dim: int = 128, expansion_factor: int = 4,
                 k_active: int = 8, context_vars: List[str] = None):
        self.state_discoverer = SAEStateDiscoverer(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            k_active=k_active
        )
        self.context_vars = context_vars or []
        self.guard_learner = TransitionGuardLearner(
            context_vars=self.context_vars,
            population_size=20
        )

        self.current_state: Optional[DiscoveredState] = None
        self.observations: List[Tuple[mx.array, Dict, DiscoveredState]] = []

    def process(self, hidden: mx.array, context: Dict[str, float]) -> DiscoveredState:
        """
        Process an observation through the pipeline.

        Args:
            hidden: Neural network hidden state (input_dim,)
            context: Context variables for guard synthesis

        Returns:
            The discovered state for this observation
        """
        if len(hidden.shape) == 1:
            hidden = hidden.reshape(1, -1)

        # Discover state via SAE
        _, active_features = self.state_discoverer.encode(hidden)
        state = self.state_discoverer.get_or_create_state(active_features)

        # Record transition if state changed
        if self.current_state is not None and self.current_state != state:
            self.guard_learner.record_transition(
                self.current_state, state, context
            )

        self.observations.append((hidden, context, state))
        self.current_state = state
        return state

    def learn(self, n_generations: int = 30,
              min_state_frequency: int = 2) -> LearnedStatechart:
        """
        Learn the complete statechart from recorded observations.

        Args:
            n_generations: Generations for guard evolution
            min_state_frequency: Minimum state occurrences to keep

        Returns:
            LearnedStatechart with states, transitions, and guards
        """
        # Prune rare states
        self.state_discoverer.prune_rare_states(min_state_frequency)

        # Learn guards for transitions
        learned_guards = self.guard_learner.learn_guards(n_generations)

        # Build output
        states = list(self.state_discoverer.discovered_states.values())

        transitions = []
        guards = {}

        for (from_id, to_id), genome in learned_guards.items():
            trans = {
                'from': from_id,
                'to': to_id,
                'guard': str(genome.expr) if genome.expr else 'true',
                'fitness': genome.fitness
            }
            transitions.append(trans)
            guards[(from_id, to_id)] = str(genome.expr) if genome.expr else 'true'

        return LearnedStatechart(
            states=states,
            transitions=transitions,
            guards=guards
        )


# =============================================================================
# DEMO: SIMPLE GAME
# =============================================================================

class SimpleGameEnv:
    """
    Simple game for testing the pipeline.

    States:
    - Idle (x near 0, y near 0)
    - Moving (x or y changing)
    - Active (score > 0)
    - Critical (health < 20)
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.score = 0
        self.health = 100
        self.step_count = 0

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.score = 0
        self.health = 100
        self.step_count = 0

    def step(self, action: int) -> Tuple[np.ndarray, Dict]:
        """
        Take action and return (hidden_state, context).

        Actions: 0=idle, 1=move_x, 2=move_y, 3=score, 4=damage
        """
        self.step_count += 1

        if action == 1:
            self.x += random.uniform(-1, 1)
        elif action == 2:
            self.y += random.uniform(-1, 1)
        elif action == 3:
            self.score += 1
        elif action == 4:
            self.health -= random.randint(5, 15)

        # Create hidden state (simulated neural network output)
        hidden = np.zeros(128)
        hidden[0:10] = self.x
        hidden[10:20] = self.y
        hidden[20:30] = self.score / 10.0
        hidden[30:40] = self.health / 100.0
        hidden[40:50] = 1.0 if abs(self.x) > 0.5 or abs(self.y) > 0.5 else 0.0
        hidden[50:60] = 1.0 if self.health < 30 else 0.0
        # Add some noise
        hidden += np.random.randn(128) * 0.1

        context = {
            'x': self.x,
            'y': self.y,
            'score': float(self.score),
            'health': float(self.health),
            'moving': 1.0 if abs(self.x) > 0.5 or abs(self.y) > 0.5 else 0.0,
            'low_health': 1.0 if self.health < 30 else 0.0,
        }

        return hidden, context


def demo_pipeline():
    """Demonstrate the SAE + Guard Synthesis pipeline."""
    print("=" * 70)
    print("SAE + GUARD SYNTHESIS INTEGRATED PIPELINE DEMO")
    print("=" * 70)
    print()
    print("Goal: Automatically discover STATES and GUARDS from game data")
    print()

    # Create pipeline
    context_vars = ['x', 'y', 'score', 'health', 'moving', 'low_health']
    pipeline = SAEGuardPipeline(
        input_dim=128,
        expansion_factor=4,  # 128 * 4 = 512 latent features
        k_active=5,
        context_vars=context_vars
    )

    # Generate game data
    env = SimpleGameEnv()
    print("Generating game data...")

    n_episodes = 20
    steps_per_episode = 50

    for ep in range(n_episodes):
        env.reset()
        for _ in range(steps_per_episode):
            # Random action with some bias
            action = random.choices(
                [0, 1, 2, 3, 4],
                weights=[0.3, 0.2, 0.2, 0.2, 0.1]
            )[0]

            hidden, context = env.step(action)
            state = pipeline.process(mx.array(hidden), context)

    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Total observations: {len(pipeline.observations)}")
    print(f"  Unique state patterns discovered: {len(pipeline.state_discoverer.discovered_states)}")
    print()

    # Learn statechart
    print("Learning statechart...")
    chart = pipeline.learn(n_generations=20, min_state_frequency=5)

    print()
    print("=" * 70)
    print("DISCOVERED STATECHART")
    print("=" * 70)
    print()

    print(f"States: {len(chart.states)}")
    for state in sorted(chart.states, key=lambda s: -s.frequency)[:10]:
        print(f"  {state.label}: {len(state.active_features)} features, "
              f"frequency={state.frequency}")

    print()
    print(f"Transitions with Learned Guards: {len(chart.transitions)}")
    for trans in sorted(chart.transitions, key=lambda t: -t['fitness'])[:10]:
        print(f"  S{trans['from']} → S{trans['to']}: "
              f"guard=[{trans['guard']}], fitness={trans['fitness']:.3f}")

    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The pipeline demonstrates that:

1. SAE discovers states automatically from neural activations
   - Stable activation patterns = discrete states
   - Feature sparsity = interpretable state representation

2. Guard synthesis evolves transition conditions
   - Positive/negative examples from observed transitions
   - Guards are human-readable expressions

3. Combined: Complete statechart learned from data
   - No hand-coding of states or guards
   - Emergent structure from observation
""")


if __name__ == "__main__":
    demo_pipeline()
