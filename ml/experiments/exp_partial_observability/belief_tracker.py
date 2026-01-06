"""
Belief Tracker: Bayesian Updates for Statechart Belief States

Implements belief tracking using:
1. Observation model: P(observation | hidden_state)
2. Transition model: P(next_hidden | current_hidden, action)
3. Bayesian update: P(hidden | obs) ∝ P(obs | hidden) * P(hidden)

The tracker maintains beliefs across all parallel dimensions
and updates them based on observations (opponent actions, game events).

KEY INSIGHT: Observation triggers in statechart = likelihood updates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import math
import random

from .belief_state import (
    BeliefDimension,
    BeliefConfiguration,
    ParallelBeliefState,
    BeliefValue,
)


@dataclass
class Observation:
    """
    An observation from the environment.

    In poker: opponent's action (check, bet, raise, fold)
    In general: any visible event that provides info about hidden state.
    """
    name: str
    value: Any
    timestamp: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationModel:
    """
    Defines P(observation | hidden_state) for a belief dimension.

    This is the likelihood function used in Bayesian updates.

    Example for opponent_hand dimension:
    - P(bet | strong) = 0.8
    - P(bet | weak) = 0.2 (bluff)
    - P(check | strong) = 0.2 (trap)
    - P(check | weak) = 0.8
    """
    dimension: str
    likelihoods: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # likelihoods[observation][hidden_value] = probability

    def get_likelihood(self, observation: str, hidden_value: str) -> float:
        """Get P(observation | hidden_value)."""
        if observation in self.likelihoods:
            return self.likelihoods[observation].get(hidden_value, 0.5)
        return 0.5  # Default: uninformative

    def add_likelihood(self, observation: str, hidden_value: str, probability: float):
        """Add a likelihood entry."""
        if observation not in self.likelihoods:
            self.likelihoods[observation] = {}
        self.likelihoods[observation][hidden_value] = probability

    def get_likelihood_dict(self, observation: str) -> Dict[str, float]:
        """Get likelihood dictionary for an observation."""
        return self.likelihoods.get(observation, {})


@dataclass
class TransitionModel:
    """
    Defines P(next_hidden | current_hidden, action) for a dimension.

    Models how hidden state changes over time.
    For poker: hand strength doesn't change, but deck state does.

    Also models belief transitions (strategy inference):
    - After seeing multiple bluffs: P(bluffer | previous) increases
    """
    dimension: str
    transitions: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    # transitions[(current_value, action)][next_value] = probability

    def get_transition_prob(self, current: str, action: str, next_val: str) -> float:
        """Get P(next | current, action)."""
        key = (current, action)
        if key in self.transitions:
            return self.transitions[key].get(next_val, 0.0)
        # Default: stay in same state
        return 1.0 if current == next_val else 0.0

    def add_transition(self, current: str, action: str, next_val: str, probability: float):
        """Add a transition entry."""
        key = (current, action)
        if key not in self.transitions:
            self.transitions[key] = {}
        self.transitions[key][next_val] = probability


class BayesianUpdater:
    """
    Performs Bayesian updates on belief state.

    Core update:
    P(hidden | obs) = P(obs | hidden) * P(hidden) / P(obs)

    Where P(obs) = Σ P(obs | h) * P(h) (normalization constant)
    """

    def __init__(self):
        self.observation_models: Dict[str, ObservationModel] = {}
        self.transition_models: Dict[str, TransitionModel] = {}

    def add_observation_model(self, model: ObservationModel):
        """Register an observation model for a dimension."""
        self.observation_models[model.dimension] = model

    def add_transition_model(self, model: TransitionModel):
        """Register a transition model for a dimension."""
        self.transition_models[model.dimension] = model

    def update_on_observation(
        self,
        belief: ParallelBeliefState,
        observation: Observation,
    ) -> ParallelBeliefState:
        """
        Update belief state given an observation.

        For each dimension that has an observation model for this obs type,
        apply Bayesian update.
        """
        for dim in belief.dimensions:
            if dim.name in self.observation_models:
                model = self.observation_models[dim.name]
                likelihood = model.get_likelihood_dict(observation.name)

                if likelihood:
                    # Apply Bayesian update: posterior ∝ likelihood * prior
                    new_dist = []
                    for i, value in enumerate(dim.possible_values):
                        prior = dim.distribution[i]
                        lik = likelihood.get(value, 0.5)
                        new_dist.append(prior * lik)

                    # Normalize
                    total = sum(new_dist)
                    if total > 0:
                        dim.distribution = [p / total for p in new_dist]

        return belief

    def predict_next(
        self,
        belief: ParallelBeliefState,
        action: str,
    ) -> ParallelBeliefState:
        """
        Predict belief after taking an action (transition update).

        P(next_hidden) = Σ P(next | current, action) * P(current)
        """
        new_belief = belief.copy()

        for dim in new_belief.dimensions:
            if dim.name in self.transition_models:
                model = self.transition_models[dim.name]

                # Compute new distribution
                new_dist = [0.0] * len(dim.possible_values)

                for next_idx, next_val in enumerate(dim.possible_values):
                    for curr_idx, curr_val in enumerate(dim.possible_values):
                        trans_prob = model.get_transition_prob(curr_val, action, next_val)
                        new_dist[next_idx] += trans_prob * dim.distribution[curr_idx]

                # Normalize
                total = sum(new_dist)
                if total > 0:
                    dim.distribution = [p / total for p in new_dist]

        return new_belief


class BeliefTracker:
    """
    Complete belief tracking system.

    Maintains:
    - Current belief state (parallel AND-states)
    - History of observations
    - Observation and transition models
    - Bayesian updater

    Provides:
    - Update on observation
    - Predict after action
    - Belief queries (certainty, entropy, configuration)
    """

    def __init__(self, initial_belief: ParallelBeliefState):
        self.belief = initial_belief.copy()
        self.updater = BayesianUpdater()
        self.observation_history: List[Observation] = []
        self.belief_history: List[BeliefConfiguration] = []

        # Track initial configuration
        self.belief_history.append(self.belief.get_configuration())

    def add_observation_model(self, model: ObservationModel):
        """Add observation model for a dimension."""
        self.updater.add_observation_model(model)

    def add_transition_model(self, model: TransitionModel):
        """Add transition model for a dimension."""
        self.updater.add_transition_model(model)

    def observe(self, observation: Observation) -> BeliefConfiguration:
        """
        Process an observation and update beliefs.

        Returns the new belief configuration.
        """
        observation.timestamp = len(self.observation_history)
        self.observation_history.append(observation)

        # Update belief
        self.belief = self.updater.update_on_observation(self.belief, observation)

        # Track new configuration
        config = self.belief.get_configuration()
        self.belief_history.append(config)

        return config

    def predict(self, action: str) -> ParallelBeliefState:
        """Predict belief after taking an action."""
        return self.updater.predict_next(self.belief, action)

    def get_current_config(self) -> BeliefConfiguration:
        """Get current belief configuration."""
        return self.belief.get_configuration()

    def get_dimension_belief(self, dim_name: str) -> Optional[BeliefDimension]:
        """Get belief for a specific dimension."""
        for dim in self.belief.dimensions:
            if dim.name == dim_name:
                return dim
        return None

    def get_most_likely_value(self, dim_name: str) -> Optional[str]:
        """Get most likely value for a dimension."""
        dim = self.get_dimension_belief(dim_name)
        if dim:
            max_idx = max(range(len(dim.distribution)), key=lambda i: dim.distribution[i])
            return dim.possible_values[max_idx]
        return None

    def get_probability(self, dim_name: str, value: str) -> float:
        """Get probability of a specific value in a dimension."""
        dim = self.get_dimension_belief(dim_name)
        if dim:
            return dim.get_probability(value)
        return 0.0

    def total_certainty(self) -> float:
        """Get average certainty across all dimensions."""
        return self.belief.average_certainty()

    def reset(self):
        """Reset to initial beliefs."""
        self.belief.reset_to_uniform()
        self.observation_history = []
        self.belief_history = [self.belief.get_configuration()]


def create_poker_observation_models() -> List[ObservationModel]:
    """
    Create observation models for poker.

    Maps opponent actions to likelihoods over hidden states.
    """
    models = []

    # Opponent hand model: P(action | hand_strength)
    hand_model = ObservationModel(dimension="opponent_hand")

    # BET observation
    hand_model.add_likelihood("bet", "weak", 0.15)     # Bluff
    hand_model.add_likelihood("bet", "medium", 0.4)   # Value
    hand_model.add_likelihood("bet", "strong", 0.85)  # Strong bet

    # CHECK observation
    hand_model.add_likelihood("check", "weak", 0.85)
    hand_model.add_likelihood("check", "medium", 0.5)
    hand_model.add_likelihood("check", "strong", 0.25)  # Trap

    # RAISE observation
    hand_model.add_likelihood("raise", "weak", 0.1)    # Big bluff
    hand_model.add_likelihood("raise", "medium", 0.3)
    hand_model.add_likelihood("raise", "strong", 0.9)

    # FOLD observation
    hand_model.add_likelihood("fold", "weak", 0.7)
    hand_model.add_likelihood("fold", "medium", 0.3)
    hand_model.add_likelihood("fold", "strong", 0.05)

    # CALL observation
    hand_model.add_likelihood("call", "weak", 0.2)
    hand_model.add_likelihood("call", "medium", 0.6)
    hand_model.add_likelihood("call", "strong", 0.7)

    models.append(hand_model)

    # Opponent strategy model: P(action | strategy_type)
    strategy_model = ObservationModel(dimension="opponent_strategy")

    # BET patterns by strategy
    strategy_model.add_likelihood("bet", "passive", 0.2)
    strategy_model.add_likelihood("bet", "aggressive", 0.7)
    strategy_model.add_likelihood("bet", "bluffer", 0.5)

    # CHECK patterns
    strategy_model.add_likelihood("check", "passive", 0.8)
    strategy_model.add_likelihood("check", "aggressive", 0.3)
    strategy_model.add_likelihood("check", "bluffer", 0.4)

    # RAISE patterns
    strategy_model.add_likelihood("raise", "passive", 0.1)
    strategy_model.add_likelihood("raise", "aggressive", 0.6)
    strategy_model.add_likelihood("raise", "bluffer", 0.4)

    # FOLD patterns
    strategy_model.add_likelihood("fold", "passive", 0.5)
    strategy_model.add_likelihood("fold", "aggressive", 0.2)
    strategy_model.add_likelihood("fold", "bluffer", 0.3)

    models.append(strategy_model)

    return models


def create_poker_transition_models() -> List[TransitionModel]:
    """
    Create transition models for poker.

    Deck state changes as cards are revealed.
    Strategy inference updates over time.
    """
    models = []

    # Deck state transitions (after reveal)
    deck_model = TransitionModel(dimension="deck_state")

    # If high card revealed, deck becomes less high-rich
    deck_model.add_transition("high_rich", "high_revealed", "balanced", 0.5)
    deck_model.add_transition("high_rich", "high_revealed", "low_rich", 0.3)
    deck_model.add_transition("high_rich", "high_revealed", "high_rich", 0.2)

    # Low card revealed
    deck_model.add_transition("low_rich", "low_revealed", "balanced", 0.5)
    deck_model.add_transition("low_rich", "low_revealed", "high_rich", 0.3)
    deck_model.add_transition("low_rich", "low_revealed", "low_rich", 0.2)

    models.append(deck_model)

    return models


def demo():
    """Demonstrate belief tracking."""
    print("=" * 60)
    print("BELIEF TRACKER: Bayesian Updates on Statechart")
    print("=" * 60)

    from .belief_state import create_poker_belief_state

    # Create initial belief
    initial_belief = create_poker_belief_state()
    tracker = BeliefTracker(initial_belief)

    # Add observation models
    for model in create_poker_observation_models():
        tracker.add_observation_model(model)

    print("\n--- Initial Beliefs ---")
    for dim in tracker.belief.dimensions:
        print(f"{dim.name}: {dict(zip(dim.possible_values, [f'{p:.2f}' for p in dim.distribution]))}")

    # Sequence of observations
    observations = [
        Observation(name="bet", value="small"),
        Observation(name="raise", value="big"),
        Observation(name="bet", value="medium"),
    ]

    for obs in observations:
        print(f"\n--- Observation: {obs.name} ---")
        config = tracker.observe(obs)

        print("Updated beliefs:")
        for dim in tracker.belief.dimensions:
            most_likely = tracker.get_most_likely_value(dim.name)
            certainty = dim.certainty()
            print(f"  {dim.name}: {most_likely} (certainty: {certainty:.2f})")

    print(f"\n--- Final Configuration ---")
    print(f"Active states: {tracker.get_current_config().active_states}")
    print(f"Total certainty: {tracker.total_certainty():.3f}")

    print(f"\n--- Observation History ---")
    for obs in tracker.observation_history:
        print(f"  t={obs.timestamp}: {obs.name}")

    return tracker


if __name__ == "__main__":
    demo()
