"""
Poker Beliefs: Hidden Card Belief Tracking

Specialized belief tracking for poker domain:
1. OpponentHandBelief - What cards does opponent have?
2. OpponentStrategyBelief - How do they play?
3. DeckStateBelief - What's left in the deck?

Extends exp_poker_bluff with belief-based reasoning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import random
import math

from .belief_state import (
    BeliefDimension,
    ParallelBeliefState,
    BeliefConfiguration,
)
from .belief_tracker import (
    BeliefTracker,
    Observation,
    ObservationModel,
    TransitionModel,
)


class HandStrength(Enum):
    """Categories of hand strength."""
    VERY_WEAK = 0     # Bottom 20%
    WEAK = 1          # 20-40%
    MEDIUM = 2        # 40-60%
    STRONG = 3        # 60-80%
    VERY_STRONG = 4   # Top 20%

    @staticmethod
    def from_percentile(p: float) -> 'HandStrength':
        """Convert percentile (0-1) to strength category."""
        if p < 0.2:
            return HandStrength.VERY_WEAK
        elif p < 0.4:
            return HandStrength.WEAK
        elif p < 0.6:
            return HandStrength.MEDIUM
        elif p < 0.8:
            return HandStrength.STRONG
        else:
            return HandStrength.VERY_STRONG


class PokerAction(Enum):
    """Poker actions (observations from opponent)."""
    CHECK = "check"
    BET_SMALL = "bet_small"
    BET_BIG = "bet_big"
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"
    ALL_IN = "all_in"


@dataclass
class PokerObservation(Observation):
    """Poker-specific observation."""
    action: PokerAction = None
    bet_size: float = 0.0
    pot_odds: float = 0.0
    position: str = ""

    def __post_init__(self):
        if self.action:
            self.name = self.action.value
            self.value = self.bet_size


class OpponentHandBelief(BeliefDimension):
    """
    Belief about opponent's hand strength.

    Updates based on their betting actions:
    - Aggressive actions → likely strong (or bluffing)
    - Passive actions → likely weak (or trapping)
    """

    def __init__(self):
        super().__init__(
            name="opponent_hand",
            possible_values=["very_weak", "weak", "medium", "strong", "very_strong"],
            distribution=[0.2, 0.2, 0.2, 0.2, 0.2],  # Uniform prior
            state_prefix="hand",
        )

        # Action → hand strength likelihood profiles
        self.action_profiles: Dict[str, List[float]] = {
            # [very_weak, weak, medium, strong, very_strong]
            "check": [0.7, 0.6, 0.4, 0.2, 0.1],       # Likely weak, trap possible
            "bet_small": [0.15, 0.3, 0.5, 0.6, 0.4],  # Value or blocking bet
            "bet_big": [0.1, 0.15, 0.3, 0.7, 0.85],   # Strong or bluff
            "call": [0.2, 0.4, 0.6, 0.5, 0.3],        # Drawing or medium
            "raise": [0.1, 0.1, 0.25, 0.6, 0.9],      # Very likely strong
            "fold": [0.8, 0.6, 0.3, 0.1, 0.05],       # Likely weak
            "all_in": [0.15, 0.1, 0.2, 0.5, 0.95],    # Polarized: nuts or bluff
        }

    def update_on_action(self, action: str):
        """Update belief based on observed action."""
        if action in self.action_profiles:
            likelihoods = self.action_profiles[action]
            likelihood_dict = dict(zip(self.possible_values, likelihoods))
            self.update_bayesian(likelihood_dict)

    def get_expected_strength(self) -> float:
        """Get expected strength as percentile."""
        percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]  # Midpoints
        return sum(p * d for p, d in zip(percentiles, self.distribution))


class OpponentStrategyBelief(BeliefDimension):
    """
    Belief about opponent's playing style.

    Inferred from patterns of actions over time:
    - Passive: checks/calls a lot
    - Aggressive: bets/raises frequently
    - Tight: folds often
    - Loose: plays many hands
    - Bluffer: bets with weak hands
    """

    def __init__(self):
        super().__init__(
            name="opponent_strategy",
            possible_values=["passive_tight", "passive_loose", "aggressive_tight",
                            "aggressive_loose", "maniac_bluffer"],
            distribution=[0.25, 0.2, 0.25, 0.2, 0.1],  # Bluffers less common
            state_prefix="strategy",
        )

        # Track action counts for pattern detection
        self.action_counts: Dict[str, int] = {
            "check": 0, "bet": 0, "call": 0, "raise": 0, "fold": 0
        }
        self.total_actions = 0

    def record_action(self, action: str):
        """Record an action for pattern detection."""
        if action.startswith("bet"):
            action = "bet"
        if action in self.action_counts:
            self.action_counts[action] += 1
            self.total_actions += 1

        # Update belief based on action patterns
        self._update_from_patterns()

    def _update_from_patterns(self):
        """Update strategy belief from action patterns."""
        if self.total_actions < 3:
            return  # Need more data

        total = max(1, self.total_actions)
        bet_rate = (self.action_counts["bet"] + self.action_counts["raise"]) / total
        fold_rate = self.action_counts["fold"] / total
        call_rate = self.action_counts["call"] / total

        # Compute likelihoods based on patterns
        likelihoods = {}

        # Passive tight: low bet, high fold
        likelihoods["passive_tight"] = (1 - bet_rate) * fold_rate * 2

        # Passive loose: low bet, low fold, high call
        likelihoods["passive_loose"] = (1 - bet_rate) * call_rate * 2

        # Aggressive tight: high bet, moderate fold
        likelihoods["aggressive_tight"] = bet_rate * (1 + fold_rate)

        # Aggressive loose: high bet, low fold
        likelihoods["aggressive_loose"] = bet_rate * (1 - fold_rate)

        # Maniac bluffer: very high bet, low fold
        likelihoods["maniac_bluffer"] = bet_rate * bet_rate * (1 - fold_rate)

        # Normalize and update
        total_lik = sum(likelihoods.values())
        if total_lik > 0:
            likelihoods = {k: v / total_lik for k, v in likelihoods.items()}
            self.update_bayesian(likelihoods)

    def is_likely_bluffer(self) -> bool:
        """Check if opponent is likely a bluffer."""
        bluff_prob = self.get_probability("maniac_bluffer")
        aggressive_prob = self.get_probability("aggressive_loose")
        return (bluff_prob + aggressive_prob * 0.5) > 0.4


class DeckStateBelief(BeliefDimension):
    """
    Belief about remaining cards in deck.

    Tracks whether remaining deck is:
    - High-card rich (many high cards left)
    - Low-card rich (many low cards left)
    - Balanced

    Updates based on revealed cards.
    """

    def __init__(self):
        super().__init__(
            name="deck_state",
            possible_values=["high_rich", "balanced", "low_rich"],
            distribution=[0.33, 0.34, 0.33],  # Uniform prior
            state_prefix="deck",
        )

        # Track revealed cards
        self.revealed_high = 0  # 10, J, Q, K, A
        self.revealed_low = 0   # 2-9
        self.total_revealed = 0

    def observe_card(self, card_value: int):
        """Observe a revealed card (2-14)."""
        self.total_revealed += 1

        if card_value >= 10:  # 10, J, Q, K, A
            self.revealed_high += 1
        else:
            self.revealed_low += 1

        self._update_belief()

    def _update_belief(self):
        """Update deck state belief based on revealed cards."""
        if self.total_revealed == 0:
            return

        # Compute remaining proportions
        # Standard deck: 20 high cards (10-A), 32 low cards (2-9)
        high_remaining = max(0, 20 - self.revealed_high)
        low_remaining = max(0, 32 - self.revealed_low)
        total_remaining = high_remaining + low_remaining

        if total_remaining == 0:
            return

        high_ratio = high_remaining / total_remaining

        # Update likelihoods
        if high_ratio > 0.45:
            likelihoods = {"high_rich": 0.7, "balanced": 0.25, "low_rich": 0.05}
        elif high_ratio < 0.35:
            likelihoods = {"high_rich": 0.05, "balanced": 0.25, "low_rich": 0.7}
        else:
            likelihoods = {"high_rich": 0.2, "balanced": 0.6, "low_rich": 0.2}

        self.update_bayesian(likelihoods)


@dataclass
class PokerBeliefState:
    """
    Complete poker belief state using parallel regions.

    Integrates:
    - Opponent hand belief
    - Opponent strategy belief
    - Deck state belief

    Each is a parallel region in the statechart.
    """
    opponent_hand: OpponentHandBelief = field(default_factory=OpponentHandBelief)
    opponent_strategy: OpponentStrategyBelief = field(default_factory=OpponentStrategyBelief)
    deck_state: DeckStateBelief = field(default_factory=DeckStateBelief)

    # History
    action_history: List[PokerObservation] = field(default_factory=list)
    card_history: List[int] = field(default_factory=list)

    def observe_action(self, action: PokerAction, bet_size: float = 0.0):
        """Process an opponent action."""
        obs = PokerObservation(action=action, bet_size=bet_size)
        self.action_history.append(obs)

        # Update beliefs
        self.opponent_hand.update_on_action(action.value)
        self.opponent_strategy.record_action(action.value)

    def observe_card(self, card_value: int):
        """Process a revealed card."""
        self.card_history.append(card_value)
        self.deck_state.observe_card(card_value)

    def get_parallel_belief(self) -> ParallelBeliefState:
        """Get as ParallelBeliefState for statechart integration."""
        belief = ParallelBeliefState(name="PokerBelief")
        belief.add_dimension(self.opponent_hand)
        belief.add_dimension(self.opponent_strategy)
        belief.add_dimension(self.deck_state)
        return belief

    def get_configuration(self) -> BeliefConfiguration:
        """Get current belief configuration."""
        return self.get_parallel_belief().get_configuration()

    def expected_opponent_strength(self) -> float:
        """Get expected opponent hand strength."""
        return self.opponent_hand.get_expected_strength()

    def opponent_is_likely_bluffing(self) -> bool:
        """Estimate if opponent is bluffing."""
        # Strong bet + likely bluffer + weak expected hand
        if not self.action_history:
            return False

        last_action = self.action_history[-1].action
        is_aggressive = last_action in [PokerAction.BET_BIG, PokerAction.RAISE, PokerAction.ALL_IN]
        is_bluffer = self.opponent_strategy.is_likely_bluffer()
        expected_weak = self.expected_opponent_strength() < 0.4

        return is_aggressive and (is_bluffer or expected_weak)

    def certainty_score(self) -> float:
        """Overall certainty about hidden information."""
        return (
            self.opponent_hand.certainty() * 0.5 +
            self.opponent_strategy.certainty() * 0.3 +
            self.deck_state.certainty() * 0.2
        )

    def reset(self):
        """Reset all beliefs to priors."""
        self.opponent_hand = OpponentHandBelief()
        self.opponent_strategy = OpponentStrategyBelief()
        self.deck_state = DeckStateBelief()
        self.action_history = []
        self.card_history = []

    def copy(self) -> 'PokerBeliefState':
        """Deep copy."""
        new = PokerBeliefState()
        new.opponent_hand = self.opponent_hand.copy()
        new.opponent_strategy = self.opponent_strategy.copy()
        new.deck_state = self.deck_state.copy()
        new.action_history = list(self.action_history)
        new.card_history = list(self.card_history)
        return new


def create_poker_belief_tracker() -> Tuple[BeliefTracker, PokerBeliefState]:
    """Create a complete poker belief tracking system."""
    poker_belief = PokerBeliefState()
    parallel_belief = poker_belief.get_parallel_belief()
    tracker = BeliefTracker(parallel_belief)

    # Add observation models
    from .belief_tracker import create_poker_observation_models, create_poker_transition_models

    for model in create_poker_observation_models():
        tracker.add_observation_model(model)

    for model in create_poker_transition_models():
        tracker.add_transition_model(model)

    return tracker, poker_belief


def demo():
    """Demonstrate poker belief tracking."""
    print("=" * 60)
    print("POKER BELIEFS: Hidden Card Tracking")
    print("=" * 60)

    poker_belief = PokerBeliefState()

    print("\n--- Initial Beliefs ---")
    print(f"Expected opponent strength: {poker_belief.expected_opponent_strength():.2f}")
    print(f"Opponent likely bluffing: {poker_belief.opponent_is_likely_bluffing()}")
    print(f"Certainty: {poker_belief.certainty_score():.2f}")

    # Simulate a hand
    print("\n--- Hand Simulation ---")

    # Opponent bets big
    print("\nOpponent bets BIG")
    poker_belief.observe_action(PokerAction.BET_BIG, bet_size=100)
    print(f"  Expected strength: {poker_belief.expected_opponent_strength():.2f}")
    print(f"  Likely bluffing: {poker_belief.opponent_is_likely_bluffing()}")

    # We call, opponent raises
    print("\nOpponent RAISES")
    poker_belief.observe_action(PokerAction.RAISE, bet_size=200)
    print(f"  Expected strength: {poker_belief.expected_opponent_strength():.2f}")
    print(f"  Likely bluffing: {poker_belief.opponent_is_likely_bluffing()}")

    # Community cards revealed
    print("\nCommunity cards: K, 7, 3")
    poker_belief.observe_card(13)  # King
    poker_belief.observe_card(7)
    poker_belief.observe_card(3)
    print(f"  Deck state: {poker_belief.deck_state.get_active_state()}")

    # More aggression
    print("\nOpponent goes ALL-IN")
    poker_belief.observe_action(PokerAction.ALL_IN, bet_size=500)
    print(f"  Expected strength: {poker_belief.expected_opponent_strength():.2f}")
    print(f"  Likely bluffing: {poker_belief.opponent_is_likely_bluffing()}")
    print(f"  Strategy belief: {poker_belief.opponent_strategy.get_active_state()}")

    print("\n--- Final Belief Configuration ---")
    config = poker_belief.get_configuration()
    print(f"Active states: {config.active_states}")
    print(f"Certainty: {poker_belief.certainty_score():.2f}")

    # Show statechart representation
    print("\n--- Statechart Representation ---")
    parallel = poker_belief.get_parallel_belief()
    sc = parallel.to_statechart_dict()
    print(f"Root: {sc['root_state']['label']} (AND)")
    for region in sc['root_state']['children']:
        active_child = [c for c in region['children'] if c.get('is_initial')]
        active_name = active_child[0]['label'] if active_child else 'unknown'
        print(f"  Region {region['label']}: {active_name}")

    return poker_belief


if __name__ == "__main__":
    demo()
