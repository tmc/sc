"""
Poker Bluffing Evolution from Self-Play

KEY RESEARCH CONTRIBUTION:
Demonstrate that DECEPTION states emerge automatically from imperfect
information games through evolution.

Approach:
1. Simplified poker: 2 players, 3 cards each, high card wins
2. Actions: check, bet, call, fold, raise
3. Imperfect information: can't see opponent's hand
4. Self-play generates games with evolved strategies
5. Evolution discovers profitable bluffing/deception

This shows: DECEPTION STATES ARE LEARNABLE, NOT HAND-CODED!
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# Simple Poker Game
# =============================================================================

class Action(Enum):
    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"
    RAISE = "raise"


@dataclass
class PokerHand:
    """A simplified poker hand (3 cards, high card wins)."""
    cards: List[int]  # Cards are integers 2-14 (2-10, J=11, Q=12, K=13, A=14)

    def strength(self) -> int:
        """Hand strength = highest card."""
        return max(self.cards)

    def normalized_strength(self) -> float:
        """Strength normalized to [0, 1]."""
        return (self.strength() - 2) / 12.0


class SimplePoker:
    """
    Simplified 2-player poker.

    Rules:
    - Each player gets 3 cards (integers 2-14)
    - Betting round: check/bet, then call/fold/raise
    - High card wins
    - Pot starts at 2 (1 ante each)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Deal cards (simple: random integers, may have duplicates)
        self.hands = [
            PokerHand([random.randint(2, 14) for _ in range(3)]),
            PokerHand([random.randint(2, 14) for _ in range(3)])
        ]
        self.pot = 2  # 1 ante each
        self.bets = [1, 1]  # Current round bets
        self.current_player = 0  # Player 0 acts first
        self.action_history: List[Tuple[int, Action]] = []
        self.is_over = False
        self.winner = None
        self.bet_to_call = 0  # Amount needed to call

    def get_legal_actions(self, player: int) -> List[Action]:
        """Get legal actions for player."""
        if self.is_over:
            return []

        if self.bet_to_call == 0:
            # No bet to face
            return [Action.CHECK, Action.BET]
        else:
            # Facing a bet
            return [Action.FOLD, Action.CALL, Action.RAISE]

    def take_action(self, player: int, action: Action) -> bool:
        """Execute an action."""
        if player != self.current_player:
            return False
        if action not in self.get_legal_actions(player):
            return False

        self.action_history.append((player, action))
        opponent = 1 - player

        if action == Action.FOLD:
            self.is_over = True
            self.winner = opponent
            return True

        if action == Action.CHECK:
            if len(self.action_history) >= 2:
                # Both checked - showdown
                self._showdown()
            else:
                self.current_player = opponent
            return True

        if action == Action.BET:
            bet_amount = 2  # Fixed bet size
            self.bets[player] += bet_amount
            self.pot += bet_amount
            self.bet_to_call = bet_amount
            self.current_player = opponent
            return True

        if action == Action.CALL:
            self.bets[player] += self.bet_to_call
            self.pot += self.bet_to_call
            self.bet_to_call = 0
            self._showdown()
            return True

        if action == Action.RAISE:
            raise_amount = self.bet_to_call + 2  # Call + raise
            self.bets[player] += raise_amount
            self.pot += raise_amount
            self.bet_to_call = 2  # New bet for opponent
            self.current_player = opponent
            return True

        return False

    def _showdown(self):
        """Compare hands and determine winner."""
        self.is_over = True
        s0 = self.hands[0].strength()
        s1 = self.hands[1].strength()

        if s0 > s1:
            self.winner = 0
        elif s1 > s0:
            self.winner = 1
        else:
            self.winner = None  # Tie - split pot

    def get_payoff(self, player: int) -> float:
        """Get payoff for player (can be negative)."""
        if not self.is_over:
            return 0

        if self.winner is None:
            # Tie - get back what you put in
            return 0

        if self.winner == player:
            # Won - get pot minus what you put in
            return self.pot - self.bets[player]
        else:
            # Lost - lose your bets
            return -self.bets[player]


# =============================================================================
# Strategy Genome (Evolvable Bluffing States)
# =============================================================================

class StrategyState(Enum):
    """Strategy states that can evolve."""
    PASSIVE = 0      # Check/call only
    AGGRESSIVE = 1   # Bet/raise often
    BLUFF = 2        # Bet with weak hands
    VALUE = 3        # Bet with strong hands only
    TRAP = 4         # Check strong hands, call


@dataclass
class PokerGenome:
    """
    Evolvable poker strategy.

    Key insight: Evolution can discover that bluffing is profitable
    by finding that BLUFF state wins against certain opponents.
    """
    # Thresholds for different behaviors
    bluff_threshold: float = 0.3  # Bluff with hands below this strength
    value_threshold: float = 0.7  # Value bet with hands above this

    # Action probabilities by state
    check_prob: float = 0.5
    bet_prob: float = 0.5
    call_prob: float = 0.5
    raise_prob: float = 0.3
    fold_prob: float = 0.2

    # Bluff frequency
    bluff_frequency: float = 0.2  # How often to bluff

    # Fitness tracking
    fitness: float = 0.0
    total_games: int = 0
    total_profit: float = 0.0
    bluffs_attempted: int = 0
    bluffs_successful: int = 0

    def copy(self) -> 'PokerGenome':
        return PokerGenome(
            bluff_threshold=self.bluff_threshold,
            value_threshold=self.value_threshold,
            check_prob=self.check_prob,
            bet_prob=self.bet_prob,
            call_prob=self.call_prob,
            raise_prob=self.raise_prob,
            fold_prob=self.fold_prob,
            bluff_frequency=self.bluff_frequency
        )

    def get_action(self, game: SimplePoker, player: int) -> Action:
        """Choose action based on evolved strategy."""
        hand = game.hands[player]
        strength = hand.normalized_strength()
        legal = game.get_legal_actions(player)

        if not legal:
            return None

        # Determine if we're bluffing or value betting
        is_bluffing = strength < self.bluff_threshold and random.random() < self.bluff_frequency
        is_value = strength > self.value_threshold

        if Action.CHECK in legal and Action.BET in legal:
            # No bet to face
            if is_value or is_bluffing:
                return Action.BET if random.random() < self.bet_prob else Action.CHECK
            else:
                return Action.CHECK if random.random() < self.check_prob else Action.BET

        if Action.FOLD in legal:
            # Facing a bet
            if strength < self.bluff_threshold and random.random() < self.fold_prob:
                return Action.FOLD

            if is_value and Action.RAISE in legal and random.random() < self.raise_prob:
                return Action.RAISE

            if random.random() < self.call_prob:
                return Action.CALL
            else:
                return Action.FOLD

        return random.choice(legal)


def mutate_poker_genome(genome: PokerGenome) -> PokerGenome:
    """Mutate a poker genome."""
    new = genome.copy()

    # Randomly mutate one or more parameters
    if random.random() < 0.3:
        new.bluff_threshold = max(0.1, min(0.5, new.bluff_threshold + random.gauss(0, 0.1)))
    if random.random() < 0.3:
        new.value_threshold = max(0.5, min(0.9, new.value_threshold + random.gauss(0, 0.1)))
    if random.random() < 0.3:
        new.bluff_frequency = max(0.0, min(0.5, new.bluff_frequency + random.gauss(0, 0.1)))
    if random.random() < 0.2:
        new.bet_prob = max(0.1, min(0.9, new.bet_prob + random.gauss(0, 0.15)))
    if random.random() < 0.2:
        new.call_prob = max(0.1, min(0.9, new.call_prob + random.gauss(0, 0.15)))
    if random.random() < 0.2:
        new.fold_prob = max(0.0, min(0.5, new.fold_prob + random.gauss(0, 0.1)))

    return new


def crossover_poker(g1: PokerGenome, g2: PokerGenome) -> PokerGenome:
    """Crossover two poker genomes."""
    return PokerGenome(
        bluff_threshold=random.choice([g1.bluff_threshold, g2.bluff_threshold]),
        value_threshold=random.choice([g1.value_threshold, g2.value_threshold]),
        bluff_frequency=random.choice([g1.bluff_frequency, g2.bluff_frequency]),
        check_prob=random.choice([g1.check_prob, g2.check_prob]),
        bet_prob=random.choice([g1.bet_prob, g2.bet_prob]),
        call_prob=random.choice([g1.call_prob, g2.call_prob]),
        fold_prob=random.choice([g1.fold_prob, g2.fold_prob])
    )


# =============================================================================
# Self-Play Evaluation
# =============================================================================

def play_match(genome1: PokerGenome, genome2: PokerGenome, n_hands: int = 50) -> Tuple[float, float]:
    """
    Play a match between two strategies.

    Returns payoffs for each player.
    """
    payoffs = [0.0, 0.0]

    for hand_num in range(n_hands):
        game = SimplePoker()

        # Alternate who acts first
        if hand_num % 2 == 1:
            genomes = [genome2, genome1]
            player_map = [1, 0]
        else:
            genomes = [genome1, genome2]
            player_map = [0, 1]

        # Play the hand
        max_actions = 10
        for _ in range(max_actions):
            if game.is_over:
                break

            current = game.current_player
            action = genomes[current].get_action(game, current)
            if action:
                game.take_action(current, action)
            else:
                break

        # Collect payoffs
        for i in range(2):
            payoffs[player_map[i]] += game.get_payoff(i)

    return payoffs[0], payoffs[1]


def evaluate_against_population(genome: PokerGenome, opponents: List[PokerGenome],
                                 games_per_opponent: int = 20) -> float:
    """Evaluate a genome against a population of opponents."""
    total_profit = 0.0
    total_games = 0

    for opp in opponents:
        p1, p2 = play_match(genome, opp, games_per_opponent)
        total_profit += p1
        total_games += games_per_opponent

    genome.fitness = total_profit / max(1, len(opponents))
    genome.total_profit = total_profit
    genome.total_games = total_games

    return genome.fitness


# =============================================================================
# Evolution Loop
# =============================================================================

def evolve_bluff_strategy(
    n_generations: int = 100,
    population_size: int = 30,
    games_per_opponent: int = 30,
    elite_size: int = 3,
    verbose: bool = True
) -> PokerGenome:
    """
    Evolve poker bluffing strategies from self-play.

    Key insight: Bluffing emerges as profitable against passive opponents.
    """

    if verbose:
        print("=" * 60)
        print("POKER BLUFFING EVOLUTION")
        print("=" * 60)
        print(f"Population: {population_size}")
        print(f"Generations: {n_generations}")
        print("-" * 60)

    # Initialize random population
    population = [PokerGenome(
        bluff_threshold=random.uniform(0.1, 0.4),
        value_threshold=random.uniform(0.6, 0.9),
        bluff_frequency=random.uniform(0.0, 0.3),
        bet_prob=random.uniform(0.3, 0.7),
        call_prob=random.uniform(0.3, 0.7),
        fold_prob=random.uniform(0.1, 0.3)
    ) for _ in range(population_size)]

    # Evaluate initial population
    for genome in population:
        evaluate_against_population(genome, population, games_per_opponent)

    best_ever = max(population, key=lambda g: g.fitness).copy()

    for gen in range(n_generations):
        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Elite
        new_population = [g.copy() for g in population[:elite_size]]

        # Generate offspring
        while len(new_population) < population_size:
            if random.random() < 0.7:
                p1 = random.choice(population[:population_size//2])
                p2 = random.choice(population[:population_size//2])
                child = crossover_poker(p1, p2)
            else:
                parent = random.choice(population[:population_size//2])
                child = parent.copy()

            child = mutate_poker_genome(child)
            new_population.append(child)

        # Evaluate against current population
        for genome in new_population:
            evaluate_against_population(genome, new_population, games_per_opponent)

        population = new_population

        # Track best
        gen_best = max(population, key=lambda g: g.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best.copy()
            best_ever.fitness = gen_best.fitness

        # Progress
        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            avg_bluff = sum(g.bluff_frequency for g in population) / len(population)
            avg_value = sum(g.value_threshold for g in population) / len(population)
            print(f"Gen {gen:3d} | Best: {gen_best.fitness:+.2f} "
                  f"| Bluff%: {gen_best.bluff_frequency:.2f} "
                  f"| Value: {gen_best.value_threshold:.2f} "
                  f"| Pop avg bluff: {avg_bluff:.2f}")

    if verbose:
        print("-" * 60)
        print(f"\nBEST EVOLVED STRATEGY:")
        print(f"  Bluff frequency: {best_ever.bluff_frequency:.2f}")
        print(f"  Bluff threshold: {best_ever.bluff_threshold:.2f}")
        print(f"  Value threshold: {best_ever.value_threshold:.2f}")
        print(f"  Bet probability: {best_ever.bet_prob:.2f}")
        print(f"  Call probability: {best_ever.call_prob:.2f}")
        print(f"  Fold probability: {best_ever.fold_prob:.2f}")
        print(f"  Fitness: {best_ever.fitness:+.2f}")

        if best_ever.bluff_frequency > 0.15:
            print(f"\n  *** DECEPTION (BLUFFING) EMERGED! ***")
            print(f"  Strategy bluffs {best_ever.bluff_frequency*100:.0f}% of the time with weak hands")

    return best_ever


# =============================================================================
# Test
# =============================================================================

def test_poker_evolution():
    """Test poker bluffing evolution."""
    print("Testing Poker Bluffing Evolution...\n")

    best = evolve_bluff_strategy(
        n_generations=50,
        population_size=25,
        games_per_opponent=30,
        verbose=True
    )

    print(f"\nFinal bluff frequency: {best.bluff_frequency:.2f}")
    return best


if __name__ == "__main__":
    test_poker_evolution()
