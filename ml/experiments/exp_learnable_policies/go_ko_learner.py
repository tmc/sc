"""
Learning Ko Rule from Self-Play

Goal: Evolve a history state mechanism that detects Ko violations
without explicitly encoding the Ko rule.

Ko Rule: Cannot immediately recapture a single stone that just captured
a single stone of yours.

What we're learning:
- That we need to track "last captured position"
- That we need to track "was it a single capture"
- That recapturing at that position is forbidden
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from enum import Enum

# Import Go statechart
from exp_go_9x9.go_statechart import (
    Go9x9Statechart, BoardState, KoState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE, EMPTY,
    get_neighbors, opponent, xy_to_idx, idx_to_xy
)

try:
    from .framework import (
        StateGenome, PolicyEvolver, GameEnvironment,
        EvolutionConfig, StateType
    )
except ImportError:
    from framework import (
        StateGenome, PolicyEvolver, GameEnvironment,
        EvolutionConfig, StateType
    )


# =============================================================================
# GO ENVIRONMENT FOR KO LEARNING
# =============================================================================

class GoKoEnvironment(GameEnvironment):
    """
    Go environment for learning the Ko rule.

    Key insight: We provide the game WITHOUT Ko checking, and let
    evolution discover what state structure is needed to prevent
    Ko violations.
    """

    def __init__(self):
        self.game = Go9x9Statechart()
        self._last_board: Optional[List[int]] = None
        self._last_capture: Optional[Tuple[int, int]] = None
        self._was_single_capture: bool = False
        self.move_count = 0

    def reset(self):
        self.game = Go9x9Statechart()
        self._last_board = None
        self._last_capture = None
        self._was_single_capture = False
        self.move_count = 0
        return self._get_obs()

    def _get_obs(self):
        """Get current observation."""
        return {
            'board': self.game.board.stones.copy(),
            'turn': self.game.current_player(),
            'move_count': self.move_count,
        }

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get legal moves (ignoring Ko for learning purposes)."""
        # Return moves that are legal EXCEPT we don't filter Ko
        # This lets us detect Ko violations
        moves = []
        turn = self.game.current_player()

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.game.board.get(x, y) == EMPTY:
                    # Check basic legality (not suicide)
                    if self._is_basic_legal(x, y, turn):
                        moves.append((x, y))

        return moves

    def _is_basic_legal(self, x: int, y: int, turn: int) -> bool:
        """Check if move is legal ignoring Ko rule."""
        # Must be empty
        if self.game.board.get(x, y) != EMPTY:
            return False

        # Try placing and check for suicide
        test_board = self.game.board.copy()
        test_board.set(x, y, turn)

        # Would capture anything?
        would_capture = False
        for nx, ny in get_neighbors(x, y):
            if test_board.get(nx, ny) == opponent(turn):
                if test_board.count_liberties(nx, ny) == 0:
                    would_capture = True
                    break

        # Check suicide
        if not would_capture and test_board.count_liberties(x, y) == 0:
            return False

        return True

    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """Take action, return (obs, reward, done, info)."""
        x, y = action

        # Record board before move
        self._last_board = self.game.board.stones.copy()

        # Play move (using the actual game which enforces Ko)
        if self.game.is_legal_move(x, y):
            self.game.play_move(x, y)

            # Track captures
            captures = []
            for idx in range(TOTAL_POINTS):
                if self._last_board[idx] != EMPTY and self.game.board.stones[idx] == EMPTY:
                    cx, cy = idx_to_xy(idx)
                    captures.append((cx, cy))

            if len(captures) == 1:
                self._last_capture = captures[0]
                self._was_single_capture = True
            else:
                self._last_capture = None
                self._was_single_capture = False

            reward = 0.0
            done = self.game.is_game_over()
        else:
            # Illegal move (Ko violation) - this is what we're learning to prevent
            reward = -1.0
            done = False

        self.move_count += 1
        if self.move_count > 200:
            done = True

        return self._get_obs(), reward, done, {}

    def check_special_rule_violation(self, action: Tuple[int, int]) -> bool:
        """Check if this action would violate the Ko rule."""
        x, y = action

        # It's a Ko violation if:
        # 1. Last move captured exactly one stone
        # 2. This move is at the captured position
        # 3. This move would capture exactly one stone (the one that just captured)

        if not self._was_single_capture or self._last_capture is None:
            return False

        if (x, y) != self._last_capture:
            return False

        # Would this capture exactly one stone?
        turn = self.game.current_player()
        test_board = self.game.board.copy()
        test_board.set(x, y, turn)

        capture_count = 0
        for nx, ny in get_neighbors(x, y):
            if test_board.get(nx, ny) == opponent(turn):
                if test_board.count_liberties(nx, ny) == 0:
                    capture_count += len(test_board.get_group(nx, ny))

        return capture_count == 1

    def get_context(self) -> Dict:
        """Get context for guard evaluation."""
        return {
            'board': self.game.board.stones.copy(),
            'turn': self.game.current_player(),
            'last_capture': self._last_capture,
            'was_single': self._was_single_capture,
            'move_count': self.move_count,
        }

    def get_special_rule_name(self) -> str:
        return "Ko Rule"


# =============================================================================
# KO-SPECIFIC GENOME WITH LEARNABLE FEATURES
# =============================================================================

@dataclass
class KoGenome:
    """
    Genome for learning Ko detection.

    Evolves:
    - Whether to track "last captured position"
    - Whether to track "was single capture"
    - Whether to track "would recapture single"
    - Threshold/weights for combining features
    """
    # Feature flags (what to track)
    track_last_capture: bool = False
    track_single_capture: bool = False
    track_would_recapture: bool = False

    # Weights for combining features
    w_single_removed: float = 0.0
    w_move_at_removed: float = 0.0
    w_would_capture_single: float = 0.0
    bias: float = 0.0

    # Fitness
    fitness: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def copy(self) -> 'KoGenome':
        new = KoGenome(
            track_last_capture=self.track_last_capture,
            track_single_capture=self.track_single_capture,
            track_would_recapture=self.track_would_recapture,
            w_single_removed=self.w_single_removed,
            w_move_at_removed=self.w_move_at_removed,
            w_would_capture_single=self.w_would_capture_single,
            bias=self.bias,
        )
        new.fitness = self.fitness
        new.precision = self.precision
        new.recall = self.recall
        new.f1 = self.f1
        return new

    def predict_ko(self, ctx: Dict, action: Tuple[int, int]) -> bool:
        """Predict if this action is a Ko violation."""
        features = self._extract_features(ctx, action)

        logit = self.bias
        if self.track_single_capture:
            logit += self.w_single_removed * features['single_removed']
        if self.track_last_capture:
            logit += self.w_move_at_removed * features['move_at_removed']
        if self.track_would_recapture:
            logit += self.w_would_capture_single * features['would_capture_single']

        return logit > 0

    def _extract_features(self, ctx: Dict, action: Tuple[int, int]) -> Dict[str, float]:
        """Extract features for Ko prediction."""
        x, y = action

        single_removed = 1.0 if ctx.get('was_single', False) else 0.0
        move_at_removed = 1.0 if ctx.get('last_capture') == (x, y) else 0.0

        # Would this move capture exactly one stone?
        would_capture_single = 0.0
        if ctx.get('board'):
            board = ctx['board']
            turn = ctx.get('turn', BLACK)

            # Create test board
            test_board = BoardState()
            test_board.stones = board.copy()
            test_board.set(x, y, turn)

            capture_count = 0
            for nx, ny in get_neighbors(x, y):
                if test_board.get(nx, ny) == opponent(turn):
                    if test_board.count_liberties(nx, ny) == 0:
                        capture_count += len(test_board.get_group(nx, ny))

            would_capture_single = 1.0 if capture_count == 1 else 0.0

        return {
            'single_removed': single_removed,
            'move_at_removed': move_at_removed,
            'would_capture_single': would_capture_single,
        }


# =============================================================================
# KO EVOLVER
# =============================================================================

class KoEvolver:
    """Evolves Ko detection from self-play."""

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig(
            population_size=50,
            n_generations=100,
            games_per_eval=50,
            moves_per_game=100,
        )
        self.population: List[KoGenome] = []
        self.best_genome: Optional[KoGenome] = None
        self.generation = 0
        self.history: List[Dict] = []

    def initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            genome = KoGenome(
                track_last_capture=random.random() > 0.5,
                track_single_capture=random.random() > 0.5,
                track_would_recapture=random.random() > 0.5,
                w_single_removed=random.gauss(0, 5),
                w_move_at_removed=random.gauss(0, 5),
                w_would_capture_single=random.gauss(0, 5),
                bias=random.gauss(0, 5),
            )
            self.population.append(genome)

    def evaluate_genome(self, genome: KoGenome) -> float:
        """Evaluate genome on Ko detection task."""
        env = GoKoEnvironment()

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for _ in range(self.config.games_per_eval):
            env.reset()

            for _ in range(self.config.moves_per_game):
                legal_moves = env.get_legal_moves()
                if not legal_moves:
                    break

                ctx = env.get_context()

                # Test all legal moves
                for action in legal_moves[:10]:  # Sample moves for efficiency
                    is_ko = env.check_special_rule_violation(action)
                    pred_ko = genome.predict_ko(ctx, action)

                    if is_ko and pred_ko:
                        true_positives += 1
                    elif is_ko and not pred_ko:
                        false_negatives += 1
                    elif not is_ko and pred_ko:
                        false_positives += 1
                    else:
                        true_negatives += 1

                # Play a random move
                action = random.choice(legal_moves)
                _, _, done, _ = env.step(action)
                if done:
                    break

        # Compute metrics
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        genome.precision = precision
        genome.recall = recall
        genome.f1 = f1
        genome.fitness = f1  # F1 is our primary metric

        return f1

    def mutate(self, genome: KoGenome) -> KoGenome:
        """Mutate genome."""
        new = genome.copy()

        if random.random() < self.config.mutation_rate:
            mutation = random.choice([
                'toggle_track', 'mutate_weight', 'mutate_bias'
            ])

            if mutation == 'toggle_track':
                choice = random.choice(['last', 'single', 'recapture'])
                if choice == 'last':
                    new.track_last_capture = not new.track_last_capture
                elif choice == 'single':
                    new.track_single_capture = not new.track_single_capture
                else:
                    new.track_would_recapture = not new.track_would_recapture

            elif mutation == 'mutate_weight':
                choice = random.choice(['single', 'at', 'capture'])
                delta = random.gauss(0, 2)
                if choice == 'single':
                    new.w_single_removed += delta
                elif choice == 'at':
                    new.w_move_at_removed += delta
                else:
                    new.w_would_capture_single += delta

            elif mutation == 'mutate_bias':
                new.bias += random.gauss(0, 2)

        return new

    def crossover(self, p1: KoGenome, p2: KoGenome) -> KoGenome:
        """Crossover two genomes."""
        return KoGenome(
            track_last_capture=random.choice([p1.track_last_capture, p2.track_last_capture]),
            track_single_capture=random.choice([p1.track_single_capture, p2.track_single_capture]),
            track_would_recapture=random.choice([p1.track_would_recapture, p2.track_would_recapture]),
            w_single_removed=random.choice([p1.w_single_removed, p2.w_single_removed]),
            w_move_at_removed=random.choice([p1.w_move_at_removed, p2.w_move_at_removed]),
            w_would_capture_single=random.choice([p1.w_would_capture_single, p2.w_would_capture_single]),
            bias=random.choice([p1.bias, p2.bias]),
        )

    def evolve_generation(self) -> None:
        """Run one generation."""
        # Evaluate
        for genome in self.population:
            self.evaluate_genome(genome)

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = self.population[0].copy()

        # Record history
        self.history.append({
            'generation': self.generation,
            'best_f1': self.population[0].f1,
            'best_precision': self.population[0].precision,
            'best_recall': self.population[0].recall,
            'avg_f1': sum(g.f1 for g in self.population) / len(self.population),
        })

        # Create new population
        new_pop = []

        # Elitism
        new_pop.extend([g.copy() for g in self.population[:self.config.elite_count]])

        # Generate offspring
        while len(new_pop) < self.config.population_size:
            # Tournament selection
            t1 = random.sample(self.population, 3)
            t2 = random.sample(self.population, 3)
            p1 = max(t1, key=lambda g: g.fitness)
            p2 = max(t2, key=lambda g: g.fitness)

            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)

        self.population = new_pop[:self.config.population_size]
        self.generation += 1

    def evolve(self, n_generations: int = 100, verbose: bool = True) -> KoGenome:
        """Run evolution."""
        if not self.population:
            self.initialize_population()

        for gen in range(n_generations):
            self.evolve_generation()

            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                h = self.history[-1]
                best = self.population[0]
                print(f"Gen {gen:3d}: F1={h['best_f1']:.3f}, "
                      f"P={h['best_precision']:.3f}, R={h['best_recall']:.3f}")
                print(f"         Features: last={best.track_last_capture}, "
                      f"single={best.track_single_capture}, recapture={best.track_would_recapture}")
                print(f"         Weights: single={best.w_single_removed:.2f}, "
                      f"at={best.w_move_at_removed:.2f}, capture={best.w_would_capture_single:.2f}, "
                      f"bias={best.bias:.2f}")

        return self.best_genome


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("LEARNING KO RULE FROM SELF-PLAY")
    print("=" * 70)
    print()
    print("Goal: Evolve a mechanism that detects Ko violations")
    print("without explicitly encoding the Ko rule.")
    print()

    config = EvolutionConfig(
        population_size=30,
        n_generations=50,
        elite_count=3,
        mutation_rate=0.4,
        games_per_eval=30,
        moves_per_game=80,
    )

    evolver = KoEvolver(config)

    print("-" * 40)
    print("EVOLVING...")
    print("-" * 40)

    best = evolver.evolve(n_generations=50, verbose=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
Learned Ko Detection:

Features Discovered:
  - Track last capture position: {best.track_last_capture}
  - Track single capture:        {best.track_single_capture}
  - Track would recapture:       {best.track_would_recapture}

Learned Weights:
  - w_single_removed:       {best.w_single_removed:+.2f}
  - w_move_at_removed:      {best.w_move_at_removed:+.2f}
  - w_would_capture_single: {best.w_would_capture_single:+.2f}
  - bias:                   {best.bias:+.2f}

Performance:
  - Precision: {best.precision:.1%}
  - Recall:    {best.recall:.1%}
  - F1 Score:  {best.f1:.1%}

Interpretation:
  Ko violation detected when:
    {best.bias:.2f}""")

    if best.track_single_capture:
        print(f"    + {best.w_single_removed:.2f} * (last move captured single stone)")
    if best.track_last_capture:
        print(f"    + {best.w_move_at_removed:.2f} * (move at last captured position)")
    if best.track_would_recapture:
        print(f"    + {best.w_would_capture_single:.2f} * (would capture single stone)")
    print("    > 0")

    # Compare to known optimal
    print()
    print("-" * 40)
    print("COMPARISON TO KNOWN OPTIMAL")
    print("-" * 40)
    print("""
Known optimal (from history_mechanisms.py):
  - single_removed:       +2.4
  - move_at_removed:      +11.5
  - would_capture_single: +10.8
  - bias:                 -19.1

Key insight: The rule is:
  single_removed AND move_at_removed AND would_capture_single

Evolution should discover that ALL THREE features are needed
with positive weights and negative bias (requiring all true).
""")


if __name__ == "__main__":
    main()
