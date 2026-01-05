#!/usr/bin/env python3
"""
Evolve Ko Rule from Self-Play

CRITICAL: The Ko structure should EMERGE from training, not be hardcoded!

Approach:
1. Start with minimal statechart (just occupation check, NO Ko handling)
2. Self-play generates games where Ko loops occur (infinite games = bad fitness)
3. Evolution discovers history states that prevent Ko
4. Result: Ko structure EMERGES from training data

This proves statecharts aren't just constraints - they're LEARNABLE!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import copy
from tqdm import tqdm
import json

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState, BoardState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE, EMPTY,
    xy_to_idx, idx_to_xy, get_neighbors, opponent
)


# =============================================================================
# EVOLVABLE KO STATECHART
# =============================================================================

@dataclass
class KoGene:
    """
    Evolvable gene for Ko detection.

    Evolution can discover these features:
    - Remember last capture position
    - Remember last capture count
    - Check if current move would recapture
    - Check if it's a single-stone capture
    """
    # History features (evolution can enable these)
    track_last_capture: bool = False
    track_capture_count: bool = False
    track_board_hash: bool = False
    history_depth: int = 0

    # Gate weights (evolution can tune these)
    single_capture_weight: float = 0.0
    recapture_weight: float = 0.0
    same_position_weight: float = 0.0

    # Threshold for Ko detection
    ko_threshold: float = 0.5


@dataclass
class EvolvableKoState:
    """Runtime state for evolvable Ko detection."""
    last_capture_pos: Optional[Tuple[int, int]] = None
    last_capture_count: int = 0
    last_capturer_pos: Optional[Tuple[int, int]] = None
    board_hashes: List[int] = field(default_factory=list)


class EvolvableKoStatechart:
    """
    Statechart with EVOLVABLE Ko detection.

    Starts with NO Ko handling - evolution must discover it!
    """

    def __init__(self, genome: Optional[Dict] = None):
        """
        Initialize with evolvable genome.

        Default genome has NO Ko handling - must be evolved!
        """
        self.genome = genome or self._minimal_genome()
        self.ko_gene = self._parse_ko_gene(self.genome)
        self.state = EvolvableKoState()

        # Metrics
        self.ko_violations = 0
        self.ko_preventions = 0
        self.infinite_loops = 0

    def _minimal_genome(self) -> Dict:
        """Minimal genome with NO Ko handling."""
        return {
            'version': '1.0',
            'ko': {
                'track_last_capture': False,
                'track_capture_count': False,
                'track_board_hash': False,
                'history_depth': 0,
                'single_capture_weight': 0.0,
                'recapture_weight': 0.0,
                'same_position_weight': 0.0,
                'ko_threshold': 0.5,
            }
        }

    def _parse_ko_gene(self, genome: Dict) -> KoGene:
        """Parse Ko gene from genome."""
        ko_dict = genome.get('ko', {})
        return KoGene(
            track_last_capture=ko_dict.get('track_last_capture', False),
            track_capture_count=ko_dict.get('track_capture_count', False),
            track_board_hash=ko_dict.get('track_board_hash', False),
            history_depth=ko_dict.get('history_depth', 0),
            single_capture_weight=ko_dict.get('single_capture_weight', 0.0),
            recapture_weight=ko_dict.get('recapture_weight', 0.0),
            same_position_weight=ko_dict.get('same_position_weight', 0.0),
            ko_threshold=ko_dict.get('ko_threshold', 0.5),
        )

    def is_ko_forbidden(self, board: BoardState, x: int, y: int,
                        color: int) -> Tuple[bool, float, str]:
        """
        Check if move is Ko-forbidden using evolved features.

        Returns:
            (is_forbidden, confidence, explanation)
        """
        if not self.ko_gene.track_last_capture:
            # No history tracking - can't detect Ko
            return False, 0.0, "No history tracking enabled"

        ko_score = 0.0
        reasons = []

        # Feature 1: Single stone capture at last captured position
        if self.ko_gene.track_capture_count and self.state.last_capture_count == 1:
            if self.state.last_capture_pos == (x, y):
                score = self.ko_gene.single_capture_weight
                ko_score += score
                reasons.append(f"single_capture@same_pos: {score:.2f}")

        # Feature 2: Would this move recapture at the capturer's position?
        if self.state.last_capturer_pos is not None:
            # Check if we're playing adjacent to where opponent just played
            cx, cy = self.state.last_capturer_pos
            if abs(x - cx) + abs(y - cy) == 1:
                score = self.ko_gene.recapture_weight
                ko_score += score
                reasons.append(f"adjacent_to_capturer: {score:.2f}")

        # Feature 3: Same position as last capture
        if self.state.last_capture_pos == (x, y):
            score = self.ko_gene.same_position_weight
            ko_score += score
            reasons.append(f"same_as_captured: {score:.2f}")

        # Feature 4: Board hash repetition (superko)
        if self.ko_gene.track_board_hash and self.ko_gene.history_depth > 0:
            # Simulate the move and check hash
            test_board = board.copy()
            test_board.set(x, y, color)
            test_hash = hash(tuple(test_board.stones))

            if test_hash in self.state.board_hashes[-self.ko_gene.history_depth:]:
                ko_score += 1.0  # Strong signal
                reasons.append("board_hash_repeat: 1.0")

        is_forbidden = ko_score >= self.ko_gene.ko_threshold
        explanation = ", ".join(reasons) if reasons else "no Ko signals"

        return is_forbidden, ko_score, explanation

    def record_move(self, board: BoardState, x: int, y: int, color: int,
                    captured_positions: List[Tuple[int, int]]):
        """Record move for history tracking."""
        if self.ko_gene.track_last_capture and captured_positions:
            self.state.last_capture_count = len(captured_positions)
            if len(captured_positions) == 1:
                self.state.last_capture_pos = captured_positions[0]
            else:
                self.state.last_capture_pos = None
            self.state.last_capturer_pos = (x, y)
        else:
            self.state.last_capture_pos = None
            self.state.last_capture_count = 0
            self.state.last_capturer_pos = None

        if self.ko_gene.track_board_hash:
            board_hash = hash(tuple(board.stones))
            self.state.board_hashes.append(board_hash)
            # Keep limited history
            max_history = max(10, self.ko_gene.history_depth * 2)
            if len(self.state.board_hashes) > max_history:
                self.state.board_hashes = self.state.board_hashes[-max_history:]

    def reset(self):
        """Reset state for new game."""
        self.state = EvolvableKoState()

    def to_genome(self) -> Dict:
        """Export as genome."""
        return {
            'version': '1.0',
            'ko': {
                'track_last_capture': self.ko_gene.track_last_capture,
                'track_capture_count': self.ko_gene.track_capture_count,
                'track_board_hash': self.ko_gene.track_board_hash,
                'history_depth': self.ko_gene.history_depth,
                'single_capture_weight': self.ko_gene.single_capture_weight,
                'recapture_weight': self.ko_gene.recapture_weight,
                'same_position_weight': self.ko_gene.same_position_weight,
                'ko_threshold': self.ko_gene.ko_threshold,
            }
        }


# =============================================================================
# EVOLUTION OPERATORS
# =============================================================================

class KoEvolution:
    """Evolution operators for Ko detection."""

    @staticmethod
    def mutate(genome: Dict, rate: float = 0.2) -> Dict:
        """Mutate Ko genome."""
        new_genome = copy.deepcopy(genome)
        ko = new_genome.setdefault('ko', {})

        # Structural mutations (can enable history tracking)
        if np.random.random() < rate:
            ko['track_last_capture'] = not ko.get('track_last_capture', False)

        if np.random.random() < rate:
            ko['track_capture_count'] = not ko.get('track_capture_count', False)

        if np.random.random() < rate:
            ko['track_board_hash'] = not ko.get('track_board_hash', False)

        if np.random.random() < rate:
            ko['history_depth'] = max(0, ko.get('history_depth', 0) +
                                      np.random.randint(-1, 3))

        # Weight mutations
        if np.random.random() < rate:
            ko['single_capture_weight'] = np.clip(
                ko.get('single_capture_weight', 0) + np.random.normal(0, 0.3),
                0, 2.0
            )

        if np.random.random() < rate:
            ko['recapture_weight'] = np.clip(
                ko.get('recapture_weight', 0) + np.random.normal(0, 0.3),
                0, 2.0
            )

        if np.random.random() < rate:
            ko['same_position_weight'] = np.clip(
                ko.get('same_position_weight', 0) + np.random.normal(0, 0.3),
                0, 2.0
            )

        if np.random.random() < rate:
            ko['ko_threshold'] = np.clip(
                ko.get('ko_threshold', 0.5) + np.random.normal(0, 0.1),
                0.1, 1.5
            )

        return new_genome

    @staticmethod
    def crossover(genome1: Dict, genome2: Dict) -> Dict:
        """Crossover two genomes."""
        child = copy.deepcopy(genome1)
        ko1 = genome1.get('ko', {})
        ko2 = genome2.get('ko', {})
        child_ko = child.setdefault('ko', {})

        for key in ['track_last_capture', 'track_capture_count', 'track_board_hash']:
            if np.random.random() < 0.5:
                child_ko[key] = ko2.get(key, False)

        for key in ['history_depth', 'single_capture_weight', 'recapture_weight',
                    'same_position_weight', 'ko_threshold']:
            if np.random.random() < 0.5:
                child_ko[key] = ko2.get(key, 0)

        return child


# =============================================================================
# GAME WITH EVOLVABLE KO
# =============================================================================

class EvolvableKoGame:
    """
    Go game with evolvable Ko detection.

    This is used to evaluate fitness - games with Ko loops get penalized.
    """

    def __init__(self, ko_statechart: EvolvableKoStatechart):
        self.ko_sc = ko_statechart
        self.board = BoardState()
        self.turn = BLACK
        self.consecutive_passes = 0
        self.move_count = 0
        self.position_history: List[str] = []

        # Metrics
        self.ko_violations = 0
        self.position_repeats = 0

    def is_legal_move(self, x: int, y: int) -> bool:
        """Check if move is legal (occupation + suicide + evolved Ko)."""
        # Occupation check
        if self.board.get(x, y) != EMPTY:
            return False

        # Suicide check (simplified)
        test_board = self.board.copy()
        test_board.set(x, y, self.turn)

        # Check if we capture anything
        captures = False
        opp = opponent(self.turn)
        for nx, ny in get_neighbors(x, y):
            if test_board.get(nx, ny) == opp:
                if test_board.count_liberties(nx, ny) == 0:
                    captures = True
                    break

        if not captures:
            if test_board.count_liberties(x, y) == 0:
                return False  # Suicide

        # Evolved Ko check
        is_ko, _, _ = self.ko_sc.is_ko_forbidden(self.board, x, y, self.turn)
        if is_ko:
            self.ko_sc.ko_preventions += 1
            return False

        return True

    def play_move(self, x: int, y: int) -> bool:
        """Execute a move."""
        if not self.is_legal_move(x, y):
            return False

        # Place stone
        self.board.set(x, y, self.turn)

        # Capture opponent stones
        captured = []
        opp = opponent(self.turn)
        for nx, ny in get_neighbors(x, y):
            if self.board.get(nx, ny) == opp:
                if self.board.count_liberties(nx, ny) == 0:
                    group = self.board.get_group(nx, ny)
                    captured.extend(group)
                    for gx, gy in group:
                        self.board.set(gx, gy, EMPTY)

        # Record for Ko detection
        self.ko_sc.record_move(self.board, x, y, self.turn, captured)

        # Check for position repetition (would indicate Ko failure)
        pos_str = ''.join(str(s) for s in self.board.stones)
        if pos_str in self.position_history:
            self.position_repeats += 1
        self.position_history.append(pos_str)

        # Switch turn
        self.turn = opponent(self.turn)
        self.consecutive_passes = 0
        self.move_count += 1

        return True

    def play_pass(self):
        """Pass."""
        self.turn = opponent(self.turn)
        self.consecutive_passes += 1
        self.move_count += 1
        # Clear Ko state on pass
        self.ko_sc.state.last_capture_pos = None
        self.ko_sc.state.last_capture_count = 0

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.consecutive_passes >= 2

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves."""
        moves = []
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if self.is_legal_move(x, y):
                moves.append((x, y))
        return moves


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

def evaluate_ko_fitness(genome: Dict, num_games: int = 10,
                        max_moves: int = 150) -> Dict:
    """
    Evaluate fitness of a Ko genome.

    Fitness rewards:
    - Completing games without infinite loops
    - Preventing position repetition
    - Having decisive outcomes

    Fitness penalties:
    - Position repeats (failed Ko detection)
    - Infinite games (no convergence)
    """
    total_fitness = 0.0
    metrics = {
        'games_completed': 0,
        'games_infinite': 0,
        'total_repeats': 0,
        'ko_preventions': 0,
        'avg_game_length': 0,
    }

    game_lengths = []

    for _ in range(num_games):
        ko_sc = EvolvableKoStatechart(genome)
        game = EvolvableKoGame(ko_sc)

        # Play random game
        while not game.is_game_over() and game.move_count < max_moves:
            legal = game.get_legal_moves()

            if not legal or np.random.random() < 0.05:
                game.play_pass()
            else:
                x, y = legal[np.random.randint(len(legal))]
                game.play_move(x, y)

        # Evaluate
        game_lengths.append(game.move_count)

        if game.move_count >= max_moves:
            # Infinite game - bad!
            metrics['games_infinite'] += 1
            total_fitness -= 1.0
        else:
            # Completed game - good!
            metrics['games_completed'] += 1
            total_fitness += 0.5

        # Penalize position repeats
        metrics['total_repeats'] += game.position_repeats
        total_fitness -= game.position_repeats * 0.2

        # Reward Ko preventions
        metrics['ko_preventions'] += ko_sc.ko_preventions
        total_fitness += ko_sc.ko_preventions * 0.1

    metrics['avg_game_length'] = np.mean(game_lengths)
    metrics['fitness'] = total_fitness / num_games

    return metrics


# =============================================================================
# EVOLUTION EXPERIMENT
# =============================================================================

def evolve_ko_detection(generations: int = 20, population_size: int = 20,
                        games_per_eval: int = 10):
    """
    Evolve Ko detection from scratch.

    Starting point: NO Ko handling
    Goal: Discover history states that prevent Ko loops
    """
    print("=" * 70)
    print("EVOLVING KO DETECTION FROM SELF-PLAY")
    print("=" * 70)
    print("\nStarting with MINIMAL genome (no Ko handling)...")
    print("Evolution must DISCOVER history states and Ko detection!\n")

    # Initialize population with minimal genomes
    population = []
    base_genome = EvolvableKoStatechart()._minimal_genome()

    for i in range(population_size):
        if i == 0:
            genome = copy.deepcopy(base_genome)
        else:
            # Random mutations to create diversity
            genome = KoEvolution.mutate(base_genome, rate=0.5)
        population.append({'genome': genome, 'fitness': 0.0, 'metrics': {}})

    history = []

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        print("-" * 40)

        # Evaluate fitness
        for individual in tqdm(population, desc="Evaluating"):
            metrics = evaluate_ko_fitness(
                individual['genome'],
                num_games=games_per_eval
            )
            individual['fitness'] = metrics['fitness']
            individual['metrics'] = metrics

        # Sort by fitness
        population.sort(key=lambda x: -x['fitness'])

        # Record history
        best = population[0]
        avg_fitness = np.mean([p['fitness'] for p in population])

        history.append({
            'generation': gen + 1,
            'best_fitness': best['fitness'],
            'avg_fitness': avg_fitness,
            'best_genome': copy.deepcopy(best['genome']),
            'best_metrics': best['metrics'],
        })

        # Print progress
        ko_gene = EvolvableKoStatechart(best['genome']).ko_gene
        print(f"\nBest fitness: {best['fitness']:.3f}")
        print(f"Avg fitness: {avg_fitness:.3f}")
        print(f"Best genome Ko features:")
        print(f"  track_last_capture: {ko_gene.track_last_capture}")
        print(f"  track_capture_count: {ko_gene.track_capture_count}")
        print(f"  track_board_hash: {ko_gene.track_board_hash}")
        print(f"  history_depth: {ko_gene.history_depth}")
        print(f"  single_capture_weight: {ko_gene.single_capture_weight:.2f}")
        print(f"  recapture_weight: {ko_gene.recapture_weight:.2f}")
        print(f"  same_position_weight: {ko_gene.same_position_weight:.2f}")
        print(f"  ko_threshold: {ko_gene.ko_threshold:.2f}")
        print(f"Metrics:")
        print(f"  Games completed: {best['metrics']['games_completed']}/{games_per_eval}")
        print(f"  Position repeats: {best['metrics']['total_repeats']}")
        print(f"  Ko preventions: {best['metrics']['ko_preventions']}")

        # Evolution (except last generation)
        if gen < generations - 1:
            elite_count = population_size // 4
            new_population = population[:elite_count]

            while len(new_population) < population_size:
                # Tournament selection
                idx1, idx2 = np.random.choice(elite_count * 2, size=2, replace=False)
                parent1 = population[idx1]['genome']
                parent2 = population[idx2]['genome']

                # Crossover and mutation
                child = KoEvolution.crossover(parent1, parent2)
                child = KoEvolution.mutate(child, rate=0.3)

                new_population.append({
                    'genome': child,
                    'fitness': 0.0,
                    'metrics': {}
                })

            population = new_population

    # Final analysis
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)

    best = population[0]
    ko_gene = EvolvableKoStatechart(best['genome']).ko_gene

    print("\nFinal Evolved Ko Detection:")
    print(f"  Fitness: {best['fitness']:.3f}")
    print(f"\nDiscovered Features:")

    discovered = []
    if ko_gene.track_last_capture:
        discovered.append("- Track last capture position")
    if ko_gene.track_capture_count:
        discovered.append("- Track capture count (detect single captures)")
    if ko_gene.track_board_hash:
        discovered.append(f"- Track board hash (depth={ko_gene.history_depth})")

    if ko_gene.single_capture_weight > 0.3:
        discovered.append(f"- Single capture weight: {ko_gene.single_capture_weight:.2f}")
    if ko_gene.recapture_weight > 0.3:
        discovered.append(f"- Recapture weight: {ko_gene.recapture_weight:.2f}")
    if ko_gene.same_position_weight > 0.3:
        discovered.append(f"- Same position weight: {ko_gene.same_position_weight:.2f}")

    if discovered:
        print("\n".join(discovered))
    else:
        print("  No Ko features discovered yet (need more generations)")

    # Compare to baseline
    print("\n" + "-" * 40)
    print("Comparison: Baseline vs Evolved")
    print("-" * 40)

    baseline_metrics = evaluate_ko_fitness(
        EvolvableKoStatechart()._minimal_genome(),
        num_games=20
    )
    evolved_metrics = evaluate_ko_fitness(best['genome'], num_games=20)

    print(f"\nBaseline (no Ko handling):")
    print(f"  Fitness: {baseline_metrics['fitness']:.3f}")
    print(f"  Position repeats: {baseline_metrics['total_repeats']}")
    print(f"  Infinite games: {baseline_metrics['games_infinite']}")

    print(f"\nEvolved:")
    print(f"  Fitness: {evolved_metrics['fitness']:.3f}")
    print(f"  Position repeats: {evolved_metrics['total_repeats']}")
    print(f"  Infinite games: {evolved_metrics['games_infinite']}")
    print(f"  Ko preventions: {evolved_metrics['ko_preventions']}")

    improvement = evolved_metrics['fitness'] - baseline_metrics['fitness']
    print(f"\nImprovement: {improvement:+.3f}")

    if improvement > 0:
        print("\n*** Ko structure EMERGED from evolution! ***")
    else:
        print("\n(Need more generations for Ko to fully emerge)")

    # Save results
    results = {
        'generations': generations,
        'population_size': population_size,
        'history': history,
        'best_genome': best['genome'],
        'baseline_metrics': baseline_metrics,
        'evolved_metrics': evolved_metrics,
        'improvement': improvement,
    }

    output_path = Path(__file__).parent / "evolved_ko_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    evolve_ko_detection(generations=15, population_size=15, games_per_eval=8)
