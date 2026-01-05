"""
Evolve Statechart Structure from Gameplay.

Instead of hand-coding which positions matter (corners, edges, X-squares),
EVOLVE the statechart signals through self-play.

Genome encodes:
1. Position values: 64 floats (which squares are valuable?)
2. Flip direction weights: 8 floats (which directions matter?)
3. Phase thresholds: 2 floats (when do phases change?)
4. Signal combination weights: learned importance

Fitness: Win rate against random opponent

Goal: Discover that corners are valuable, X-squares are bad, etc.
without being told.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from othello_game import (
    BOARD_SIZE, ROWS, COLS, BLACK, WHITE, EMPTY,
    CORNERS, X_SQUARES, C_SQUARES, EDGES, DIRECTIONS,
    get_initial_board, get_valid_moves, make_move, is_game_over,
    get_winner, count_pieces, get_all_flips, opponent, idx_to_rc
)


@dataclass
class Genome:
    """Genome encoding statechart structure."""
    # Position values: how valuable is each square?
    position_values: list  # [64]

    # Direction weights: which flip directions matter more?
    direction_weights: list  # [8]

    # Phase thresholds: when do phases change?
    phase_thresholds: list  # [2]

    # Signal combination: [position, flip_count, direction_score, phase_bonus]
    signal_weights: list  # [4]

    # Fitness (cached)
    fitness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "position_values": self.position_values,
            "direction_weights": self.direction_weights,
            "phase_thresholds": self.phase_thresholds,
            "signal_weights": self.signal_weights,
            "fitness": self.fitness,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        return Genome(
            position_values=d["position_values"],
            direction_weights=d["direction_weights"],
            phase_thresholds=d["phase_thresholds"],
            signal_weights=d["signal_weights"],
            fitness=d.get("fitness", 0.0),
        )


def randn():
    """Simple Box-Muller transform for normal distribution."""
    u1 = random.random()
    u2 = random.random()
    import math
    return math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)


def random_genome() -> Genome:
    """Create random genome."""
    return Genome(
        position_values=[randn() * 0.5 for _ in range(64)],
        direction_weights=[randn() * 0.5 for _ in range(8)],
        phase_thresholds=[15.0 + randn() * 5, 45.0 + randn() * 5],
        signal_weights=[abs(randn()) + 0.1 for _ in range(4)],
    )


def mutate(genome: Genome, mutation_rate: float = 0.1, mutation_strength: float = 0.3) -> Genome:
    """Mutate genome with Gaussian noise."""
    new_pos = genome.position_values.copy()
    new_dir = genome.direction_weights.copy()
    new_phase = genome.phase_thresholds.copy()
    new_sig = genome.signal_weights.copy()

    # Position mutations
    for i in range(64):
        if random.random() < mutation_rate:
            new_pos[i] += randn() * mutation_strength

    # Direction mutations
    for i in range(8):
        if random.random() < mutation_rate:
            new_dir[i] += randn() * mutation_strength

    # Phase threshold mutations
    if random.random() < mutation_rate:
        new_phase[0] = max(5, min(40, new_phase[0] + randn() * 3.0))
        new_phase[1] = max(20, min(60, new_phase[1] + randn() * 3.0))

    # Signal weight mutations
    for i in range(4):
        if random.random() < mutation_rate:
            new_sig[i] += randn() * mutation_strength
            new_sig[i] = abs(new_sig[i]) + 0.01

    return Genome(
        position_values=new_pos,
        direction_weights=new_dir,
        phase_thresholds=new_phase,
        signal_weights=new_sig,
    )


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """Uniform crossover between two parents."""
    new_pos = [p1 if random.random() > 0.5 else p2
               for p1, p2 in zip(parent1.position_values, parent2.position_values)]

    new_dir = [p1 if random.random() > 0.5 else p2
               for p1, p2 in zip(parent1.direction_weights, parent2.direction_weights)]

    # Blend phase thresholds
    alpha = random.random()
    new_phase = [alpha * p1 + (1 - alpha) * p2
                 for p1, p2 in zip(parent1.phase_thresholds, parent2.phase_thresholds)]

    new_sig = [p1 if random.random() > 0.5 else p2
               for p1, p2 in zip(parent1.signal_weights, parent2.signal_weights)]

    return Genome(
        position_values=new_pos,
        direction_weights=new_dir,
        phase_thresholds=new_phase,
        signal_weights=new_sig,
    )


class EvolvedPlayer:
    """Player that uses evolved genome for move selection."""

    def __init__(self, genome: Genome):
        self.genome = genome

    def get_phase(self, board: list) -> int:
        """Determine game phase from evolved thresholds."""
        black, white, _ = count_pieces(board)
        total = black + white
        if total < self.genome.phase_thresholds[0]:
            return 0  # Opening
        elif total < self.genome.phase_thresholds[1]:
            return 1  # Midgame
        return 2  # Endgame

    def score_move(self, board: list, pos: int, player: int) -> float:
        """Score a move using evolved genome."""
        flips = get_all_flips(board, pos, player)
        if not flips:
            return -1e9  # Invalid

        # 1. Position value (evolved)
        pos_score = self.genome.position_values[pos]

        # 2. Flip count (weighted by direction)
        flip_score = 0.0
        row, col = idx_to_rc(pos)
        for i, (dr, dc) in enumerate(DIRECTIONS):
            # Count flips in this direction
            dir_flips = 0
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                idx = r * COLS + c
                if board[idx] == opponent(player):
                    if idx in flips:
                        dir_flips += 1
                    r, c = r + dr, c + dc
                else:
                    break
            flip_score += dir_flips * self.genome.direction_weights[i]

        # 3. Phase bonus (different strategies per phase)
        phase = self.get_phase(board)
        if phase == 0:  # Opening: prefer center
            center_dist = abs(row - 3.5) + abs(col - 3.5)
            phase_bonus = -center_dist * 0.1
        elif phase == 1:  # Midgame: use position values more
            phase_bonus = pos_score * 0.5
        else:  # Endgame: maximize flips
            phase_bonus = len(flips) * 0.2

        # 4. Combine with evolved weights
        w = self.genome.signal_weights
        total = (w[0] * pos_score +
                 w[1] * len(flips) +
                 w[2] * flip_score +
                 w[3] * phase_bonus)

        return total

    def get_move(self, board: list, player: int) -> int:
        """Get best move according to evolved genome."""
        valid = get_valid_moves(board, player)
        if not valid:
            return -1

        best_move = valid[0]
        best_score = -1e9

        for move in valid:
            score = self.score_move(board, move, player)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


def evaluate_genome(genome: Genome, num_games: int = 20) -> float:
    """Evaluate genome fitness via games against random."""
    player = EvolvedPlayer(genome)
    wins = 0
    total_margin = 0

    for i in range(num_games):
        board = get_initial_board()
        model_player = BLACK if i % 2 == 0 else WHITE
        current = BLACK
        passes = 0

        while not is_game_over(board) and passes < 2:
            valid = get_valid_moves(board, current)

            if not valid:
                passes += 1
                current = opponent(current)
                continue
            else:
                passes = 0

            if current == model_player:
                move = player.get_move(board, current)
            else:
                move = valid[random.randint(0, len(valid) - 1)]

            board = make_move(board, move, current)
            current = opponent(current)

        winner = get_winner(board)
        black, white, _ = count_pieces(board)
        margin = black - white if model_player == BLACK else white - black

        if winner == model_player:
            wins += 1
            total_margin += margin
        elif winner != 0:
            total_margin += margin  # Negative margin for loss

    # Fitness = win rate + small margin bonus
    win_rate = wins / num_games
    margin_bonus = total_margin / (num_games * 64) * 0.1  # Normalized

    return win_rate + margin_bonus


def run_evolution(
    pop_size: int = 30,
    generations: int = 50,
    elite_size: int = 5,
    mutation_rate: float = 0.15,
    games_per_eval: int = 20,
) -> Tuple[Genome, List[float]]:
    """Run evolutionary algorithm."""

    print("=" * 70)
    print("EVOLVING STATECHART STRUCTURE")
    print("=" * 70)
    print()
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"Elite: {elite_size}, Mutation rate: {mutation_rate}")
    print()

    # Initialize population
    population = [random_genome() for _ in range(pop_size)]

    # Evaluate initial population
    print("Evaluating initial population...")
    for genome in population:
        genome.fitness = evaluate_genome(genome, games_per_eval)

    population.sort(key=lambda g: g.fitness, reverse=True)
    best_fitness_history = [population[0].fitness]

    avg_fit = sum(g.fitness for g in population) / len(population)
    print(f"Gen  0: best={population[0].fitness:.3f}, avg={avg_fit:.3f}")

    # Evolution loop
    for gen in range(1, generations + 1):
        # Select parents (tournament selection)
        new_pop = []

        # Keep elites
        new_pop.extend(population[:elite_size])

        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = [population[random.randint(0, pop_size - 1)] for _ in range(3)]
            t2 = [population[random.randint(0, pop_size - 1)] for _ in range(3)]
            p1 = max(t1, key=lambda g: g.fitness)
            p2 = max(t2, key=lambda g: g.fitness)

            # Crossover
            child = crossover(p1, p2)

            # Mutate
            child = mutate(child, mutation_rate)

            new_pop.append(child)

        population = new_pop[:pop_size]

        # Evaluate new individuals (elites keep their fitness)
        for genome in population[elite_size:]:
            genome.fitness = evaluate_genome(genome, games_per_eval)

        population.sort(key=lambda g: g.fitness, reverse=True)
        best_fitness_history.append(population[0].fitness)

        # Report
        if gen % 5 == 0 or gen == generations:
            avg_fit = sum(g.fitness for g in population) / len(population)
            print(f"Gen {gen:2d}: best={population[0].fitness:.3f}, avg={avg_fit:.3f}")

    print()
    return population[0], best_fitness_history


def mean(vals):
    """Simple mean function."""
    return sum(vals) / len(vals) if vals else 0.0


def analyze_evolved_genome(genome: Genome):
    """Analyze what the evolution discovered."""
    print("=" * 70)
    print("ANALYSIS: EVOLVED STATECHART STRUCTURE")
    print("=" * 70)
    print()

    # 1. Position values heatmap
    print("Position Values (discovered):")
    print("-" * 40)
    pos_vals = genome.position_values

    # Normalize for display
    vmin, vmax = min(pos_vals), max(pos_vals)
    normalized = [(v - vmin) / (vmax - vmin + 1e-6) for v in pos_vals]

    symbols = " ░▒▓█"
    for row in range(8):
        line = ""
        for col in range(8):
            idx = min(4, int(normalized[row * 8 + col] * 4.99))
            line += symbols[idx] + " "
        print(f"  {line}")
    print()

    # 2. Analyze corner vs non-corner
    corner_vals = [genome.position_values[c] for c in CORNERS]
    x_square_vals = [genome.position_values[x] for x in X_SQUARES]
    edge_vals = [genome.position_values[e] for e in EDGES if e not in CORNERS]
    center = [27, 28, 35, 36]  # Center 4 squares
    center_vals = [genome.position_values[c] for c in center]

    print("Position Value Means:")
    print(f"  Corners:   {mean(corner_vals):+.3f}")
    print(f"  X-squares: {mean(x_square_vals):+.3f}")
    print(f"  Edges:     {mean(edge_vals):+.3f}")
    print(f"  Center:    {mean(center_vals):+.3f}")
    print()

    # 3. Direction weights
    print("Direction Weights:")
    dir_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
    for i, (name, weight) in enumerate(zip(dir_names, genome.direction_weights)):
        bar = "█" * max(0, int(abs(weight) * 5))
        sign = "+" if weight >= 0 else "-"
        print(f"  {name}: {sign}{bar} ({weight:.3f})")
    print()

    # 4. Phase thresholds
    print("Phase Thresholds:")
    print(f"  Opening -> Midgame: {genome.phase_thresholds[0]:.1f} pieces")
    print(f"  Midgame -> Endgame: {genome.phase_thresholds[1]:.1f} pieces")
    print()

    # 5. Signal weights
    print("Signal Importance:")
    sig_names = ["Position", "Flip count", "Direction", "Phase bonus"]
    total = sum(genome.signal_weights)
    for name, weight in zip(sig_names, genome.signal_weights):
        pct = weight / total * 100
        bar = "█" * max(0, int(pct / 5))
        print(f"  {name:<12}: {bar} ({pct:.1f}%)")
    print()

    # 6. Compare to hand-coded knowledge
    print("=" * 70)
    print("DISCOVERY VALIDATION")
    print("=" * 70)
    print()

    # Did evolution discover corners are good?
    corner_better = mean(corner_vals) > mean(edge_vals)
    x_bad = mean(x_square_vals) < mean(corner_vals)

    print("Did evolution discover key patterns?")
    corner_mean = mean(corner_vals)
    edge_mean = mean(edge_vals)
    x_mean = mean(x_square_vals)
    print(f"  Corners valuable: {'YES' if corner_better else 'NO'} "
          f"(corners={corner_mean:.2f} vs edges={edge_mean:.2f})")
    print(f"  X-squares risky:  {'YES' if x_bad else 'NO'} "
          f"(x-squares={x_mean:.2f} vs corners={corner_mean:.2f})")
    print()


def compare_evolved_vs_handcoded(evolved_genome: Genome, num_games: int = 100):
    """Compare evolved genome against hand-coded version."""
    print("=" * 70)
    print("EVOLVED VS HAND-CODED COMPARISON")
    print("=" * 70)
    print()

    # Evolved player
    evolved_player = EvolvedPlayer(evolved_genome)

    # Hand-coded "optimal" genome
    handcoded_pos = [0.0] * 64
    for c in CORNERS:
        handcoded_pos[c] = 10.0
    for x in X_SQUARES:
        handcoded_pos[x] = -8.0
    for c in C_SQUARES:
        handcoded_pos[c] = -3.0
    for e in EDGES:
        if e not in CORNERS and e not in C_SQUARES:
            handcoded_pos[e] = 2.0

    handcoded = Genome(
        position_values=handcoded_pos,
        direction_weights=[1.0] * 8,
        phase_thresholds=[20.0, 50.0],
        signal_weights=[1.0, 1.0, 0.5, 0.5],
    )

    handcoded_player = EvolvedPlayer(handcoded)

    # Evaluate both against random
    print("Evaluating vs random (100 games each)...")

    evolved_wins = 0
    handcoded_wins = 0

    for i in range(num_games):
        board = get_initial_board()
        player_color = BLACK if i % 2 == 0 else WHITE
        current = BLACK
        passes = 0

        # Evolved
        board_e = board.copy()
        while not is_game_over(board_e) and passes < 2:
            valid = get_valid_moves(board_e, current)
            if not valid:
                passes += 1
                current = opponent(current)
                continue
            passes = 0

            if current == player_color:
                move = evolved_player.get_move(board_e, current)
            else:
                move = valid[random.randint(0, len(valid) - 1)]

            board_e = make_move(board_e, move, current)
            current = opponent(current)

        if get_winner(board_e) == player_color:
            evolved_wins += 1

        # Hand-coded
        board_h = board.copy()
        current = BLACK
        passes = 0
        while not is_game_over(board_h) and passes < 2:
            valid = get_valid_moves(board_h, current)
            if not valid:
                passes += 1
                current = opponent(current)
                continue
            passes = 0

            if current == player_color:
                move = handcoded_player.get_move(board_h, current)
            else:
                move = valid[random.randint(0, len(valid) - 1)]

            board_h = make_move(board_h, move, current)
            current = opponent(current)

        if get_winner(board_h) == player_color:
            handcoded_wins += 1

    print()
    print(f"{'Method':<15} | {'Win Rate':<10}")
    print("-" * 30)
    print(f"{'Evolved':<15} | {evolved_wins}%")
    print(f"{'Hand-coded':<15} | {handcoded_wins}%")
    print()

    diff = evolved_wins - handcoded_wins
    if diff > 0:
        print(f"Evolution outperforms hand-coding by {diff}%!")
    elif diff < 0:
        print(f"Hand-coding still better by {-diff}%")
    else:
        print("Tied!")
    print()


def run_experiment():
    """Full evolution experiment."""
    random.seed(42)

    # Run evolution
    best_genome, history = run_evolution(
        pop_size=30,
        generations=30,
        elite_size=5,
        mutation_rate=0.15,
        games_per_eval=20,
    )

    # Analyze what was discovered
    analyze_evolved_genome(best_genome)

    # Compare to hand-coded
    compare_evolved_vs_handcoded(best_genome, num_games=100)

    # Save best genome
    with open("evolved_genome.json", "w") as f:
        json.dump(best_genome.to_dict(), f, indent=2)
    print("Saved best genome to evolved_genome.json")

    return best_genome, history


if __name__ == "__main__":
    run_experiment()
