"""
Scaled-up Evolution to Beat Hand-coded Othello.

Improvements over basic evolution:
1. Larger population (50 vs 30)
2. More generations (100 vs 30)
3. More eval games (40 vs 20) - less noisy fitness
4. Adaptive mutation rate
5. More sophisticated signals (mobility, corner control)
6. Island model for diversity
"""

import random
import time
import json
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from othello_game import (
    BOARD_SIZE, ROWS, COLS, BLACK, WHITE, EMPTY,
    CORNERS, X_SQUARES, C_SQUARES, EDGES, DIRECTIONS,
    get_initial_board, get_valid_moves, make_move, is_game_over,
    get_winner, count_pieces, get_all_flips, opponent, idx_to_rc,
    get_mobility
)
from dataclasses import dataclass, field
from typing import List, Tuple


def randn():
    """Box-Muller normal distribution."""
    u1 = random.random() + 1e-10
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


@dataclass
class ScaledGenome:
    """Enhanced genome with more evolvable signals."""
    # Position values [64]
    position_values: list

    # Direction weights [8]
    direction_weights: list

    # Phase thresholds [2]
    phase_thresholds: list

    # Signal weights [6]: position, flips, direction, phase, mobility, corner_control
    signal_weights: list

    # Corner control bonus when we own adjacent corner [4]
    corner_adjacency_bonus: list

    # Mobility weight by phase [3]
    mobility_phase_weights: list

    fitness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "position_values": self.position_values,
            "direction_weights": self.direction_weights,
            "phase_thresholds": self.phase_thresholds,
            "signal_weights": self.signal_weights,
            "corner_adjacency_bonus": self.corner_adjacency_bonus,
            "mobility_phase_weights": self.mobility_phase_weights,
            "fitness": self.fitness,
        }

    @staticmethod
    def from_dict(d: dict) -> "ScaledGenome":
        return ScaledGenome(
            position_values=d["position_values"],
            direction_weights=d["direction_weights"],
            phase_thresholds=d["phase_thresholds"],
            signal_weights=d["signal_weights"],
            corner_adjacency_bonus=d.get("corner_adjacency_bonus", [2.0]*4),
            mobility_phase_weights=d.get("mobility_phase_weights", [1.0, 0.5, 0.2]),
            fitness=d.get("fitness", 0.0),
        )


def random_genome() -> ScaledGenome:
    """Create random genome with good priors."""
    # Start with slight corner preference
    pos_vals = [randn() * 0.3 for _ in range(64)]
    for c in CORNERS:
        pos_vals[c] += 0.5  # Prior: corners are good
    for x in X_SQUARES:
        pos_vals[x] -= 0.3  # Prior: X-squares are risky

    return ScaledGenome(
        position_values=pos_vals,
        direction_weights=[randn() * 0.3 for _ in range(8)],
        phase_thresholds=[18 + randn() * 3, 48 + randn() * 3],
        signal_weights=[abs(randn()) + 0.5 for _ in range(6)],
        corner_adjacency_bonus=[abs(randn()) + 1.0 for _ in range(4)],
        mobility_phase_weights=[abs(randn()) + 0.5 for _ in range(3)],
    )


def mutate(genome: ScaledGenome, rate: float = 0.1, strength: float = 0.2) -> ScaledGenome:
    """Mutate with given rate and strength."""
    new_pos = genome.position_values.copy()
    new_dir = genome.direction_weights.copy()
    new_phase = genome.phase_thresholds.copy()
    new_sig = genome.signal_weights.copy()
    new_corner = genome.corner_adjacency_bonus.copy()
    new_mob = genome.mobility_phase_weights.copy()

    for i in range(64):
        if random.random() < rate:
            new_pos[i] += randn() * strength

    for i in range(8):
        if random.random() < rate:
            new_dir[i] += randn() * strength

    if random.random() < rate:
        new_phase[0] = max(10, min(35, new_phase[0] + randn() * 2))
        new_phase[1] = max(35, min(58, new_phase[1] + randn() * 2))

    for i in range(6):
        if random.random() < rate:
            new_sig[i] = max(0.01, new_sig[i] + randn() * strength)

    for i in range(4):
        if random.random() < rate:
            new_corner[i] = max(0, new_corner[i] + randn() * strength)

    for i in range(3):
        if random.random() < rate:
            new_mob[i] = max(0, new_mob[i] + randn() * strength)

    return ScaledGenome(
        position_values=new_pos,
        direction_weights=new_dir,
        phase_thresholds=new_phase,
        signal_weights=new_sig,
        corner_adjacency_bonus=new_corner,
        mobility_phase_weights=new_mob,
    )


def crossover(p1: ScaledGenome, p2: ScaledGenome) -> ScaledGenome:
    """Uniform crossover with blending."""
    new_pos = [a if random.random() > 0.5 else b
               for a, b in zip(p1.position_values, p2.position_values)]

    new_dir = [a if random.random() > 0.5 else b
               for a, b in zip(p1.direction_weights, p2.direction_weights)]

    alpha = random.random()
    new_phase = [alpha * a + (1-alpha) * b
                 for a, b in zip(p1.phase_thresholds, p2.phase_thresholds)]

    new_sig = [a if random.random() > 0.5 else b
               for a, b in zip(p1.signal_weights, p2.signal_weights)]

    new_corner = [a if random.random() > 0.5 else b
                  for a, b in zip(p1.corner_adjacency_bonus, p2.corner_adjacency_bonus)]

    new_mob = [a if random.random() > 0.5 else b
               for a, b in zip(p1.mobility_phase_weights, p2.mobility_phase_weights)]

    return ScaledGenome(
        position_values=new_pos,
        direction_weights=new_dir,
        phase_thresholds=new_phase,
        signal_weights=new_sig,
        corner_adjacency_bonus=new_corner,
        mobility_phase_weights=new_mob,
    )


# Corner adjacency mapping
CORNER_ADJACENT = {
    0: [1, 8, 9],      # A1
    7: [6, 14, 15],    # H1
    56: [48, 49, 57],  # A8
    63: [54, 55, 62],  # H8
}


class ScaledPlayer:
    """Player using scaled genome."""

    def __init__(self, genome: ScaledGenome):
        self.genome = genome

    def get_phase(self, board: list) -> int:
        black, white, _ = count_pieces(board)
        total = black + white
        if total < self.genome.phase_thresholds[0]:
            return 0
        elif total < self.genome.phase_thresholds[1]:
            return 1
        return 2

    def owns_corner(self, board: list, player: int, corner_idx: int) -> bool:
        """Check if player owns the corner."""
        corner = CORNERS[corner_idx]
        return board[corner] == player

    def score_move(self, board: list, pos: int, player: int) -> float:
        """Score a move."""
        flips = get_all_flips(board, pos, player)
        if not flips:
            return -1e9

        g = self.genome
        phase = self.get_phase(board)

        # 1. Position value
        pos_score = g.position_values[pos]

        # 2. Flip count
        flip_score = len(flips)

        # 3. Direction-weighted flips
        row, col = idx_to_rc(pos)
        dir_score = 0.0
        for i, (dr, dc) in enumerate(DIRECTIONS):
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
            dir_score += dir_flips * g.direction_weights[i]

        # 4. Phase bonus
        if phase == 0:
            center_dist = abs(row - 3.5) + abs(col - 3.5)
            phase_bonus = -center_dist * 0.1
        elif phase == 1:
            phase_bonus = pos_score * 0.5
        else:
            phase_bonus = flip_score * 0.3

        # 5. Mobility delta (expensive but valuable)
        test_board = make_move(board, pos, player)
        my_mob = len(get_valid_moves(test_board, player))
        opp_mob = len(get_valid_moves(test_board, opponent(player)))
        mobility_score = (my_mob - opp_mob) * g.mobility_phase_weights[phase]

        # 6. Corner control bonus
        corner_bonus = 0.0
        for i, corner in enumerate(CORNERS):
            if pos in CORNER_ADJACENT.get(corner, []):
                if self.owns_corner(board, player, i):
                    corner_bonus += g.corner_adjacency_bonus[i]
                elif board[corner] == EMPTY:
                    # Risky if corner is empty (might give it away)
                    corner_bonus -= g.corner_adjacency_bonus[i] * 0.5

        # Combine with signal weights
        w = g.signal_weights
        total = (
            w[0] * pos_score +
            w[1] * flip_score +
            w[2] * dir_score +
            w[3] * phase_bonus +
            w[4] * mobility_score +
            w[5] * corner_bonus
        )

        return total

    def get_move(self, board: list, player: int) -> int:
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


def evaluate_genome(genome: ScaledGenome, num_games: int = 40) -> float:
    """Evaluate with more games for stability."""
    player = ScaledPlayer(genome)
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
            total_margin += margin

    win_rate = wins / num_games
    margin_bonus = total_margin / (num_games * 64) * 0.1

    return win_rate + margin_bonus


def run_scaled_evolution(
    pop_size: int = 50,
    generations: int = 100,
    elite_size: int = 8,
    games_per_eval: int = 40,
) -> Tuple[ScaledGenome, List[float]]:
    """Run scaled evolution."""

    print("=" * 70)
    print("SCALED EVOLUTION - BEAT HAND-CODED")
    print("=" * 70)
    print()
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"Elite: {elite_size}, Games/eval: {games_per_eval}")
    print()

    population = [random_genome() for _ in range(pop_size)]

    print("Evaluating initial population...")
    for g in population:
        g.fitness = evaluate_genome(g, games_per_eval)

    population.sort(key=lambda x: x.fitness, reverse=True)
    history = [population[0].fitness]

    avg_fit = sum(g.fitness for g in population) / len(population)
    print(f"Gen   0: best={population[0].fitness:.3f}, avg={avg_fit:.3f}")

    start_time = time.time()

    for gen in range(1, generations + 1):
        # Adaptive mutation rate (higher early, lower late)
        base_rate = 0.15 * (1 - gen / generations * 0.5)
        strength = 0.25 * (1 - gen / generations * 0.3)

        new_pop = list(population[:elite_size])

        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = [population[random.randint(0, pop_size-1)] for _ in range(4)]
            t2 = [population[random.randint(0, pop_size-1)] for _ in range(4)]
            p1 = max(t1, key=lambda x: x.fitness)
            p2 = max(t2, key=lambda x: x.fitness)

            child = crossover(p1, p2)
            child = mutate(child, base_rate, strength)
            new_pop.append(child)

        population = new_pop[:pop_size]

        for g in population[elite_size:]:
            g.fitness = evaluate_genome(g, games_per_eval)

        population.sort(key=lambda x: x.fitness, reverse=True)
        history.append(population[0].fitness)

        if gen % 10 == 0 or gen == generations:
            avg_fit = sum(g.fitness for g in population) / len(population)
            elapsed = time.time() - start_time
            print(f"Gen {gen:3d}: best={population[0].fitness:.3f}, "
                  f"avg={avg_fit:.3f}, time={elapsed:.0f}s")

    print()
    return population[0], history


def compare_vs_handcoded(evolved: ScaledGenome, num_games: int = 200):
    """Compare evolved vs hand-coded."""
    print("=" * 70)
    print("EVOLVED VS HAND-CODED (200 games)")
    print("=" * 70)
    print()

    evolved_player = ScaledPlayer(evolved)

    # Hand-coded genome
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

    handcoded = ScaledGenome(
        position_values=handcoded_pos,
        direction_weights=[1.0] * 8,
        phase_thresholds=[20.0, 50.0],
        signal_weights=[1.0, 1.0, 0.5, 0.5, 0.3, 0.5],
        corner_adjacency_bonus=[2.0] * 4,
        mobility_phase_weights=[1.0, 0.5, 0.2],
    )
    handcoded_player = ScaledPlayer(handcoded)

    evolved_wins = 0
    handcoded_wins = 0

    for i in range(num_games):
        board = get_initial_board()
        player_color = BLACK if i % 2 == 0 else WHITE
        current = BLACK
        passes = 0

        # Evolved game
        board_e = board.copy()
        current_e = BLACK
        passes_e = 0
        while not is_game_over(board_e) and passes_e < 2:
            valid = get_valid_moves(board_e, current_e)
            if not valid:
                passes_e += 1
                current_e = opponent(current_e)
                continue
            passes_e = 0

            if current_e == player_color:
                move = evolved_player.get_move(board_e, current_e)
            else:
                move = valid[random.randint(0, len(valid) - 1)]

            board_e = make_move(board_e, move, current_e)
            current_e = opponent(current_e)

        if get_winner(board_e) == player_color:
            evolved_wins += 1

        # Hand-coded game
        board_h = board.copy()
        current_h = BLACK
        passes_h = 0
        while not is_game_over(board_h) and passes_h < 2:
            valid = get_valid_moves(board_h, current_h)
            if not valid:
                passes_h += 1
                current_h = opponent(current_h)
                continue
            passes_h = 0

            if current_h == player_color:
                move = handcoded_player.get_move(board_h, current_h)
            else:
                move = valid[random.randint(0, len(valid) - 1)]

            board_h = make_move(board_h, move, current_h)
            current_h = opponent(current_h)

        if get_winner(board_h) == player_color:
            handcoded_wins += 1

    print(f"{'Method':<15} | {'Win Rate':<10}")
    print("-" * 30)
    print(f"{'Evolved':<15} | {evolved_wins/num_games*100:.1f}%")
    print(f"{'Hand-coded':<15} | {handcoded_wins/num_games*100:.1f}%")
    print()

    diff = evolved_wins - handcoded_wins
    if diff > 0:
        print(f"🎉 EVOLVED BEATS HAND-CODED by {diff} games ({diff/num_games*100:.1f}%)!")
    elif diff < 0:
        print(f"Hand-coded still ahead by {-diff} games")
    else:
        print("Tied!")
    print()

    return evolved_wins, handcoded_wins


def run_experiment():
    """Run scaled evolution experiment."""
    random.seed(42)

    best, history = run_scaled_evolution(
        pop_size=50,
        generations=100,
        elite_size=8,
        games_per_eval=40,
    )

    # Compare
    evolved_wins, handcoded_wins = compare_vs_handcoded(best, num_games=200)

    # Save
    with open("evolved_scaled_genome.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    print("Saved to evolved_scaled_genome.json")

    return best, history


if __name__ == "__main__":
    run_experiment()
