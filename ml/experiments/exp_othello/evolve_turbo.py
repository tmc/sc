"""
Turbo Evolution - Fast version to beat hand-coded.

Key insight: Use smarter initialization and focused search.
- Start with priors (corners good, X-squares bad)
- Smaller population but more targeted
- Faster games with early termination
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


def randn():
    u1 = random.random() + 1e-10
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


class TurboGenome:
    """Focused genome with strong priors."""

    def __init__(self):
        # Start with KNOWN good structure, evolve refinements
        self.corner_value = 25.0 + randn() * 5
        self.x_square_penalty = -12.0 + randn() * 3
        self.c_square_penalty = -4.0 + randn() * 2
        self.edge_value = 3.0 + randn() * 1

        # Center value by distance
        self.center_values = [1.0 + randn() * 0.3 for _ in range(4)]  # dist 0-3

        # Flip weights
        self.flip_weight = 1.5 + randn() * 0.3
        self.flip_phase_mod = [0.5, 1.0, 2.0]  # opening/mid/end

        # Mobility weight
        self.mobility_weight = 0.8 + randn() * 0.2
        self.mobility_phase = [1.5, 0.8, 0.2]  # matters less in endgame

        # Phase thresholds
        self.phase1 = 20 + randn() * 3
        self.phase2 = 48 + randn() * 3

        self.fitness = 0.0

    def copy(self):
        g = TurboGenome.__new__(TurboGenome)
        g.corner_value = self.corner_value
        g.x_square_penalty = self.x_square_penalty
        g.c_square_penalty = self.c_square_penalty
        g.edge_value = self.edge_value
        g.center_values = self.center_values.copy()
        g.flip_weight = self.flip_weight
        g.flip_phase_mod = self.flip_phase_mod.copy()
        g.mobility_weight = self.mobility_weight
        g.mobility_phase = self.mobility_phase.copy()
        g.phase1 = self.phase1
        g.phase2 = self.phase2
        g.fitness = 0.0
        return g

    def mutate(self, rate=0.15, strength=0.3):
        g = self.copy()
        if random.random() < rate:
            g.corner_value += randn() * strength * 3
        if random.random() < rate:
            g.x_square_penalty += randn() * strength * 2
        if random.random() < rate:
            g.c_square_penalty += randn() * strength
        if random.random() < rate:
            g.edge_value += randn() * strength
        for i in range(4):
            if random.random() < rate:
                g.center_values[i] += randn() * strength * 0.5
        if random.random() < rate:
            g.flip_weight = max(0.1, g.flip_weight + randn() * strength)
        for i in range(3):
            if random.random() < rate:
                g.flip_phase_mod[i] = max(0.1, g.flip_phase_mod[i] + randn() * strength)
        if random.random() < rate:
            g.mobility_weight = max(0, g.mobility_weight + randn() * strength)
        for i in range(3):
            if random.random() < rate:
                g.mobility_phase[i] = max(0, g.mobility_phase[i] + randn() * strength)
        if random.random() < rate:
            g.phase1 = max(10, min(35, g.phase1 + randn() * 2))
        if random.random() < rate:
            g.phase2 = max(35, min(58, g.phase2 + randn() * 2))
        return g

    def crossover(self, other):
        g = TurboGenome.__new__(TurboGenome)
        g.corner_value = self.corner_value if random.random() > 0.5 else other.corner_value
        g.x_square_penalty = self.x_square_penalty if random.random() > 0.5 else other.x_square_penalty
        g.c_square_penalty = self.c_square_penalty if random.random() > 0.5 else other.c_square_penalty
        g.edge_value = self.edge_value if random.random() > 0.5 else other.edge_value
        g.center_values = [a if random.random() > 0.5 else b for a, b in zip(self.center_values, other.center_values)]
        g.flip_weight = self.flip_weight if random.random() > 0.5 else other.flip_weight
        g.flip_phase_mod = [a if random.random() > 0.5 else b for a, b in zip(self.flip_phase_mod, other.flip_phase_mod)]
        g.mobility_weight = self.mobility_weight if random.random() > 0.5 else other.mobility_weight
        g.mobility_phase = [a if random.random() > 0.5 else b for a, b in zip(self.mobility_phase, other.mobility_phase)]
        alpha = random.random()
        g.phase1 = alpha * self.phase1 + (1-alpha) * other.phase1
        g.phase2 = alpha * self.phase2 + (1-alpha) * other.phase2
        g.fitness = 0.0
        return g


class TurboPlayer:
    def __init__(self, genome: TurboGenome):
        self.g = genome

    def get_phase(self, board):
        black, white, _ = count_pieces(board)
        total = black + white
        if total < self.g.phase1:
            return 0
        elif total < self.g.phase2:
            return 1
        return 2

    def center_dist(self, pos):
        row, col = pos // 8, pos % 8
        return int(abs(row - 3.5) + abs(col - 3.5))

    def score_move(self, board, pos, player):
        flips = get_all_flips(board, pos, player)
        if not flips:
            return -1e9

        phase = self.get_phase(board)
        score = 0.0

        # Position value
        if pos in CORNERS:
            score += self.g.corner_value
        elif pos in X_SQUARES:
            score += self.g.x_square_penalty
        elif pos in C_SQUARES:
            score += self.g.c_square_penalty
        elif pos in EDGES:
            score += self.g.edge_value
        else:
            dist = min(3, self.center_dist(pos))
            score += self.g.center_values[dist]

        # Flip value (phase-dependent)
        score += len(flips) * self.g.flip_weight * self.g.flip_phase_mod[phase]

        # Mobility (expensive, only compute in opening/mid)
        if phase < 2 and self.g.mobility_weight > 0.1:
            test_board = make_move(board, pos, player)
            my_mob = len(get_valid_moves(test_board, player))
            opp_mob = len(get_valid_moves(test_board, opponent(player)))
            score += (my_mob - opp_mob) * self.g.mobility_weight * self.g.mobility_phase[phase]

        return score

    def get_move(self, board, player):
        valid = get_valid_moves(board, player)
        if not valid:
            return -1
        return max(valid, key=lambda m: self.score_move(board, m, player))


def evaluate(genome, num_games=30):
    """Fast evaluation."""
    player = TurboPlayer(genome)
    wins = 0
    margin = 0

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
        m = black - white if model_player == BLACK else white - black

        if winner == model_player:
            wins += 1
            margin += m
        elif winner != 0:
            margin += m

    return wins / num_games + margin / (num_games * 64) * 0.1


def run_turbo_evolution():
    """Fast evolution with strong priors."""
    print("=" * 70)
    print("TURBO EVOLUTION - BEAT HAND-CODED")
    print("=" * 70)

    pop_size = 40
    generations = 60
    elite = 6

    print(f"Pop: {pop_size}, Gens: {generations}, Elite: {elite}")
    print()

    pop = [TurboGenome() for _ in range(pop_size)]

    print("Evaluating initial...")
    for g in pop:
        g.fitness = evaluate(g, 30)

    pop.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Gen  0: best={pop[0].fitness:.3f}")

    start = time.time()

    for gen in range(1, generations + 1):
        # Adaptive mutation
        rate = 0.15 * (1 - gen/generations * 0.4)
        strength = 0.3 * (1 - gen/generations * 0.3)

        new_pop = [g.copy() for g in pop[:elite]]

        while len(new_pop) < pop_size:
            t1 = [pop[random.randint(0, pop_size-1)] for _ in range(3)]
            t2 = [pop[random.randint(0, pop_size-1)] for _ in range(3)]
            p1 = max(t1, key=lambda x: x.fitness)
            p2 = max(t2, key=lambda x: x.fitness)

            child = p1.crossover(p2)
            child = child.mutate(rate, strength)
            new_pop.append(child)

        pop = new_pop[:pop_size]

        for g in pop[elite:]:
            g.fitness = evaluate(g, 30)

        pop.sort(key=lambda x: x.fitness, reverse=True)

        if gen % 1 == 0 or gen == generations:
            elapsed = time.time() - start
            avg = sum(g.fitness for g in pop) / len(pop)
            print(f"Gen {gen:2d}: best={pop[0].fitness:.3f}, avg={avg:.3f}, time={elapsed:.0f}s")

    print()
    return pop[0]


def compare_final(evolved):
    """Final comparison with more games."""
    print("=" * 70)
    print("FINAL COMPARISON (300 games each)")
    print("=" * 70)

    evolved_player = TurboPlayer(evolved)

    # Hand-coded baseline
    handcoded = TurboGenome.__new__(TurboGenome)
    handcoded.corner_value = 25.0
    handcoded.x_square_penalty = -12.0
    handcoded.c_square_penalty = -4.0
    handcoded.edge_value = 3.0
    handcoded.center_values = [1.0, 0.8, 0.5, 0.2]
    handcoded.flip_weight = 1.0
    handcoded.flip_phase_mod = [0.5, 1.0, 2.0]
    handcoded.mobility_weight = 0.5
    handcoded.mobility_phase = [1.0, 0.5, 0.1]
    handcoded.phase1 = 20
    handcoded.phase2 = 50
    handcoded_player = TurboPlayer(handcoded)

    num_games = 300
    evolved_wins = 0
    handcoded_wins = 0

    for i in range(num_games):
        board = get_initial_board()
        player_color = BLACK if i % 2 == 0 else WHITE

        # Evolved
        board_e = board.copy()
        current = BLACK
        passes = 0
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
    print(f"{'Method':<15} | {'Wins':<6} | {'Win Rate':<10}")
    print("-" * 40)
    print(f"{'Evolved':<15} | {evolved_wins:<6} | {evolved_wins/num_games*100:.1f}%")
    print(f"{'Hand-coded':<15} | {handcoded_wins:<6} | {handcoded_wins/num_games*100:.1f}%")
    print()

    diff = evolved_wins - handcoded_wins
    if diff > 0:
        print(f"🎉 EVOLVED BEATS HAND-CODED by {diff} games!")
    elif diff < 0:
        print(f"Hand-coded ahead by {-diff} games")
    else:
        print("Tied!")
    print()

    # Show what was evolved
    print("=" * 70)
    print("EVOLVED PARAMETERS")
    print("=" * 70)
    print(f"Corner value:     {evolved.corner_value:.2f} (hand-coded: 25.0)")
    print(f"X-square penalty: {evolved.x_square_penalty:.2f} (hand-coded: -12.0)")
    print(f"C-square penalty: {evolved.c_square_penalty:.2f} (hand-coded: -4.0)")
    print(f"Edge value:       {evolved.edge_value:.2f} (hand-coded: 3.0)")
    print(f"Flip weight:      {evolved.flip_weight:.2f}")
    print(f"Mobility weight:  {evolved.mobility_weight:.2f}")
    print(f"Phase 1 threshold: {evolved.phase1:.1f} pieces")
    print(f"Phase 2 threshold: {evolved.phase2:.1f} pieces")
    print()

    return evolved_wins, handcoded_wins


if __name__ == "__main__":
    random.seed(42)
    best = run_turbo_evolution()
    compare_final(best)
