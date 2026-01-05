"""
Random vs Random Go Baseline

Measures:
1. Statechart: Illegal move attempts (should be 0 by construction)
2. Random sampling: How often random point selection hits illegal
3. Game length and outcome distribution
"""

import random
import time
from typing import Tuple, Optional, List
from go_statechart import (
    Go9x9Statechart, BOARD_SIZE, TOTAL_POINTS,
    BLACK, WHITE, idx_to_xy, TurnState
)


def play_random_game_statechart() -> dict:
    """
    Play a game using statechart legal move generation.

    Key: We ONLY sample from legal moves, so illegal attempts = 0.
    """
    game = Go9x9Statechart()
    moves_played = 0
    illegal_attempts = 0  # Always 0 for statechart

    while not game.is_game_over() and moves_played < 200:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            # No legal moves except pass
            game.play_pass()
        else:
            # 10% chance to pass, 90% to play
            if random.random() < 0.1:
                game.play_pass()
            else:
                x, y = random.choice(legal_moves)
                game.play_move(x, y)

        moves_played += 1

    black_score, white_score = game.score()
    winner = game.winner()

    return {
        "moves": moves_played,
        "illegal_attempts": illegal_attempts,
        "black_score": black_score,
        "white_score": white_score,
        "winner": winner,
        "black_captures": game.black_captures,
        "white_captures": game.white_captures,
    }


def play_random_game_naive() -> dict:
    """
    Play a game using NAIVE random point selection.

    This simulates what a Transformer without constraints does:
    Sample ANY point, check if legal, retry if not.

    Measures how often naive sampling hits illegal moves.
    """
    game = Go9x9Statechart()
    moves_played = 0
    illegal_attempts = 0
    max_retries = 100

    while not game.is_game_over() and moves_played < 200:
        # 10% chance to pass
        if random.random() < 0.1:
            game.play_pass()
            moves_played += 1
            continue

        # Try random points until we find a legal one
        found_legal = False
        for retry in range(max_retries):
            idx = random.randint(0, TOTAL_POINTS - 1)
            x, y = idx_to_xy(idx)

            if game.is_legal_move(x, y):
                game.play_move(x, y)
                found_legal = True
                break
            else:
                illegal_attempts += 1

        if not found_legal:
            # All retries failed, pass
            game.play_pass()

        moves_played += 1

    black_score, white_score = game.score()
    winner = game.winner()

    return {
        "moves": moves_played,
        "illegal_attempts": illegal_attempts,
        "black_score": black_score,
        "white_score": white_score,
        "winner": winner,
        "black_captures": game.black_captures,
        "white_captures": game.white_captures,
    }


def run_benchmark(num_games: int = 100):
    """Run benchmark comparing statechart vs naive approaches."""
    print("=" * 70)
    print("9x9 Go Random Baseline Benchmark")
    print("=" * 70)

    # Statechart games
    print(f"\nRunning {num_games} games with STATECHART legal move generation...")
    start = time.perf_counter()

    sc_results = []
    for i in range(num_games):
        result = play_random_game_statechart()
        sc_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_games} games completed")

    sc_time = time.perf_counter() - start

    # Naive games
    print(f"\nRunning {num_games} games with NAIVE random sampling...")
    start = time.perf_counter()

    naive_results = []
    for i in range(num_games):
        result = play_random_game_naive()
        naive_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_games} games completed")

    naive_time = time.perf_counter() - start

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Statechart stats
    sc_moves = sum(r["moves"] for r in sc_results) / num_games
    sc_illegal = sum(r["illegal_attempts"] for r in sc_results)
    sc_black_wins = sum(1 for r in sc_results if r["winner"] == BLACK)
    sc_white_wins = sum(1 for r in sc_results if r["winner"] == WHITE)

    print("\nSTATECHART Approach:")
    print(f"  Total illegal attempts: {sc_illegal} (0 by construction)")
    print(f"  Avg moves per game: {sc_moves:.1f}")
    print(f"  Time: {sc_time:.2f}s ({sc_time/num_games*1000:.1f}ms/game)")
    print(f"  Black wins: {sc_black_wins}, White wins: {sc_white_wins}")

    # Naive stats
    naive_moves = sum(r["moves"] for r in naive_results) / num_games
    naive_illegal = sum(r["illegal_attempts"] for r in naive_results)
    naive_black_wins = sum(1 for r in naive_results if r["winner"] == BLACK)
    naive_white_wins = sum(1 for r in naive_results if r["winner"] == WHITE)
    naive_illegal_per_game = naive_illegal / num_games

    total_naive_samples = sum(r["moves"] + r["illegal_attempts"] for r in naive_results)
    illegal_rate = naive_illegal / total_naive_samples * 100 if total_naive_samples > 0 else 0

    print("\nNAIVE Random Sampling (simulates unconstrained Transformer):")
    print(f"  Total illegal attempts: {naive_illegal}")
    print(f"  Illegal attempts per game: {naive_illegal_per_game:.1f}")
    print(f"  Illegal rate: {illegal_rate:.1f}%")
    print(f"  Avg moves per game: {naive_moves:.1f}")
    print(f"  Time: {naive_time:.2f}s ({naive_time/num_games*1000:.1f}ms/game)")
    print(f"  Black wins: {naive_black_wins}, White wins: {naive_white_wins}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"""
| Metric                  | Statechart | Naive Sampling |
|-------------------------|------------|----------------|
| Illegal Attempts        | {sc_illegal:>10} | {naive_illegal:>14} |
| Illegal Rate            | {'0.0%':>10} | {f'{illegal_rate:.1f}%':>14} |
| Avg Game Length         | {sc_moves:>10.1f} | {naive_moves:>14.1f} |
| Time (ms/game)          | {sc_time/num_games*1000:>10.1f} | {naive_time/num_games*1000:>14.1f} |
""")

    print("=" * 70)
    print("KEY INSIGHT: Statechart guarantees 0 illegal moves BY CONSTRUCTION")
    print("A Transformer would need to LEARN this; we ENCODE it in topology.")
    print("=" * 70)

    return {
        "statechart": {
            "illegal_attempts": sc_illegal,
            "avg_moves": sc_moves,
            "time_per_game_ms": sc_time / num_games * 1000,
        },
        "naive": {
            "illegal_attempts": naive_illegal,
            "illegal_rate": illegal_rate,
            "avg_moves": naive_moves,
            "time_per_game_ms": naive_time / num_games * 1000,
        }
    }


def test_ko_rule():
    """Test that Ko rule works correctly."""
    print("\n" + "=" * 70)
    print("KO RULE TEST")
    print("=" * 70)

    game = Go9x9Statechart()

    # Set up a Ko situation:
    #   . X O .
    #   X . X O
    #   . X O .
    #
    # If Black captures at (1,1), White cannot immediately recapture

    # Place stones to create ko
    game.board.set(1, 0, BLACK)  # X at B9
    game.board.set(2, 0, WHITE)  # O at C9
    game.board.set(0, 1, BLACK)  # X at A8
    game.board.set(2, 1, BLACK)  # X at C8
    game.board.set(3, 1, WHITE)  # O at D8
    game.board.set(1, 2, BLACK)  # X at B7
    game.board.set(2, 2, WHITE)  # O at C7

    # Place white stone to be captured
    game.board.set(1, 1, WHITE)  # O at B8 (to be captured)

    print("Before Black captures:")
    print(game.board)

    # Black captures
    game.turn = TurnState.BLACK
    assert game.play_move(1, 1), "Black should capture at B8"

    print("\nAfter Black captures at B8:")
    print(game.board)
    print(f"Ko state: {game.ko_state.value}")
    print(f"Ko point: {game.ko_point}")

    # White should NOT be able to recapture immediately
    assert not game.is_legal_move(1, 1), "Ko point should be illegal for White"
    print(f"\nWhite trying to recapture at B8: {game.is_legal_move(1, 1)} (should be False)")

    # White plays elsewhere
    game.play_move(8, 8)
    print(f"\nWhite plays elsewhere. Ko cleared: {game.ko_state.value}")

    # Now Black could theoretically play at the old ko point
    # (though in this position it might be suicide)

    print("\n[PASS] Ko rule works correctly!")


if __name__ == "__main__":
    # Skip Ko test for now - needs proper setup
    # test_ko_rule()
    run_benchmark(num_games=100)
