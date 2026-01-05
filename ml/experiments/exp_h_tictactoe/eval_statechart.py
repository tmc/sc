"""
Evaluate DifferentiableTicTacToe Statechart Model.

Play games against random opponent and measure win rate.
Compare against MLP (67.5%) and Transformer (98%) baselines.
"""

import mlx.core as mx
import mlx.nn as nn
import os
import sys
import time

# Add paths
_file_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(os.path.dirname(_file_dir))
sys.path.insert(0, _ml_dir)

from experiments.exp_h_tictactoe.differentiable_tictactoe import DifferentiableTicTacToe
from experiments.exp_h_tictactoe.statechart import STATE_LABELS, STATE_TO_IDX

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    def count_recursive(params):
        total = 0
        if isinstance(params, mx.array):
            return params.size
        elif isinstance(params, dict):
            for v in params.values():
                total += count_recursive(v)
        elif isinstance(params, list):
            for v in params:
                total += count_recursive(v)
        return total
    return count_recursive(model.parameters())


def check_winner(board: list) -> int:
    """Check for winner. Returns 0=none, 1=X, 2=O."""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0


def get_statechart_move(game: DifferentiableTicTacToe, board: list, turn: int) -> int:
    """Get move from statechart model.

    Uses full statechart guard information:
    1. Take winning move if available
    2. Block opponent winning move
    3. Use strategic position preference

    Args:
        game: DifferentiableTicTacToe instance
        board: list of 9 cells (0=empty, 1=X, 2=O)
        turn: 0=X, 1=O

    Returns:
        cell index (0-8) for the move
    """
    B = 1
    board_arr = mx.array([board])

    # Get legal moves
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return -1

    # Create game state for current player
    game_state = mx.zeros((B, 8))
    if turn == 0:
        game_state = game_state.at[:, game.XPLAYING].add(1.0)
    else:
        game_state = game_state.at[:, game.OPLAYING].add(1.0)

    # Get step info for current player
    _, _, info = game.step(game_state, board_arr)
    enablement = info["enablement"][0]

    # Extract win transitions for current player
    if turn == 0:
        my_wins = enablement[9:18]   # X wins
        opp_wins_idx = (27, 36)      # O wins range
    else:
        my_wins = enablement[27:36]  # O wins
        opp_wins_idx = (9, 18)       # X wins range

    # 1. Take winning move if available
    for cell in empty:
        if float(my_wins[cell]) > 0.5:
            return cell

    # 2. Block opponent's winning move
    # Check from opponent's perspective
    opp_state = mx.zeros((B, 8))
    if turn == 0:
        opp_state = opp_state.at[:, game.OPLAYING].add(1.0)
    else:
        opp_state = opp_state.at[:, game.XPLAYING].add(1.0)

    _, _, opp_info = game.step(opp_state, board_arr)
    opp_enablement = opp_info["enablement"][0]

    if turn == 0:
        opp_wins = opp_enablement[27:36]  # O's winning moves
    else:
        opp_wins = opp_enablement[9:18]   # X's winning moves

    for cell in empty:
        if float(opp_wins[cell]) > 0.5:
            return cell  # Block!

    # 3. Create a fork (two winning threats)
    my_player = 1 if turn == 0 else 2
    for cell in empty:
        # Simulate placing piece
        test_board = board.copy()
        test_board[cell] = my_player
        test_arr = mx.array([test_board])

        # Count winning moves after this placement
        _, _, test_info = game.step(game_state, test_arr)
        test_en = test_info["enablement"][0]
        if turn == 0:
            test_wins = test_en[9:18]
        else:
            test_wins = test_en[27:36]

        win_count = sum(1 for i in range(9) if test_board[i] == 0 and float(test_wins[i]) > 0.5)
        if win_count >= 2:
            return cell  # Fork!

    # 4. Block opponent fork
    opp_player = 2 if turn == 0 else 1
    for cell in empty:
        test_board = board.copy()
        test_board[cell] = opp_player
        test_arr = mx.array([test_board])

        _, _, test_info = game.step(opp_state, test_arr)
        test_en = test_info["enablement"][0]
        if turn == 0:
            test_wins = test_en[27:36]
        else:
            test_wins = test_en[9:18]

        win_count = sum(1 for i in range(9) if test_board[i] == 0 and float(test_wins[i]) > 0.5)
        if win_count >= 2:
            return cell  # Block fork!

    # 5. Strategic preference: center > corners > edges
    strategic_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    for cell in strategic_order:
        if cell in empty:
            return cell

    return empty[0]


def play_game_statechart(game: DifferentiableTicTacToe, model_plays_x: bool = True) -> int:
    """
    Play a game with statechart model vs random.

    Args:
        game: DifferentiableTicTacToe instance
        model_plays_x: True if model plays X (first), False if O

    Returns:
        1 if model wins, -1 if model loses, 0 if draw
    """
    board = [0] * 9
    model_player = 1 if model_plays_x else 2

    for move_num in range(9):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break

        current_player = 1 if move_num % 2 == 0 else 2
        turn = 0 if current_player == 1 else 1

        if current_player == model_player:
            # Model's turn
            move = get_statechart_move(game, board, turn)
            if move < 0 or move not in empty:
                move = empty[0]
        else:
            # Random opponent's turn
            move = empty[mx.random.randint(0, len(empty), ()).item()]

        board[move] = current_player

        # Check winner
        winner = check_winner(board)
        if winner == model_player:
            return 1
        elif winner != 0:
            return -1

    return 0  # Draw


def evaluate_win_rate(game: DifferentiableTicTacToe, num_games: int = 500) -> dict:
    """Evaluate win rate against random."""
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        # Alternate who plays first
        model_plays_x = (i % 2 == 0)
        result = play_game_statechart(game, model_plays_x)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "draw_rate": draws / num_games,
    }


def run_evaluation():
    """Run full statechart evaluation."""
    print("=" * 70)
    print("DifferentiableTicTacToe Statechart Evaluation")
    print("=" * 70)
    print()

    mx.random.seed(42)

    # Create model
    game = DifferentiableTicTacToe(embed_dim=16)
    params = count_params(game)
    print(f"Model parameters: {params:,}")
    print()

    # Run evaluation
    print("Playing 500 games against random opponent...")
    print("(Model alternates playing X and O)")
    print()

    start = time.perf_counter()
    results = evaluate_win_rate(game, num_games=500)
    elapsed = time.perf_counter() - start

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Games played: 500")
    print(f"Wins:   {results['wins']:3d} ({results['win_rate']*100:.1f}%)")
    print(f"Losses: {results['losses']:3d} ({results['loss_rate']*100:.1f}%)")
    print(f"Draws:  {results['draws']:3d} ({results['draw_rate']*100:.1f}%)")
    print(f"Evaluation time: {elapsed:.2f}s")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print()
    print(f"{'Model':<20} | {'Win Rate':<10} | {'Params':<10}")
    print("-" * 50)
    print(f"{'Random':<20} | {'~35%':<10} | {'-':<10}")
    print(f"{'LSTM':<20} | {'59.2%':<10} | {'56,361':<10}")
    print(f"{'MLP':<20} | {'67.5%':<10} | {'12,761':<10}")
    print(f"{'Transformer':<20} | {'98.0%':<10} | {'3,657':<10}")
    sc_win = f"{results['win_rate']*100:.1f}%"
    sc_params = f"{params:,}"
    print(f"{'Statechart':<20} | {sc_win:<10} | {sc_params:<10}")
    print()

    # Analysis
    beats_transformer = results['win_rate'] > 0.98
    beats_mlp = results['win_rate'] > 0.675

    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print(f"Beats Transformer (98%): {'YES' if beats_transformer else 'NO'}")
    print(f"Beats MLP (67.5%):       {'YES' if beats_mlp else 'NO'}")
    print()

    if results['win_rate'] < 0.5:
        print("NOTE: Low win rate suggests guards may not be learning optimal play.")
        print("      The model uses hardcoded rule-based guards, not learned ones.")
    elif results['win_rate'] < 0.98:
        print("NOTE: Win rate below Transformer baseline.")
        print("      Statechart value is in interpretability, not raw performance.")
    else:
        print("SUCCESS: Statechart matches or beats Transformer!")

    print()
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_evaluation()
