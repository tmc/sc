"""
Train DifferentiableTicTacToe Policy via Self-Play.

The statechart provides structure (states, transitions, guards).
We LEARN the policy (which transition to take) via gradient descent.

Training approach: REINFORCE with self-play
- Play games using current policy
- Reward: +1 win, -1 loss, 0 draw
- Update policy to increase winning actions
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import os
import sys

# Add paths
_file_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(os.path.dirname(_file_dir))
sys.path.insert(0, _ml_dir)

from experiments.exp_h_tictactoe.differentiable_tictactoe import DifferentiableTicTacToe

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def count_params(model: nn.Module) -> int:
    """Count parameters."""
    def count_recursive(p):
        if isinstance(p, mx.array):
            return p.size
        elif isinstance(p, dict):
            return sum(count_recursive(v) for v in p.values())
        elif isinstance(p, list):
            return sum(count_recursive(v) for v in p)
        return 0
    return count_recursive(model.parameters())


def check_winner(board: list) -> int:
    """Check winner. 0=none, 1=X, 2=O."""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0


class TrainablePolicy(nn.Module):
    """
    Trainable policy USING statechart structure.

    The statechart computes transition enablement (move validity + win detection).
    We learn weights to combine these signals into move selection.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.game = DifferentiableTicTacToe(embed_dim=embed_dim)

        # Learned position values: [center, corners, edges]
        # Initialize with strategic priors
        self.pos_proj = nn.Linear(9, 9, bias=False)
        # Initialize: center=1.0, corners=0.5, edges=0.25
        init_pos = mx.array([0.5, 0.25, 0.5, 0.25, 1.0, 0.25, 0.5, 0.25, 0.5])
        self.pos_proj.weight = mx.eye(9) * init_pos[:, None]

        # Signal combination weights (learnable)
        self.signal_weights = nn.Linear(4, 1, bias=False)
        # Initialize: [win, block, move, position] = [5, 4, 1, 0.5]
        self.signal_weights.weight = mx.array([[5.0, 4.0, 1.0, 0.5]])

    def get_move_logits(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Get move logits using statechart enablement.

        Uses the statechart's guard computation to identify:
        1. Winning moves (highest priority)
        2. Blocking moves (block opponent wins)
        3. Regular moves (weighted by position value)
        """
        B = board.shape[0]

        # Get game state
        game_state = mx.zeros((B, 8))
        is_x = (turn == 0).astype(mx.float32)[:, None]
        is_o = (turn == 1).astype(mx.float32)[:, None]

        # Set active state
        for b in range(B):
            if int(turn[b].item()) == 0:
                game_state = game_state.at[b, self.game.XPLAYING].add(1.0)
            else:
                game_state = game_state.at[b, self.game.OPLAYING].add(1.0)

        # Get enablement from statechart
        _, _, info = self.game.step(game_state, board)
        enablement = info["enablement"]  # [B, 37]

        # Extract signals for current player
        # X: moves=0-8, wins=9-17
        # O: moves=18-26, wins=27-35
        x_moves = enablement[:, :9]
        x_wins = enablement[:, 9:18]
        o_moves = enablement[:, 18:27]
        o_wins = enablement[:, 27:36]

        # My moves and wins
        my_moves = is_x * x_moves + is_o * o_moves
        my_wins = is_x * x_wins + is_o * o_wins

        # Opponent wins (for blocking)
        opp_wins = is_x * o_wins + is_o * x_wins

        # Position values
        pos_vals = self.pos_proj(mx.ones((B, 9)))  # [B, 9]

        # Stack signals: [B, 9, 4]
        signals = mx.stack([my_wins, opp_wins, my_moves, pos_vals], axis=-1)

        # Combine with learned weights
        weights = self.signal_weights.weight[0]  # [4]
        logits = mx.sum(signals * weights[None, None, :], axis=-1)  # [B, 9]

        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        """Get move probabilities (masked to legal moves)."""
        logits = self.get_move_logits(board, turn)

        # Mask illegal moves
        legal_mask = (board == 0).astype(mx.float32)
        masked_logits = logits + (1 - legal_mask) * (-1e9)

        return mx.softmax(masked_logits, axis=-1)

    def sample_move(self, board: mx.array, turn: mx.array) -> tuple:
        """Sample a move from the policy."""
        probs = self.get_move_probs(board, turn)[0]

        # Greedy for evaluation, sample for training
        move = int(mx.argmax(probs).item())

        # Fallback if illegal
        empty = [i for i in range(9) if int(board[0, i].item()) == 0]
        if move not in empty and empty:
            move = empty[0]

        log_prob = mx.log(probs[move] + 1e-10)
        return move, log_prob


def play_game_for_training(policy: TrainablePolicy) -> tuple:
    """
    Play a game and collect trajectory for training.

    Returns: (reward, boards, moves)
        - reward: +1 if X wins, -1 if O wins, 0 draw
        - boards: list of board states before each move
        - moves: list of (turn, move_idx) for each move
    """
    board = [0] * 9
    boards = []
    moves = []

    for move_num in range(9):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break

        turn = move_num % 2  # 0=X, 1=O
        boards.append(board.copy())

        board_arr = mx.array([board])
        turn_arr = mx.array([turn])

        # Get move
        move, _ = policy.sample_move(board_arr, turn_arr)
        moves.append((turn, move))

        # Make move
        player = 1 if turn == 0 else 2
        board[move] = player

        # Check winner
        winner = check_winner(board)
        if winner == 1:
            return 1.0, boards, moves  # X wins
        elif winner == 2:
            return -1.0, boards, moves  # O wins

    return 0.0, boards, moves  # Draw


def play_vs_random(policy: TrainablePolicy, policy_is_x: bool = True) -> int:
    """Play policy vs random. Returns +1 win, -1 loss, 0 draw."""
    board = [0] * 9
    policy_player = 1 if policy_is_x else 2

    for move_num in range(9):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break

        current_player = 1 if move_num % 2 == 0 else 2
        turn = move_num % 2

        if current_player == policy_player:
            # Policy's turn
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])
            probs = policy.get_move_probs(board_arr, turn_arr)[0]
            move = int(mx.argmax(probs).item())
            if move not in empty:
                move = empty[0]
        else:
            # Random's turn
            move = empty[mx.random.randint(0, len(empty), ()).item()]

        board[move] = current_player

        winner = check_winner(board)
        if winner == policy_player:
            return 1
        elif winner != 0:
            return -1

    return 0


def evaluate_vs_random(policy: TrainablePolicy, num_games: int = 200) -> dict:
    """Evaluate win rate vs random (alternating who goes first)."""
    wins = losses = draws = 0

    for i in range(num_games):
        result = play_vs_random(policy, policy_is_x=(i % 2 == 0))
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
    }


def train_reinforce(
    policy: TrainablePolicy,
    num_episodes: int = 5000,
    lr: float = 0.001,
    eval_every: int = 500,
) -> dict:
    """Train policy using REINFORCE."""
    optimizer = optim.Adam(learning_rate=lr)

    history = {"episode": [], "loss": [], "win_rate": []}
    start_time = time.perf_counter()

    for ep in range(num_episodes):
        # Play a game
        reward, boards, moves = play_game_for_training(policy)

        if not moves:
            continue

        # Compute loss: -reward * sum(log_probs)
        # X wants positive reward, O wants negative
        def loss_fn(p):
            total_log_prob = mx.array(0.0)
            for i, (turn, move_idx) in enumerate(moves):
                board_arr = mx.array([boards[i]])
                turn_arr = mx.array([turn])
                probs = p.get_move_probs(board_arr, turn_arr)[0]
                log_p = mx.log(probs[move_idx] + 1e-10)
                # X (turn=0) wants +reward, O (turn=1) wants -reward
                sign = 1.0 if turn == 0 else -1.0
                total_log_prob = total_log_prob + sign * log_p

            return -reward * total_log_prob

        loss, grads = nn.value_and_grad(policy, loss_fn)(policy)
        optimizer.update(policy, grads)
        mx.eval(policy.parameters())

        # Evaluate periodically
        if (ep + 1) % eval_every == 0:
            eval_result = evaluate_vs_random(policy, num_games=100)
            elapsed = time.perf_counter() - start_time

            history["episode"].append(ep + 1)
            history["loss"].append(float(loss) if hasattr(loss, 'item') else float(loss))
            history["win_rate"].append(eval_result["win_rate"])

            print(f"Episode {ep+1:5d}: win_rate={eval_result['win_rate']:.3f}, "
                  f"W/L/D={eval_result['wins']}/{eval_result['losses']}/{eval_result['draws']}, "
                  f"time={elapsed:.1f}s")

    return history


def run_training():
    """Run full training experiment."""
    print("=" * 70)
    print("Train DifferentiableTicTacToe Policy")
    print("=" * 70)
    print()

    mx.random.seed(42)

    # Create policy
    policy = TrainablePolicy(embed_dim=16)
    params = count_params(policy)
    print(f"Policy parameters: {params:,}")
    print()

    # Evaluate untrained
    print("Evaluating UNTRAINED policy...")
    untrained_result = evaluate_vs_random(policy, num_games=200)
    print(f"  Win rate: {untrained_result['win_rate']:.3f}")
    print(f"  W/L/D: {untrained_result['wins']}/{untrained_result['losses']}/{untrained_result['draws']}")
    print()

    # Train
    print("Training with REINFORCE...")
    print("-" * 70)
    history = train_reinforce(
        policy,
        num_episodes=5000,
        lr=0.001,
        eval_every=500,
    )
    print("-" * 70)
    print()

    # Final evaluation
    print("Evaluating TRAINED policy...")
    trained_result = evaluate_vs_random(policy, num_games=500)
    print(f"  Win rate: {trained_result['win_rate']:.3f}")
    print(f"  W/L/D: {trained_result['wins']}/{trained_result['losses']}/{trained_result['draws']}")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Model':<25} | {'Win Rate':<10} | {'Params':<10}")
    print("-" * 55)
    print(f"{'Random baseline':<25} | {'~35%':<10} | {'-':<10}")
    print(f"{'Statechart (untrained)':<25} | {untrained_result['win_rate']*100:.1f}%{'':<5} | {params:,}{'':<1}")
    print(f"{'Statechart (trained)':<25} | {trained_result['win_rate']*100:.1f}%{'':<5} | {params:,}{'':<1}")
    print(f"{'MLP baseline':<25} | {'67.5%':<10} | {'12,761':<10}")
    print(f"{'Transformer baseline':<25} | {'98.0%':<10} | {'3,657':<10}")
    print()

    return {
        "untrained": untrained_result,
        "trained": trained_result,
        "history": history,
        "params": params,
    }


if __name__ == "__main__":
    run_training()
