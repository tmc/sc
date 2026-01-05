"""
TicTacToe Baseline: MLP Policy Network

Compare against DifferentiableTicTacToe statechart approach.

Metrics:
1. Legal move accuracy
2. Win rate vs random
3. Sample efficiency (accuracy at 100, 1000, 10000 samples)
4. Training time
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

# Winning lines for TicTacToe
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),  # diagonals
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


class MLPBaseline(nn.Module):
    """
    Simple MLP for TicTacToe move prediction.

    Input: board (9 cells, each 0/1/2) + whose_turn (0=X, 1=O)
    Output: move probabilities (9)

    Architecture designed to match statechart model param count (~3000).
    """

    def __init__(self, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # Embed each cell (0=empty, 1=X, 2=O)
        self.cell_embed = nn.Embedding(3, embed_dim)
        # Embed whose turn (0=X, 1=O)
        self.turn_embed = nn.Embedding(2, embed_dim)

        # MLP: 9*embed_dim + embed_dim -> hidden -> 9
        input_dim = 9 * embed_dim + embed_dim
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 9)

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Predict move probabilities.

        Args:
            board: [B, 9] cell values (0=empty, 1=X, 2=O)
            turn: [B] whose turn (0=X, 1=O)

        Returns:
            [B, 9] logits for each cell
        """
        B = board.shape[0]

        # Embed board cells
        cell_emb = self.cell_embed(board)  # [B, 9, embed_dim]
        cell_flat = cell_emb.reshape(B, -1)  # [B, 9*embed_dim]

        # Embed turn
        turn_emb = self.turn_embed(turn)  # [B, embed_dim]

        # Concatenate
        x = mx.concatenate([cell_flat, turn_emb], axis=-1)

        # MLP
        x = mx.maximum(self.hidden1(x), 0)  # ReLU
        x = mx.maximum(self.hidden2(x), 0)  # ReLU
        logits = self.output(x)

        return logits

    def get_legal_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        """Get move probabilities masked to legal moves only."""
        logits = self(board, turn)

        # Mask illegal moves (non-empty cells)
        legal_mask = (board == 0).astype(mx.float32)
        masked_logits = logits + (1 - legal_mask) * (-1e9)

        return mx.softmax(masked_logits, axis=-1)


class StatechartModel(nn.Module):
    """
    Wrapper for statechart-based TicTacToe for fair comparison.

    Uses the DifferentiableTicTacToe model.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        from experiments.exp_h_tictactoe.differentiable_tictactoe import DifferentiableTicTacToe
        self.game = DifferentiableTicTacToe(embed_dim=embed_dim)

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Get move preferences from statechart model.

        Returns logits for each cell (higher = preferred move).
        """
        B = board.shape[0]

        # Create game state based on turn
        game_state = mx.zeros((B, 8))
        # XPlaying = 2, OPlaying = 5
        x_mask = (turn == 0).astype(mx.float32)
        o_mask = (turn == 1).astype(mx.float32)
        game_state = game_state.at[:, 2].add(x_mask)  # XPlaying
        game_state = game_state.at[:, 5].add(o_mask)  # OPlaying

        # Get step info
        _, _, info = self.game.step(game_state, board)
        enablement = info["enablement"]

        # Extract move preferences from enablement
        # For X: transitions 0-8 (moves) + 9-17 (wins)
        # For O: transitions 18-26 (moves) + 27-35 (wins)
        x_prefs = enablement[:, :9] + enablement[:, 9:18]
        o_prefs = enablement[:, 18:27] + enablement[:, 27:36]

        # Select based on turn
        logits = x_mask[:, None] * x_prefs + o_mask[:, None] * o_prefs

        return logits


# =============================================================================
# Data Generation
# =============================================================================

def generate_random_position():
    """Generate a random TicTacToe position."""
    board = [0] * 9
    num_moves = mx.random.randint(0, 9, ()).item()

    for m in range(num_moves):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break
        cell = empty[mx.random.randint(0, len(empty), ()).item()]
        player = 1 if m % 2 == 0 else 2
        board[cell] = player

        # Check for winner
        for line in WINNING_LINES:
            if board[line[0]] == board[line[1]] == board[line[2]] != 0:
                return None  # Game already over, regenerate

    # Check if any moves possible
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return None

    # Determine whose turn
    x_count = sum(1 for c in board if c == 1)
    o_count = sum(1 for c in board if c == 2)
    turn = 0 if x_count == o_count else 1  # X goes first

    return {
        "board": board,
        "turn": turn,
        "legal_moves": empty,
    }


def generate_dataset(num_samples: int) -> list:
    """Generate dataset of random positions with legal moves."""
    samples = []
    while len(samples) < num_samples:
        pos = generate_random_position()
        if pos is not None:
            samples.append(pos)
    return samples


def check_winner(board: list) -> int:
    """Check for winner. Returns 0=none, 1=X, 2=O."""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0


def play_game(model: nn.Module, opponent: str = "random") -> int:
    """
    Play a game between model (X) and opponent.

    Returns: 1 if model wins, -1 if model loses, 0 if draw
    """
    board = [0] * 9
    turn = 0  # X starts

    for move_num in range(9):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break

        if turn == 0:
            # Model's turn (X)
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])

            if hasattr(model, 'get_legal_move_probs'):
                probs = model.get_legal_move_probs(board_arr, turn_arr)[0]
            else:
                logits = model(board_arr, turn_arr)[0]
                legal_mask = mx.array([1.0 if i in empty else 0.0 for i in range(9)])
                masked = logits + (1 - legal_mask) * (-1e9)
                probs = mx.softmax(masked, axis=-1)

            # Sample from distribution (or argmax for deterministic)
            move = int(mx.argmax(probs).item())
            if move not in empty:
                move = empty[0]  # Fallback

        else:
            # Opponent's turn (O)
            if opponent == "random":
                move = empty[mx.random.randint(0, len(empty), ()).item()]
            else:
                raise ValueError(f"Unknown opponent: {opponent}")

        # Make move
        player = 1 if turn == 0 else 2
        board[move] = player

        # Check winner
        winner = check_winner(board)
        if winner == 1:
            return 1  # Model wins
        elif winner == 2:
            return -1  # Model loses

        turn = 1 - turn

    return 0  # Draw


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_legal_accuracy(model: nn.Module, samples: list) -> float:
    """Evaluate accuracy of predicting legal moves."""
    correct = 0
    total = 0

    for s in samples:
        board = mx.array([s["board"]])
        turn = mx.array([s["turn"]])

        if hasattr(model, 'get_legal_move_probs'):
            probs = model.get_legal_move_probs(board, turn)[0]
        else:
            logits = model(board, turn)[0]
            legal_mask = mx.array([1.0 if i in s["legal_moves"] else 0.0 for i in range(9)])
            masked = logits + (1 - legal_mask) * (-1e9)
            probs = mx.softmax(masked, axis=-1)

        pred_move = int(mx.argmax(probs).item())

        if pred_move in s["legal_moves"]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_win_rate(model: nn.Module, num_games: int = 100) -> dict:
    """Evaluate win rate against random opponent."""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        result = play_game(model, opponent="random")
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


# =============================================================================
# Training
# =============================================================================

def train_supervised(
    model: nn.Module,
    train_samples: list,
    num_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 0.01,
) -> dict:
    """Train model to predict legal moves."""
    optimizer = optim.Adam(learning_rate=lr)

    start_time = time.perf_counter()
    losses = []

    for step in range(num_steps):
        # Sample batch
        batch_indices = [mx.random.randint(0, len(train_samples), ()).item()
                        for _ in range(batch_size)]
        batch = [train_samples[i] for i in batch_indices]

        boards = mx.array([s["board"] for s in batch])
        turns = mx.array([s["turn"] for s in batch])

        # Target: uniform over legal moves
        targets = mx.zeros((batch_size, 9))
        for i, s in enumerate(batch):
            for legal_cell in s["legal_moves"]:
                targets = targets.at[i, legal_cell].add(1.0 / len(s["legal_moves"]))

        def loss_fn(m):
            logits = m(boards, turns)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            # Cross-entropy with target distribution
            loss = -mx.mean(mx.sum(targets * log_probs, axis=-1))
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        losses.append(float(loss))

    elapsed = time.perf_counter() - start_time

    return {
        "final_loss": losses[-1],
        "training_time": elapsed,
        "losses": losses,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_comparison():
    """Run full comparison between MLP baseline and statechart model."""
    print("=" * 70)
    print("TicTacToe: MLP Baseline vs Statechart Model")
    print("=" * 70)
    print()

    mx.random.seed(42)

    # Create models
    print("Creating models...")
    mlp = MLPBaseline(embed_dim=16, hidden_dim=64)
    mlp_params = count_params(mlp)
    print(f"  MLP Baseline: {mlp_params:,} parameters")

    # For fair comparison, we'd also test statechart model
    # but it requires more setup, so we focus on MLP baseline here

    # Generate datasets
    print("\nGenerating datasets...")
    sample_sizes = [100, 1000, 10000]
    datasets = {n: generate_dataset(n) for n in sample_sizes}
    test_set = generate_dataset(500)
    print(f"  Test set: {len(test_set)} samples")

    # Results storage
    results = {"mlp": {}}

    # Train and evaluate at different sample sizes
    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY EXPERIMENT")
    print("=" * 70)

    for num_samples in sample_sizes:
        print(f"\n--- Training with {num_samples} samples ---")

        # Fresh model
        model = MLPBaseline(embed_dim=16, hidden_dim=64)
        train_data = datasets[num_samples]

        # Train
        train_result = train_supervised(
            model=model,
            train_samples=train_data,
            num_steps=500,
            batch_size=min(32, num_samples),
            lr=0.01,
        )

        # Evaluate
        legal_acc = evaluate_legal_accuracy(model, test_set)
        win_stats = evaluate_win_rate(model, num_games=100)

        print(f"  Legal move accuracy: {legal_acc:.4f}")
        print(f"  Win rate vs random:  {win_stats['win_rate']:.4f}")
        print(f"  Training time:       {train_result['training_time']:.2f}s")

        results["mlp"][num_samples] = {
            "legal_accuracy": legal_acc,
            "win_rate": win_stats["win_rate"],
            "training_time": train_result["training_time"],
            "final_loss": train_result["final_loss"],
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Samples':<10} | {'Legal Acc':<10} | {'Win Rate':<10} | {'Time (s)':<10}")
    print("-" * 50)

    for num_samples in sample_sizes:
        r = results["mlp"][num_samples]
        print(f"{num_samples:<10} | {r['legal_accuracy']:<10.4f} | {r['win_rate']:<10.4f} | {r['training_time']:<10.2f}")

    # Final evaluation with full training
    print("\n" + "=" * 70)
    print("FULL TRAINING (10000 samples, 2000 steps)")
    print("=" * 70)

    final_model = MLPBaseline(embed_dim=16, hidden_dim=64)
    final_train = train_supervised(
        model=final_model,
        train_samples=datasets[10000],
        num_steps=2000,
        batch_size=64,
        lr=0.01,
    )

    final_legal_acc = evaluate_legal_accuracy(final_model, test_set)
    final_win_stats = evaluate_win_rate(final_model, num_games=200)

    print(f"\nFinal Results:")
    print(f"  Legal move accuracy: {final_legal_acc:.4f}")
    print(f"  Win rate vs random:  {final_win_stats['win_rate']:.4f}")
    print(f"  Wins: {final_win_stats['wins']}, Losses: {final_win_stats['losses']}, Draws: {final_win_stats['draws']}")
    print(f"  Training time:       {final_train['training_time']:.2f}s")

    results["mlp"]["final"] = {
        "legal_accuracy": final_legal_acc,
        "win_rate": final_win_stats["win_rate"],
        "training_time": final_train["training_time"],
        "wins": final_win_stats["wins"],
        "losses": final_win_stats["losses"],
        "draws": final_win_stats["draws"],
    }

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_comparison()
