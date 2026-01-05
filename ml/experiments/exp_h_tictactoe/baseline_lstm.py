"""
LSTM Baseline for TicTacToe

Compares against DifferentiableTicTacToe statechart model.

Architecture: BoardEmbed -> LSTM -> Linear -> Softmax
Input: sequence of (board_state, move) history
Output: next move probabilities (9 cells)

Metrics:
1. Legal move accuracy (% of moves that are valid)
2. Win rate vs random player
3. Sample efficiency (games to reach 50% win rate)
4. Training time per epoch
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Winning lines for TicTacToe
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Cols
    (0, 4, 8), (2, 4, 6),             # Diagonals
]


class LSTMTicTacToe(nn.Module):
    """LSTM baseline for TicTacToe move prediction.

    Input: Board state sequence [B, T, 9] where each cell is 0/1/2
    Output: Move probabilities [B, 9]
    """

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Board cell embedding: 0=empty, 1=X, 2=O
        self.cell_embed = nn.Embedding(3, embed_dim)

        # Project 9 cells to input dim
        self.input_proj = nn.Linear(9 * embed_dim, hidden_dim)

        # LSTM layers (MLX doesn't have nn.LSTM, use manual GRU-like cells)
        self.lstm_cell = LSTMCell(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9),
        )

    def __call__(self, board_seq: mx.array) -> mx.array:
        """Forward pass.

        Args:
            board_seq: [B, T, 9] board states over time

        Returns:
            move_probs: [B, 9] next move probabilities
        """
        B, T, _ = board_seq.shape

        # Embed and project each timestep
        # [B, T, 9] -> [B, T, 9, embed_dim] -> [B, T, 9*embed_dim]
        embedded = self.cell_embed(board_seq.astype(mx.int32))
        embedded = embedded.reshape(B, T, -1)

        # Project to hidden dim
        x = mx.tanh(self.input_proj(embedded))  # [B, T, hidden_dim]

        # Run LSTM
        h = mx.zeros((B, self.hidden_dim))
        c = mx.zeros((B, self.hidden_dim))

        for t in range(T):
            h, c = self.lstm_cell(x[:, t, :], h, c)

        # Output
        logits = self.output_proj(h)  # [B, 9]
        return mx.softmax(logits, axis=-1)

    def predict_move(self, board: mx.array) -> mx.array:
        """Predict next move given current board.

        Args:
            board: [B, 9] current board

        Returns:
            move_probs: [B, 9] probabilities
        """
        # Treat as sequence of length 1
        board_seq = board[:, None, :]  # [B, 1, 9]
        return self(board_seq)


class LSTMCell(nn.Module):
    """Single LSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates: input, forget, cell, output
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def __call__(self, x: mx.array, h: mx.array, c: mx.array) -> tuple:
        """LSTM cell forward.

        Args:
            x: [B, input_dim] input
            h: [B, hidden_dim] hidden state
            c: [B, hidden_dim] cell state

        Returns:
            (new_h, new_c)
        """
        combined = mx.concatenate([x, h], axis=-1)
        gates = self.gates(combined)

        # Split into 4 gates
        i, f, g, o = mx.split(gates, 4, axis=-1)

        i = mx.sigmoid(i)  # Input gate
        f = mx.sigmoid(f)  # Forget gate
        g = mx.tanh(g)     # Cell candidate
        o = mx.sigmoid(o)  # Output gate

        new_c = f * c + i * g
        new_h = o * mx.tanh(new_c)

        return new_h, new_c


class TicTacToeEnv:
    """Simple TicTacToe environment for training and evaluation."""

    def __init__(self):
        self.reset()

    def reset(self) -> mx.array:
        """Reset to empty board, X to play."""
        self.board = mx.zeros((9,), dtype=mx.int32)
        self.current_player = 1  # 1=X, 2=O
        self.done = False
        self.winner = 0  # 0=none, 1=X, 2=O, 3=draw
        return self.board

    def get_legal_moves(self) -> list:
        """Get list of legal move indices."""
        return [i for i in range(9) if int(self.board[i]) == 0]

    def check_winner(self) -> int:
        """Check for winner. Returns 0=none, 1=X, 2=O, 3=draw."""
        board_list = self.board.tolist()

        for line in WINNING_LINES:
            vals = [board_list[i] for i in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return vals[0]

        # Check for draw
        if 0 not in board_list:
            return 3

        return 0

    def step(self, move: int) -> tuple:
        """Take a move.

        Args:
            move: Cell index 0-8

        Returns:
            (board, reward, done, info)
        """
        if self.done or int(self.board[move]) != 0:
            return self.board, -1.0, True, {"illegal": True}

        # Place piece
        self.board = self.board.at[move].add(self.current_player)

        # Check winner
        self.winner = self.check_winner()

        if self.winner > 0:
            self.done = True
            if self.winner == 3:  # Draw
                reward = 0.0
            elif self.winner == self.current_player:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0
            # Switch player
            self.current_player = 3 - self.current_player

        return self.board, reward, self.done, {"winner": self.winner}


def generate_training_data(num_games: int = 1000) -> tuple:
    """Generate training data from random self-play.

    Returns:
        boards: [N, 9] board states
        moves: [N] move indices taken
        outcomes: [N] game outcomes (1=win, 0=draw, -1=loss)
    """
    boards = []
    moves = []
    outcomes = []

    env = TicTacToeEnv()

    for _ in range(num_games):
        board = env.reset()
        game_boards = []
        game_moves = []
        game_players = []

        while not env.done:
            legal = env.get_legal_moves()
            if not legal:
                break

            # Random move
            move = legal[mx.random.randint(0, len(legal), ()).item()]

            game_boards.append(board.tolist())
            game_moves.append(move)
            game_players.append(env.current_player)

            board, _, done, info = env.step(move)

        # Assign outcomes based on winner
        winner = env.winner
        for i, player in enumerate(game_players):
            if winner == 0 or winner == 3:  # No winner or draw
                outcome = 0.0
            elif winner == player:
                outcome = 1.0
            else:
                outcome = -1.0

            boards.append(game_boards[i])
            moves.append(game_moves[i])
            outcomes.append(outcome)

    return (
        mx.array(boards, dtype=mx.int32),
        mx.array(moves, dtype=mx.int32),
        mx.array(outcomes, dtype=mx.float32),
    )


def compute_legal_move_accuracy(model: LSTMTicTacToe, boards: mx.array, moves: mx.array) -> float:
    """Compute accuracy of predicting legal moves."""
    probs = model.predict_move(boards)
    pred_moves = mx.argmax(probs, axis=-1)

    # Check if predicted moves are legal (cell is empty)
    legal = mx.array([int(boards[i, int(pred_moves[i])]) == 0 for i in range(len(boards))])
    return float(mx.mean(legal))


def evaluate_vs_random(model: LSTMTicTacToe, num_games: int = 100) -> dict:
    """Evaluate model playing as X against random O player.

    Returns:
        dict with wins, losses, draws, illegal_moves
    """
    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0

    env = TicTacToeEnv()

    for _ in range(num_games):
        board = env.reset()

        while not env.done:
            if env.current_player == 1:  # X = model
                probs = model.predict_move(board[None, :])[0]

                # Mask illegal moves
                legal_mask = mx.array([1.0 if int(board[i]) == 0 else 0.0 for i in range(9)])
                masked_probs = probs * legal_mask

                if float(mx.sum(masked_probs)) > 0:
                    move = int(mx.argmax(masked_probs))
                else:
                    # All moves illegal (shouldn't happen)
                    illegal_moves += 1
                    break
            else:  # O = random
                legal = env.get_legal_moves()
                if not legal:
                    break
                move = legal[mx.random.randint(0, len(legal), ()).item()]

            board, _, done, info = env.step(move)

            if info.get("illegal"):
                illegal_moves += 1
                break

        if env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        elif env.winner == 3:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "illegal_moves": illegal_moves,
    }


def train_lstm_baseline():
    """Train LSTM baseline and report metrics."""
    print("=" * 70)
    print("LSTM Baseline for TicTacToe")
    print("=" * 70)

    # Create model with params ~48k to match statechart
    # LSTM params: gates (input+hidden -> 4*hidden) = (hidden_dim + hidden_dim) * 4 * hidden_dim
    # For hidden_dim=56: (56+56)*4*56 = 25088
    # Plus embeddings and projections
    model = LSTMTicTacToe(embed_dim=28, hidden_dim=56)

    # Count params
    params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"\nModel parameters: {params:,}")
    print(f"Target (statechart): ~48,853")

    # Generate training data
    print("\nGenerating training data...")
    mx.random.seed(42)
    boards, moves, outcomes = generate_training_data(num_games=5000)
    print(f"Training samples: {len(boards):,}")

    # Split train/val
    split = int(0.9 * len(boards))
    train_boards, val_boards = boards[:split], boards[split:]
    train_moves, val_moves = moves[:split], moves[split:]

    # Training setup
    optimizer = optim.Adam(learning_rate=0.001)
    batch_size = 64
    num_epochs = 20

    def loss_fn(model, batch_boards, batch_moves):
        probs = model.predict_move(batch_boards)
        # Cross-entropy loss
        log_probs = mx.log(probs + 1e-10)
        loss = -mx.mean(log_probs[mx.arange(len(batch_moves)), batch_moves])
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print("\nTraining...")
    print("-" * 70)

    epoch_times = []
    win_rates = []

    # Initial evaluation
    init_eval = evaluate_vs_random(model, num_games=100)
    print(f"Epoch   0: win_rate={init_eval['win_rate']:.2%} (before training)")
    win_rates.append(init_eval['win_rate'])

    for epoch in range(1, num_epochs + 1):
        start_time = time.perf_counter()

        # Shuffle
        perm = mx.random.permutation(mx.arange(len(train_boards)))
        train_boards = train_boards[perm]
        train_moves = train_moves[perm]

        # Train batches
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_boards), batch_size):
            batch_b = train_boards[i:i+batch_size]
            batch_m = train_moves[i:i+batch_size]

            loss, grads = loss_and_grad(model, batch_b, batch_m)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += float(loss)
            num_batches += 1

        epoch_time = time.perf_counter() - start_time
        epoch_times.append(epoch_time)
        avg_loss = total_loss / num_batches

        # Evaluate
        eval_result = evaluate_vs_random(model, num_games=100)
        win_rates.append(eval_result['win_rate'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, "
                  f"win_rate={eval_result['win_rate']:.2%}, "
                  f"time={epoch_time:.2f}s")

    print("-" * 70)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Legal move accuracy
    legal_acc = compute_legal_move_accuracy(model, val_boards, val_moves)
    print(f"\n1. Legal Move Accuracy: {legal_acc:.2%}")

    # Win rate vs random
    final_eval = evaluate_vs_random(model, num_games=500)
    print(f"\n2. Win Rate vs Random (500 games):")
    print(f"   Wins:   {final_eval['wins']} ({final_eval['wins']/5:.1f}%)")
    print(f"   Losses: {final_eval['losses']} ({final_eval['losses']/5:.1f}%)")
    print(f"   Draws:  {final_eval['draws']} ({final_eval['draws']/5:.1f}%)")
    print(f"   Illegal: {final_eval['illegal_moves']}")

    # Sample efficiency
    epochs_to_50 = -1
    for i, wr in enumerate(win_rates):
        if wr >= 0.50:
            epochs_to_50 = i
            break
    print(f"\n3. Sample Efficiency:")
    print(f"   Epochs to 50% win rate: {epochs_to_50 if epochs_to_50 >= 0 else '>20'}")
    print(f"   Training samples: {len(train_boards):,}")

    # Training time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\n4. Training Time:")
    print(f"   Avg epoch time: {avg_epoch_time:.2f}s")
    print(f"   Total training: {sum(epoch_times):.1f}s")

    print("\n" + "=" * 70)

    return {
        "params": params,
        "legal_move_accuracy": legal_acc,
        "win_rate": final_eval['win_rate'],
        "wins": final_eval['wins'],
        "losses": final_eval['losses'],
        "draws": final_eval['draws'],
        "epochs_to_50": epochs_to_50,
        "avg_epoch_time": avg_epoch_time,
        "total_time": sum(epoch_times),
    }


if __name__ == "__main__":
    results = train_lstm_baseline()

    print("\nSUMMARY FOR NOTES.md:")
    print("-" * 40)
    print(f"Parameters: {results['params']:,}")
    print(f"Legal Move Accuracy: {results['legal_move_accuracy']:.1%}")
    print(f"Win Rate vs Random: {results['win_rate']:.1%}")
    print(f"Epochs to 50% WR: {results['epochs_to_50']}")
    print(f"Training Time: {results['total_time']:.1f}s")
