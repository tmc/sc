"""
Transformer Baseline for 9x9 Go

Measures: What % of Transformer outputs are illegal moves?

Unlike the statechart which GUARANTEES legal moves, a Transformer
must LEARN to avoid illegal moves from training data.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time
from typing import List, Tuple, Optional
from go_statechart import (
    Go9x9Statechart, BOARD_SIZE, TOTAL_POINTS,
    BLACK, WHITE, EMPTY, TurnState
)


# =============================================================================
# MODEL
# =============================================================================

class GoTransformer(nn.Module):
    """
    Simple Transformer for Go move prediction.

    Input: 81 board positions (0=empty, 1=black, 2=white) + turn (1=black, 2=white)
    Output: 82 logits (81 positions + pass)
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.stone_embed = nn.Embedding(3, d_model)  # empty, black, white
        self.pos_embed = nn.Embedding(TOTAL_POINTS, d_model)
        self.turn_embed = nn.Embedding(3, d_model)  # 0=unused, 1=black, 2=white

        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'attn': nn.MultiHeadAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm2': nn.LayerNorm(d_model),
            })

        # Output head
        self.output_head = nn.Linear(d_model, TOTAL_POINTS + 1)  # 81 + pass

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            board: (batch, 81) board positions
            turn: (batch,) turn (1=black, 2=white)

        Returns:
            logits: (batch, 82) move logits
        """
        batch_size = board.shape[0]

        # Embed board positions
        pos_ids = mx.arange(TOTAL_POINTS)
        pos_ids = mx.broadcast_to(pos_ids, (batch_size, TOTAL_POINTS))

        x = self.stone_embed(board) + self.pos_embed(pos_ids)

        # Add turn embedding to all positions
        turn_emb = self.turn_embed(turn)  # (batch, d_model)
        turn_emb = mx.expand_dims(turn_emb, axis=1)  # (batch, 1, d_model)
        x = x + turn_emb

        # Transformer layers
        for layer in self.layers:
            # Self-attention
            attn_out = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)

            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # Pool and output
        x = mx.mean(x, axis=1)  # (batch, d_model)
        logits = self.output_head(x)  # (batch, 82)

        return logits


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_game_data(num_games: int = 100) -> List[Tuple[mx.array, mx.array, int]]:
    """
    Generate training data from random games.

    Each example: (board_state, turn, legal_move_index)
    """
    data = []

    for _ in range(num_games):
        game = Go9x9Statechart()

        while not game.is_game_over() and len(game.move_history) < 100:
            legal_moves = game.get_legal_moves()

            if not legal_moves or random.random() < 0.1:
                # Pass
                board = mx.array(game.board.stones)
                turn = mx.array([game.current_player()])
                data.append((board, turn, TOTAL_POINTS))  # Pass = index 81
                game.play_pass()
            else:
                # Record state and legal move
                board = mx.array(game.board.stones)
                turn = mx.array([game.current_player()])

                x, y = random.choice(legal_moves)
                move_idx = y * BOARD_SIZE + x
                data.append((board, turn, move_idx))

                game.play_move(x, y)

    return data


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_illegal_rate(model: GoTransformer, num_games: int = 50) -> dict:
    """
    Evaluate how often the model produces illegal moves.
    """
    total_moves = 0
    illegal_moves = 0
    games_completed = 0

    for game_idx in range(num_games):
        game = Go9x9Statechart()

        while not game.is_game_over() and len(game.move_history) < 150:
            # Get model prediction
            board = mx.array([game.board.stones])
            turn = mx.array([game.current_player()])

            logits = model(board, turn)
            probs = mx.softmax(logits, axis=-1)

            # Sample from distribution
            move_idx = int(mx.argmax(probs[0]).item())

            total_moves += 1

            if move_idx == TOTAL_POINTS:
                # Pass
                game.play_pass()
            else:
                x, y = move_idx % BOARD_SIZE, move_idx // BOARD_SIZE

                if game.is_legal_move(x, y):
                    game.play_move(x, y)
                else:
                    # Illegal move!
                    illegal_moves += 1

                    # Retry with legal move (so game can continue)
                    legal = game.get_legal_moves()
                    if legal:
                        lx, ly = random.choice(legal)
                        game.play_move(lx, ly)
                    else:
                        game.play_pass()

        games_completed += 1

    illegal_rate = illegal_moves / total_moves * 100 if total_moves > 0 else 0

    return {
        "total_moves": total_moves,
        "illegal_moves": illegal_moves,
        "illegal_rate": illegal_rate,
        "games": games_completed,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model: GoTransformer, data: list, epochs: int = 10, batch_size: int = 32):
    """Train the model on generated data."""
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(model, batch_boards, batch_turns, batch_labels):
        logits = model(batch_boards, batch_turns)
        return mx.mean(nn.losses.cross_entropy(logits, batch_labels))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < batch_size:
                continue

            boards = mx.stack([b[0] for b in batch])
            turns = mx.concatenate([b[1] for b in batch])
            labels = mx.array([b[2] for b in batch])

            loss, grads = loss_and_grad_fn(model, boards, turns, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("9x9 Go Transformer Baseline")
    print("=" * 70)

    # Create model
    model = GoTransformer(d_model=64, n_heads=4, n_layers=2)

    # Count parameters
    def count_params(params, total=0):
        for v in params.values():
            if isinstance(v, dict):
                total = count_params(v, total)
            elif hasattr(v, 'size'):
                total += v.size
        return total

    print(f"\nModel parameters: {count_params(model.parameters()):,}")

    # Evaluate BEFORE training
    print("\n" + "-" * 40)
    print("BEFORE TRAINING (random weights)")
    print("-" * 40)

    before = evaluate_illegal_rate(model, num_games=20)
    print(f"Total moves: {before['total_moves']}")
    print(f"Illegal moves: {before['illegal_moves']}")
    print(f"Illegal rate: {before['illegal_rate']:.1f}%")

    # Generate training data
    print("\n" + "-" * 40)
    print("Generating training data...")
    print("-" * 40)

    data = generate_game_data(num_games=200)
    print(f"Generated {len(data)} training examples")

    # Train
    print("\n" + "-" * 40)
    print("Training...")
    print("-" * 40)

    train_model(model, data, epochs=10, batch_size=32)

    # Evaluate AFTER training
    print("\n" + "-" * 40)
    print("AFTER TRAINING")
    print("-" * 40)

    after = evaluate_illegal_rate(model, num_games=20)
    print(f"Total moves: {after['total_moves']}")
    print(f"Illegal moves: {after['illegal_moves']}")
    print(f"Illegal rate: {after['illegal_rate']:.1f}%")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"""
| Approach          | Illegal Rate |
|-------------------|--------------|
| Statechart        | 0.0%         |
| Transformer (raw) | {before['illegal_rate']:.1f}%        |
| Transformer (trained) | {after['illegal_rate']:.1f}%    |
""")

    print("=" * 70)
    print("KEY INSIGHT: Statechart is 0% by CONSTRUCTION")
    print("Transformer must LEARN to avoid illegal moves")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
