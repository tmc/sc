"""
Transformer Baseline for TicTacToe

Compare transformer architecture against statechart-based model.
Matches parameter count for fair comparison.

Metrics:
1. Legal move accuracy
2. Win rate vs random
3. Sample efficiency
4. Training time
"""

import time
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from statechart import check_winner, get_legal_moves, is_winning_move, board_full


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for 9 board positions."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.pos_embed = mx.zeros((9, embed_dim))
    
    def __call__(self, x: mx.array) -> mx.array:
        """Add positional encoding to embeddings.
        
        Args:
            x: [B, 9, embed_dim] token embeddings
        Returns:
            [B, 9, embed_dim] with positional info
        """
        return x + self.pos_embed[None, :, :]


class TransformerTicTacToe(nn.Module):
    """Transformer baseline for TicTacToe move prediction.
    
    Architecture:
        - Token embedding: board cells (0=empty, 1=X, 2=O)
        - Positional encoding
        - Self-attention layer
        - Output projection to move probabilities
    
    Parameter budget: ~3000 to match statechart model
    """
    
    def __init__(self, embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding for cell values
        self.token_embed = nn.Embedding(3, embed_dim)  # 3 * 16 = 48
        
        # Positional encoding
        self.pos_encode = PositionalEncoding(embed_dim)  # 9 * 16 = 144
        
        # Self-attention layer
        self.attention = nn.MultiHeadAttention(
            dims=embed_dim,
            num_heads=num_heads,
        )  # 4 * embed_dim^2 = 4 * 256 = 1024
        
        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)  # 2 * 16 = 32
        self.norm2 = nn.LayerNorm(embed_dim)  # 2 * 16 = 32
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # 16*32 + 32 = 544
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),  # 32*16 + 16 = 528
        )
        
        # Output: aggregate and project to 9 move probabilities
        self.output_proj = nn.Linear(embed_dim * 9, 9)  # 144*9 + 9 = 1305
        
        # Total: ~3657 params
    
    def __call__(self, board: mx.array) -> mx.array:
        """Predict move probabilities.
        
        Args:
            board: [B, 9] cell values (0=empty, 1=X, 2=O)
        
        Returns:
            [B, 9] move probabilities (softmax)
        """
        B = board.shape[0]
        
        # Embed tokens
        x = self.token_embed(board)  # [B, 9, embed_dim]
        
        # Add positional encoding
        x = self.pos_encode(x)
        
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Flatten and project to moves
        x_flat = x.reshape(B, -1)  # [B, 9 * embed_dim]
        logits = self.output_proj(x_flat)  # [B, 9]
        
        return mx.softmax(logits, axis=-1)
    
    def get_legal_move_probs(self, board: mx.array) -> mx.array:
        """Get probabilities masked to legal moves only.
        
        Args:
            board: [B, 9] cell values
        
        Returns:
            [B, 9] probabilities for legal moves, 0 for occupied
        """
        probs = self(board)
        
        # Mask illegal moves
        legal_mask = mx.where(board == 0, 1.0, 0.0)
        masked_probs = probs * legal_mask
        
        # Renormalize
        total = mx.sum(masked_probs, axis=-1, keepdims=True)
        return masked_probs / (total + 1e-8)


def generate_training_data(num_games: int = 1000):
    """Generate training data from random games.
    
    Each sample: (board_state, optimal_move)
    Optimal move = winning move if available, else block opponent win, else random
    
    Returns:
        boards: [N, 9] board states
        moves: [N] optimal move indices
        legal_masks: [N, 9] legal move masks
    """
    boards = []
    moves = []
    legal_masks = []
    
    for _ in range(num_games):
        board = [0] * 9
        current_player = 1  # X starts
        
        while True:
            legal = get_legal_moves(board)
            if not legal:
                break
            
            # Record state before move
            boards.append(board.copy())
            legal_mask = [1.0 if board[i] == 0 else 0.0 for i in range(9)]
            legal_masks.append(legal_mask)
            
            # Find optimal move
            optimal = None
            
            # 1. Check for winning move
            for m in legal:
                if is_winning_move(board, current_player, m):
                    optimal = m
                    break
            
            # 2. Block opponent win
            if optimal is None:
                opponent = 3 - current_player
                for m in legal:
                    if is_winning_move(board, opponent, m):
                        optimal = m
                        break
            
            # 3. Take center if available
            if optimal is None and 4 in legal:
                optimal = 4
            
            # 4. Take corner
            if optimal is None:
                corners = [c for c in [0, 2, 6, 8] if c in legal]
                if corners:
                    optimal = random.choice(corners)
            
            # 5. Random
            if optimal is None:
                optimal = random.choice(legal)
            
            moves.append(optimal)
            
            # Make move
            board[optimal] = current_player
            
            # Check for win
            if check_winner(board, current_player):
                break
            
            # Switch player
            current_player = 3 - current_player
    
    return (
        mx.array(boards, dtype=mx.int32),
        mx.array(moves, dtype=mx.int32),
        mx.array(legal_masks),
    )


def compute_legal_accuracy(model, boards, moves, legal_masks):
    """Compute accuracy for legal moves."""
    probs = model(boards)
    
    # Mask illegal moves
    masked_probs = probs * legal_masks
    total = mx.sum(masked_probs, axis=-1, keepdims=True)
    normalized = masked_probs / (total + 1e-8)
    
    pred_moves = mx.argmax(normalized, axis=-1)
    correct = mx.sum(pred_moves == moves)
    
    return float(correct) / boards.shape[0]


def play_vs_random(model, num_games: int = 100) -> dict:
    """Play model (as X) vs random player (as O).
    
    Returns:
        wins, losses, draws counts
    """
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        board = [0] * 9
        current = 1  # X (model) starts
        
        while True:
            legal = get_legal_moves(board)
            if not legal:
                draws += 1
                break
            
            if current == 1:  # Model's turn
                board_arr = mx.array([board], dtype=mx.int32)
                probs = model.get_legal_move_probs(board_arr)[0]
                mx.eval(probs)
                move = int(mx.argmax(probs))
            else:  # Random opponent
                move = random.choice(legal)
            
            board[move] = current
            
            if check_winner(board, current):
                if current == 1:
                    wins += 1
                else:
                    losses += 1
                break
            
            current = 3 - current
    
    return {'wins': wins, 'losses': losses, 'draws': draws}


def train(num_epochs: int = 100, num_games: int = 500, learning_rate: float = 0.01):
    """Train transformer baseline.
    
    Returns:
        model, metrics dict
    """
    print("=" * 60)
    print("TRANSFORMER BASELINE: TicTacToe")
    print("=" * 60)
    
    # Create model
    model = TransformerTicTacToe(embed_dim=16, num_heads=2)
    
    # Count parameters
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, (list, tuple)):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total

    total_params = count_params(model.parameters())
    print(f"\nModel parameters: {total_params}")
    
    # Generate training data
    print(f"\nGenerating {num_games} training games...")
    start_gen = time.time()
    boards, moves, legal_masks = generate_training_data(num_games)
    gen_time = time.time() - start_gen
    print(f"Generated {boards.shape[0]} samples in {gen_time:.2f}s")
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Loss function
    def loss_fn(model):
        probs = model(boards)
        
        # Mask illegal moves before computing loss
        masked_probs = probs * legal_masks
        total = mx.sum(masked_probs, axis=-1, keepdims=True)
        normalized = masked_probs / (total + 1e-8)
        
        # Cross-entropy loss
        one_hot = mx.zeros_like(probs)
        for i in range(moves.shape[0]):
            one_hot = one_hot.at[i, moves[i]].add(1.0)
        
        # Clip for stability
        normalized = mx.clip(normalized, 1e-7, 1.0)
        return -mx.mean(mx.sum(one_hot * mx.log(normalized), axis=-1))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)
    
    start_train = time.time()
    history = {'loss': [], 'accuracy': [], 'win_rate': []}
    
    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        
        # Gradient clipping
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
        
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # Compute accuracy
        acc = compute_legal_accuracy(model, boards, moves, legal_masks)
        
        history['loss'].append(float(loss))
        history['accuracy'].append(acc)
        
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, accuracy={acc:.4f}")
    
    train_time = time.time() - start_train
    
    # Evaluate vs random
    print("\n" + "-" * 60)
    print("Evaluating vs random player...")
    results = play_vs_random(model, num_games=100)
    win_rate = results['wins'] / 100
    
    print(f"Win: {results['wins']}, Loss: {results['losses']}, Draw: {results['draws']}")
    print(f"Win rate: {win_rate:.1%}")
    
    metrics = {
        'total_params': total_params,
        'num_samples': boards.shape[0],
        'final_loss': history['loss'][-1],
        'final_accuracy': history['accuracy'][-1],
        'win_rate': win_rate,
        'train_time': train_time,
        'gen_time': gen_time,
    }
    
    return model, metrics


def main():
    """Run transformer baseline."""
    model, metrics = train(num_epochs=200, num_games=1000)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Transformer Baseline")
    print("=" * 60)
    
    print(f"\nModel:")
    print(f"  Parameters: {metrics['total_params']}")
    print(f"  Training samples: {metrics['num_samples']}")
    
    print(f"\nPerformance:")
    print(f"  Legal move accuracy: {metrics['final_accuracy']:.1%}")
    print(f"  Win rate vs random: {metrics['win_rate']:.1%}")
    
    print(f"\nEfficiency:")
    print(f"  Data generation: {metrics['gen_time']:.2f}s")
    print(f"  Training time: {metrics['train_time']:.2f}s")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    
    return metrics


if __name__ == "__main__":
    metrics = main()
