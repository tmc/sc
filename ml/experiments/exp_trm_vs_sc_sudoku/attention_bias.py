"""
Attention-Based Constraint Encoding for Sudoku.

Key insight: Instead of hard logit masking (blocks gradients),
modulate attention patterns to encode Sudoku constraints as
soft inductive bias.

Hard guards:  logits += log(valid_mask)  → Gradient blocked
Soft attention: attention[i,j] *= constraint_affinity[i,j] → Gradients flow

The model learns WHEN constraints matter, not just WHAT they are.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class AttentionBiasConfig:
    """Configuration for attention-biased TRM."""
    # Base TRM config
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    num_cells: int = 81
    num_digits: int = 9
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1

    # Attention bias config
    use_constraint_bias: bool = True
    initial_bias_scale: float = 1.0      # Starting strength of constraint bias
    final_bias_scale: float = 5.0        # Ending strength (after annealing)
    learnable_bias: bool = True          # Learn to modulate bias per-head
    bias_temperature: float = 1.0        # Temperature for bias softmax


def build_constraint_affinity_matrix() -> mx.array:
    """
    Build a 81x81 constraint affinity matrix for Sudoku.

    affinity[i,j] = 1 if cells i and j share a constraint (row/col/box)
                  = 0 otherwise

    This matrix encodes which cell pairs need to "attend to each other"
    for constraint checking.
    """
    affinity = mx.zeros((81, 81))

    # For each pair of cells
    for i in range(81):
        row_i = i // 9
        col_i = i % 9
        box_i = (row_i // 3) * 3 + (col_i // 3)

        for j in range(81):
            row_j = j // 9
            col_j = j % 9
            box_j = (row_j // 3) * 3 + (col_j // 3)

            # Same row, column, or box?
            same_row = row_i == row_j
            same_col = col_i == col_j
            same_box = box_i == box_j

            if same_row or same_col or same_box:
                # Use a one-hot style assignment since MLX doesn't have item assignment
                pass  # Will build with a different approach

    # Build more efficiently using broadcasting
    cells = mx.arange(81)
    rows = cells // 9
    cols = cells % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    # Compare all pairs
    rows_i = rows[:, None]  # [81, 1]
    rows_j = rows[None, :]  # [1, 81]
    cols_i = cols[:, None]
    cols_j = cols[None, :]
    boxes_i = boxes[:, None]
    boxes_j = boxes[None, :]

    same_row = (rows_i == rows_j).astype(mx.float32)
    same_col = (cols_i == cols_j).astype(mx.float32)
    same_box = (boxes_i == boxes_j).astype(mx.float32)

    # Union of constraints (max because we just need "is related")
    affinity = mx.maximum(mx.maximum(same_row, same_col), same_box)

    # Remove self-connections (diagonal)
    eye = mx.eye(81)
    affinity = affinity * (1 - eye)

    return affinity


def build_detailed_constraint_matrix() -> Dict[str, mx.array]:
    """
    Build separate affinity matrices for row, column, and box constraints.

    Returns dict with 'row', 'col', 'box' matrices, each [81, 81].
    """
    cells = mx.arange(81)
    rows = cells // 9
    cols = cells % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    rows_i = rows[:, None]
    rows_j = rows[None, :]
    cols_i = cols[:, None]
    cols_j = cols[None, :]
    boxes_i = boxes[:, None]
    boxes_j = boxes[None, :]

    eye = mx.eye(81)

    return {
        'row': (rows_i == rows_j).astype(mx.float32) * (1 - eye),
        'col': (cols_i == cols_j).astype(mx.float32) * (1 - eye),
        'box': (boxes_i == boxes_j).astype(mx.float32) * (1 - eye),
    }


class ConstraintBiasedAttention(nn.Module):
    """
    Multi-head attention with learnable constraint bias.

    Adds constraint affinity matrix to attention logits, with
    learnable per-head scaling.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        constraint_affinity: mx.array,
        learnable_scale: bool = True,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Constraint affinity matrix [81, 81]
        self.constraint_affinity = constraint_affinity

        # Per-head learnable bias scale
        if learnable_scale:
            # Initialize with small positive values
            self.bias_scales = mx.ones((num_heads,)) * initial_scale

    def __call__(
        self,
        x: mx.array,
        bias_strength: float = 1.0,
    ) -> mx.array:
        """
        Apply attention with constraint bias.

        Args:
            x: Input [batch, 81, hidden_dim]
            bias_strength: Global multiplier for constraint bias (for annealing)

        Returns:
            Output [batch, 81, hidden_dim]
        """
        B, N, D = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Shape: [B, num_heads, 81, head_dim]

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B, num_heads, 81, 81]

        # Add constraint bias
        # bias_scales: [num_heads] -> [1, num_heads, 1, 1]
        scales = self.bias_scales[None, :, None, None]
        # constraint_affinity: [81, 81] -> [1, 1, 81, 81]
        bias = self.constraint_affinity[None, None, :, :] * scales * bias_strength

        scores = scores + bias

        # Softmax and weighted sum
        attn_weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn_weights, v)  # [B, num_heads, 81, head_dim]

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        out = self.o_proj(out)

        return out


class TransformerBlockWithBias(nn.Module):
    """Transformer block with constraint-biased attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        constraint_affinity: mx.array,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = ConstraintBiasedAttention(
            hidden_dim, num_heads, constraint_affinity, learnable_scale=True
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, bias_strength: float = 1.0) -> mx.array:
        # Attention with residual
        h = self.ln1(x)
        h = self.attn(h, bias_strength=bias_strength)
        h = self.dropout(h)
        x = x + h

        # FFN with residual
        h = self.ln2(x)
        h = self.ff(h)
        h = self.dropout(h)
        x = x + h

        return x


class AttentionBiasTRM(nn.Module):
    """
    TRM with attention-based constraint encoding.

    Instead of hard guards on output logits, encodes Sudoku
    constraints as attention bias. This provides:
    1. Soft inductive bias (gradients flow freely)
    2. Learnable constraint importance per head
    3. Annealing from soft to firm constraints
    """

    def __init__(self, config: AttentionBiasConfig):
        super().__init__()
        self.config = config

        # Build constraint affinity matrix
        self.constraint_affinity = build_constraint_affinity_matrix()

        # Cell embedding (0-9: empty + digits)
        self.cell_embed = nn.Embedding(10, config.hidden_dim)

        # Position embedding (81 cells)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # H-level context network
        self.h_context_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Transformer layers with constraint bias
        self.layers = [
            TransformerBlockWithBias(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
                self.constraint_affinity,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ]

        # Output head
        self.output_head = nn.Linear(config.hidden_dim, config.num_digits)

        # Layer norm
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def encode(self, puzzle: mx.array) -> mx.array:
        """Encode puzzle to initial hidden states."""
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        # Get embeddings
        cell_emb = self.cell_embed(puzzle.astype(mx.int32))  # [B, 81, D]
        pos_emb = self.pos_embed(positions)  # [B, 81, D]

        return cell_emb + pos_emb

    def refine(
        self,
        h: mx.array,
        h_context: mx.array,
        bias_strength: float = 1.0,
    ) -> mx.array:
        """One L-cycle refinement step."""
        # Add H-context
        h = h + h_context[:, None, :]

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, bias_strength=bias_strength)

        return h

    def predict(self, h: mx.array) -> mx.array:
        """Predict digit logits from hidden states."""
        h = self.ln_out(h)
        return self.output_head(h)  # [B, 81, 9]

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        bias_anneal: bool = True,
    ) -> Dict[str, mx.array]:
        """
        Solve puzzles using iterative refinement with constraint bias.

        Args:
            puzzle: [B, 81] puzzle (0=empty, 1-9=given)
            bias_anneal: If True, increase bias strength across H-cycles
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Compute bias strength schedule
        if bias_anneal:
            initial = self.config.initial_bias_scale
            final = self.config.final_bias_scale
            bias_schedule = [
                initial + (final - initial) * h / (H - 1) if H > 1 else final
                for h in range(H)
            ]
        else:
            bias_schedule = [self.config.initial_bias_scale] * H

        # Initial encoding
        h = self.encode(puzzle)

        for hi in range(H):
            bias_strength = bias_schedule[hi]

            # Compute H-level context
            pooled = mx.mean(h, axis=1)
            h_context = self.h_context_net(pooled)

            for li in range(L):
                h = self.refine(h, h_context, bias_strength=bias_strength)

        # Final prediction
        logits = self.predict(h)  # [B, 81, 9]
        predictions = mx.argmax(logits, axis=-1) + 1  # [B, 81]

        return {
            'logits': logits,
            'predictions': predictions,
        }

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        bias_anneal: bool = True,
    ) -> Tuple[mx.array, Dict]:
        """Compute cross-entropy loss."""
        result = self.solve(puzzle, bias_anneal=bias_anneal)
        logits = result['logits']

        # Cross-entropy loss
        targets = (solution - 1).astype(mx.int32)  # [B, 81], 0-indexed
        B, C, D = logits.shape

        # Flatten for loss
        logits_flat = logits.reshape(-1, D)
        targets_flat = targets.reshape(-1)

        # Stable cross-entropy
        logits_flat = mx.clip(logits_flat, -30, 30)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

        # Gather correct class
        batch_indices = mx.arange(logits_flat.shape[0])
        correct_log_probs = log_probs[batch_indices, targets_flat]
        loss = -mx.mean(correct_log_probs)

        # Metrics
        predictions = result['predictions']
        accuracy = mx.mean((predictions == solution).astype(mx.float32))

        metrics = {
            'loss': float(loss.item()),
            'accuracy': float(accuracy.item()),
        }

        return loss, metrics


def train_and_evaluate():
    """Train attention-biased TRM and compare with vanilla."""
    from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 70)
    print("ATTENTION BIAS EXPERIMENT")
    print("=" * 70)

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(1000, 200, seed=42)

    # Train vanilla TRM for comparison
    print("\n" + "-" * 70)
    print("Training Vanilla TRM (baseline)...")
    print("-" * 70)

    vanilla_config = VanillaTRMConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
    )
    vanilla_model = VanillaTRM(vanilla_config)
    vanilla_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def vanilla_loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    vanilla_loss_and_grad = nn.value_and_grad(vanilla_model, vanilla_loss_fn)

    for epoch in range(30):
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        for i in range(0, train_p.shape[0], 32):
            p = train_p_shuf[i:i+32]
            s = train_s_shuf[i:i+32]
            loss, grads = vanilla_loss_and_grad(vanilla_model, p, s)
            vanilla_optimizer.update(vanilla_model, grads)
            mx.eval(vanilla_model.parameters())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    # Evaluate vanilla
    vanilla_result = vanilla_model.solve(test_p)
    vanilla_acc = float(mx.mean((vanilla_result['predictions'] == test_s).astype(mx.float32)).item())
    print(f"Vanilla TRM accuracy: {vanilla_acc:.1%}")

    # Train attention-biased TRM
    print("\n" + "-" * 70)
    print("Training Attention-Biased TRM...")
    print("-" * 70)

    bias_config = AttentionBiasConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
        use_constraint_bias=True,
        initial_bias_scale=0.5,
        final_bias_scale=3.0,
        learnable_bias=True,
    )
    bias_model = AttentionBiasTRM(bias_config)
    bias_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def bias_loss_fn(m, p, s):
        loss, _ = m.loss(p, s, bias_anneal=True)
        return loss

    bias_loss_and_grad = nn.value_and_grad(bias_model, bias_loss_fn)

    for epoch in range(30):
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        for i in range(0, train_p.shape[0], 32):
            p = train_p_shuf[i:i+32]
            s = train_s_shuf[i:i+32]
            loss, grads = bias_loss_and_grad(bias_model, p, s)
            bias_optimizer.update(bias_model, grads)
            mx.eval(bias_model.parameters())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    # Evaluate attention-biased
    bias_result = bias_model.solve(test_p, bias_anneal=True)
    bias_acc = float(mx.mean((bias_result['predictions'] == test_s).astype(mx.float32)).item())
    print(f"Attention-Biased TRM accuracy: {bias_acc:.1%}")

    # Compare
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Vanilla TRM:         {vanilla_acc:.1%}")
    print(f"Attention-Biased:    {bias_acc:.1%}")
    print(f"Improvement:         {bias_acc - vanilla_acc:+.1%}")

    # Analyze learned bias scales
    print("\n" + "-" * 70)
    print("LEARNED BIAS SCALES PER HEAD")
    print("-" * 70)
    for layer_idx, layer in enumerate(bias_model.layers):
        scales = layer.attn.bias_scales
        mx.eval(scales)
        print(f"Layer {layer_idx}: {[f'{float(s.item()):.3f}' for s in scales]}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Attention bias provides soft inductive bias for Sudoku constraints
2. Gradients flow freely (unlike hard logit masking)
3. Bias annealing: soft early, firm late (exploration → exploitation)
4. Per-head scales show which heads use constraints most
5. Model learns WHEN constraints matter, not just WHAT they are
""")


if __name__ == "__main__":
    train_and_evaluate()
