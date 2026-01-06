#!/usr/bin/env python3
"""
Hard Mask TRM: Multiplicative constraint masking instead of additive bias.

Hypothesis: Additive attention bias is too soft - the model can learn to ignore it.
Hard masking enforces constraints by zeroing out attention to unrelated cells.

Key difference from attention_bias:
- attention_bias: attn = softmax(QK^T + bias)  # Can still attend to anything
- hard_mask: attn = softmax(QK^T) * mask       # Forced to zero for unrelated cells
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardMaskConfig:
    """Config for hard mask TRM."""
    vocab_size: int = 10
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 81
    mask_type: str = "sudoku"  # "sudoku", "local", or "global"
    mask_strength: float = 1.0  # 1.0 = full mask, <1.0 = partial


def create_sudoku_mask() -> mx.array:
    """Create mask where 1 = cells that share row/col/box, 0 = unrelated."""
    n = 81
    mask = np.eye(n)  # Self-attention always allowed

    for i in range(n):
        row_i, col_i = i // 9, i % 9
        box_i = (row_i // 3) * 3 + (col_i // 3)

        for j in range(n):
            row_j, col_j = j // 9, j % 9
            box_j = (row_j // 3) * 3 + (col_j // 3)

            # Allow attention if same row, col, or box
            if row_i == row_j or col_i == col_j or box_i == box_j:
                mask[i, j] = 1.0

    return mx.array(mask)


def create_local_mask(window: int = 9) -> mx.array:
    """Create local attention mask (each cell attends to nearby cells)."""
    n = 81
    mask = np.zeros((n, n))

    for i in range(n):
        row_i, col_i = i // 9, i % 9
        for j in range(n):
            row_j, col_j = j // 9, j % 9
            # Manhattan distance
            if abs(row_i - row_j) + abs(col_i - col_j) <= window:
                mask[i, j] = 1.0

    return mx.array(mask)


class HardMaskAttention(nn.Module):
    """Attention with hard masking for Sudoku constraints."""

    def __init__(self, config: HardMaskConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Create constraint mask
        if config.mask_type == "sudoku":
            self.mask = create_sudoku_mask()
        elif config.mask_type == "local":
            self.mask = create_local_mask()
        else:
            self.mask = mx.ones((81, 81))  # Global attention

        # Learnable mask refinement (optional)
        self.mask_scale = mx.array([config.mask_strength])

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Apply softmax first
        attn_probs = mx.softmax(attn_scores, axis=-1)

        # HARD MASK: Zero out attention to unrelated cells
        # This is the key difference from additive bias
        effective_mask = self.mask * self.mask_scale + (1 - self.mask_scale)
        attn_probs = attn_probs * effective_mask

        # Renormalize (important for maintaining attention distribution)
        attn_probs = attn_probs / (mx.sum(attn_probs, axis=-1, keepdims=True) + 1e-8)

        out = attn_probs @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        return self.o_proj(out)


class HardMaskBlock(nn.Module):
    """Transformer block with hard masking."""

    def __init__(self, config: HardMaskConfig):
        super().__init__()
        self.attention = HardMaskAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HardMaskTRM(nn.Module):
    """
    TRM with hard constraint masking.

    Instead of soft attention bias, this model uses multiplicative masking
    to enforce that cells can only attend to related cells (same row/col/box).
    """

    def __init__(self, config: Optional[HardMaskConfig] = None):
        super().__init__()
        self.config = config or HardMaskConfig()

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        self.blocks = [HardMaskBlock(self.config) for _ in range(self.config.num_layers)]
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, L = input_ids.shape
        positions = mx.arange(L)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)

    def solve(self, puzzle: mx.array, temperature: float = 0.0, **kwargs) -> dict:
        """Solve a puzzle, returning dict with predictions and logits."""
        logits = self(puzzle)

        if temperature > 0:
            predictions = mx.argmax(mx.softmax(logits / temperature, axis=-1), axis=-1)
        else:
            predictions = mx.argmax(logits, axis=-1)

        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)
        return {"predictions": predictions, "logits": logits}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        """Compute cross-entropy loss for training."""
        logits = self(puzzle)
        target = solution
        empty_mask = (puzzle == 0).astype(mx.float32)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        target_expanded = target[:, :, None]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)
        loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == target).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        return loss, {"cell_accuracy": cell_acc}


# Variant: Hard mask with iterative refinement
class HardMaskIterativeTRM(nn.Module):
    """Hard mask + iterative refinement for maximum constraint propagation."""

    def __init__(self, config: Optional[HardMaskConfig] = None, num_iterations: int = 3):
        super().__init__()
        self.config = config or HardMaskConfig()
        self.num_iterations = num_iterations

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        # Prediction -> embedding projection
        self.pred_proj = nn.Linear(self.config.vocab_size, self.config.hidden_size)

        self.blocks = [HardMaskBlock(self.config) for _ in range(self.config.num_layers)]
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def _forward_once(self, x: mx.array, positions: mx.array) -> mx.array:
        """Single forward pass."""
        x = x + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, L = input_ids.shape
        positions = mx.arange(L)

        # Mask for given cells
        given_mask = (input_ids > 0).astype(mx.float32)[..., None]

        # Initial embedding
        x = self.embedding(input_ids)
        logits = None

        for i in range(self.num_iterations):
            if i > 0 and logits is not None:
                # Mix predictions with original input
                soft_pred = mx.softmax(logits, axis=-1)
                pred_embed = self.pred_proj(soft_pred)
                # Given cells keep original embedding
                x = given_mask * self.embedding(input_ids) + (1 - given_mask) * pred_embed

            logits = self._forward_once(x, positions)

        return logits

    def solve(self, puzzle: mx.array, temperature: float = 0.0, **kwargs) -> dict:
        logits = self(puzzle)
        if temperature > 0:
            predictions = mx.argmax(mx.softmax(logits / temperature, axis=-1), axis=-1)
        else:
            predictions = mx.argmax(logits, axis=-1)
        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)
        return {"predictions": predictions, "logits": logits}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        """Compute cross-entropy loss for training."""
        logits = self(puzzle)
        target = solution
        empty_mask = (puzzle == 0).astype(mx.float32)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        target_expanded = target[:, :, None]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)
        loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == target).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        return loss, {"cell_accuracy": cell_acc}


def create_model(config: Optional[HardMaskConfig] = None) -> HardMaskTRM:
    """Factory function."""
    return HardMaskTRM(config)


def create_iterative_model(config: Optional[HardMaskConfig] = None, num_iterations: int = 3) -> HardMaskIterativeTRM:
    """Factory function for iterative variant."""
    return HardMaskIterativeTRM(config, num_iterations)


if __name__ == "__main__":
    print("Testing Hard Mask TRM...")
    config = HardMaskConfig(mask_type="sudoku")
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")
    print(f"  Mask type: {config.mask_type}")

    # Check mask sparsity
    mask = create_sudoku_mask()
    sparsity = 1 - (mx.sum(mask) / (81 * 81)).item()
    print(f"  Mask sparsity: {sparsity:.1%} zeros")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Parameters: {num_params:,}")

    print("\nTesting Hard Mask Iterative TRM...")
    model_iter = create_iterative_model(config, num_iterations=3)
    logits_iter = model_iter(batch)
    print(f"  Output: {logits_iter.shape}")
    print(f"  Iterations: 3")
