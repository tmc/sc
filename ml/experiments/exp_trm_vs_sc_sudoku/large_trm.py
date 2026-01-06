#!/usr/bin/env python3
"""
Large TRM: Increased model capacity to test if plateau is due to size.

Hypothesis: The 60% plateau might be due to insufficient model capacity
to represent full Sudoku constraint reasoning.

This model increases:
- hidden_size: 64 -> 256
- num_layers: 2 -> 4
- num_heads: 4 -> 8

With attention bias for Sudoku constraints.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LargeTRMConfig:
    """Config for large capacity TRM."""
    vocab_size: int = 10
    hidden_size: int = 256  # 4x larger
    num_heads: int = 8  # 2x more heads
    num_layers: int = 4  # 2x more layers
    dropout: float = 0.1
    max_seq_len: int = 81
    use_attention_bias: bool = True


def create_sudoku_bias() -> mx.array:
    """Create attention bias for Sudoku constraints."""
    n = 81
    bias = np.zeros((n, n))

    for i in range(n):
        row_i, col_i = i // 9, i % 9
        box_i = (row_i // 3) * 3 + (col_i // 3)

        for j in range(n):
            row_j, col_j = j // 9, j % 9
            box_j = (row_j // 3) * 3 + (col_j // 3)

            if i != j:
                if row_i == row_j:
                    bias[i, j] += 1.0
                if col_i == col_j:
                    bias[i, j] += 1.0
                if box_i == box_j:
                    bias[i, j] += 1.0

    return mx.array(bias)


class LargeAttention(nn.Module):
    """Scaled-up attention with Sudoku bias."""

    def __init__(self, config: LargeTRMConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        if config.use_attention_bias:
            self.sudoku_bias = create_sudoku_bias()
            self.bias_scale = mx.array([1.0])

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        if self.config.use_attention_bias:
            attn = attn + self.sudoku_bias * self.bias_scale

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        return self.o_proj(out)


class LargeBlock(nn.Module):
    """Large transformer block."""

    def __init__(self, config: LargeTRMConfig):
        super().__init__()
        self.attention = LargeAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LargeTRM(nn.Module):
    """
    Large capacity TRM with attention bias.

    Tests whether increased model size breaks through the 60% ceiling.
    """

    def __init__(self, config: Optional[LargeTRMConfig] = None):
        super().__init__()
        self.config = config or LargeTRMConfig()

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        self.blocks = [LargeBlock(self.config) for _ in range(self.config.num_layers)]
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


# Even larger variant
@dataclass
class XLargeTRMConfig:
    """Extra large config."""
    vocab_size: int = 10
    hidden_size: int = 512  # 8x larger than base
    num_heads: int = 16
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 81
    use_attention_bias: bool = True


class XLargeTRM(LargeTRM):
    """Extra large TRM."""

    def __init__(self, config: Optional[XLargeTRMConfig] = None):
        # Convert XLarge config to Large config format
        xl_config = config or XLargeTRMConfig()
        large_config = LargeTRMConfig(
            vocab_size=xl_config.vocab_size,
            hidden_size=xl_config.hidden_size,
            num_heads=xl_config.num_heads,
            num_layers=xl_config.num_layers,
            dropout=xl_config.dropout,
            max_seq_len=xl_config.max_seq_len,
            use_attention_bias=xl_config.use_attention_bias,
        )
        super().__init__(large_config)


def create_model(config: Optional[LargeTRMConfig] = None) -> LargeTRM:
    """Factory function."""
    return LargeTRM(config)


def create_xlarge_model(config: Optional[XLargeTRMConfig] = None) -> XLargeTRM:
    """Factory function for XLarge."""
    return XLargeTRM(config)


if __name__ == "__main__":
    print("Testing Large TRM...")
    config = LargeTRMConfig()
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")
    print(f"  hidden_size={config.hidden_size}, num_layers={config.num_layers}, num_heads={config.num_heads}")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Parameters: {num_params:,}")

    print("\nTesting XLarge TRM...")
    xl_config = XLargeTRMConfig()
    xl_model = create_xlarge_model(xl_config)

    xl_logits = xl_model(batch)
    print(f"  Output: {xl_logits.shape}")
    print(f"  hidden_size={xl_config.hidden_size}, num_layers={xl_config.num_layers}")

    xl_params = sum(p.size for p in xl_model.parameters().values())
    print(f"  Parameters: {xl_params:,}")

    # Compare to base
    base_params = 64 * 64 * 4 + 64 * 10 * 2  # Rough estimate
    print(f"\n  Large is ~{num_params / 50000:.1f}x base model")
    print(f"  XLarge is ~{xl_params / 50000:.1f}x base model")
