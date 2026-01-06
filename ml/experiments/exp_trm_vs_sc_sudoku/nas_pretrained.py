#!/usr/bin/env python3
"""
NAS with Pre-training: Initialize architecture weights with known-good constraints.

Hypothesis: Pure NAS struggles because the search space is too large and starts uniform.
Pre-training with known constraints (row/col/box) gives the model a warm start.

Key changes from nas_sc_trm.py:
1. Initialize arch weights to favor row/col/box (not uniform)
2. Allow search to discover additional useful constraints
3. Use curriculum: fix base constraints early, then search for additions
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import math


@dataclass
class NASPretrainedConfig:
    """Config for pre-trained NAS."""
    vocab_size: int = 10
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 81

    # NAS settings
    num_constraint_ops: int = 8  # row, col, box + 5 searchable
    num_base_constraints: int = 3  # row, col, box (always on)
    arch_temp: float = 1.0  # Softmax temperature for arch weights

    # Pre-training initialization
    base_constraint_weight: float = 2.0  # Initial weight for row/col/box
    search_constraint_weight: float = 0.0  # Initial weight for searchable ops


def make_row_mask() -> np.ndarray:
    mask = np.zeros((81, 81))
    for row in range(9):
        for i in range(9):
            for j in range(9):
                if i != j:
                    mask[row*9 + i, row*9 + j] = 1.0
    return mask


def make_col_mask() -> np.ndarray:
    mask = np.zeros((81, 81))
    for col in range(9):
        for i in range(9):
            for j in range(9):
                if i != j:
                    mask[i*9 + col, j*9 + col] = 1.0
    return mask


def make_box_mask() -> np.ndarray:
    mask = np.zeros((81, 81))
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for r in range(3):
                for c in range(3):
                    cells.append((box_row*3 + r) * 9 + (box_col*3 + c))
            for i in cells:
                for j in cells:
                    if i != j:
                        mask[i, j] = 1.0
    return mask


def make_diagonal_mask() -> np.ndarray:
    """Diagonal constraints (like in diagonal Sudoku variant)."""
    mask = np.zeros((81, 81))
    # Main diagonal
    main_diag = [i*9 + i for i in range(9)]
    for i in main_diag:
        for j in main_diag:
            if i != j:
                mask[i, j] = 1.0
    # Anti-diagonal
    anti_diag = [i*9 + (8-i) for i in range(9)]
    for i in anti_diag:
        for j in anti_diag:
            if i != j:
                mask[i, j] = 1.0
    return mask


def make_knight_mask() -> np.ndarray:
    """Knight's move constraint (anti-knight Sudoku)."""
    mask = np.zeros((81, 81))
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
    for i in range(81):
        row, col = i // 9, i % 9
        for dr, dc in knight_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                j = nr * 9 + nc
                mask[i, j] = 1.0
    return mask


def make_king_mask() -> np.ndarray:
    """King's move constraint (anti-king Sudoku)."""
    mask = np.zeros((81, 81))
    king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(81):
        row, col = i // 9, i % 9
        for dr, dc in king_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                j = nr * 9 + nc
                mask[i, j] = 1.0
    return mask


def make_orthogonal_mask() -> np.ndarray:
    """Orthogonal neighbor constraint."""
    mask = np.zeros((81, 81))
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(81):
        row, col = i // 9, i % 9
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                j = nr * 9 + nc
                mask[i, j] = 1.0
    return mask


def make_global_mask() -> np.ndarray:
    """Global attention (attend to all)."""
    mask = np.ones((81, 81))
    np.fill_diagonal(mask, 0)  # No self for constraint
    return mask


class ConstraintSupernet(nn.Module):
    """Supernet with pre-initialized architecture weights."""

    def __init__(self, config: NASPretrainedConfig):
        super().__init__()
        self.config = config

        # Create all constraint masks
        self.constraint_masks = mx.array(np.stack([
            make_row_mask(),      # 0 - base
            make_col_mask(),      # 1 - base
            make_box_mask(),      # 2 - base
            make_diagonal_mask(), # 3 - searchable
            make_knight_mask(),   # 4 - searchable
            make_king_mask(),     # 5 - searchable
            make_orthogonal_mask(), # 6 - searchable
            make_global_mask(),   # 7 - searchable
        ]))  # [8, 81, 81]

        # PRE-INITIALIZED architecture weights
        # Row/col/box start high, others start low
        init_weights = np.array([
            config.base_constraint_weight,    # row
            config.base_constraint_weight,    # col
            config.base_constraint_weight,    # box
            config.search_constraint_weight,  # diagonal
            config.search_constraint_weight,  # knight
            config.search_constraint_weight,  # king
            config.search_constraint_weight,  # orthogonal
            config.search_constraint_weight,  # global
        ])
        self.arch_weights = mx.array(init_weights)

        # Per-constraint learnable scales
        self.constraint_scales = mx.ones((config.num_constraint_ops,))

    def get_arch_probs(self) -> mx.array:
        """Get architecture probabilities via softmax."""
        return mx.softmax(self.arch_weights / self.config.arch_temp)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply weighted constraint attention bias.

        Args:
            x: [B, L, D] input

        Returns:
            bias: [L, L] attention bias
        """
        probs = self.get_arch_probs()  # [8]
        scales = self.constraint_scales  # [8]

        # Weighted sum of constraint masks
        weighted_masks = self.constraint_masks * (probs * scales).reshape(-1, 1, 1)
        combined_mask = mx.sum(weighted_masks, axis=0)  # [81, 81]

        return combined_mask

    def get_discovered_structure(self) -> dict:
        """Return the discovered architecture."""
        probs = self.get_arch_probs()
        names = ['row', 'col', 'box', 'diagonal', 'knight', 'king', 'orthogonal', 'global']
        return {name: float(prob) for name, prob in zip(names, probs.tolist())}


class PretrainedNASAttention(nn.Module):
    """Attention with pre-trained NAS constraint bias."""

    def __init__(self, config: NASPretrainedConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.constraint_supernet = ConstraintSupernet(config)
        self.bias_scale = mx.array([1.0])

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Add learned constraint bias
        constraint_bias = self.constraint_supernet(x) * self.bias_scale
        attn = attn + constraint_bias

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        return self.o_proj(out)


class PretrainedNASBlock(nn.Module):
    """Transformer block with pre-trained NAS."""

    def __init__(self, config: NASPretrainedConfig):
        super().__init__()
        self.attention = PretrainedNASAttention(config)
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


class NASPretrainedTRM(nn.Module):
    """
    NAS with pre-trained constraint weights.

    Starts with row/col/box highly weighted, then searches for
    additional useful constraints.
    """

    def __init__(self, config: Optional[NASPretrainedConfig] = None):
        super().__init__()
        self.config = config or NASPretrainedConfig()

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        self.blocks = [PretrainedNASBlock(self.config) for _ in range(self.config.num_layers)]
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

    def get_discovered_structure(self) -> dict:
        """Get the discovered architecture from all blocks."""
        structures = []
        for i, block in enumerate(self.blocks):
            struct = block.attention.constraint_supernet.get_discovered_structure()
            structures.append({f"layer_{i}_{k}": v for k, v in struct.items()})

        # Aggregate
        combined = {}
        for struct in structures:
            for k, v in struct.items():
                combined[k] = v
        return combined


def create_model(config: Optional[NASPretrainedConfig] = None) -> NASPretrainedTRM:
    """Factory function."""
    return NASPretrainedTRM(config)


if __name__ == "__main__":
    print("Testing NAS Pre-trained TRM...")
    config = NASPretrainedConfig(
        base_constraint_weight=2.0,
        search_constraint_weight=0.0
    )
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")

    # Show initial architecture weights
    supernet = model.blocks[0].attention.constraint_supernet
    print("\n  Initial architecture weights:")
    for name, prob in supernet.get_discovered_structure().items():
        print(f"    {name}: {prob:.3f}")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"\n  Parameters: {num_params:,}")
