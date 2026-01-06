"""
Hybrid Models: Combining best techniques from top performers.

Based on 100-epoch benchmark results:
1. attention (79.5%) - attention bias for constraints
2. hybrid (51.4%) - SC-TRM hybrid with constraint encoding
3. faithful (45.4%) - gradient truncation

This module creates new hybrids combining these techniques.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention_bias import (
    AttentionBiasConfig,
    AttentionBiasTRM,
    ConstraintBiasedAttention,
    TransformerBlockWithBias,
    build_constraint_affinity_matrix,
)
from .faithful_trm_v3 import FaithfulTRMv3Config, rms_norm, SwiGLU
from .hierarchical_sc_trm import HierarchicalConfig, UnitEncoder, BoardEncoder, TopDownGuidance


# =============================================================================
# HYBRID 1: Attention Bias + Gradient Truncation
# =============================================================================

@dataclass
class AttentionBiasGradTruncConfig:
    """Attention bias model with gradient truncation."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1
    initial_bias_scale: float = 0.5
    final_bias_scale: float = 3.0
    init_std: float = 1.0


class AttentionBiasGradTrunc(nn.Module):
    """
    Combines attention_bias (79.5%) with gradient truncation from faithful.

    Key insight: Attention bias provides structural inductive bias,
    gradient truncation provides memory efficiency and prevents overfitting.
    """

    def __init__(self, config: AttentionBiasGradTruncConfig):
        super().__init__()
        self.config = config

        # Constraint affinity matrix
        self.constraint_affinity = build_constraint_affinity_matrix()

        # Embeddings
        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # H-level context
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

        # Output
        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def encode(self, puzzle: mx.array) -> mx.array:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))
        cell_emb = self.cell_embed(puzzle.astype(mx.int32))
        pos_emb = self.pos_embed(positions)
        return cell_emb + pos_emb

    def refine(self, h: mx.array, h_context: mx.array, bias_strength: float) -> mx.array:
        h = h + h_context[:, None, :]
        for layer in self.layers:
            h = layer(h, bias_strength=bias_strength)
        return h

    def predict(self, h: mx.array) -> mx.array:
        h = self.ln_out(h)
        return self.output_head(h)

    def solve(self, puzzle: mx.array, truncate_grad: bool = False, **kwargs) -> Dict[str, mx.array]:
        H = self.config.H_cycles
        L = self.config.L_cycles

        # Bias annealing schedule
        bias_schedule = [
            self.config.initial_bias_scale +
            (self.config.final_bias_scale - self.config.initial_bias_scale) * hi / (H - 1)
            if H > 1 else self.config.final_bias_scale
            for hi in range(H)
        ]

        h = self.encode(puzzle)

        # H-1 cycles WITHOUT gradient (if truncate_grad)
        if truncate_grad and H > 1:
            for hi in range(H - 1):
                bias_strength = bias_schedule[hi]
                pooled = mx.mean(h, axis=1)
                h_context = self.h_context_net(pooled)
                for _ in range(L):
                    h = self.refine(h, h_context, bias_strength)
            h = mx.stop_gradient(h)

        # Final H cycle WITH gradient
        hi = H - 1 if truncate_grad else 0
        for current_h in range(hi, H):
            bias_strength = bias_schedule[current_h]
            pooled = mx.mean(h, axis=1)
            h_context = self.h_context_net(pooled)
            for _ in range(L):
                h = self.refine(h, h_context, bias_strength)

        logits = self.predict(h)
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, truncate_grad: bool = True, **kwargs) -> Tuple[mx.array, Dict]:
        result = self.solve(puzzle, truncate_grad=truncate_grad)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        B, C, D = logits.shape

        logits_flat = logits.reshape(-1, D)
        targets_flat = targets.reshape(-1)

        logits_flat = mx.clip(logits_flat, -30, 30)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

        batch_indices = mx.arange(logits_flat.shape[0])
        correct_log_probs = log_probs[batch_indices, targets_flat]
        loss = -mx.mean(correct_log_probs)

        predictions = result['predictions']
        cell_accuracy = mx.mean((predictions == solution).astype(mx.float32))

        return loss, {'cell_accuracy': cell_accuracy, 'accuracy': cell_accuracy}


# =============================================================================
# HYBRID 2: Hierarchical + Attention Bias
# =============================================================================

@dataclass
class HierarchicalAttentionConfig:
    """Hierarchical structure with attention bias at each level."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1
    unit_hidden_dim: int = 64
    board_hidden_dim: int = 32
    initial_bias_scale: float = 0.5
    final_bias_scale: float = 3.0


class HierarchicalAttention(nn.Module):
    """
    Combines hierarchical SC (59.8% in old benchmark) with attention bias.

    Uses 3-level hierarchy with attention bias at cell level.
    """

    def __init__(self, config: HierarchicalAttentionConfig):
        super().__init__()
        self.config = config

        # Constraint affinity
        self.constraint_affinity = build_constraint_affinity_matrix()

        # Embeddings
        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # Cell-level with constraint-biased attention
        self.cell_layers = [
            TransformerBlockWithBias(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
                self.constraint_affinity,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ]

        # Hierarchy encoders
        self.unit_encoder = UnitEncoder(config.hidden_dim, config.unit_hidden_dim)
        self.board_encoder = BoardEncoder(config.unit_hidden_dim, config.board_hidden_dim)
        self.guidance = TopDownGuidance(config.board_hidden_dim, config.unit_hidden_dim, config.hidden_dim)

        # H-context
        self.h_context_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Output
        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def encode(self, puzzle: mx.array) -> mx.array:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))
        return self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

    def refine_with_hierarchy(self, h: mx.array, h_context: mx.array,
                               bias_strength: float, guidance_strength: float) -> mx.array:
        # Add H-context
        h = h + h_context[:, None, :]

        # Cell-level with attention bias
        for layer in self.cell_layers:
            h = layer(h, bias_strength=bias_strength)

        # Hierarchy: cells -> units -> board -> cells
        unit_states = self.unit_encoder(h)
        board_state = self.board_encoder(unit_states)
        h = self.guidance(h, unit_states, board_state, strength=guidance_strength)

        return h

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        H = self.config.H_cycles
        L = self.config.L_cycles

        # Schedules
        bias_schedule = [
            self.config.initial_bias_scale +
            (self.config.final_bias_scale - self.config.initial_bias_scale) * hi / (H - 1)
            if H > 1 else self.config.final_bias_scale
            for hi in range(H)
        ]
        guidance_schedule = [0.5 + 0.5 * hi / (H - 1) if H > 1 else 1.0 for hi in range(H)]

        h = self.encode(puzzle)

        for hi in range(H):
            pooled = mx.mean(h, axis=1)
            h_context = self.h_context_net(pooled)
            for _ in range(L):
                h = self.refine_with_hierarchy(h, h_context, bias_schedule[hi], guidance_schedule[hi])

        logits = self.ln_out(h)
        logits = self.output_head(logits)
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs) -> Tuple[mx.array, Dict]:
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy, 'accuracy': cell_accuracy}


# =============================================================================
# HYBRID 3: Faithful v3 + Attention (replace MLP-T)
# =============================================================================

@dataclass
class FaithfulAttentionConfig:
    """Faithful TRM with attention instead of MLP-T."""
    vocab_size: int = 10
    hidden_size: int = 128
    num_heads: int = 4
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 81
    init_std: float = 1.0
    use_constraint_bias: bool = True
    initial_bias_scale: float = 0.5
    final_bias_scale: float = 3.0


class FaithfulAttentionBlock(nn.Module):
    """Reasoning block with constraint-biased attention instead of MLP-T."""

    def __init__(self, config: FaithfulAttentionConfig, constraint_affinity: mx.array):
        super().__init__()
        self.config = config

        # Constraint-biased attention
        self.attn = ConstraintBiasedAttention(
            config.hidden_size, config.num_heads, constraint_affinity,
            learnable_scale=True, initial_scale=config.initial_bias_scale
        )

        # FFN (SwiGLU style)
        inter = ((int(2.0 * config.hidden_size * 2 / 3) + 255) // 256) * 256
        if inter == 0:
            inter = max(64, int(2.0 * config.hidden_size))
        self.gate_up = nn.Linear(config.hidden_size, inter * 2, bias=False)
        self.down = nn.Linear(inter, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, bias_strength: float = 1.0) -> mx.array:
        # Attention with constraint bias
        h = rms_norm(x, self.config.rms_norm_eps)
        h = self.attn(h, bias_strength=bias_strength)
        x = x + h

        # FFN
        h = rms_norm(x, self.config.rms_norm_eps)
        gate_up = self.gate_up(h)
        gate, up = mx.split(gate_up, 2, axis=-1)
        h = self.down(nn.silu(gate) * up)
        x = x + h

        return x


class FaithfulAttention(nn.Module):
    """
    Faithful TRM v3 with attention instead of MLP-T.

    Keeps gradient truncation but uses constraint-biased attention.
    """

    def __init__(self, config: FaithfulAttentionConfig):
        super().__init__()
        self.config = config

        # Constraint affinity
        self.constraint_affinity = build_constraint_affinity_matrix()

        # Embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_pos = nn.Embedding(config.max_seq_len, config.hidden_size)

        # Reasoning layers with attention
        self.layers = [
            FaithfulAttentionBlock(config, self.constraint_affinity)
            for _ in range(config.L_layers)
        ]

        # Initial states
        self.H_init = mx.clip(
            mx.random.normal(shape=(config.hidden_size,)) * config.init_std,
            -2 * config.init_std, 2 * config.init_std
        )
        self.L_init = mx.clip(
            mx.random.normal(shape=(config.hidden_size,)) * config.init_std,
            -2 * config.init_std, 2 * config.init_std
        )

        # Output
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def L_level(self, z: mx.array, injection: mx.array, bias_strength: float) -> mx.array:
        x = z + injection
        for layer in self.layers:
            x = layer(x, bias_strength=bias_strength)
        return x

    def __call__(self, input_ids: mx.array, truncate_grad: bool = True) -> mx.array:
        B, L = input_ids.shape

        # Embeddings
        positions = mx.arange(L)
        input_embed = self.embed_tokens(input_ids) + self.embed_pos(positions)
        input_embed = self.embed_scale * input_embed

        # Initialize states
        z_H = mx.broadcast_to(self.H_init, (B, L, self.config.hidden_size))
        z_L = mx.broadcast_to(self.L_init, (B, L, self.config.hidden_size))

        H = self.config.H_cycles

        # Bias schedule
        bias_schedule = [
            self.config.initial_bias_scale +
            (self.config.final_bias_scale - self.config.initial_bias_scale) * hi / (H - 1)
            if H > 1 else self.config.final_bias_scale
            for hi in range(H)
        ]

        # H-1 cycles WITHOUT gradient
        if truncate_grad and H > 1:
            for hi in range(H - 1):
                bias = bias_schedule[hi]
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embed, bias)
                z_H = self.L_level(z_H, z_L, bias)
            z_H = mx.stop_gradient(z_H)
            z_L = mx.stop_gradient(z_L)

        # Final cycle WITH gradient
        bias = bias_schedule[-1]
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embed, bias)
        z_H = self.L_level(z_H, z_L, bias)

        return self.lm_head(z_H)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        logits = self(puzzle, truncate_grad=False)
        predictions = mx.argmax(logits, axis=-1)
        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)
        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, truncate_grad: bool = True, **kwargs):
        logits = self(puzzle, truncate_grad=truncate_grad)
        target = solution

        empty_mask = (puzzle == 0).astype(mx.float32)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

        target_expanded = target[:, :, None]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)

        loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == target).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        return loss, {'cell_accuracy': cell_acc}


# =============================================================================
# HYBRID 4: Deep Attention (more layers, more heads)
# =============================================================================

@dataclass
class DeepAttentionConfig:
    """Deeper attention model with more capacity."""
    hidden_dim: int = 192  # Larger
    num_heads: int = 6     # More heads
    num_layers: int = 4    # More layers
    ff_dim: int = 384
    H_cycles: int = 4
    L_cycles: int = 4
    dropout: float = 0.1
    initial_bias_scale: float = 0.5
    final_bias_scale: float = 4.0


class DeepAttention(nn.Module):
    """
    Deeper version of attention bias model with more capacity.
    """

    def __init__(self, config: DeepAttentionConfig):
        super().__init__()
        self.config = config

        self.constraint_affinity = build_constraint_affinity_matrix()

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        self.h_context_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.layers = [
            TransformerBlockWithBias(
                config.hidden_dim, config.num_heads, config.ff_dim,
                self.constraint_affinity, config.dropout,
            )
            for _ in range(config.num_layers)
        ]

        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, truncate_grad: bool = False, **kwargs) -> Dict[str, mx.array]:
        H, L = self.config.H_cycles, self.config.L_cycles
        B = puzzle.shape[0]

        bias_schedule = [
            self.config.initial_bias_scale +
            (self.config.final_bias_scale - self.config.initial_bias_scale) * hi / (H - 1)
            if H > 1 else self.config.final_bias_scale
            for hi in range(H)
        ]

        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))
        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        if truncate_grad and H > 1:
            for hi in range(H - 1):
                pooled = mx.mean(h, axis=1)
                h_ctx = self.h_context_net(pooled)
                h = h + h_ctx[:, None, :]
                for _ in range(L):
                    for layer in self.layers:
                        h = layer(h, bias_strength=bias_schedule[hi])
            h = mx.stop_gradient(h)

        for hi in range(H - 1 if truncate_grad else 0, H):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context_net(pooled)
            h = h + h_ctx[:, None, :]
            for _ in range(L):
                for layer in self.layers:
                    h = layer(h, bias_strength=bias_schedule[hi])

        logits = self.output_head(self.ln_out(h))
        return {'logits': logits, 'predictions': mx.argmax(logits, axis=-1) + 1}

    def loss(self, puzzle: mx.array, solution: mx.array, truncate_grad: bool = True, **kwargs):
        result = self.solve(puzzle, truncate_grad=truncate_grad)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy, 'accuracy': cell_accuracy}


# Factory function
def create_hybrid_model(model_type: str):
    """Create a hybrid model by type."""
    if model_type == 'attn_grad_trunc':
        return AttentionBiasGradTrunc(AttentionBiasGradTruncConfig())
    elif model_type == 'hier_attn':
        return HierarchicalAttention(HierarchicalAttentionConfig())
    elif model_type == 'faithful_attn':
        return FaithfulAttention(FaithfulAttentionConfig())
    elif model_type == 'deep_attn':
        return DeepAttention(DeepAttentionConfig())
    else:
        raise ValueError(f"Unknown hybrid model type: {model_type}")


if __name__ == "__main__":
    print("Testing hybrid models...")

    for name in ['attn_grad_trunc', 'hier_attn', 'faithful_attn', 'deep_attn']:
        print(f"\n=== {name} ===")
        model = create_hybrid_model(name)
        batch = mx.zeros((2, 81), dtype=mx.int32)
        solution = mx.ones((2, 81), dtype=mx.int32)

        result = model.solve(batch)
        print(f"  Output shape: {result['predictions'].shape}")

        loss, metrics = model.loss(batch, solution)
        print(f"  Loss: {float(loss):.4f}")
        print(f"  Cell accuracy: {float(metrics.get('cell_accuracy', 0)):.4f}")
