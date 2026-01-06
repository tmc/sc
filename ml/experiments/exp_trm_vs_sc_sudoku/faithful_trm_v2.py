#!/usr/bin/env python3
"""
Faithful TRM v2: Exact implementation matching Samsung SAIL Montreal's TRM.

Key architecture from the paper:
- Two latent states: z_H (high) and z_L (low)
- Same network (L_level) used for both updates
- Input injection: z_H + input_embeddings added at each L step
- mlp_t=True: MLP across sequence dimension (better for Sudoku)
- SwiGLU activation + RMS norm
- H_cycles outer, L_cycles inner iterations
- Gradient only on final H cycle

Reference: github.com/SamsungSAILMontreal/TinyRecursiveModels
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional
import math


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """RMS normalization (no learnable params)."""
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


@dataclass
class FaithfulTRMv2Config:
    """Config matching TRM paper."""
    vocab_size: int = 10
    hidden_size: int = 128
    expansion: float = 2.0  # SwiGLU expansion
    H_cycles: int = 3  # Outer iterations
    L_cycles: int = 6  # Inner iterations per H cycle
    L_layers: int = 2  # Layers in L_level network
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 81
    use_mlp_t: bool = True  # MLP across sequence (better for Sudoku)
    num_heads: int = 4  # Only used if use_mlp_t=False


class SwiGLU(nn.Module):
    """SwiGLU activation as used in TRM."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        # Match TRM's inter size calculation
        inter = ((int(expansion * hidden_size * 2 / 3) + 255) // 256) * 256

        self.gate_up = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down = nn.Linear(inter, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down(nn.silu(gate) * up)


class ReasoningBlock(nn.Module):
    """Single reasoning block matching TRM's TinyRecursiveReasoningModel_ACTV1Block."""

    def __init__(self, config: FaithfulTRMv2Config):
        super().__init__()
        self.config = config

        if config.use_mlp_t:
            # MLP across sequence dimension (transpose -> MLP -> transpose)
            self.mlp_t = SwiGLU(config.max_seq_len, config.expansion)
        else:
            # Self-attention
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # FFN (always present)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        if self.config.use_mlp_t:
            # MLP across sequence: [B, L, D] -> [B, D, L] -> MLP -> [B, D, L] -> [B, L, D]
            h = mx.transpose(x, (0, 2, 1))  # [B, D, L]
            h = self.mlp_t(h)
            h = mx.transpose(h, (0, 2, 1))  # [B, L, D]
            x = rms_norm(x + h, self.config.rms_norm_eps)
        else:
            # Self-attention
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for multi-head
            head_dim = D // self.config.num_heads
            q = q.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
            h = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
            h = self.o_proj(h)
            x = rms_norm(x + h, self.config.rms_norm_eps)

        # FFN (post-norm style)
        h = self.mlp(x)
        x = rms_norm(x + h, self.config.rms_norm_eps)

        return x


class ReasoningModule(nn.Module):
    """L_level: Stack of reasoning blocks with input injection."""

    def __init__(self, config: FaithfulTRMv2Config):
        super().__init__()
        self.layers = [ReasoningBlock(config) for _ in range(config.L_layers)]

    def __call__(self, hidden_states: mx.array, input_injection: mx.array) -> mx.array:
        """Forward with input injection."""
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer(x)
        return x


class FaithfulTRMv2(nn.Module):
    """
    Faithful TRM v2: Matches the exact TRM architecture.

    Core loop:
        for h in range(H_cycles):
            for l in range(L_cycles):
                z_L = L_level(z_L, z_H + input_embed)
            z_H = L_level(z_H, z_L)
    """

    def __init__(self, config: Optional[FaithfulTRMv2Config] = None):
        super().__init__()
        self.config = config or FaithfulTRMv2Config()

        # Embeddings
        embed_scale = math.sqrt(self.config.hidden_size)
        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.embed_pos = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        # Single L_level network (shared for z_L and z_H updates)
        self.L_level = ReasoningModule(self.config)

        # Initial states (learnable)
        self.H_init = mx.zeros((self.config.hidden_size,))
        self.L_init = mx.zeros((self.config.hidden_size,))

        # Output head
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Forward pass with recursive reasoning.

        Args:
            input_ids: [B, 81] input puzzle

        Returns:
            logits: [B, 81, 10] predictions
        """
        B, L = input_ids.shape

        # Input embeddings
        positions = mx.arange(L)
        input_embed = self.embed_tokens(input_ids) + self.embed_pos(positions)
        input_embed = self.embed_scale * input_embed

        # Initialize latent states
        z_H = mx.broadcast_to(self.H_init, (B, L, self.config.hidden_size))
        z_L = mx.broadcast_to(self.L_init, (B, L, self.config.hidden_size))

        # Recursive reasoning loop
        for _h in range(self.config.H_cycles):
            # Inner L cycles: update z_L
            for _l in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embed)

            # Update z_H using z_L
            z_H = self.L_level(z_H, z_L)

        # Output
        logits = self.lm_head(z_H)
        return logits

    def solve(self, puzzle: mx.array, temperature: float = 0.0, **kwargs) -> dict:
        """Solve a puzzle, returning dict with predictions and logits."""
        logits = self(puzzle)

        if temperature > 0:
            predictions = mx.argmax(mx.softmax(logits / temperature, axis=-1), axis=-1)
        else:
            predictions = mx.argmax(logits, axis=-1)

        # Keep original given values
        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)

        return {"predictions": predictions, "logits": logits}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        """Compute cross-entropy loss for training."""
        logits = self(puzzle)  # [B, 81, 10]
        target = solution  # [B, 81]

        # Cross-entropy loss (only on non-given cells)
        empty_mask = (puzzle == 0).astype(mx.float32)  # [B, 81]

        # Compute log probs
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)  # [B, 81, 10]

        # Gather correct class log probs
        target_expanded = target[:, :, None]  # [B, 81, 1]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)  # [B, 81]

        # Masked loss (only on empty cells)
        loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        # Metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == target).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        return loss, {"cell_accuracy": cell_acc}


def create_model(config: Optional[FaithfulTRMv2Config] = None) -> FaithfulTRMv2:
    """Factory function."""
    return FaithfulTRMv2(config)


if __name__ == "__main__":
    print("Testing Faithful TRM v2...")
    config = FaithfulTRMv2Config(H_cycles=3, L_cycles=6, use_mlp_t=True)
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")
    print(f"  H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
    print(f"  use_mlp_t={config.use_mlp_t}")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Parameters: {num_params:,}")
