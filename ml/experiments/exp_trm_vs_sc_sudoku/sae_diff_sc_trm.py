#!/usr/bin/env python3
"""
SAE + Differentiable Statechart TRM: Hybrid approach combining sparse feature
discovery with learnable topology.

Key insight: Cold start is the main problem with NAS and differentiable topology.
This model:
1. Uses SAE to discover constraint patterns from hidden states (warm start)
2. Initializes SoftTopology from SAE-discovered features
3. Fine-tunes with Gumbel-Softmax gradient descent

Architecture:
- TopK SAE: Discovers K sparse constraint patterns from transformer hidden states
- Soft Topology: Learnable [81, 81] adjacency matrix initialized from SAE
- Gumbel-Softmax: Enables gradient flow through discrete topology decisions
- Standard transformer backbone with topology-biased attention

This combines:
- exp_differentiable_statecharts: Gumbel-Softmax, SoftTopology
- exp_grammar_induction: TopK SAE for sparse feature discovery
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class SAEDiffSCConfig:
    """Config for SAE + Differentiable SC TRM."""
    vocab_size: int = 10
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 81

    # SAE settings
    sae_latent_dim: int = 32  # Number of sparse features to discover
    sae_k: int = 8  # Top-K sparsity
    sae_warmup_epochs: int = 10  # Train SAE alone first

    # Soft topology settings
    topology_temp: float = 1.0  # Gumbel-Softmax temperature
    topology_sparsity: float = 0.1  # L1 regularization strength
    topology_init_from_sae: bool = True  # Initialize from SAE features

    # Training
    dropout: float = 0.1


def gumbel_softmax(logits: mx.array, temperature: float = 1.0, hard: bool = False) -> mx.array:
    """
    Gumbel-Softmax for differentiable discrete sampling.

    Args:
        logits: Unnormalized log probabilities
        temperature: Controls discreteness (lower = more discrete)
        hard: If True, use straight-through estimator

    Returns:
        Soft (or hard) sample from categorical distribution
    """
    # Sample Gumbel noise
    u = mx.random.uniform(shape=logits.shape, low=1e-10, high=1.0)
    gumbel = -mx.log(-mx.log(u))

    # Add noise and apply temperature
    y = mx.softmax((logits + gumbel) / temperature, axis=-1)

    if hard:
        # Straight-through estimator: hard forward, soft backward
        y_hard = mx.zeros_like(y)
        indices = mx.argmax(y, axis=-1, keepdims=True)
        # One-hot encoding via scatter would be ideal, but we approximate
        y_hard = mx.where(
            mx.arange(y.shape[-1]) == indices,
            mx.ones_like(y),
            mx.zeros_like(y)
        )
        y = mx.stop_gradient(y_hard - y) + y

    return y


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder for discovering constraint patterns.

    Learns to represent hidden states using K sparse features.
    These features often correspond to meaningful constraint patterns.
    """

    def __init__(self, config: SAEDiffSCConfig):
        super().__init__()
        self.config = config
        self.k = config.sae_k

        # Encoder: hidden -> latent
        self.encoder = nn.Linear(config.hidden_size, config.sae_latent_dim)

        # Decoder: latent -> hidden
        self.decoder = nn.Linear(config.sae_latent_dim, config.hidden_size)

        # Feature-to-topology projection (81 cells)
        self.feature_to_topology = nn.Linear(config.sae_latent_dim, config.max_seq_len)

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Encode with Top-K sparsity.

        Args:
            x: [B, L, D] hidden states

        Returns:
            sparse_codes: [B, L, sae_latent_dim] with only K non-zero
            indices: [B, L, K] indices of top-K features
        """
        # Get pre-activation
        pre_act = self.encoder(x)  # [B, L, latent_dim]

        # Find top-K activations
        abs_acts = mx.abs(pre_act)

        # Sort descending to find threshold
        sorted_acts = mx.sort(abs_acts, axis=-1)  # Ascending order
        latent_dim = pre_act.shape[-1]
        k = min(self.k, latent_dim)

        # Get k-th largest value as threshold (index from end for ascending sort)
        threshold_idx = max(latent_dim - k, 0)
        threshold = sorted_acts[:, :, threshold_idx:threshold_idx+1]  # [B, L, 1]

        # Zero out below threshold
        mask = abs_acts >= threshold
        sparse_codes = pre_act * mask.astype(mx.float32)

        # Get indices of active features (top-K indices)
        indices = mx.argsort(abs_acts, axis=-1)[:, :, -k:]

        return sparse_codes, indices

    def decode(self, sparse_codes: mx.array) -> mx.array:
        """Decode sparse codes back to hidden space."""
        return self.decoder(sparse_codes)

    def get_feature_topology(self) -> mx.array:
        """
        Convert SAE features to topology weights.

        Returns:
            topology: [81, 81] adjacency weights
        """
        # Get decoder weights and project to cell space
        # decoder.weight: [hidden_size, latent_dim]
        # feature_to_topology.weight: [81, latent_dim]

        # Each latent feature votes for which cells should attend to which
        # Compute outer product of feature projections
        feat_proj = self.feature_to_topology.weight  # [81, latent_dim]

        # Similarity between cell projections = topology
        topology = feat_proj @ feat_proj.T  # [81, 81]

        return topology

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass with reconstruction.

        Returns:
            reconstruction: [B, L, D]
            sparse_codes: [B, L, latent_dim]
            topology: [81, 81]
        """
        sparse_codes, _ = self.encode(x)
        reconstruction = self.decode(sparse_codes)
        topology = self.get_feature_topology()

        return reconstruction, sparse_codes, topology


class SoftTopology(nn.Module):
    """
    Learnable soft adjacency matrix with Gumbel-Softmax sampling.

    Can be initialized from SAE-discovered features or from known
    Sudoku constraints (row/col/box).
    """

    def __init__(self, config: SAEDiffSCConfig):
        super().__init__()
        self.config = config
        n = config.max_seq_len

        # Learnable logits for adjacency
        # Initialize to slight positive (encourage connectivity initially)
        self.adjacency_logits = mx.zeros((n, n))

        # Per-cell importance weights
        self.cell_weights = mx.ones((n,))

    def initialize_from_sudoku(self):
        """Initialize with known Sudoku constraints as warm start."""
        n = 81
        init = np.zeros((n, n))

        for i in range(n):
            row_i, col_i = i // 9, i % 9
            box_i = (row_i // 3) * 3 + (col_i // 3)

            for j in range(n):
                row_j, col_j = j // 9, j % 9
                box_j = (row_j // 3) * 3 + (col_j // 3)

                if i != j:
                    if row_i == row_j:
                        init[i, j] += 2.0  # Same row
                    if col_i == col_j:
                        init[i, j] += 2.0  # Same column
                    if box_i == box_j:
                        init[i, j] += 2.0  # Same box

        self.adjacency_logits = mx.array(init)

    def initialize_from_sae(self, sae_topology: mx.array):
        """Initialize from SAE-discovered topology."""
        # Normalize and scale
        normalized = sae_topology / (mx.max(mx.abs(sae_topology)) + 1e-8)
        self.adjacency_logits = normalized * 3.0  # Scale to reasonable logit range

    def get_soft_adjacency(self, temperature: float = 1.0) -> mx.array:
        """Get soft adjacency weights via sigmoid."""
        return mx.sigmoid(self.adjacency_logits / temperature)

    def get_hard_adjacency(self, temperature: float = 1.0) -> mx.array:
        """Get hard adjacency via Gumbel-Softmax."""
        # Convert to binary choice per edge
        # Stack logits for [no_edge, edge] choice
        logits = mx.stack([
            mx.zeros_like(self.adjacency_logits),
            self.adjacency_logits
        ], axis=-1)  # [81, 81, 2]

        # Gumbel-softmax sample
        samples = gumbel_softmax(logits, temperature, hard=True)

        # Return probability of edge
        return samples[:, :, 1]

    def __call__(self, temperature: float = 1.0, hard: bool = False) -> mx.array:
        """Get adjacency matrix."""
        if hard:
            return self.get_hard_adjacency(temperature)
        return self.get_soft_adjacency(temperature)

    def sparsity_loss(self) -> mx.array:
        """L1 regularization on adjacency weights."""
        soft_adj = self.get_soft_adjacency()
        return mx.mean(mx.abs(soft_adj))


class SAEDiffAttention(nn.Module):
    """Attention with SAE-initialized differentiable topology bias."""

    def __init__(self, config: SAEDiffSCConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Soft topology
        self.topology = SoftTopology(config)

        # Learnable bias scale
        self.bias_scale = mx.array([1.0])

    def __call__(self, x: mx.array, use_hard_topology: bool = False) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Add topology bias
        adj = self.topology(self.config.topology_temp, hard=use_hard_topology)
        attn = attn + adj * self.bias_scale

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        return self.o_proj(out)


class SAEDiffBlock(nn.Module):
    """Transformer block with SAE-initialized differentiable topology."""

    def __init__(self, config: SAEDiffSCConfig):
        super().__init__()
        self.attention = SAEDiffAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, use_hard_topology: bool = False) -> mx.array:
        x = x + self.dropout(self.attention(self.norm1(x), use_hard_topology))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SAEDiffSCTRM(nn.Module):
    """
    SAE + Differentiable Statechart TRM.

    Training phases:
    1. SAE warmup: Train SAE on hidden states to discover constraint patterns
    2. Topology init: Initialize soft topology from SAE features
    3. Joint training: Fine-tune everything with Gumbel-Softmax

    This avoids the cold start problem by using SAE as a warm start
    for the differentiable topology.
    """

    def __init__(self, config: Optional[SAEDiffSCConfig] = None):
        super().__init__()
        self.config = config or SAEDiffSCConfig()

        # Embeddings
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        # SAE for feature discovery
        self.sae = TopKSAE(self.config)

        # Transformer blocks with differentiable topology
        self.blocks = [SAEDiffBlock(self.config) for _ in range(self.config.num_layers)]

        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        # Track training phase
        self.sae_warmup_done = False
        self.topology_initialized = False

    def initialize_topology_from_sudoku(self):
        """Initialize all topology modules with Sudoku constraints."""
        for block in self.blocks:
            block.attention.topology.initialize_from_sudoku()
        self.topology_initialized = True

    def initialize_topology_from_sae(self, hidden_states: mx.array):
        """
        Initialize topology from SAE-discovered features.

        Args:
            hidden_states: [B, 81, hidden_size] sample of hidden states
        """
        # Get SAE topology
        _, _, sae_topology = self.sae(hidden_states)

        # Initialize all attention blocks
        for block in self.blocks:
            block.attention.topology.initialize_from_sae(sae_topology)

        self.topology_initialized = True

    def __call__(self, input_ids: mx.array, use_hard_topology: bool = False) -> mx.array:
        B, L = input_ids.shape
        positions = mx.arange(L)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x, use_hard_topology)

        x = self.norm(x)
        return self.head(x)

    def forward_with_sae(self, input_ids: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass that also returns SAE reconstruction for auxiliary loss.

        Returns:
            logits: [B, 81, 10]
            sae_recon: [B, 81, hidden_size]
            hidden_states: [B, 81, hidden_size]
        """
        B, L = input_ids.shape
        positions = mx.arange(L)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Store first block output for SAE
        x = self.blocks[0](x)
        hidden_states = x

        # SAE reconstruction
        sae_recon, _, _ = self.sae(hidden_states)

        # Continue through remaining blocks
        for block in self.blocks[1:]:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)

        return logits, sae_recon, hidden_states

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

    def loss(self, puzzle: mx.array, solution: mx.array,
             include_sae_loss: bool = True, include_sparsity_loss: bool = True, **kwargs):
        """
        Compute loss with optional auxiliary losses.

        Args:
            puzzle: [B, 81] input
            solution: [B, 81] target
            include_sae_loss: Add SAE reconstruction loss
            include_sparsity_loss: Add topology sparsity regularization
        """
        if include_sae_loss:
            logits, sae_recon, hidden_states = self.forward_with_sae(puzzle)
        else:
            logits = self(puzzle)

        target = solution
        empty_mask = (puzzle == 0).astype(mx.float32)

        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        target_expanded = target[:, :, None]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)
        ce_loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        total_loss = ce_loss

        # SAE reconstruction loss
        if include_sae_loss:
            sae_loss = mx.mean((sae_recon - hidden_states) ** 2)
            total_loss = total_loss + 0.1 * sae_loss

        # Topology sparsity loss
        if include_sparsity_loss:
            sparsity_loss = sum(
                block.attention.topology.sparsity_loss()
                for block in self.blocks
            ) / len(self.blocks)
            total_loss = total_loss + self.config.topology_sparsity * sparsity_loss

        # Metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == target).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        return total_loss, {"cell_accuracy": cell_acc}

    def get_discovered_topology(self) -> dict:
        """Return the discovered topology structure."""
        topologies = {}
        for i, block in enumerate(self.blocks):
            adj = block.attention.topology.get_soft_adjacency()
            topologies[f"layer_{i}"] = {
                "mean_weight": float(mx.mean(adj)),
                "max_weight": float(mx.max(adj)),
                "sparsity": float(mx.mean((adj < 0.5).astype(mx.float32))),
            }
        return topologies


# Variant: Pre-initialized with Sudoku constraints
class SAEDiffSCTRM_Sudoku(SAEDiffSCTRM):
    """Variant pre-initialized with Sudoku row/col/box constraints."""

    def __init__(self, config: Optional[SAEDiffSCConfig] = None):
        super().__init__(config)
        self.initialize_topology_from_sudoku()


def create_model(config: Optional[SAEDiffSCConfig] = None) -> SAEDiffSCTRM:
    """Factory function."""
    return SAEDiffSCTRM(config)


def create_sudoku_model(config: Optional[SAEDiffSCConfig] = None) -> SAEDiffSCTRM_Sudoku:
    """Factory function for Sudoku-initialized variant."""
    return SAEDiffSCTRM_Sudoku(config)


if __name__ == "__main__":
    print("Testing SAE + Differentiable SC TRM...")
    config = SAEDiffSCConfig()
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")
    print(f"  SAE latent dim: {config.sae_latent_dim}, K: {config.sae_k}")

    # Test with SAE loss
    solution = mx.ones((2, 81), dtype=mx.int32)
    loss, metrics = model.loss(batch, solution, include_sae_loss=True)
    print(f"  Loss: {float(loss):.4f}, Cell acc: {float(metrics['cell_accuracy']):.1%}")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Parameters: {num_params:,}")

    print("\nTesting Sudoku-initialized variant...")
    model_sudoku = create_sudoku_model(config)
    logits_sudoku = model_sudoku(batch)
    print(f"  Output: {logits_sudoku.shape}")
    print(f"  Topology initialized: {model_sudoku.topology_initialized}")

    # Show discovered topology
    topo = model_sudoku.get_discovered_topology()
    for layer, stats in topo.items():
        print(f"  {layer}: mean={stats['mean_weight']:.3f}, sparsity={stats['sparsity']:.1%}")
