#!/usr/bin/env python3
"""
Iterative TRM: Faithful implementation of recursive reasoning from the TRM paper.

Key insight from Samsung SAIL Montreal's TRM paper:
- z ← net(x, y, z)  # Update latent reasoning
- y ← net(y, z)     # Refine output answer
- T=3 outer recursions, n=6 inner iterations for Sudoku
- Only 2 layers works best (prevents overfitting)
- Halting mechanism learns when to stop

Reference: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class IterativeConfig:
    """Config for iterative refinement TRM."""
    vocab_size: int = 10  # 0-9 for Sudoku
    hidden_size: int = 128  # Latent dimension
    num_layers: int = 2  # Paper finds 2 optimal
    T: int = 3  # Outer recursion steps
    n: int = 6  # Inner latent updates per recursion
    dropout: float = 0.1
    max_seq_len: int = 81
    use_halting: bool = True  # Learn when to stop
    use_attention: bool = True  # Paper: MLP better for Sudoku, attention for ARC


class LatentUpdateNet(nn.Module):
    """Network for: z ← net(x, y, z)"""

    def __init__(self, config: IterativeConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size

        # Combine x, y, z
        self.combine = nn.Linear(hidden * 3, hidden)

        if config.use_attention:
            self.layers = [
                nn.MultiHeadAttention(hidden, num_heads=4)
                for _ in range(config.num_layers)
            ]
            self.norms = [nn.LayerNorm(hidden) for _ in range(config.num_layers)]
        else:
            # MLP version (paper says better for Sudoku)
            self.layers = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
            )

        self.out = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array, y: mx.array, z: mx.array) -> mx.array:
        # Combine inputs: [B, L, H*3] -> [B, L, H]
        combined = self.combine(mx.concatenate([x, y, z], axis=-1))

        if self.config.use_attention:
            h = combined
            for layer, norm in zip(self.layers, self.norms):
                h = h + layer(norm(h), norm(h), norm(h))
        else:
            h = self.layers(combined)

        return self.out(h)


class AnswerUpdateNet(nn.Module):
    """Network for: y ← net(y, z)"""

    def __init__(self, config: IterativeConfig):
        super().__init__()
        hidden = config.hidden_size

        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        combined = mx.concatenate([y, z], axis=-1)
        return self.net(combined)


class HaltingNet(nn.Module):
    """Learns when to stop recursion: q ← net(y, z)"""

    def __init__(self, config: IterativeConfig):
        super().__init__()
        hidden = config.hidden_size

        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        combined = mx.concatenate([y, z], axis=-1)
        # Global pooling then predict
        pooled = mx.mean(combined, axis=1)  # [B, H*2]
        return self.net(pooled)  # [B, 1]


class IterativeTRM(nn.Module):
    """
    Iterative TRM with recursive reasoning.

    Implements the core TRM loop:
        for t in range(T):
            for i in range(n):
                z ← latent_net(x, y, z)
            y ← answer_net(y, z)
            if halting_net(y, z) > 0: break
    """

    def __init__(self, config: Optional[IterativeConfig] = None):
        super().__init__()
        self.config = config or IterativeConfig()

        # Embeddings
        self.input_embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_embed = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        # Initial z projection
        self.z_init = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        # Core networks
        self.latent_net = LatentUpdateNet(self.config)
        self.answer_net = AnswerUpdateNet(self.config)

        if self.config.use_halting:
            self.halting_net = HaltingNet(self.config)

        # Output head
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def __call__(self, input_ids: mx.array, return_halting: bool = False) -> mx.array:
        """
        Forward pass with recursive reasoning.

        Args:
            input_ids: [B, 81] input puzzle
            return_halting: If True, also return halting logits

        Returns:
            logits: [B, 81, 10] predictions
            halting_logits: [B, T] if return_halting
        """
        B, L = input_ids.shape

        # Embed input (x is fixed throughout)
        positions = mx.arange(L)
        x = self.input_embed(input_ids) + self.pos_embed(positions)

        # Initialize y (answer) and z (latent)
        y = x  # Start with input embedding
        z = self.z_init(x)  # Initialize latent

        halting_logits = []

        # Outer recursion loop
        for t in range(self.config.T):
            # Inner latent update loop
            for i in range(self.config.n):
                z = z + self.latent_net(x, y, z)  # Residual update

            # Update answer
            y = y + self.answer_net(y, z)  # Residual update

            # Halting prediction (for training)
            if self.config.use_halting:
                h_logit = self.halting_net(y, z)
                halting_logits.append(h_logit)

        # Final prediction
        logits = self.head(self.norm(y))

        if return_halting and halting_logits:
            return logits, mx.concatenate(halting_logits, axis=-1)
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

        # Target: values 1-9 in solution, shift to 0-9 for vocab
        # Actually vocab is 0-9 where 0 = empty, 1-9 = values
        # So we use solution directly as target
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


# Also create MLP-only version (paper says better for Sudoku)
class IterativeTRM_MLP(IterativeTRM):
    """MLP version without attention (paper finds this better for Sudoku)."""

    def __init__(self, config: Optional[IterativeConfig] = None):
        config = config or IterativeConfig()
        config.use_attention = False
        super().__init__(config)


def create_model(config: Optional[IterativeConfig] = None) -> IterativeTRM:
    """Factory function."""
    return IterativeTRM(config)


def create_mlp_model(config: Optional[IterativeConfig] = None) -> IterativeTRM_MLP:
    """Factory function for MLP version."""
    return IterativeTRM_MLP(config)


if __name__ == "__main__":
    print("Testing Iterative TRM (attention version)...")
    config = IterativeConfig(T=3, n=6, use_attention=True)
    model = create_model(config)

    batch = mx.zeros((2, 81), dtype=mx.int32)
    logits = model(batch)
    print(f"  Input: {batch.shape}, Output: {logits.shape}")
    print(f"  T={config.T}, n={config.n}")

    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Parameters: {num_params:,}")

    print("\nTesting Iterative TRM (MLP version)...")
    config_mlp = IterativeConfig(T=3, n=6, use_attention=False)
    model_mlp = create_mlp_model(config_mlp)

    logits_mlp = model_mlp(batch)
    print(f"  Input: {batch.shape}, Output: {logits_mlp.shape}")

    num_params_mlp = sum(p.size for p in model_mlp.parameters().values())
    print(f"  Parameters: {num_params_mlp:,}")
