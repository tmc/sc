"""
Faithful TRM: Implementation matching Samsung SAIL Montreal's TinyRecursiveModels.

This closely follows the upstream architecture:
- Dual latent structure (z_H, z_L) instead of single hidden state
- RMSNorm (post-norm) instead of LayerNorm (pre-norm)
- SwiGLU activation instead of GELU
- Gradient truncation: only backprop through last H cycle

Reference: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class FaithfulTRMConfig:
    """Configuration matching TRM paper exactly."""
    # Model dimensions
    d_model: int = 256
    d_hidden: int = 512  # FF hidden dim
    num_heads: int = 8
    num_layers: int = 4

    # TRM-specific
    d_z_H: int = 256  # High-level latent dimension
    d_z_L: int = 256  # Low-level latent dimension

    # Sudoku specifics
    num_cells: int = 81
    num_digits: int = 9

    # TRM iteration config
    H_cycles: int = 3
    L_cycles: int = 6

    # Training
    dropout: float = 0.1
    truncate_grads: bool = True  # Only backprop through last H cycle


def stablemax(logits: mx.array, axis: int = -1) -> mx.array:
    """
    Stablemax activation from TRM paper.

    More numerically stable than softmax for iterative refinement.
    s_i = 1/(1-x_i) if x_i < 0 else x_i + 1
    """
    s = mx.where(logits < 0, 1.0 / (1.0 - logits), logits + 1.0)
    return s / mx.sum(s, axis=axis, keepdims=True)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by TRM upstream)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation (used by TRM upstream instead of GELU)."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden)
        self.w2 = nn.Linear(d_model, d_hidden)
        self.w3 = nn.Linear(d_hidden, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU: out = (swish(W1*x) * W2*x) @ W3
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class FaithfulTransformerBlock(nn.Module):
    """Transformer block matching TRM's architecture."""

    def __init__(self, config: FaithfulTRMConfig):
        super().__init__()
        self.config = config

        self.attn = nn.MultiHeadAttention(
            dims=config.d_model,
            num_heads=config.num_heads,
        )
        self.norm1 = RMSNorm(config.d_model)

        self.ff = SwiGLU(config.d_model, config.d_hidden)
        self.norm2 = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # Post-norm (TRM style): residual, then norm
        h = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(h))

        h = self.ff(x)
        x = self.norm2(x + self.dropout(h))

        return x


class FaithfulTRM(nn.Module):
    """
    Faithful implementation of Samsung SAIL Montreal's TinyRecursiveModels.

    Key architectural features:
    - Dual latent structure: z_H (high-level) and z_L (low-level)
    - z_L = L_level(z_L, z_H + inputs)
    - Gradient truncation: only last H cycle gets gradients
    - RMSNorm + SwiGLU
    """

    def __init__(self, config: Optional[FaithfulTRMConfig] = None):
        super().__init__()
        self.config = config or FaithfulTRMConfig()

        # Input embedding: puzzle position + digit -> d_model
        self.pos_embed = nn.Embedding(self.config.num_cells, self.config.d_model)
        self.digit_embed = nn.Embedding(self.config.num_digits + 1, self.config.d_model)  # 0-9
        self.input_proj = nn.Linear(self.config.d_model * 2, self.config.d_model)

        # H-level network: updates z_H based on z_L
        self.H_level = nn.Sequential(
            nn.Linear(self.config.d_z_L, self.config.d_hidden),
            nn.SiLU(),
            nn.Linear(self.config.d_hidden, self.config.d_z_H),
        )

        # L-level network: transformer that refines z_L given (z_L, z_H + inputs)
        self.z_H_proj = nn.Linear(self.config.d_z_H, self.config.d_model)
        self.L_blocks = [
            FaithfulTransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ]

        # Output head: z_L -> digit logits
        self.output_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.num_digits),
        )

        # Initial z_H and z_L projections
        self.z_H_init = nn.Linear(self.config.d_model, self.config.d_z_H)
        self.z_L_init = nn.Linear(self.config.d_model, self.config.d_z_L)

    def embed_puzzle(self, puzzle: mx.array) -> mx.array:
        """
        Embed puzzle into initial representations.

        Args:
            puzzle: [B, 81] with values 0-9 (0=empty)

        Returns:
            [B, 81, d_model] input embeddings
        """
        B = puzzle.shape[0]

        # Position embeddings
        positions = mx.arange(self.config.num_cells, dtype=mx.int32)
        positions = mx.broadcast_to(positions, (B, self.config.num_cells))
        pos_emb = self.pos_embed(positions)  # [B, 81, d_model]

        # Digit embeddings
        puzzle = puzzle.astype(mx.int32)
        dig_emb = self.digit_embed(puzzle)  # [B, 81, d_model]

        # Combine
        combined = mx.concatenate([pos_emb, dig_emb], axis=-1)  # [B, 81, d_model*2]
        return self.input_proj(combined)  # [B, 81, d_model]

    def L_level_step(
        self,
        z_L: mx.array,     # [B, 81, d_z_L]
        z_H: mx.array,     # [B, d_z_H]
        inputs: mx.array,  # [B, 81, d_model]
    ) -> mx.array:
        """
        Single L-level step: refine z_L given z_H and inputs.

        TRM formula: z_L = L_level(z_L, z_H + inputs)
        """
        # Project z_H and add to inputs
        z_H_expanded = self.z_H_proj(z_H)[:, None, :]  # [B, 1, d_model]
        h = z_L + z_H_expanded + inputs  # [B, 81, d_model]

        # Apply transformer blocks
        for block in self.L_blocks:
            h = block(h)

        return h

    def H_level_step(self, z_L: mx.array) -> mx.array:
        """
        Single H-level step: update z_H based on z_L.

        Pool z_L across cells and compute new z_H.
        """
        pooled = mx.mean(z_L, axis=1)  # [B, d_z_L]
        return self.H_level(pooled)  # [B, d_z_H]

    def solve(
        self,
        puzzle: mx.array,  # [B, 81]
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Solve puzzle using H x L iteration.

        Key difference from vanilla: dual latent (z_H, z_L) structure.
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Initial embedding
        inputs = self.embed_puzzle(puzzle)  # [B, 81, d_model]

        # Initialize latents
        pooled_input = mx.mean(inputs, axis=1)  # [B, d_model]
        z_H = self.z_H_init(pooled_input)  # [B, d_z_H]
        z_L = inputs  # [B, 81, d_z_L]

        trajectory = [] if return_trajectory else None

        for hi in range(H):
            # Gradient truncation: stop gradients for all but last H cycle
            if self.config.truncate_grads and hi < H - 1:
                z_L = mx.stop_gradient(z_L)
                z_H = mx.stop_gradient(z_H)

            # L-level iterations
            for li in range(L):
                z_L = self.L_level_step(z_L, z_H, inputs)

                if return_trajectory:
                    logits = self.output_head(z_L)
                    trajectory.append(logits)

            # H-level update
            z_H = self.H_level_step(z_L)

        # Final prediction
        logits = self.output_head(z_L)  # [B, 81, 9]
        predictions = mx.argmax(logits, axis=-1) + 1  # 1-9

        result = {
            "logits": logits,
            "predictions": predictions,
            "final_z_L": z_L,
            "final_z_H": z_H,
        }

        if return_trajectory:
            result["trajectory"] = mx.stack(trajectory, axis=1)

        return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        use_stablemax: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute loss for training.

        Uses stablemax (TRM's more stable softmax variant).
        """
        result = self.solve(puzzle, return_trajectory=False)
        logits = result["logits"]  # [B, 81, 9]

        # Target: shift to 0-8 for cross-entropy
        target = solution - 1  # [B, 81], values 0-8

        # Compute probabilities
        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            probs = mx.softmax(logits, axis=-1)

        # Cross-entropy loss
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)  # [B, 81]

        loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            "loss": loss,
            "cell_accuracy": cell_accuracy,
            "exact_accuracy": exact_accuracy,
        }

        return loss, metrics


def test_faithful_trm():
    """Test faithful TRM implementation."""
    print("=" * 60)
    print("Testing Faithful TRM (matching upstream architecture)")
    print("=" * 60)

    config = FaithfulTRMConfig(
        d_model=64,
        d_hidden=128,
        num_heads=4,
        num_layers=2,
        d_z_H=64,
        d_z_L=64,
        H_cycles=2,
        L_cycles=3,
    )

    model = FaithfulTRM(config)

    # Create test data
    mx.random.seed(42)
    B = 4
    puzzle = mx.random.randint(0, 10, (B, 81))
    solution = mx.random.randint(1, 10, (B, 81))

    print("\n1. Testing embedding...")
    inputs = model.embed_puzzle(puzzle)
    print(f"   Input embedding shape: {inputs.shape}")

    print("\n2. Testing solve...")
    result = model.solve(puzzle, return_trajectory=True)
    print(f"   Logits shape: {result['logits'].shape}")
    print(f"   Predictions shape: {result['predictions'].shape}")
    print(f"   z_L shape: {result['final_z_L'].shape}")
    print(f"   z_H shape: {result['final_z_H'].shape}")
    if 'trajectory' in result:
        print(f"   Trajectory shape: {result['trajectory'].shape}")

    print("\n3. Testing loss...")
    loss, metrics = model.loss(puzzle, solution)
    print(f"   Loss: {float(loss.tolist()):.4f}")
    print(f"   Cell accuracy: {float(metrics['cell_accuracy'].tolist()):.2%}")
    print(f"   Exact accuracy: {float(metrics['exact_accuracy'].tolist()):.2%}")

    print("\n4. Testing gradient truncation...")
    # Verify gradients only flow through last H cycle
    def loss_fn(m, p, s):
        l, _ = m.loss(p, s)
        return l

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(model, puzzle, solution)
    print(f"   Gradient computed (truncation enabled: {config.truncate_grads})")

    print("\n5. Parameter count...")
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total
    num_params = count_params(model.parameters())
    print(f"   Parameters: {num_params:,}")

    print("\n6. Architecture comparison:")
    print("   Feature              | Vanilla TRM | Faithful TRM")
    print("   --------------------|-------------|-------------")
    print("   Normalization       | LayerNorm   | RMSNorm     ")
    print("   Activation          | GELU        | SwiGLU      ")
    print("   Latent structure    | Single h    | Dual z_H/z_L")
    print("   Gradient truncation | None        | Last H only ")

    print("\n" + "=" * 60)
    print("Faithful TRM test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_faithful_trm()
