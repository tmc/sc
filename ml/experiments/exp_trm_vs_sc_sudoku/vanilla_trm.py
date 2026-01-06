"""
Vanilla TRM: Faithful implementation of TinyRecursiveModels for Sudoku.

This is the baseline - no statechart structure, just the raw TRM architecture
with H×L iteration and learned halting.

Reference: Samsung SAIL Montreal's TinyRecursiveModels
- 7M parameters, 87% on Sudoku-Extreme
- H_cycles=3, L_cycles=6 (18 total iterations)
- Non-autoregressive: predict all 81 cells in parallel
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class VanillaTRMConfig:
    """Configuration matching TRM paper."""
    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512

    # Sudoku specifics
    num_cells: int = 81
    num_digits: int = 9

    # TRM iteration config
    H_cycles: int = 3
    L_cycles: int = 6

    # Training
    dropout: float = 0.1
    use_halting: bool = True


def stablemax(logits: mx.array, axis: int = -1) -> mx.array:
    """
    Stablemax activation from TRM paper.

    More numerically stable than softmax for iterative refinement.
    """
    # s_i = 1/(1-x_i) if x_i < 0 else x_i + 1
    s = mx.where(logits < 0, 1.0 / (1.0 - logits), logits + 1.0)
    return s / mx.sum(s, axis=axis, keepdims=True)


class CellEmbedding(nn.Module):
    """Embed cell position and current digit prediction."""

    def __init__(self, config: VanillaTRMConfig):
        super().__init__()
        self.config = config

        # Position embedding for 81 cells
        self.pos_embed = nn.Embedding(config.num_cells, config.hidden_dim)

        # Digit embedding (0=empty, 1-9=digits)
        self.digit_embed = nn.Embedding(config.num_digits + 1, config.hidden_dim)

        # Combine
        self.combine = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def __call__(
        self,
        positions: mx.array,  # [B, 81]
        digits: mx.array,     # [B, 81]
    ) -> mx.array:
        """
        Embed cells.

        Returns:
            [B, 81, hidden_dim]
        """
        pos_emb = self.pos_embed(positions)
        dig_emb = self.digit_embed(digits)
        combined = mx.concatenate([pos_emb, dig_emb], axis=-1)
        return self.combine(combined)


class TransformerBlock(nn.Module):
    """Standard transformer block."""

    def __init__(self, config: VanillaTRMConfig):
        super().__init__()
        self.config = config

        self.attn = nn.MultiHeadAttention(
            dims=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention
        h = self.norm1(x)
        h = self.attn(h, h, h)
        x = x + self.dropout(h)

        # Feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        return x


class VanillaTRM(nn.Module):
    """
    Vanilla TinyRecursiveModel for Sudoku.

    Key differences from SC version:
    - No explicit constraint guards
    - No statechart structure
    - Pure learned iteration dynamics
    """

    def __init__(self, config: Optional[VanillaTRMConfig] = None):
        super().__init__()
        self.config = config or VanillaTRMConfig()

        # Cell embedding
        self.cell_embed = CellEmbedding(self.config)

        # H-level context network
        self.h_context_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # L-level transformer blocks
        self.blocks = [
            TransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ]

        # Output head: predict digit for each cell
        self.output_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_digits),
        )

        # Halting head: predict when to stop
        if self.config.use_halting:
            self.halt_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # Position indices (fixed)
        self._positions = None

    def _get_positions(self, batch_size: int) -> mx.array:
        """Get position indices for batch."""
        if self._positions is None or self._positions.shape[0] != batch_size:
            pos = mx.arange(self.config.num_cells, dtype=mx.int32)
            self._positions = mx.broadcast_to(pos, (batch_size, self.config.num_cells))
        return self._positions

    def encode(
        self,
        puzzle: mx.array,  # [B, 81] input puzzle (0=empty, 1-9=given)
    ) -> mx.array:
        """
        Initial encoding of puzzle.

        Returns:
            [B, 81, hidden_dim]
        """
        B = puzzle.shape[0]
        positions = self._get_positions(B)
        # Ensure integer types for embedding lookup
        positions = positions.astype(mx.int32)
        puzzle = puzzle.astype(mx.int32)
        return self.cell_embed(positions, puzzle)

    def refine(
        self,
        h: mx.array,           # [B, 81, hidden_dim]
        h_context: mx.array,   # [B, hidden_dim]
    ) -> mx.array:
        """
        Single L-level refinement step.

        Returns:
            [B, 81, hidden_dim] refined representations
        """
        # Add H-context to all cells
        h = h + h_context[:, None, :]

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)

        return h

    def predict(self, h: mx.array) -> mx.array:
        """
        Predict digit logits.

        Args:
            h: [B, 81, hidden_dim]

        Returns:
            [B, 81, 9] digit logits
        """
        return self.output_head(h)

    def predict_halt(self, h: mx.array) -> mx.array:
        """
        Predict halting probability.

        Args:
            h: [B, 81, hidden_dim]

        Returns:
            [B] halt probabilities
        """
        if not self.config.use_halting:
            return mx.zeros((h.shape[0],))

        # Pool over cells
        pooled = mx.mean(h, axis=1)  # [B, hidden_dim]
        halt_logit = self.halt_head(pooled)  # [B, 1]
        return mx.sigmoid(halt_logit.squeeze(-1))

    def solve(
        self,
        puzzle: mx.array,  # [B, 81]
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Solve puzzle using H×L iteration.

        Args:
            puzzle: [B, 81] input puzzle
            H_cycles: Override H cycles
            L_cycles: Override L cycles
            return_trajectory: Return intermediate predictions

        Returns:
            Dict with predictions and optionally trajectory
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Initial encoding
        h = self.encode(puzzle)

        trajectory = [] if return_trajectory else None
        halt_probs = []

        for hi in range(H):
            # Compute H-level context
            pooled = mx.mean(h, axis=1)  # [B, hidden_dim]
            h_context = self.h_context_net(pooled)

            for li in range(L):
                # L-level refinement
                h = self.refine(h, h_context)

                # Track predictions
                if return_trajectory:
                    logits = self.predict(h)
                    trajectory.append(logits)

                # Check halting
                if self.config.use_halting:
                    halt_p = self.predict_halt(h)
                    halt_probs.append(halt_p)

        # Final prediction
        logits = self.predict(h)
        predictions = mx.argmax(logits, axis=-1) + 1  # 1-9

        result = {
            "logits": logits,
            "predictions": predictions,
            "final_h": h,
        }

        if return_trajectory:
            result["trajectory"] = mx.stack(trajectory, axis=1)  # [B, T, 81, 9]

        if halt_probs:
            result["halt_probs"] = mx.stack(halt_probs, axis=1)  # [B, T]

        return result

    def loss(
        self,
        puzzle: mx.array,   # [B, 81]
        solution: mx.array, # [B, 81] target (1-9)
        use_stablemax: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute loss for training.

        Args:
            puzzle: Input puzzle
            solution: Target solution (1-9)
            use_stablemax: Use stablemax instead of softmax

        Returns:
            (loss, metrics_dict)
        """
        result = self.solve(puzzle, return_trajectory=True)
        logits = result["logits"]  # [B, 81, 9]

        # Target: shift to 0-8 for cross-entropy
        target = solution - 1  # [B, 81], values 0-8

        # Compute probabilities
        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            probs = mx.softmax(logits, axis=-1)

        # Cross-entropy loss
        # Gather correct class probabilities
        B, C = target.shape
        batch_idx = mx.arange(B)[:, None]
        cell_idx = mx.arange(C)[None, :]

        # Manual gather for correct probs
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)  # [B, 81]

        # Negative log likelihood
        loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Compute metrics
        predictions = mx.argmax(logits, axis=-1)  # [B, 81], 0-8
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))

        # Exact match (all 81 cells correct)
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            "loss": loss,
            "cell_accuracy": cell_accuracy,
            "exact_accuracy": exact_accuracy,
        }

        return loss, metrics


def test_vanilla_trm():
    """Test vanilla TRM."""
    print("=" * 60)
    print("Testing Vanilla TRM")
    print("=" * 60)

    config = VanillaTRMConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=3,
    )

    model = VanillaTRM(config)

    # Create test data
    mx.random.seed(42)
    B = 4
    puzzle = mx.random.randint(0, 10, (B, 81))  # 0=empty, 1-9=given
    solution = mx.random.randint(1, 10, (B, 81))  # 1-9

    print("\n1. Testing encoding...")
    h = model.encode(puzzle)
    print(f"   Encoded shape: {h.shape}")

    print("\n2. Testing solve...")
    result = model.solve(puzzle, return_trajectory=True)
    print(f"   Logits shape: {result['logits'].shape}")
    print(f"   Predictions shape: {result['predictions'].shape}")
    print(f"   Trajectory shape: {result['trajectory'].shape}")

    print("\n3. Testing loss...")
    loss, metrics = model.loss(puzzle, solution)
    print(f"   Loss: {float(loss.tolist()):.4f}")
    print(f"   Cell accuracy: {float(metrics['cell_accuracy'].tolist()):.2%}")
    print(f"   Exact accuracy: {float(metrics['exact_accuracy'].tolist()):.2%}")

    print("\n4. Parameter count...")
    num_params = sum(p.size for p in model.parameters().values())
    print(f"   Parameters: {num_params:,}")

    print("\n" + "=" * 60)
    print("Vanilla TRM test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_vanilla_trm()
