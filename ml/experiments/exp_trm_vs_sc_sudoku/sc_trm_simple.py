"""
Simplified SC-TRM: Minimal statechart augmentation on vanilla TRM.

Key differences from full SC-TRM:
1. Same architecture as vanilla TRM (no extra complexity)
2. Optional soft constraint loss (auxiliary, not blocking)
3. No guard masking during training (gradients flow freely)
4. Guards only applied at inference time

Hypothesis: Learning works better without guard blocking during training.
"""

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .vanilla_trm import VanillaTRM, VanillaTRMConfig, stablemax


@dataclass
class SimpleSCConfig:
    """Configuration for simplified SC-TRM."""
    # Same as vanilla TRM
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512
    num_cells: int = 81
    num_digits: int = 9
    H_cycles: int = 3
    L_cycles: int = 6
    dropout: float = 0.1
    use_halting: bool = True

    # SC-specific (soft constraints)
    constraint_loss_weight: float = 0.1  # Weight for auxiliary constraint loss
    use_inference_guards: bool = True    # Apply guards only at inference


class SimpleSCTRM(VanillaTRM):
    """
    Simplified SC-TRM: Vanilla TRM + soft constraint loss.

    Inherits from VanillaTRM to ensure proper gradient flow.
    Training: Pure cross-entropy + optional constraint violation penalty
    Inference: Apply constraint guards to filter predictions
    """

    def __init__(self, config: Optional[SimpleSCConfig] = None):
        self.sc_config = config or SimpleSCConfig()

        # Initialize parent VanillaTRM with matching config
        vanilla_config = VanillaTRMConfig(
            hidden_dim=self.sc_config.hidden_dim,
            num_heads=self.sc_config.num_heads,
            num_layers=self.sc_config.num_layers,
            ff_dim=self.sc_config.ff_dim,
            num_cells=self.sc_config.num_cells,
            num_digits=self.sc_config.num_digits,
            H_cycles=self.sc_config.H_cycles,
            L_cycles=self.sc_config.L_cycles,
            dropout=self.sc_config.dropout,
            use_halting=self.sc_config.use_halting,
        )
        super().__init__(vanilla_config)

    def compute_constraint_violations(
        self,
        probs: mx.array,  # [B, 81, 9]
    ) -> mx.array:
        """
        Compute soft constraint violations.

        Returns scalar loss penalizing predictions that violate Sudoku rules.
        """
        B = probs.shape[0]

        # Reshape to [B, 9, 9, 9] = [batch, row, col, digit]
        board = probs.reshape(B, 9, 9, 9)

        # Row violations: sum of probs for same digit in same row should be ~1
        row_sums = mx.sum(board, axis=2)  # [B, 9, 9] = [batch, row, digit]
        row_violation = mx.mean(mx.square(row_sums - 1.0))

        # Col violations: sum of probs for same digit in same col should be ~1
        col_sums = mx.sum(board, axis=1)  # [B, 9, 9] = [batch, col, digit]
        col_violation = mx.mean(mx.square(col_sums - 1.0))

        # Box violations
        board_boxes = board.reshape(B, 3, 3, 3, 3, 9)
        box_sums = mx.sum(board_boxes, axis=(2, 4))  # [B, 3, 3, 9]
        box_violation = mx.mean(mx.square(box_sums - 1.0))

        return row_violation + col_violation + box_violation

    def apply_inference_guards(
        self,
        logits: mx.array,  # [B, 81, 9]
        puzzle: mx.array,  # [B, 81]
    ) -> mx.array:
        """
        Apply hard constraint guards at inference time.

        For each cell, mask out digits that are already GIVEN in same row/col/box.
        Only uses the known puzzle cells, not predictions.
        """
        B = logits.shape[0]

        # Use only given cells from puzzle (0 for empty cells)
        board = puzzle.reshape(B, 9, 9)  # [B, 9, 9], values 0-9 where 0=empty

        # Build validity mask: [B, 81, 9]
        # For each cell and each digit, check if that digit is already in row/col/box
        valid = mx.ones((B, 81, 9))

        for cell in range(81):
            row, col = cell // 9, cell % 9
            box_r, box_c = (row // 3) * 3, (col // 3) * 3

            # Get given digits in same row/col/box
            row_digits = board[:, row, :]  # [B, 9]
            col_digits = board[:, :, col]  # [B, 9]
            box_digits = board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)

            # For each digit 1-9, check if it's already given
            for d in range(9):
                digit = d + 1
                in_row = mx.any(row_digits == digit, axis=1)  # [B]
                in_col = mx.any(col_digits == digit, axis=1)
                in_box = mx.any(box_digits == digit, axis=1)
                blocked = in_row | in_col | in_box
                # Update validity for this cell/digit combo
                cell_mask = mx.arange(81)[None, :, None] == cell
                digit_mask = mx.arange(9)[None, None, :] == d
                valid = mx.where(
                    blocked[:, None, None] & cell_mask & digit_mask,
                    mx.zeros_like(valid),
                    valid
                )

        # Apply mask
        masked_logits = logits + mx.log(valid + 1e-10)
        return masked_logits

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
        use_guards: bool = None,
    ) -> Dict[str, mx.array]:
        """Solve puzzle using inherited TRM model."""
        # Use parent's solve method
        result = super().solve(
            puzzle,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            return_trajectory=return_trajectory,
        )

        # Optionally apply inference guards
        if use_guards is None:
            use_guards = self.sc_config.use_inference_guards

        if use_guards:
            result['logits'] = self.apply_inference_guards(result['logits'], puzzle)
            result['predictions'] = mx.argmax(result['logits'], axis=-1) + 1

        return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        use_stablemax: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute loss with optional constraint penalty.

        Uses single forward pass for both losses to ensure gradient connection.
        """
        # Single forward pass
        result = super().solve(puzzle, return_trajectory=True)
        logits = result['logits']  # [B, 81, 9]

        # Target: shift to 0-8 for cross-entropy
        target = solution - 1  # [B, 81], values 0-8

        # Compute probabilities
        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            probs = mx.softmax(logits, axis=-1)

        # Cross-entropy loss (same as VanillaTRM)
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)  # [B, 81]
        base_loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Compute constraint violation on same forward pass
        constraint_loss = self.compute_constraint_violations(probs)

        # Total loss
        total_loss = base_loss + self.sc_config.constraint_loss_weight * constraint_loss

        # Compute metrics
        predictions = mx.argmax(logits, axis=-1)  # [B, 81], 0-8
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            'loss': total_loss,
            'base_loss': base_loss,
            'constraint_loss': constraint_loss,
            'cell_accuracy': cell_accuracy,
            'exact_accuracy': exact_accuracy,
        }

        return total_loss, metrics


def test_simple_sc_trm():
    """Test simplified SC-TRM."""
    print("="*60)
    print("Testing Simplified SC-TRM")
    print("="*60)

    config = SimpleSCConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=2,
        constraint_loss_weight=0.1,
        use_inference_guards=False,
    )

    model = SimpleSCTRM(config)

    # Test data
    B = 4
    puzzle = mx.zeros((B, 81), dtype=mx.int32)
    solution = mx.ones((B, 81), dtype=mx.int32)

    print("\n1. Testing solve...")
    result = model.solve(puzzle, use_guards=False)
    print(f"   Logits shape: {result['logits'].shape}")
    print(f"   Predictions shape: {result['predictions'].shape}")

    print("\n2. Testing loss...")
    loss, metrics = model.loss(puzzle, solution)
    print(f"   Total loss: {float(loss.item()):.4f}")
    print(f"   Base loss: {float(metrics['base_loss'].item()):.4f}")
    print(f"   Constraint loss: {float(metrics['constraint_loss'].item()):.4f}")

    print("\n3. Parameter count...")
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, mx.array):
            total += params.size
        elif isinstance(params, (list, tuple)):
            for v in params:
                total += count_params(v)
        return total

    num_params = count_params(model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    import mlx.nn as nn_module
    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss
    loss_and_grad = nn_module.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, puzzle, solution)
    mx.eval(loss, grads)

    def count_grad_norms(grads, prefix=""):
        norms = []
        if isinstance(grads, dict):
            for k, v in grads.items():
                norms.extend(count_grad_norms(v, f"{prefix}{k}."))
        elif isinstance(grads, mx.array):
            norm = float(mx.sqrt(mx.sum(grads * grads)).item())
            norms.append((prefix.rstrip('.'), norm))
        elif isinstance(grads, (list, tuple)):
            for i, v in enumerate(grads):
                norms.extend(count_grad_norms(v, f"{prefix}{i}."))
        return norms

    all_norms = count_grad_norms(grads)
    mean_norm = sum(n for _, n in all_norms) / len(all_norms) if all_norms else 0
    print(f"   Mean gradient norm: {mean_norm:.2e}")
    print(f"   Total param groups: {len(all_norms)}")

    print("\n" + "="*60)
    print("Simplified SC-TRM test complete")
    print("="*60)


if __name__ == "__main__":
    test_simple_sc_trm()
