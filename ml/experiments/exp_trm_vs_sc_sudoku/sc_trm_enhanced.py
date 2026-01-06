"""
Enhanced SC-TRM: Advanced statechart-augmented TRM variants.

Explores multiple ways to integrate constraints:
1. Iterative inference guards (apply after each H-cycle)
2. Guard-informed refinement (feed constraint info back)
3. Multi-pass inference (run multiple times with filtering)
4. Constraint propagation (iteratively eliminate impossible values)
"""

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .vanilla_trm import VanillaTRM, VanillaTRMConfig, stablemax


@dataclass
class EnhancedSCConfig:
    """Configuration for enhanced SC-TRM."""
    # Base TRM config
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

    # Enhanced SC options
    iterative_guards: bool = True       # Apply guards after each H-cycle
    multi_pass: int = 1                 # Number of inference passes
    constraint_propagation: bool = True  # Use constraint propagation
    confidence_threshold: float = 0.8   # Confidence for constraint prop


class EnhancedSCTRM(VanillaTRM):
    """
    Enhanced SC-TRM with multiple constraint integration strategies.
    """

    def __init__(self, config: Optional[EnhancedSCConfig] = None):
        self.sc_config = config or EnhancedSCConfig()

        # Initialize parent VanillaTRM
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

    def compute_validity_mask(
        self,
        puzzle: mx.array,      # [B, 81] given cells
        predictions: mx.array,  # [B, 81] current predictions (optional)
        confidence: Optional[mx.array] = None,  # [B, 81] confidence scores
        use_predictions: bool = False,
    ) -> mx.array:
        """
        Compute validity mask: which digits are valid for each cell.

        If use_predictions=True and confidence is provided, also block
        digits that are confidently predicted in same row/col/box.

        Returns:
            [B, 81, 9] validity mask (1=valid, 0=blocked)
        """
        B = puzzle.shape[0]

        # Start with all valid
        valid = mx.ones((B, 81, 9))

        # Build board from givens
        board = puzzle.reshape(B, 9, 9)  # [B, 9, 9]

        # Block digits already given in row/col/box
        for cell in range(81):
            row, col = cell // 9, cell % 9
            box_r, box_c = (row // 3) * 3, (col // 3) * 3

            # Get given digits
            row_digits = board[:, row, :]
            col_digits = board[:, :, col]
            box_digits = board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)

            for d in range(9):
                digit = d + 1
                in_row = mx.any(row_digits == digit, axis=1)
                in_col = mx.any(col_digits == digit, axis=1)
                in_box = mx.any(box_digits == digit, axis=1)
                blocked = in_row | in_col | in_box

                cell_mask = mx.arange(81)[None, :, None] == cell
                digit_mask = mx.arange(9)[None, None, :] == d
                valid = mx.where(
                    blocked[:, None, None] & cell_mask & digit_mask,
                    mx.zeros_like(valid),
                    valid
                )

        # Optionally also consider confident predictions
        if use_predictions and confidence is not None:
            threshold = self.sc_config.confidence_threshold
            pred_board = predictions.reshape(B, 9, 9)
            conf_board = confidence.reshape(B, 9, 9)

            # For each cell, check if any other cell in row/col/box
            # has a confident prediction of each digit
            for cell in range(81):
                row, col = cell // 9, cell % 9
                box_r, box_c = (row // 3) * 3, (col // 3) * 3

                for d in range(9):
                    digit = d + 1

                    # Check row (excluding self)
                    row_preds = pred_board[:, row, :]
                    row_conf = conf_board[:, row, :]
                    row_confident = (row_preds == digit) & (row_conf > threshold)
                    row_confident_other = mx.sum(row_confident.astype(mx.float32), axis=1) > 0

                    # Check col
                    col_preds = pred_board[:, :, col]
                    col_conf = conf_board[:, :, col]
                    col_confident = (col_preds == digit) & (col_conf > threshold)
                    col_confident_other = mx.sum(col_confident.astype(mx.float32), axis=1) > 0

                    # Check box
                    box_preds = pred_board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)
                    box_conf = conf_board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)
                    box_confident = (box_preds == digit) & (box_conf > threshold)
                    box_confident_other = mx.sum(box_confident.astype(mx.float32), axis=1) > 0

                    blocked = row_confident_other | col_confident_other | box_confident_other

                    cell_mask = mx.arange(81)[None, :, None] == cell
                    digit_mask = mx.arange(9)[None, None, :] == d
                    valid = mx.where(
                        blocked[:, None, None] & cell_mask & digit_mask,
                        mx.zeros_like(valid),
                        valid
                    )

        return valid

    def apply_constraint_propagation(
        self,
        logits: mx.array,      # [B, 81, 9]
        puzzle: mx.array,      # [B, 81]
        max_iters: int = 5,
    ) -> mx.array:
        """
        Iterative constraint propagation.

        1. Compute valid mask from givens
        2. Apply mask to logits
        3. If any cell has only one valid digit, treat as "given"
        4. Repeat until convergence
        """
        B = logits.shape[0]

        # Start with puzzle givens (MLX uses copy via slicing or mx.array())
        current_board = mx.array(puzzle)
        current_logits = mx.array(logits)

        for _ in range(max_iters):
            # Compute validity
            valid = self.compute_validity_mask(current_board, None, None, False)

            # Apply mask
            masked_logits = current_logits + mx.log(valid + 1e-10)

            # Get predictions and confidence
            probs = mx.softmax(masked_logits, axis=-1)
            predictions = mx.argmax(probs, axis=-1) + 1  # [B, 81]
            confidence = mx.max(probs, axis=-1)  # [B, 81]

            # Check for naked singles (only one valid digit)
            num_valid = mx.sum(valid, axis=-1)  # [B, 81]
            naked_singles = (num_valid == 1) & (current_board.reshape(B, 81) == 0)

            if not mx.any(naked_singles):
                break

            # Update board with naked singles
            single_digit = mx.argmax(valid, axis=-1) + 1  # [B, 81]
            current_board = mx.where(
                naked_singles,
                single_digit,
                current_board.reshape(B, 81)
            )
            current_logits = masked_logits

        return current_logits

    def solve_with_iterative_guards(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """
        Solve with guards applied after each H-cycle.

        This allows the model to benefit from constraint filtering
        during the refinement process.
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Initial encoding
        h = self.encode(puzzle)

        for hi in range(H):
            # Compute H-level context
            pooled = mx.mean(h, axis=1)
            h_context = self.h_context_net(pooled)

            for li in range(L):
                h = self.refine(h, h_context)

            # After each H-cycle, apply soft constraint filtering
            if hi < H - 1:  # Not the last cycle
                logits = self.predict(h)
                valid = self.compute_validity_mask(puzzle, None, None, False)

                # Soft filtering: boost valid, suppress invalid
                filtered_logits = logits + mx.log(valid + 1e-10)

                # Convert back to representation via embedding
                probs = mx.softmax(filtered_logits, axis=-1)
                predictions = mx.argmax(probs, axis=-1) + 1

                # Re-encode with filtered predictions
                B = puzzle.shape[0]
                positions = self._get_positions(B)
                positions = positions.astype(mx.int32)

                # Use predictions for empty cells, puzzle for givens
                given_mask = puzzle > 0
                board_state = mx.where(given_mask, puzzle, predictions)
                board_state = board_state.astype(mx.int32)

                h = self.cell_embed(positions, board_state)

        # Final prediction with constraint propagation
        logits = self.predict(h)

        if self.sc_config.constraint_propagation:
            logits = self.apply_constraint_propagation(logits, puzzle)
        else:
            valid = self.compute_validity_mask(puzzle, None, None, False)
            logits = logits + mx.log(valid + 1e-10)

        predictions = mx.argmax(logits, axis=-1) + 1

        return {
            "logits": logits,
            "predictions": predictions,
            "final_h": h,
        }

    def solve_multi_pass(
        self,
        puzzle: mx.array,
        num_passes: int = 3,
    ) -> Dict[str, mx.array]:
        """
        Multi-pass inference: run model multiple times, using
        confident predictions to inform next pass.
        """
        B = puzzle.shape[0]
        current_board = mx.array(puzzle)

        for pass_idx in range(num_passes):
            # Run standard solve
            result = super().solve(current_board)
            logits = result['logits']

            # Apply constraints
            valid = self.compute_validity_mask(current_board, None, None, False)
            masked_logits = logits + mx.log(valid + 1e-10)

            probs = mx.softmax(masked_logits, axis=-1)
            predictions = mx.argmax(probs, axis=-1) + 1
            confidence = mx.max(probs, axis=-1)

            if pass_idx < num_passes - 1:
                # For intermediate passes, lock in high-confidence predictions
                threshold = self.sc_config.confidence_threshold
                high_conf = confidence > threshold
                empty_cells = current_board == 0
                should_lock = high_conf & empty_cells

                current_board = mx.where(should_lock, predictions, current_board)

        return {
            "logits": masked_logits,
            "predictions": predictions,
            "final_h": result.get('final_h'),
        }

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
        method: str = "basic",  # "iterative", "multi_pass", "basic", "raw"
    ) -> Dict[str, mx.array]:
        """
        Solve puzzle using specified method.

        Args:
            method: "basic" (guards at end only), "iterative" (guards each H-cycle),
                   "multi_pass" (multiple inference passes), "raw" (no guards at all)
        """
        if method == "iterative" and self.sc_config.iterative_guards:
            return self.solve_with_iterative_guards(puzzle, H_cycles, L_cycles)
        elif method == "multi_pass" and self.sc_config.multi_pass > 1:
            return self.solve_multi_pass(puzzle, self.sc_config.multi_pass)
        elif method == "raw":
            # No guards - for training
            return super().solve(puzzle, H_cycles, L_cycles, return_trajectory)
        else:
            # Basic: apply guards at end only
            result = super().solve(puzzle, H_cycles, L_cycles, return_trajectory)
            valid = self.compute_validity_mask(puzzle, None, None, False)
            result['logits'] = result['logits'] + mx.log(valid + 1e-10)
            result['predictions'] = mx.argmax(result['logits'], axis=-1) + 1
            return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        use_stablemax: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Override loss to use raw solve (no guards during training).
        """
        # Call solve with method="raw" to avoid guard application
        result = self.solve(puzzle, return_trajectory=True, method="raw")
        logits = result['logits']  # [B, 81, 9]

        # Target: shift to 0-8 for cross-entropy
        target = solution - 1

        # Compute probabilities
        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            probs = mx.softmax(logits, axis=-1)

        # Cross-entropy loss
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)
        base_loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Compute metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            'loss': base_loss,
            'cell_accuracy': cell_accuracy,
            'exact_accuracy': exact_accuracy,
        }

        return base_loss, metrics


def test_enhanced_sc_trm():
    """Test enhanced SC-TRM variants."""
    print("=" * 60)
    print("Testing Enhanced SC-TRM")
    print("=" * 60)

    # Test data
    B = 4
    puzzle = mx.zeros((B, 81), dtype=mx.int32)
    # Set some givens
    puzzle = puzzle.at[:, 0].set(1)  # Top-left is 1
    puzzle = puzzle.at[:, 10].set(2)  # (1,1) is 2

    solution = mx.ones((B, 81), dtype=mx.int32)

    # Test iterative guards
    print("\n1. Testing iterative guards...")
    config = EnhancedSCConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=2,
        iterative_guards=True,
        constraint_propagation=True,
    )
    model = EnhancedSCTRM(config)

    result = model.solve(puzzle, method="iterative")
    print(f"   Predictions shape: {result['predictions'].shape}")
    print(f"   Sample prediction: {result['predictions'][0, :9].tolist()}")

    # Test multi-pass
    print("\n2. Testing multi-pass inference...")
    config2 = EnhancedSCConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=2,
        multi_pass=3,
        confidence_threshold=0.9,
    )
    model2 = EnhancedSCTRM(config2)

    result2 = model2.solve(puzzle, method="multi_pass")
    print(f"   Predictions shape: {result2['predictions'].shape}")

    # Test constraint propagation
    print("\n3. Testing constraint propagation...")
    logits = mx.random.normal((B, 81, 9))
    cp_logits = model.apply_constraint_propagation(logits, puzzle)
    print(f"   Propagated logits shape: {cp_logits.shape}")

    print("\n" + "=" * 60)
    print("Enhanced SC-TRM tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_enhanced_sc_trm()
