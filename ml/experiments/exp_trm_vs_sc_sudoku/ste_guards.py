"""
Straight-Through Estimator (STE) Guards for SC-TRM.

Key insight: Hard masks block gradients during backprop.
Solution: Forward uses hard mask, backward uses soft approximation.

This allows:
- Hard constraint satisfaction at inference (100% validity)
- Smooth gradient flow during training (better learning)

Based on quantization literature:
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons"
- Courbariaux et al., "BinaryConnect: Training Deep Neural Networks with binary weights"
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig, stablemax


@dataclass
class STEConfig:
    """Configuration for STE-augmented SC-TRM."""
    # Base TRM config
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    num_cells: int = 81
    num_digits: int = 9
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1
    use_halting: bool = True

    # STE options
    soft_epsilon: float = 0.1  # How soft the backward mask is (0.1 = 90% hard)
    use_ste_training: bool = True  # Use STE during training
    hard_inference: bool = True  # Always use hard guards at inference


class STEGuard(nn.Module):
    """
    Straight-Through Estimator for constraint guards.

    Forward pass: Hard masking (log(valid) added to logits)
    Backward pass: Soft approximation (gradients flow through soft mask)
    """

    def __init__(self, soft_epsilon: float = 0.1):
        super().__init__()
        self.soft_epsilon = soft_epsilon

    def __call__(self, logits: mx.array, valid_mask: mx.array, training: bool = True) -> mx.array:
        """
        Apply STE guard to logits.

        Args:
            logits: [B, 81, 9] raw logits from model
            valid_mask: [B, 81, 9] binary mask (1=valid, 0=invalid)
            training: Whether in training mode

        Returns:
            Masked logits with STE for gradient flow
        """
        if not training:
            # Inference: always hard mask
            return logits + mx.log(valid_mask + 1e-10)

        # Training: STE trick
        # Forward: hard masking
        hard_masked = logits + mx.log(valid_mask + 1e-10)

        # Backward: soft approximation
        # valid_mask * (1 - epsilon) + epsilon = soft version
        soft_mask = valid_mask * (1 - self.soft_epsilon) + self.soft_epsilon
        soft_masked = logits + mx.log(soft_mask)

        # STE: forward uses hard, backward uses soft
        # Achieved via: hard - stop_gradient(hard - soft)
        # This makes forward = hard, but gradient flows through soft
        return hard_masked - mx.stop_gradient(hard_masked - soft_masked)


class STETRM(VanillaTRM):
    """
    TRM with Straight-Through Estimator guards.

    Guards are applied:
    - During training: STE (hard forward, soft backward)
    - During inference: Hard masks only
    """

    def __init__(self, config: Optional[STEConfig] = None):
        self.ste_config = config or STEConfig()

        # Initialize parent VanillaTRM
        vanilla_config = VanillaTRMConfig(
            hidden_dim=self.ste_config.hidden_dim,
            num_heads=self.ste_config.num_heads,
            num_layers=self.ste_config.num_layers,
            ff_dim=self.ste_config.ff_dim,
            num_cells=self.ste_config.num_cells,
            num_digits=self.ste_config.num_digits,
            H_cycles=self.ste_config.H_cycles,
            L_cycles=self.ste_config.L_cycles,
            dropout=self.ste_config.dropout,
            use_halting=self.ste_config.use_halting,
        )
        super().__init__(vanilla_config)

        # STE guard module
        self.ste_guard = STEGuard(soft_epsilon=self.ste_config.soft_epsilon)

    def compute_validity_mask(self, puzzle: mx.array) -> mx.array:
        """
        Compute validity mask from puzzle givens.

        For each cell, a digit is invalid if it already appears
        in the same row, column, or 3x3 box.

        Args:
            puzzle: [B, 81] puzzle with givens (0 = empty)

        Returns:
            [B, 81, 9] validity mask (1=valid, 0=blocked)
        """
        B = puzzle.shape[0]
        board = puzzle.reshape(B, 9, 9)  # [B, 9, 9]

        # Start with all valid
        valid = mx.ones((B, 81, 9))

        # For each cell position
        for cell in range(81):
            row, col = cell // 9, cell % 9
            box_r, box_c = (row // 3) * 3, (col // 3) * 3

            # Get givens in same row/col/box
            row_vals = board[:, row, :]  # [B, 9]
            col_vals = board[:, :, col]  # [B, 9]
            box_vals = board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)  # [B, 9]

            # For each digit, check if blocked
            for d in range(9):
                digit = d + 1

                # Check if digit appears in row/col/box
                in_row = mx.any(row_vals == digit, axis=1)  # [B]
                in_col = mx.any(col_vals == digit, axis=1)  # [B]
                in_box = mx.any(box_vals == digit, axis=1)  # [B]

                blocked = in_row | in_col | in_box  # [B]

                # Update validity mask
                # valid[b, cell, d] = 0 if blocked[b]
                cell_idx = mx.array([cell])
                digit_idx = mx.array([d])

                # Expand blocked to match valid shape
                blocked_expanded = blocked[:, None, None]  # [B, 1, 1]
                cell_match = mx.arange(81)[None, :, None] == cell  # [1, 81, 1]
                digit_match = mx.arange(9)[None, None, :] == d  # [1, 1, 9]

                mask = blocked_expanded & cell_match & digit_match  # [B, 81, 9]
                valid = mx.where(mask, mx.zeros_like(valid), valid)

        return valid

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
        use_guards: bool = True,
        training: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Solve puzzle with optional STE guards.

        Args:
            puzzle: [B, 81] puzzle with givens
            use_guards: Whether to apply constraint guards
            training: Whether in training mode (affects STE behavior)
        """
        # Get base result from parent
        result = super().solve(puzzle, H_cycles, L_cycles, return_trajectory)

        if not use_guards:
            return result

        # Compute validity mask
        valid_mask = self.compute_validity_mask(puzzle)

        # Apply STE guard
        logits = result['logits']
        masked_logits = self.ste_guard(logits, valid_mask, training=training)

        # Update predictions
        predictions = mx.argmax(masked_logits, axis=-1) + 1

        result['logits'] = masked_logits
        result['predictions'] = predictions
        result['valid_mask'] = valid_mask

        return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        use_stablemax: bool = True,
        use_guards: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute loss with STE guards during training.

        The STE allows gradients to flow through the guard masking,
        enabling the model to learn WITH constraint awareness.
        """
        # Solve with guards in training mode
        result = self.solve(
            puzzle,
            return_trajectory=True,
            use_guards=use_guards and self.ste_config.use_ste_training,
            training=True,
        )
        logits = result['logits']  # [B, 81, 9]

        # Target: shift to 0-8 for cross-entropy
        target = solution - 1

        # Clip logits to prevent extreme values
        logits = mx.clip(logits, -30, 30)

        # Compute probabilities with numerical stability
        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            # Numerically stable softmax
            logits_max = mx.max(logits, axis=-1, keepdims=True)
            logits_shifted = logits - logits_max
            exp_logits = mx.exp(logits_shifted)
            probs = exp_logits / (mx.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Cross-entropy loss with clamping
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)
        # Clamp probabilities to prevent log(0)
        correct_probs = mx.clip(correct_probs, 1e-7, 1.0)
        base_loss = -mx.mean(mx.log(correct_probs))

        # Check for NaN and replace with high loss
        base_loss = mx.where(mx.isnan(base_loss), mx.array(10.0), base_loss)

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


def train_and_evaluate(
    model,
    train_p,
    train_s,
    test_p,
    test_s,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    use_guards=True,
    print_every=5,
):
    """Train model and return metrics."""
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s, use_guards=use_guards)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    import time
    start = time.time()
    best_cell_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, train_p.shape[0], batch_size):
            p = train_p_shuf[i:i+batch_size]
            s = train_s_shuf[i:i+batch_size]

            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += float(loss.item())
            n_batches += 1

        # Periodic evaluation
        if epoch % print_every == 0 or epoch == epochs:
            # Evaluate with hard guards (inference mode)
            result = model.solve(test_p[:100], use_guards=True, training=False)
            pred = result['predictions']
            mx.eval(pred)

            cell_acc = float(mx.mean((pred == test_s[:100]).astype(mx.float32)).item())
            exact_acc = float(mx.mean(mx.all(pred == test_s[:100], axis=1).astype(mx.float32)).item())
            best_cell_acc = max(best_cell_acc, cell_acc)

            print(f"  Epoch {epoch:4d}: loss={total_loss/n_batches:.4f}, "
                  f"cell_acc={cell_acc:.1%}, exact_acc={exact_acc:.1%}")

    elapsed = time.time() - start

    # Final evaluation
    result = model.solve(test_p, use_guards=True, training=False)
    pred = result['predictions']
    mx.eval(pred)

    final_cell_acc = float(mx.mean((pred == test_s).astype(mx.float32)).item())
    final_exact_acc = float(mx.mean(mx.all(pred == test_s, axis=1).astype(mx.float32)).item())

    return {
        'cell_acc': final_cell_acc,
        'exact_acc': final_exact_acc,
        'best_cell_acc': best_cell_acc,
        'time': elapsed,
    }


def compare_ste_vs_inference_guards():
    """Compare STE guards (training) vs inference-only guards."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 70)
    print("STE GUARDS vs INFERENCE-ONLY GUARDS")
    print("=" * 70)

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(1000, 200, seed=42)

    # 1. Vanilla TRM (no guards)
    print("\n" + "-" * 70)
    print("1. VANILLA TRM (no guards)")
    print("-" * 70)

    vanilla_config = VanillaTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
    )
    vanilla_model = VanillaTRM(vanilla_config)

    vanilla_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def vanilla_loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    vanilla_loss_and_grad = nn.value_and_grad(vanilla_model, vanilla_loss_fn)

    import time
    start = time.time()
    for epoch in range(30):
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        for i in range(0, train_p.shape[0], 32):
            p = train_p_shuf[i:i+32]
            s = train_s_shuf[i:i+32]
            loss, grads = vanilla_loss_and_grad(vanilla_model, p, s)
            vanilla_optimizer.update(vanilla_model, grads)
            mx.eval(vanilla_model.parameters())

        if (epoch + 1) % 10 == 0:
            result = vanilla_model.solve(test_p[:100])
            pred = result['predictions']
            mx.eval(pred)
            cell_acc = float(mx.mean((pred == test_s[:100]).astype(mx.float32)).item())
            print(f"  Epoch {epoch+1}: cell_acc={cell_acc:.1%}")

    vanilla_time = time.time() - start
    vanilla_result = vanilla_model.solve(test_p)
    vanilla_pred = vanilla_result['predictions']
    mx.eval(vanilla_pred)
    vanilla_cell_acc = float(mx.mean((vanilla_pred == test_s).astype(mx.float32)).item())

    # 2. STE-TRM (guards during training via STE)
    print("\n" + "-" * 70)
    print("2. STE-TRM (guards during training via STE)")
    print("-" * 70)

    ste_config = STEConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        soft_epsilon=0.1,
        use_ste_training=True,
    )
    ste_model = STETRM(ste_config)

    ste_result = train_and_evaluate(
        ste_model, train_p, train_s, test_p, test_s,
        epochs=30, use_guards=True, print_every=10
    )

    # 3. Inference-only guards (train without, apply at test)
    print("\n" + "-" * 70)
    print("3. INFERENCE-ONLY GUARDS (train without, apply at test)")
    print("-" * 70)

    inf_config = STEConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        use_ste_training=False,  # No guards during training
    )
    inf_model = STETRM(inf_config)

    # Train WITHOUT guards
    inf_result_no_guards = train_and_evaluate(
        inf_model, train_p, train_s, test_p, test_s,
        epochs=30, use_guards=False, print_every=10
    )

    # Now evaluate WITH guards (inference only)
    final_result = inf_model.solve(test_p, use_guards=True, training=False)
    final_pred = final_result['predictions']
    mx.eval(final_pred)
    inf_cell_acc_with_guards = float(mx.mean((final_pred == test_s).astype(mx.float32)).item())

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Cell Acc':>10} {'Time':>10}")
    print("-" * 70)
    print(f"{'Vanilla TRM (no guards)':<35} {vanilla_cell_acc:>9.1%} {vanilla_time:>9.1f}s")
    print(f"{'STE-TRM (guards via STE)':<35} {ste_result['cell_acc']:>9.1%} {ste_result['time']:>9.1f}s")
    print(f"{'Inference-only guards':<35} {inf_cell_acc_with_guards:>9.1%} {inf_result_no_guards['time']:>9.1f}s")
    print("-" * 70)

    print("\nKey findings:")
    ste_vs_vanilla = ste_result['cell_acc'] - vanilla_cell_acc
    inf_vs_vanilla = inf_cell_acc_with_guards - vanilla_cell_acc
    ste_vs_inf = ste_result['cell_acc'] - inf_cell_acc_with_guards

    print(f"  STE vs Vanilla: {ste_vs_vanilla:+.1%}")
    print(f"  Inference-only vs Vanilla: {inf_vs_vanilla:+.1%}")
    print(f"  STE vs Inference-only: {ste_vs_inf:+.1%}")


def test_ste_gradient_flow():
    """Verify that STE allows gradients to flow."""
    print("=" * 60)
    print("Testing STE Gradient Flow")
    print("=" * 60)

    # Create simple test case
    B = 4
    logits = mx.random.normal((B, 81, 9))
    valid_mask = mx.ones((B, 81, 9))
    # Block some digits using indexing
    # Create mask for blocked positions
    block_mask = mx.zeros((B, 81, 9))
    # Block digit 0 and 1 for cell 0
    block_mask = block_mask + (
        (mx.arange(81)[None, :, None] == 0) &
        ((mx.arange(9)[None, None, :] == 0) | (mx.arange(9)[None, None, :] == 1))
    ).astype(mx.float32)
    valid_mask = valid_mask * (1 - block_mask)

    ste_guard = STEGuard(soft_epsilon=0.1)

    # Test forward pass
    print("\n1. Testing forward pass...")
    masked_training = ste_guard(logits, valid_mask, training=True)
    masked_inference = ste_guard(logits, valid_mask, training=False)

    print(f"   Logits[0,0,:3]: {logits[0, 0, :3].tolist()}")
    print(f"   Valid mask[0,0,:3]: {valid_mask[0, 0, :3].tolist()}")
    print(f"   Masked (training)[0,0,:3]: {masked_training[0, 0, :3].tolist()}")
    print(f"   Masked (inference)[0,0,:3]: {masked_inference[0, 0, :3].tolist()}")

    # Test gradient flow
    print("\n2. Testing gradient flow...")

    def loss_fn(logits_in):
        masked = ste_guard(logits_in, valid_mask, training=True)
        return mx.mean(masked)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(logits)
    mx.eval(grads)

    # Check that gradients exist and are non-zero
    grad_norm = float(mx.sqrt(mx.sum(grads * grads)).item())
    print(f"   Gradient norm: {grad_norm:.6f}")
    print(f"   Gradient at blocked position [0,0,0]: {grads[0, 0, 0].item():.6f}")
    print(f"   Gradient at valid position [0,0,2]: {grads[0, 0, 2].item():.6f}")

    # Without STE, gradient at blocked position would be 0
    print("\n3. Comparison with hard masking (no STE)...")

    def hard_loss_fn(logits_in):
        masked = logits_in + mx.log(valid_mask + 1e-10)  # Hard masking
        return mx.mean(masked)

    hard_grad_fn = mx.grad(hard_loss_fn)
    hard_grads = hard_grad_fn(logits)
    mx.eval(hard_grads)

    hard_grad_norm = float(mx.sqrt(mx.sum(hard_grads * hard_grads)).item())
    print(f"   Hard gradient norm: {hard_grad_norm:.6f}")
    print(f"   Hard gradient at blocked [0,0,0]: {hard_grads[0, 0, 0].item():.6f}")

    print("\n" + "=" * 60)
    print("STE gradient flow test complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['gradient', 'compare', 'all'], default='all')
    args = parser.parse_args()

    if args.test in ['gradient', 'all']:
        test_ste_gradient_flow()

    if args.test in ['compare', 'all']:
        compare_ste_vs_inference_guards()
