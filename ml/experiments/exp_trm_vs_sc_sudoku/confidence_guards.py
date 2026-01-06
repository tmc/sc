"""
Confidence-Weighted Dynamic Guards for SC-TRM.

Key insight: Guard strength should depend on model confidence.
- Low confidence (high entropy) → Soft guards (exploration)
- High confidence (low entropy) → Hard guards (exploitation)

This creates self-regulating constraints that adapt during inference.
The model "earns" hard constraints by being confident.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig, stablemax
from experiments.exp_trm_vs_sc_sudoku.ste_guards import STETRM, STEConfig


@dataclass
class ConfidenceGuardConfig:
    """Configuration for confidence-weighted SC-TRM."""
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

    # Confidence guard options
    confidence_temperature: float = 5.0  # Controls sharpness of confidence→strength mapping
    confidence_threshold: float = 0.5  # Midpoint for sigmoid
    min_guard_strength: float = 0.1  # Minimum guard strength (always some softness)
    max_guard_strength: float = 1.0  # Maximum guard strength (full hard)

    # Training options
    use_guards_training: bool = True  # Apply confidence guards during training
    soft_epsilon: float = 0.1  # For STE-style backward pass


class ConfidenceGuard(nn.Module):
    """
    Confidence-weighted constraint guard.

    Guard strength = sigmoid((confidence - threshold) * temperature)
    Where confidence = 1 - normalized_entropy

    High confidence → strong guards (near-hard masking)
    Low confidence → weak guards (soft masking, exploration)
    """

    def __init__(
        self,
        temperature: float = 5.0,
        threshold: float = 0.5,
        min_strength: float = 0.1,
        max_strength: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.min_strength = min_strength
        self.max_strength = max_strength

    def compute_confidence(self, logits: mx.array) -> mx.array:
        """
        Compute per-cell confidence from logits.

        Confidence = 1 - normalized_entropy
        Where normalized_entropy = H(p) / log(9)

        Returns:
            [B, 81] confidence scores in [0, 1]
        """
        # Stable softmax
        logits_max = mx.max(logits, axis=-1, keepdims=True)
        logits_shifted = logits - logits_max
        exp_logits = mx.exp(logits_shifted)
        probs = exp_logits / (mx.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Compute entropy: H(p) = -sum(p * log(p))
        # Clip probs to prevent log(0)
        probs_clipped = mx.clip(probs, 1e-10, 1.0)
        entropy = -mx.sum(probs * mx.log(probs_clipped), axis=-1)  # [B, 81]

        # Normalize by max entropy (log(9) for 9 classes)
        max_entropy = mx.log(mx.array(9.0))
        normalized_entropy = entropy / max_entropy

        # Confidence = 1 - normalized_entropy
        confidence = 1.0 - normalized_entropy

        return confidence

    def compute_guard_strength(self, confidence: mx.array) -> mx.array:
        """
        Map confidence to guard strength via sigmoid.

        strength = min + (max - min) * sigmoid((conf - thresh) * temp)

        Returns:
            [B, 81] guard strengths in [min_strength, max_strength]
        """
        # Sigmoid mapping
        sigmoid_input = (confidence - self.threshold) * self.temperature
        sigmoid_output = mx.sigmoid(sigmoid_input)

        # Scale to [min, max]
        strength = self.min_strength + (self.max_strength - self.min_strength) * sigmoid_output

        return strength

    def __call__(
        self,
        logits: mx.array,
        valid_mask: mx.array,
        training: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply confidence-weighted guards.

        Args:
            logits: [B, 81, 9] raw logits
            valid_mask: [B, 81, 9] validity mask
            training: Whether in training mode

        Returns:
            Tuple of (masked_logits, guard_strengths)
        """
        # Compute confidence and guard strength
        confidence = self.compute_confidence(logits)  # [B, 81]
        strength = self.compute_guard_strength(confidence)  # [B, 81]

        # Expand strength to match logits shape
        strength_expanded = strength[:, :, None]  # [B, 81, 1]

        # Interpolate between soft and hard masking
        # Soft: valid_mask * 0.9 + 0.1 (always some probability)
        # Hard: valid_mask + epsilon (near-zero for invalid)
        soft_mask = valid_mask * 0.9 + 0.1
        hard_mask = valid_mask + 1e-10

        # Interpolate: strength * hard + (1 - strength) * soft
        interpolated_mask = strength_expanded * hard_mask + (1 - strength_expanded) * soft_mask

        # Apply as log-mask
        masked_logits = logits + mx.log(interpolated_mask)

        # Clip to prevent extreme values
        masked_logits = mx.clip(masked_logits, -30, 30)

        return masked_logits, strength


class ConfidenceTRM(VanillaTRM):
    """
    TRM with confidence-weighted dynamic guards.
    """

    def __init__(self, config: Optional[ConfidenceGuardConfig] = None):
        self.conf_config = config or ConfidenceGuardConfig()

        # Initialize parent VanillaTRM
        vanilla_config = VanillaTRMConfig(
            hidden_dim=self.conf_config.hidden_dim,
            num_heads=self.conf_config.num_heads,
            num_layers=self.conf_config.num_layers,
            ff_dim=self.conf_config.ff_dim,
            num_cells=self.conf_config.num_cells,
            num_digits=self.conf_config.num_digits,
            H_cycles=self.conf_config.H_cycles,
            L_cycles=self.conf_config.L_cycles,
            dropout=self.conf_config.dropout,
            use_halting=self.conf_config.use_halting,
        )
        super().__init__(vanilla_config)

        # Confidence guard module
        self.confidence_guard = ConfidenceGuard(
            temperature=self.conf_config.confidence_temperature,
            threshold=self.conf_config.confidence_threshold,
            min_strength=self.conf_config.min_guard_strength,
            max_strength=self.conf_config.max_guard_strength,
        )

    def compute_validity_mask(self, puzzle: mx.array) -> mx.array:
        """Compute validity mask from puzzle givens."""
        B = puzzle.shape[0]
        board = puzzle.reshape(B, 9, 9)

        valid = mx.ones((B, 81, 9))

        for cell in range(81):
            row, col = cell // 9, cell % 9
            box_r, box_c = (row // 3) * 3, (col // 3) * 3

            row_vals = board[:, row, :]
            col_vals = board[:, :, col]
            box_vals = board[:, box_r:box_r+3, box_c:box_c+3].reshape(B, 9)

            for d in range(9):
                digit = d + 1
                in_row = mx.any(row_vals == digit, axis=1)
                in_col = mx.any(col_vals == digit, axis=1)
                in_box = mx.any(box_vals == digit, axis=1)
                blocked = in_row | in_col | in_box

                blocked_expanded = blocked[:, None, None]
                cell_match = mx.arange(81)[None, :, None] == cell
                digit_match = mx.arange(9)[None, None, :] == d
                mask = blocked_expanded & cell_match & digit_match
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
        """Solve with confidence-weighted guards."""
        result = super().solve(puzzle, H_cycles, L_cycles, return_trajectory)

        if not use_guards:
            return result

        valid_mask = self.compute_validity_mask(puzzle)
        logits = result['logits']

        masked_logits, guard_strengths = self.confidence_guard(
            logits, valid_mask, training=training
        )

        predictions = mx.argmax(masked_logits, axis=-1) + 1

        result['logits'] = masked_logits
        result['predictions'] = predictions
        result['valid_mask'] = valid_mask
        result['guard_strengths'] = guard_strengths

        return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        use_stablemax: bool = True,
        use_guards: bool = True,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute loss with confidence guards."""
        result = self.solve(
            puzzle,
            return_trajectory=True,
            use_guards=use_guards and self.conf_config.use_guards_training,
            training=True,
        )
        logits = result['logits']

        target = solution - 1
        logits = mx.clip(logits, -30, 30)

        if use_stablemax:
            probs = stablemax(logits, axis=-1)
        else:
            logits_max = mx.max(logits, axis=-1, keepdims=True)
            logits_shifted = logits - logits_max
            exp_logits = mx.exp(logits_shifted)
            probs = exp_logits / (mx.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1)
        correct_probs = correct_probs.squeeze(-1)
        correct_probs = mx.clip(correct_probs, 1e-7, 1.0)
        base_loss = -mx.mean(mx.log(correct_probs))
        base_loss = mx.where(mx.isnan(base_loss), mx.array(10.0), base_loss)

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


def compare_confidence_guards():
    """Compare confidence guards with other approaches."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku
    import time

    print("=" * 70)
    print("CONFIDENCE-WEIGHTED GUARDS COMPARISON")
    print("=" * 70)

    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(1000, 200, seed=42)

    def train_model(model, train_p, train_s, epochs=30, use_guards=True, model_type='vanilla'):
        optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

        def loss_fn(m, p, s):
            if model_type == 'vanilla':
                loss, _ = m.loss(p, s)
            else:
                loss, _ = m.loss(p, s, use_guards=use_guards)
            return loss

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        start = time.time()
        for epoch in range(epochs):
            perm = mx.random.permutation(train_p.shape[0])
            train_p_shuf = train_p[perm]
            train_s_shuf = train_s[perm]

            for i in range(0, train_p.shape[0], 32):
                p = train_p_shuf[i:i+32]
                s = train_s_shuf[i:i+32]
                loss, grads = loss_and_grad(model, p, s)
                optimizer.update(model, grads)
                mx.eval(model.parameters())

            if (epoch + 1) % 10 == 0:
                if model_type == 'vanilla':
                    result = model.solve(test_p[:100])
                else:
                    result = model.solve(test_p[:100], use_guards=True, training=False)
                pred = result['predictions']
                mx.eval(pred)
                cell_acc = float(mx.mean((pred == test_s[:100]).astype(mx.float32)).item())
                print(f"  Epoch {epoch+1}: cell_acc={cell_acc:.1%}")

        elapsed = time.time() - start
        return elapsed

    results = {}

    # 1. Vanilla TRM
    print("\n" + "-" * 70)
    print("1. VANILLA TRM (baseline)")
    print("-" * 70)

    vanilla_config = VanillaTRMConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
    )
    vanilla_model = VanillaTRM(vanilla_config)
    vanilla_time = train_model(vanilla_model, train_p, train_s, use_guards=False, model_type='vanilla')

    vanilla_result = vanilla_model.solve(test_p)
    vanilla_pred = vanilla_result['predictions']
    mx.eval(vanilla_pred)
    vanilla_acc = float(mx.mean((vanilla_pred == test_s).astype(mx.float32)).item())
    results['vanilla'] = {'acc': vanilla_acc, 'time': vanilla_time}

    # 2. STE-TRM
    print("\n" + "-" * 70)
    print("2. STE-TRM (guards via STE)")
    print("-" * 70)

    ste_config = STEConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4, soft_epsilon=0.1, use_ste_training=True,
    )
    ste_model = STETRM(ste_config)
    ste_time = train_model(ste_model, train_p, train_s, use_guards=True, model_type='ste')

    ste_result = ste_model.solve(test_p, use_guards=True, training=False)
    ste_pred = ste_result['predictions']
    mx.eval(ste_pred)
    ste_acc = float(mx.mean((ste_pred == test_s).astype(mx.float32)).item())
    results['ste'] = {'acc': ste_acc, 'time': ste_time}

    # 3. Confidence-weighted guards
    print("\n" + "-" * 70)
    print("3. CONFIDENCE-TRM (confidence-weighted guards)")
    print("-" * 70)

    conf_config = ConfidenceGuardConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
        confidence_temperature=5.0,
        confidence_threshold=0.5,
        min_guard_strength=0.1,
        max_guard_strength=1.0,
        use_guards_training=True,
    )
    conf_model = ConfidenceTRM(conf_config)
    conf_time = train_model(conf_model, train_p, train_s, use_guards=True, model_type='confidence')

    conf_result = conf_model.solve(test_p, use_guards=True, training=False)
    conf_pred = conf_result['predictions']
    mx.eval(conf_pred)
    conf_acc = float(mx.mean((conf_pred == test_s).astype(mx.float32)).item())
    results['confidence'] = {'acc': conf_acc, 'time': conf_time}

    # Analyze guard strengths
    guard_strengths = conf_result['guard_strengths']
    mx.eval(guard_strengths)
    mean_strength = float(mx.mean(guard_strengths).item())
    std_strength = float(mx.std(guard_strengths).item())

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Cell Acc':>10} {'Time':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['acc']:>9.1%} {r['time']:>9.1f}s")
    print("-" * 70)

    print(f"\nConfidence guard statistics:")
    print(f"  Mean strength: {mean_strength:.3f}")
    print(f"  Std strength: {std_strength:.3f}")

    print("\nImprovements vs Vanilla:")
    for name, r in results.items():
        if name != 'vanilla':
            diff = r['acc'] - results['vanilla']['acc']
            print(f"  {name}: {diff:+.1%}")


def analyze_confidence_adaptation():
    """Analyze how confidence adapts during inference."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 60)
    print("CONFIDENCE ADAPTATION ANALYSIS")
    print("=" * 60)

    # Generate a small test set
    _, _, test_p, test_s = generate_random_sudoku(100, 20, seed=42)

    # Train model
    print("\nTraining model...")
    train_p, train_s, _, _ = generate_random_sudoku(500, 100, seed=123)

    conf_config = ConfidenceGuardConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
    )
    model = ConfidenceTRM(conf_config)

    optimizer = optim.AdamW(learning_rate=1e-4)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(20):
        perm = mx.random.permutation(train_p.shape[0])
        for i in range(0, train_p.shape[0], 32):
            p = train_p[perm][i:i+32]
            s = train_s[perm][i:i+32]
            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

    # Analyze confidence on test puzzles
    print("\nAnalyzing confidence patterns...")

    result = model.solve(test_p, use_guards=True, training=False)
    predictions = result['predictions']
    guard_strengths = result['guard_strengths']
    mx.eval(predictions, guard_strengths)

    # Compute correctness per cell
    correct = (predictions == test_s).astype(mx.float32)  # [B, 81]

    # Correlation between confidence and correctness
    strengths_flat = guard_strengths.reshape(-1)
    correct_flat = correct.reshape(-1)
    mx.eval(strengths_flat, correct_flat)

    # Bin by strength and compute accuracy
    print("\nAccuracy by confidence level:")
    print(f"{'Strength Range':<20} {'Accuracy':>10} {'Count':>10}")
    print("-" * 50)

    strength_np = strengths_flat.tolist()
    correct_np = correct_flat.tolist()

    for low, high in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        mask = [(low <= s < high) for s in strength_np]
        count = sum(mask)
        if count > 0:
            acc = sum(c for c, m in zip(correct_np, mask) if m) / count
            print(f"[{low:.1f}, {high:.1f}){'':<12} {acc:>9.1%} {count:>10}")

    # Overall stats
    overall_acc = float(mx.mean(correct).item())
    print(f"\nOverall cell accuracy: {overall_acc:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['compare', 'analyze', 'all'], default='all')
    args = parser.parse_args()

    if args.test in ['compare', 'all']:
        compare_confidence_guards()

    if args.test in ['analyze', 'all']:
        analyze_confidence_adaptation()
