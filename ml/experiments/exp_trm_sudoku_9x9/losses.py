"""
Loss functions for Sudoku statechart training.

Includes stablemax cross-entropy from TinyRecursiveModels, which is more
numerically stable than standard softmax for iterative refinement.
"""

import mlx.core as mx


def stablemax(logits: mx.array, axis: int = -1) -> mx.array:
    """
    Stablemax: numerically stable alternative to softmax.

    From TRM paper:
    s(x) = 1/(1-x) if x < 0 else x + 1

    Properties:
    - Bounded: s(x) >= 1 for all x
    - Monotonic: s(x1) < s(x2) if x1 < x2
    - Stable: No exp() overflow for large positive values
    - Differentiable everywhere

    Args:
        logits: [*, C] logits
        axis: Axis to normalize over

    Returns:
        probs: [*, C] normalized probabilities
    """
    # s(x) = 1/(1-x) if x < 0 else x + 1
    s = mx.where(
        logits < 0,
        1.0 / (1.0 - logits),
        logits + 1.0
    )

    # Normalize
    return s / mx.sum(s, axis=axis, keepdims=True)


def stablemax_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> mx.array:
    """
    Cross-entropy loss using stablemax instead of softmax.

    More numerically stable for iterative refinement where logits
    can vary widely across iterations.

    Args:
        logits: [B, ..., C] logits
        targets: [B, ...] integer class indices
        ignore_index: Index to ignore in loss computation
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value (scalar if reduction != "none")
    """
    # Get probabilities via stablemax
    probs = stablemax(logits, axis=-1)

    # Clamp for numerical stability
    probs = mx.clip(probs, 1e-7, 1.0 - 1e-7)

    # Gather the probability of the correct class
    # Shape: [B, ...]
    num_classes = logits.shape[-1]

    # Create mask for valid targets
    valid_mask = targets != ignore_index

    # Replace ignore_index with 0 for gathering (will be masked anyway)
    safe_targets = mx.where(valid_mask, targets, mx.zeros_like(targets))

    # One-hot encode and gather
    one_hot = mx.one_hot(safe_targets, num_classes)
    target_probs = mx.sum(probs * one_hot, axis=-1)

    # Negative log probability
    nll = -mx.log(target_probs)

    # Apply mask
    nll = mx.where(valid_mask, nll, mx.zeros_like(nll))

    # Reduction
    if reduction == "none":
        return nll
    elif reduction == "sum":
        return mx.sum(nll)
    elif reduction == "mean":
        num_valid = mx.sum(valid_mask.astype(mx.float32))
        return mx.sum(nll) / mx.maximum(num_valid, mx.array(1.0))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def softmax_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> mx.array:
    """
    Standard softmax cross-entropy for comparison.

    Args:
        logits: [B, ..., C] logits
        targets: [B, ...] integer class indices
        ignore_index: Index to ignore in loss computation
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Softmax probabilities
    probs = mx.softmax(logits, axis=-1)
    probs = mx.clip(probs, 1e-7, 1.0 - 1e-7)

    # Valid mask
    num_classes = logits.shape[-1]
    valid_mask = targets != ignore_index
    safe_targets = mx.where(valid_mask, targets, mx.zeros_like(targets))

    # Gather target probabilities
    one_hot = mx.one_hot(safe_targets, num_classes)
    target_probs = mx.sum(probs * one_hot, axis=-1)

    # NLL
    nll = -mx.log(target_probs)
    nll = mx.where(valid_mask, nll, mx.zeros_like(nll))

    # Reduction
    if reduction == "none":
        return nll
    elif reduction == "sum":
        return mx.sum(nll)
    elif reduction == "mean":
        num_valid = mx.sum(valid_mask.astype(mx.float32))
        return mx.sum(nll) / mx.maximum(num_valid, mx.array(1.0))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def sudoku_loss(
    logits: mx.array,
    targets: mx.array,
    use_stablemax: bool = True,
    constraint_weight: float = 0.0,
) -> mx.array:
    """
    Combined loss for Sudoku solving.

    Args:
        logits: [B, 81, 10] logits for each cell (10 = empty + 1-9)
        targets: [B, 81] target values (1-9)
        use_stablemax: Use stablemax vs softmax
        constraint_weight: Weight for constraint violation penalty

    Returns:
        Total loss
    """
    # Classification loss
    if use_stablemax:
        ce_loss = stablemax_cross_entropy(logits, targets)
    else:
        ce_loss = softmax_cross_entropy(logits, targets)

    # Constraint violation penalty (optional)
    if constraint_weight > 0:
        probs = stablemax(logits, axis=-1) if use_stablemax else mx.softmax(logits, axis=-1)
        constraint_loss = _constraint_violation_loss(probs)
        return ce_loss + constraint_weight * constraint_loss

    return ce_loss


def _constraint_violation_loss(probs: mx.array) -> mx.array:
    """
    Penalize constraint violations in predicted probabilities.

    For each row/col/box, penalize if multiple cells have high
    probability for the same digit.

    Args:
        probs: [B, 81, 10] soft state probabilities

    Returns:
        Constraint violation penalty
    """
    B = probs.shape[0]
    total_violation = mx.zeros((B,))

    # Digit probabilities (skip empty state)
    digit_probs = probs[:, :, 1:]  # [B, 81, 9]

    # Row constraints
    for row in range(9):
        row_cells = list(range(row * 9, (row + 1) * 9))
        row_probs = digit_probs[:, row_cells, :]  # [B, 9, 9]
        # Sum of probabilities per digit (should be <= 1 for valid)
        digit_sums = mx.sum(row_probs, axis=1)  # [B, 9]
        violation = mx.sum(mx.relu(digit_sums - 1.0), axis=-1)
        total_violation = total_violation + violation

    # Column constraints
    for col in range(9):
        col_cells = list(range(col, 81, 9))
        col_probs = digit_probs[:, col_cells, :]  # [B, 9, 9]
        digit_sums = mx.sum(col_probs, axis=1)  # [B, 9]
        violation = mx.sum(mx.relu(digit_sums - 1.0), axis=-1)
        total_violation = total_violation + violation

    # Box constraints
    for box_row in range(3):
        for box_col in range(3):
            box_cells = []
            for r in range(3):
                for c in range(3):
                    box_cells.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
            box_probs = digit_probs[:, box_cells, :]  # [B, 9, 9]
            digit_sums = mx.sum(box_probs, axis=1)  # [B, 9]
            violation = mx.sum(mx.relu(digit_sums - 1.0), axis=-1)
            total_violation = total_violation + violation

    return mx.mean(total_violation)


def test_losses():
    """Test loss functions."""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Create test data
    B, C = 4, 10
    logits = mx.random.normal((B, 81, C))
    targets = mx.random.randint(1, C, (B, 81))  # 1-9

    print("\n1. Testing stablemax...")
    probs = stablemax(logits)
    sums = mx.sum(probs, axis=-1)
    print(f"   Prob sums (should be 1): min={float(mx.min(sums)):.4f}, max={float(mx.max(sums)):.4f}")

    print("\n2. Testing stablemax cross-entropy...")
    loss_stable = stablemax_cross_entropy(logits, targets)
    print(f"   Stablemax CE loss: {float(loss_stable):.4f}")

    print("\n3. Testing softmax cross-entropy...")
    loss_softmax = softmax_cross_entropy(logits, targets)
    print(f"   Softmax CE loss: {float(loss_softmax):.4f}")

    print("\n4. Testing combined sudoku loss...")
    loss_combined = sudoku_loss(logits, targets, constraint_weight=0.1)
    print(f"   Combined loss: {float(loss_combined):.4f}")

    print("\n5. Testing gradient flow...")
    def loss_fn(logits):
        return stablemax_cross_entropy(logits, targets)

    loss, grads = mx.value_and_grad(loss_fn)(logits)
    has_grad = mx.any(grads != 0).item()
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")

    print("\n6. Testing with extreme logits...")
    extreme_logits = logits * 100  # Large values
    loss_extreme = stablemax_cross_entropy(extreme_logits, targets)
    is_finite = mx.isfinite(loss_extreme).item()
    print(f"   Extreme logits (×100): loss={float(loss_extreme):.4f}, finite={is_finite}")

    print("\n" + "=" * 60)
    print("Loss function tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
