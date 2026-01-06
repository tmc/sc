"""
Hook Injection: Steering vector injection into model forward pass.

Creates hooks compatible with mlux HookedModel.run_with_hooks() that
add steering vectors to hidden states at specified layers.

Hook signature for mlux:
    hook(args, output, wrapper) -> modified_output | None

The steering vector is added to the hidden states (residual stream)
to push generation towards desired behavior.
"""

import sys
from dataclasses import dataclass
from typing import Callable, Any, Optional, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


# =============================================================================
# STEERING HOOK TYPES
# =============================================================================

@dataclass
class SteeringConfig:
    """Configuration for steering hook."""
    layer: int
    alpha: float = 1.0
    position: str = "last"  # "last", "all", or specific index
    normalize: bool = False


# =============================================================================
# HOOK CREATION
# =============================================================================

def create_steering_hook(
    steering_vector: Any,
    alpha: float = 1.0,
    position: str = "last",
    normalize: bool = False,
) -> Callable:
    """
    Create a steering hook function for mlux.

    Args:
        steering_vector: Vector to add to hidden states (shape: hidden_dim)
        alpha: Scaling factor for steering strength
        position: Where to apply steering:
            - "last": Only last token position
            - "all": All positions
            - int: Specific position index
        normalize: Whether to normalize the steering vector

    Returns:
        Hook function compatible with mlux run_with_hooks
    """
    # Prepare steering vector
    vec = steering_vector
    if normalize and MLX_AVAILABLE:
        norm = mx.sqrt(mx.sum(vec * vec))
        vec = vec / (norm + 1e-8)

    # Scale by alpha
    scaled_vec = vec * alpha if MLX_AVAILABLE else vec

    def hook(args: tuple, output: Any, wrapper: Any) -> Any:
        """
        Steering hook that modifies layer output.

        mlux hook signature: hook(args, output, wrapper) -> modified_output
        - args: Input arguments to the layer
        - output: Layer output (hidden states)
        - wrapper: HookWrapper with metadata
        """
        if output is None:
            return None

        # output shape: (batch, seq_len, hidden_dim)
        modified = output

        if position == "last":
            # Add steering to last position only
            # This affects the token being generated
            modified = mx.concatenate([
                output[:, :-1, :],
                output[:, -1:, :] + scaled_vec.reshape(1, 1, -1)
            ], axis=1)
        elif position == "all":
            # Add steering to all positions
            modified = output + scaled_vec.reshape(1, 1, -1)
        elif isinstance(position, int):
            # Add steering to specific position
            before = output[:, :position, :]
            target = output[:, position:position+1, :] + scaled_vec.reshape(1, 1, -1)
            after = output[:, position+1:, :]
            modified = mx.concatenate([before, target, after], axis=1)

        return modified

    return hook


def create_multi_layer_hooks(
    steering_vector: Any,
    layers: List[int],
    alpha: float = 1.0,
    decay: float = 1.0,
) -> List[Tuple[str, Callable]]:
    """
    Create steering hooks for multiple layers.

    Args:
        steering_vector: Base steering vector
        layers: List of layer indices
        alpha: Base steering strength
        decay: Decay factor per layer (1.0 = no decay)

    Returns:
        List of (layer_name, hook_fn) tuples for run_with_hooks
    """
    hooks = []
    for i, layer in enumerate(layers):
        # Apply decay: earlier layers get weaker steering
        layer_alpha = alpha * (decay ** i)
        hook_fn = create_steering_hook(steering_vector, alpha=layer_alpha)
        layer_name = f"model.layers.{layer}"
        hooks.append((layer_name, hook_fn))
    return hooks


# =============================================================================
# CONTRASTIVE STEERING
# =============================================================================

def create_contrastive_hook(
    positive_vector: Any,
    negative_vector: Any,
    alpha: float = 1.0,
) -> Callable:
    """
    Create hook that adds positive and subtracts negative directions.

    This pushes generation towards positive examples and away from negative.

    Args:
        positive_vector: Direction to move towards
        negative_vector: Direction to move away from
        alpha: Steering strength

    Returns:
        Hook function
    """
    # Compute contrastive direction
    direction = positive_vector - negative_vector
    return create_steering_hook(direction, alpha=alpha)


# =============================================================================
# ADAPTIVE STEERING
# =============================================================================

class AdaptiveSteeringHook:
    """
    Steering hook that adapts strength based on generation progress.

    Can increase/decrease steering as generation continues.
    """

    def __init__(
        self,
        steering_vector: Any,
        initial_alpha: float = 1.0,
        decay_rate: float = 0.95,
        min_alpha: float = 0.1,
    ):
        self.steering_vector = steering_vector
        self.initial_alpha = initial_alpha
        self.decay_rate = decay_rate
        self.min_alpha = min_alpha
        self.step = 0
        self.current_alpha = initial_alpha

    def reset(self):
        """Reset step counter."""
        self.step = 0
        self.current_alpha = self.initial_alpha

    def get_hook(self) -> Callable:
        """Get current hook function."""
        return create_steering_hook(
            self.steering_vector,
            alpha=self.current_alpha
        )

    def step_forward(self):
        """Update alpha for next step."""
        self.step += 1
        self.current_alpha = max(
            self.min_alpha,
            self.initial_alpha * (self.decay_rate ** self.step)
        )


# =============================================================================
# VALIDITY-SPECIFIC HOOKS
# =============================================================================

def create_validity_hook(
    valid_vector: Any,
    invalid_vector: Any,
    alpha: float = 1.0,
) -> Callable:
    """
    Create steering hook specifically for statechart validity.

    Uses contrastive direction between valid and invalid examples.
    """
    return create_contrastive_hook(valid_vector, invalid_vector, alpha)


def create_syntax_hook(
    syntax_vector: Any,
    alpha: float = 1.5,  # Stronger for syntax
) -> Callable:
    """
    Create steering hook for JSON syntax validity.

    Uses higher default alpha since syntax is critical.
    """
    return create_steering_hook(syntax_vector, alpha=alpha)


def create_semantic_hook(
    semantic_vector: Any,
    alpha: float = 1.0,
) -> Callable:
    """
    Create steering hook for semantic validity (initial states, transitions).
    """
    return create_steering_hook(semantic_vector, alpha=alpha)


# =============================================================================
# HOOK UTILITIES
# =============================================================================

def get_available_layers(model: Any) -> List[str]:
    """Get available hook points in model."""
    if hasattr(model, 'available_hooks'):
        return model.available_hooks()
    return []


def find_layer_hooks(model: Any, pattern: str = "layers") -> List[str]:
    """Find layer hook points matching pattern."""
    if hasattr(model, 'find_hooks'):
        return model.find_hooks(pattern)
    return []


# =============================================================================
# TESTING
# =============================================================================

def test_hook_injection():
    """Test hook creation and injection."""
    print("=" * 60)
    print("HOOK INJECTION TEST")
    print("=" * 60)

    print(f"\nMLX available: {MLX_AVAILABLE}")

    if not MLX_AVAILABLE:
        print("Skipping tests (MLX not available)")
        return True

    # Test basic hook creation
    print("\nCreating steering hook...")
    vec = mx.random.normal((896,))
    hook = create_steering_hook(vec, alpha=1.0)
    print(f"Hook created: {hook}")

    # Test with mock output
    print("\nTesting hook with mock data...")
    mock_output = mx.random.normal((1, 10, 896))
    modified = hook((), mock_output, None)
    print(f"Input shape: {mock_output.shape}")
    print(f"Output shape: {modified.shape}")

    # Check steering was applied to last position
    diff = mx.abs(modified[:, -1, :] - mock_output[:, -1, :])
    print(f"Max diff at last position: {mx.max(diff).item():.4f}")

    # Test multi-layer hooks
    print("\nCreating multi-layer hooks...")
    hooks = create_multi_layer_hooks(vec, layers=[6, 12, 18], alpha=1.0, decay=0.8)
    print(f"Created {len(hooks)} hooks")
    for name, _ in hooks:
        print(f"  {name}")

    # Test contrastive hook
    print("\nCreating contrastive hook...")
    pos_vec = mx.random.normal((896,))
    neg_vec = mx.random.normal((896,))
    contrast_hook = create_contrastive_hook(pos_vec, neg_vec, alpha=1.0)
    print(f"Contrastive hook created: {contrast_hook}")

    # Test adaptive steering
    print("\nTesting adaptive steering...")
    adaptive = AdaptiveSteeringHook(vec, initial_alpha=1.0, decay_rate=0.9)
    print(f"Initial alpha: {adaptive.current_alpha}")
    for _ in range(5):
        adaptive.step_forward()
    print(f"Alpha after 5 steps: {adaptive.current_alpha:.3f}")

    print("\n[PASS] Hook injection test complete")
    return True


if __name__ == "__main__":
    test_hook_injection()
