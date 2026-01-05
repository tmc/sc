"""
Experiment A: Soft State Configuration for Differentiable Statecharts

This module implements differentiable statechart primitives using MLX.
The key insight is replacing discrete state membership with continuous
probability distributions, enabling gradient-based optimization.

Core components:
- SoftStateConfiguration: Represents state activations as soft probabilities
- DifferentiableGuard: Learnable guard conditions that output [0,1] values
"""

import mlx.core as mx
import mlx.nn as nn


class SoftStateConfiguration(nn.Module):
    """
    Soft state configuration using differentiable probability distributions.

    Instead of discrete state membership (state ∈ {0, 1}), we use
    soft activations (state ∈ [0, 1]) via softmax over state logits.

    For a 2-state toggle (On/Off):
    - State logits: [B, 2] learnable parameters
    - State probs: softmax(logits / temperature) -> [B, 2]

    Temperature controls sharpness:
    - T → 0: approaches discrete (one-hot)
    - T → ∞: uniform distribution
    """

    def __init__(self, num_states: int = 2, temperature: float = 1.0):
        super().__init__()
        self.num_states = num_states
        self.temperature = temperature
        # Initialize state logits - equal probability initially
        self.state_logits = mx.zeros((num_states,))

    def get_config(self, batch_size: int = 1) -> mx.array:
        """
        Get soft state configuration as probability distribution.

        Returns:
            [B, num_states] tensor of state probabilities (sum to 1)
        """
        # Apply temperature-scaled softmax
        scaled_logits = self.state_logits / self.temperature
        probs = mx.softmax(scaled_logits)
        # Broadcast to batch dimension
        return mx.broadcast_to(probs, (batch_size, self.num_states))

    def set_logits(self, logits: mx.array):
        """Set state logits directly (for gradient updates)."""
        self.state_logits = logits

    def get_state_prob(self, state_idx: int, batch_size: int = 1) -> mx.array:
        """Get probability of being in a specific state."""
        config = self.get_config(batch_size)
        return config[:, state_idx]

    def entropy(self) -> mx.array:
        """Compute entropy of state distribution (measure of uncertainty)."""
        probs = mx.softmax(self.state_logits / self.temperature)
        # H = -sum(p * log(p)), with numerical stability
        log_probs = mx.log(probs + 1e-10)
        return -mx.sum(probs * log_probs)


class DifferentiableGuard(nn.Module):
    """
    Differentiable guard condition using a small MLP.

    Maps context features to a [0, 1] guard satisfaction value.
    The guard can be thought of as "how satisfied is this condition?"

    Architecture:
        context [B, context_dim] -> MLP -> sigmoid -> [B, 1]

    Temperature parameter controls sharpness of the sigmoid:
    - Low temp: more binary-like decisions
    - High temp: more gradual transitions
    """

    def __init__(self, context_dim: int, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Simple MLP: context -> hidden -> 1
        self.l1 = nn.Linear(context_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

    def __call__(self, context: mx.array) -> mx.array:
        """
        Evaluate guard condition on context.

        Args:
            context: [B, context_dim] context features

        Returns:
            [B, 1] guard satisfaction values in [0, 1]
        """
        x = self.l1(context)
        x = mx.tanh(x)  # Bounded activation for stability
        x = self.l2(x)
        # Temperature-scaled sigmoid
        return mx.sigmoid(x / self.temperature)


class SoftToggle(nn.Module):
    """
    Complete 2-state soft toggle with differentiable transitions.

    States: On (idx=0), Off (idx=1)
    Transitions:
        On --[turn_off]--> Off
        Off --[turn_on]--> On

    Guards determine transition probability based on context.
    """

    def __init__(self, context_dim: int, temperature: float = 1.0):
        super().__init__()
        self.config = SoftStateConfiguration(num_states=2, temperature=temperature)

        # Guards for each transition
        self.guard_on_to_off = DifferentiableGuard(context_dim, temperature=temperature)
        self.guard_off_to_on = DifferentiableGuard(context_dim, temperature=temperature)

    def step(self, context: mx.array) -> mx.array:
        """
        Perform one soft step of the toggle.

        The new state distribution is computed by:
        1. Get current state probs
        2. Compute guard values for each transition
        3. Apply soft transition semantics

        Returns:
            [B, 2] new state configuration
        """
        batch_size = context.shape[0]
        current = self.config.get_config(batch_size)

        # Get guard values
        g_on_off = self.guard_on_to_off(context)  # [B, 1]
        g_off_on = self.guard_off_to_on(context)  # [B, 1]

        # Current state probs
        p_on = current[:, 0:1]   # [B, 1]
        p_off = current[:, 1:2]  # [B, 1]

        # Soft transition: probability mass flows based on guard * source prob
        # New p_on = p_on * (1 - g_on_off) + p_off * g_off_on
        # New p_off = p_off * (1 - g_off_on) + p_on * g_on_off

        new_p_on = p_on * (1 - g_on_off) + p_off * g_off_on
        new_p_off = p_off * (1 - g_off_on) + p_on * g_on_off

        new_config = mx.concatenate([new_p_on, new_p_off], axis=1)

        # Normalize to ensure valid probability distribution
        return new_config / mx.sum(new_config, axis=1, keepdims=True)


def test_gradient_flow():
    """Test that gradients flow through the soft statechart."""
    print("=" * 60)
    print("Testing Gradient Flow in Differentiable Statechart")
    print("=" * 60)

    # Create a simple soft toggle
    context_dim = 4
    toggle = SoftToggle(context_dim=context_dim, temperature=1.0)

    # Random context
    mx.random.seed(42)
    context = mx.random.normal((1, context_dim))

    print("\n1. Initial State Configuration:")
    initial_config = toggle.config.get_config(1)
    print(f"   State probs [On, Off]: {initial_config}")

    print("\n2. Testing DifferentiableGuard:")
    guard = DifferentiableGuard(context_dim)
    guard_value = guard(context)
    print(f"   Guard output: {guard_value}")

    # Test gradient flow through the guard
    print("\n3. Testing Guard Gradient Computation:")

    def guard_loss_fn(model, x):
        """Loss for guard: just return the output."""
        return model(x)[0, 0]

    loss_value, grads = nn.value_and_grad(guard, guard_loss_fn)(guard, context)
    print(f"   Guard output: {loss_value:.6f}")

    # grads is a dict of parameter name -> gradient tensor
    print(f"   Number of gradient tensors: {len(grads)}")

    # Check guard gradients
    guard_has_grad = False

    def print_grad_tree(grads, prefix=""):
        """Recursively print gradient tree."""
        nonlocal guard_has_grad
        if isinstance(grads, dict):
            for name, val in grads.items():
                print_grad_tree(val, f"{prefix}{name}.")
        elif isinstance(grads, mx.array):
            grad_norm = float(mx.sqrt(mx.sum(grads * grads)))
            print(f"   {prefix[:-1]}: shape={grads.shape}, norm={grad_norm:.6f}")
            if grad_norm > 1e-10:
                guard_has_grad = True

    print_grad_tree(grads)

    print("\n4. Testing SoftToggle Gradient Computation:")

    def toggle_loss_fn(model, x):
        """Loss: P(Off) after step - minimize to encourage On state."""
        new_config = model.step(x)
        return new_config[0, 1]  # P(Off)

    toggle_loss, toggle_grads = nn.value_and_grad(toggle, toggle_loss_fn)(toggle, context)
    print(f"   Loss (P(Off)): {toggle_loss:.6f}")

    # Check toggle gradients
    toggle_has_grad = False

    def check_toggle_grads(grads, prefix=""):
        """Recursively check gradient tree."""
        nonlocal toggle_has_grad
        if isinstance(grads, dict):
            for name, val in grads.items():
                check_toggle_grads(val, f"{prefix}{name}.")
        elif isinstance(grads, mx.array):
            grad_norm = float(mx.sqrt(mx.sum(grads * grads)))
            print(f"   {prefix[:-1]}: shape={grads.shape}, norm={grad_norm:.6f}")
            if grad_norm > 1e-10:
                toggle_has_grad = True

    check_toggle_grads(toggle_grads)

    print("\n5. Gradient Flow Verification:")
    if guard_has_grad:
        print("   [checkmark] Guard gradients flow correctly!")
    else:
        print("   [X] Warning: Guard gradients are zero")

    if toggle_has_grad:
        print("   [checkmark] Toggle gradients flow correctly!")
    else:
        print("   [X] Warning: Toggle gradients are zero")

    # Test entropy computation
    print("\n6. State Configuration Entropy:")
    soft_config = SoftStateConfiguration(num_states=2)
    entropy = soft_config.entropy()
    print(f"   Entropy (uniform init): {entropy:.4f}")
    print(f"   Max entropy for 2 states: {float(mx.log(mx.array(2.0))):.4f}")

    # Test with biased configuration
    soft_config.set_logits(mx.array([2.0, -2.0]))  # Heavily favor state 0
    entropy_biased = soft_config.entropy()
    print(f"   Entropy (biased init): {entropy_biased:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

    return {
        "initial_config": initial_config,
        "guard_loss": float(loss_value),
        "guard_has_grad": guard_has_grad,
        "toggle_has_grad": toggle_has_grad
    }


if __name__ == "__main__":
    results = test_gradient_flow()
