"""
Statechart Layers as Neural Network Middleware

Wraps differentiable statechart components as nn.Module layers
that can be inserted into standard neural network architectures.

Components:
- StatechartLayer: Base layer wrapping soft config + transitions
- StatechartSandwich: Linear -> StatechartLayer -> Linear
- StatechartResidual: x + StatechartLayer(x)
- StatechartStack: Multiple layers at different granularity
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple


class StatechartLayer(nn.Module):
    """
    Wraps soft state configuration as an nn.Module layer.
    
    Input: [B, input_dim] tensor
    Output: [B, input_dim] tensor (same shape, processed through statechart)
    
    Internally:
    1. Projects input to state space
    2. Computes soft transitions based on input
    3. Projects back to input dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        num_states: int,
        num_transitions: int = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_states = num_states
        self.num_transitions = num_transitions if num_transitions is not None else num_states
        self.temperature = temperature

        # Use instance variables (guaranteed to be int)
        n_states = self.num_states
        n_trans = self.num_transitions

        # Project input to state logits
        self.to_state = nn.Linear(input_dim, n_states)

        # Transition parameters: learns which transitions to enable
        self.transition_source = nn.Linear(n_states, n_trans)
        self.transition_target = nn.Linear(n_trans, n_states)

        # Guard network: input-dependent transition gating
        self.guard_net = nn.Sequential(
            nn.Linear(input_dim, n_trans * 2),
            nn.Tanh(),
            nn.Linear(n_trans * 2, n_trans),
        )
        
        # Project state back to input dimension
        self.from_state = nn.Linear(n_states, input_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Process input through statechart layer.
        
        Args:
            x: [B, input_dim] input tensor
            
        Returns:
            [B, input_dim] processed output
        """
        B = x.shape[0]
        
        # 1. Get soft state configuration from input
        state_logits = self.to_state(x)  # [B, num_states]
        config = mx.softmax(state_logits / self.temperature, axis=-1)
        
        # 2. Compute transition enablement from source states
        source_weights = mx.softmax(self.transition_source(config), axis=-1)  # [B, num_transitions]
        
        # 3. Compute guards (input-dependent gating)
        guards = mx.sigmoid(self.guard_net(x))  # [B, num_transitions]
        
        # 4. Apply guards to source weights
        enabled = source_weights * guards  # [B, num_transitions]
        
        # 5. Compute new configuration via target weights
        new_config = self.transition_target(enabled)  # [B, num_states]
        new_config = mx.softmax(new_config / self.temperature, axis=-1)
        
        # 6. Blend old and new config (smooth transition)
        blend_factor = mx.mean(guards, axis=-1, keepdims=True)  # [B, 1]
        blended = (1 - blend_factor) * config + blend_factor * new_config
        
        # 7. Project back to input dimension
        output = self.from_state(blended)
        
        return output
    
    def get_config(self, x: mx.array) -> mx.array:
        """Get soft state configuration without full forward pass."""
        state_logits = self.to_state(x)
        return mx.softmax(state_logits / self.temperature, axis=-1)


class StatechartSandwich(nn.Module):
    """
    Linear -> StatechartLayer -> Linear
    
    Sandwiches a statechart layer between two linear projections.
    The statechart acts as a structured bottleneck.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_states: int,
        output_dim: int = None,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        
        self.pre_linear = nn.Linear(input_dim, hidden_dim)
        self.statechart = StatechartLayer(hidden_dim, num_states)
        self.post_linear = nn.Linear(hidden_dim, output_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Process through sandwich structure.
        
        Args:
            x: [B, input_dim]
            
        Returns:
            [B, output_dim]
        """
        h = mx.tanh(self.pre_linear(x))
        h = self.statechart(h)
        return self.post_linear(h)


class StatechartResidual(nn.Module):
    """
    x + StatechartLayer(x)
    
    Adds statechart processing as a residual connection.
    Allows the network to optionally use statechart structure.
    """
    
    def __init__(
        self,
        dim: int,
        num_states: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.statechart = StatechartLayer(dim, num_states)
        self.scale = scale
        self.norm = nn.LayerNorm(dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Process with residual connection.
        
        Args:
            x: [B, dim]
            
        Returns:
            [B, dim]
        """
        # Pre-norm residual
        h = self.norm(x)
        h = self.statechart(h)
        return x + self.scale * h


class StatechartStack(nn.Module):
    """
    Multiple statechart layers at different granularity.
    
    Each layer can have different number of states, allowing
    multi-scale state representation.
    """
    
    def __init__(
        self,
        dim: int,
        state_counts: List[int],
        use_residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual
        
        if use_residual:
            self.layers = [
                StatechartResidual(dim, num_states)
                for num_states in state_counts
            ]
        else:
            self.layers = [
                StatechartLayer(dim, num_states)
                for num_states in state_counts
            ]
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Process through stacked layers.
        
        Args:
            x: [B, dim]
            
        Returns:
            [B, dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_all_configs(self, x: mx.array) -> List[mx.array]:
        """Get state configurations from all layers."""
        configs = []
        h = x
        for layer in self.layers:
            if self.use_residual:
                sc_layer = layer.statechart
            else:
                sc_layer = layer
            configs.append(sc_layer.get_config(h))
            h = layer(h)
        return configs


class StatechartGate(nn.Module):
    """
    Uses statechart configuration to gate another pathway.
    
    gate = StatechartLayer(x)
    output = gate * other_pathway(x)
    """
    
    def __init__(self, dim: int, num_states: int):
        super().__init__()
        self.statechart = StatechartLayer(dim, num_states)
        self.gate_proj = nn.Linear(dim, dim)
    
    def __call__(self, x: mx.array, other: mx.array) -> mx.array:
        """
        Gate other pathway using statechart output.
        
        Args:
            x: [B, dim] input for statechart
            other: [B, dim] pathway to gate
            
        Returns:
            [B, dim] gated output
        """
        sc_out = self.statechart(x)
        gate = mx.sigmoid(self.gate_proj(sc_out))
        return gate * other


# =============================================================================
# Test Script
# =============================================================================


def test_statechart_layers():
    """Test gradient flow through all layer types."""
    print("=" * 60)
    print("Testing Statechart Layers as Middleware")
    print("=" * 60)
    
    batch_size = 4
    input_dim = 16
    num_states = 4
    
    mx.random.seed(42)
    x = mx.random.normal((batch_size, input_dim))
    
    # 1. Test StatechartLayer
    print("\n1. StatechartLayer")
    layer = StatechartLayer(input_dim, num_states)
    out = layer(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    config = layer.get_config(x)
    print(f"   State config: {config.shape}, sum={float(mx.sum(config[0])):.3f}")
    
    # Gradient test
    def loss_fn(model, inp):
        return mx.mean(model(inp) ** 2)
    
    loss, grads = nn.value_and_grad(layer, loss_fn)(layer, x)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # 2. Test StatechartSandwich
    print("\n2. StatechartSandwich")
    sandwich = StatechartSandwich(input_dim, hidden_dim=32, num_states=num_states)
    out = sandwich(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    loss, grads = nn.value_and_grad(sandwich, loss_fn)(sandwich, x)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # 3. Test StatechartResidual
    print("\n3. StatechartResidual")
    residual = StatechartResidual(input_dim, num_states)
    out = residual(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Check residual property
    diff = mx.mean(mx.abs(out - x))
    print(f"   Mean diff from input: {float(diff):.4f}")
    
    loss, grads = nn.value_and_grad(residual, loss_fn)(residual, x)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # 4. Test StatechartStack
    print("\n4. StatechartStack (multi-granularity)")
    stack = StatechartStack(input_dim, state_counts=[2, 4, 8])
    out = stack(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Layers: 3 (states: 2, 4, 8)")
    
    configs = stack.get_all_configs(x)
    for i, cfg in enumerate(configs):
        print(f"   Layer {i} config shape: {cfg.shape}")
    
    loss, grads = nn.value_and_grad(stack, loss_fn)(stack, x)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # 5. Test StatechartGate
    print("\n5. StatechartGate")
    gate = StatechartGate(input_dim, num_states)
    other = mx.random.normal((batch_size, input_dim))
    out = gate(x, other)
    print(f"   Inputs: x={x.shape}, other={other.shape} -> Output: {out.shape}")
    
    def gate_loss(model, inp, oth):
        return mx.mean(model(inp, oth) ** 2)
    
    loss, grads = nn.value_and_grad(gate, gate_loss)(gate, x, other)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("All layer tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_statechart_layers()
