"""Continuum Memory System (CMS) as Multi-Frequency Statechart Hierarchy.

Maps the HOPE architecture's CMS concept to statecharts:
- Multiple MLP blocks at different update frequencies
- Higher levels (slower) consolidate knowledge from lower levels (faster)
- Creates spectrum from short-term to long-term memory

Statechart mapping:
- Level 0 (τ=1): Leaf states, updated every step
- Level 1 (τ=10): Region states, updated every 10 steps
- Level 2 (τ=100): Composite states, updated every 100 steps
- Level 3 (τ=1000): Root level, slow consolidation
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple


class ContinuumMemoryLevel(nn.Module):
    """Single level of the Continuum Memory System.

    Each level operates at a specific frequency τ, updating only
    every τ steps. This creates temporal hierarchy analogous to
    statechart hierarchy levels.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 4.0,
        update_frequency: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_frequency = update_frequency
        self.step_counter = 0

        # SwiGLU-style MLP (from Titans)
        inner_dim = int(hidden_dim * expansion)
        self.gate_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=False)

        # Persistent state (memory)
        self._memory_state = None

    def __call__(self, x: mx.array, force_update: bool = False) -> mx.array:
        """Process input, updating memory only at frequency intervals.

        Args:
            x: Input tensor [B, seq_len, hidden_dim]
            force_update: Override frequency check and update

        Returns:
            Output tensor [B, seq_len, hidden_dim]
        """
        self.step_counter += 1
        should_update = force_update or (self.step_counter % self.update_frequency == 0)

        if should_update:
            # SwiGLU: gate * up, then down
            gate = mx.sigmoid(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
            output = self.down_proj(hidden)

            # Update memory state
            self._memory_state = output.mean(axis=1, keepdims=True)  # Pool over sequence

            return output
        else:
            # Use cached memory state if available
            if self._memory_state is not None:
                # Broadcast memory to sequence length
                return mx.broadcast_to(self._memory_state, x.shape)
            else:
                return x  # Identity on first step before any update

    def reset(self):
        """Reset step counter and memory state."""
        self.step_counter = 0
        self._memory_state = None


class ContinuumMemorySystem(nn.Module):
    """Full Continuum Memory System with multiple frequency levels.

    Implements the HOPE architecture's key innovation: multiple memory
    levels operating at different timescales, enabling continual learning
    without catastrophic forgetting.

    Statechart interpretation:
    - Level 0: Leaf state dynamics (fast, reactive)
    - Level 1: Region coordination (medium, tactical)
    - Level 2: Composite state patterns (slow, strategic)
    - Level 3: Root level consolidation (slowest, foundational)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int = 4,
        expansion: float = 4.0,
        base_frequency: int = 1,
        frequency_multiplier: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        # Create levels with exponentially increasing frequencies
        self.levels = []
        for i in range(num_levels):
            freq = base_frequency * (frequency_multiplier ** i)
            level = ContinuumMemoryLevel(hidden_dim, expansion, freq)
            self.levels.append(level)

        # Level mixing (learnable combination weights)
        self.level_weights = mx.ones(num_levels) / num_levels

        # Cross-level communication
        self.level_projections = [
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_levels - 1)
        ]

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Process input through all CMS levels.

        Args:
            x: Input tensor [B, seq_len, hidden_dim]

        Returns:
            Tuple of:
                - Combined output [B, seq_len, hidden_dim]
                - List of per-level outputs for analysis
        """
        level_outputs = []
        current = x

        # Process through each level
        for i, level in enumerate(self.levels):
            level_out = level(current)
            level_outputs.append(level_out)

            # Pass information up to next level
            if i < len(self.level_projections):
                current = self.level_projections[i](level_out)

        # Combine level outputs with learned weights
        weights = mx.softmax(self.level_weights)
        combined = sum(w * out for w, out in zip(weights, level_outputs))

        return combined, level_outputs

    def reset_all(self):
        """Reset all levels."""
        for level in self.levels:
            level.reset()


class SelfReferentialStatechartLayer(nn.Module):
    """Self-Referential layer that learns its own update algorithm.

    Key innovation from Titans: values are a function of memory state,
    creating a self-modifying system that controls its own learning.

    Statechart interpretation: Guards that can modify other guards,
    enabling meta-level state machine adaptation.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key projections (standard)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Value projection depends on memory state (self-referential)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_state_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Memory state for self-reference
        self._memory_state = None

        # Meta-learning parameters (learnable learning rate and weight decay)
        self.meta_lr = mx.array([0.01])  # Learnable learning rate
        self.meta_wd = mx.array([0.001])  # Learnable weight decay

    def __call__(self, x: mx.array) -> mx.array:
        """Self-referential attention where values depend on memory state.

        Args:
            x: Input [B, seq_len, hidden_dim]

        Returns:
            Output [B, seq_len, hidden_dim]
        """
        B, L, D = x.shape

        # Standard Q, K computation
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)

        # Self-referential V: depends on both input AND memory state
        v_input = self.v_proj(x)
        if self._memory_state is not None:
            v_state = self.v_state_proj(self._memory_state)
            v = v_input + v_state  # Combine input and memory-dependent values
        else:
            v = v_input
        v = v.reshape(B, L, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        q = q.transpose(0, 2, 1, 3)  # [B, heads, L, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        attn = mx.softmax(scores, axis=-1)

        out = mx.matmul(attn, v)  # [B, heads, L, head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.o_proj(out)

        # Update memory state with meta-learning (detach to prevent gradient through update)
        new_state = mx.stop_gradient(out.mean(axis=1, keepdims=True))
        if self._memory_state is not None:
            lr = mx.sigmoid(self.meta_lr)  # Bound to [0, 1]
            wd = mx.sigmoid(self.meta_wd)
            self._memory_state = mx.stop_gradient(
                (1 - lr) * self._memory_state * (1 - wd) + lr * new_state
            )
        else:
            self._memory_state = new_state

        return out

    def reset(self):
        """Reset memory state."""
        self._memory_state = None


class HOPEStatechart(nn.Module):
    """Complete HOPE architecture implemented as differentiable statechart.

    Combines:
    - Self-Referential Titans layer (self-modifying guards)
    - Continuum Memory System (multi-frequency hierarchy)
    - Standard statechart structure (states, transitions, configurations)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        # Self-referential attention layer
        self.self_ref = SelfReferentialStatechartLayer(hidden_dim, num_heads)

        # Continuum Memory System
        self.cms = ContinuumMemorySystem(hidden_dim, num_levels)

        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass through HOPE statechart.

        Args:
            x: Input [B, seq_len, hidden_dim]

        Returns:
            Tuple of:
                - Output [B, seq_len, hidden_dim]
                - Per-level CMS outputs
        """
        # Self-referential attention with residual
        h = x + self.self_ref(self.norm1(x))

        # Continuum Memory System with residual
        cms_out, level_outputs = self.cms(self.norm2(h))
        out = h + cms_out

        return out, level_outputs

    def reset(self):
        """Reset all stateful components."""
        self.self_ref.reset()
        self.cms.reset_all()


def test_hope_statechart():
    """Test the HOPE statechart implementation."""
    print("=" * 60)
    print("Testing HOPE Statechart (Titans + Nested Learning)")
    print("=" * 60)

    # Create model
    hidden_dim = 64
    model = HOPEStatechart(hidden_dim=hidden_dim, num_levels=4, num_heads=4)

    # Random input
    mx.random.seed(42)
    batch_size, seq_len = 2, 16
    x = mx.random.normal((batch_size, seq_len, hidden_dim))

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    out, level_outputs = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Number of CMS levels: {len(level_outputs)}")

    for i, level_out in enumerate(level_outputs):
        freq = 10 ** i
        print(f"  Level {i} (τ={freq}): {level_out.shape}")

    # Test gradient flow
    print("\nTesting gradient flow...")

    def loss_fn(model, x):
        out, _ = model(x)
        return mx.mean(out ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x)
    print(f"Loss: {loss:.6f}")

    # Check gradients exist
    has_grads = False
    def check_grads(g, prefix=""):
        nonlocal has_grads
        if isinstance(g, dict):
            for k, v in g.items():
                check_grads(v, f"{prefix}{k}.")
        elif isinstance(g, mx.array):
            norm = float(mx.sqrt(mx.sum(g * g)))
            if norm > 1e-10:
                has_grads = True

    check_grads(grads)
    print(f"Gradients flow: {'YES' if has_grads else 'NO'}")

    # Test continual updates (multiple steps)
    print("\nTesting continual updates (20 steps)...")
    model.reset()

    for step in range(20):
        x_step = mx.random.normal((batch_size, seq_len, hidden_dim))
        out, levels = model(x_step)

        if step % 5 == 0:
            level_norms = [float(mx.mean(mx.abs(l))) for l in levels]
            print(f"  Step {step}: level norms = {[f'{n:.4f}' for n in level_norms]}")

    print("\n" + "=" * 60)
    print("HOPE Statechart test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hope_statechart()
