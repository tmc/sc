"""
Hybrid Memory: Hull (exact structural) + SSM (long-horizon temporal).

Implements HybridMemoryConfig from the proto proposal (§3.12).

DESIGN RATIONALE:
  Hulls and SSMs serve complementary roles:
  - Hull: O(log n) exact retrieval for deterministic topology execution
    and Deep History (H*) restoration. Returns exact argmax.
  - SSM: O(n log n) long-horizon temporal modeling via selective
    state-space model. Handles continuous-time event loops over
    thousands of steps.

COMBINATION MODES:
  1. CONCAT:    [hull_out; ssm_out] -> Linear(2D, D)
  2. ADD:       hull_out + ssm_out
  3. GATE:      g * hull_out + (1-g) * ssm_out, g = sigmoid(W[h;s])
  4. ATTENTION: Cross-attention between hull and SSM outputs

ROUTING:
  Optionally learn which queries go to hull (exact, structural) vs.
  SSM (approximate, temporal) via a routing network.

References:
  - [Gu 2023] Mamba: Linear-Time Sequence Modeling
  - [Barber 1996] Quickhull algorithm for convex hulls
"""

import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List

from .hull_memory import HullKVCache, HullConfig


class MemoryCombination(IntEnum):
    """How to combine hull and SSM outputs (mirrors proto enum)."""
    CONCAT = 1
    ADD = 2
    GATE = 3
    ATTENTION = 4


@dataclass
class SSMConfig:
    """Selective state-space model configuration."""
    state_dim: int = 64       # SSM hidden state dimension
    ssm_rank: int = 16        # Low-rank projection for A/B
    dt_rank: int = 8          # Discretization step rank
    expand_factor: int = 2    # Channel expansion in SSM block
    n_layers: int = 1         # Number of SSM blocks


@dataclass
class HybridMemoryConfig:
    """Configuration mirroring proto HybridMemoryConfig."""
    hull: HullConfig = None
    ssm: SSMConfig = None
    combination: MemoryCombination = MemoryCombination.GATE
    learned_routing: bool = True
    routing_threshold: float = 0.5
    memory_dim: int = 64

    def __post_init__(self):
        if self.hull is None:
            self.hull = HullConfig(memory_dim=self.memory_dim)
        if self.ssm is None:
            self.ssm = SSMConfig(state_dim=self.memory_dim)


# ---------------------------------------------------------------------------
# Selective SSM Block (simplified Mamba-like)
# ---------------------------------------------------------------------------

class SelectiveSSMBlock(nn.Module):
    """
    Simplified selective state-space model block.

    Implements the core S6 selective scan: the A and B matrices are
    input-dependent (selective), enabling content-aware temporal modeling.

    x_t = A_t * x_{t-1} + B_t * u_t
    y_t = C * x_t

    Where A_t, B_t are computed from the input via learned projections.
    """

    def __init__(self, d_model: int, ssm_rank: int = 16, dt_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.ssm_rank = ssm_rank

        # Input-dependent discretization
        self.dt_proj = nn.Linear(d_model, dt_rank)
        self.dt_out = nn.Linear(dt_rank, d_model)

        # Input-dependent B projection
        self.b_proj = nn.Linear(d_model, ssm_rank)

        # Fixed A (diagonal, initialized to negative values for stability)
        # In MLX we store as learnable parameter
        self._a_log = mx.random.uniform(
            low=-4.0, high=-1.0, shape=(d_model, ssm_rank)
        )

        # Output C: [D, R] learned projection, sum over R -> [D]
        self._c_weight = mx.random.normal((d_model, ssm_rank)) * 0.01

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Hidden state
        self._h = None

    def reset_state(self):
        """Reset SSM hidden state."""
        self._h = None

    def _discretize(self, u: mx.array):
        """Compute input-dependent A_bar, B_bar via ZOH discretization."""
        # dt = softplus(dt_proj(u))
        dt = nn.softplus(self.dt_out(nn.silu(self.dt_proj(u))))  # [D]

        # A = -exp(a_log) (diagonal, negative for stability)
        a = -mx.exp(self._a_log)  # [D, R]

        # B is input-dependent
        b = self.b_proj(u)  # [R]

        # ZOH discretization: A_bar = exp(A * dt), B_bar = (A_bar - I) * A^{-1} * B
        # Simplified: A_bar ≈ 1 + A*dt (first-order for small dt)
        dt_expanded = mx.expand_dims(dt, -1)  # [D, 1]
        a_bar = mx.exp(a * dt_expanded)  # [D, R]
        b_bar = mx.expand_dims(b, 0) * dt_expanded  # [D, R]

        return a_bar, b_bar

    def step(self, u: mx.array) -> mx.array:
        """
        One SSM step.

        Args:
            u: [d_model] input vector.

        Returns:
            y: [d_model] output vector.
        """
        u = self.norm(u)
        a_bar, b_bar = self._discretize(u)

        if self._h is None:
            self._h = mx.zeros((self.d_model, self.ssm_rank))

        # State update: h = A_bar * h + B_bar * u
        u_expanded = mx.expand_dims(u, -1)  # [D, 1]
        self._h = a_bar * self._h + b_bar * u_expanded  # [D, R]

        # Output: y = C * h, reduce over ssm_rank dimension
        # _h is [D, R], _c_weight is [D, R] -> element-wise -> sum over R -> [D]
        y = mx.sum(self._h * self._c_weight, axis=-1)  # [D]

        return y

    def forward_sequence(self, inputs: mx.array) -> mx.array:
        """
        Process a sequence through the SSM.

        Args:
            inputs: [seq_len, d_model] input sequence.

        Returns:
            outputs: [seq_len, d_model] output sequence.
        """
        self.reset_state()
        outputs = []
        for t in range(inputs.shape[0]):
            y = self.step(inputs[t])
            outputs.append(y)
        return mx.stack(outputs)


# ---------------------------------------------------------------------------
# SSM Memory Backend
# ---------------------------------------------------------------------------

class SSMMemory(nn.Module):
    """
    SSM-based temporal memory backend.

    Processes execution traces through selective SSM blocks to capture
    long-horizon temporal patterns. Unlike the Hull (which provides
    exact structural lookups), the SSM learns approximate temporal
    representations over potentially thousands of steps.
    """

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config

        # Stack of SSM blocks
        self.blocks = [
            SelectiveSSMBlock(config.state_dim, config.ssm_rank, config.dt_rank)
            for _ in range(config.n_layers)
        ]

        # Input/output projections
        self.in_proj = nn.Linear(config.state_dim, config.state_dim)
        self.out_proj = nn.Linear(config.state_dim, config.state_dim)

    def reset(self):
        """Reset all SSM hidden states."""
        for block in self.blocks:
            block.reset_state()

    def step(self, x: mx.array) -> mx.array:
        """
        Process one input through the SSM stack.

        Args:
            x: [state_dim] input vector.

        Returns:
            output: [state_dim] SSM memory output.
        """
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block.step(h)  # Residual connection
        return self.out_proj(h)

    def forward_sequence(self, inputs: mx.array) -> mx.array:
        """Process a sequence, return all outputs."""
        self.reset()
        outputs = []
        for t in range(inputs.shape[0]):
            y = self.step(inputs[t])
            outputs.append(y)
        return mx.stack(outputs)


# ---------------------------------------------------------------------------
# Hybrid Memory (Hull + SSM)
# ---------------------------------------------------------------------------

class MemoryRouter(nn.Module):
    """
    Learned routing network: decides hull vs SSM per query.

    Outputs a scalar routing weight r in [0, 1]:
      r > 0.5 -> prefer hull (exact structural)
      r < 0.5 -> prefer SSM (temporal approximate)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def __call__(self, query: mx.array) -> mx.array:
        """Returns routing weight in [0, 1]."""
        return mx.sigmoid(self.net(query).squeeze())


class HybridMemory(nn.Module):
    """
    Hybrid memory combining Hull (exact) and SSM (temporal) backends.

    Mirrors HybridMemoryConfig from the proto proposal (§3.12).

    COMBINATION MODES:
      CONCAT:    [hull; ssm] -> Linear(2D, D)
      ADD:       hull + ssm
      GATE:      g * hull + (1-g) * ssm
      ATTENTION: cross-attention between hull and SSM outputs

    ROUTING:
      When learned_routing=True, a routing network decides per-query
      whether to emphasize hull (structural) or SSM (temporal) output.
      When False, queries above routing_threshold go to hull.
    """

    def __init__(self, config: HybridMemoryConfig):
        super().__init__()
        self.config = config
        dim = config.memory_dim

        # Hull backend
        self.hull = HullKVCache(config.hull)

        # SSM backend
        self.ssm = SSMMemory(config.ssm)

        # Combination layer
        if config.combination == MemoryCombination.CONCAT:
            self.combine_proj = nn.Linear(dim * 2, dim)
        elif config.combination == MemoryCombination.GATE:
            self.gate_proj = nn.Linear(dim * 2, 1)
        elif config.combination == MemoryCombination.ATTENTION:
            self.cross_q = nn.Linear(dim, dim)
            self.cross_k = nn.Linear(dim, dim)
            self.cross_v = nn.Linear(dim, dim)
            self.cross_out = nn.Linear(dim, dim)

        # Routing
        if config.learned_routing:
            self.router = MemoryRouter(dim)

        # Step counter for hull positions
        self._step_count = 0

    def reset(self):
        """Reset both backends."""
        self.hull.reset(namespace=0)
        self.ssm.reset()
        self._step_count = 0

    def write(self, key: mx.array, value: mx.array):
        """Write to both backends."""
        self.hull.append(key, value, namespace=0)
        self.ssm.step(key)
        self._step_count += 1

    def _combine(
        self,
        hull_out: mx.array,
        ssm_out: mx.array,
        query: mx.array,
    ) -> mx.array:
        """Combine hull and SSM outputs according to config."""
        mode = self.config.combination

        if mode == MemoryCombination.ADD:
            return hull_out + ssm_out

        elif mode == MemoryCombination.CONCAT:
            cat = mx.concatenate([hull_out, ssm_out])
            return self.combine_proj(cat)

        elif mode == MemoryCombination.GATE:
            cat = mx.concatenate([hull_out, ssm_out])
            g = mx.sigmoid(self.gate_proj(cat))
            return g * hull_out + (1.0 - g) * ssm_out

        elif mode == MemoryCombination.ATTENTION:
            # Cross-attention: query attends to [hull, ssm]
            q = self.cross_q(query)                    # [D]
            kv = mx.stack([hull_out, ssm_out])         # [2, D]
            k = self.cross_k(kv)                       # [2, D]
            v = self.cross_v(kv)                       # [2, D]

            scale = 1.0 / math.sqrt(q.shape[-1])
            scores = mx.matmul(mx.expand_dims(q, 0), k.T) * scale  # [1, 2]
            weights = mx.softmax(scores, axis=-1)
            out = mx.matmul(weights, v).squeeze()      # [D]
            return self.cross_out(out)

        return hull_out + ssm_out

    def __call__(
        self,
        query: mx.array,
        query_pos: Optional[int] = None,
    ) -> mx.array:
        """
        Query hybrid memory.

        Args:
            query: [memory_dim] query vector.
            query_pos: Position for hull lookup (defaults to current step).

        Returns:
            output: [memory_dim] combined memory output.
        """
        if query_pos is None:
            query_pos = max(0, self._step_count - 1)

        # Hull output
        hull_out = self.hull(query, query_pos, namespace=0, use_hull=True)

        # SSM output (last hidden state via a step with the query)
        ssm_out = self.ssm.step(query)

        # Route or combine
        if self.config.learned_routing:
            r = self.router(query)
            hull_out = hull_out * r
            ssm_out = ssm_out * (1.0 - r)

        return self._combine(hull_out, ssm_out, query)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hybrid_memory():
    """Test hybrid memory with all combination modes."""
    print("=" * 60)
    print("Hybrid Memory (Hull + SSM) Tests")
    print("=" * 60)

    dim = 32

    # Test 1: SSM block
    print("\n1. Selective SSM block")
    ssm_block = SelectiveSSMBlock(d_model=dim, ssm_rank=8, dt_rank=4)
    x = mx.random.normal((dim,))
    y = ssm_block.step(x)
    mx.eval(y)
    print(f"   Input dim: {x.shape}, Output dim: {y.shape}")
    assert y.shape == (dim,), f"Expected ({dim},), got {y.shape}"

    # Test sequence processing
    seq = mx.random.normal((10, dim))
    out_seq = ssm_block.forward_sequence(seq)
    mx.eval(out_seq)
    print(f"   Sequence: {seq.shape} -> {out_seq.shape}")
    assert out_seq.shape == (10, dim)

    # Test 2: SSM memory
    print("\n2. SSM memory backend")
    ssm_cfg = SSMConfig(state_dim=dim, ssm_rank=8, n_layers=2)
    ssm_mem = SSMMemory(ssm_cfg)
    out = ssm_mem.step(mx.random.normal((dim,)))
    mx.eval(out)
    print(f"   Step output: {out.shape}")
    assert out.shape == (dim,)

    # Test 3: All combination modes
    print("\n3. Combination modes")
    for mode in MemoryCombination:
        config = HybridMemoryConfig(
            hull=HullConfig(memory_dim=dim, num_heads=4, top_k=4),
            ssm=SSMConfig(state_dim=dim, ssm_rank=8),
            combination=mode,
            learned_routing=True,
            memory_dim=dim,
        )
        hybrid = HybridMemory(config)

        # Fill with data
        for j in range(20):
            key = mx.random.normal((dim,))
            val = mx.random.normal((dim,))
            hybrid.write(key, val)

        # Query
        q = mx.random.normal((dim,))
        result = hybrid(q)
        mx.eval(result)
        norm = float(mx.sqrt(mx.sum(result ** 2)).item())
        print(f"   {mode.name:10s}: output norm = {norm:.4f}")
        assert result.shape == (dim,), f"Expected ({dim},), got {result.shape}"

    # Test 4: Routing
    print("\n4. Learned routing")
    config = HybridMemoryConfig(
        hull=HullConfig(memory_dim=dim, num_heads=4, top_k=4),
        ssm=SSMConfig(state_dim=dim, ssm_rank=8),
        combination=MemoryCombination.GATE,
        learned_routing=True,
        memory_dim=dim,
    )
    hybrid = HybridMemory(config)
    for j in range(20):
        hybrid.write(mx.random.normal((dim,)), mx.random.normal((dim,)))

    q = mx.random.normal((dim,))
    r = hybrid.router(q)
    mx.eval(r)
    print(f"   Routing weight: {float(r.item()):.4f} (0=SSM, 1=Hull)")

    # Test 5: Reset
    print("\n5. Reset")
    hybrid.reset()
    out_after_reset = hybrid(mx.random.normal((dim,)))
    mx.eval(out_after_reset)
    print(f"   After reset norm: {float(mx.sqrt(mx.sum(out_after_reset**2)).item()):.4f}")

    print("\n" + "=" * 60)
    print("All hybrid memory tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_memory()
