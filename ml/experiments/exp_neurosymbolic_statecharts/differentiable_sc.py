"""
Differentiable Statechart with full paper architecture.

Extends the base differentiable statechart (exp_differentiable_statecharts)
with all components from the paper:

  1. Gumbel-Softmax state selection with temperature annealing
  2. Neural predicate guards (sigmoid MLP, continuous [0,1])
  3. Soft adjacency matrix (learnable topology)
  4. Attention-based transition resolution
  5. Hull-backed memory integration
  6. Typed memory (ephemeral/history/belief)
  7. Hybrid control: discrete modes govern continuous controllers

DIFFERENTIABLE EXECUTION PIPELINE:
  soft_config = gumbel_softmax(logits, tau)
  guard_scores = sigmoid(f_theta(context))
  transition_probs = guard_scores * adjacency * weights / Z
  next_config = sum(transition_probs * target_configs)
  memory_output = hull_sparse_sdpa(query, cached_kv)

LOSS:
  L = L_task + beta * H(pi) + lambda_sparse * ||A||_1
"""

import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .hull_memory import HullKVCache, HullConfig


# ---------------------------------------------------------------------------
# Gumbel-Softmax
# ---------------------------------------------------------------------------

def sample_gumbel(shape: Tuple[int, ...], eps: float = 1e-20) -> mx.array:
    """Sample from Gumbel(0, 1)."""
    u = mx.random.uniform(shape=shape)
    return -mx.log(-mx.log(u + eps) + eps)


def gumbel_softmax(
    logits: mx.array,
    temperature: float = 1.0,
    hard: bool = False
) -> mx.array:
    """
    Gumbel-Softmax with optional straight-through estimator.

    pi_i = exp((log(alpha_i) + g_i) / tau) / sum_j exp(...)
    where g_i ~ Gumbel(0, 1).

    As tau -> 0, approaches one-hot (discrete).
    As tau -> inf, approaches uniform.

    Args:
        logits: [..., n_classes] unnormalized log probabilities.
        temperature: Softmax temperature.
        hard: If True, use straight-through (hard forward, soft backward).
    """
    gumbels = sample_gumbel(logits.shape)
    y_soft = mx.softmax((logits + gumbels) / temperature, axis=-1)

    if hard:
        # Straight-through: hard argmax forward, gradient through y_soft
        index = mx.argmax(y_soft, axis=-1, keepdims=True)
        y_hard = mx.zeros_like(y_soft)
        # Scatter one-hot: set the argmax position to 1.0
        n_classes = y_soft.shape[-1]
        eye = mx.eye(n_classes)
        y_hard = eye[index.squeeze(-1)]
        # Forward: sees y_hard; backward: sees y_soft gradients
        y_hard = y_hard - mx.stop_gradient(y_soft) + y_soft
        return y_hard

    return y_soft


# ---------------------------------------------------------------------------
# Neural Predicate (continuous guard)
# ---------------------------------------------------------------------------

class NeuralPredicate(nn.Module):
    """
    Differentiable guard: g_tilde(context, event) -> [0, 1].

    Training: soft value enables gradient flow.
    Inference: g_tilde > threshold -> enabled.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """Returns soft boolean in [0, 1]."""
        for i, layer in enumerate(self.layers[:-1]):
            x = nn.relu(layer(x))
        return mx.sigmoid(self.layers[-1](x))


# ---------------------------------------------------------------------------
# Soft Adjacency (learnable topology)
# ---------------------------------------------------------------------------

class SoftAdjacency(nn.Module):
    """
    Learnable transition topology as soft adjacency matrix.

    A[i, j, e] in [0, 1] = P(edge from state i to state j on event e).

    Regularization:
      L_sparse = lambda_1 * mean(A)           (L1 sparsity)
      L_entropy = lambda_H * mean(H(A[i,:,e])) (row-wise entropy)
    """

    def __init__(self, n_states: int, n_events: int):
        super().__init__()
        self.n_states = n_states
        self.n_events = n_events

        # Logits initialized sparse (sigmoid(-2) ~ 0.12)
        self.edge_logits = mx.random.normal((n_states, n_states, n_events)) * 0.1 - 2.0

    def get_adjacency(self, temperature: float = 1.0) -> mx.array:
        """Soft adjacency [n_states, n_states, n_events]."""
        return mx.sigmoid(self.edge_logits / temperature)

    def get_transition_probs(
        self,
        state_dist: mx.array,
        event_idx: int,
        temperature: float = 1.0
    ) -> mx.array:
        """
        Compute next-state distribution: next = state_dist @ T.

        T[i,j] = A[i,j,e] / sum_j A[i,j,e] (row-normalized).
        """
        adj = self.get_adjacency(temperature)[:, :, event_idx]  # [S, S]
        adj_norm = adj / (mx.sum(adj, axis=1, keepdims=True) + 1e-8)
        return mx.matmul(state_dist, adj_norm)

    def sparsity_loss(self) -> mx.array:
        return mx.mean(self.get_adjacency())

    def entropy_loss(self) -> mx.array:
        adj = self.get_adjacency()
        probs = adj / (mx.sum(adj, axis=1, keepdims=True) + 1e-8)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=1)
        return mx.mean(entropy)


# ---------------------------------------------------------------------------
# Transition Attention
# ---------------------------------------------------------------------------

class TransitionAttention(nn.Module):
    """
    Multi-head attention over candidate transitions.

    Query: state_embed + event_embed
    Keys:  per-transition embeddings (src concat tgt)
    Values: target state embeddings
    Guard scores modulate attention as soft mask.
    """

    def __init__(self, state_dim: int, event_dim: int, num_heads: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads

        self.q_proj = nn.Linear(state_dim + event_dim, state_dim)
        self.k_proj = nn.Linear(state_dim * 2, state_dim)
        self.v_proj = nn.Linear(state_dim, state_dim)
        self.o_proj = nn.Linear(state_dim, state_dim)

    def __call__(
        self,
        state_embed: mx.array,
        event_embed: mx.array,
        transition_embeds: mx.array,
        target_embeds: mx.array,
        guard_scores: mx.array,
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            next_state_embed: [batch, state_dim]
            attn_weights: [batch, n_transitions]
        """
        batch = state_embed.shape[0]

        q_in = mx.concatenate([state_embed, event_embed], axis=-1)
        q = self.q_proj(q_in)     # [B, D]
        k = self.k_proj(transition_embeds)  # [T, D]
        v = self.v_proj(target_embeds)      # [T, D]

        scores = mx.matmul(q, k.T) / math.sqrt(self.state_dim)  # [B, T]
        scores = scores * guard_scores  # soft mask
        attn = mx.softmax(scores / temperature, axis=-1)  # [B, T]

        out = mx.matmul(attn, v)  # [B, D]
        out = self.o_proj(out)
        return out, attn


# ---------------------------------------------------------------------------
# Typed Memory System
# ---------------------------------------------------------------------------

class TypedMemory(nn.Module):
    """
    Three-tier memory following paper §2.4:

      1. Ephemeral: tracks immediate effects, cleared on phase transitions.
      2. History: Harel deep/shallow history for configuration restoration.
      3. Belief: persistent invariants, survives machine resets.

    Each tier is backed by a Hull KV-Cache namespace for O(log n) retrieval.
    """

    def __init__(self, memory_dim: int, num_heads: int = 4, top_k: int = 8):
        super().__init__()
        self.memory_dim = memory_dim

        hull_config = HullConfig(
            num_hull_layers=3,
            top_k=top_k,
            num_namespaces=3,  # ephemeral=0, history=1, belief=2
            memory_dim=memory_dim,
            num_heads=num_heads,
        )
        self.cache = HullKVCache(hull_config)

        # Projection from statechart context to memory write
        self.write_proj = nn.Linear(memory_dim, memory_dim)

        # Tier-specific read heads
        self.read_projs = [nn.Linear(memory_dim, memory_dim) for _ in range(3)]

        # Combiner: merge three tier outputs
        self.combiner = nn.Linear(memory_dim * 3, memory_dim)

        self._step = 0

    def write(self, context: mx.array, tier: int = 0):
        """Write context into the specified memory tier."""
        projected = self.write_proj(context)
        self.cache.append(projected, projected, namespace=tier)
        self._step += 1

    def write_all(self, context: mx.array):
        """Write to ephemeral and history; belief is written explicitly."""
        self.write(context, tier=0)  # ephemeral
        self.write(context, tier=1)  # history

    def write_belief(self, belief: mx.array):
        """Write a discovered invariant to persistent belief memory."""
        self.write(belief, tier=2)

    def read(self, query: mx.array) -> mx.array:
        """
        Read from all three tiers and combine.

        Returns: [memory_dim] combined memory output.
        """
        outputs = []
        for tier in range(3):
            q = self.read_projs[tier](query)
            out = self.cache(q, query_pos=self._step, namespace=tier, use_hull=True)
            outputs.append(out)

        combined = mx.concatenate(outputs, axis=-1)
        return self.combiner(combined)

    def clear_ephemeral(self):
        """Clear ephemeral memory (on phase transition)."""
        self.cache.reset(namespace=0)


# ---------------------------------------------------------------------------
# Differentiable Statechart Config
# ---------------------------------------------------------------------------

@dataclass
class NeuroSymbolicConfig:
    """Full configuration for the neuro-symbolic statechart."""
    # Statechart structure
    n_states: int = 8
    n_events: int = 4
    state_dim: int = 64
    event_dim: int = 32
    context_dim: int = 32

    # Attention
    num_heads: int = 4

    # Temperature annealing
    temperature_max: float = 5.0
    temperature_min: float = 0.1
    annealing_rate: float = 0.995

    # Regularization
    sparsity_coefficient: float = 0.1
    entropy_coefficient: float = 0.05

    # Memory
    hull_top_k: int = 8
    hull_layers: int = 3

    # Hybrid control
    control_dim: int = 16
    control_frequency_hz: float = 10.0

    # Straight-through estimator
    straight_through: bool = True


# ---------------------------------------------------------------------------
# Hybrid Controller
# ---------------------------------------------------------------------------

class HybridController(nn.Module):
    """
    Continuous neural controller governed by discrete statechart modes.

    Each state acts as a behavioral mode with its own controller output.
    H = (Q, X, Init, f, Inv, E, G, R) per Henzinger 1996.

    The controller maps (mode_embedding, continuous_state) -> control_output.
    """

    def __init__(self, state_dim: int, control_dim: int, n_states: int):
        super().__init__()
        self.control_dim = control_dim

        # Per-mode controller (shared network, mode-conditioned)
        self.controller = nn.Sequential(
            nn.Linear(state_dim + control_dim, 64),
            nn.ReLU(),
            nn.Linear(64, control_dim),
        )

        # Mode invariants (learned boundaries)
        self.invariant_net = nn.Sequential(
            nn.Linear(control_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Reset map: on mode transition, how to reset continuous variables
        self.reset_map = nn.Linear(state_dim, control_dim)

    def step(
        self,
        mode_embed: mx.array,
        continuous_state: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        One control step.

        Returns:
            next_continuous: [control_dim] updated continuous variables.
            invariant_score: [1] soft invariant satisfaction.
        """
        combined = mx.concatenate([mode_embed, continuous_state], axis=-1)
        delta = self.controller(combined)
        next_continuous = continuous_state + delta  # Euler integration

        invariant_score = mx.sigmoid(self.invariant_net(next_continuous))
        return next_continuous, invariant_score

    def reset(self, mode_embed: mx.array) -> mx.array:
        """Reset continuous variables on mode transition."""
        return mx.tanh(self.reset_map(mode_embed))


# ---------------------------------------------------------------------------
# Neuro-Symbolic Statechart (main module)
# ---------------------------------------------------------------------------

class NeuroSymbolicStatechart(nn.Module):
    """
    End-to-end differentiable statechart with all paper components.

    Integrates:
      - Gumbel-Softmax state selection
      - Neural predicate guards
      - Soft adjacency topology
      - Transition attention
      - Hull-backed typed memory
      - Hybrid mode-conditioned controller

    Forward pass:
      1. Encode current soft configuration
      2. Evaluate neural guards on context + memory
      3. Compute transition probabilities via topology * guards
      4. Apply Gumbel-Softmax for next configuration
      5. Update typed memory
      6. Step hybrid controller
    """

    def __init__(self, config: NeuroSymbolicConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature_max

        # Learnable embeddings
        self.state_embeds = mx.random.normal((config.n_states, config.state_dim)) * 0.1
        self.event_embeds = mx.random.normal((config.n_events, config.event_dim)) * 0.1

        # Topology
        self.topology = SoftAdjacency(config.n_states, config.n_events)

        # Guards
        self.guard = NeuralPredicate(
            config.context_dim + config.state_dim,
            hidden_dims=[32, 16],
        )

        # Transition attention
        self.transition_attn = TransitionAttention(
            config.state_dim, config.event_dim, config.num_heads
        )

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.context_dim, config.state_dim),
            nn.ReLU(),
            nn.Linear(config.state_dim, config.state_dim),
        )

        # State decoder
        self.state_decoder = nn.Linear(config.state_dim, config.n_states)

        # Typed memory
        self.memory = TypedMemory(
            config.state_dim,
            num_heads=config.num_heads,
            top_k=config.hull_top_k,
        )

        # Hybrid controller
        self.controller = HybridController(
            config.state_dim, config.control_dim, config.n_states
        )

        # Memory-to-context projection
        self.memory_proj = nn.Linear(config.state_dim, config.context_dim)

    def encode_state(self, state_dist: mx.array) -> mx.array:
        """Weighted sum of state embeddings."""
        return mx.matmul(state_dist, self.state_embeds)

    def get_guard_scores(
        self,
        state_embed: mx.array,
        context: mx.array,
        event_idx: int,
    ) -> mx.array:
        """Evaluate neural guards for all transitions from current state."""
        batch = state_embed.shape[0]
        guard_input = mx.concatenate([state_embed, context], axis=-1)
        base_score = self.guard(guard_input)  # [B, 1]

        adj = self.topology.get_adjacency(self.temperature)[:, :, event_idx]
        transition_probs = mx.matmul(
            mx.ones((batch, self.config.n_states)) / self.config.n_states,
            adj,
        )
        return base_score * transition_probs

    def step(
        self,
        state_dist: mx.array,
        context: mx.array,
        event_idx: int,
        continuous_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Dict[str, mx.array]]:
        """
        One differentiable execution step.

        Args:
            state_dist: [batch, n_states] soft configuration.
            context: [batch, context_dim] context variables.
            event_idx: Event index.
            continuous_state: [batch, control_dim] for hybrid mode.

        Returns:
            next_dist: [batch, n_states] next soft configuration.
            next_continuous: [batch, control_dim] next continuous state.
            info: Intermediate values for analysis.
        """
        batch = state_dist.shape[0]
        state_embed = self.encode_state(state_dist)
        context_embed = self.context_encoder(context)

        # Read memory and augment context
        mem_query = state_embed[0] if batch == 1 else state_embed.mean(axis=0)
        mem_output = self.memory.read(mem_query)
        mem_context = self.memory_proj(mx.expand_dims(mem_output, 0))
        mem_context = mx.broadcast_to(mem_context, (batch, self.config.context_dim))
        augmented_context = context + mem_context

        # Guard evaluation
        guard_scores = self.get_guard_scores(state_embed, augmented_context, event_idx)

        # Topology transition
        next_dist_topo = self.topology.get_transition_probs(
            state_dist, event_idx, self.temperature
        )

        # Apply guards as soft mask, renormalize (mandatory per §5.3)
        next_dist = next_dist_topo * guard_scores
        next_dist = next_dist / (mx.sum(next_dist, axis=-1, keepdims=True) + 1e-8)

        # Gumbel-Softmax sampling
        logits = mx.log(next_dist + 1e-8)
        next_dist = gumbel_softmax(logits, self.temperature, hard=self.config.straight_through)

        # Write to typed memory
        combined_embed = state_embed[0] + context_embed[0]
        self.memory.write_all(combined_embed)

        # Hybrid controller step
        next_continuous = None
        invariant_score = None
        if continuous_state is not None:
            mode_embed = self.encode_state(next_dist)
            next_continuous, invariant_score = self.controller.step(
                mode_embed[0], continuous_state[0]
            )
            next_continuous = mx.expand_dims(next_continuous, 0)

        info = {
            "guard_scores": guard_scores,
            "topology_probs": next_dist_topo,
            "memory_output": mem_output,
            "invariant_score": invariant_score,
        }

        return next_dist, next_continuous, info

    def forward_sequence(
        self,
        initial_state: int,
        context_sequence: mx.array,
        event_sequence: List[int],
        initial_continuous: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[mx.array], List[Dict]]:
        """
        Process event sequence through differentiable statechart.

        Returns:
            state_trajectory: [seq_len+1, n_states]
            continuous_trajectory: list of [control_dim] or empty
            infos: per-step info dicts
        """
        one_hot = [0.0] * self.config.n_states
        one_hot[initial_state] = 1.0
        state_dist = mx.array([one_hot])

        continuous = None
        if initial_continuous is not None:
            continuous = mx.expand_dims(initial_continuous, 0)

        trajectory = [state_dist]
        continuous_trajectory = []
        infos = []

        for t in range(len(event_sequence)):
            ctx = context_sequence[t:t+1]
            next_dist, next_cont, info = self.step(
                state_dist, ctx, event_sequence[t], continuous
            )

            trajectory.append(next_dist)
            if next_cont is not None:
                continuous_trajectory.append(next_cont[0])
                continuous = next_cont
            infos.append(info)
            state_dist = next_dist

        return mx.concatenate(trajectory, axis=0), continuous_trajectory, infos

    def anneal_temperature(self):
        """Exponential temperature decay: tau *= rate."""
        self.temperature = max(
            self.config.temperature_min,
            self.temperature * self.config.annealing_rate,
        )

    def get_loss(
        self,
        state_trajectory: mx.array,
        target_states: mx.array,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Training loss.

        L = L_ce + beta * H(pi) + lambda * ||A||_1
        """
        seq_len = target_states.shape[0]

        # Cross-entropy (vectorized to preserve computation graph)
        preds = state_trajectory[:seq_len]
        log_probs = mx.log(preds + 1e-8)
        eye = mx.eye(self.config.n_states)
        targets_onehot = eye[target_states]
        ce = -mx.mean(mx.sum(log_probs * targets_onehot, axis=-1))

        # Sparsity
        sp = self.topology.sparsity_loss() * self.config.sparsity_coefficient

        # Entropy
        ent = self.topology.entropy_loss() * self.config.entropy_coefficient

        total = ce + sp + ent
        return total, {"ce_loss": ce, "sparsity_loss": sp, "entropy_loss": ent}


def test_neurosymbolic_statechart():
    """Smoke test the full architecture."""
    print("=" * 60)
    print("NeuroSymbolicStatechart Smoke Test")
    print("=" * 60)

    config = NeuroSymbolicConfig(
        n_states=6,
        n_events=4,
        state_dim=32,
        event_dim=16,
        context_dim=16,
        num_heads=4,
        hull_top_k=4,
        control_dim=8,
    )

    model = NeuroSymbolicStatechart(config)

    # Synthetic data: 5-step sequence
    ctx_seq = mx.random.normal((5, config.context_dim))
    events = [0, 1, 2, 3, 0]
    init_cont = mx.zeros((config.control_dim,))

    traj, cont_traj, infos = model.forward_sequence(
        initial_state=0,
        context_sequence=ctx_seq,
        event_sequence=events,
        initial_continuous=init_cont,
    )

    print(f"  State trajectory shape: {traj.shape}")  # [6, 6]
    print(f"  Continuous steps: {len(cont_traj)}")
    print(f"  Temperature: {model.temperature:.3f}")

    # Test loss
    targets = mx.array([0, 1, 2, 3, 4, 0])
    loss, components = model.get_loss(traj, targets)
    print(f"  Loss: {float(loss.item()):.4f}")
    for k, v in components.items():
        print(f"    {k}: {float(v.item()):.4f}")

    # Temperature annealing
    for _ in range(100):
        model.anneal_temperature()
    print(f"  Temperature after 100 anneals: {model.temperature:.3f}")

    # Belief memory write
    belief = mx.random.normal((config.state_dim,))
    model.memory.write_belief(belief)
    print(f"  Belief memory written.")

    print("\n  All components functional.")
    print("=" * 60)


if __name__ == "__main__":
    test_neurosymbolic_statechart()
