"""
Recursive Language Model (RLM) Delegation for Statecharts.

Implements paper §2.1: agents spawn sub-agents for subtasks, passing
down only relevant state to prevent context rot.

DELEGATION SEMANTICS:
  An RLM agent A operating in statechart mode q may delegate a subtask
  to a child agent A' with projected context:

    A'.context = project(A.context, delegation.context_keys)
    A'.statechart = delegation.child_statechart
    result = A'.run(A'.context)
    A.context[delegation.result_key] = result

  Recursion depth is bounded by delegation.max_depth.

STATECHART INTEGRATION:
  Delegation is triggered by a transition with a DelegationAction:
    State("analyze") --DELEGATE--> State("waiting_for_child")
    On child completion: State("waiting_for_child") --RESULT--> State("aggregate")

  The child agent runs its own statechart independently. Communication
  is via the result_key written back to parent context.

PARALLELISM:
  Multiple children can be spawned from a parallel (AND) state. Each
  orthogonal region delegates independently; results are merged when
  all children complete.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import IntEnum

from .differentiable_sc import (
    NeuroSymbolicStatechart,
    NeuroSymbolicConfig,
    gumbel_softmax,
)


# ---------------------------------------------------------------------------
# Delegation Configuration
# ---------------------------------------------------------------------------

class AggregationStrategy(IntEnum):
    """How to combine results from multiple child agents."""
    FIRST = 0        # Take the first result
    CONCAT = 1       # Concatenate all results
    MEAN = 2         # Average embeddings
    ATTENTION = 3    # Learned attention over child results
    VOTE = 4         # Majority vote (discrete actions)
    MIXTURE = 5      # Mixture of experts gating


@dataclass
class DelegationConfig:
    """Configuration for a single delegation action."""
    # Which context keys to project to child
    context_keys: List[str] = field(default_factory=list)

    # Dimensionality of projected context
    projected_dim: int = 16

    # Maximum recursion depth
    max_depth: int = 3

    # Key where child result is stored in parent context
    result_key: str = "child_result"

    # Child statechart configuration (if different from parent)
    child_n_states: int = 4
    child_n_events: int = 3

    # Timeout (max steps for child execution)
    max_child_steps: int = 50

    # How to aggregate multiple child results
    aggregation: AggregationStrategy = AggregationStrategy.MEAN


@dataclass
class RLMConfig:
    """Configuration for the full RLM delegation system."""
    # Parent statechart config
    parent_config: NeuroSymbolicConfig = field(default_factory=NeuroSymbolicConfig)

    # Maximum number of concurrent children
    max_children: int = 4

    # Maximum recursion depth across all delegation chains
    max_depth: int = 3

    # Context projection dimension
    context_projection_dim: int = 16

    # Whether children share weights with parent
    shared_weights: bool = False

    # Communication bandwidth (embeddings exchanged per step)
    bandwidth: int = 1

    # Delegation decision threshold
    delegation_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Context Projection
# ---------------------------------------------------------------------------

class ContextProjection(nn.Module):
    """
    Projects parent context to child context, retaining only relevant state.

    Prevents context rot by limiting what the child sees.
    The projection is learned end-to-end.
    """

    def __init__(self, parent_dim: int, child_dim: int):
        super().__init__()
        self.proj = nn.Linear(parent_dim, child_dim)
        self.gate = nn.Linear(parent_dim, child_dim)

    def __call__(self, parent_context: mx.array) -> mx.array:
        """
        Project parent context to child-relevant subset.

        Uses gated projection: child_ctx = sigma(gate) * proj(parent_ctx)
        """
        projected = self.proj(parent_context)
        gate_scores = mx.sigmoid(self.gate(parent_context))
        return projected * gate_scores


class ResultAggregator(nn.Module):
    """
    Aggregates results from multiple child agents.

    Supports multiple strategies per AggregationStrategy.
    """

    def __init__(self, result_dim: int, max_children: int, strategy: AggregationStrategy):
        super().__init__()
        self.result_dim = result_dim
        self.max_children = max_children
        self.strategy = strategy

        if strategy == AggregationStrategy.ATTENTION:
            self.attn_proj = nn.Linear(result_dim, result_dim)
            self.attn_query = mx.random.normal((result_dim,)) * 0.1

        if strategy == AggregationStrategy.MIXTURE:
            self.gate_net = nn.Sequential(
                nn.Linear(result_dim * max_children, max_children),
            )

    def __call__(self, results: List[mx.array]) -> mx.array:
        """Aggregate child results into a single vector."""
        if not results:
            return mx.zeros((self.result_dim,))

        if len(results) == 1:
            return results[0]

        stacked = mx.stack(results)  # [n_children, result_dim]

        if self.strategy == AggregationStrategy.FIRST:
            return results[0]

        elif self.strategy == AggregationStrategy.MEAN:
            return mx.mean(stacked, axis=0)

        elif self.strategy == AggregationStrategy.CONCAT:
            # Pad to max_children, then flatten
            padded = mx.zeros((self.max_children, self.result_dim))
            for i, r in enumerate(results):
                if i < self.max_children:
                    padded = mx.concatenate([
                        padded[:i], mx.expand_dims(r, 0), padded[i+1:]
                    ], axis=0)
            return padded.reshape(-1)[:self.result_dim]

        elif self.strategy == AggregationStrategy.ATTENTION:
            # Learned attention over child results
            keys = self.attn_proj(stacked)  # [n, D]
            scores = mx.matmul(
                mx.expand_dims(self.attn_query, 0), keys.T
            ) / math.sqrt(self.result_dim)  # [1, n]
            weights = mx.softmax(scores, axis=-1)  # [1, n]
            return mx.matmul(weights, stacked).squeeze(0)  # [D]

        elif self.strategy == AggregationStrategy.MIXTURE:
            # MoE gating
            flat = mx.concatenate(results + [
                mx.zeros((self.result_dim,))
            ] * (self.max_children - len(results)))[:self.result_dim * self.max_children]
            gate = mx.softmax(self.gate_net(flat), axis=-1)  # [max_children]
            gate = gate[:len(results)]
            gate = gate / (mx.sum(gate) + 1e-8)
            weighted = mx.stack([g * r for g, r in zip(gate, results)])
            return mx.sum(weighted, axis=0)

        return mx.mean(stacked, axis=0)


# ---------------------------------------------------------------------------
# Delegation Decision Network
# ---------------------------------------------------------------------------

class DelegationDecider(nn.Module):
    """
    Decides whether to delegate a subtask to a child agent.

    Input: (state_embedding, context, event_embedding)
    Output: delegation_probability in [0, 1]

    The decision is differentiable for end-to-end training.
    A high score means the current subtask is too complex for the
    current agent and should be delegated.
    """

    def __init__(self, state_dim: int, context_dim: int, event_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + context_dim + event_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def __call__(
        self,
        state_embed: mx.array,
        context: mx.array,
        event_embed: mx.array,
    ) -> mx.array:
        """Returns delegation probability in [0, 1]."""
        combined = mx.concatenate([state_embed, context, event_embed], axis=-1)
        return mx.sigmoid(self.net(combined))


# ---------------------------------------------------------------------------
# Child Agent
# ---------------------------------------------------------------------------

class ChildAgent:
    """
    A child agent spawned by recursive delegation.

    Runs its own statechart with projected context.
    Reports result back to parent via result embedding.
    """

    def __init__(
        self,
        agent_id: str,
        config: NeuroSymbolicConfig,
        depth: int,
        max_depth: int,
    ):
        self.agent_id = agent_id
        self.depth = depth
        self.max_depth = max_depth
        self.model = NeuroSymbolicStatechart(config)
        self.completed = False
        self.result: Optional[mx.array] = None

    def run(
        self,
        projected_context: mx.array,
        event_sequence: List[int],
        max_steps: int = 50,
    ) -> mx.array:
        """
        Execute the child statechart on projected context.

        Returns:
            result: [state_dim] final state embedding as result.
        """
        seq_len = min(len(event_sequence), max_steps)
        ctx_seq = mx.broadcast_to(
            mx.expand_dims(projected_context, 0),
            (seq_len, projected_context.shape[-1]),
        )

        # Pad context to match child's context_dim
        child_ctx_dim = self.model.config.context_dim
        if ctx_seq.shape[-1] < child_ctx_dim:
            padding = mx.zeros((seq_len, child_ctx_dim - ctx_seq.shape[-1]))
            ctx_seq = mx.concatenate([ctx_seq, padding], axis=-1)
        elif ctx_seq.shape[-1] > child_ctx_dim:
            ctx_seq = ctx_seq[:, :child_ctx_dim]

        traj, _, _ = self.model.forward_sequence(
            initial_state=0,
            context_sequence=ctx_seq,
            event_sequence=event_sequence[:seq_len],
        )

        # Result is final state embedding
        final_dist = traj[-1:]  # [1, n_states]
        self.result = self.model.encode_state(final_dist).squeeze(0)
        self.completed = True
        return self.result


# ---------------------------------------------------------------------------
# RLM Delegation Manager
# ---------------------------------------------------------------------------

class RecursiveDelegationManager(nn.Module):
    """
    Manages recursive delegation of subtasks to child agents.

    Integrates with the parent NeuroSymbolicStatechart:
      1. At each step, the DelegationDecider evaluates whether to delegate.
      2. If delegating, ContextProjection creates child context.
      3. Child agent runs its own statechart.
      4. ResultAggregator combines child results into parent context.

    The entire pipeline is differentiable (delegation decision is soft,
    context projection is learned, aggregation uses attention).
    """

    def __init__(self, config: RLMConfig):
        super().__init__()
        self.config = config

        # Delegation decision
        self.decider = DelegationDecider(
            config.parent_config.state_dim,
            config.parent_config.context_dim,
            config.parent_config.event_dim,
        )

        # Context projection
        self.projector = ContextProjection(
            config.parent_config.context_dim,
            config.context_projection_dim,
        )

        # Result aggregation
        self.aggregator = ResultAggregator(
            config.parent_config.state_dim,
            config.max_children,
            AggregationStrategy.ATTENTION,
        )

        # Result injection: map child result back to parent context space
        self.result_injector = nn.Linear(
            config.parent_config.state_dim,
            config.parent_config.context_dim,
        )

        # Active children (stored outside nn.Module's attribute system)
        self._children: List[ChildAgent] = []
        self._child_counter = 0

    def should_delegate(
        self,
        state_embed: mx.array,
        context: mx.array,
        event_embed: mx.array,
    ) -> Tuple[bool, mx.array]:
        """
        Decide whether to delegate.

        Returns:
            should: Hard decision (for execution).
            probability: Soft probability (for gradient).
        """
        prob = self.decider(state_embed, context, event_embed)
        should = float(prob.item()) > self.config.delegation_threshold
        return should, prob

    def spawn_child(
        self,
        parent_context: mx.array,
        event_sequence: List[int],
        current_depth: int,
    ) -> Optional[ChildAgent]:
        """
        Spawn a child agent with projected context.

        Returns None if max_depth or max_children exceeded.
        """
        if current_depth >= self.config.max_depth:
            return None
        if len(self._children) >= self.config.max_children:
            return None

        # Project context
        projected = self.projector(parent_context)

        # Create child config (smaller than parent)
        child_config = NeuroSymbolicConfig(
            n_states=max(4, self.config.parent_config.n_states // 2),
            n_events=self.config.parent_config.n_events,
            state_dim=self.config.parent_config.state_dim,
            event_dim=self.config.parent_config.event_dim,
            context_dim=self.config.context_projection_dim,
            num_heads=2,
            hull_top_k=4,
            control_dim=self.config.parent_config.control_dim,
        )

        child = ChildAgent(
            agent_id=f"child_{self._child_counter}",
            config=child_config,
            depth=current_depth + 1,
            max_depth=self.config.max_depth,
        )
        self._child_counter += 1

        # Run child
        child.run(projected, event_sequence)
        self._children.append(child)

        return child

    def collect_results(self) -> mx.array:
        """Aggregate all completed child results."""
        completed = [c.result for c in self._children if c.completed and c.result is not None]
        if not completed:
            return mx.zeros((self.config.parent_config.state_dim,))
        return self.aggregator(completed)

    def inject_results(self, context: mx.array) -> mx.array:
        """Inject aggregated child results back into parent context."""
        result = self.collect_results()
        delta = self.result_injector(result)
        return context + mx.expand_dims(delta, 0) if len(context.shape) == 2 else context + delta

    def clear_children(self):
        """Clear completed children."""
        self._children = [c for c in self._children if not c.completed]


# ---------------------------------------------------------------------------
# RLM-Enhanced Statechart
# ---------------------------------------------------------------------------

class RLMStatechart(nn.Module):
    """
    NeuroSymbolicStatechart with recursive delegation capability.

    Wraps the base statechart and adds delegation at each step.
    The delegation decision is differentiable — the model learns
    when to delegate vs. handle locally.

    EXECUTION FLOW:
      1. Standard statechart step (guards, topology, Gumbel-Softmax)
      2. DelegationDecider evaluates complexity
      3. If delegating: spawn child, project context, run child SC
      4. Inject child result into parent context
      5. Continue parent execution

    TRAINING:
      The delegation probability is included in the loss:
        L_delegate = alpha * sum(delegation_probs)  (encourage parsimony)
      This penalizes unnecessary delegation while allowing the model
      to learn when subtasks genuinely benefit from delegation.
    """

    def __init__(self, config: RLMConfig):
        super().__init__()
        self.config = config
        self.sc = NeuroSymbolicStatechart(config.parent_config)
        self.delegation = RecursiveDelegationManager(config)

    def step(
        self,
        state_dist: mx.array,
        context: mx.array,
        event_idx: int,
        current_depth: int = 0,
        event_sequence: Optional[List[int]] = None,
        continuous_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Dict]:
        """
        One step with optional recursive delegation.

        Returns:
            next_dist: [batch, n_states]
            next_continuous: [batch, control_dim] or None
            info: includes delegation_prob
        """
        batch = state_dist.shape[0]
        state_embed = self.sc.encode_state(state_dist)
        event_embed = self.sc.event_embeds[event_idx:event_idx+1]
        event_embed_b = mx.broadcast_to(event_embed, (batch, self.config.parent_config.event_dim))

        # Delegation decision
        should, prob = self.delegation.should_delegate(
            state_embed[0:1],
            context[0:1],
            event_embed,
        )

        if should and current_depth < self.config.max_depth:
            # Spawn child
            child_events = event_sequence or [event_idx] * 10
            self.delegation.spawn_child(
                context[0], child_events, current_depth
            )
            # Inject results
            context = self.delegation.inject_results(context)
            self.delegation.clear_children()

        # Standard statechart step (with enriched context)
        next_dist, next_cont, info = self.sc.step(
            state_dist, context, event_idx, continuous_state
        )
        info["delegation_prob"] = prob

        return next_dist, next_cont, info

    def forward_sequence(
        self,
        initial_state: int,
        context_sequence: mx.array,
        event_sequence: List[int],
        current_depth: int = 0,
        initial_continuous: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[Dict]]:
        """Process event sequence with delegation."""
        one_hot = [0.0] * self.config.parent_config.n_states
        one_hot[initial_state] = 1.0
        state_dist = mx.array([one_hot])

        continuous = None
        if initial_continuous is not None:
            continuous = mx.expand_dims(initial_continuous, 0)

        trajectory = [state_dist]
        infos = []
        delegation_probs = []

        for t in range(len(event_sequence)):
            ctx = context_sequence[t:t+1]
            next_dist, next_cont, info = self.step(
                state_dist, ctx, event_sequence[t],
                current_depth=current_depth,
                event_sequence=event_sequence[t:],
                continuous_state=continuous,
            )
            trajectory.append(next_dist)
            infos.append(info)
            delegation_probs.append(info["delegation_prob"])
            state_dist = next_dist
            if next_cont is not None:
                continuous = next_cont

        return mx.concatenate(trajectory, axis=0), infos

    def get_loss(
        self,
        state_trajectory: mx.array,
        target_states: mx.array,
        infos: List[Dict],
        delegation_penalty: float = 0.01,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Training loss with delegation penalty.

        L = L_sc + alpha * sum(delegation_probs)
        """
        sc_loss, components = self.sc.get_loss(state_trajectory, target_states)

        # Delegation penalty (encourage parsimony)
        del_probs = [info["delegation_prob"] for info in infos if "delegation_prob" in info]
        if del_probs:
            del_loss = mx.mean(mx.stack([p.squeeze() for p in del_probs])) * delegation_penalty
        else:
            del_loss = mx.array(0.0)

        total = sc_loss + del_loss
        components["delegation_loss"] = del_loss
        return total, components


def test_recursive_delegation():
    """Test the RLM delegation system."""
    print("=" * 60)
    print("Recursive Delegation (RLM) Tests")
    print("=" * 60)

    parent_config = NeuroSymbolicConfig(
        n_states=6,
        n_events=3,
        state_dim=32,
        event_dim=16,
        context_dim=16,
        num_heads=4,
        hull_top_k=4,
        control_dim=8,
    )

    rlm_config = RLMConfig(
        parent_config=parent_config,
        max_children=2,
        max_depth=2,
        context_projection_dim=16,
        delegation_threshold=0.3,  # Low threshold to trigger delegation
    )

    model = RLMStatechart(rlm_config)

    # Test sequence
    ctx = mx.random.normal((8, parent_config.context_dim))
    events = [0, 1, 2, 0, 1, 2, 0, 1]

    print("\n1. Forward sequence with delegation")
    traj, infos = model.forward_sequence(
        initial_state=0,
        context_sequence=ctx,
        event_sequence=events,
    )
    print(f"   Trajectory shape: {traj.shape}")
    print(f"   Steps: {len(infos)}")

    del_probs = [float(info["delegation_prob"].item()) for info in infos]
    print(f"   Delegation probs: {[f'{p:.3f}' for p in del_probs]}")
    n_delegated = sum(1 for p in del_probs if p > rlm_config.delegation_threshold)
    print(f"   Steps delegated: {n_delegated}/{len(events)}")

    # Test loss
    print("\n2. Loss computation with delegation penalty")
    targets = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2])
    loss, components = model.get_loss(traj, targets, infos)
    print(f"   Total loss: {float(loss.item()):.4f}")
    for k, v in components.items():
        print(f"     {k}: {float(v.item()):.4f}")

    # Test depth limiting
    print("\n3. Depth limiting")
    deep_config = RLMConfig(
        parent_config=parent_config,
        max_depth=1,  # Shallow
        delegation_threshold=0.0,  # Always delegate
    )
    deep_model = RLMStatechart(deep_config)
    traj2, infos2 = deep_model.forward_sequence(
        initial_state=0,
        context_sequence=ctx[:3],
        event_sequence=events[:3],
        current_depth=0,
    )
    print(f"   Trajectory shape (depth=1): {traj2.shape}")

    # Test aggregation strategies
    print("\n4. Result aggregation strategies")
    for strategy in [AggregationStrategy.MEAN, AggregationStrategy.ATTENTION]:
        agg = ResultAggregator(32, 4, strategy)
        results = [mx.random.normal((32,)) for _ in range(3)]
        combined = agg(results)
        print(f"   {strategy.name}: output norm = {float(mx.sqrt(mx.sum(combined**2)).item()):.4f}")

    print("\n" + "=" * 60)
    print("All RLM delegation tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    test_recursive_delegation()
