"""
Attention-Based Transition Priority Selection

Uses transformer attention to learn which transition should fire when multiple
are enabled. Replaces the specificity heuristic from exp_transition_priorities
with a learned, context-aware attention mechanism.

Key insight: When multiple transitions are enabled, the "right" choice depends
on the current context. A learned attention mechanism can capture complex
context-dependent priority patterns that hand-coded heuristics cannot.

Architecture:
- TransitionEncoder: Embeds transitions (guard type, source, target, params)
- ContextEncoder: Embeds execution context (grid, state variables)
- PriorityAttention: Cross-attention where context attends to transitions
- Output: Attention weights become transition selection probabilities

Training:
- Can use evolution (like exp_transition_priorities)
- Can use gradient descent (differentiable attention)
- Supervision from oracle priority decisions
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from abc import ABC, abstractmethod
import random
import math
import json


# =============================================================================
# Core Types (reused from exp_transition_priorities)
# =============================================================================

@dataclass
class Guard:
    """Guard condition with embedding support."""
    predicate: str
    params: Dict[str, Any] = field(default_factory=dict)

    # Guard type vocabulary
    PREDICATES = [
        "true", "false", "has_nonzero", "all_zero",
        "count_gt", "count_lt", "has_color",
        "state_eq", "state_gt", "state_lt",
    ]

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate guard in context."""
        grid = context.get("grid")
        state = context.get("state", {})

        if self.predicate == "true":
            return True
        elif self.predicate == "false":
            return False
        elif self.predicate == "has_nonzero":
            return bool(mx.any(grid != 0)) if grid is not None else False
        elif self.predicate == "all_zero":
            return bool(mx.all(grid == 0)) if grid is not None else True
        elif self.predicate == "count_gt":
            threshold = self.params.get("threshold", 0)
            return bool(mx.sum(grid != 0) > threshold) if grid is not None else False
        elif self.predicate == "count_lt":
            threshold = self.params.get("threshold", 5)
            return bool(mx.sum(grid != 0) < threshold) if grid is not None else True
        elif self.predicate == "has_color":
            color = self.params.get("color", 1)
            return bool(mx.any(grid == color)) if grid is not None else False
        elif self.predicate == "state_eq":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key) == value
        elif self.predicate == "state_gt":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key, 0) > value
        elif self.predicate == "state_lt":
            key = self.params.get("key", "x")
            value = self.params.get("value", 0)
            return state.get(key, 0) < value
        return True

    def to_embedding_indices(self) -> Tuple[int, float, float]:
        """Convert guard to embedding indices.

        Returns:
            (predicate_idx, param1_normalized, param2_normalized)
        """
        pred_idx = self.PREDICATES.index(self.predicate) if self.predicate in self.PREDICATES else 0

        # Normalize parameters to [0, 1]
        threshold = self.params.get("threshold", 0) / 10.0
        color = self.params.get("color", 0) / 5.0
        value = self.params.get("value", 0) / 10.0

        return pred_idx, max(threshold, color, value), 0.0


@dataclass
class Transition:
    """A transition with attention-based priority."""
    source: str
    target: str
    guard: Guard

    # Explicit priority (can be used as prior)
    priority: float = 0.0

    # Tracking
    fire_count: int = 0
    attention_scores: List[float] = field(default_factory=list)

    def is_enabled(self, context: Dict[str, Any]) -> bool:
        """Check if transition is enabled."""
        return self.guard.evaluate(context)


# =============================================================================
# Attention Modules
# =============================================================================

class TransitionEncoder(nn.Module):
    """Encodes transitions into embeddings."""

    def __init__(self, embed_dim: int = 64, n_predicates: int = 10, n_states: int = 20):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings
        self.predicate_embed = nn.Embedding(n_predicates, embed_dim // 2)
        self.state_embed = nn.Embedding(n_states, embed_dim // 4)

        # Project to final embedding
        # Input: predicate_embed (32) + source_embed (16) + target_embed (16) + params (8) = 72
        self.proj = nn.Linear(embed_dim // 2 + embed_dim // 4 + embed_dim // 4 + 8, embed_dim)

    def __call__(
        self,
        pred_indices: mx.array,  # [B, N]
        source_indices: mx.array,  # [B, N]
        target_indices: mx.array,  # [B, N]
        params: mx.array,  # [B, N, 8] - guard parameters
    ) -> mx.array:
        """Encode transitions.

        Args:
            pred_indices: Predicate type indices
            source_indices: Source state indices
            target_indices: Target state indices
            params: Guard parameters (threshold, value, etc.)

        Returns:
            Transition embeddings [B, N, embed_dim]
        """
        pred_emb = self.predicate_embed(pred_indices)  # [B, N, 32]
        src_emb = self.state_embed(source_indices)  # [B, N, 16]
        tgt_emb = self.state_embed(target_indices)  # [B, N, 16]

        # Concatenate all features
        combined = mx.concatenate([pred_emb, src_emb, tgt_emb, params], axis=-1)

        return self.proj(combined)


class ContextEncoder(nn.Module):
    """Encodes execution context into embeddings."""

    def __init__(self, embed_dim: int = 64, grid_size: int = 5, n_state_vars: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Grid encoder (simple CNN)
        self.grid_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.grid_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.grid_pool = nn.AvgPool2d(kernel_size=grid_size)  # Global avg pool

        # State variable encoder
        self.state_proj = nn.Linear(n_state_vars, embed_dim // 2)

        # Combine grid and state
        self.combine = nn.Linear(32 + embed_dim // 2, embed_dim)

    def __call__(
        self,
        grid: mx.array,  # [B, H, W] or [B, H, W, 1]
        state_vars: mx.array,  # [B, n_state_vars]
    ) -> mx.array:
        """Encode context.

        Args:
            grid: Grid state
            state_vars: State variable values

        Returns:
            Context embedding [B, embed_dim]
        """
        # MLX Conv2d uses NHWC format (Batch, Height, Width, Channels)
        if grid.ndim == 3:
            grid = mx.expand_dims(grid, axis=-1)  # [B, H, W, 1]

        # Grid features (NHWC format)
        x = nn.relu(self.grid_conv1(grid))  # [B, H, W, 16]
        x = nn.relu(self.grid_conv2(x))  # [B, H, W, 32]

        # Global average pool - over spatial dimensions (H, W)
        # x shape: [B, H, W, C]
        grid_feat = mx.mean(x, axis=(1, 2))  # [B, 32]

        # State features
        state_feat = nn.relu(self.state_proj(state_vars))  # [B, 32]

        # Combine
        combined = mx.concatenate([grid_feat, state_feat], axis=-1)
        return self.combine(combined)


class PriorityAttention(nn.Module):
    """
    Cross-attention for transition priority selection.

    Context (Q) attends to Transitions (K, V) to produce attention weights
    that serve as transition selection probabilities.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Query from context
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Key and Value from transitions
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Final score head
        self.score_head = nn.Linear(embed_dim, 1)

        # Temperature for softmax (learnable)
        self.temperature = mx.array([1.0])

    def __call__(
        self,
        context_emb: mx.array,  # [B, embed_dim]
        transition_emb: mx.array,  # [B, N, embed_dim]
        mask: Optional[mx.array] = None,  # [B, N] - True for enabled transitions
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute attention-based priority scores.

        Args:
            context_emb: Context embedding
            transition_emb: Transition embeddings
            mask: Mask for enabled transitions (False = disabled)

        Returns:
            (priority_scores, attention_weights)
            - priority_scores: [B, N] scores for each transition
            - attention_weights: [B, N] normalized attention weights
        """
        B, N, _ = transition_emb.shape

        # Expand context to match transitions
        context_emb = mx.expand_dims(context_emb, axis=1)  # [B, 1, D]

        # Project to Q, K, V
        Q = self.q_proj(context_emb)  # [B, 1, D]
        K = self.k_proj(transition_emb)  # [B, N, D]
        V = self.v_proj(transition_emb)  # [B, N, D]

        # Reshape for multi-head attention
        Q = Q.reshape(B, 1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # [B, H, 1, D/H]
        K = K.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # [B, H, N, D/H]
        V = V.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # [B, H, N, D/H]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # [B, H, 1, N]

        # Apply mask (set disabled transitions to -inf)
        if mask is not None:
            mask = mx.expand_dims(mx.expand_dims(mask, axis=1), axis=1)  # [B, 1, 1, N]
            scores = mx.where(mask, scores, mx.array([-1e9]))

        # Softmax with temperature
        attn_weights = mx.softmax(scores / self.temperature, axis=-1)  # [B, H, 1, N]

        # Attention output
        attn_out = mx.matmul(attn_weights, V)  # [B, H, 1, D/H]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, self.embed_dim)  # [B, 1, D]
        attn_out = self.out_proj(attn_out)  # [B, 1, D]

        # Combine attention output with transitions for final scores
        combined = transition_emb + attn_out  # [B, N, D]
        priority_scores = self.score_head(combined).squeeze(-1)  # [B, N]

        # Average attention weights across heads
        attn_weights_avg = mx.mean(attn_weights, axis=1).squeeze(1)  # [B, N]

        return priority_scores, attn_weights_avg


class PriorityAttentionModel(nn.Module):
    """
    Complete model for attention-based transition priority selection.

    Combines:
    - TransitionEncoder: Embeds each transition
    - ContextEncoder: Embeds the current execution context
    - PriorityAttention: Cross-attention to select transition
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_predicates: int = 10,
        n_states: int = 20,
        grid_size: int = 5,
        n_state_vars: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders
        self.transition_encoder = TransitionEncoder(
            embed_dim=embed_dim,
            n_predicates=n_predicates,
            n_states=n_states,
        )
        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            grid_size=grid_size,
            n_state_vars=n_state_vars,
        )

        # Attention
        self.priority_attention = PriorityAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
        )

    def __call__(
        self,
        pred_indices: mx.array,
        source_indices: mx.array,
        target_indices: mx.array,
        guard_params: mx.array,
        grid: mx.array,
        state_vars: mx.array,
        enabled_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            pred_indices: [B, N] predicate type indices
            source_indices: [B, N] source state indices
            target_indices: [B, N] target state indices
            guard_params: [B, N, 8] guard parameters
            grid: [B, H, W] grid state
            state_vars: [B, n_state_vars] state variables
            enabled_mask: [B, N] which transitions are enabled

        Returns:
            (priority_scores, attention_weights)
        """
        # Encode transitions
        trans_emb = self.transition_encoder(
            pred_indices, source_indices, target_indices, guard_params
        )

        # Encode context
        ctx_emb = self.context_encoder(grid, state_vars)

        # Compute attention-based priorities
        return self.priority_attention(ctx_emb, trans_emb, enabled_mask)


# =============================================================================
# Attention Priority Strategy
# =============================================================================

class AttentionPriorityStrategy:
    """
    Priority strategy that uses a learned attention model.

    Replaces the specificity heuristic with learned context-dependent attention.
    """

    def __init__(
        self,
        model: Optional[PriorityAttentionModel] = None,
        embed_dim: int = 64,
        state_vocab: Optional[Dict[str, int]] = None,
    ):
        self.model = model or PriorityAttentionModel(embed_dim=embed_dim)
        self.state_vocab = state_vocab or {}
        self._next_state_id = 0

    def get_state_id(self, state_name: str) -> int:
        """Get or create state ID."""
        if state_name not in self.state_vocab:
            self.state_vocab[state_name] = self._next_state_id
            self._next_state_id += 1
        return self.state_vocab[state_name]

    def prepare_inputs(
        self,
        transitions: List[Transition],
        context: Dict[str, Any],
    ) -> Tuple[mx.array, ...]:
        """Prepare model inputs from transitions and context."""
        N = len(transitions)

        # Transition features
        pred_indices = []
        source_indices = []
        target_indices = []
        guard_params = []

        for t in transitions:
            pred_idx, p1, p2 = t.guard.to_embedding_indices()
            pred_indices.append(pred_idx)
            source_indices.append(self.get_state_id(t.source))
            target_indices.append(self.get_state_id(t.target))

            # Pack params into 8-dim vector
            params = [
                p1, p2,
                t.guard.params.get("threshold", 0) / 10.0,
                t.guard.params.get("color", 0) / 5.0,
                t.guard.params.get("value", 0) / 10.0,
                t.priority / 10.0,  # Include explicit priority as feature
                0.0, 0.0,  # Reserved
            ]
            guard_params.append(params)

        # Context features
        grid = context.get("grid")
        if grid is None:
            grid = mx.zeros((5, 5))
        # Ensure grid is float for convolution
        grid = grid.astype(mx.float32)

        state = context.get("state", {})
        state_vars = [
            state.get("x", 0) / 10.0,
            state.get("y", 0) / 10.0,
            state.get("z", 0) / 10.0,
            state.get("count", 0) / 100.0,
            0.0, 0.0, 0.0, 0.0,  # Reserved
        ]

        # Enabled mask
        enabled_mask = mx.array([t.is_enabled(context) for t in transitions])

        # Add batch dimension
        return (
            mx.expand_dims(mx.array(pred_indices), axis=0),
            mx.expand_dims(mx.array(source_indices), axis=0),
            mx.expand_dims(mx.array(target_indices), axis=0),
            mx.expand_dims(mx.array(guard_params), axis=0),
            mx.expand_dims(grid, axis=0),
            mx.expand_dims(mx.array(state_vars), axis=0),
            mx.expand_dims(enabled_mask, axis=0),
        )

    def compute_priorities(
        self,
        transitions: List[Transition],
        context: Dict[str, Any],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute priority scores for all transitions.

        Returns:
            (scores, attention_weights) - Both as lists
        """
        inputs = self.prepare_inputs(transitions, context)
        scores, attn = self.model(*inputs)

        # Remove batch dimension
        scores = scores.squeeze(0).tolist()
        attn = attn.squeeze(0).tolist()

        return scores, attn

    def select_transition(
        self,
        transitions: List[Transition],
        context: Dict[str, Any],
        deterministic: bool = True,
    ) -> Tuple[Transition, float]:
        """
        Select which transition to fire.

        Args:
            transitions: List of enabled transitions
            context: Current execution context
            deterministic: If True, select highest score; else sample

        Returns:
            (selected_transition, attention_score)
        """
        if len(transitions) == 0:
            raise ValueError("No transitions to select from")

        if len(transitions) == 1:
            return transitions[0], 1.0

        scores, attn = self.compute_priorities(transitions, context)

        if deterministic:
            idx = max(range(len(scores)), key=lambda i: scores[i])
        else:
            # Sample proportional to softmax of scores
            probs = mx.softmax(mx.array(scores))
            idx = int(mx.argmax(mx.random.categorical(mx.log(probs + 1e-10))))

        # Track attention for analysis
        transitions[idx].attention_scores.append(attn[idx])

        return transitions[idx], attn[idx]


# =============================================================================
# Statechart with Attention Priority
# =============================================================================

@dataclass
class AttentionStatechart:
    """Statechart using attention-based priority resolution."""

    states: Set[str] = field(default_factory=set)
    transitions: List[Transition] = field(default_factory=list)
    initial_state: str = "start"
    final_states: Set[str] = field(default_factory=set)

    current_state: str = ""
    strategy: Optional[AttentionPriorityStrategy] = None

    # Tracking
    conflicts_resolved: int = 0
    attention_history: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.current_state:
            self.current_state = self.initial_state
        if self.strategy is None:
            self.strategy = AttentionPriorityStrategy()

    def reset(self):
        """Reset to initial state."""
        self.current_state = self.initial_state
        self.conflicts_resolved = 0
        self.attention_history = []

    def get_enabled_transitions(self, context: Dict[str, Any]) -> List[Transition]:
        """Get all transitions enabled from current state."""
        return [
            t for t in self.transitions
            if t.source == self.current_state and t.is_enabled(context)
        ]

    def step(self, context: Dict[str, Any]) -> Tuple[str, bool, Optional[List[float]]]:
        """
        Take one step.

        Returns:
            (new_state, is_done, attention_weights)
        """
        enabled = self.get_enabled_transitions(context)

        if not enabled:
            return self.current_state, self.current_state in self.final_states, None

        if len(enabled) > 1:
            self.conflicts_resolved += 1

        # Use attention to select
        selected, attn_score = self.strategy.select_transition(enabled, context)
        selected.fire_count += 1

        # Get attention weights for all enabled
        _, attn_weights = self.strategy.compute_priorities(enabled, context)
        self.attention_history.append(attn_weights)

        self.current_state = selected.target

        return self.current_state, self.current_state in self.final_states, attn_weights

    def execute(self, context: Dict[str, Any], max_steps: int = 100) -> List[str]:
        """Execute until done, return state sequence."""
        self.reset()
        sequence = [self.current_state]

        for _ in range(max_steps):
            state, done, _ = self.step(context)
            sequence.append(state)
            if done:
                break

        return sequence


# =============================================================================
# Training with Evolution
# =============================================================================

class AttentionEvolver:
    """
    Evolves attention model parameters.

    Uses evolution to train the attention model since MLX doesn't have
    built-in optimizers for all use cases, and evolution is more robust
    for non-differentiable objectives.
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.1,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std

        # Population of models
        self.population: List[AttentionPriorityStrategy] = []
        self.fitnesses: List[float] = []

        self.generation = 0
        self.best_fitness = 0.0
        self.best_model: Optional[AttentionPriorityStrategy] = None

    def _mutate_params(self, params: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Mutate model parameters."""
        new_params = {}
        for k, v in params.items():
            if random.random() < self.mutation_rate:
                noise = mx.random.normal(v.shape) * self.mutation_std
                new_params[k] = v + noise
            else:
                new_params[k] = v
        return new_params

    def initialize_population(self):
        """Create initial population."""
        self.population = []
        for _ in range(self.population_size):
            strategy = AttentionPriorityStrategy()
            self.population.append(strategy)
        self.fitnesses = [0.0] * self.population_size

    def evaluate(
        self,
        strategy: AttentionPriorityStrategy,
        test_cases: List[Tuple[AttentionStatechart, Dict[str, Any], List[str]]],
    ) -> float:
        """
        Evaluate a strategy on test cases.

        Args:
            strategy: Strategy to evaluate
            test_cases: List of (statechart, context, expected_sequence)

        Returns:
            Fitness score
        """
        total_score = 0.0

        for sc_template, context, expected in test_cases:
            # Create copy with this strategy
            sc = AttentionStatechart(
                states=set(sc_template.states),
                transitions=[
                    Transition(
                        source=t.source,
                        target=t.target,
                        guard=Guard(predicate=t.guard.predicate, params=dict(t.guard.params)),
                        priority=t.priority,
                    )
                    for t in sc_template.transitions
                ],
                initial_state=sc_template.initial_state,
                final_states=set(sc_template.final_states),
                strategy=strategy,
            )

            actual = sc.execute(context)

            # Score: sequence matching
            matches = sum(1 for a, e in zip(actual, expected) if a == e)
            max_len = max(len(actual), len(expected))
            score = matches / max_len if max_len > 0 else 0

            # Bonus for reaching final state
            if actual[-1] in sc.final_states:
                score += 0.2

            total_score += score

        return total_score / len(test_cases) if test_cases else 0.0

    def evolve(
        self,
        test_cases: List[Tuple[AttentionStatechart, Dict[str, Any], List[str]]],
        n_generations: int = 50,
        verbose: bool = True,
    ) -> AttentionPriorityStrategy:
        """
        Evolve optimal attention model.

        Args:
            test_cases: Training data
            n_generations: Number of generations
            verbose: Print progress

        Returns:
            Best evolved strategy
        """
        if not self.population:
            self.initialize_population()

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate
            for i, strategy in enumerate(self.population):
                self.fitnesses[i] = self.evaluate(strategy, test_cases)

                if self.fitnesses[i] > self.best_fitness:
                    self.best_fitness = self.fitnesses[i]
                    self.best_model = strategy

            if verbose and gen % 10 == 0:
                avg_fit = sum(self.fitnesses) / len(self.fitnesses)
                print(f"Gen {gen}: best={self.best_fitness:.3f}, avg={avg_fit:.3f}")

            if self.best_fitness >= 1.0:
                break

            # Selection and reproduction
            sorted_pop = sorted(
                zip(self.fitnesses, self.population),
                key=lambda x: -x[0]
            )

            # Elitism
            next_pop = [sorted_pop[0][1]]

            # Tournament selection + mutation
            while len(next_pop) < self.population_size:
                # Tournament
                candidates = random.sample(sorted_pop[:len(sorted_pop)//2], min(3, len(sorted_pop)//2))
                parent = max(candidates, key=lambda x: x[0])[1]

                # Create child with mutated parameters
                child = AttentionPriorityStrategy()

                # Copy and mutate parameters
                parent_params = dict(parent.model.parameters())
                child_params = self._mutate_params(parent_params)
                child.model.update(child_params)

                next_pop.append(child)

            self.population = next_pop
            self.fitnesses = [0.0] * len(next_pop)

        return self.best_model


# =============================================================================
# Test Case Generation
# =============================================================================

def create_test_statechart() -> AttentionStatechart:
    """Create a test statechart with conflicts."""
    states = {"start", "s0", "s1", "s2", "s3", "end"}

    transitions = [
        # From start - multiple options based on context
        Transition("start", "s0", Guard("all_zero"), priority=0.0),
        Transition("start", "s1", Guard("has_nonzero"), priority=0.0),
        Transition("start", "s2", Guard("count_gt", {"threshold": 5}), priority=0.0),

        # From s0
        Transition("s0", "s1", Guard("true"), priority=0.0),
        Transition("s0", "end", Guard("state_eq", {"key": "x", "value": 0}), priority=0.0),

        # From s1
        Transition("s1", "s2", Guard("count_gt", {"threshold": 3}), priority=0.0),
        Transition("s1", "end", Guard("true"), priority=0.0),

        # From s2
        Transition("s2", "s3", Guard("has_color", {"color": 1}), priority=0.0),
        Transition("s2", "end", Guard("true"), priority=0.0),

        # From s3
        Transition("s3", "end", Guard("true"), priority=0.0),
    ]

    return AttentionStatechart(
        states=states,
        transitions=transitions,
        initial_state="start",
        final_states={"end"},
    )


def generate_test_cases(
    sc_template: AttentionStatechart,
    n_cases: int = 20,
    grid_size: int = 5,
) -> List[Tuple[AttentionStatechart, Dict[str, Any], List[str]]]:
    """
    Generate test cases with expected sequences.

    The expected sequence is determined by a "correct" priority scheme
    (specificity-based), which the attention model should learn to mimic.
    """
    test_cases = []

    for _ in range(n_cases):
        # Create context
        grid = mx.array([
            [random.randint(0, 3) for _ in range(grid_size)]
            for _ in range(grid_size)
        ])

        state = {
            "x": random.randint(0, 5),
            "y": random.randint(0, 5),
        }

        context = {"grid": grid, "state": state}

        # Determine expected sequence based on specificity rules
        nonzero_count = int(mx.sum(grid != 0))
        has_ones = bool(mx.any(grid == 1))

        # Specificity-based expectations:
        # More specific guards should win
        if nonzero_count > 5:
            # count_gt(5) is most specific when true
            expected = ["start", "s2"]
            if has_ones:
                expected.append("s3")
            expected.append("end")
        elif nonzero_count > 0:
            # has_nonzero wins over all_zero
            expected = ["start", "s1"]
            if nonzero_count > 3:
                expected.extend(["s2", "end"])
            else:
                expected.append("end")
        elif state["x"] == 0:
            # Exact match is most specific
            expected = ["start", "s0", "end"]
        else:
            expected = ["start", "s0", "s1", "end"]

        test_cases.append((sc_template, context, expected))

    return test_cases


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate attention-based priority selection."""
    print("=" * 60)
    print("Attention-Based Transition Priority Selection")
    print("=" * 60)

    # Create test statechart
    sc = create_test_statechart()
    print(f"\nStatechart: {len(sc.states)} states, {len(sc.transitions)} transitions")

    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = generate_test_cases(sc, n_cases=30)
    print(f"Created {len(test_cases)} test cases")

    # Create evolver
    evolver = AttentionEvolver(
        population_size=15,
        mutation_rate=0.2,
        mutation_std=0.15,
    )

    print("\nEvolving attention model...")
    best = evolver.evolve(test_cases, n_generations=30, verbose=True)

    print(f"\nBest fitness: {evolver.best_fitness:.3f}")

    # Test on a specific case
    print("\n--- Test Case Analysis ---")
    grid = mx.array([
        [1, 0, 2, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    context = {"grid": grid, "state": {"x": 2, "y": 1}}

    test_sc = AttentionStatechart(
        states=set(sc.states),
        transitions=[
            Transition(
                source=t.source,
                target=t.target,
                guard=Guard(predicate=t.guard.predicate, params=dict(t.guard.params)),
                priority=t.priority,
            )
            for t in sc.transitions
        ],
        initial_state=sc.initial_state,
        final_states=set(sc.final_states),
        strategy=best,
    )

    sequence = test_sc.execute(context)
    print(f"Grid nonzero count: {int(mx.sum(grid != 0))}")
    print(f"Execution sequence: {' -> '.join(sequence)}")
    print(f"Conflicts resolved: {test_sc.conflicts_resolved}")

    # Show attention weights
    if test_sc.attention_history:
        print(f"\nAttention weights at first conflict:")
        for i, w in enumerate(test_sc.attention_history[0]):
            print(f"  Transition {i}: {w:.3f}")

    return evolver, best


def compare_to_specificity():
    """Compare attention strategy to specificity heuristic."""
    print("\n" + "=" * 60)
    print("Comparison: Attention vs Specificity")
    print("=" * 60)

    sc = create_test_statechart()
    test_cases = generate_test_cases(sc, n_cases=50)

    # Train attention model
    evolver = AttentionEvolver(population_size=15)
    attention_strategy = evolver.evolve(test_cases, n_generations=30, verbose=False)

    # Evaluate both strategies
    attention_score = evolver.evaluate(attention_strategy, test_cases)

    print(f"\nResults on {len(test_cases)} test cases:")
    print(f"  Attention model: {attention_score:.3f}")
    print(f"  (Specificity baseline: ~0.56 from exp_transition_priorities)")

    improvement = (attention_score - 0.56) / 0.56 * 100
    print(f"\n  {'Improvement' if improvement > 0 else 'Gap'}: {abs(improvement):.1f}%")

    return attention_score


if __name__ == "__main__":
    demo()
    compare_to_specificity()
