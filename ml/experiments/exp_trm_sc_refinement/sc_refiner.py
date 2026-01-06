"""
SC Refiner: Core model for TRM-style statechart refinement.

Uses iterative refinement to transform partial/invalid statecharts
into valid ones by learning to fix structural issues.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class SCRefinerConfig:
    """Configuration for statechart refiner."""
    # Model dimensions
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2

    # Statechart dimensions
    max_states: int = 32
    max_transitions: int = 64
    state_types: int = 4  # BASIC, NORMAL, PARALLEL, INITIAL

    # TRM iteration config
    H_cycles: int = 3
    L_cycles: int = 6

    # Refinement targets
    fix_missing_initial: bool = True
    fix_unreachable: bool = True
    fix_invalid_hierarchy: bool = True
    fix_transition_targets: bool = True


class StateEncoder(nn.Module):
    """Encode state information into embeddings."""

    def __init__(self, config: SCRefinerConfig):
        super().__init__()
        self.config = config

        # State type embedding
        self.type_embed = nn.Embedding(config.state_types, config.hidden_dim)

        # Position embedding for hierarchy
        self.depth_embed = nn.Embedding(10, config.hidden_dim)  # Max depth 10

        # Label embedding (character-level)
        self.char_embed = nn.Embedding(128, config.hidden_dim // 4)
        self.label_proj = nn.Linear(config.hidden_dim // 4, config.hidden_dim)

        # Combine embeddings
        self.combine = nn.Linear(config.hidden_dim * 3, config.hidden_dim)

    def __call__(
        self,
        state_types: mx.array,   # [B, S]
        state_depths: mx.array,  # [B, S]
        label_chars: mx.array,   # [B, S, max_label_len]
    ) -> mx.array:
        """
        Encode states.

        Returns:
            [B, S, hidden_dim] state embeddings
        """
        # Type embedding
        type_emb = self.type_embed(state_types)  # [B, S, H]

        # Depth embedding
        depth_emb = self.depth_embed(state_depths)  # [B, S, H]

        # Label embedding (average over chars)
        char_emb = self.char_embed(label_chars)  # [B, S, L, H/4]
        label_emb = mx.mean(char_emb, axis=2)  # [B, S, H/4]
        label_emb = self.label_proj(label_emb)  # [B, S, H]

        # Combine
        combined = mx.concatenate([type_emb, depth_emb, label_emb], axis=-1)
        return self.combine(combined)


class TransitionEncoder(nn.Module):
    """Encode transition information."""

    def __init__(self, config: SCRefinerConfig):
        super().__init__()
        self.config = config

        # Source/target state indices
        self.state_ref_embed = nn.Embedding(config.max_states + 1, config.hidden_dim)

        # Event embedding
        self.event_embed = nn.Embedding(64, config.hidden_dim)  # Max 64 unique events

        # Combine
        self.combine = nn.Linear(config.hidden_dim * 3, config.hidden_dim)

    def __call__(
        self,
        source_indices: mx.array,  # [B, T]
        target_indices: mx.array,  # [B, T]
        event_ids: mx.array,       # [B, T]
    ) -> mx.array:
        """
        Encode transitions.

        Returns:
            [B, T, hidden_dim] transition embeddings
        """
        src_emb = self.state_ref_embed(source_indices)
        tgt_emb = self.state_ref_embed(target_indices)
        evt_emb = self.event_embed(event_ids)

        combined = mx.concatenate([src_emb, tgt_emb, evt_emb], axis=-1)
        return self.combine(combined)


class RefinementBlock(nn.Module):
    """Single refinement block with attention and MLP."""

    def __init__(self, config: SCRefinerConfig):
        super().__init__()
        self.config = config

        # Self-attention for states
        self.state_attn = nn.MultiHeadAttention(
            dims=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.state_norm1 = nn.LayerNorm(config.hidden_dim)

        # Cross-attention: states attend to transitions
        self.cross_attn = nn.MultiHeadAttention(
            dims=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.state_norm2 = nn.LayerNorm(config.hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.state_norm3 = nn.LayerNorm(config.hidden_dim)

    def __call__(
        self,
        state_emb: mx.array,       # [B, S, H]
        trans_emb: mx.array,       # [B, T, H]
        state_mask: mx.array,      # [B, S] valid states
        trans_mask: mx.array,      # [B, T] valid transitions
    ) -> mx.array:
        """
        Refine state embeddings.

        Returns:
            [B, S, H] refined state embeddings
        """
        # Self-attention on states
        h = state_emb
        h = h + self.state_attn(h, h, h)
        h = self.state_norm1(h)

        # Cross-attention to transitions
        h = h + self.cross_attn(h, trans_emb, trans_emb)
        h = self.state_norm2(h)

        # MLP
        h = h + self.mlp(h)
        h = self.state_norm3(h)

        return h


class RefinementHead(nn.Module):
    """Predict refinement actions for states and transitions."""

    def __init__(self, config: SCRefinerConfig):
        super().__init__()
        self.config = config

        # State refinement predictions
        self.type_head = nn.Linear(config.hidden_dim, config.state_types)
        self.initial_head = nn.Linear(config.hidden_dim, 2)  # is_initial
        self.parent_head = nn.Linear(config.hidden_dim, config.max_states + 1)

        # Transition refinement predictions
        self.trans_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.source_head = nn.Linear(config.hidden_dim, config.max_states)
        self.target_head = nn.Linear(config.hidden_dim, config.max_states)
        self.valid_head = nn.Linear(config.hidden_dim, 2)  # keep/remove

    def __call__(
        self,
        state_emb: mx.array,   # [B, S, H]
        trans_emb: mx.array,   # [B, T, H]
    ) -> Dict[str, mx.array]:
        """
        Predict refinement actions.

        Returns:
            Dict with prediction logits
        """
        # State predictions
        type_logits = self.type_head(state_emb)
        initial_logits = self.initial_head(state_emb)
        parent_logits = self.parent_head(state_emb)

        # Transition predictions
        trans_h = self.trans_proj(trans_emb)
        source_logits = self.source_head(trans_h)
        target_logits = self.target_head(trans_h)
        valid_logits = self.valid_head(trans_h)

        return {
            "state_type": type_logits,
            "state_initial": initial_logits,
            "state_parent": parent_logits,
            "trans_source": source_logits,
            "trans_target": target_logits,
            "trans_valid": valid_logits,
        }


class SCRefiner(nn.Module):
    """
    TRM-style Statechart Refiner.

    Uses iterative refinement to fix invalid statecharts:
    1. Encode current SC state
    2. Apply H×L refinement iterations
    3. Predict corrections at each step
    4. Apply soft corrections, continue refining
    """

    def __init__(self, config: Optional[SCRefinerConfig] = None):
        super().__init__()
        self.config = config or SCRefinerConfig()

        # Encoders
        self.state_encoder = StateEncoder(self.config)
        self.trans_encoder = TransitionEncoder(self.config)

        # Refinement blocks
        self.blocks = [
            RefinementBlock(self.config)
            for _ in range(self.config.num_layers)
        ]

        # H-level context
        self.h_context_net = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Output heads
        self.head = RefinementHead(self.config)

        # Validity score predictor
        self.validity_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode(
        self,
        state_types: mx.array,
        state_depths: mx.array,
        label_chars: mx.array,
        source_indices: mx.array,
        target_indices: mx.array,
        event_ids: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode statechart components.

        Returns:
            (state_embeddings, transition_embeddings)
        """
        state_emb = self.state_encoder(state_types, state_depths, label_chars)
        trans_emb = self.trans_encoder(source_indices, target_indices, event_ids)
        return state_emb, trans_emb

    def step(
        self,
        state_emb: mx.array,
        trans_emb: mx.array,
        h_context: mx.array,
        state_mask: mx.array,
        trans_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Single refinement step (L-level).

        Args:
            state_emb: [B, S, H] current state embeddings
            trans_emb: [B, T, H] current transition embeddings
            h_context: [B, H] H-level context
            state_mask: [B, S] valid states
            trans_mask: [B, T] valid transitions

        Returns:
            Updated (state_emb, trans_emb)
        """
        # Add H-context to state embeddings
        h_ctx_expanded = h_context[:, None, :]  # [B, 1, H]
        state_emb = state_emb + h_ctx_expanded

        # Apply refinement blocks
        for block in self.blocks:
            state_emb = block(state_emb, trans_emb, state_mask, trans_mask)

        return state_emb, trans_emb

    def predict_validity(self, state_emb: mx.array) -> mx.array:
        """
        Predict overall validity score.

        Args:
            state_emb: [B, S, H] state embeddings

        Returns:
            [B] validity scores (sigmoid)
        """
        # Pool over states
        pooled = mx.mean(state_emb, axis=1)  # [B, H]
        score = self.validity_head(pooled)  # [B, 1]
        return mx.sigmoid(score.squeeze(-1))

    def refine(
        self,
        state_types: mx.array,
        state_depths: mx.array,
        label_chars: mx.array,
        source_indices: mx.array,
        target_indices: mx.array,
        event_ids: mx.array,
        state_mask: mx.array,
        trans_mask: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """
        Run full refinement with H×L iterations.

        Returns:
            Dict with final predictions and validity scores
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Initial encoding
        state_emb, trans_emb = self.encode(
            state_types, state_depths, label_chars,
            source_indices, target_indices, event_ids,
        )

        all_validity_scores = []

        for h in range(H):
            # Compute H-level context
            pooled = mx.mean(state_emb, axis=1)  # [B, H]
            h_context = self.h_context_net(pooled)

            for l in range(L):
                # L-level refinement step
                state_emb, trans_emb = self.step(
                    state_emb, trans_emb, h_context,
                    state_mask, trans_mask,
                )

                # Track validity
                validity = self.predict_validity(state_emb)
                all_validity_scores.append(validity)

        # Final predictions
        predictions = self.head(state_emb, trans_emb)
        predictions["validity_trajectory"] = mx.stack(all_validity_scores, axis=1)
        predictions["final_validity"] = all_validity_scores[-1]

        return predictions


def refine_statechart(
    model: SCRefiner,
    statechart_data: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """
    Convenience function to refine a statechart.

    Args:
        model: Trained SCRefiner
        statechart_data: Dict with encoded statechart

    Returns:
        Refinement predictions
    """
    return model.refine(
        state_types=statechart_data["state_types"],
        state_depths=statechart_data["state_depths"],
        label_chars=statechart_data["label_chars"],
        source_indices=statechart_data["source_indices"],
        target_indices=statechart_data["target_indices"],
        event_ids=statechart_data["event_ids"],
        state_mask=statechart_data["state_mask"],
        trans_mask=statechart_data["trans_mask"],
    )


def test_sc_refiner():
    """Test the SC refiner model."""
    print("=" * 60)
    print("Testing SC Refiner")
    print("=" * 60)

    config = SCRefinerConfig(
        hidden_dim=64,
        max_states=16,
        max_transitions=32,
        H_cycles=2,
        L_cycles=3,
    )

    model = SCRefiner(config)

    # Create test data
    B, S, T = 2, 8, 12
    max_label_len = 16

    mx.random.seed(42)

    state_types = mx.random.randint(0, 4, (B, S))
    state_depths = mx.random.randint(0, 4, (B, S))
    label_chars = mx.random.randint(0, 128, (B, S, max_label_len))
    source_indices = mx.random.randint(0, S, (B, T))
    target_indices = mx.random.randint(0, S, (B, T))
    event_ids = mx.random.randint(0, 10, (B, T))
    state_mask = mx.ones((B, S))
    trans_mask = mx.ones((B, T))

    print("\n1. Testing encoding...")
    state_emb, trans_emb = model.encode(
        state_types, state_depths, label_chars,
        source_indices, target_indices, event_ids,
    )
    print(f"   State embeddings: {state_emb.shape}")
    print(f"   Trans embeddings: {trans_emb.shape}")

    print("\n2. Testing single step...")
    h_context = mx.zeros((B, config.hidden_dim))
    new_state_emb, new_trans_emb = model.step(
        state_emb, trans_emb, h_context,
        state_mask, trans_mask,
    )
    print(f"   Refined state emb: {new_state_emb.shape}")

    print("\n3. Testing full refinement...")
    predictions = model.refine(
        state_types, state_depths, label_chars,
        source_indices, target_indices, event_ids,
        state_mask, trans_mask,
    )

    print(f"   Predictions: {list(predictions.keys())}")
    print(f"   State type logits: {predictions['state_type'].shape}")
    print(f"   Validity trajectory: {predictions['validity_trajectory'].shape}")
    print(f"   Final validity: {predictions['final_validity']}")

    print("\n" + "=" * 60)
    print("SC Refiner test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sc_refiner()
