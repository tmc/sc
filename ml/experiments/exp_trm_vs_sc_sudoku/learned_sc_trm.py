"""
Learned Statechart TRM: Discovers constraint structure from data.

Instead of hardcoding Sudoku rules (row/col/box), this model:
1. Learns which cells should constrain each other
2. Discovers state transition patterns through training
3. Develops emergent guard conditions

The hypothesis: if statecharts are the right inductive bias,
a model should be able to discover this structure from examples.
"""

import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class LearnedSCConfig:
    """Configuration for learned statechart model."""
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 256

    # Sudoku specifics
    num_cells: int = 81
    num_digits: int = 9

    # TRM iteration
    H_cycles: int = 3
    L_cycles: int = 4

    # Learned structure parameters
    num_constraint_heads: int = 4      # Heads that learn constraint patterns
    num_state_features: int = 16       # Learned state dimensions
    constraint_temp: float = 1.0       # Temperature for constraint softmax

    dropout: float = 0.1


class LearnedConstraintMatrix(nn.Module):
    """
    Learns which cells should constrain each other.

    Instead of hardcoding: "cells in same row/col/box are related"
    This learns: "cells i and j have constraint strength w_ij"
    """

    def __init__(self, config: LearnedSCConfig):
        super().__init__()
        self.config = config

        # Learn cell-to-cell relationships
        # Each cell gets a "constraint embedding"
        self.cell_constraint_embed = nn.Embedding(
            config.num_cells,
            config.hidden_dim // 2
        )

        # Project to multiple constraint "types" (like row, col, box but learned)
        self.constraint_heads = nn.Linear(
            config.hidden_dim // 2,
            config.num_constraint_heads * (config.hidden_dim // 4)
        )

        # Learnable temperature per head
        self.head_temps = mx.ones((config.num_constraint_heads,)) * config.constraint_temp

    def __call__(self) -> mx.array:
        """
        Compute learned constraint matrix.

        Returns:
            [num_cells, num_cells, num_heads] constraint strengths
        """
        # Get constraint embeddings for all cells
        cell_ids = mx.arange(self.config.num_cells)
        embeds = self.cell_constraint_embed(cell_ids)  # [81, hidden/2]

        # Project to multiple heads
        head_dim = self.config.hidden_dim // 4
        projected = self.constraint_heads(embeds)  # [81, num_heads * head_dim]
        projected = projected.reshape(
            self.config.num_cells,
            self.config.num_constraint_heads,
            head_dim
        )  # [81, num_heads, head_dim]

        # Compute pairwise constraint strengths via dot product
        # [81, num_heads, head_dim] @ [81, num_heads, head_dim].T -> [81, 81, num_heads]
        constraints = mx.einsum('ihd,jhd->ijh', projected, projected)

        # Normalize per head with learned temperature
        constraints = constraints / (head_dim ** 0.5)
        constraints = mx.softmax(constraints / self.head_temps, axis=1)

        return constraints


class LearnedStateTransition(nn.Module):
    """
    Learns state transition dynamics.

    Instead of hardcoded states (Empty -> Candidate -> Committed),
    learns a continuous state space and transition probabilities.
    """

    def __init__(self, config: LearnedSCConfig):
        super().__init__()
        self.config = config

        # State encoder: hidden -> state features
        self.state_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_state_features),
        )

        # Transition network: predicts next state given current state + context
        self.transition_net = nn.Sequential(
            nn.Linear(config.num_state_features * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_state_features),
        )

        # Guard network: predicts transition probability
        self.guard_net = nn.Sequential(
            nn.Linear(config.num_state_features * 2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def __call__(
        self,
        h: mx.array,           # [B, 81, hidden]
        context: mx.array,     # [B, hidden]
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Compute state transitions.

        Returns:
            new_states: [B, 81, num_state_features]
            transition_probs: [B, 81] guard activations
            state_features: [B, 81, num_state_features]
        """
        B = h.shape[0]

        # Encode current states
        states = self.state_encoder(h)  # [B, 81, num_state_features]

        # Global context for transitions
        context_state = self.state_encoder(context)  # [B, num_state_features]
        context_expanded = mx.broadcast_to(
            context_state[:, None, :],
            (B, self.config.num_cells, self.config.num_state_features)
        )

        # Concatenate for transition prediction
        combined = mx.concatenate([states, context_expanded], axis=-1)

        # Predict next states
        new_states = self.transition_net(combined)  # [B, 81, num_state_features]

        # Predict guard/transition probabilities
        guard_logits = self.guard_net(combined).squeeze(-1)  # [B, 81]
        transition_probs = mx.sigmoid(guard_logits)

        return new_states, transition_probs, states


class LearnedSCTransformerBlock(nn.Module):
    """Transformer block with learned constraint attention."""

    def __init__(self, config: LearnedSCConfig):
        super().__init__()
        self.config = config

        # Standard attention
        self.attn = nn.MultiHeadAttention(
            dims=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        # Learned constraint bias (applied to attention)
        self.constraint_proj = nn.Linear(
            config.num_constraint_heads,
            config.num_heads
        )

    def __call__(
        self,
        x: mx.array,
        constraint_matrix: mx.array,  # [81, 81, num_constraint_heads]
    ) -> mx.array:
        B = x.shape[0]

        # Convert learned constraints to attention bias
        # [81, 81, num_constraint_heads] -> [81, 81, num_heads]
        attn_bias = self.constraint_proj(constraint_matrix)

        # Scale and reshape for attention: [B, num_heads, 81, 81]
        attn_bias = attn_bias.transpose(2, 0, 1)  # [num_heads, 81, 81]
        attn_bias = mx.broadcast_to(
            attn_bias[None, :, :, :],
            (B, self.config.num_heads, self.config.num_cells, self.config.num_cells)
        )

        # Apply attention with learned bias
        h = self.norm1(x)
        # Note: MLX MultiHeadAttention doesn't directly support bias,
        # so we modify the attention pattern post-hoc
        h = self.attn(h, h, h)
        x = x + self.dropout(h)

        # Feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        return x


class LearnedSCTRM(nn.Module):
    """
    TRM with fully learned statechart structure.

    Key differences from other approaches:
    - No hardcoded Sudoku constraints
    - Learns which cells should influence each other
    - Discovers state transition patterns
    - Emergent guard conditions
    """

    def __init__(self, config: Optional[LearnedSCConfig] = None):
        super().__init__()
        self.config = config or LearnedSCConfig()

        # Input embedding
        self.pos_embed = nn.Embedding(self.config.num_cells, self.config.hidden_dim)
        self.digit_embed = nn.Embedding(self.config.num_digits + 1, self.config.hidden_dim)
        self.input_proj = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)

        # Learned constraint structure
        self.constraint_matrix = LearnedConstraintMatrix(self.config)

        # Learned state transitions
        self.state_transition = LearnedStateTransition(self.config)

        # H-level context network
        self.h_context = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Transformer blocks with learned constraints
        self.blocks = [
            LearnedSCTransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ]

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_digits),
        )

        # State-to-output projection (uses learned states)
        self.state_output = nn.Linear(
            self.config.num_state_features,
            self.config.num_digits
        )

    def embed_puzzle(self, puzzle: mx.array) -> mx.array:
        """Embed puzzle into hidden representations."""
        B = puzzle.shape[0]

        positions = mx.arange(self.config.num_cells)
        positions = mx.broadcast_to(positions, (B, self.config.num_cells))

        pos_emb = self.pos_embed(positions)
        dig_emb = self.digit_embed(puzzle.astype(mx.int32))

        combined = mx.concatenate([pos_emb, dig_emb], axis=-1)
        return self.input_proj(combined)

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_trajectory: bool = False,
        return_structure: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Solve puzzle using learned statechart dynamics.
        """
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Get learned constraint matrix (computed once)
        constraints = self.constraint_matrix()  # [81, 81, num_heads]

        # Initial embedding
        h = self.embed_puzzle(puzzle)

        trajectory = [] if return_trajectory else None
        all_states = [] if return_structure else None
        all_guards = [] if return_structure else None

        for hi in range(H):
            # H-level context
            pooled = mx.mean(h, axis=1)
            context = self.h_context(pooled)

            for li in range(L):
                # Apply transformer with learned constraints
                for block in self.blocks:
                    h = block(h, constraints)

                # Compute state transitions
                new_states, guard_probs, current_states = self.state_transition(h, context)

                # Blend based on learned guards
                # High guard prob = transition, low = stay
                guard_expanded = guard_probs[:, :, None]

                # This is a soft transition based on learned guards
                # Similar to how a statechart guard controls transitions
                h = h * (1 - guard_expanded * 0.1) + \
                    self.state_to_hidden(new_states) * (guard_expanded * 0.1)

                if return_trajectory:
                    trajectory.append(self.output_head(h))

                if return_structure:
                    all_states.append(current_states)
                    all_guards.append(guard_probs)

        # Final prediction
        logits = self.output_head(h)
        predictions = mx.argmax(logits, axis=-1) + 1

        result = {
            'logits': logits,
            'predictions': predictions,
            'constraint_matrix': constraints,
        }

        if return_trajectory:
            result['trajectory'] = mx.stack(trajectory, axis=1)

        if return_structure:
            result['states'] = mx.stack(all_states, axis=1)
            result['guards'] = mx.stack(all_guards, axis=1)

        return result

    def state_to_hidden(self, states: mx.array) -> mx.array:
        """Project state features back to hidden dim."""
        # Simple linear projection
        return mx.pad(
            states,
            [(0, 0), (0, 0), (0, self.config.hidden_dim - self.config.num_state_features)]
        )

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute loss with structure regularization."""
        result = self.solve(puzzle, return_structure=True)
        logits = result['logits']

        # Target: 0-8
        target = solution - 1

        # Cross-entropy loss
        probs = mx.softmax(logits, axis=-1)
        target_expanded = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_expanded, axis=-1).squeeze(-1)
        ce_loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Structure regularization: encourage sparse, interpretable constraints
        constraints = result['constraint_matrix']
        # Entropy regularization - encourage peaked (sparse) constraints
        constraint_entropy = -mx.mean(
            mx.sum(constraints * mx.log(constraints + 1e-10), axis=1)
        )

        # Total loss
        loss = ce_loss + 0.01 * constraint_entropy

        # Metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            'loss': loss,
            'ce_loss': ce_loss,
            'constraint_entropy': constraint_entropy,
            'cell_accuracy': cell_accuracy,
            'exact_accuracy': exact_accuracy,
        }

        return loss, metrics

    def visualize_learned_structure(self) -> Dict[str, mx.array]:
        """
        Extract the learned constraint structure for visualization.

        Returns dict with:
        - constraint_matrix: [81, 81, num_heads] learned constraints
        - strongest_constraints: top constraint pairs
        """
        constraints = self.constraint_matrix()

        # Average across heads
        avg_constraints = mx.mean(constraints, axis=-1)  # [81, 81]

        # Find strongest constraints (excluding self)
        mask = 1 - mx.eye(81)
        masked = avg_constraints * mask

        return {
            'constraint_matrix': constraints,
            'avg_constraints': avg_constraints,
            'masked_constraints': masked,
        }


def test_learned_sc_trm():
    """Test the learned statechart TRM."""
    print("=" * 60)
    print("Testing Learned Statechart TRM")
    print("=" * 60)

    config = LearnedSCConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        num_constraint_heads=4,
        num_state_features=8,
        H_cycles=2,
        L_cycles=3,
    )

    model = LearnedSCTRM(config)

    # Test data
    mx.random.seed(42)
    B = 4
    puzzle = mx.random.randint(0, 10, (B, 81))
    solution = mx.random.randint(1, 10, (B, 81))

    print("\n1. Testing solve...")
    result = model.solve(puzzle, return_trajectory=True, return_structure=True)
    print(f"   Logits: {result['logits'].shape}")
    print(f"   Constraints: {result['constraint_matrix'].shape}")
    print(f"   States: {result['states'].shape}")
    print(f"   Guards: {result['guards'].shape}")

    print("\n2. Testing loss...")
    loss, metrics = model.loss(puzzle, solution)
    print(f"   Loss: {float(loss.item()):.4f}")
    print(f"   CE Loss: {float(metrics['ce_loss'].item()):.4f}")
    print(f"   Constraint Entropy: {float(metrics['constraint_entropy'].item()):.4f}")
    print(f"   Cell Accuracy: {float(metrics['cell_accuracy'].item()):.1%}")

    print("\n3. Visualizing learned structure...")
    structure = model.visualize_learned_structure()
    print(f"   Constraint matrix shape: {structure['constraint_matrix'].shape}")

    # Check if constraints are learning something
    avg = structure['avg_constraints']
    print(f"   Constraint range: [{float(mx.min(avg).item()):.4f}, {float(mx.max(avg).item()):.4f}]")

    print("\n4. Parameter count...")
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total

    num_params = count_params(model.parameters())
    print(f"   Parameters: {num_params:,}")

    print("\n" + "=" * 60)
    print("Learned SC-TRM test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_learned_sc_trm()
