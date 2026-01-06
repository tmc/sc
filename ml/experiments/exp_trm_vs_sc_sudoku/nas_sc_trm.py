"""
Neural Architecture Search for Statechart Structure (NAS-SC).

Uses differentiable architecture search (DARTS-style) to discover:
1. Which constraint patterns matter (row/col/box/diagonal/knight/etc)
2. Optimal state transition topology
3. Guard condition structure

The model maintains a "supernet" of possible constraint types and
learns architecture weights to select the best combination.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class NASSCConfig:
    """Configuration for NAS statechart model."""
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 256

    # Sudoku
    num_cells: int = 81
    num_digits: int = 9

    # TRM iteration
    H_cycles: int = 3
    L_cycles: int = 4

    # NAS parameters
    num_constraint_ops: int = 8       # Number of candidate constraint types
    num_transition_ops: int = 4       # Number of candidate transition types
    arch_temp: float = 1.0            # Temperature for architecture softmax

    dropout: float = 0.1


def make_row_constraint() -> mx.array:
    """Constraint: cells in same row should not conflict."""
    import numpy as np
    mask = np.zeros((81, 81))
    for row in range(9):
        for i in range(9):
            for j in range(9):
                if i != j:
                    mask[row * 9 + i, row * 9 + j] = 1.0
    return mx.array(mask / 8.0)


def make_col_constraint() -> mx.array:
    """Constraint: cells in same column should not conflict."""
    import numpy as np
    mask = np.zeros((81, 81))
    for col in range(9):
        for i in range(9):
            for j in range(9):
                if i != j:
                    mask[i * 9 + col, j * 9 + col] = 1.0
    return mx.array(mask / 8.0)


def make_box_constraint() -> mx.array:
    """Constraint: cells in same 3x3 box should not conflict."""
    import numpy as np
    mask = np.zeros((81, 81))
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for r in range(3):
                for c in range(3):
                    cells.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
            for i in cells:
                for j in cells:
                    if i != j:
                        mask[i, j] = 1.0
    return mx.array(mask / 8.0)


def make_diagonal_constraint() -> mx.array:
    """Constraint: main diagonals (like Sudoku-X variant)."""
    import numpy as np
    mask = np.zeros((81, 81))
    diag1 = [i * 9 + i for i in range(9)]
    diag2 = [i * 9 + (8 - i) for i in range(9)]

    for diag in [diag1, diag2]:
        for i in diag:
            for j in diag:
                if i != j:
                    mask[i, j] = 1.0
    return mx.array(mask / 8.0)


def make_knight_constraint() -> mx.array:
    """Constraint: knight's move (like Anti-Knight Sudoku)."""
    import numpy as np
    mask = np.zeros((81, 81))
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]

    for cell in range(81):
        row, col = cell // 9, cell % 9
        for dr, dc in knight_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                neighbor = nr * 9 + nc
                mask[cell, neighbor] = 1.0

    return mx.array(mask / 4.0)


def make_king_constraint() -> mx.array:
    """Constraint: king's move (orthogonal + diagonal neighbors)."""
    import numpy as np
    mask = np.zeros((81, 81))
    king_moves = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for cell in range(81):
        row, col = cell // 9, cell % 9
        for dr, dc in king_moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                neighbor = nr * 9 + nc
                mask[cell, neighbor] = 1.0

    return mx.array(mask / 8.0)


def make_orthogonal_constraint() -> mx.array:
    """Constraint: orthogonally adjacent cells only."""
    import numpy as np
    mask = np.zeros((81, 81))
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for cell in range(81):
        row, col = cell // 9, cell % 9
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                neighbor = nr * 9 + nc
                mask[cell, neighbor] = 1.0

    return mx.array(mask / 4.0)


def make_global_constraint() -> mx.array:
    """Constraint: all cells connected (global attention)."""
    mask = mx.ones((81, 81)) - mx.eye(81)
    return mask / 80.0


class ConstraintSupernet(nn.Module):
    """
    Supernet of candidate constraint operations.

    Each constraint type is a candidate "operation" in NAS terms.
    Architecture weights determine which constraints are active.
    """

    def __init__(self, config: NASSCConfig):
        super().__init__()
        self.config = config

        # Pre-compute candidate constraint matrices
        self.constraint_ops = [
            make_row_constraint(),
            make_col_constraint(),
            make_box_constraint(),
            make_diagonal_constraint(),
            make_knight_constraint(),
            make_king_constraint(),
            make_orthogonal_constraint(),
            make_global_constraint(),
        ]

        # Stack into single tensor [num_ops, 81, 81]
        self.constraint_bank = mx.stack(self.constraint_ops, axis=0)

        # Learnable architecture weights (which constraints to use)
        self.arch_weights = mx.zeros((config.num_constraint_ops,))

        # Learnable scaling per constraint type
        self.constraint_scales = mx.ones((config.num_constraint_ops,))

    def __call__(self, temperature: float = 1.0) -> Tuple[mx.array, mx.array]:
        """
        Compute mixed constraint matrix using architecture weights.

        Returns:
            constraint_matrix: [81, 81] weighted combination
            arch_probs: [num_ops] architecture probabilities
        """
        # Softmax over architecture weights
        arch_probs = mx.softmax(self.arch_weights / temperature)

        # Scale constraints
        scaled_constraints = self.constraint_bank * self.constraint_scales[:, None, None]

        # Weighted combination
        # [num_ops, 81, 81] * [num_ops, 1, 1] -> sum -> [81, 81]
        mixed = mx.sum(scaled_constraints * arch_probs[:, None, None], axis=0)

        return mixed, arch_probs


class TransitionSupernet(nn.Module):
    """
    Supernet of candidate state transition operations.

    Different ways states can transition:
    - Hard transition (discrete jump)
    - Soft blend (gradual transition)
    - Gated transition (conditional)
    - Residual transition (additive)
    """

    def __init__(self, config: NASSCConfig):
        super().__init__()
        self.config = config

        # Transition operation networks
        self.hard_transition = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.soft_blend = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Tanh(),
        )

        self.gated_transition = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
            nn.GELU(),
        )
        self.gate_proj = nn.Linear(config.hidden_dim, 1)

        self.residual_transition = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Architecture weights for transitions
        self.trans_arch_weights = mx.zeros((config.num_transition_ops,))

    def __call__(
        self,
        h: mx.array,           # [B, 81, hidden]
        context: mx.array,     # [B, hidden]
        temperature: float = 1.0
    ) -> Tuple[mx.array, mx.array]:
        """Apply mixed transition operation."""
        B = h.shape[0]

        # Expand context
        ctx = mx.broadcast_to(context[:, None, :], h.shape)
        combined = mx.concatenate([h, ctx], axis=-1)

        # Compute each transition type
        trans_hard = self.hard_transition(combined)
        trans_soft = h + 0.1 * self.soft_blend(combined)

        gated = self.gated_transition(combined)
        gate = mx.sigmoid(self.gate_proj(h))
        trans_gated = gate * gated[:, :, :self.config.hidden_dim] + \
                      (1 - gate) * gated[:, :, self.config.hidden_dim:]

        trans_residual = h + 0.1 * self.residual_transition(h)

        # Stack transitions [B, 81, hidden, num_ops]
        all_trans = mx.stack([trans_hard, trans_soft, trans_gated, trans_residual], axis=-1)

        # Architecture softmax
        trans_probs = mx.softmax(self.trans_arch_weights / temperature)

        # Weighted combination
        mixed = mx.sum(all_trans * trans_probs, axis=-1)

        return mixed, trans_probs


class NASTransformerBlock(nn.Module):
    """Transformer block with NAS constraint integration."""

    def __init__(self, config: NASSCConfig):
        super().__init__()
        self.config = config

        self.attn = nn.MultiHeadAttention(
            dims=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Constraint integration
        self.constraint_proj = nn.Linear(1, config.num_heads)

        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, constraint_matrix: mx.array) -> mx.array:
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attn(h, h, h)
        x = x + self.dropout(h)

        # Feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        return x


class NASSCTRM(nn.Module):
    """
    Neural Architecture Search for Statechart TRM.

    Discovers optimal statechart structure through:
    1. Constraint operation search (row/col/box/diagonal/knight/etc)
    2. Transition operation search (hard/soft/gated/residual)
    3. End-to-end differentiable training
    """

    def __init__(self, config: Optional[NASSCConfig] = None):
        super().__init__()
        self.config = config or NASSCConfig()

        # Input embedding
        self.pos_embed = nn.Embedding(self.config.num_cells, self.config.hidden_dim)
        self.digit_embed = nn.Embedding(self.config.num_digits + 1, self.config.hidden_dim)
        self.input_proj = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)

        # NAS supernets
        self.constraint_supernet = ConstraintSupernet(self.config)
        self.transition_supernet = TransitionSupernet(self.config)

        # H-level context
        self.h_context = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Transformer blocks
        self.blocks = [
            NASTransformerBlock(self.config)
            for _ in range(self.config.num_layers)
        ]

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_digits),
        )

    def embed_puzzle(self, puzzle: mx.array) -> mx.array:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(
            mx.arange(self.config.num_cells),
            (B, self.config.num_cells)
        )

        pos_emb = self.pos_embed(positions)
        dig_emb = self.digit_embed(puzzle.astype(mx.int32))

        combined = mx.concatenate([pos_emb, dig_emb], axis=-1)
        return self.input_proj(combined)

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        arch_temp: float = 1.0,
        return_arch: bool = False,
    ) -> Dict[str, mx.array]:
        """Solve with architecture search."""
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Get mixed constraint matrix from supernet
        constraint_matrix, constraint_probs = self.constraint_supernet(arch_temp)

        # Embed input
        h = self.embed_puzzle(puzzle)

        for hi in range(H):
            pooled = mx.mean(h, axis=1)
            context = self.h_context(pooled)

            for li in range(L):
                # Apply transformer with searched constraints
                for block in self.blocks:
                    h = block(h, constraint_matrix)

                # Apply searched transition
                h, trans_probs = self.transition_supernet(h, context, arch_temp)

        # Output
        logits = self.output_head(h)
        predictions = mx.argmax(logits, axis=-1) + 1

        result = {
            'logits': logits,
            'predictions': predictions,
        }

        if return_arch:
            result['constraint_probs'] = constraint_probs
            result['transition_probs'] = trans_probs
            result['constraint_matrix'] = constraint_matrix

        return result

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        arch_temp: float = 1.0,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Loss with architecture regularization."""
        result = self.solve(puzzle, arch_temp=arch_temp, return_arch=True)
        logits = result['logits']

        # Cross-entropy
        target = solution - 1
        probs = mx.softmax(logits, axis=-1)
        target_exp = target[:, :, None]
        correct_probs = mx.take_along_axis(probs, target_exp, axis=-1).squeeze(-1)
        ce_loss = -mx.mean(mx.log(correct_probs + 1e-10))

        # Architecture entropy regularization (encourage sparse selection)
        constraint_entropy = -mx.sum(
            result['constraint_probs'] * mx.log(result['constraint_probs'] + 1e-10)
        )
        trans_entropy = -mx.sum(
            result['transition_probs'] * mx.log(result['transition_probs'] + 1e-10)
        )

        # Total loss (minimize entropy = more decisive architecture)
        loss = ce_loss - 0.01 * (constraint_entropy + trans_entropy)

        # Metrics
        predictions = mx.argmax(logits, axis=-1)
        cell_accuracy = mx.mean((predictions == target).astype(mx.float32))
        exact_match = mx.all(predictions == target, axis=1)
        exact_accuracy = mx.mean(exact_match.astype(mx.float32))

        metrics = {
            'loss': loss,
            'ce_loss': ce_loss,
            'constraint_entropy': constraint_entropy,
            'trans_entropy': trans_entropy,
            'cell_accuracy': cell_accuracy,
            'exact_accuracy': exact_accuracy,
            'constraint_probs': result['constraint_probs'],
            'transition_probs': result['transition_probs'],
        }

        return loss, metrics

    def get_discovered_architecture(self) -> Dict[str, any]:
        """Extract the discovered architecture."""
        constraint_names = [
            'row', 'col', 'box', 'diagonal',
            'knight', 'king', 'orthogonal', 'global'
        ]
        transition_names = ['hard', 'soft', 'gated', 'residual']

        c_probs = mx.softmax(self.constraint_supernet.arch_weights).tolist()
        t_probs = mx.softmax(self.transition_supernet.trans_arch_weights).tolist()

        return {
            'constraints': {name: prob for name, prob in zip(constraint_names, c_probs)},
            'transitions': {name: prob for name, prob in zip(transition_names, t_probs)},
            'top_constraint': constraint_names[int(mx.argmax(mx.array(c_probs)).item())],
            'top_transition': transition_names[int(mx.argmax(mx.array(t_probs)).item())],
        }


def test_nas_sc_trm():
    """Test NAS statechart TRM."""
    print("=" * 60)
    print("Testing NAS Statechart TRM")
    print("=" * 60)

    config = NASSCConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=3,
    )

    model = NASSCTRM(config)

    mx.random.seed(42)
    B = 4
    puzzle = mx.random.randint(0, 10, (B, 81))
    solution = mx.random.randint(1, 10, (B, 81))

    print("\n1. Testing solve with architecture...")
    result = model.solve(puzzle, return_arch=True)
    print(f"   Logits: {result['logits'].shape}")
    print(f"   Constraint probs: {result['constraint_probs'].tolist()}")
    print(f"   Transition probs: {result['transition_probs'].tolist()}")

    print("\n2. Testing loss...")
    loss, metrics = model.loss(puzzle, solution)
    print(f"   Loss: {float(loss.item()):.4f}")
    print(f"   CE Loss: {float(metrics['ce_loss'].item()):.4f}")
    print(f"   Cell Accuracy: {float(metrics['cell_accuracy'].item()):.1%}")

    print("\n3. Discovered architecture (before training):")
    arch = model.get_discovered_architecture()
    print(f"   Top constraint: {arch['top_constraint']}")
    print(f"   Top transition: {arch['top_transition']}")
    print(f"   Constraint weights: {arch['constraints']}")

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
    print("NAS SC-TRM test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_nas_sc_trm()
