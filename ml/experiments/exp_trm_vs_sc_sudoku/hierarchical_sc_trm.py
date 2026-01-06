"""
Hierarchical Multi-Level Statechart TRM for Sudoku.

Sudoku has natural hierarchy - model it explicitly:

Level 3: Board State (Solving → Stuck → Backtrack → Solved)
Level 2: Unit State (Row/Col/Box completion progress)
Level 1: Cell State (Empty → Candidate → Committed)

Messages flow: Board → Units → Cells (top-down guidance)

This mirrors human solving strategy: easy cells first, use
solved cells to constrain harder ones.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical SC-TRM."""
    # Base config
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    num_cells: int = 81
    num_digits: int = 9
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1

    # Hierarchy config
    unit_hidden_dim: int = 64    # Dimension for unit-level states
    board_hidden_dim: int = 32   # Dimension for board-level state
    top_down_strength: float = 1.0  # Strength of top-down guidance


class UnitEncoder(nn.Module):
    """
    Encodes constraint unit states (row/column/box).

    Takes cell states and aggregates them into unit-level representations.
    """

    def __init__(self, cell_dim: int, unit_dim: int, num_units: int = 27):
        super().__init__()
        self.cell_dim = cell_dim
        self.unit_dim = unit_dim
        self.num_units = num_units  # 9 rows + 9 cols + 9 boxes

        # Project from cells to unit representation
        self.cell_to_unit = nn.Linear(cell_dim * 9, unit_dim)  # 9 cells per unit

        # Unit state refinement
        self.unit_net = nn.Sequential(
            nn.Linear(unit_dim, unit_dim * 2),
            nn.GELU(),
            nn.Linear(unit_dim * 2, unit_dim),
        )
        self.ln = nn.LayerNorm(unit_dim)

    def get_unit_cells(self, unit_idx: int) -> List[int]:
        """Get cell indices for a constraint unit."""
        if unit_idx < 9:  # Row
            row = unit_idx
            return [row * 9 + c for c in range(9)]
        elif unit_idx < 18:  # Column
            col = unit_idx - 9
            return [r * 9 + col for r in range(9)]
        else:  # Box
            box = unit_idx - 18
            box_row = box // 3
            box_col = box % 3
            cells = []
            for r in range(3):
                for c in range(3):
                    cells.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
            return cells

    def __call__(self, cell_states: mx.array) -> mx.array:
        """
        Compute unit states from cell states.

        Args:
            cell_states: [B, 81, cell_dim]

        Returns:
            unit_states: [B, 27, unit_dim]
        """
        B = cell_states.shape[0]

        unit_states = []
        for unit_idx in range(self.num_units):
            # Get cells for this unit
            cell_indices = self.get_unit_cells(unit_idx)

            # Gather and concatenate cell states
            # [B, 9, cell_dim] -> [B, 9 * cell_dim]
            unit_cells = mx.concatenate([
                cell_states[:, idx:idx+1, :] for idx in cell_indices
            ], axis=1)  # [B, 9, cell_dim]
            unit_cells_flat = unit_cells.reshape(B, -1)  # [B, 9 * cell_dim]

            # Project to unit representation
            unit_state = self.cell_to_unit(unit_cells_flat)  # [B, unit_dim]
            unit_states.append(unit_state[:, None, :])

        # Stack all units
        unit_states = mx.concatenate(unit_states, axis=1)  # [B, 27, unit_dim]

        # Refine unit states
        unit_states = self.ln(unit_states)
        unit_states = unit_states + self.unit_net(unit_states)

        return unit_states


class BoardEncoder(nn.Module):
    """
    Encodes board-level state from unit states.

    Aggregates all unit states into global board representation.
    """

    def __init__(self, unit_dim: int, board_dim: int):
        super().__init__()
        self.unit_dim = unit_dim
        self.board_dim = board_dim

        # Attention-based aggregation
        self.query = nn.Linear(unit_dim, board_dim)
        self.key = nn.Linear(unit_dim, board_dim)
        self.value = nn.Linear(unit_dim, board_dim)

        # Board state refinement
        self.board_net = nn.Sequential(
            nn.Linear(board_dim, board_dim * 2),
            nn.GELU(),
            nn.Linear(board_dim * 2, board_dim),
        )
        self.ln = nn.LayerNorm(board_dim)

    def __call__(self, unit_states: mx.array) -> mx.array:
        """
        Compute board state from unit states.

        Args:
            unit_states: [B, 27, unit_dim]

        Returns:
            board_state: [B, board_dim]
        """
        B = unit_states.shape[0]

        # Global query (learnable)
        global_query = mx.mean(unit_states, axis=1, keepdims=True)  # [B, 1, unit_dim]
        q = self.query(global_query)  # [B, 1, board_dim]
        k = self.key(unit_states)  # [B, 27, board_dim]
        v = self.value(unit_states)  # [B, 27, board_dim]

        # Attention
        scale = 1.0 / math.sqrt(self.board_dim)
        scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale  # [B, 1, 27]
        attn = mx.softmax(scores, axis=-1)
        board_state = mx.matmul(attn, v).squeeze(1)  # [B, board_dim]

        # Refine
        board_state = self.ln(board_state)
        board_state = board_state + self.board_net(board_state)

        return board_state


class TopDownGuidance(nn.Module):
    """
    Propagates guidance from board → units → cells.

    Top-down messages help cells understand global solving progress.
    """

    def __init__(
        self,
        board_dim: int,
        unit_dim: int,
        cell_dim: int,
    ):
        super().__init__()
        self.board_dim = board_dim
        self.unit_dim = unit_dim
        self.cell_dim = cell_dim

        # Board → Unit projection
        self.board_to_unit = nn.Linear(board_dim, unit_dim)

        # Unit → Cell projection (per unit type)
        self.unit_to_cell = nn.Linear(unit_dim, cell_dim)

        # Combine cell's own state with top-down signal
        self.gate = nn.Linear(cell_dim * 2, cell_dim)

    def __call__(
        self,
        cell_states: mx.array,
        unit_states: mx.array,
        board_state: mx.array,
        strength: float = 1.0,
    ) -> mx.array:
        """
        Apply top-down guidance to cell states.

        Args:
            cell_states: [B, 81, cell_dim]
            unit_states: [B, 27, unit_dim]
            board_state: [B, board_dim]

        Returns:
            guided_cell_states: [B, 81, cell_dim]
        """
        B = cell_states.shape[0]

        # Board → Unit guidance
        board_signal = self.board_to_unit(board_state)  # [B, unit_dim]
        unit_states = unit_states + board_signal[:, None, :]  # [B, 27, unit_dim]

        # Unit → Cell guidance
        # Each cell gets guidance from its 3 units (row, col, box)
        cell_guidance = mx.zeros((B, 81, self.cell_dim))

        for cell_idx in range(81):
            row = cell_idx // 9
            col = cell_idx % 9
            box = (row // 3) * 3 + (col // 3)

            # Unit indices
            row_unit = row
            col_unit = 9 + col
            box_unit = 18 + box

            # Average guidance from 3 units
            guidance = (
                self.unit_to_cell(unit_states[:, row_unit, :]) +
                self.unit_to_cell(unit_states[:, col_unit, :]) +
                self.unit_to_cell(unit_states[:, box_unit, :])
            ) / 3.0

            # Add to guidance tensor using broadcast
            mask = mx.arange(81) == cell_idx
            mask = mask[None, :, None]  # [1, 81, 1]
            cell_guidance = cell_guidance + mask * guidance[:, None, :]

        # Gate: combine cell state with guidance
        combined = mx.concatenate([cell_states, cell_guidance * strength], axis=-1)
        guided = mx.sigmoid(self.gate(combined)) * cell_states + \
                 (1 - mx.sigmoid(self.gate(combined))) * cell_guidance

        return guided


class HierarchicalSCTRM(nn.Module):
    """
    Hierarchical Statechart TRM with 3 levels:
    - Board level: global solving progress
    - Unit level: row/col/box completion
    - Cell level: individual cell predictions
    """

    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config

        # Cell embedding
        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # Cell-level transformer layers (using simple attention + FFN)
        self.cell_attns = [nn.MultiHeadAttention(config.hidden_dim, config.num_heads)
                          for _ in range(config.num_layers)]
        self.cell_ffns = [nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
        ) for _ in range(config.num_layers)]
        self.cell_lns1 = [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
        self.cell_lns2 = [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]

        # Unit encoder
        self.unit_encoder = UnitEncoder(
            config.hidden_dim, config.unit_hidden_dim
        )

        # Board encoder
        self.board_encoder = BoardEncoder(
            config.unit_hidden_dim, config.board_hidden_dim
        )

        # Top-down guidance
        self.guidance = TopDownGuidance(
            config.board_hidden_dim,
            config.unit_hidden_dim,
            config.hidden_dim,
        )

        # H-level context
        self.h_context_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Output head
        self.output_head = nn.Linear(config.hidden_dim, config.num_digits)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def encode(self, puzzle: mx.array) -> mx.array:
        """Encode puzzle to initial cell states."""
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        cell_emb = self.cell_embed(puzzle.astype(mx.int32))
        pos_emb = self.pos_embed(positions)

        return cell_emb + pos_emb

    def refine_with_hierarchy(
        self,
        cell_states: mx.array,
        h_context: mx.array,
        guidance_strength: float = 1.0,
    ) -> mx.array:
        """
        One L-cycle with hierarchical processing.

        1. Cell-level transformer refinement
        2. Compute unit states from cells
        3. Compute board state from units
        4. Apply top-down guidance
        """
        B = cell_states.shape[0]

        # Add H-context
        h = cell_states + h_context[:, None, :]

        # Cell-level refinement with attention + FFN
        for i in range(len(self.cell_attns)):
            # Attention with residual
            h_norm = self.cell_lns1[i](h)
            h = h + self.cell_attns[i](h_norm, h_norm, h_norm)
            # FFN with residual
            h_norm = self.cell_lns2[i](h)
            h = h + self.cell_ffns[i](h_norm)

        # Compute hierarchy
        unit_states = self.unit_encoder(h)
        board_state = self.board_encoder(unit_states)

        # Apply top-down guidance
        h = self.guidance(h, unit_states, board_state, strength=guidance_strength)

        return h

    def predict(self, h: mx.array) -> mx.array:
        """Predict digit logits."""
        h = self.ln_out(h)
        return self.output_head(h)

    def solve(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Solve with hierarchical refinement."""
        H = H_cycles or self.config.H_cycles
        L = L_cycles or self.config.L_cycles

        # Guidance strength increases across H-cycles
        guidance_schedule = [0.5 + 0.5 * h / (H - 1) if H > 1 else 1.0 for h in range(H)]

        h = self.encode(puzzle)

        for hi in range(H):
            strength = guidance_schedule[hi]

            pooled = mx.mean(h, axis=1)
            h_context = self.h_context_net(pooled)

            for li in range(L):
                h = self.refine_with_hierarchy(h, h_context, guidance_strength=strength)

        logits = self.predict(h)
        predictions = mx.argmax(logits, axis=-1) + 1

        return {
            'logits': logits,
            'predictions': predictions,
        }

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
    ) -> Tuple[mx.array, Dict]:
        """Compute loss."""
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        B, C, D = logits.shape

        logits_flat = logits.reshape(-1, D)
        targets_flat = targets.reshape(-1)

        logits_flat = mx.clip(logits_flat, -30, 30)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

        batch_indices = mx.arange(logits_flat.shape[0])
        correct_log_probs = log_probs[batch_indices, targets_flat]
        loss = -mx.mean(correct_log_probs)

        predictions = result['predictions']
        accuracy = mx.mean((predictions == solution).astype(mx.float32))

        return loss, {'loss': float(loss.item()), 'accuracy': float(accuracy.item())}


def train_and_evaluate():
    """Train hierarchical SC-TRM and compare."""
    from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 70)
    print("HIERARCHICAL SC-TRM EXPERIMENT")
    print("=" * 70)

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(1000, 200, seed=42)

    # Train vanilla TRM baseline
    print("\n" + "-" * 70)
    print("Training Vanilla TRM (baseline)...")
    print("-" * 70)

    vanilla_config = VanillaTRMConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
    )
    vanilla_model = VanillaTRM(vanilla_config)
    vanilla_opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def vanilla_loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    vanilla_grad = nn.value_and_grad(vanilla_model, vanilla_loss_fn)

    for epoch in range(30):
        perm = mx.random.permutation(train_p.shape[0])
        for i in range(0, train_p.shape[0], 32):
            p = train_p[perm[i:i+32]]
            s = train_s[perm[i:i+32]]
            loss, grads = vanilla_grad(vanilla_model, p, s)
            vanilla_opt.update(vanilla_model, grads)
            mx.eval(vanilla_model.parameters())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    vanilla_result = vanilla_model.solve(test_p)
    vanilla_acc = float(mx.mean((vanilla_result['predictions'] == test_s).astype(mx.float32)).item())
    print(f"Vanilla TRM accuracy: {vanilla_acc:.1%}")

    # Train hierarchical SC-TRM
    print("\n" + "-" * 70)
    print("Training Hierarchical SC-TRM...")
    print("-" * 70)

    hier_config = HierarchicalConfig(
        hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
        H_cycles=3, L_cycles=4,
        unit_hidden_dim=64, board_hidden_dim=32,
    )
    hier_model = HierarchicalSCTRM(hier_config)
    hier_opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def hier_loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    hier_grad = nn.value_and_grad(hier_model, hier_loss_fn)

    for epoch in range(30):
        perm = mx.random.permutation(train_p.shape[0])
        for i in range(0, train_p.shape[0], 32):
            p = train_p[perm[i:i+32]]
            s = train_s[perm[i:i+32]]
            loss, grads = hier_grad(hier_model, p, s)
            hier_opt.update(hier_model, grads)
            mx.eval(hier_model.parameters())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    hier_result = hier_model.solve(test_p)
    hier_acc = float(mx.mean((hier_result['predictions'] == test_s).astype(mx.float32)).item())
    print(f"Hierarchical SC-TRM accuracy: {hier_acc:.1%}")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Vanilla TRM:       {vanilla_acc:.1%}")
    print(f"Hierarchical SC:   {hier_acc:.1%}")
    print(f"Improvement:       {hier_acc - vanilla_acc:+.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Hierarchical SC-TRM models Sudoku's natural structure:
   - Board level: global solving progress
   - Unit level: row/col/box completion
   - Cell level: individual predictions

2. Top-down guidance helps cells understand context:
   - Easy cells inform harder ones
   - Mirrors human solving strategy

3. Information flows both ways:
   - Bottom-up: cells → units → board
   - Top-down: board → units → cells
""")


if __name__ == "__main__":
    train_and_evaluate()
