"""
Statechart-Inspired Neural Architectures for Sudoku.

Creative explorations of encoding SC semantics in neural layers.

Key Ideas:
1. StateHeadAttention - each head "owns" states, competes for activation
2. GuardGatedTransition - explicit guard networks with STE
3. OrthogonalRegionNetwork - parallel independent processing streams
4. HistoryAugmentedState - explicit memory for history states
5. HierarchicalLCA - proper LCA computation for transition semantics
6. EventDrivenProcessor - event queue with priority processing
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention_bias import build_constraint_affinity_matrix


# =============================================================================
# IDEA 1: State-Head Attention
# Each attention head specializes in a subset of "states" (cell configurations)
# Heads compete to determine which state is active
# =============================================================================

@dataclass
class StateHeadConfig:
    """Config for state-head attention model."""
    hidden_dim: int = 128
    num_state_heads: int = 9  # One head per digit possibility
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1
    state_temperature: float = 1.0  # For state competition softmax


class StateHeadAttention(nn.Module):
    """
    Attention where each head specializes in one "state" (digit).

    Key insight: In a statechart, only one state in an OR-region is active.
    We model this as heads competing via softmax.
    """

    def __init__(self, hidden_dim: int, num_heads: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads  # = 9 for Sudoku (one per digit)
        # Ensure head_dim divides evenly
        self.head_dim = max(hidden_dim // num_heads, 1)
        self.temperature = temperature

        # Use full hidden dim for projections
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_dim)

        # Competition gate: heads compete for activation
        self.competition_gate = nn.Linear(hidden_dim, num_heads)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Returns: (output, state_activations)
        state_activations: [B, 81, num_heads] - which head is "active" for each cell
        """
        B, N, D = x.shape

        # Standard attention per head
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_scores, axis=-1)

        head_outputs = mx.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]

        # Compute state competition: which head should be active for each position?
        competition_logits = self.competition_gate(x)  # [B, N, num_heads]
        state_activations = mx.softmax(competition_logits / self.temperature, axis=-1)

        # Weight head outputs by state activations
        # head_outputs: [B, num_heads, N, head_dim]
        # state_activations: [B, N, num_heads] -> [B, num_heads, N, 1]
        weights = state_activations.transpose(0, 2, 1)[:, :, :, None]
        weighted = head_outputs * weights

        # Combine all heads
        out = weighted.transpose(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        return out, state_activations


class StateHeadTRM(nn.Module):
    """TRM with state-head attention."""

    def __init__(self, config: StateHeadConfig):
        super().__init__()
        self.config = config

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                'attn': StateHeadAttention(config.hidden_dim, config.num_state_heads, config.state_temperature),
                'ff': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.ff_dim),
                    nn.GELU(),
                    nn.Linear(config.ff_dim, config.hidden_dim),
                ),
                'ln1': nn.LayerNorm(config.hidden_dim),
                'ln2': nn.LayerNorm(config.hidden_dim),
            })

        self.h_context = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Output: use state activations directly for prediction
        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        all_state_activations = []

        for hi in range(self.config.H_cycles):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context(pooled)

            for li in range(self.config.L_cycles):
                h = h + h_ctx[:, None, :]

                for layer in self.layers:
                    h_norm = layer['ln1'](h)
                    attn_out, state_acts = layer['attn'](h_norm)
                    h = h + attn_out

                    h_norm = layer['ln2'](h)
                    h = h + layer['ff'](h_norm)

                    all_state_activations.append(state_acts)

        logits = self.output_head(self.ln_out(h))
        predictions = mx.argmax(logits, axis=-1) + 1

        return {
            'logits': logits,
            'predictions': predictions,
            'state_activations': all_state_activations[-1] if all_state_activations else None,
        }

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy}


# =============================================================================
# IDEA 2: Guard-Gated Transitions
# Explicit guard networks that produce hard gates with STE
# =============================================================================

@dataclass
class GuardGatedConfig:
    """Config for guard-gated model."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_guards: int = 27  # 9 rows + 9 cols + 9 boxes
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    guard_threshold: float = 0.5


def ste_hard_gate(soft_gate: mx.array, threshold: float = 0.5) -> mx.array:
    """Straight-through estimator for hard gating."""
    hard = (soft_gate > threshold).astype(mx.float32)
    # STE: forward uses hard, backward uses soft gradient
    return soft_gate + mx.stop_gradient(hard - soft_gate)


class GuardNetwork(nn.Module):
    """
    Network that computes guard conditions for transitions.

    In statecharts, guards are boolean conditions that enable/disable transitions.
    We model them as neural networks that output gate values.
    """

    def __init__(self, hidden_dim: int, num_guards: int):
        super().__init__()
        self.num_guards = num_guards

        # Guard network: computes condition for each constraint unit
        self.guard_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_guards),
            nn.Sigmoid(),
        )

        # Context aggregation for guard evaluation
        self.context_agg = nn.Linear(hidden_dim * 9, hidden_dim)

    def get_unit_cells(self, unit_idx: int) -> List[int]:
        """Get cell indices for a constraint unit."""
        if unit_idx < 9:  # Row
            return [unit_idx * 9 + c for c in range(9)]
        elif unit_idx < 18:  # Column
            col = unit_idx - 9
            return [r * 9 + col for r in range(9)]
        else:  # Box
            box = unit_idx - 18
            box_row, box_col = box // 3, box % 3
            return [(box_row * 3 + r) * 9 + (box_col * 3 + c) for r in range(3) for c in range(3)]

    def __call__(self, cell_states: mx.array) -> mx.array:
        """
        Compute guard values for each constraint unit.

        Args:
            cell_states: [B, 81, hidden_dim]

        Returns:
            guards: [B, 27] - soft gate value for each constraint
        """
        B = cell_states.shape[0]

        # Aggregate context for each unit
        unit_contexts = []
        for unit_idx in range(self.num_guards):
            cell_indices = self.get_unit_cells(unit_idx)
            unit_cells = mx.concatenate([
                cell_states[:, idx:idx+1, :] for idx in cell_indices
            ], axis=1)  # [B, 9, hidden_dim]
            unit_context = self.context_agg(unit_cells.reshape(B, -1))  # [B, hidden_dim]
            unit_contexts.append(unit_context[:, None, :])

        unit_contexts = mx.concatenate(unit_contexts, axis=1)  # [B, 27, hidden_dim]

        # Pool and compute guards
        pooled = mx.mean(unit_contexts, axis=1)  # [B, hidden_dim]
        guards = self.guard_net(pooled)  # [B, 27]

        return guards


class GuardGatedAttention(nn.Module):
    """Attention gated by guard conditions."""

    def __init__(self, hidden_dim: int, num_heads: int, num_guards: int, threshold: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.threshold = threshold

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Map guards to attention mask
        self.guard_to_mask = nn.Linear(num_guards, 81)

    def __call__(self, x: mx.array, guards: mx.array) -> mx.array:
        B, N, D = x.shape

        # Compute attention
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Apply guard-based gating
        # guards: [B, 27] -> mask: [B, 81]
        guard_mask = mx.sigmoid(self.guard_to_mask(guards))  # [B, 81]
        hard_mask = ste_hard_gate(guard_mask, self.threshold)

        # Apply mask to attention scores
        mask_expanded = hard_mask[:, None, None, :]  # [B, 1, 1, 81]
        scores = scores * mask_expanded + (1 - mask_expanded) * (-1e9)

        attn_weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

        return self.o_proj(out)


class GuardGatedTRM(nn.Module):
    """TRM with guard-gated transitions."""

    def __init__(self, config: GuardGatedConfig):
        super().__init__()
        self.config = config

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        self.guard_net = GuardNetwork(config.hidden_dim, config.num_guards)

        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                'attn': GuardGatedAttention(config.hidden_dim, config.num_heads, config.num_guards, config.guard_threshold),
                'ff': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.ff_dim),
                    nn.GELU(),
                    nn.Linear(config.ff_dim, config.hidden_dim),
                ),
                'ln1': nn.LayerNorm(config.hidden_dim),
                'ln2': nn.LayerNorm(config.hidden_dim),
            })

        self.h_context = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        for hi in range(self.config.H_cycles):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context(pooled)

            for li in range(self.config.L_cycles):
                h = h + h_ctx[:, None, :]

                # Compute guards
                guards = self.guard_net(h)

                for layer in self.layers:
                    h_norm = layer['ln1'](h)
                    h = h + layer['attn'](h_norm, guards)

                    h_norm = layer['ln2'](h)
                    h = h + layer['ff'](h_norm)

        logits = self.output_head(self.ln_out(h))
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy}


# =============================================================================
# IDEA 3: Orthogonal Region Network
# Parallel independent processing streams that synchronize
# =============================================================================

@dataclass
class OrthogonalConfig:
    """Config for orthogonal region model."""
    hidden_dim: int = 128
    num_regions: int = 3  # Row, Column, Box regions
    region_dim: int = 42  # Per-region hidden dim
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    sync_every: int = 2  # Synchronize regions every N L-cycles


class OrthogonalRegion(nn.Module):
    """
    Independent processing region (like parallel states in SC).

    Each region processes its view independently, then syncs.
    """

    def __init__(self, region_type: str, hidden_dim: int, region_dim: int):
        super().__init__()
        self.region_type = region_type  # 'row', 'col', 'box'
        self.hidden_dim = hidden_dim
        self.region_dim = region_dim

        # Project from cell space to region space
        self.project_in = nn.Linear(hidden_dim, region_dim)

        # Region-specific processing
        self.region_net = nn.Sequential(
            nn.Linear(region_dim * 9, region_dim * 9),  # 9 cells per unit
            nn.GELU(),
            nn.Linear(region_dim * 9, region_dim * 9),
        )
        self.ln = nn.LayerNorm(region_dim)

        # Project back to cell space
        self.project_out = nn.Linear(region_dim, hidden_dim)

    def get_unit_indices(self, unit_idx: int) -> List[int]:
        """Get cell indices for a unit in this region type."""
        if self.region_type == 'row':
            return [unit_idx * 9 + c for c in range(9)]
        elif self.region_type == 'col':
            return [r * 9 + unit_idx for r in range(9)]
        else:  # box
            box_row, box_col = unit_idx // 3, unit_idx % 3
            return [(box_row * 3 + r) * 9 + (box_col * 3 + c) for r in range(3) for c in range(3)]

    def __call__(self, cell_states: mx.array) -> mx.array:
        """
        Process cells through region-specific view.

        Args:
            cell_states: [B, 81, hidden_dim]

        Returns:
            region_output: [B, 81, hidden_dim] - contribution from this region
        """
        B = cell_states.shape[0]

        # Project to region space
        region_states = self.project_in(cell_states)  # [B, 81, region_dim]

        # Process each unit (9 units per region type)
        outputs = mx.zeros((B, 81, self.region_dim))

        for unit_idx in range(9):
            cell_indices = self.get_unit_indices(unit_idx)

            # Gather cells for this unit
            unit_cells = mx.concatenate([
                region_states[:, idx:idx+1, :] for idx in cell_indices
            ], axis=1)  # [B, 9, region_dim]

            # Process as a unit
            unit_flat = unit_cells.reshape(B, -1)  # [B, 9*region_dim]
            unit_processed = self.region_net(unit_flat)  # [B, 9*region_dim]
            unit_processed = unit_processed.reshape(B, 9, self.region_dim)
            unit_processed = self.ln(unit_processed)

            # Scatter back to cell positions
            for i, idx in enumerate(cell_indices):
                mask = mx.arange(81) == idx
                mask = mask[None, :, None]
                outputs = outputs + mask * unit_processed[:, i:i+1, :]

        return self.project_out(outputs)


class OrthogonalTRM(nn.Module):
    """
    TRM with orthogonal (parallel) regions.

    Mimics parallel states in statecharts: Row, Col, Box regions
    process independently and synchronize periodically.
    """

    def __init__(self, config: OrthogonalConfig):
        super().__init__()
        self.config = config

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # Orthogonal regions
        self.row_region = OrthogonalRegion('row', config.hidden_dim, config.region_dim)
        self.col_region = OrthogonalRegion('col', config.hidden_dim, config.region_dim)
        self.box_region = OrthogonalRegion('box', config.hidden_dim, config.region_dim)

        # Synchronization layer
        self.sync_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.sync_ln = nn.LayerNorm(config.hidden_dim)

        # Global refinement
        self.global_attn = nn.MultiHeadAttention(config.hidden_dim, 4)
        self.global_ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
        )
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        self.h_context = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        for hi in range(self.config.H_cycles):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context(pooled)

            for li in range(self.config.L_cycles):
                h = h + h_ctx[:, None, :]

                # Process through orthogonal regions
                row_out = self.row_region(h)
                col_out = self.col_region(h)
                box_out = self.box_region(h)

                # Synchronize if needed
                if (li + 1) % self.config.sync_every == 0:
                    combined = mx.concatenate([row_out, col_out, box_out], axis=-1)
                    sync_out = self.sync_layer(combined)
                    h = self.sync_ln(h + sync_out)
                else:
                    # Average contributions
                    h = h + (row_out + col_out + box_out) / 3.0

                # Global refinement
                h_norm = self.ln1(h)
                h = h + self.global_attn(h_norm, h_norm, h_norm)
                h_norm = self.ln2(h)
                h = h + self.global_ff(h_norm)

        logits = self.output_head(self.ln_out(h))
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy}


# =============================================================================
# IDEA 4: History-Augmented Model
# Explicit memory for history states
# =============================================================================

@dataclass
class HistoryConfig:
    """Config for history-augmented model."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    history_depth: int = 2  # How many past states to remember
    use_deep_history: bool = True  # Deep vs shallow history


class HistoryMemory(nn.Module):
    """
    Explicit memory for history states.

    In statecharts, history pseudo-states remember the last active substate.
    We model this as an explicit memory mechanism.
    """

    def __init__(self, hidden_dim: int, history_depth: int, use_deep: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_depth = history_depth
        self.use_deep = use_deep

        # Memory write gate
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Memory read attention
        self.read_query = nn.Linear(hidden_dim, hidden_dim)
        self.read_key = nn.Linear(hidden_dim, hidden_dim)
        self.read_value = nn.Linear(hidden_dim, hidden_dim)

        # Combine current with history
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def __call__(self, current: mx.array, history_stack: List[mx.array]) -> Tuple[mx.array, List[mx.array]]:
        """
        Update history and read from it.

        Args:
            current: [B, 81, hidden_dim] - current state
            history_stack: list of past states

        Returns:
            output: [B, 81, hidden_dim] - state augmented with history
            new_history: updated history stack
        """
        B = current.shape[0]

        # Read from history
        if len(history_stack) > 0:
            # Stack history: [depth, B, 81, hidden_dim]
            history_tensor = mx.stack(history_stack, axis=0)

            # Attention over history
            q = self.read_query(current)  # [B, 81, hidden_dim]

            if self.use_deep:
                # Deep history: attend over all past states
                k = self.read_key(history_tensor)  # [depth, B, 81, hidden_dim]
                v = self.read_value(history_tensor)

                # Reshape for attention
                depth = k.shape[0]
                k = k.transpose(1, 2, 0, 3)  # [B, 81, depth, hidden_dim]
                v = v.transpose(1, 2, 0, 3)
                q = q[:, :, None, :]  # [B, 81, 1, hidden_dim]

                scores = mx.sum(q * k, axis=-1) / math.sqrt(self.hidden_dim)  # [B, 81, depth]
                attn = mx.softmax(scores, axis=-1)  # [B, 81, depth]
                history_read = mx.sum(attn[:, :, :, None] * v, axis=2)  # [B, 81, hidden_dim]
            else:
                # Shallow history: only most recent
                history_read = self.read_value(history_stack[-1])

            # Combine current with history
            combined = mx.concatenate([current, history_read], axis=-1)
            output = self.combine(combined)
        else:
            output = current

        # Write to history
        new_history = history_stack.copy()
        new_history.append(current)
        if len(new_history) > self.history_depth:
            new_history = new_history[-self.history_depth:]

        return output, new_history


class HistoryTRM(nn.Module):
    """TRM with history memory."""

    def __init__(self, config: HistoryConfig):
        super().__init__()
        self.config = config

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        self.history_mem = HistoryMemory(config.hidden_dim, config.history_depth, config.use_deep_history)

        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                'attn': nn.MultiHeadAttention(config.hidden_dim, config.num_heads),
                'ff': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.ff_dim),
                    nn.GELU(),
                    nn.Linear(config.ff_dim, config.hidden_dim),
                ),
                'ln1': nn.LayerNorm(config.hidden_dim),
                'ln2': nn.LayerNorm(config.hidden_dim),
            })

        self.h_context = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        history_stack = []

        for hi in range(self.config.H_cycles):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context(pooled)

            for li in range(self.config.L_cycles):
                h = h + h_ctx[:, None, :]

                # Process with history
                h, history_stack = self.history_mem(h, history_stack)

                for layer in self.layers:
                    h_norm = layer['ln1'](h)
                    h = h + layer['attn'](h_norm, h_norm, h_norm)

                    h_norm = layer['ln2'](h)
                    h = h + layer['ff'](h_norm)

        logits = self.output_head(self.ln_out(h))
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy}


# =============================================================================
# IDEA 5: Evolved Constraint Network
# Neural network that learns constraint satisfaction patterns
# =============================================================================

@dataclass
class EvolvedConstraintConfig:
    """Config for evolved constraint model."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    H_cycles: int = 4
    L_cycles: int = 4
    constraint_dim: int = 64
    num_constraint_types: int = 3  # row, col, box


class ConstraintSatisfactionLayer(nn.Module):
    """
    Layer that explicitly checks and enforces constraints.

    Learns to detect violations and propose corrections.
    """

    def __init__(self, hidden_dim: int, constraint_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.constraint_dim = constraint_dim

        # Violation detector: for each cell, detect if it violates constraints
        self.violation_detector = nn.Sequential(
            nn.Linear(hidden_dim * 9, constraint_dim),  # 9 cells in unit
            nn.GELU(),
            nn.Linear(constraint_dim, 9),  # Score for each cell
            nn.Sigmoid(),
        )

        # Correction proposer: suggest fix for violations
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim + constraint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Constraint embedding
        self.constraint_embed = nn.Linear(9, constraint_dim)  # 9 digits

    def get_unit_cells(self, unit_type: str, unit_idx: int) -> List[int]:
        if unit_type == 'row':
            return [unit_idx * 9 + c for c in range(9)]
        elif unit_type == 'col':
            return [r * 9 + unit_idx for r in range(9)]
        else:  # box
            box_row, box_col = unit_idx // 3, unit_idx % 3
            return [(box_row * 3 + r) * 9 + (box_col * 3 + c) for r in range(3) for c in range(3)]

    def __call__(self, h: mx.array, predictions: mx.array) -> mx.array:
        """
        Check constraints and propose corrections.

        Args:
            h: [B, 81, hidden_dim] - hidden states
            predictions: [B, 81, 9] - current prediction logits

        Returns:
            corrections: [B, 81, hidden_dim] - correction signals
        """
        B = h.shape[0]
        corrections = mx.zeros_like(h)

        # Check each constraint unit
        for unit_type in ['row', 'col', 'box']:
            for unit_idx in range(9):
                cell_indices = self.get_unit_cells(unit_type, unit_idx)

                # Gather unit cells
                unit_h = mx.concatenate([
                    h[:, idx:idx+1, :] for idx in cell_indices
                ], axis=1)  # [B, 9, hidden_dim]

                unit_pred = mx.concatenate([
                    predictions[:, idx:idx+1, :] for idx in cell_indices
                ], axis=1)  # [B, 9, 9]

                # Detect violations
                unit_flat = unit_h.reshape(B, -1)  # [B, 9*hidden_dim]
                violation_scores = self.violation_detector(unit_flat)  # [B, 9]

                # Compute constraint context
                pred_probs = mx.softmax(unit_pred, axis=-1)  # [B, 9, 9]
                digit_counts = mx.sum(pred_probs, axis=1)  # [B, 9] - count per digit
                constraint_ctx = self.constraint_embed(digit_counts)  # [B, constraint_dim]

                # Propose corrections for violated cells
                for i, idx in enumerate(cell_indices):
                    cell_h = h[:, idx, :]  # [B, hidden_dim]
                    cell_input = mx.concatenate([cell_h, constraint_ctx], axis=-1)
                    cell_correction = self.correction(cell_input)  # [B, hidden_dim]

                    # Weight by violation score
                    weight = violation_scores[:, i:i+1]  # [B, 1]

                    mask = mx.arange(81) == idx
                    mask = mask[None, :, None]
                    corrections = corrections + mask * (cell_correction[:, None, :] * weight[:, :, None])

        return corrections


class EvolvedConstraintTRM(nn.Module):
    """TRM with learned constraint satisfaction."""

    def __init__(self, config: EvolvedConstraintConfig):
        super().__init__()
        self.config = config

        self.cell_embed = nn.Embedding(10, config.hidden_dim)
        self.pos_embed = nn.Embedding(81, config.hidden_dim)

        # Constraint affinity for attention bias
        self.constraint_affinity = build_constraint_affinity_matrix()

        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                'attn': nn.MultiHeadAttention(config.hidden_dim, config.num_heads),
                'constraint': ConstraintSatisfactionLayer(config.hidden_dim, config.constraint_dim),
                'ff': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.ff_dim),
                    nn.GELU(),
                    nn.Linear(config.ff_dim, config.hidden_dim),
                ),
                'ln1': nn.LayerNorm(config.hidden_dim),
                'ln2': nn.LayerNorm(config.hidden_dim),
                'ln3': nn.LayerNorm(config.hidden_dim),
            })

        self.h_context = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Intermediate prediction head for constraint checking
        self.pred_head = nn.Linear(config.hidden_dim, 9)

        self.output_head = nn.Linear(config.hidden_dim, 9)
        self.ln_out = nn.LayerNorm(config.hidden_dim)

    def solve(self, puzzle: mx.array, **kwargs) -> Dict[str, mx.array]:
        B = puzzle.shape[0]
        positions = mx.broadcast_to(mx.arange(81)[None, :], (B, 81))

        h = self.cell_embed(puzzle.astype(mx.int32)) + self.pos_embed(positions)

        for hi in range(self.config.H_cycles):
            pooled = mx.mean(h, axis=1)
            h_ctx = self.h_context(pooled)

            for li in range(self.config.L_cycles):
                h = h + h_ctx[:, None, :]

                for layer in self.layers:
                    # Attention
                    h_norm = layer['ln1'](h)
                    h = h + layer['attn'](h_norm, h_norm, h_norm)

                    # Constraint satisfaction
                    h_norm = layer['ln2'](h)
                    current_pred = self.pred_head(h_norm)
                    corrections = layer['constraint'](h_norm, current_pred)
                    h = h + corrections

                    # FFN
                    h_norm = layer['ln3'](h)
                    h = h + layer['ff'](h_norm)

        logits = self.output_head(self.ln_out(h))
        predictions = mx.argmax(logits, axis=-1) + 1

        return {'logits': logits, 'predictions': predictions}

    def loss(self, puzzle: mx.array, solution: mx.array, **kwargs):
        result = self.solve(puzzle)
        logits = result['logits']

        targets = (solution - 1).astype(mx.int32)
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)

        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        loss = -mx.mean(log_probs[batch_indices, targets_flat])

        cell_accuracy = mx.mean((result['predictions'] == solution).astype(mx.float32))
        return loss, {'cell_accuracy': cell_accuracy}


# =============================================================================
# Factory and Testing
# =============================================================================

def create_sc_neural_model(model_type: str):
    """Create an SC-inspired neural model by type."""
    if model_type == 'state_head':
        return StateHeadTRM(StateHeadConfig())
    elif model_type == 'guard_gated':
        return GuardGatedTRM(GuardGatedConfig())
    elif model_type == 'orthogonal':
        return OrthogonalTRM(OrthogonalConfig())
    elif model_type == 'history':
        return HistoryTRM(HistoryConfig())
    elif model_type == 'evolved_constraint':
        return EvolvedConstraintTRM(EvolvedConstraintConfig())
    else:
        raise ValueError(f"Unknown SC neural model type: {model_type}")


if __name__ == "__main__":
    print("Testing SC-inspired neural models...")

    for name in ['state_head', 'guard_gated', 'orthogonal', 'history', 'evolved_constraint']:
        print(f"\n=== {name} ===")
        try:
            model = create_sc_neural_model(name)
            batch = mx.zeros((2, 81), dtype=mx.int32)
            solution = mx.ones((2, 81), dtype=mx.int32)

            result = model.solve(batch)
            print(f"  Output shape: {result['predictions'].shape}")

            loss, metrics = model.loss(batch, solution)
            mx.eval(loss)
            print(f"  Loss: {float(loss):.4f}")
            print(f"  Cell accuracy: {float(metrics.get('cell_accuracy', 0)):.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
