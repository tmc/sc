"""
9x9 Sudoku as Statechart with TRM-style Iterative Refinement

Structure:
- 81 cells in a 9x9 grid
- Each cell has 10 states: empty (0), 1, 2, ..., 9
- Guards: row_valid, col_valid, box_valid (3x3 boxes)
- TRM-style two-level iteration: H_cycles (outer) × L_cycles (inner)

Statechart interpretation:
- Board state = parallel composition of 81 cell statecharts
- Each iteration = one statechart step across all cells
- Constraint satisfaction = guard evaluation on transitions
- Non-autoregressive: all cells update in parallel
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SolveResult:
    """Result from solve() method."""
    final_board: mx.array      # [B, 81, 10] soft states
    predictions: mx.array      # [B, 81] argmax predictions
    halt_step: Optional[int]   # Step where halting occurred (if learned)
    all_iterations: Optional[List[mx.array]]  # All intermediate states


class SudokuCell9x9(nn.Module):
    """
    Single cell in 9x9 Sudoku grid.

    States: 0 (empty), 1, 2, ..., 9
    Transitions: PLACE_1 through PLACE_9
    """

    NUM_VALUES = 10  # empty + 9 digits

    def __init__(self, cell_id: int, hidden_dim: int = 64):
        super().__init__()
        self.cell_id = cell_id
        self.row = cell_id // 9
        self.col = cell_id % 9
        self.box = (self.row // 3) * 3 + (self.col // 3)

        # State embedding
        self.state_embed = nn.Embedding(self.NUM_VALUES, hidden_dim)

        # Transition network: context → scores for placing 1-9
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),  # Score for placing 1-9
        )

    def forward_transition(
        self,
        current_state: mx.array,  # [B, NUM_VALUES] soft state
        context: mx.array,        # [B, hidden_dim] from board
        guard_values: mx.array,   # [B, 9] guard satisfaction for each digit
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Compute new state after considering transitions.

        Args:
            current_state: [B, 10] soft state (empty, 1, 2, ..., 9)
            context: [B, hidden_dim] context from board encoder
            guard_values: [B, 9] constraint satisfaction for placing 1-9
            temperature: Softmax temperature (lower = more confident)

        Returns:
            new_state: [B, 10] updated soft state
        """
        B = current_state.shape[0]

        # Transition scores
        trans_scores = self.transition_net(context)  # [B, 9]

        # Apply temperature
        trans_scores = trans_scores / temperature

        # Apply guards (mask invalid placements)
        masked_scores = trans_scores + mx.log(guard_values + 1e-7)

        # Probability of transitioning to each digit
        trans_probs = mx.softmax(masked_scores, axis=-1)  # [B, 9]

        # Current probability of being empty
        p_empty = current_state[:, 0:1]  # [B, 1]

        # Transition rate (how much to move from empty to placed)
        # Higher when model is confident, modulated by guards
        trans_rate = 0.3  # Can be learned

        # New digit probabilities
        # p(d) = old_p(d) + p_empty * trans_rate * trans_prob(d) * guard(d)
        new_digit_probs = (
            current_state[:, 1:] +
            p_empty * trans_rate * trans_probs * guard_values
        )

        # New empty probability
        total_transition = mx.sum(trans_probs * guard_values, axis=-1, keepdims=True)
        new_empty = p_empty * (1.0 - trans_rate * total_transition)

        # Combine
        new_state = mx.concatenate([new_empty, new_digit_probs], axis=-1)

        # Normalize to valid distribution
        return new_state / mx.sum(new_state, axis=-1, keepdims=True)


class SudokuConstraintGuard9x9(nn.Module):
    """
    Evaluates Sudoku constraints as soft guards.

    For placing digit d in cell (r, c):
    - row_valid: d not already in row r
    - col_valid: d not already in col c
    - box_valid: d not already in 3x3 box

    Guards are soft (differentiable) approximations.
    """

    def __init__(self, hidden_dim: int = 64, learned: bool = True):
        super().__init__()
        self.grid_size = 9
        self.box_size = 3
        self.learned = learned

        if learned:
            # Learnable constraint networks (can discover Sudoku rules)
            self.row_net = nn.Linear(9 * 10, 9)  # 9 cells × 10 states → 9 digit guards
            self.col_net = nn.Linear(9 * 10, 9)
            self.box_net = nn.Linear(9 * 10, 9)
        else:
            # Use explicit constraint logic (no learning)
            pass

    def _get_row_cells(self, cell_id: int) -> List[int]:
        """Cell IDs in same row."""
        row = cell_id // 9
        return [row * 9 + c for c in range(9)]

    def _get_col_cells(self, cell_id: int) -> List[int]:
        """Cell IDs in same column."""
        col = cell_id % 9
        return [r * 9 + col for r in range(9)]

    def _get_box_cells(self, cell_id: int) -> List[int]:
        """Cell IDs in same 3x3 box."""
        row, col = cell_id // 9, cell_id % 9
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        cells = []
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                cells.append(r * 9 + c)
        return cells

    def __call__(
        self,
        board_states: mx.array,  # [B, 81, 10] soft states for all cells
        cell_id: int,
    ) -> mx.array:
        """
        Compute guard values for placing each digit in cell.

        Args:
            board_states: [B, 81, 10] soft state for each cell
            cell_id: Which cell to check

        Returns:
            guards: [B, 9] satisfaction for placing 1-9
        """
        B = board_states.shape[0]

        if self.learned:
            # Get states of cells in same row/col/box
            row_cells = self._get_row_cells(cell_id)
            col_cells = self._get_col_cells(cell_id)
            box_cells = self._get_box_cells(cell_id)

            # Flatten states
            row_states = board_states[:, row_cells, :].reshape(B, -1)  # [B, 90]
            col_states = board_states[:, col_cells, :].reshape(B, -1)
            box_states = board_states[:, box_cells, :].reshape(B, -1)

            # Compute guards via learned networks
            row_guard = mx.sigmoid(self.row_net(row_states))  # [B, 9]
            col_guard = mx.sigmoid(self.col_net(col_states))
            box_guard = mx.sigmoid(self.box_net(box_states))

            # Combined: all constraints must be satisfied
            return row_guard * col_guard * box_guard
        else:
            # Explicit constraint logic
            return self._explicit_guards(board_states, cell_id)

    def _explicit_guards(
        self,
        board_states: mx.array,
        cell_id: int
    ) -> mx.array:
        """
        Explicit (non-learned) constraint guards.

        For each digit d, guard = 1 if d not already placed in row/col/box.
        """
        B = board_states.shape[0]

        # Get digit probabilities (columns 1-9)
        digit_probs = board_states[:, :, 1:]  # [B, 81, 9]

        # Row constraint: for each digit, check if it's NOT in the row
        row = cell_id // 9
        row_cells = [row * 9 + c for c in range(9) if row * 9 + c != cell_id]
        row_probs = digit_probs[:, row_cells, :]  # [B, 8, 9]
        row_max = mx.max(row_probs, axis=1)  # [B, 9] - max prob for each digit
        row_guard = 1.0 - row_max  # High if digit NOT in row

        # Column constraint
        col = cell_id % 9
        col_cells = [r * 9 + col for r in range(9) if r * 9 + col != cell_id]
        col_probs = digit_probs[:, col_cells, :]
        col_max = mx.max(col_probs, axis=1)
        col_guard = 1.0 - col_max

        # Box constraint
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        box_cells = [
            r * 9 + c
            for r in range(box_row, box_row + 3)
            for c in range(box_col, box_col + 3)
            if r * 9 + c != cell_id
        ]
        box_probs = digit_probs[:, box_cells, :]
        box_max = mx.max(box_probs, axis=1)
        box_guard = 1.0 - box_max

        # Combined guard
        return row_guard * col_guard * box_guard


class SudokuStatechart9x9(nn.Module):
    """
    Full 9x9 Sudoku as differentiable statechart with TRM-style iteration.

    Architecture:
    - 81 parallel cell statecharts
    - Constraint guards for row/col/box validity
    - Board encoder for global context
    - Two-level iteration: H_cycles (macro) × L_cycles (micro)
    - Optional learned halting
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        H_cycles: int = 3,
        L_cycles: int = 6,
        learned_guards: bool = True,
        learned_halting: bool = True,
    ):
        super().__init__()
        self.grid_size = 9
        self.num_cells = 81
        self.hidden_dim = hidden_dim
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Cell statecharts (shared weights for efficiency)
        self.cell_transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),  # Score for placing 1-9
        )

        # Constraint guard
        self.guard = SudokuConstraintGuard9x9(hidden_dim, learned=learned_guards)

        # Board encoder: convert soft states to context
        self.board_encoder = nn.Sequential(
            nn.Linear(81 * 10, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Cell position embedding
        self.pos_embed = nn.Embedding(81, hidden_dim)

        # H-cycle context updater (optional)
        self.h_context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned halting
        self.learned_halting = learned_halting
        if learned_halting:
            self.halt_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Output projection to logits
        self.output_head = nn.Linear(10, 10)

    def encode_board(self, board_states: mx.array) -> mx.array:
        """
        Encode board state to context vector.

        Args:
            board_states: [B, 81, 10] soft states

        Returns:
            context: [B, hidden_dim]
        """
        B = board_states.shape[0]
        flat = board_states.reshape(B, -1)  # [B, 810]
        return self.board_encoder(flat)

    def step(
        self,
        board_states: mx.array,
        context: mx.array,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        One L-cycle step: update all cells in parallel.

        Args:
            board_states: [B, 81, 10] current soft states
            context: [B, hidden_dim] global context
            temperature: Softmax temperature

        Returns:
            new_states: [B, 81, 10] updated soft states
        """
        B = board_states.shape[0]

        new_states = []
        for cell_id in range(self.num_cells):
            # Position-specific context
            pos_emb = self.pos_embed(mx.array([cell_id]))  # [1, hidden_dim]
            pos_emb = mx.broadcast_to(pos_emb, (B, self.hidden_dim))
            cell_context = context + pos_emb

            # Get guard values
            guards = self.guard(board_states, cell_id)  # [B, 9]

            # Transition scores
            trans_scores = self.cell_transition(cell_context)  # [B, 9]
            trans_scores = trans_scores / temperature

            # Apply guards
            masked_scores = trans_scores + mx.log(guards + 1e-7)
            trans_probs = mx.softmax(masked_scores, axis=-1)

            # Current state
            current = board_states[:, cell_id, :]  # [B, 10]
            p_empty = current[:, 0:1]

            # Update probabilities
            trans_rate = 0.3
            new_digit_probs = (
                current[:, 1:] +
                p_empty * trans_rate * trans_probs * guards
            )
            total_trans = mx.sum(trans_probs * guards, axis=-1, keepdims=True)
            new_empty = p_empty * (1.0 - trans_rate * total_trans)

            new_state = mx.concatenate([new_empty, new_digit_probs], axis=-1)
            new_state = new_state / mx.sum(new_state, axis=-1, keepdims=True)
            new_states.append(new_state)

        return mx.stack(new_states, axis=1)  # [B, 81, 10]

    def solve(
        self,
        initial_board: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_all: bool = False,
        temperature_schedule: Optional[List[float]] = None,
    ) -> SolveResult:
        """
        Solve Sudoku via TRM-style iterative refinement.

        Args:
            initial_board: [B, 81, 10] initial soft states
            H_cycles: Number of outer cycles (default: self.H_cycles)
            L_cycles: Number of inner cycles per outer (default: self.L_cycles)
            return_all: Return all intermediate states
            temperature_schedule: Temperature per H-cycle (for annealing)

        Returns:
            SolveResult with final board and predictions
        """
        H = H_cycles or self.H_cycles
        L = L_cycles or self.L_cycles

        board = initial_board
        all_iterations = [] if return_all else None

        for h in range(H):
            # Temperature for this H-cycle
            if temperature_schedule is not None:
                temp = temperature_schedule[min(h, len(temperature_schedule) - 1)]
            else:
                # Default: anneal from 1.0 to 0.5
                temp = 1.0 - 0.5 * (h / max(H - 1, 1))

            # Get H-level context (updated each outer cycle)
            h_context = self.encode_board(board)
            h_context = self.h_context_net(h_context)

            # L-cycles: inner refinement loop
            for l in range(L):
                # Get current context
                context = self.encode_board(board) + h_context

                # Step all cells
                board = self.step(board, context, temperature=temp)

                if return_all:
                    all_iterations.append(board)

            # Check halting (optional)
            if self.learned_halting:
                halt_context = self.encode_board(board)
                halt_logit = self.halt_net(halt_context)  # [B, 1]
                # For now, just track but don't early stop during training
                # Could use for adaptive computation during inference

        # Get predictions
        logits = self.output_head(board)  # [B, 81, 10]
        predictions = mx.argmax(logits, axis=-1)  # [B, 81]

        return SolveResult(
            final_board=board,
            predictions=predictions,
            halt_step=None,
            all_iterations=all_iterations,
        )

    def forward(
        self,
        board: mx.array,  # [B, 81] int values 0-9
        return_all: bool = False,
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        """
        Forward pass: board → logits.

        Args:
            board: [B, 81] integer board (0 = empty, 1-9 = digit)
            return_all: Return all intermediate states

        Returns:
            logits: [B, 81, 10] logits for each cell
            all_iterations: List of intermediate states (if return_all)
        """
        # Convert to soft states
        soft_board = self.board_to_soft(board)

        # Solve
        result = self.solve(soft_board, return_all=return_all)

        # Get logits
        logits = self.output_head(result.final_board)

        return logits, result.all_iterations

    def board_to_soft(self, board: mx.array) -> mx.array:
        """
        Convert discrete board to soft states.

        Args:
            board: [B, 81] int values 0-9

        Returns:
            soft: [B, 81, 10] one-hot encoded
        """
        # One-hot encode
        one_hot = mx.one_hot(board, 10)  # [B, 81, 10]

        # For empty cells, start with uniform over 1-9
        empty_mask = (board == 0)[:, :, None]  # [B, 81, 1]
        uniform = mx.concatenate([
            mx.zeros((board.shape[0], 81, 1)),  # No prob for empty
            mx.ones((board.shape[0], 81, 9)) / 9,  # Uniform over 1-9
        ], axis=-1)

        return mx.where(empty_mask, uniform, one_hot)

    def soft_to_board(self, soft: mx.array) -> mx.array:
        """
        Convert soft states to discrete board.

        Args:
            soft: [B, 81, 10]

        Returns:
            board: [B, 81] argmax values
        """
        return mx.argmax(soft, axis=-1)

    def is_valid(self, board: mx.array) -> Tuple[mx.array, dict]:
        """
        Check if board satisfies Sudoku constraints.

        Args:
            board: [B, 81] discrete values 1-9

        Returns:
            valid: [B] boolean per board
            details: Dict with per-constraint validity
        """
        B = board.shape[0]
        row_valid = mx.ones((B,), dtype=mx.bool_)
        col_valid = mx.ones((B,), dtype=mx.bool_)
        box_valid = mx.ones((B,), dtype=mx.bool_)

        # Check rows
        for row in range(9):
            row_vals = board[:, row * 9:(row + 1) * 9]
            for d in range(1, 10):
                count = mx.sum(row_vals == d, axis=-1)
                row_valid = row_valid & (count <= 1)

        # Check columns
        for col in range(9):
            col_vals = board[:, col::9]
            for d in range(1, 10):
                count = mx.sum(col_vals == d, axis=-1)
                col_valid = col_valid & (count <= 1)

        # Check boxes
        for box_row in range(3):
            for box_col in range(3):
                box_cells = []
                for r in range(3):
                    for c in range(3):
                        box_cells.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
                box_vals = board[:, box_cells]
                for d in range(1, 10):
                    count = mx.sum(box_vals == d, axis=-1)
                    box_valid = box_valid & (count <= 1)

        valid = row_valid & col_valid & box_valid

        return valid, {
            "row_valid": row_valid,
            "col_valid": col_valid,
            "box_valid": box_valid,
        }


def test_sudoku_statechart_9x9():
    """Test the 9x9 Sudoku statechart."""
    print("=" * 60)
    print("Testing Sudoku Statechart 9x9")
    print("=" * 60)

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
        learned_guards=True,
    )

    # Create a simple test puzzle
    mx.random.seed(42)
    B = 2

    # Generate random incomplete boards
    puzzle = mx.random.randint(0, 10, (B, 81))
    # Make ~50% cells empty
    mask = mx.random.uniform(shape=(B, 81)) < 0.5
    puzzle = mx.where(mask, mx.zeros_like(puzzle), puzzle)

    print(f"\n1. Input puzzle shape: {puzzle.shape}")
    print(f"   Empty cells: {mx.sum(puzzle == 0).item()}")

    # Forward pass
    print("\n2. Running forward pass...")
    logits, _ = model.forward(puzzle)
    print(f"   Output logits shape: {logits.shape}")

    # Check gradient flow
    print("\n3. Testing gradient flow...")
    targets = mx.random.randint(1, 10, (B, 81))

    def loss_fn(model, puzzle, targets):
        logits, _ = model.forward(puzzle)
        # Simple CE loss
        probs = mx.softmax(logits, axis=-1)
        one_hot = mx.one_hot(targets, 10)
        return -mx.mean(mx.sum(one_hot * mx.log(probs + 1e-7), axis=-1))

    loss, grads = nn.value_and_grad(model, loss_fn)(model, puzzle, targets)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Loss: {float(loss):.4f}")
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")

    # Test solve with all iterations
    print("\n4. Testing solve with iteration tracking...")
    soft_puzzle = model.board_to_soft(puzzle)
    result = model.solve(soft_puzzle, H_cycles=2, L_cycles=3, return_all=True)
    print(f"   Final board shape: {result.final_board.shape}")
    print(f"   Predictions shape: {result.predictions.shape}")
    print(f"   Number of iterations: {len(result.all_iterations) if result.all_iterations else 0}")

    # Validate output
    print("\n5. Checking output validity...")
    valid, details = model.is_valid(result.predictions)
    print(f"   Valid boards: {valid.tolist()}")
    print(f"   Row valid: {details['row_valid'].tolist()}")
    print(f"   Col valid: {details['col_valid'].tolist()}")
    print(f"   Box valid: {details['box_valid'].tolist()}")

    print("\n" + "=" * 60)
    print("9x9 Sudoku statechart test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sudoku_statechart_9x9()
