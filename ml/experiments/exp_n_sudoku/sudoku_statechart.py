"""
Sudoku as Statechart

Models 4x4 Sudoku (simpler than 9x9) as a differentiable statechart.

Structure:
- 16 cells in a 4x4 grid
- Each cell has 5 states: empty (0), 1, 2, 3, 4
- Guards: row_valid, col_valid, box_valid (2x2 boxes)
- Transitions: PLACE_1..PLACE_4 per cell

Statechart interpretation:
- Board state = parallel composition of 16 cell states
- Placing a digit = transition in one cell's statechart
- Constraint satisfaction = guard evaluation

Inspired by TinyRecursiveModels Sudoku benchmark.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional


class SudokuCell(nn.Module):
    """
    Single cell in Sudoku grid.
    
    States: 0 (empty), 1, 2, 3, 4
    Transitions: PLACE_1, PLACE_2, PLACE_3, PLACE_4
    """
    
    NUM_VALUES = 5  # empty + 4 digits
    
    def __init__(self, cell_id: int, hidden_dim: int = 16):
        super().__init__()
        self.cell_id = cell_id
        self.row = cell_id // 4
        self.col = cell_id % 4
        self.box = (self.row // 2) * 2 + (self.col // 2)  # 0, 1, 2, 3
        
        # State embedding
        self.state_embed = nn.Embedding(self.NUM_VALUES, hidden_dim)
        
        # Transition scores: given context, score each possible placement
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),  # Score for placing 1, 2, 3, 4
        )
    
    def get_state_probs(self, state_logits: mx.array) -> mx.array:
        """Convert logits to soft state distribution."""
        return mx.softmax(state_logits, axis=-1)
    
    def forward_transition(
        self,
        current_state: mx.array,  # [B, NUM_VALUES] soft state
        context: mx.array,        # [B, hidden_dim] from neighbors
        guard_values: mx.array,   # [B, 4] guard satisfaction for each digit
    ) -> mx.array:
        """
        Compute new state after considering transitions.
        
        Args:
            current_state: [B, 5] soft state (empty, 1, 2, 3, 4)
            context: [B, hidden_dim] context from board
            guard_values: [B, 4] constraint satisfaction for placing 1, 2, 3, 4
        
        Returns:
            new_state: [B, 5] updated soft state
        """
        B = current_state.shape[0]
        
        # Transition scores
        trans_scores = self.transition_net(context)  # [B, 4]
        
        # Apply guards (only allow valid placements)
        masked_scores = trans_scores * guard_values  # [B, 4]
        
        # Probability of transitioning (vs staying)
        trans_probs = mx.softmax(masked_scores, axis=-1)  # [B, 4]
        
        # Current probability of being empty
        p_empty = current_state[:, 0:1]  # [B, 1]
        
        # New state: redistribute empty probability to digits
        # For each digit d: p(d) = old_p(d) + p_empty * trans_prob(d) * guard(d)
        new_digit_probs = current_state[:, 1:] + p_empty * trans_probs * guard_values
        
        # New empty probability (decreases if we place)
        total_transition = mx.sum(trans_probs * guard_values, axis=-1, keepdims=True)
        new_empty = p_empty * (1 - total_transition * 0.5)  # Soft decrease
        
        new_state = mx.concatenate([new_empty, new_digit_probs], axis=-1)
        
        # Normalize
        return new_state / mx.sum(new_state, axis=-1, keepdims=True)


class SudokuConstraintGuard(nn.Module):
    """
    Evaluates Sudoku constraints as soft guards.
    
    For placing digit d in cell (r, c):
    - row_valid: d not already in row r
    - col_valid: d not already in col c
    - box_valid: d not already in 2x2 box
    """
    
    def __init__(self, grid_size: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.box_size = 2  # 2x2 boxes for 4x4 grid
        
        # Learnable constraint networks (can learn Sudoku rules)
        self.row_net = nn.Linear(grid_size * 5, 4)  # 4 cells per row, 5 states each
        self.col_net = nn.Linear(grid_size * 5, 4)
        self.box_net = nn.Linear(4 * 5, 4)  # 4 cells per 2x2 box
    
    def get_row_cells(self, cell_id: int) -> List[int]:
        """Get cell IDs in same row."""
        row = cell_id // self.grid_size
        return [row * self.grid_size + c for c in range(self.grid_size)]
    
    def get_col_cells(self, cell_id: int) -> List[int]:
        """Get cell IDs in same column."""
        col = cell_id % self.grid_size
        return [r * self.grid_size + col for r in range(self.grid_size)]
    
    def get_box_cells(self, cell_id: int) -> List[int]:
        """Get cell IDs in same 2x2 box."""
        row, col = cell_id // self.grid_size, cell_id % self.grid_size
        box_row, box_col = (row // 2) * 2, (col // 2) * 2
        cells = []
        for r in range(box_row, box_row + 2):
            for c in range(box_col, box_col + 2):
                cells.append(r * self.grid_size + c)
        return cells
    
    def __call__(
        self,
        board_states: mx.array,  # [B, 16, 5] soft states for all cells
        cell_id: int,
    ) -> mx.array:
        """
        Compute guard values for placing each digit in cell.
        
        Args:
            board_states: [B, 16, 5] soft state for each cell
            cell_id: Which cell to check
        
        Returns:
            guards: [B, 4] satisfaction for placing 1, 2, 3, 4
        """
        B = board_states.shape[0]
        
        # Get states of cells in same row/col/box
        row_cells = self.get_row_cells(cell_id)
        col_cells = self.get_col_cells(cell_id)
        box_cells = self.get_box_cells(cell_id)
        
        # Flatten states for each constraint group
        row_states = board_states[:, row_cells, :].reshape(B, -1)  # [B, 20]
        col_states = board_states[:, col_cells, :].reshape(B, -1)  # [B, 20]
        box_states = board_states[:, box_cells, :].reshape(B, -1)  # [B, 20]
        
        # Compute constraint satisfaction
        row_guard = mx.sigmoid(self.row_net(row_states))  # [B, 4]
        col_guard = mx.sigmoid(self.col_net(col_states))  # [B, 4]
        box_guard = mx.sigmoid(self.box_net(box_states))  # [B, 4]
        
        # Combined guard: all constraints must be satisfied
        combined = row_guard * col_guard * box_guard
        
        return combined


class SudokuStatechart(nn.Module):
    """
    Full 4x4 Sudoku as differentiable statechart.
    
    Parallel composition of 16 cell statecharts with constraint guards.
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.grid_size = 4
        self.num_cells = 16
        self.hidden_dim = hidden_dim
        
        # Cell statecharts
        self.cells = [SudokuCell(i, hidden_dim) for i in range(self.num_cells)]
        
        # Constraint guard
        self.guard = SudokuConstraintGuard(self.grid_size, hidden_dim)
        
        # Board encoder: convert board to context
        self.board_encoder = nn.Sequential(
            nn.Linear(self.num_cells * 5, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Cell position embedding
        self.pos_embed = nn.Embedding(self.num_cells, hidden_dim)
    
    def encode_board(self, board_states: mx.array) -> mx.array:
        """
        Encode board state to context vector.
        
        Args:
            board_states: [B, 16, 5] soft states
        
        Returns:
            context: [B, hidden_dim]
        """
        B = board_states.shape[0]
        flat = board_states.reshape(B, -1)  # [B, 80]
        return self.board_encoder(flat)
    
    def step(self, board_states: mx.array) -> mx.array:
        """
        One step of statechart transitions.
        
        Each cell considers placing each digit based on constraints.
        
        Args:
            board_states: [B, 16, 5] current soft states
        
        Returns:
            new_states: [B, 16, 5] updated soft states
        """
        B = board_states.shape[0]
        
        # Encode board context
        context = self.encode_board(board_states)  # [B, hidden_dim]
        
        new_states = []
        for cell_id, cell in enumerate(self.cells):
            # Get position-specific context
            pos_emb = self.pos_embed(mx.array([cell_id]))  # [1, hidden_dim]
            pos_emb = mx.broadcast_to(pos_emb, (B, self.hidden_dim))
            cell_context = context + pos_emb
            
            # Get guard values for this cell
            guards = self.guard(board_states, cell_id)  # [B, 4]
            
            # Current cell state
            current = board_states[:, cell_id, :]  # [B, 5]
            
            # Compute transition
            new = cell.forward_transition(current, cell_context, guards)
            new_states.append(new)
        
        return mx.stack(new_states, axis=1)  # [B, 16, 5]
    
    def solve(self, initial_board: mx.array, num_steps: int = 10) -> mx.array:
        """
        Attempt to solve Sudoku via iterative transitions.
        
        Args:
            initial_board: [B, 16, 5] initial soft states
            num_steps: Number of transition steps
        
        Returns:
            final_board: [B, 16, 5] final soft states
        """
        board = initial_board
        for _ in range(num_steps):
            board = self.step(board)
        return board
    
    def board_to_soft(self, board: mx.array) -> mx.array:
        """
        Convert discrete board to soft states.
        
        Args:
            board: [B, 16] int values 0-4
        
        Returns:
            soft: [B, 16, 5] one-hot encoded
        """
        return mx.eye(5)[board]  # [B, 16, 5]
    
    def soft_to_board(self, soft: mx.array) -> mx.array:
        """
        Convert soft states to discrete board.
        
        Args:
            soft: [B, 16, 5]
        
        Returns:
            board: [B, 16] argmax values
        """
        return mx.argmax(soft, axis=-1)
    
    def is_valid(self, board: mx.array) -> mx.array:
        """
        Check if board satisfies Sudoku constraints.
        
        Args:
            board: [B, 16] discrete values 0-4
        
        Returns:
            valid: [B] boolean per board
        """
        B = board.shape[0]
        valid = mx.ones((B,), dtype=mx.bool_)
        
        for row in range(4):
            row_vals = board[:, row*4:(row+1)*4]  # [B, 4]
            # Check no duplicates (excluding 0)
            for d in range(1, 5):
                count = mx.sum(row_vals == d, axis=-1)
                valid = valid & (count <= 1)
        
        for col in range(4):
            col_vals = board[:, col::4]  # [B, 4]
            for d in range(1, 5):
                count = mx.sum(col_vals == d, axis=-1)
                valid = valid & (count <= 1)
        
        # Check 2x2 boxes
        for box_row in range(2):
            for box_col in range(2):
                box_start = box_row * 8 + box_col * 2
                box_vals = mx.concatenate([
                    board[:, box_start:box_start+2],
                    board[:, box_start+4:box_start+6]
                ], axis=-1)  # [B, 4]
                for d in range(1, 5):
                    count = mx.sum(box_vals == d, axis=-1)
                    valid = valid & (count <= 1)
        
        return valid


def generate_sudoku_4x4(num_puzzles: int = 10, num_clues: int = 6) -> Tuple[mx.array, mx.array]:
    """
    Generate simple 4x4 Sudoku puzzles.

    Returns:
        puzzles: [num_puzzles, 16] with some cells empty (0)
        solutions: [num_puzzles, 16] full solutions
    """
    # Simple valid 4x4 solution
    base_solution = [
        1, 2, 3, 4,
        3, 4, 1, 2,
        2, 1, 4, 3,
        4, 3, 2, 1,
    ]

    puzzles = []
    solutions = []

    for _ in range(num_puzzles):
        # Permute digits (using Python list)
        perm_arr = mx.random.permutation(mx.array([0, 1, 2, 3]))
        perm = [int(perm_arr[i]) + 1 for i in range(4)]  # 1-4 shuffled

        # Apply permutation to base solution
        sol = [perm[base_solution[i] - 1] for i in range(16)]

        # Create puzzle by removing some cells
        remove_arr = mx.random.permutation(mx.arange(16))
        remove_indices = set(int(remove_arr[i]) for i in range(16 - num_clues))
        puzzle = [0 if i in remove_indices else sol[i] for i in range(16)]

        puzzles.append(puzzle)
        solutions.append(sol)

    return mx.array(puzzles, dtype=mx.int32), mx.array(solutions, dtype=mx.int32)


def test_sudoku_statechart():
    """Test the Sudoku statechart."""
    print("=" * 60)
    print("Testing Sudoku Statechart (4x4)")
    print("=" * 60)
    
    # Create model
    model = SudokuStatechart(hidden_dim=32)
    
    # Generate test puzzles
    mx.random.seed(42)
    puzzles, solutions = generate_sudoku_4x4(num_puzzles=2, num_clues=8)
    
    print("\n1. Generated puzzles:")
    for i in range(2):
        puzzle = puzzles[i].reshape(4, 4)
        print(f"   Puzzle {i}: {puzzle.tolist()}")
    
    # Convert to soft states
    soft_puzzles = model.board_to_soft(puzzles)
    print(f"\n2. Soft state shape: {soft_puzzles.shape}")
    
    # Run one step
    print("\n3. Running statechart step...")
    new_states = model.step(soft_puzzles)
    print(f"   Output shape: {new_states.shape}")
    
    # Check state sums (should be 1)
    sums = mx.sum(new_states, axis=-1)
    print(f"   State sums (should be 1): min={float(mx.min(sums)):.3f}, max={float(mx.max(sums)):.3f}")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    
    def loss_fn(model, soft_board, target):
        output = model.step(soft_board)
        target_soft = model.board_to_soft(target)
        return mx.mean((output - target_soft) ** 2)
    
    loss, grads = nn.value_and_grad(model, loss_fn)(model, soft_puzzles, solutions)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Loss: {float(loss):.4f}")
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # Test solving
    print("\n5. Testing solve (10 steps)...")
    solved = model.solve(soft_puzzles, num_steps=10)
    solved_discrete = model.soft_to_board(solved)
    
    print(f"   Solved boards:")
    for i in range(2):
        board = solved_discrete[i].reshape(4, 4)
        print(f"   Board {i}: {board.tolist()}")
    
    # Validate
    print("\n6. Checking validity...")
    valid = model.is_valid(solved_discrete)
    print(f"   Valid boards: {valid.tolist()}")
    
    print("\n" + "=" * 60)
    print("Sudoku statechart test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sudoku_statechart()
