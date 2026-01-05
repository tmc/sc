"""
Experiment H Sketch: TicTacToe with DifferentiableTransitionSelector

This demonstrates how the TransitionSelector models game state transitions.
"""

import mlx.core as mx
import mlx.nn as nn
from exp_c_transitions import DifferentiableTransitionSelector, StatechartMachine


# TicTacToe State Design:
#
# Game States (4): PLAYING, X_WINS, O_WINS, DRAW
#
# Transitions (11):
#   - PLACE_X_i (i=0..8): PLAYING -> PLAYING (if cell empty, X turn)
#   - PLACE_O_i (i=0..8): PLAYING -> PLAYING (if cell empty, O turn)
#   - X_WINS: PLAYING -> X_WINS (when X has 3 in row)
#   - O_WINS: PLAYING -> O_WINS (when O has 3 in row)
#   - DRAW: PLAYING -> DRAW (when board full, no winner)
#
# Simplified for demo: 4 game states, transitions check win conditions


class TicTacToeBoard(nn.Module):
    """Differentiable TicTacToe board representation."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        # Board: 9 cells, each can be EMPTY(0), X(1), O(2)
        self.cell_embed = nn.Embedding(3, embed_dim)
        self.board_proj = nn.Linear(9 * embed_dim, embed_dim)

    def encode(self, board: mx.array) -> mx.array:
        """Encode board state.

        Args:
            board: [B, 9] cell values (0=empty, 1=X, 2=O)

        Returns:
            [B, embed_dim] board embedding
        """
        cell_embs = self.cell_embed(board)  # [B, 9, embed_dim]
        flat = cell_embs.reshape(board.shape[0], -1)  # [B, 9*embed_dim]
        return self.board_proj(flat)  # [B, embed_dim]


class TicTacToeStatechart(nn.Module):
    """TicTacToe as a differentiable statechart."""

    # Game phase states
    PLAYING = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3
    NUM_STATES = 4

    # Win patterns (indices of 3-in-a-row)
    WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6],              # diagonals
    ]

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Board encoder
        self.board_encoder = TicTacToeBoard(embed_dim)

        # Transitions:
        # 0-8: PLACE moves (PLAYING -> PLAYING)
        # 9: X_WINS transition
        # 10: O_WINS transition
        # 11: DRAW transition
        NUM_TRANSITIONS = 12

        transitions = (
            # Place moves: PLAYING -> PLAYING
            [(self.PLAYING, self.PLAYING, i) for i in range(9)] +
            # Terminal transitions
            [(self.PLAYING, self.X_WINS, 9),
             (self.PLAYING, self.O_WINS, 10),
             (self.PLAYING, self.DRAW, 11)]
        )

        self.selector = DifferentiableTransitionSelector(
            num_states=self.NUM_STATES,
            num_transitions=NUM_TRANSITIONS,
            embed_dim=embed_dim,
            num_events=12  # 9 place + 3 terminal
        )
        self.selector.configure_transitions(
            source_states=[t[0] for t in transitions],
            target_states=[t[1] for t in transitions],
            event_ids=[t[2] for t in transitions]
        )

    def check_winner(self, board: mx.array) -> mx.array:
        """Check for winner (differentiable approximation).

        Args:
            board: [B, 9] cell values

        Returns:
            [B, 3] soft indicators for (x_wins, o_wins, draw)
        """
        B = board.shape[0]

        # Check each win pattern
        x_wins_any = mx.zeros((B,))
        o_wins_any = mx.zeros((B,))

        for pattern in self.WIN_PATTERNS:
            cells = board[:, pattern]  # [B, 3]
            # X wins if all 3 cells are 1
            x_match = mx.prod(mx.where(cells == 1, 1.0, 0.0), axis=1)
            # O wins if all 3 cells are 2
            o_match = mx.prod(mx.where(cells == 2, 1.0, 0.0), axis=1)
            x_wins_any = mx.maximum(x_wins_any, x_match)
            o_wins_any = mx.maximum(o_wins_any, o_match)

        # Draw if board full and no winner
        board_full = mx.prod(mx.where(board > 0, 1.0, 0.0), axis=1)
        is_draw = board_full * (1 - x_wins_any) * (1 - o_wins_any)

        return mx.stack([x_wins_any, o_wins_any, is_draw], axis=1)

    def step(self, game_state: mx.array, board: mx.array) -> tuple:
        """Take one game step.

        Args:
            game_state: [B, 4] soft game phase
            board: [B, 9] current board

        Returns:
            (new_game_state, enablement, selection)
        """
        # Encode board as context for guards
        context = self.board_encoder.encode(board)

        # Use transition selector
        new_state, enablement, selection = self.selector(game_state, context)

        return new_state, enablement, selection


def demo():
    """Quick demonstration."""
    print("TicTacToe Statechart Demo")
    print("=" * 40)

    game = TicTacToeStatechart(embed_dim=16)

    # Initial state: PLAYING with empty board
    game_state = mx.array([[1.0, 0.0, 0.0, 0.0]])  # PLAYING
    board = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])  # empty

    print(f"Initial game state: PLAYING")
    print(f"Board: {board[0].tolist()}")

    # Take a step
    new_state, enablement, selection = game.step(game_state, board)

    print(f"\nAfter step:")
    print(f"  Game state: {[f'{x:.3f}' for x in new_state[0].tolist()]}")
    print(f"  (PLAYING, X_WINS, O_WINS, DRAW)")
    print(f"  Enablement (first 9 = place moves): {[f'{x:.2f}' for x in enablement[0, :9].tolist()]}")

    # Test gradient flow through game state (not board embedding)
    def loss_fn(game_st):
        ctx = game.board_encoder.encode(board)  # Fixed board
        new_state, _, _ = game.selector(game_st, ctx)
        return mx.mean(new_state[:, 1])  # X_WINS probability

    loss, grad = mx.value_and_grad(loss_fn)(game_state)
    grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))

    print(f"\nGradient check:")
    print(f"  Loss (X_WINS prob): {float(loss):.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  ✓ Gradients flow: {grad_norm > 0}")


if __name__ == "__main__":
    demo()
