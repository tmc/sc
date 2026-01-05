"""
Go 9x9 Game Interface for AlphaZero

Implements the alpha-zero-general Game interface using the Go statechart.
This provides the bridge between AlphaZero and statechart-guaranteed moves.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple, Optional
import copy

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, BoardState, TurnState, KoState,
    BOARD_SIZE, TOTAL_POINTS, EMPTY, BLACK, WHITE,
    xy_to_idx, idx_to_xy
)


class Go9x9Game:
    """
    AlphaZero Game interface for 9x9 Go using statechart backend.

    Action space: 82 actions
      - 0-80: Board positions (y * 9 + x)
      - 81: Pass

    Board representation for neural network:
      - Shape: (9, 9, 3) where channels are (empty, black, white)
      - Or soft config: (327,) = turn(2) + ko(82) + board(81*3)
    """

    def __init__(self):
        self.n = BOARD_SIZE  # 9
        self.action_size = TOTAL_POINTS + 1  # 82 (81 positions + pass)

    def getInitBoard(self) -> Go9x9Statechart:
        """Return initial game state (statechart)."""
        return Go9x9Statechart()

    def getBoardSize(self) -> Tuple[int, int]:
        """Return board dimensions."""
        return (self.n, self.n)

    def getActionSize(self) -> int:
        """Return number of possible actions."""
        return self.action_size

    def getNextState(self, state: Go9x9Statechart, player: int,
                     action: int) -> Tuple[Go9x9Statechart, int]:
        """
        Execute action and return (next_state, next_player).

        Args:
            state: Current statechart state
            player: Current player (1 for Black, -1 for White)
            action: Action index (0-80 for positions, 81 for pass)

        Returns:
            (new_state, next_player)
        """
        new_state = copy.deepcopy(state)

        if action == 81:
            new_state.play_pass()
        else:
            x, y = idx_to_xy(action)
            success = new_state.play_move(x, y)
            if not success:
                # Should never happen if using getValidMoves
                raise ValueError(f"Illegal move: action={action}, pos=({x},{y})")

        next_player = -player
        return new_state, next_player

    def getValidMoves(self, state: Go9x9Statechart, player: int) -> np.ndarray:
        """
        Return binary vector of valid moves.

        This is where statechart guards guarantee 100% legal moves.
        """
        valid = np.zeros(self.action_size, dtype=np.float32)

        # Check each position via statechart guards
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if state.is_legal_move(x, y):
                valid[idx] = 1.0

        # Pass is always legal
        valid[81] = 1.0

        return valid

    def getGameEnded(self, state: Go9x9Statechart, player: int) -> float:
        """
        Return game result from perspective of player.

        Returns:
            0 if game not ended
            1 if player won
            -1 if player lost
            small value (1e-4) for draw
        """
        if not state.is_game_over():
            return 0

        winner = state.winner()
        if winner is None:
            return 1e-4  # Draw

        # Map winner to player perspective
        winner_player = 1 if winner == BLACK else -1
        return 1 if winner_player == player else -1

    def getCanonicalForm(self, state: Go9x9Statechart,
                         player: int) -> Go9x9Statechart:
        """
        Return state from perspective of current player.

        For Go, the canonical form swaps colors if player is White.
        """
        if player == 1:
            return state

        # Swap colors for White's perspective
        canonical = copy.deepcopy(state)

        # Swap board colors
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            stone = canonical.board.get(x, y)
            if stone == BLACK:
                canonical.board.set(x, y, WHITE)
            elif stone == WHITE:
                canonical.board.set(x, y, BLACK)

        # Swap turn
        canonical.turn = (TurnState.BLACK if canonical.turn == TurnState.WHITE
                         else TurnState.WHITE)

        # Swap captures
        canonical.black_captures, canonical.white_captures = (
            canonical.white_captures, canonical.black_captures
        )

        return canonical

    def getSymmetries(self, state: Go9x9Statechart,
                      pi: np.ndarray) -> List[Tuple[Go9x9Statechart, np.ndarray]]:
        """
        Return 8-fold symmetries (4 rotations x 2 reflections).

        Go has D4 symmetry: rotations by 90 degrees and reflections.
        """
        symmetries = []

        # Get board as numpy array
        board = self._state_to_board_array(state)
        pi_board = pi[:81].reshape(self.n, self.n)
        pi_pass = pi[81]

        for rot in range(4):  # 0, 90, 180, 270 degrees
            for flip in [False, True]:
                new_board = np.rot90(board, rot)
                new_pi_board = np.rot90(pi_board, rot)

                if flip:
                    new_board = np.fliplr(new_board)
                    new_pi_board = np.fliplr(new_pi_board)

                new_state = self._board_array_to_state(new_board, state)
                new_pi = np.concatenate([new_pi_board.ravel(), [pi_pass]])
                symmetries.append((new_state, new_pi))

        return symmetries

    def stringRepresentation(self, state: Go9x9Statechart) -> str:
        """Return unique string representation for MCTS dictionary keys."""
        # Include turn, ko state, and board
        parts = [
            'B' if state.turn == TurnState.BLACK else 'W',
            f"ko:{state.ko_point}" if state.ko_state == KoState.KO_FORBIDDEN else "noko",
            ''.join(str(s) for s in state.board.stones)
        ]
        return '|'.join(parts)

    def _state_to_board_array(self, state: Go9x9Statechart) -> np.ndarray:
        """Convert statechart to numpy board array."""
        board = np.zeros((self.n, self.n), dtype=np.int8)
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            board[y, x] = state.board.get(x, y)
        return board

    def _board_array_to_state(self, board: np.ndarray,
                               template: Go9x9Statechart) -> Go9x9Statechart:
        """Convert numpy board array back to statechart state."""
        new_state = copy.deepcopy(template)
        for y in range(self.n):
            for x in range(self.n):
                new_state.board.set(x, y, int(board[y, x]))
        return new_state

    def getBoardArray(self, state: Go9x9Statechart) -> np.ndarray:
        """Get board as (9, 9, 3) one-hot encoded array."""
        board = np.zeros((self.n, self.n, 3), dtype=np.float32)
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            stone = state.board.get(x, y)
            board[y, x, stone] = 1.0
        return board


def test_game_interface():
    """Test the game interface."""
    print("Testing Go9x9Game interface...")

    game = Go9x9Game()

    # Test initialization
    state = game.getInitBoard()
    print(f"Board size: {game.getBoardSize()}")
    print(f"Action size: {game.getActionSize()}")

    # Test valid moves
    valid = game.getValidMoves(state, 1)
    print(f"Initial valid moves: {valid.sum()} (should be 82)")
    assert valid.sum() == 82, "All positions + pass should be valid initially"

    # Test state transitions
    state, player = game.getNextState(state, 1, 40)  # Play center
    print(f"After Black plays center, next player: {player}")

    valid = game.getValidMoves(state, player)
    print(f"Valid moves after one play: {valid.sum()} (should be 81)")
    assert valid.sum() == 81, "Center should be occupied"

    # Test canonical form
    canonical = game.getCanonicalForm(state, -1)
    print(f"Canonical turn: {canonical.turn}")

    # Test symmetries
    pi = np.ones(82) / 82
    syms = game.getSymmetries(state, pi)
    print(f"Symmetries generated: {len(syms)} (should be 8)")
    assert len(syms) == 8

    # Test string representation
    s = game.stringRepresentation(state)
    print(f"State string length: {len(s)}")

    print("All tests passed!")


if __name__ == "__main__":
    test_game_interface()
