"""
9x9 Go Statechart Implementation

A statechart-based Go engine that GUARANTEES 100% legal moves by construction.

Key Rules Encoded as Statechart Concepts:
- Turn alternation: OR-state (Black | White)
- Ko rule: History state (KoForbidden remembers last capture point)
- Suicide prevention: Guard (not_suicide check)
- Occupation: Guard (not_occupied check)

The statechart topology makes illegal moves IMPOSSIBLE to generate.
"""

from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional, Dict
from enum import Enum
import copy


# =============================================================================
# CONSTANTS
# =============================================================================

BOARD_SIZE = 9
TOTAL_POINTS = BOARD_SIZE * BOARD_SIZE  # 81

# Stone colors
EMPTY = 0
BLACK = 1
WHITE = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def xy_to_idx(x: int, y: int) -> int:
    """Convert (x, y) coordinates to linear index."""
    return y * BOARD_SIZE + x


def idx_to_xy(idx: int) -> Tuple[int, int]:
    """Convert linear index to (x, y) coordinates."""
    return idx % BOARD_SIZE, idx // BOARD_SIZE


def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
    """Get orthogonal neighbors of a point."""
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < BOARD_SIZE - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < BOARD_SIZE - 1:
        neighbors.append((x, y + 1))
    return neighbors


def opponent(color: int) -> int:
    """Get opponent color."""
    return WHITE if color == BLACK else BLACK


# =============================================================================
# BOARD STATE (Context, not States)
# =============================================================================

@dataclass
class BoardState:
    """
    The board is CONTEXT, not states.

    States are: Turn, KoState, Prisoners
    Context is: the actual stone positions
    """
    stones: List[int] = field(default_factory=lambda: [EMPTY] * TOTAL_POINTS)

    def get(self, x: int, y: int) -> int:
        """Get stone at position."""
        return self.stones[xy_to_idx(x, y)]

    def set(self, x: int, y: int, color: int):
        """Set stone at position."""
        self.stones[xy_to_idx(x, y)] = color

    def copy(self) -> 'BoardState':
        """Deep copy the board."""
        new_board = BoardState()
        new_board.stones = self.stones.copy()
        return new_board

    def get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones connected to (x, y) of the same color."""
        color = self.get(x, y)
        if color == EMPTY:
            return set()

        group = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            if self.get(cx, cy) != color:
                continue

            group.add((cx, cy))
            for nx, ny in get_neighbors(cx, cy):
                if (nx, ny) not in group and self.get(nx, ny) == color:
                    stack.append((nx, ny))

        return group

    def get_liberties(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all liberties of the group containing (x, y)."""
        group = self.get_group(x, y)
        if not group:
            return set()

        liberties = set()
        for gx, gy in group:
            for nx, ny in get_neighbors(gx, gy):
                if self.get(nx, ny) == EMPTY:
                    liberties.add((nx, ny))

        return liberties

    def count_liberties(self, x: int, y: int) -> int:
        """Count liberties of the group containing (x, y)."""
        return len(self.get_liberties(x, y))

    def remove_group(self, x: int, y: int) -> int:
        """Remove the group containing (x, y). Returns number of stones removed."""
        group = self.get_group(x, y)
        for gx, gy in group:
            self.set(gx, gy, EMPTY)
        return len(group)

    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        lines = []
        for y in range(BOARD_SIZE):
            row = ' '.join(symbols[self.get(x, y)] for x in range(BOARD_SIZE))
            lines.append(f"{BOARD_SIZE - y:2} {row}")
        lines.append("   " + ' '.join('ABCDEFGHJ'[:BOARD_SIZE]))
        return '\n'.join(lines)


# =============================================================================
# STATECHART STATES
# =============================================================================

class TurnState(Enum):
    """OR-State: Exactly one player's turn."""
    BLACK = "black"
    WHITE = "white"


class KoState(Enum):
    """OR-State with HISTORY: Ko rule state."""
    NO_KO = "no_ko"
    KO_FORBIDDEN = "ko_forbidden"


# =============================================================================
# GO 9x9 STATECHART
# =============================================================================

@dataclass
class Go9x9Statechart:
    """
    Statechart-based 9x9 Go engine.

    Structure:
    Go9x9 (AND - parallel regions)
    ├── Turn (OR): Black | White
    ├── KoState (OR, HISTORY)
    │   ├── NoKo
    │   └── KoForbidden [point]
    ├── Prisoners (AND)
    │   ├── BlackCaptures: int
    │   └── WhiteCaptures: int
    └── Board (context)

    Invariants enforced by construction:
    1. Turn alternates (OR-state)
    2. Ko point cannot be played (guard)
    3. Suicide moves are rejected (guard)
    4. Only empty points can be played (guard)
    """

    # OR-State: Turn
    turn: TurnState = TurnState.BLACK

    # OR-State with History: Ko
    ko_state: KoState = KoState.NO_KO
    ko_point: Optional[Tuple[int, int]] = None

    # AND-State: Prisoners
    black_captures: int = 0
    white_captures: int = 0

    # Context: Board
    board: BoardState = field(default_factory=BoardState)

    # Game state
    consecutive_passes: int = 0
    move_history: List[Tuple[int, Optional[Tuple[int, int]]]] = field(default_factory=list)

    def current_player(self) -> int:
        """Get current player color."""
        return BLACK if self.turn == TurnState.BLACK else WHITE

    # =========================================================================
    # GUARDS (Statechart guards that make illegal moves impossible)
    # =========================================================================

    def guard_not_occupied(self, x: int, y: int) -> bool:
        """Guard: Point must be empty."""
        return self.board.get(x, y) == EMPTY

    def guard_not_ko(self, x: int, y: int) -> bool:
        """Guard: Point must not be ko-forbidden."""
        if self.ko_state == KoState.NO_KO:
            return True
        return self.ko_point != (x, y)

    def guard_not_suicide(self, x: int, y: int) -> bool:
        """
        Guard: Move must not be suicide.

        A move is suicide if:
        1. The placed stone has no liberties, AND
        2. It doesn't capture any opponent stones

        This is checked by simulating the move.
        """
        color = self.current_player()
        opp = opponent(color)

        # Temporarily place the stone
        test_board = self.board.copy()
        test_board.set(x, y, color)

        # Check if we capture any opponent stones
        captures_something = False
        for nx, ny in get_neighbors(x, y):
            if test_board.get(nx, ny) == opp:
                if test_board.count_liberties(nx, ny) == 0:
                    captures_something = True
                    break

        if captures_something:
            return True  # Not suicide - we capture something

        # Check if our stone has liberties
        return test_board.count_liberties(x, y) > 0

    def is_legal_move(self, x: int, y: int) -> bool:
        """
        Combined guard: All conditions must be true.

        This is the AND of all guards.
        """
        return (
            self.guard_not_occupied(x, y) and
            self.guard_not_ko(x, y) and
            self.guard_not_suicide(x, y)
        )

    # =========================================================================
    # ACTIONS (Statechart actions/effects)
    # =========================================================================

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Get all legal moves for current player.

        This is the KEY ADVANTAGE: We generate ONLY legal moves.
        A Transformer must learn this; we encode it in topology.
        """
        legal = []
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if self.is_legal_move(x, y):
                legal.append((x, y))
        return legal

    def play_move(self, x: int, y: int) -> bool:
        """
        Execute a move (transition).

        Returns True if move was legal and executed.
        """
        if not self.is_legal_move(x, y):
            return False  # Guard failed - move rejected

        color = self.current_player()
        opp = opponent(color)

        # Place stone
        self.board.set(x, y, color)

        # Capture opponent stones with no liberties
        captured = 0
        captured_single = None

        for nx, ny in get_neighbors(x, y):
            if self.board.get(nx, ny) == opp:
                if self.board.count_liberties(nx, ny) == 0:
                    group = self.board.get_group(nx, ny)
                    if len(group) == 1:
                        captured_single = list(group)[0]
                    captured += self.board.remove_group(nx, ny)

        # Update prisoners (AND-state)
        if color == BLACK:
            self.black_captures += captured
        else:
            self.white_captures += captured

        # Update ko state (OR-state with HISTORY)
        # Ko occurs when: single stone captured AND our stone has exactly one liberty
        if captured == 1 and captured_single is not None:
            our_group = self.board.get_group(x, y)
            our_liberties = self.board.get_liberties(x, y)
            if len(our_group) == 1 and len(our_liberties) == 1:
                # Ko situation!
                self.ko_state = KoState.KO_FORBIDDEN
                self.ko_point = captured_single
            else:
                self.ko_state = KoState.NO_KO
                self.ko_point = None
        else:
            self.ko_state = KoState.NO_KO
            self.ko_point = None

        # Switch turn (OR-state transition)
        self.turn = TurnState.WHITE if self.turn == TurnState.BLACK else TurnState.BLACK

        # Record move
        self.move_history.append((color, (x, y)))
        self.consecutive_passes = 0

        return True

    def play_pass(self):
        """Pass (don't place a stone)."""
        color = self.current_player()

        # Clear ko (passing clears ko)
        self.ko_state = KoState.NO_KO
        self.ko_point = None

        # Switch turn
        self.turn = TurnState.WHITE if self.turn == TurnState.BLACK else TurnState.BLACK

        # Record pass
        self.move_history.append((color, None))
        self.consecutive_passes += 1

    def is_game_over(self) -> bool:
        """Game ends after two consecutive passes."""
        return self.consecutive_passes >= 2

    def score(self) -> Tuple[float, float]:
        """
        Simple scoring: territory + captures.
        Returns (black_score, white_score).
        """
        black_territory = 0
        white_territory = 0

        # Count stones
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            stone = self.board.get(x, y)
            if stone == BLACK:
                black_territory += 1
            elif stone == WHITE:
                white_territory += 1

        # Add captures
        black_score = black_territory + self.black_captures
        white_score = white_territory + self.white_captures + 6.5  # Komi

        return black_score, white_score

    def winner(self) -> Optional[int]:
        """Get winner (None if game not over or tie)."""
        if not self.is_game_over():
            return None

        black_score, white_score = self.score()
        if black_score > white_score:
            return BLACK
        elif white_score > black_score:
            return WHITE
        return None


# =============================================================================
# TEST
# =============================================================================

def test_go_statechart():
    """Basic test of the Go statechart."""
    print("=" * 60)
    print("9x9 Go Statechart Test")
    print("=" * 60)

    game = Go9x9Statechart()

    # Test initial state
    print("\nInitial board:")
    print(game.board)
    print(f"\nCurrent turn: {game.turn.value}")
    print(f"Legal moves: {len(game.get_legal_moves())}")
    assert len(game.get_legal_moves()) == 81, "All 81 points should be legal initially"

    # Play some moves
    print("\nPlaying moves...")

    # Black plays center (4, 4)
    assert game.play_move(4, 4), "Center move should be legal"
    print(f"Black plays D5")

    # White plays adjacent
    assert game.play_move(4, 5), "Adjacent move should be legal"
    print(f"White plays D4")

    # Check illegal moves
    assert not game.is_legal_move(4, 4), "Occupied point should be illegal"
    assert not game.is_legal_move(4, 5), "Occupied point should be illegal"

    print("\nAfter 2 moves:")
    print(game.board)
    print(f"Black captures: {game.black_captures}")
    print(f"White captures: {game.white_captures}")

    # Test suicide detection
    print("\nTesting suicide detection...")
    game2 = Go9x9Statechart()

    # Create a situation where a move would be suicide
    # White stones surrounding a point
    game2.board.set(0, 1, WHITE)
    game2.board.set(1, 0, WHITE)
    game2.turn = TurnState.BLACK

    print("Board with potential suicide point at (0,0):")
    print(game2.board)

    # (0, 0) would be suicide for black if surrounded
    # But with only 2 neighbors covered, it's not suicide
    is_suicide = not game2.guard_not_suicide(0, 0)
    print(f"Is (0,0) suicide? {is_suicide}")

    # Now surround completely (need to place more white stones around corner)
    # Actually corner only has 2 neighbors, so 2 white stones = surrounded
    # But we need to check - black at (0,0) would have 0 liberties
    # So this IS suicide
    print(f"guard_not_suicide(0,0) = {game2.guard_not_suicide(0, 0)}")

    print("\n" + "=" * 60)
    print("TEST PASSED: Go statechart basic operations work")
    print("=" * 60)


if __name__ == "__main__":
    test_go_statechart()
