"""
Game Environment Abstraction for Topology Evolution

Provides a unified interface for different games (TicTacToe, Connect4, Othello, etc.)
so that topology evolution can work across all of them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, ClassVar
import random
import copy


# =============================================================================
# Game Configuration
# =============================================================================

@dataclass
class GameConfig:
    """Configuration for a game environment."""
    name: str
    n_positions: int      # Board size (9 for TicTacToe, 42 for Connect4)
    n_players: int        # Usually 2
    max_game_length: int  # Typical maximum game length

    # For guard evolution
    base_features: List[str]  # Feature names available for guards


# =============================================================================
# Abstract Game Environment
# =============================================================================

class GameEnvironment(ABC):
    """
    Abstract base class for game environments.

    Implementations must provide:
    - State representation
    - Legal move detection
    - Game progression
    - Feature extraction for guards
    """

    config: ClassVar[GameConfig]

    @abstractmethod
    def reset(self) -> 'GameEnvironment':
        """Reset to initial state. Returns self for chaining."""
        pass

    @abstractmethod
    def copy(self) -> 'GameEnvironment':
        """Create a deep copy of the current state."""
        pass

    @abstractmethod
    def get_state(self) -> Tuple[int, ...]:
        """Get current board state as tuple."""
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices."""
        pass

    @abstractmethod
    def is_legal(self, move: int) -> bool:
        """Check if a specific move is legal."""
        pass

    @abstractmethod
    def make_move(self, move: int) -> bool:
        """Make a move. Returns True if successful."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game is over."""
        pass

    @abstractmethod
    def get_winner(self) -> int:
        """Get winner: 1=player1, 2=player2, 0=none/draw."""
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Get current player (1 or 2)."""
        pass

    @classmethod
    @abstractmethod
    def generate_random_position(cls, max_moves: Optional[int] = None) -> 'GameEnvironment':
        """Generate a random non-terminal position."""
        pass

    @abstractmethod
    def get_features(self) -> Dict[str, float]:
        """
        Extract features for guard evaluation.

        Returns dict mapping feature names to values.
        Features should be normalized to [0, 1] range where possible.
        """
        pass

    def get_legal_mask(self) -> List[bool]:
        """Get boolean mask of legal moves."""
        return [self.is_legal(i) for i in range(self.config.n_positions)]


# =============================================================================
# TicTacToe Implementation
# =============================================================================

class TicTacToeEnv(GameEnvironment):
    """TicTacToe game environment."""

    config = GameConfig(
        name="TicTacToe",
        n_positions=9,
        n_players=2,
        max_game_length=9,
        base_features=[
            "cell_empty",      # Is target cell empty?
            "move_count",      # How many moves played?
            "is_corner",       # Is move a corner?
            "is_center",       # Is move the center?
            "is_edge",         # Is move an edge?
            "player_pieces",   # Current player's piece count
            "opponent_pieces", # Opponent's piece count
        ]
    )

    # Win lines (rows, cols, diagonals)
    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]

    CORNERS = {0, 2, 6, 8}
    CENTER = {4}
    EDGES = {1, 3, 5, 7}

    def __init__(self):
        self.board = [0] * 9  # 0=empty, 1=X, 2=O
        self.current_player = 1  # X starts

    def reset(self) -> 'TicTacToeEnv':
        self.board = [0] * 9
        self.current_player = 1
        return self

    def copy(self) -> 'TicTacToeEnv':
        new = TicTacToeEnv()
        new.board = self.board.copy()
        new.current_player = self.current_player
        return new

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.board)

    def get_legal_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]

    def is_legal(self, move: int) -> bool:
        return 0 <= move < 9 and self.board[move] == 0

    def make_move(self, move: int) -> bool:
        if not self.is_legal(move):
            return False
        self.board[move] = self.current_player
        self.current_player = 3 - self.current_player  # Switch 1<->2
        return True

    def is_terminal(self) -> bool:
        return self.get_winner() != 0 or len(self.get_legal_moves()) == 0

    def get_winner(self) -> int:
        for line in self.WIN_LINES:
            if self.board[line[0]] != 0:
                if self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
                    return self.board[line[0]]
        return 0

    def get_current_player(self) -> int:
        return self.current_player

    @classmethod
    def generate_random_position(cls, max_moves: Optional[int] = None) -> 'TicTacToeEnv':
        game = cls()
        n_moves = random.randint(0, max_moves or 6)
        for _ in range(n_moves):
            legal = game.get_legal_moves()
            if not legal or game.is_terminal():
                break
            game.make_move(random.choice(legal))
        return game

    def get_features(self) -> Dict[str, float]:
        """Extract features for guard evaluation."""
        move_count = sum(1 for c in self.board if c != 0)
        player_pieces = sum(1 for c in self.board if c == self.current_player)
        opponent_pieces = sum(1 for c in self.board if c == 3 - self.current_player)

        return {
            "move_count": move_count / 9.0,
            "player_pieces": player_pieces / 5.0,
            "opponent_pieces": opponent_pieces / 5.0,
        }

    def get_move_features(self, move: int) -> Dict[str, float]:
        """Get features specific to a move position."""
        return {
            "cell_empty": 1.0 if self.board[move] == 0 else 0.0,
            "is_corner": 1.0 if move in self.CORNERS else 0.0,
            "is_center": 1.0 if move in self.CENTER else 0.0,
            "is_edge": 1.0 if move in self.EDGES else 0.0,
        }


# =============================================================================
# Connect4 Implementation
# =============================================================================

class Connect4Env(GameEnvironment):
    """Connect4 game environment."""

    config = GameConfig(
        name="Connect4",
        n_positions=7,  # 7 columns (moves are column indices)
        n_players=2,
        max_game_length=42,
        base_features=[
            "column_height",   # Height of column (0-6)
            "column_playable", # Is column not full?
            "move_count",      # Total moves played
            "center_control",  # Pieces in center columns
            "threats",         # Number of 3-in-a-rows
        ]
    )

    ROWS = 6
    COLS = 7

    def __init__(self):
        # Board: 6 rows x 7 cols, 0=empty, 1=red, 2=yellow
        self.board = [[0] * self.COLS for _ in range(self.ROWS)]
        self.current_player = 1
        self.column_heights = [0] * self.COLS  # Track column fill levels

    def reset(self) -> 'Connect4Env':
        self.board = [[0] * self.COLS for _ in range(self.ROWS)]
        self.current_player = 1
        self.column_heights = [0] * self.COLS
        return self

    def copy(self) -> 'Connect4Env':
        new = Connect4Env()
        new.board = [row.copy() for row in self.board]
        new.current_player = self.current_player
        new.column_heights = self.column_heights.copy()
        return new

    def get_state(self) -> Tuple[int, ...]:
        # Flatten board to tuple
        return tuple(cell for row in self.board for cell in row)

    def get_legal_moves(self) -> List[int]:
        return [col for col in range(self.COLS) if self.column_heights[col] < self.ROWS]

    def is_legal(self, move: int) -> bool:
        return 0 <= move < self.COLS and self.column_heights[move] < self.ROWS

    def make_move(self, col: int) -> bool:
        if not self.is_legal(col):
            return False
        row = self.column_heights[col]
        self.board[row][col] = self.current_player
        self.column_heights[col] += 1
        self.current_player = 3 - self.current_player
        return True

    def is_terminal(self) -> bool:
        return self.get_winner() != 0 or len(self.get_legal_moves()) == 0

    def get_winner(self) -> int:
        # Check all possible 4-in-a-rows
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.board[row][col] == 0:
                    continue
                player = self.board[row][col]

                for dr, dc in directions:
                    count = 1
                    for i in range(1, 4):
                        r, c = row + dr * i, col + dc * i
                        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                            if self.board[r][c] == player:
                                count += 1
                            else:
                                break
                        else:
                            break
                    if count >= 4:
                        return player
        return 0

    def get_current_player(self) -> int:
        return self.current_player

    @classmethod
    def generate_random_position(cls, max_moves: Optional[int] = None) -> 'Connect4Env':
        game = cls()
        n_moves = random.randint(0, max_moves or 20)
        for _ in range(n_moves):
            legal = game.get_legal_moves()
            if not legal or game.is_terminal():
                break
            game.make_move(random.choice(legal))
        return game

    def get_features(self) -> Dict[str, float]:
        """Extract features for guard evaluation."""
        move_count = sum(self.column_heights)
        center_control = sum(
            1 for row in range(self.ROWS)
            for col in [2, 3, 4]
            if self.board[row][col] == self.current_player
        )

        return {
            "move_count": move_count / 42.0,
            "center_control": center_control / 18.0,
        }

    def get_move_features(self, col: int) -> Dict[str, float]:
        """Get features specific to a column."""
        return {
            "column_playable": 1.0 if self.is_legal(col) else 0.0,
            "column_height": self.column_heights[col] / 6.0,
            "is_center": 1.0 if col in [2, 3, 4] else 0.0,
        }


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_legal_move_dataset(
    env_class: type,
    n_positions: int = 1000,
    max_moves: Optional[int] = None
) -> List[Tuple[Tuple[int, ...], List[bool]]]:
    """
    Generate dataset of (position, legal_moves) pairs.

    Args:
        env_class: GameEnvironment class to use
        n_positions: Number of positions to generate
        max_moves: Max moves for random position generation

    Returns:
        List of (board_state, legal_mask) tuples
    """
    dataset = []

    for _ in range(n_positions):
        game = env_class.generate_random_position(max_moves)
        if game.is_terminal():
            continue

        state = game.get_state()
        legal_mask = game.get_legal_mask()
        dataset.append((state, legal_mask))

    return dataset


# =============================================================================
# Othello Implementation
# =============================================================================

class OthelloEnv(GameEnvironment):
    """Othello (Reversi) game environment."""

    config = GameConfig(
        name="Othello",
        n_positions=64,  # 8x8 board
        n_players=2,
        max_game_length=60,  # Typically 60 moves max
        base_features=[
            "cell_empty",      # Is target cell empty?
            "is_corner",       # Is move a corner?
            "is_edge",         # Is move on edge?
            "is_x_square",     # Is move an X-square (diagonal to corner)?
            "is_c_square",     # Is move a C-square (adjacent to corner)?
            "move_count",      # How many moves played?
            "player_pieces",   # Current player's piece count
            "opponent_pieces", # Opponent's piece count
            "mobility",        # Number of legal moves
            "flip_count",      # How many pieces this move flips
        ]
    )

    SIZE = 8

    # Strategic positions
    CORNERS = {0, 7, 56, 63}
    X_SQUARES = {9, 14, 49, 54}  # Diagonal to corners
    C_SQUARES = {1, 6, 8, 15, 48, 55, 57, 62}  # Adjacent to corners

    # 8 directions: (row_delta, col_delta)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def __init__(self):
        # Board: 0=empty, 1=black, 2=white
        self.board = [0] * 64
        # Initial position: center 4 squares
        self.board[27] = 2  # d4 = white
        self.board[28] = 1  # e4 = black
        self.board[35] = 1  # d5 = black
        self.board[36] = 2  # e5 = white
        self.current_player = 1  # Black starts
        self._legal_moves_cache = None
        self._pass_count = 0  # Track consecutive passes

    def reset(self) -> 'OthelloEnv':
        self.board = [0] * 64
        self.board[27] = 2
        self.board[28] = 1
        self.board[35] = 1
        self.board[36] = 2
        self.current_player = 1
        self._legal_moves_cache = None
        self._pass_count = 0
        return self

    def copy(self) -> 'OthelloEnv':
        new = OthelloEnv.__new__(OthelloEnv)
        new.board = self.board.copy()
        new.current_player = self.current_player
        new._legal_moves_cache = None
        new._pass_count = self._pass_count
        return new

    def _idx_to_rc(self, idx: int) -> Tuple[int, int]:
        """Convert board index to (row, col)."""
        return idx // self.SIZE, idx % self.SIZE

    def _rc_to_idx(self, row: int, col: int) -> int:
        """Convert (row, col) to board index."""
        return row * self.SIZE + col

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if position is on the board."""
        return 0 <= row < self.SIZE and 0 <= col < self.SIZE

    def _get_flips(self, move: int, player: int) -> List[int]:
        """
        Get list of positions that would be flipped by this move.

        Returns empty list if move is not legal.
        """
        if self.board[move] != 0:
            return []

        row, col = self._idx_to_rc(move)
        opponent = 3 - player
        all_flips = []

        for dr, dc in self.DIRECTIONS:
            flips = []
            r, c = row + dr, col + dc

            # Move along direction while finding opponent pieces
            while self._in_bounds(r, c) and self.board[self._rc_to_idx(r, c)] == opponent:
                flips.append(self._rc_to_idx(r, c))
                r += dr
                c += dc

            # Check if we hit our own piece (completing the sandwich)
            if flips and self._in_bounds(r, c) and self.board[self._rc_to_idx(r, c)] == player:
                all_flips.extend(flips)

        return all_flips

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.board)

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices."""
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        legal = []
        for idx in range(64):
            if self._get_flips(idx, self.current_player):
                legal.append(idx)

        self._legal_moves_cache = legal
        return legal

    def is_legal(self, move: int) -> bool:
        """Check if a specific move is legal."""
        if not (0 <= move < 64):
            return False
        return bool(self._get_flips(move, self.current_player))

    def make_move(self, move: int) -> bool:
        """Make a move. Returns True if successful."""
        flips = self._get_flips(move, self.current_player)
        if not flips:
            return False

        # Place piece
        self.board[move] = self.current_player

        # Flip captured pieces
        for idx in flips:
            self.board[idx] = self.current_player

        # Switch player
        self.current_player = 3 - self.current_player
        self._legal_moves_cache = None
        self._pass_count = 0

        # Check if next player must pass
        if not self.get_legal_moves():
            self.current_player = 3 - self.current_player
            self._legal_moves_cache = None
            self._pass_count += 1

        return True

    def is_terminal(self) -> bool:
        """Check if game is over (no legal moves for either player)."""
        if self._pass_count >= 2:
            return True

        # Check if current player has moves
        if self.get_legal_moves():
            return False

        # Check if opponent has moves
        self.current_player = 3 - self.current_player
        self._legal_moves_cache = None
        has_moves = bool(self.get_legal_moves())
        self.current_player = 3 - self.current_player
        self._legal_moves_cache = None

        return not has_moves

    def get_winner(self) -> int:
        """Get winner: 1=black, 2=white, 0=draw."""
        black = sum(1 for c in self.board if c == 1)
        white = sum(1 for c in self.board if c == 2)

        if black > white:
            return 1
        elif white > black:
            return 2
        return 0

    def get_current_player(self) -> int:
        return self.current_player

    @classmethod
    def generate_random_position(cls, max_moves: Optional[int] = None) -> 'OthelloEnv':
        """Generate a random non-terminal position."""
        game = cls()
        n_moves = random.randint(0, max_moves or 30)

        for _ in range(n_moves):
            legal = game.get_legal_moves()
            if not legal or game.is_terminal():
                break
            game.make_move(random.choice(legal))

        return game

    def get_features(self) -> Dict[str, float]:
        """Extract features for guard evaluation."""
        black = sum(1 for c in self.board if c == 1)
        white = sum(1 for c in self.board if c == 2)
        total = black + white
        legal_moves = self.get_legal_moves()

        player_pieces = black if self.current_player == 1 else white
        opponent_pieces = white if self.current_player == 1 else black

        return {
            "move_count": total / 64.0,
            "player_pieces": player_pieces / 32.0,
            "opponent_pieces": opponent_pieces / 32.0,
            "mobility": len(legal_moves) / 20.0,  # Normalize by typical max
        }

    def get_move_features(self, move: int) -> Dict[str, float]:
        """Get features specific to a move position."""
        flips = self._get_flips(move, self.current_player)

        return {
            "cell_empty": 1.0 if self.board[move] == 0 else 0.0,
            "is_corner": 1.0 if move in self.CORNERS else 0.0,
            "is_edge": 1.0 if (move // 8 in [0, 7] or move % 8 in [0, 7]) else 0.0,
            "is_x_square": 1.0 if move in self.X_SQUARES else 0.0,
            "is_c_square": 1.0 if move in self.C_SQUARES else 0.0,
            "flip_count": len(flips) / 10.0,  # Normalize
        }

    def display(self) -> str:
        """Return string representation of the board."""
        symbols = {0: ".", 1: "●", 2: "○"}
        lines = ["  a b c d e f g h"]
        for row in range(8):
            line = f"{row+1} "
            for col in range(8):
                line += symbols[self.board[row * 8 + col]] + " "
            lines.append(line)
        return "\n".join(lines)


# =============================================================================
# Go 9x9 Implementation
# =============================================================================

class Go9x9Env(GameEnvironment):
    """
    Go 9x9 game environment with Ko rule.

    Key rules encoded:
    - Empty intersection + not suicide + not Ko = legal
    - Suicide: placing a stone with no liberties that doesn't capture
    - Ko: can't immediately recapture single stone

    Uses existing Go9x9Statechart from exp_go_9x9 for reference.
    """

    config = GameConfig(
        name="Go9x9",
        n_positions=81,  # 9x9 board
        n_players=2,
        max_game_length=150,  # Typical game length
        base_features=[
            "cell_empty",       # Is target cell empty?
            "is_ko_point",      # Is this the Ko-forbidden point?
            "is_suicide",       # Would this be suicide?
            "is_corner",        # Corner position (4 corners)
            "is_edge",          # Edge position
            "is_center",        # Center region (3x3)
            "move_count",       # How many moves played?
            "player_stones",    # Current player's stone count
            "opponent_stones",  # Opponent's stone count
            "liberties_after",  # Liberties after placing
        ]
    )

    SIZE = 9
    TOTAL = 81

    # Strategic positions
    CORNERS = {0, 8, 72, 80}  # a1, j1, a9, j9
    STAR_POINTS = {20, 24, 40, 56, 60}  # 3-3 and center

    # Empty=0, Black=1, White=2
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self):
        self.board = [self.EMPTY] * self.TOTAL
        self.current_player = 1  # Black starts
        self.ko_point = None  # Position forbidden by Ko rule
        self.consecutive_passes = 0
        self._legal_cache = None

    def reset(self) -> 'Go9x9Env':
        self.board = [self.EMPTY] * self.TOTAL
        self.current_player = 1
        self.ko_point = None
        self.consecutive_passes = 0
        self._legal_cache = None
        return self

    def copy(self) -> 'Go9x9Env':
        new = Go9x9Env.__new__(Go9x9Env)
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.ko_point = self.ko_point
        new.consecutive_passes = self.consecutive_passes
        new._legal_cache = None
        return new

    def _idx_to_xy(self, idx: int) -> Tuple[int, int]:
        """Convert index to (x, y)."""
        return idx % self.SIZE, idx // self.SIZE

    def _xy_to_idx(self, x: int, y: int) -> int:
        """Convert (x, y) to index."""
        return y * self.SIZE + x

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE

    def _get_neighbors(self, idx: int) -> List[int]:
        """Get orthogonal neighbor indices."""
        x, y = self._idx_to_xy(idx)
        neighbors = []
        if x > 0:
            neighbors.append(self._xy_to_idx(x - 1, y))
        if x < self.SIZE - 1:
            neighbors.append(self._xy_to_idx(x + 1, y))
        if y > 0:
            neighbors.append(self._xy_to_idx(x, y - 1))
        if y < self.SIZE - 1:
            neighbors.append(self._xy_to_idx(x, y + 1))
        return neighbors

    def _get_group(self, idx: int) -> Set[int]:
        """Get all stones connected to idx of same color."""
        color = self.board[idx]
        if color == self.EMPTY:
            return set()

        group = set()
        stack = [idx]

        while stack:
            current = stack.pop()
            if current in group:
                continue
            if self.board[current] != color:
                continue

            group.add(current)
            for n in self._get_neighbors(current):
                if n not in group and self.board[n] == color:
                    stack.append(n)

        return group

    def _get_liberties(self, idx: int) -> Set[int]:
        """Get liberties of the group containing idx."""
        group = self._get_group(idx)
        if not group:
            return set()

        liberties = set()
        for stone in group:
            for n in self._get_neighbors(stone):
                if self.board[n] == self.EMPTY:
                    liberties.add(n)

        return liberties

    def _would_be_suicide(self, idx: int, color: int) -> bool:
        """Check if placing color at idx would be suicide."""
        # Temporarily place stone
        old = self.board[idx]
        self.board[idx] = color
        opp = 3 - color

        # Check if we capture any opponent stones
        captures = False
        for n in self._get_neighbors(idx):
            if self.board[n] == opp:
                if len(self._get_liberties(n)) == 0:
                    captures = True
                    break

        if captures:
            self.board[idx] = old
            return False  # Not suicide - we capture

        # Check if our stone has liberties
        has_liberties = len(self._get_liberties(idx)) > 0
        self.board[idx] = old

        return not has_liberties

    def _is_ko_point(self, idx: int) -> bool:
        """Check if idx is forbidden by Ko rule."""
        return self.ko_point == idx

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.board)

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices."""
        if self._legal_cache is not None:
            return self._legal_cache

        legal = []
        for idx in range(self.TOTAL):
            if self.is_legal(idx):
                legal.append(idx)

        self._legal_cache = legal
        return legal

    def is_legal(self, move: int) -> bool:
        """Check if a specific move is legal."""
        if not (0 <= move < self.TOTAL):
            return False

        # Must be empty
        if self.board[move] != self.EMPTY:
            return False

        # Must not be Ko point
        if self._is_ko_point(move):
            return False

        # Must not be suicide
        if self._would_be_suicide(move, self.current_player):
            return False

        return True

    def make_move(self, move: int) -> bool:
        """Make a move. Returns True if successful."""
        if not self.is_legal(move):
            return False

        color = self.current_player
        opp = 3 - color

        # Place stone
        self.board[move] = color

        # Capture opponent stones with no liberties
        captured_stones = []
        for n in self._get_neighbors(move):
            if self.board[n] == opp:
                if len(self._get_liberties(n)) == 0:
                    group = self._get_group(n)
                    captured_stones.extend(group)
                    for stone in group:
                        self.board[stone] = self.EMPTY

        # Update Ko point
        # Ko: captured exactly one stone, and our stone has exactly one liberty
        if len(captured_stones) == 1:
            our_group = self._get_group(move)
            our_libs = self._get_liberties(move)
            if len(our_group) == 1 and len(our_libs) == 1:
                self.ko_point = captured_stones[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Switch player
        self.current_player = 3 - self.current_player
        self.consecutive_passes = 0
        self._legal_cache = None

        return True

    def pass_turn(self):
        """Pass (don't place a stone)."""
        self.ko_point = None  # Ko clears on pass
        self.current_player = 3 - self.current_player
        self.consecutive_passes += 1
        self._legal_cache = None

    def is_terminal(self) -> bool:
        """Game ends after two consecutive passes."""
        return self.consecutive_passes >= 2

    def get_winner(self) -> int:
        """Get winner by simple area counting."""
        if not self.is_terminal():
            return 0

        black = sum(1 for s in self.board if s == self.BLACK)
        white = sum(1 for s in self.board if s == self.WHITE)

        # Komi for white
        white_score = white + 6.5

        if black > white_score:
            return 1
        elif white_score > black:
            return 2
        return 0

    def get_current_player(self) -> int:
        return self.current_player

    @classmethod
    def generate_random_position(cls, max_moves: Optional[int] = None) -> 'Go9x9Env':
        """Generate a random non-terminal position."""
        game = cls()
        n_moves = random.randint(0, max_moves or 40)

        for _ in range(n_moves):
            legal = game.get_legal_moves()
            if not legal or game.is_terminal():
                break
            # Occasionally pass to add variety
            if random.random() < 0.05:
                game.pass_turn()
            else:
                game.make_move(random.choice(legal))

        return game

    def get_features(self) -> Dict[str, float]:
        """Extract features for guard evaluation."""
        black = sum(1 for s in self.board if s == self.BLACK)
        white = sum(1 for s in self.board if s == self.WHITE)
        total = black + white
        legal_moves = self.get_legal_moves()

        player_stones = black if self.current_player == 1 else white
        opp_stones = white if self.current_player == 1 else black

        return {
            "move_count": total / 81.0,
            "player_stones": player_stones / 40.0,
            "opponent_stones": opp_stones / 40.0,
            "mobility": len(legal_moves) / 81.0,
            "has_ko": 1.0 if self.ko_point is not None else 0.0,
        }

    def get_move_features(self, move: int) -> Dict[str, float]:
        """Get features specific to a move position."""
        x, y = self._idx_to_xy(move)

        is_corner = move in self.CORNERS
        is_edge = (x == 0 or x == 8 or y == 0 or y == 8) and not is_corner
        is_center = 2 <= x <= 6 and 2 <= y <= 6

        return {
            "cell_empty": 1.0 if self.board[move] == self.EMPTY else 0.0,
            "is_ko_point": 1.0 if self._is_ko_point(move) else 0.0,
            "is_corner": 1.0 if is_corner else 0.0,
            "is_edge": 1.0 if is_edge else 0.0,
            "is_center": 1.0 if is_center else 0.0,
            "is_star_point": 1.0 if move in self.STAR_POINTS else 0.0,
        }

    def display(self) -> str:
        """Return string representation."""
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for y in range(self.SIZE):
            row = " ".join(symbols[self.board[y * self.SIZE + x]] for x in range(self.SIZE))
            lines.append(f"{self.SIZE - y:2} {row}")
        lines.append("   A B C D E F G H J")
        return "\n".join(lines)


# =============================================================================
# Environment Registry
# =============================================================================

ENVIRONMENTS = {
    "tictactoe": TicTacToeEnv,
    "connect4": Connect4Env,
    "othello": OthelloEnv,
    "go9x9": Go9x9Env,
}


def get_environment(name: str) -> type:
    """Get environment class by name."""
    name = name.lower()
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENVIRONMENTS.keys())}")
    return ENVIRONMENTS[name]


# =============================================================================
# Testing
# =============================================================================

def test_environment(env_class: type, n_games: int = 10):
    """Test an environment implementation."""
    print(f"\nTesting {env_class.config.name}...")
    print(f"  Board size: {env_class.config.n_positions}")
    print(f"  Max game length: {env_class.config.max_game_length}")

    # Test random games
    wins = {0: 0, 1: 0, 2: 0}
    for _ in range(n_games):
        game = env_class()
        while not game.is_terminal():
            legal = game.get_legal_moves()
            if not legal:
                break
            game.make_move(random.choice(legal))
        wins[game.get_winner()] += 1

    print(f"  Random game results: P1={wins[1]}, P2={wins[2]}, Draw={wins[0]}")

    # Test dataset generation
    dataset = generate_legal_move_dataset(env_class, n_positions=100)
    print(f"  Dataset: {len(dataset)} positions")

    # Test features
    game = env_class.generate_random_position()
    features = game.get_features()
    print(f"  Features: {list(features.keys())}")

    print(f"  {env_class.config.name} OK!")


if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT ABSTRACTION TEST")
    print("=" * 60)

    test_environment(TicTacToeEnv)
    test_environment(Connect4Env)
    test_environment(OthelloEnv)
    test_environment(Go9x9Env)

    print("\nAll environments passed!")
