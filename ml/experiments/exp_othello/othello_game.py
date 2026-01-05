"""
Othello Game Logic.

Board: 8x8 = 64 cells
Players: Black (1), White (2)
Rules: Must flip at least one opponent piece

Statechart structure:
Othello (AND)
├── GamePhase (OR): Opening | Midgame | Endgame
├── Turn (OR): Black | White
├── Board[64] (PARALLEL): Empty | Black | White
└── ValidMoves: computed from flip constraints
"""

import mlx.core as mx

# Constants
ROWS = 8
COLS = 8
BOARD_SIZE = 64

EMPTY = 0
BLACK = 1
WHITE = 2

# 8 directions: (row_delta, col_delta)
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Strategic positions
CORNERS = [0, 7, 56, 63]  # A1, H1, A8, H8
X_SQUARES = [9, 14, 49, 54]  # Diagonal to corners (bad)
C_SQUARES = [1, 6, 8, 15, 48, 55, 57, 62]  # Adjacent to corners (risky)
EDGES = [i for i in range(64) if i // 8 == 0 or i // 8 == 7 or i % 8 == 0 or i % 8 == 7]


def idx_to_rc(idx: int) -> tuple:
    """Convert flat index to (row, col)."""
    return idx // COLS, idx % COLS


def rc_to_idx(row: int, col: int) -> int:
    """Convert (row, col) to flat index."""
    return row * COLS + col


def is_valid_rc(row: int, col: int) -> bool:
    """Check if row, col is on board."""
    return 0 <= row < ROWS and 0 <= col < COLS


def get_initial_board() -> list:
    """Get starting Othello position."""
    board = [EMPTY] * BOARD_SIZE
    # Center 4 squares: White on D4, E5; Black on D5, E4
    board[rc_to_idx(3, 3)] = WHITE
    board[rc_to_idx(3, 4)] = BLACK
    board[rc_to_idx(4, 3)] = BLACK
    board[rc_to_idx(4, 4)] = WHITE
    return board


def opponent(player: int) -> int:
    """Get opponent player."""
    return WHITE if player == BLACK else BLACK


def get_flips_in_direction(board: list, pos: int, player: int, dr: int, dc: int) -> list:
    """
    Get list of positions that would be flipped in a direction.

    This is the core "guard" function: has_flip(pos, dir).
    Returns empty list if no valid flips.
    """
    row, col = idx_to_rc(pos)
    opp = opponent(player)
    flips = []

    r, c = row + dr, col + dc
    while is_valid_rc(r, c):
        idx = rc_to_idx(r, c)
        if board[idx] == opp:
            flips.append(idx)
            r, c = r + dr, c + dc
        elif board[idx] == player:
            return flips  # Found our piece, flips are valid
        else:
            return []  # Empty cell, no valid flips

    return []  # Reached edge without finding our piece


def get_all_flips(board: list, pos: int, player: int) -> list:
    """Get all positions that would be flipped by placing at pos."""
    if board[pos] != EMPTY:
        return []

    all_flips = []
    for dr, dc in DIRECTIONS:
        all_flips.extend(get_flips_in_direction(board, pos, player, dr, dc))

    return all_flips


def is_valid_move(board: list, pos: int, player: int) -> bool:
    """Check if pos is a valid move (flips at least one piece)."""
    return len(get_all_flips(board, pos, player)) > 0


def get_valid_moves(board: list, player: int) -> list:
    """Get all valid moves for player."""
    return [pos for pos in range(BOARD_SIZE) if is_valid_move(board, pos, player)]


def make_move(board: list, pos: int, player: int) -> list:
    """Make a move and flip pieces. Returns new board."""
    flips = get_all_flips(board, pos, player)
    if not flips:
        return board  # Invalid move, no change

    new_board = board.copy()
    new_board[pos] = player
    for flip_pos in flips:
        new_board[flip_pos] = player

    return new_board


def count_pieces(board: list) -> tuple:
    """Count (black, white, empty) pieces."""
    black = sum(1 for c in board if c == BLACK)
    white = sum(1 for c in board if c == WHITE)
    empty = BOARD_SIZE - black - white
    return black, white, empty


def get_game_phase(board: list) -> int:
    """
    Determine game phase.
    0 = Opening (< 20 pieces on board)
    1 = Midgame (20-50 pieces)
    2 = Endgame (> 50 pieces)
    """
    black, white, _ = count_pieces(board)
    total = black + white
    if total < 20:
        return 0  # Opening
    elif total < 50:
        return 1  # Midgame
    else:
        return 2  # Endgame


def is_game_over(board: list) -> bool:
    """Check if game is over (neither player can move)."""
    return not get_valid_moves(board, BLACK) and not get_valid_moves(board, WHITE)


def get_winner(board: list) -> int:
    """Get winner. 0=tie, 1=black, 2=white."""
    black, white, _ = count_pieces(board)
    if black > white:
        return BLACK
    elif white > black:
        return WHITE
    return 0


def get_stability(board: list, player: int) -> int:
    """
    Count stable pieces (cannot be flipped).
    Corners are always stable. Pieces connected to corners along edges are stable.
    """
    stable = 0

    # Corners are always stable
    for corner in CORNERS:
        if board[corner] == player:
            stable += 1

    # Simplified: count edge pieces that are part of complete edge runs
    # (A more complete implementation would trace from corners)

    return stable


def get_mobility(board: list, player: int) -> int:
    """Count valid moves for player."""
    return len(get_valid_moves(board, player))


def get_frontier(board: list, player: int) -> int:
    """
    Count frontier pieces (adjacent to empty squares).
    Fewer frontier pieces is often better (less attackable).
    """
    frontier = 0
    for pos in range(BOARD_SIZE):
        if board[pos] != player:
            continue
        row, col = idx_to_rc(pos)
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if is_valid_rc(r, c) and board[rc_to_idx(r, c)] == EMPTY:
                frontier += 1
                break
    return frontier


def board_to_string(board: list) -> str:
    """Pretty print board."""
    symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
    lines = ["  A B C D E F G H"]
    for row in range(ROWS):
        line = f"{row+1} "
        for col in range(COLS):
            line += symbols[board[rc_to_idx(row, col)]] + " "
        lines.append(line)
    return "\n".join(lines)


# Test
if __name__ == "__main__":
    board = get_initial_board()
    print("Initial board:")
    print(board_to_string(board))
    print()

    print(f"Valid moves for Black: {get_valid_moves(board, BLACK)}")
    print(f"Game phase: {['Opening', 'Midgame', 'Endgame'][get_game_phase(board)]}")

    # Make a move
    board = make_move(board, 19, BLACK)  # D3
    print("\nAfter Black plays D3:")
    print(board_to_string(board))
    print(f"Valid moves for White: {get_valid_moves(board, WHITE)}")
