"""Tic-Tac-Toe as a statechart for differentiable validation.

Statechart Structure:
    TicTacToe (OR)
    ├── XTurn (OR)
    │   ├── Playing
    │   └── XWins (final)
    ├── OTurn (OR)
    │   ├── Playing
    │   └── OWins (final)
    └── Draw (final)

Guards: cell_empty(i), is_winning_move(), board_full()
Events: MOVE_0..MOVE_8
"""

# Statechart definition as dict (compatible with sc proto format)
TICTACTOE_STATECHART = {
    "name": "TicTacToe",
    "root_state": {
        "label": "__root__",
        "type": 2,  # OR (Normal)
        "children": [
            {
                "label": "XTurn",
                "type": 2,  # OR
                "is_initial": True,
                "children": [
                    {"label": "XPlaying", "type": 1, "is_initial": True},  # BASIC
                    {"label": "XWins", "type": 1, "is_final": True},
                ]
            },
            {
                "label": "OTurn",
                "type": 2,  # OR
                "children": [
                    {"label": "OPlaying", "type": 1, "is_initial": True},
                    {"label": "OWins", "type": 1, "is_final": True},
                ]
            },
            {"label": "Draw", "type": 1, "is_final": True},
        ]
    },
    "events": [
        {"label": f"MOVE_{i}"} for i in range(9)
    ],
    "transitions": [
        # X moves (from XPlaying to OPlaying, unless win/draw)
        *[{
            "label": f"x_move_{i}",
            "from": ["XPlaying"],
            "to": ["OPlaying"],
            "event": f"MOVE_{i}",
            "guard": {"expression": f"cell_empty({i}) && !is_x_winning({i})"},
        } for i in range(9)],
        # X winning moves
        *[{
            "label": f"x_wins_{i}",
            "from": ["XPlaying"],
            "to": ["XWins"],
            "event": f"MOVE_{i}",
            "guard": {"expression": f"cell_empty({i}) && is_x_winning({i})"},
        } for i in range(9)],
        # O moves
        *[{
            "label": f"o_move_{i}",
            "from": ["OPlaying"],
            "to": ["XPlaying"],
            "event": f"MOVE_{i}",
            "guard": {"expression": f"cell_empty({i}) && !is_o_winning({i})"},
        } for i in range(9)],
        # O winning moves
        *[{
            "label": f"o_wins_{i}",
            "from": ["OPlaying"],
            "to": ["OWins"],
            "event": f"MOVE_{i}",
            "guard": {"expression": f"cell_empty({i}) && is_o_winning({i})"},
        } for i in range(9)],
        # Draw detection (board full after X move without winner)
        {
            "label": "draw_after_x",
            "from": ["XPlaying"],
            "to": ["Draw"],
            "guard": {"expression": "board_full()"},
        },
    ],
}

# State indices for soft configuration
STATE_LABELS = [
    "__root__",
    "XTurn", "XPlaying", "XWins",
    "OTurn", "OPlaying", "OWins",
    "Draw"
]

STATE_TO_IDX = {label: i for i, label in enumerate(STATE_LABELS)}
IDX_TO_STATE = {i: label for i, label in enumerate(STATE_LABELS)}

# Winning lines (indices into 3x3 board)
WINNING_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6],              # diags
]


def check_winner(board, player):
    """Check if player (1=X, 2=O) has won."""
    for line in WINNING_LINES:
        if all(board[i] == player for i in line):
            return True
    return False


def is_winning_move(board, player, cell):
    """Check if placing player at cell would win."""
    if board[cell] != 0:
        return False
    test_board = list(board)
    test_board[cell] = player
    return check_winner(test_board, player)


def get_legal_moves(board):
    """Get list of empty cell indices."""
    return [i for i in range(9) if board[i] == 0]


def board_full(board):
    """Check if board is full."""
    return all(cell != 0 for cell in board)
