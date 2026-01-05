"""
Fused Attention-Statechart Model for Connect 4.

Connect 4 is a better test than TicTacToe because:
- 7x6 board (42 cells vs 9)
- 69 winning lines (vs 8)
- Column-based play with gravity
- More complex fork/threat patterns
- State space too large to memorize

Fusion approach:
1. Explicit signals: Win detection, block detection, threat counting
2. Implicit patterns: Self-attention over board positions
3. Column bias: Center columns are strategically stronger
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time

# Board dimensions
ROWS = 6
COLS = 7
BOARD_SIZE = ROWS * COLS

# Generate all winning lines (4 in a row)
def generate_winning_lines():
    lines = []
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            lines.append([r * COLS + c + i for i in range(4)])
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            lines.append([(r + i) * COLS + c for i in range(4)])
    # Diagonal (down-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            lines.append([(r + i) * COLS + (c + i) for i in range(4)])
    # Diagonal (down-left)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            lines.append([(r + i) * COLS + (c - i) for i in range(4)])
    return lines

WINNING_LINES = generate_winning_lines()
print(f"Connect 4: {len(WINNING_LINES)} winning lines")


def count_params(model: nn.Module) -> int:
    def count_recursive(p):
        if isinstance(p, mx.array):
            return p.size
        elif isinstance(p, dict):
            return sum(count_recursive(v) for v in p.values())
        elif isinstance(p, list):
            return sum(count_recursive(v) for v in p)
        return 0
    return count_recursive(model.parameters())


def check_winner(board: list) -> int:
    """Check winner. 0=none, 1=P1, 2=P2."""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] == board[line[3]] != 0:
            return board[line[0]]
    return 0


def get_valid_moves(board: list) -> list:
    """Get columns that aren't full."""
    valid = []
    for col in range(COLS):
        if board[col] == 0:  # Top row of column is empty
            valid.append(col)
    return valid


def get_drop_row(board: list, col: int) -> int:
    """Get the row where a piece would land in given column."""
    for row in range(ROWS - 1, -1, -1):
        if board[row * COLS + col] == 0:
            return row
    return -1


def make_move(board: list, col: int, player: int) -> list:
    """Make a move, returning new board."""
    row = get_drop_row(board, col)
    if row >= 0:
        new_board = board.copy()
        new_board[row * COLS + col] = player
        return new_board
    return board


class Connect4Signals(nn.Module):
    """
    Explicit statechart-derived signals for Connect 4.

    Computes:
    - Winning moves (immediate win in column)
    - Blocking moves (block opponent's immediate win)
    - Threat count (number of 3-in-a-row threats after move)
    """

    def __init__(self):
        super().__init__()

    def get_column_wins(self, board: mx.array, player: int) -> mx.array:
        """Find columns that would win for player."""
        B = board.shape[0]
        wins = mx.zeros((B, COLS))

        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            for col in range(COLS):
                row = get_drop_row(board_list, col)
                if row < 0:
                    continue
                # Simulate placing piece
                test_board = board_list.copy()
                test_board[row * COLS + col] = player
                if check_winner(test_board) == player:
                    wins = wins.at[b, col].add(1.0)

        return wins

    def get_threat_count(self, board: mx.array, player: int) -> mx.array:
        """Count threats (3-in-a-row with open 4th) after each column move."""
        B = board.shape[0]
        threats = mx.zeros((B, COLS))

        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            for col in range(COLS):
                row = get_drop_row(board_list, col)
                if row < 0:
                    continue
                # Simulate placing piece
                test_board = board_list.copy()
                test_board[row * COLS + col] = player

                # Count lines with 3 of player and 1 empty
                threat_count = 0
                for line in WINNING_LINES:
                    player_count = sum(1 for i in line if test_board[i] == player)
                    empty_count = sum(1 for i in line if test_board[i] == 0)
                    if player_count == 3 and empty_count == 1:
                        threat_count += 1

                threats = threats.at[b, col].add(float(threat_count))

        return threats

    def __call__(self, board: mx.array, turn: mx.array) -> dict:
        """Compute all signals."""
        B = board.shape[0]

        # Determine player (0=P1, 1=P2)
        is_p1 = (turn == 0).astype(mx.float32)[:, None]
        is_p2 = (turn == 1).astype(mx.float32)[:, None]

        # Winning moves
        p1_wins = self.get_column_wins(board, 1)
        p2_wins = self.get_column_wins(board, 2)
        my_wins = is_p1 * p1_wins + is_p2 * p2_wins

        # Blocking moves
        opp_wins = is_p1 * p2_wins + is_p2 * p1_wins

        # Threat creation
        p1_threats = self.get_threat_count(board, 1)
        p2_threats = self.get_threat_count(board, 2)
        my_threats = is_p1 * p1_threats + is_p2 * p2_threats

        # Valid columns
        valid = mx.zeros((B, COLS))
        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            for col in get_valid_moves(board_list):
                valid = valid.at[b, col].add(1.0)

        return {
            "my_wins": my_wins,
            "opp_wins": opp_wins,
            "my_threats": my_threats,
            "valid": valid,
        }


class BoardAttention(nn.Module):
    """
    Self-attention over the Connect 4 board.

    Learns spatial relationships between positions.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Cell embeddings (empty, P1, P2)
        self.cell_embed = nn.Embedding(3, embed_dim)

        # Position embeddings (row + column encoding)
        self.row_embed = nn.Embedding(ROWS, embed_dim // 2)
        self.col_embed = nn.Embedding(COLS, embed_dim // 2)

        # Turn embedding
        self.turn_embed = nn.Embedding(2, embed_dim)

        # Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pool columns
        self.col_pool = nn.Linear(ROWS * embed_dim, embed_dim)

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Compute attention over board, output per-column scores.

        Returns: [B, COLS] column scores
        """
        B = board.shape[0]

        # Embed cells
        cell_emb = self.cell_embed(board)  # [B, 42, d]

        # Add position embeddings
        rows = mx.arange(ROWS)[:, None].flatten()  # [0,0,0,0,0,0,0, 1,1,1,... ]
        cols = mx.arange(COLS)[None, :].flatten()  # [0,1,2,3,4,5,6, 0,1,2,...]
        # Actually for 6x7 board, indices are row-major
        row_idx = mx.arange(BOARD_SIZE) // COLS  # [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ...]
        col_idx = mx.arange(BOARD_SIZE) % COLS   # [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, ...]

        row_emb = self.row_embed(row_idx)  # [42, d/2]
        col_emb = self.col_embed(col_idx)  # [42, d/2]
        pos_emb = mx.concatenate([row_emb, col_emb], axis=-1)  # [42, d]

        x = cell_emb + pos_emb[None, :, :]  # [B, 42, d]

        # Add turn embedding to queries
        turn_emb = self.turn_embed(turn)  # [B, d]

        Q = self.q_proj(x + turn_emb[:, None, :])  # [B, 42, d]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = mx.softmax((Q @ K.transpose(0, 2, 1)) * scale, axis=-1)
        attn_out = self.out_proj(attn @ V)  # [B, 42, d]

        # Reshape to [B, ROWS, COLS, d] and pool over rows
        attn_out = attn_out.reshape(B, ROWS, COLS, self.embed_dim)
        col_features = attn_out.reshape(B, COLS, ROWS * self.embed_dim)
        col_scores = self.col_pool(col_features)  # [B, COLS, d]

        return col_scores


class FusedConnect4(nn.Module):
    """
    Fusion of explicit signals and attention for Connect 4.
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Explicit signals
        self.signals = Connect4Signals()

        # Learned attention
        self.attention = BoardAttention(embed_dim, num_heads=4)

        # Score projection
        self.score_proj = nn.Linear(embed_dim, 1)

        # Column bias (center columns are better)
        # Columns: 0, 1, 2, 3, 4, 5, 6
        # Center (3) is best, then 2,4, then 1,5, then 0,6
        self.col_bias = mx.array([0.2, 0.4, 0.6, 1.0, 0.6, 0.4, 0.2])

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Compute column logits.

        Priority:
        1. Win (if available)
        2. Block (if opponent can win)
        3. Threat creation + attention patterns
        """
        B = board.shape[0]

        # Get explicit signals
        sig = self.signals(board, turn)
        my_wins = sig["my_wins"]      # [B, 7]
        opp_wins = sig["opp_wins"]    # [B, 7]
        my_threats = sig["my_threats"]  # [B, 7]
        valid = sig["valid"]          # [B, 7]

        # Get attention patterns
        attn_scores = self.attention(board, turn)  # [B, 7, d]
        attn_col_scores = self.score_proj(attn_scores)[:, :, 0]  # [B, 7]

        # Combine
        col_bias = mx.broadcast_to(self.col_bias[None, :], (B, COLS))

        # Strategic score = attention + threats + column bias
        strategic = attn_col_scores * 2.0 + my_threats * 1.5 + col_bias * 0.5

        # Hierarchical selection
        has_win = mx.sum(my_wins, axis=-1, keepdims=True) > 0
        has_block = mx.sum(opp_wins, axis=-1, keepdims=True) > 0

        logits = mx.where(
            has_win,
            my_wins * 100.0,
            mx.where(
                has_block,
                opp_wins * 50.0,
                strategic
            )
        )

        # Mask invalid columns
        logits = logits + (1 - valid) * (-1e9)

        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        logits = self(board, turn)
        return mx.softmax(logits, axis=-1)


class PureAttentionConnect4(nn.Module):
    """
    Pure attention model for comparison (no explicit signals).
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.attention = BoardAttention(embed_dim, num_heads=4)
        self.score_proj = nn.Linear(embed_dim, 1)

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        B = board.shape[0]
        attn_scores = self.attention(board, turn)
        logits = self.score_proj(attn_scores)[:, :, 0]

        # Mask invalid columns
        valid = mx.zeros((B, COLS))
        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            for col in get_valid_moves(board_list):
                valid = valid.at[b, col].add(1.0)

        logits = logits + (1 - valid) * (-1e9)
        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        return mx.softmax(self(board, turn), axis=-1)


def play_vs_random(model, model_is_p1: bool = True) -> int:
    """Play model vs random. Returns +1 win, -1 loss, 0 draw."""
    board = [0] * BOARD_SIZE
    model_player = 1 if model_is_p1 else 2

    for move_num in range(BOARD_SIZE):
        valid = get_valid_moves(board)
        if not valid:
            break

        current_player = 1 if move_num % 2 == 0 else 2
        turn = move_num % 2

        if current_player == model_player:
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])
            probs = model.get_move_probs(board_arr, turn_arr)[0]
            col = int(mx.argmax(probs).item())
            if col not in valid:
                col = valid[0]
        else:
            col = valid[mx.random.randint(0, len(valid), ()).item()]

        board = make_move(board, col, current_player)

        winner = check_winner(board)
        if winner == model_player:
            return 1
        elif winner != 0:
            return -1

    return 0


def evaluate(model, num_games: int = 200) -> dict:
    """Evaluate model vs random."""
    wins = losses = draws = 0

    for i in range(num_games):
        result = play_vs_random(model, model_is_p1=(i % 2 == 0))
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
    }


def train_model(model, num_episodes: int = 1000, lr: float = 0.01) -> dict:
    """Train via REINFORCE."""
    optimizer = optim.Adam(learning_rate=lr)
    history = []

    print("Training...")
    print("-" * 50)

    for ep in range(num_episodes):
        board = [0] * BOARD_SIZE
        log_probs_p1 = []
        log_probs_p2 = []

        for move_num in range(BOARD_SIZE):
            valid = get_valid_moves(board)
            if not valid:
                break

            turn = move_num % 2
            current_player = 1 if turn == 0 else 2

            board_arr = mx.array([board])
            turn_arr = mx.array([turn])
            probs = model.get_move_probs(board_arr, turn_arr)[0]

            # Sample move with annealing temperature
            temp = max(0.3, 1.0 - ep / num_episodes)
            logits = mx.log(probs + 1e-10) / temp
            sample_probs = mx.softmax(logits, axis=-1)

            cumsum = mx.cumsum(sample_probs)
            rand = mx.random.uniform(shape=())
            col = int(mx.sum(cumsum < rand).item())
            if col >= COLS or col not in valid:
                col = int(mx.argmax(probs).item())
                if col not in valid:
                    col = valid[0]

            log_p = mx.log(probs[col] + 1e-10)
            if turn == 0:
                log_probs_p1.append(log_p)
            else:
                log_probs_p2.append(log_p)

            board = make_move(board, col, current_player)

            winner = check_winner(board)
            if winner != 0:
                break

        # Compute reward
        winner = check_winner(board)
        if winner == 1:
            r1, r2 = 1.0, -1.0
        elif winner == 2:
            r1, r2 = -1.0, 1.0
        else:
            r1, r2 = 0.0, 0.0

        # Update
        def loss_fn(m):
            loss = mx.array(0.0)
            for lp in log_probs_p1:
                loss = loss - r1 * lp
            for lp in log_probs_p2:
                loss = loss - r2 * lp
            return loss

        if log_probs_p1 or log_probs_p2:
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        # Evaluate periodically
        if (ep + 1) % 250 == 0:
            result = evaluate(model, num_games=100)
            history.append(result["win_rate"])
            print(f"Episode {ep+1:4d}: win_rate={result['win_rate']*100:.1f}%, "
                  f"W/L/D={result['wins']}/{result['losses']}/{result['draws']}")

    print("-" * 50)
    return history


def run_experiment():
    """Compare fused vs pure attention on Connect 4."""
    print("=" * 70)
    print("CONNECT 4: FUSED VS PURE ATTENTION")
    print("=" * 70)
    print()
    print(f"Board: {ROWS}x{COLS} = {BOARD_SIZE} cells")
    print(f"Winning lines: {len(WINNING_LINES)}")
    print()

    mx.random.seed(42)

    # Create models
    fused = FusedConnect4(embed_dim=32)
    pure = PureAttentionConnect4(embed_dim=32)

    fused_params = count_params(fused)
    pure_params = count_params(pure)

    print(f"Fused model params: {fused_params:,}")
    print(f"Pure attention params: {pure_params:,}")
    print()

    # Evaluate untrained
    print("UNTRAINED evaluation (100 games each)...")
    fused_untrained = evaluate(fused, 100)
    pure_untrained = evaluate(pure, 100)
    print(f"Fused untrained: {fused_untrained['win_rate']*100:.1f}%, "
          f"W/L/D={fused_untrained['wins']}/{fused_untrained['losses']}/{fused_untrained['draws']}")
    print(f"Pure untrained:  {pure_untrained['win_rate']*100:.1f}%, "
          f"W/L/D={pure_untrained['wins']}/{pure_untrained['losses']}/{pure_untrained['draws']}")
    print()

    # Train fused
    print("Training FUSED model...")
    train_model(fused, num_episodes=1000, lr=0.01)
    print()

    # Train pure attention
    print("Training PURE ATTENTION model...")
    train_model(pure, num_episodes=1000, lr=0.01)
    print()

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION (200 games each)")
    print("=" * 70)
    print()

    fused_final = evaluate(fused, 200)
    pure_final = evaluate(pure, 200)

    print(f"{'Model':<20} | {'Win Rate':<10} | {'W/L/D':<15} | {'Params':<10}")
    print("-" * 65)
    print(f"{'Random baseline':<20} | {'~35%':<10} | {'-':<15} | {'-':<10}")
    fused_wld = f"{fused_final['wins']}/{fused_final['losses']}/{fused_final['draws']}"
    pure_wld = f"{pure_final['wins']}/{pure_final['losses']}/{pure_final['draws']}"
    print(f"{'Pure Attention':<20} | {pure_final['win_rate']*100:.1f}%{'':<5} | {pure_wld:<15} | {pure_params:,}{'':<1}")
    print(f"{'FUSED (ours)':<20} | {fused_final['win_rate']*100:.1f}%{'':<5} | {fused_wld:<15} | {fused_params:,}{'':<1}")
    print()

    # Analysis
    improvement = (fused_final['win_rate'] - pure_final['win_rate']) * 100
    print(f"Improvement from fusion: {improvement:+.1f}%")

    if fused_final['losses'] < pure_final['losses']:
        print(f"Fusion reduces losses: {fused_final['losses']} vs {pure_final['losses']}")

    print()
    print("=" * 70)

    return {
        "fused": fused_final,
        "pure": pure_final,
    }


if __name__ == "__main__":
    run_experiment()
