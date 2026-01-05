"""
Fused Attention-Statechart Model for Othello.

Othello properties:
- 8x8 board, ~10^28 states
- Flip propagation = parallel region updates
- Game phases = hierarchical (Opening/Mid/End)
- Capture rules = guards (must flip at least one)

Statechart structure:
Othello (AND)
├── GamePhase (OR): Opening | Midgame | Endgame
├── Turn (OR): Black | White
├── Board[64] (PARALLEL): Empty | Black | White
└── ValidMoves: computed from flip constraints
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from othello_game import (
    BOARD_SIZE, ROWS, COLS, BLACK, WHITE, EMPTY,
    CORNERS, X_SQUARES, C_SQUARES, EDGES, DIRECTIONS,
    get_initial_board, get_valid_moves, make_move, is_game_over,
    get_winner, count_pieces, get_game_phase, get_all_flips,
    get_mobility, get_stability, get_frontier, opponent
)


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


class OthelloSignals(nn.Module):
    """
    Explicit statechart-derived signals for Othello.

    Computes per-move:
    - valid: Is move legal (flips at least one)?
    - flip_count: How many pieces flipped
    - is_corner: Captures a corner (very valuable)
    - is_edge: On edge (stable once filled)
    - is_x_square: X-square (dangerous, can give corner)
    - mobility_gain: Moves gained - opponent moves lost
    """

    def __init__(self):
        super().__init__()
        # Position values (learned)
        self.corner_value = 25.0
        self.edge_value = 3.0
        self.x_square_penalty = -10.0
        self.c_square_penalty = -3.0

    def compute_signals(self, board: list, player: int) -> dict:
        """Compute all signals for a single board."""
        valid = [0.0] * BOARD_SIZE
        flip_count = [0.0] * BOARD_SIZE
        is_corner = [0.0] * BOARD_SIZE
        is_edge = [0.0] * BOARD_SIZE
        is_x_square = [0.0] * BOARD_SIZE
        is_c_square = [0.0] * BOARD_SIZE

        valid_moves = get_valid_moves(board, player)

        for pos in valid_moves:
            valid[pos] = 1.0
            flips = get_all_flips(board, pos, player)
            flip_count[pos] = float(len(flips))

            if pos in CORNERS:
                is_corner[pos] = 1.0
            if pos in EDGES:
                is_edge[pos] = 1.0
            if pos in X_SQUARES:
                is_x_square[pos] = 1.0
            if pos in C_SQUARES:
                is_c_square[pos] = 1.0

        return {
            "valid": valid,
            "flip_count": flip_count,
            "is_corner": is_corner,
            "is_edge": is_edge,
            "is_x_square": is_x_square,
            "is_c_square": is_c_square,
        }

    def __call__(self, board: mx.array, turn: mx.array) -> dict:
        """Compute signals for batch."""
        B = board.shape[0]

        all_valid = []
        all_flips = []
        all_corner = []
        all_edge = []
        all_x = []
        all_c = []

        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            player = BLACK if int(turn[b].item()) == 0 else WHITE

            signals = self.compute_signals(board_list, player)
            all_valid.append(signals["valid"])
            all_flips.append(signals["flip_count"])
            all_corner.append(signals["is_corner"])
            all_edge.append(signals["is_edge"])
            all_x.append(signals["is_x_square"])
            all_c.append(signals["is_c_square"])

        return {
            "valid": mx.array(all_valid),
            "flip_count": mx.array(all_flips),
            "is_corner": mx.array(all_corner),
            "is_edge": mx.array(all_edge),
            "is_x_square": mx.array(all_x),
            "is_c_square": mx.array(all_c),
        }


class OthelloAttention(nn.Module):
    """
    Multi-head self-attention over Othello board.

    Features:
    - Position embeddings (row + col)
    - Cell embeddings (empty, black, white)
    - Phase-conditioned queries (opening/mid/end strategy differs)
    - Turn embedding
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Cell embedding
        self.cell_embed = nn.Embedding(3, embed_dim)

        # Position embeddings
        self.row_embed = nn.Embedding(ROWS, embed_dim // 2)
        self.col_embed = nn.Embedding(COLS, embed_dim // 2)

        # Phase embedding (opening, mid, end have different strategies)
        self.phase_embed = nn.Embedding(3, embed_dim)

        # Turn embedding
        self.turn_embed = nn.Embedding(2, embed_dim)

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.score_proj = nn.Linear(embed_dim, 1)

    def __call__(self, board: mx.array, turn: mx.array, phase: mx.array) -> mx.array:
        """
        Compute attention scores over board positions.

        Returns: [B, 64] position scores
        """
        B = board.shape[0]

        # Embed cells
        cell_emb = self.cell_embed(board)  # [B, 64, d]

        # Add position embeddings
        row_idx = mx.arange(BOARD_SIZE) // COLS
        col_idx = mx.arange(BOARD_SIZE) % COLS
        row_emb = self.row_embed(row_idx)  # [64, d/2]
        col_emb = self.col_embed(col_idx)  # [64, d/2]
        pos_emb = mx.concatenate([row_emb, col_emb], axis=-1)  # [64, d]

        x = cell_emb + pos_emb[None, :, :]  # [B, 64, d]

        # Phase and turn conditioning
        phase_emb = self.phase_embed(phase)  # [B, d]
        turn_emb = self.turn_embed(turn)  # [B, d]
        condition = phase_emb + turn_emb  # [B, d]

        # Compute attention (queries are conditioned)
        Q = self.q_proj(x + condition[:, None, :])  # [B, 64, d]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = mx.softmax((Q @ K.transpose(0, 2, 1)) * scale, axis=-1)
        attn_out = self.out_proj(attn @ V)  # [B, 64, d]

        # Project to scores
        scores = self.score_proj(attn_out)[:, :, 0]  # [B, 64]

        return scores


class FusedOthello(nn.Module):
    """
    Fusion of explicit statechart signals and attention.

    Strategy by phase:
    - Opening: Mobility > stability (control center, build options)
    - Midgame: Edges > corners > flips (build stable position)
    - Endgame: Flip count matters (maximize pieces)

    Hierarchical priority:
    1. Corner captures (always take)
    2. Block opponent corners (avoid giving away)
    3. Edges (stable)
    4. Strategic moves (attention + flip count)
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Explicit signals
        self.signals = OthelloSignals()

        # Learned attention
        self.attention = OthelloAttention(embed_dim, num_heads=4)

        # Phase-dependent weights
        # [corner, edge, flip, attention, x_penalty, c_penalty]
        self.phase_weights = nn.Embedding(3, 6)
        # Initialize with strategic priors
        # Opening: mobility matters, avoid corners (game starts)
        # Midgame: edges and corners are key
        # Endgame: piece count (flips) matter most
        self.phase_weights.weight = mx.array([
            [20.0, 2.0, 0.5, 3.0, -8.0, -2.0],  # Opening
            [30.0, 5.0, 1.0, 2.0, -10.0, -3.0],  # Midgame
            [25.0, 3.0, 2.0, 1.0, -5.0, -1.0],   # Endgame
        ])

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """Compute move logits."""
        B = board.shape[0]

        # Compute game phase for each board
        phases = []
        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            phases.append(get_game_phase(board_list))
        phase = mx.array(phases)

        # Get explicit signals
        sig = self.signals(board, turn)
        valid = sig["valid"]
        flip_count = sig["flip_count"]
        is_corner = sig["is_corner"]
        is_edge = sig["is_edge"]
        is_x = sig["is_x_square"]
        is_c = sig["is_c_square"]

        # Get attention scores
        attn_scores = self.attention(board, turn, phase)

        # Get phase weights
        weights = self.phase_weights(phase)  # [B, 6]

        # Combine signals with phase-dependent weights
        logits = (
            weights[:, 0:1] * is_corner +
            weights[:, 1:2] * is_edge +
            weights[:, 2:3] * flip_count +
            weights[:, 3:4] * attn_scores +
            weights[:, 4:5] * is_x +
            weights[:, 5:6] * is_c
        )

        # Mask invalid moves
        logits = logits + (1 - valid) * (-1e9)

        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        return mx.softmax(self(board, turn), axis=-1)


class PureAttentionOthello(nn.Module):
    """Pure attention model for comparison."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.attention = OthelloAttention(embed_dim, num_heads=4)

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        B = board.shape[0]

        # Compute phase
        phases = []
        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            phases.append(get_game_phase(board_list))
        phase = mx.array(phases)

        logits = self.attention(board, turn, phase)

        # Mask invalid
        for b in range(B):
            board_list = [int(x) for x in board[b].tolist()]
            player = BLACK if int(turn[b].item()) == 0 else WHITE
            valid = get_valid_moves(board_list, player)
            for pos in range(BOARD_SIZE):
                if pos not in valid:
                    logits = logits.at[b, pos].add(-1e9)

        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        return mx.softmax(self(board, turn), axis=-1)


def play_vs_random(model, model_is_black: bool = True) -> tuple:
    """Play model vs random. Returns (result, final_score_diff)."""
    board = get_initial_board()
    model_player = BLACK if model_is_black else WHITE

    current_player = BLACK
    passes = 0

    while not is_game_over(board) and passes < 2:
        valid = get_valid_moves(board, current_player)

        if not valid:
            passes += 1
            current_player = opponent(current_player)
            continue
        else:
            passes = 0

        turn = 0 if current_player == BLACK else 1

        if current_player == model_player:
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])
            probs = model.get_move_probs(board_arr, turn_arr)[0]
            move = int(mx.argmax(probs).item())
            if move not in valid:
                move = valid[0]
        else:
            move = valid[mx.random.randint(0, len(valid), ()).item()]

        board = make_move(board, move, current_player)
        current_player = opponent(current_player)

    # Get result
    winner = get_winner(board)
    black, white, _ = count_pieces(board)

    if winner == model_player:
        return 1, abs(black - white)
    elif winner != 0:
        return -1, abs(black - white)
    return 0, 0


def evaluate(model, num_games: int = 100) -> dict:
    """Evaluate model vs random."""
    wins = losses = draws = 0
    total_margin = 0

    for i in range(num_games):
        result, margin = play_vs_random(model, model_is_black=(i % 2 == 0))
        if result == 1:
            wins += 1
            total_margin += margin
        elif result == -1:
            losses += 1
            total_margin -= margin
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "avg_margin": total_margin / num_games,
    }


def train_model(model, num_episodes: int = 500, lr: float = 0.01) -> list:
    """Train via REINFORCE self-play."""
    optimizer = optim.Adam(learning_rate=lr)
    history = []

    print("Training...")
    print("-" * 50)

    for ep in range(num_episodes):
        board = get_initial_board()
        log_probs_black = []
        log_probs_white = []

        current_player = BLACK
        passes = 0

        while not is_game_over(board) and passes < 2:
            valid = get_valid_moves(board, current_player)

            if not valid:
                passes += 1
                current_player = opponent(current_player)
                continue
            else:
                passes = 0

            turn = 0 if current_player == BLACK else 1
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])

            probs = model.get_move_probs(board_arr, turn_arr)[0]

            # Sample with temperature
            temp = max(0.3, 1.0 - ep / num_episodes)
            logits = mx.log(probs + 1e-10) / temp
            sample_probs = mx.softmax(logits, axis=-1)

            cumsum = mx.cumsum(sample_probs)
            rand = mx.random.uniform(shape=())
            move = int(mx.sum(cumsum < rand).item())

            if move >= BOARD_SIZE or move not in valid:
                move = int(mx.argmax(probs).item())
                if move not in valid:
                    move = valid[0]

            log_p = mx.log(probs[move] + 1e-10)
            if current_player == BLACK:
                log_probs_black.append(log_p)
            else:
                log_probs_white.append(log_p)

            board = make_move(board, move, current_player)
            current_player = opponent(current_player)

        # Compute reward
        winner = get_winner(board)
        if winner == BLACK:
            r_black, r_white = 1.0, -1.0
        elif winner == WHITE:
            r_black, r_white = -1.0, 1.0
        else:
            r_black, r_white = 0.0, 0.0

        # Update
        def loss_fn(m):
            loss = mx.array(0.0)
            for lp in log_probs_black:
                loss = loss - r_black * lp
            for lp in log_probs_white:
                loss = loss - r_white * lp
            return loss

        if log_probs_black or log_probs_white:
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        # Evaluate periodically
        if (ep + 1) % 100 == 0:
            result = evaluate(model, num_games=50)
            history.append(result["win_rate"])
            print(f"Episode {ep+1:4d}: win_rate={result['win_rate']*100:.1f}%, "
                  f"W/L/D={result['wins']}/{result['losses']}/{result['draws']}, "
                  f"margin={result['avg_margin']:.1f}")

    print("-" * 50)
    return history


def run_experiment():
    """Compare fused vs pure attention on Othello."""
    print("=" * 70)
    print("OTHELLO: FUSED VS PURE ATTENTION")
    print("=" * 70)
    print()
    print(f"Board: {ROWS}x{COLS} = {BOARD_SIZE} cells")
    print(f"State space: ~10^28")
    print()

    mx.random.seed(42)

    # Create models
    fused = FusedOthello(embed_dim=64)
    pure = PureAttentionOthello(embed_dim=64)

    fused_params = count_params(fused)
    pure_params = count_params(pure)

    print(f"Fused model params: {fused_params:,}")
    print(f"Pure attention params: {pure_params:,}")
    print()

    # Evaluate untrained
    print("UNTRAINED evaluation (50 games each)...")
    fused_untrained = evaluate(fused, 50)
    pure_untrained = evaluate(pure, 50)
    print(f"Fused untrained: {fused_untrained['win_rate']*100:.1f}%, "
          f"W/L/D={fused_untrained['wins']}/{fused_untrained['losses']}/{fused_untrained['draws']}")
    print(f"Pure untrained:  {pure_untrained['win_rate']*100:.1f}%, "
          f"W/L/D={pure_untrained['wins']}/{pure_untrained['losses']}/{pure_untrained['draws']}")
    print()

    # Train fused
    print("Training FUSED model (500 episodes)...")
    train_model(fused, num_episodes=500, lr=0.01)
    print()

    # Train pure
    print("Training PURE ATTENTION model (500 episodes)...")
    train_model(pure, num_episodes=500, lr=0.01)
    print()

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION (100 games each)")
    print("=" * 70)
    print()

    fused_final = evaluate(fused, 100)
    pure_final = evaluate(pure, 100)

    print(f"{'Model':<20} | {'Win Rate':<10} | {'W/L/D':<12} | {'Margin':<8} | {'Params':<10}")
    print("-" * 70)
    print(f"{'Random baseline':<20} | {'~35%':<10} | {'-':<12} | {'-':<8} | {'-':<10}")

    pure_wld = f"{pure_final['wins']}/{pure_final['losses']}/{pure_final['draws']}"
    fused_wld = f"{fused_final['wins']}/{fused_final['losses']}/{fused_final['draws']}"

    print(f"{'Pure Attention':<20} | {pure_final['win_rate']*100:.1f}%{'':<5} | "
          f"{pure_wld:<12} | {pure_final['avg_margin']:.1f}{'':<4} | {pure_params:,}")
    print(f"{'FUSED (ours)':<20} | {fused_final['win_rate']*100:.1f}%{'':<5} | "
          f"{fused_wld:<12} | {fused_final['avg_margin']:.1f}{'':<4} | {fused_params:,}")
    print()

    improvement = (fused_final['win_rate'] - pure_final['win_rate']) * 100
    print(f"Improvement from fusion: {improvement:+.1f}%")

    if fused_final['losses'] < pure_final['losses']:
        print(f"Fusion reduces losses: {fused_final['losses']} vs {pure_final['losses']}")

    print()
    print("=" * 70)

    return {"fused": fused_final, "pure": pure_final}


if __name__ == "__main__":
    run_experiment()
