"""
Fused Attention-Statechart Model for TicTacToe.

Hypothesis: Combining explicit statechart structure with implicit attention
patterns should beat pure attention (98% Transformer).

Fusion points:
1. State-conditioned attention: Game state modifies attention queries
2. Guard-gated attention: Statechart guards mask attention
3. Explicit + Implicit: Rule-based signals + learned patterns

Target: Beat 98% Transformer win rate.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import os
import sys

_file_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(os.path.dirname(_file_dir))
sys.path.insert(0, _ml_dir)

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


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
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0


class StatechartSignals(nn.Module):
    """
    Explicit statechart-derived signals.

    Computes:
    - Winning moves (immediate win available)
    - Blocking moves (must block opponent win)
    - Fork opportunities (create two winning threats)
    """

    def __init__(self):
        super().__init__()

    def get_winning_moves(self, board: mx.array, player: int) -> mx.array:
        """Find cells that would win for player."""
        B = board.shape[0]
        wins = mx.zeros((B, 9))

        for line in WINNING_LINES:
            i, j, k = line
            for empty_idx, (o1, o2) in [(i, (j, k)), (j, (i, k)), (k, (i, j))]:
                is_empty = (board[:, empty_idx] == 0).astype(mx.float32)
                has_o1 = (board[:, o1] == player).astype(mx.float32)
                has_o2 = (board[:, o2] == player).astype(mx.float32)
                is_win = is_empty * has_o1 * has_o2
                wins = wins.at[:, empty_idx].add(is_win)

        return mx.minimum(wins, 1.0)

    def get_fork_moves(self, board: mx.array, player: int) -> mx.array:
        """Find cells that create a fork (two winning threats)."""
        B = board.shape[0]
        forks = mx.zeros((B, 9))

        for cell in range(9):
            # Simulate placing piece
            test_boards = []
            for b in range(B):
                test = list(board[b].tolist())
                if test[cell] == 0:
                    test[cell] = player
                test_boards.append(test)

            test_arr = mx.array(test_boards)
            wins_after = self.get_winning_moves(test_arr, player)

            # Count winning moves after placement
            win_count = mx.sum(wins_after, axis=1)
            is_fork = (win_count >= 2).astype(mx.float32)
            is_empty = (board[:, cell] == 0).astype(mx.float32)

            forks = forks.at[:, cell].add(is_fork * is_empty)

        return forks

    def get_opp_fork_moves(self, board: mx.array, player: int) -> mx.array:
        """Find moves that BLOCK opponent forks."""
        B = board.shape[0]
        opp = 2 if player == 1 else 1
        blocks = mx.zeros((B, 9))

        for cell in range(9):
            # Simulate opponent placing piece
            test_boards = []
            for b in range(B):
                test = list(board[b].tolist())
                if test[cell] == 0:
                    test[cell] = opp
                test_boards.append(test)

            test_arr = mx.array(test_boards)
            wins_after = self.get_winning_moves(test_arr, opp)

            # Count winning moves after placement
            win_count = mx.sum(wins_after, axis=1)
            is_fork = (win_count >= 2).astype(mx.float32)
            is_empty = (board[:, cell] == 0).astype(mx.float32)

            blocks = blocks.at[:, cell].add(is_fork * is_empty)

        return blocks

    def __call__(self, board: mx.array, turn: mx.array) -> dict:
        """Compute all statechart signals."""
        B = board.shape[0]

        # Determine player
        is_x = (turn == 0).astype(mx.float32)[:, None]
        is_o = (turn == 1).astype(mx.float32)[:, None]

        # My winning moves
        x_wins = self.get_winning_moves(board, 1)
        o_wins = self.get_winning_moves(board, 2)
        my_wins = is_x * x_wins + is_o * o_wins

        # Opponent winning moves (blocking targets)
        opp_wins = is_x * o_wins + is_o * x_wins

        # My forks (create two threats)
        x_forks = self.get_fork_moves(board, 1)
        o_forks = self.get_fork_moves(board, 2)
        my_forks = is_x * x_forks + is_o * o_forks

        # Opponent fork blocks
        x_block_forks = self.get_opp_fork_moves(board, 1)
        o_block_forks = self.get_opp_fork_moves(board, 2)
        opp_forks = is_x * o_block_forks + is_o * x_block_forks

        # Legal moves
        legal = (board == 0).astype(mx.float32)

        return {
            "my_wins": my_wins,
            "opp_wins": opp_wins,
            "my_forks": my_forks,
            "opp_forks": opp_forks,
            "legal": legal,
        }


class StateConditionedAttention(nn.Module):
    """
    Attention where the game state modifies what we look for.

    State embedding is added to queries, biasing attention
    toward state-relevant patterns.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Cell embeddings
        self.cell_embed = nn.Embedding(3, embed_dim)

        # Position embeddings (center, corners, edges have different values)
        self.pos_embed = nn.Embedding(9, embed_dim)

        # State embedding (2 states: X's turn, O's turn)
        self.state_embed = nn.Embedding(2, embed_dim)

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, board: mx.array, turn: mx.array) -> tuple:
        """
        Compute state-conditioned attention over board.

        Returns:
            (output, attn_weights)
        """
        B = board.shape[0]

        # Embed cells
        cell_emb = self.cell_embed(board)  # [B, 9, d]

        # Add position embedding
        positions = mx.arange(9)
        pos_emb = self.pos_embed(positions)  # [9, d]
        x = cell_emb + pos_emb[None, :, :]  # [B, 9, d]

        # Get state embedding
        state_emb = self.state_embed(turn)  # [B, d]

        # Queries are state-conditioned
        Q = self.q_proj(x + state_emb[:, None, :])  # [B, 9, d]
        K = self.k_proj(x)  # [B, 9, d]
        V = self.v_proj(x)  # [B, 9, d]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_scores = (Q @ K.transpose(0, 2, 1)) * scale  # [B, 9, 9]
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Attend
        attn_out = attn_weights @ V  # [B, 9, d]
        output = self.out_proj(attn_out)

        return output, attn_weights


class FusedModel(nn.Module):
    """
    Fusion of statechart signals and attention patterns.

    Combines:
    1. Explicit signals (wins, blocks) from statechart
    2. Implicit patterns from self-attention
    3. Learned combination weights
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Statechart signals (explicit, rule-based)
        self.statechart = StatechartSignals()

        # State-conditioned attention (implicit, learned)
        self.attention = StateConditionedAttention(embed_dim, num_heads=2)

        # Project attention output to move scores
        self.attn_to_score = nn.Linear(embed_dim, 1)

        # Learnable signal weights
        # [my_wins, opp_wins, attn_pattern, position_bias]
        self.signal_weights = nn.Linear(4, 1, bias=False)

        # Initialize with good priors
        self.signal_weights.weight = mx.array([[10.0, 8.0, 2.0, 0.5]])

        # Position bias (center > corners > edges)
        self.position_bias = mx.array([0.5, 0.25, 0.5, 0.25, 1.0, 0.25, 0.5, 0.25, 0.5])

    def __call__(self, board: mx.array, turn: mx.array) -> mx.array:
        """
        Compute move logits by fusing statechart + attention.

        PRIORITY ORDER (hard-coded for correctness):
        1. Winning move (if available, MUST take)
        2. Blocking move (if opponent can win, MUST block)
        3. Block opponent fork (prevent double threat)
        4. Create own fork (if possible)
        5. Attention-based strategic move

        Args:
            board: [B, 9] cell values
            turn: [B] whose turn (0=X, 1=O)

        Returns:
            [B, 9] move logits
        """
        B = board.shape[0]

        # 1. Get explicit statechart signals
        signals = self.statechart(board, turn)
        my_wins = signals["my_wins"]      # [B, 9]
        opp_wins = signals["opp_wins"]    # [B, 9]
        my_forks = signals["my_forks"]    # [B, 9]
        opp_forks = signals["opp_forks"]  # [B, 9]
        legal = signals["legal"]          # [B, 9]

        # 2. Get implicit attention patterns for strategic moves
        attn_out, attn_weights = self.attention(board, turn)  # [B, 9, d]
        attn_scores = self.attn_to_score(attn_out)[:, :, 0]   # [B, 9]

        # 3. Position bias
        pos_bias = mx.broadcast_to(self.position_bias[None, :], (B, 9))

        # 4. HIERARCHICAL fusion
        # Priority: win > block > block_fork > create_fork > strategic
        has_win = mx.sum(my_wins, axis=-1, keepdims=True) > 0
        has_block = mx.sum(opp_wins, axis=-1, keepdims=True) > 0
        has_opp_fork = mx.sum(opp_forks, axis=-1, keepdims=True) > 0
        has_my_fork = mx.sum(my_forks, axis=-1, keepdims=True) > 0

        # Strategic = attention + position + slight fork preference
        strategic = attn_scores * 2.0 + pos_bias * 0.5 + my_forks * 3.0

        logits = mx.where(
            has_win,
            my_wins * 100.0,
            mx.where(
                has_block,
                opp_wins * 50.0,
                mx.where(
                    has_opp_fork,
                    opp_forks * 30.0,  # Block opponent fork
                    mx.where(
                        has_my_fork,
                        my_forks * 20.0,  # Create our fork
                        strategic
                    )
                )
            )
        )

        # 5. Mask illegal moves
        logits = logits + (1 - legal) * (-1e9)

        return logits

    def get_move_probs(self, board: mx.array, turn: mx.array) -> mx.array:
        """Get move probabilities."""
        logits = self(board, turn)
        return mx.softmax(logits, axis=-1)


def play_vs_random(model: FusedModel, model_is_x: bool = True) -> int:
    """Play model vs random. Returns +1 win, -1 loss, 0 draw."""
    board = [0] * 9
    model_player = 1 if model_is_x else 2

    for move_num in range(9):
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            break

        current_player = 1 if move_num % 2 == 0 else 2
        turn = move_num % 2

        if current_player == model_player:
            board_arr = mx.array([board])
            turn_arr = mx.array([turn])
            probs = model.get_move_probs(board_arr, turn_arr)[0]
            move = int(mx.argmax(probs).item())
            if move not in empty:
                move = empty[0]
        else:
            move = empty[mx.random.randint(0, len(empty), ()).item()]

        board[move] = current_player

        winner = check_winner(board)
        if winner == model_player:
            return 1
        elif winner != 0:
            return -1

    return 0


def evaluate(model: FusedModel, num_games: int = 500) -> dict:
    """Evaluate win rate vs random."""
    wins = losses = draws = 0

    for i in range(num_games):
        result = play_vs_random(model, model_is_x=(i % 2 == 0))
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


def train_fused(model: FusedModel, num_episodes: int = 2000, lr: float = 0.01) -> dict:
    """Train fused model via REINFORCE with self-play."""
    import mlx.optimizers as optim

    optimizer = optim.Adam(learning_rate=lr)
    history = {"episode": [], "win_rate": []}

    print("Training fused model...")
    print("-" * 50)

    for ep in range(num_episodes):
        # Play a game
        board = [0] * 9
        log_probs_x = []
        log_probs_o = []

        for move_num in range(9):
            empty = [i for i in range(9) if board[i] == 0]
            if not empty:
                break

            turn = move_num % 2
            current_player = 1 if turn == 0 else 2

            board_arr = mx.array([board])
            turn_arr = mx.array([turn])

            probs = model.get_move_probs(board_arr, turn_arr)[0]

            # Sample from distribution (with temperature for exploration)
            temp = max(0.5, 1.0 - ep / num_episodes)  # Anneal temperature
            logits = mx.log(probs + 1e-10) / temp
            sample_probs = mx.softmax(logits, axis=-1)

            # Sample move
            cumsum = mx.cumsum(sample_probs)
            rand = mx.random.uniform(shape=())
            move = int(mx.sum(cumsum < rand).item())
            if move >= 9 or board[move] != 0:
                # Fallback to greedy
                move = int(mx.argmax(probs).item())
                if move not in empty:
                    move = empty[0]

            log_p = mx.log(probs[move] + 1e-10)
            if turn == 0:
                log_probs_x.append(log_p)
            else:
                log_probs_o.append(log_p)

            board[move] = current_player

            winner = check_winner(board)
            if winner != 0:
                break

        # Compute reward
        winner = check_winner(board)
        if winner == 1:
            reward_x, reward_o = 1.0, -1.0
        elif winner == 2:
            reward_x, reward_o = -1.0, 1.0
        else:
            reward_x, reward_o = 0.0, 0.0

        # REINFORCE update
        def loss_fn(m):
            loss = mx.array(0.0)
            for lp in log_probs_x:
                loss = loss - reward_x * lp
            for lp in log_probs_o:
                loss = loss - reward_o * lp
            return loss

        if log_probs_x or log_probs_o:
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        # Evaluate periodically
        if (ep + 1) % 500 == 0:
            eval_result = evaluate(model, num_games=100)
            history["episode"].append(ep + 1)
            history["win_rate"].append(eval_result["win_rate"])
            print(f"Episode {ep+1:4d}: win_rate={eval_result['win_rate']*100:.1f}%, "
                  f"W/L/D={eval_result['wins']}/{eval_result['losses']}/{eval_result['draws']}")

    print("-" * 50)
    return history


def run_experiment():
    """Run the fusion experiment."""
    print("=" * 70)
    print("FUSED ATTENTION-STATECHART MODEL")
    print("=" * 70)
    print()
    print("Hypothesis: Explicit statechart signals + implicit attention patterns")
    print("            should beat pure attention (98% Transformer).")
    print()

    mx.random.seed(42)

    # Create model
    model = FusedModel(embed_dim=32)
    params = count_params(model)
    print(f"Parameters: {params:,}")
    print()

    # Evaluate untrained
    print("UNTRAINED model vs random (200 games)...")
    untrained = evaluate(model, num_games=200)
    print(f"Win rate: {untrained['win_rate']*100:.1f}%, W/L/D: {untrained['wins']}/{untrained['losses']}/{untrained['draws']}")
    print()

    # Train
    train_fused(model, num_episodes=2000, lr=0.01)
    print()

    # Evaluate trained
    print("Evaluating TRAINED model vs random (500 games)...")
    start = time.perf_counter()
    result = evaluate(model, num_games=500)
    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Win rate: {result['win_rate']*100:.1f}%")
    print(f"W/L/D: {result['wins']}/{result['losses']}/{result['draws']}")
    print(f"Time: {elapsed:.2f}s")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Model':<25} | {'Win Rate':<10} | {'Params':<10}")
    print("-" * 55)
    print(f"{'Random baseline':<25} | {'~35%':<10} | {'-':<10}")
    print(f"{'MLP':<25} | {'67.5%':<10} | {'12,761':<10}")
    print(f"{'Statechart (rules)':<25} | {'86.2%':<10} | {'23,979':<10}")
    print(f"{'Transformer':<25} | {'98.0%':<10} | {'3,657':<10}")
    wr = f"{result['win_rate']*100:.1f}%"
    print(f"{'FUSED (trained)':<25} | {wr:<10} | {params:<10}")
    print()

    beats_transformer = result['win_rate'] > 0.98
    print(f"Beats Transformer (98%): {'YES!' if beats_transformer else 'No'}")

    if result['losses'] == 0:
        print("ZERO LOSSES against random!")

    print()
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_experiment()
