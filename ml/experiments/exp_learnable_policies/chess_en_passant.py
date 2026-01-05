"""
Learning En Passant from Self-Play

Goal: Evolve an EPHEMERAL STATE mechanism that enables en passant captures
only on the move immediately following a double pawn advance.

En Passant Rule:
- A pawn that advances two squares can be captured "in passing"
- But ONLY on the very next move
- The capture window is ephemeral - it expires after one turn

What we're learning:
- Need to track "pawn just double-advanced" (ephemeral flag)
- The flag must reset after opponent's move
- The capturing pawn must be adjacent and on correct rank
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# SIMPLIFIED CHESS BOARD FOR EN PASSANT LEARNING
# =============================================================================

class Piece(Enum):
    EMPTY = 0
    WHITE_PAWN = 1
    BLACK_PAWN = 2
    # Other pieces not needed for en passant learning


@dataclass
class ChessBoard:
    """Simplified chess board for pawn-only en passant testing."""

    # 8x8 board, indexed [rank][file] where rank 0 = rank 1 (white's back rank)
    squares: List[List[Piece]]
    turn: int  # 1 = white, -1 = black

    # En passant target (file where en passant is possible, or None)
    ep_target: Optional[int] = None

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to starting position with just pawns."""
        self.squares = [[Piece.EMPTY for _ in range(8)] for _ in range(8)]

        # White pawns on rank 2 (index 1)
        for f in range(8):
            self.squares[1][f] = Piece.WHITE_PAWN

        # Black pawns on rank 7 (index 6)
        for f in range(8):
            self.squares[6][f] = Piece.BLACK_PAWN

        self.turn = 1  # White to move
        self.ep_target = None

    def copy(self):
        new = ChessBoard.__new__(ChessBoard)
        new.squares = [[self.squares[r][f] for f in range(8)] for r in range(8)]
        new.turn = self.turn
        new.ep_target = self.ep_target
        return new

    def get(self, rank: int, file: int) -> Piece:
        if 0 <= rank < 8 and 0 <= file < 8:
            return self.squares[rank][file]
        return Piece.EMPTY

    def set(self, rank: int, file: int, piece: Piece):
        if 0 <= rank < 8 and 0 <= file < 8:
            self.squares[rank][file] = piece

    def is_white_turn(self) -> bool:
        return self.turn == 1

    def get_pawn_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], bool]]:
        """
        Get all legal pawn moves.
        Returns list of (from_square, to_square, is_en_passant).
        """
        moves = []
        pawn = Piece.WHITE_PAWN if self.is_white_turn() else Piece.BLACK_PAWN
        direction = 1 if self.is_white_turn() else -1
        start_rank = 1 if self.is_white_turn() else 6
        ep_rank = 4 if self.is_white_turn() else 3  # Rank where en passant happens

        for r in range(8):
            for f in range(8):
                if self.get(r, f) != pawn:
                    continue

                # Single advance
                new_r = r + direction
                if 0 <= new_r < 8 and self.get(new_r, f) == Piece.EMPTY:
                    moves.append(((r, f), (new_r, f), False))

                    # Double advance from starting rank
                    if r == start_rank:
                        new_r2 = r + 2 * direction
                        if self.get(new_r2, f) == Piece.EMPTY:
                            moves.append(((r, f), (new_r2, f), False))

                # Captures (including en passant)
                for df in [-1, 1]:
                    new_f = f + df
                    if not (0 <= new_f < 8):
                        continue

                    new_r = r + direction
                    if not (0 <= new_r < 8):
                        continue

                    target = self.get(new_r, new_f)

                    # Normal capture
                    if target != Piece.EMPTY:
                        enemy_pawn = Piece.BLACK_PAWN if self.is_white_turn() else Piece.WHITE_PAWN
                        if target == enemy_pawn:
                            moves.append(((r, f), (new_r, new_f), False))

                    # En passant capture
                    elif self.ep_target == new_f and r == ep_rank:
                        moves.append(((r, f), (new_r, new_f), True))

        return moves

    def make_move(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int], is_ep: bool = False):
        """Make a move on the board."""
        r1, f1 = from_sq
        r2, f2 = to_sq

        piece = self.get(r1, f1)
        self.set(r1, f1, Piece.EMPTY)
        self.set(r2, f2, piece)

        # Handle en passant capture
        if is_ep:
            # Remove the captured pawn
            captured_rank = r1  # Same rank as capturing pawn started
            self.set(captured_rank, f2, Piece.EMPTY)

        # Check if this was a double pawn advance
        if abs(r2 - r1) == 2:
            self.ep_target = f1  # En passant possible on this file
        else:
            self.ep_target = None

        self.turn *= -1

    def to_string(self) -> str:
        """String representation."""
        symbols = {
            Piece.EMPTY: '.',
            Piece.WHITE_PAWN: 'P',
            Piece.BLACK_PAWN: 'p',
        }
        lines = []
        for r in range(7, -1, -1):
            row = ''.join(symbols[self.get(r, f)] for f in range(8))
            lines.append(f"{r+1} {row}")
        lines.append("  abcdefgh")
        if self.ep_target is not None:
            lines.append(f"EP target: {chr(ord('a') + self.ep_target)}")
        return '\n'.join(lines)


# =============================================================================
# EN PASSANT ENVIRONMENT
# =============================================================================

class EnPassantEnvironment:
    """Environment for learning en passant detection."""

    def __init__(self):
        self.board = ChessBoard()
        self.move_count = 0
        self._last_double_advance: Optional[int] = None  # File of last double advance

    def reset(self):
        self.board.reset()
        self.move_count = 0
        self._last_double_advance = None

        # Randomize starting position a bit for variety
        for _ in range(random.randint(0, 10)):
            moves = self.board.get_pawn_moves()
            if moves:
                from_sq, to_sq, is_ep = random.choice(moves)
                self.board.make_move(from_sq, to_sq, is_ep)

        return self._get_obs()

    def _get_obs(self):
        return {
            'board': self.board.copy(),
            'turn': self.board.turn,
            'ep_target': self.board.ep_target,
        }

    def get_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], bool]]:
        """Get legal moves (includes en passant when legal)."""
        return self.board.get_pawn_moves()

    def get_legal_moves_without_ep(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], bool]]:
        """Get legal moves ignoring en passant (for learning)."""
        # Temporarily disable en passant
        saved_ep = self.board.ep_target
        self.board.ep_target = None
        moves = self.board.get_pawn_moves()
        self.board.ep_target = saved_ep
        return moves

    def step(self, move: Tuple):
        from_sq, to_sq, is_ep = move
        self.board.make_move(from_sq, to_sq, is_ep)
        self.move_count += 1
        done = self.move_count > 100 or not self.board.get_pawn_moves()
        return self._get_obs(), 0.0, done, {}

    def check_en_passant_available(self) -> bool:
        """Check if en passant is currently available."""
        return self.board.ep_target is not None

    def get_en_passant_moves(self) -> List[Tuple]:
        """Get just the en passant moves."""
        all_moves = self.board.get_pawn_moves()
        return [m for m in all_moves if m[2]]  # is_ep = True

    def get_context(self) -> Dict:
        return {
            'board': self.board.copy(),
            'turn': self.board.turn,
            'ep_target': self.board.ep_target,
            'last_double_advance': self._last_double_advance,
        }


# =============================================================================
# EN PASSANT GENOME
# =============================================================================

@dataclass
class EnPassantGenome:
    """
    Genome for learning en passant detection.

    Key insight: We need an EPHEMERAL flag that:
    - Is set when opponent pawn double-advances
    - Resets after one move
    - Enables capture on adjacent file
    """
    # Feature flags
    track_double_advance: bool = False
    track_adjacent_pawn: bool = False
    use_ephemeral_flag: bool = False
    check_correct_rank: bool = False

    # Weights
    w_double_advance: float = 0.0
    w_adjacent: float = 0.0
    w_ephemeral: float = 0.0
    w_rank: float = 0.0
    bias: float = 0.0

    # Fitness
    fitness: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def copy(self):
        new = EnPassantGenome(
            track_double_advance=self.track_double_advance,
            track_adjacent_pawn=self.track_adjacent_pawn,
            use_ephemeral_flag=self.use_ephemeral_flag,
            check_correct_rank=self.check_correct_rank,
            w_double_advance=self.w_double_advance,
            w_adjacent=self.w_adjacent,
            w_ephemeral=self.w_ephemeral,
            w_rank=self.w_rank,
            bias=self.bias,
        )
        new.fitness = self.fitness
        new.precision = self.precision
        new.recall = self.recall
        new.f1 = self.f1
        return new

    def predict_ep_available(self, ctx: Dict, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Predict if en passant is available for this move."""
        features = self._extract_features(ctx, from_sq, to_sq)

        logit = self.bias
        if self.track_double_advance:
            logit += self.w_double_advance * features['double_advance']
        if self.track_adjacent_pawn:
            logit += self.w_adjacent * features['adjacent']
        if self.use_ephemeral_flag:
            logit += self.w_ephemeral * features['ephemeral']
        if self.check_correct_rank:
            logit += self.w_rank * features['correct_rank']

        return logit > 0

    def _extract_features(self, ctx: Dict, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> Dict[str, float]:
        """Extract features for en passant prediction."""
        board = ctx.get('board')
        ep_target = ctx.get('ep_target')

        r1, f1 = from_sq
        r2, f2 = to_sq

        # Was there a double advance last turn?
        double_advance = 1.0 if ep_target is not None else 0.0

        # Is capturing pawn adjacent to target file?
        adjacent = 1.0 if ep_target is not None and abs(f1 - ep_target) == 1 else 0.0

        # Is this an ephemeral opportunity (would be reset)?
        ephemeral = double_advance  # Same as double_advance for now

        # Is pawn on correct rank for en passant?
        turn = ctx.get('turn', 1)
        ep_rank = 4 if turn == 1 else 3
        correct_rank = 1.0 if r1 == ep_rank else 0.0

        return {
            'double_advance': double_advance,
            'adjacent': adjacent,
            'ephemeral': ephemeral,
            'correct_rank': correct_rank,
        }


# =============================================================================
# EN PASSANT EVOLVER
# =============================================================================

class EnPassantEvolver:
    """Evolves en passant detection from self-play."""

    def __init__(self, population_size: int = 30, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[EnPassantGenome] = []
        self.best_genome: Optional[EnPassantGenome] = None
        self.generation = 0

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            genome = EnPassantGenome(
                track_double_advance=random.random() > 0.5,
                track_adjacent_pawn=random.random() > 0.5,
                use_ephemeral_flag=random.random() > 0.5,
                check_correct_rank=random.random() > 0.5,
                w_double_advance=random.gauss(0, 3),
                w_adjacent=random.gauss(0, 3),
                w_ephemeral=random.gauss(0, 3),
                w_rank=random.gauss(0, 3),
                bias=random.gauss(0, 3),
            )
            self.population.append(genome)

    def evaluate_genome(self, genome: EnPassantGenome) -> float:
        """Evaluate genome on en passant detection."""
        env = EnPassantEnvironment()

        tp, fp, tn, fn = 0, 0, 0, 0

        for _ in range(50):  # Games
            env.reset()

            for _ in range(50):  # Moves per game
                moves = env.get_legal_moves()
                if not moves:
                    break

                ctx = env.get_context()

                # Check each move
                for move in moves[:5]:
                    from_sq, to_sq, is_ep = move
                    actual_ep = is_ep
                    pred_ep = genome.predict_ep_available(ctx, from_sq, to_sq)

                    if actual_ep and pred_ep:
                        tp += 1
                    elif actual_ep and not pred_ep:
                        fn += 1
                    elif not actual_ep and pred_ep:
                        fp += 1
                    else:
                        tn += 1

                # Make random move
                move = random.choice(moves)
                _, _, done, _ = env.step(move)
                if done:
                    break

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        genome.precision = precision
        genome.recall = recall
        genome.f1 = f1
        genome.fitness = f1

        return f1

    def mutate(self, genome: EnPassantGenome) -> EnPassantGenome:
        new = genome.copy()
        if random.random() < self.mutation_rate:
            choice = random.randint(0, 4)
            if choice == 0:
                new.track_double_advance = not new.track_double_advance
            elif choice == 1:
                new.track_adjacent_pawn = not new.track_adjacent_pawn
            elif choice == 2:
                new.use_ephemeral_flag = not new.use_ephemeral_flag
            elif choice == 3:
                new.check_correct_rank = not new.check_correct_rank
            else:
                attr = random.choice(['w_double_advance', 'w_adjacent', 'w_ephemeral', 'w_rank', 'bias'])
                setattr(new, attr, getattr(new, attr) + random.gauss(0, 1))
        return new

    def evolve(self, n_generations: int = 50, verbose: bool = True) -> EnPassantGenome:
        if not self.population:
            self.initialize_population()

        for gen in range(n_generations):
            # Evaluate
            for g in self.population:
                self.evaluate_genome(g)

            # Sort
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].copy()

            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                best = self.population[0]
                print(f"Gen {gen:3d}: F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
                print(f"         double={best.track_double_advance}, adj={best.track_adjacent_pawn}, "
                      f"eph={best.use_ephemeral_flag}, rank={best.check_correct_rank}")

            # Create new population
            new_pop = [self.population[0].copy()]  # Elitism
            while len(new_pop) < self.population_size:
                parent = random.choice(self.population[:10])
                child = self.mutate(parent)
                new_pop.append(child)

            self.population = new_pop
            self.generation += 1

        return self.best_genome


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("LEARNING EN PASSANT FROM SELF-PLAY")
    print("=" * 70)
    print()
    print("Goal: Evolve an EPHEMERAL STATE mechanism for en passant")
    print()
    print("En Passant Rule:")
    print("- When a pawn double-advances, adjacent enemy pawns can capture it")
    print("- But ONLY on the very next move (ephemeral window)")
    print()

    evolver = EnPassantEvolver(population_size=30, mutation_rate=0.4)

    print("-" * 40)
    print("EVOLVING...")
    print("-" * 40)

    best = evolver.evolve(n_generations=50, verbose=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
Learned En Passant Detection:

Features Discovered:
  - Track double advance:  {best.track_double_advance}
  - Track adjacent pawn:   {best.track_adjacent_pawn}
  - Use ephemeral flag:    {best.use_ephemeral_flag}
  - Check correct rank:    {best.check_correct_rank}

Learned Weights:
  - w_double_advance: {best.w_double_advance:+.2f}
  - w_adjacent:       {best.w_adjacent:+.2f}
  - w_ephemeral:      {best.w_ephemeral:+.2f}
  - w_rank:           {best.w_rank:+.2f}
  - bias:             {best.bias:+.2f}

Performance:
  - Precision: {best.precision:.1%}
  - Recall:    {best.recall:.1%}
  - F1 Score:  {best.f1:.1%}

Key Insight:
  En passant requires EPHEMERAL STATE - a flag that exists for exactly
  one turn and then expires. Evolution should discover:
  1. Need to track when enemy pawn double-advances
  2. Flag must reset after opponent moves
  3. Capture only valid from adjacent file on correct rank
""")


if __name__ == "__main__":
    main()
