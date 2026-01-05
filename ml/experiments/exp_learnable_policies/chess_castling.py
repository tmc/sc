"""
Learning Castling Rights from Self-Play

Goal: Evolve PERSISTENT FLAG mechanism that tracks piece movement history.

Castling Rule:
- King can castle if neither king nor rook has ever moved
- Once a piece moves, that side's castling right is lost FOREVER
- This requires persistent flags that never reset

What we're learning:
- Need to track "has king moved" (persistent flag)
- Need to track "has rook moved" (per-side persistent flags)
- Flags persist until explicitly changed (never auto-reset)
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


# =============================================================================
# SIMPLIFIED CHESS FOR CASTLING
# =============================================================================

class PieceType(Enum):
    EMPTY = 0
    KING = 1
    ROOK = 2


@dataclass
class CastlingBoard:
    """
    Simplified board for castling learning.
    Only tracks kings, rooks, and castling rights.
    """
    # Piece positions: (type, color) where color is 1=white, -1=black
    # Position encoding: squares[rank][file]
    squares: List[List[Tuple[PieceType, int]]]

    # Castling rights - these are the PERSISTENT FLAGS we're learning
    white_king_side: bool = True
    white_queen_side: bool = True
    black_king_side: bool = True
    black_queen_side: bool = True

    # Has piece moved tracking (ground truth)
    white_king_moved: bool = False
    white_rook_a_moved: bool = False
    white_rook_h_moved: bool = False
    black_king_moved: bool = False
    black_rook_a_moved: bool = False
    black_rook_h_moved: bool = False

    turn: int = 1  # 1 = white, -1 = black

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to starting position."""
        self.squares = [[(PieceType.EMPTY, 0) for _ in range(8)] for _ in range(8)]

        # Place kings
        self.squares[0][4] = (PieceType.KING, 1)   # White king on e1
        self.squares[7][4] = (PieceType.KING, -1)  # Black king on e8

        # Place rooks
        self.squares[0][0] = (PieceType.ROOK, 1)   # White rook on a1
        self.squares[0][7] = (PieceType.ROOK, 1)   # White rook on h1
        self.squares[7][0] = (PieceType.ROOK, -1)  # Black rook on a8
        self.squares[7][7] = (PieceType.ROOK, -1)  # Black rook on h8

        # Reset castling rights
        self.white_king_side = True
        self.white_queen_side = True
        self.black_king_side = True
        self.black_queen_side = True

        # Reset movement tracking
        self.white_king_moved = False
        self.white_rook_a_moved = False
        self.white_rook_h_moved = False
        self.black_king_moved = False
        self.black_rook_a_moved = False
        self.black_rook_h_moved = False

        self.turn = 1

    def copy(self):
        new = CastlingBoard.__new__(CastlingBoard)
        new.squares = [[(self.squares[r][f][0], self.squares[r][f][1])
                        for f in range(8)] for r in range(8)]
        new.white_king_side = self.white_king_side
        new.white_queen_side = self.white_queen_side
        new.black_king_side = self.black_king_side
        new.black_queen_side = self.black_queen_side
        new.white_king_moved = self.white_king_moved
        new.white_rook_a_moved = self.white_rook_a_moved
        new.white_rook_h_moved = self.white_rook_h_moved
        new.black_king_moved = self.black_king_moved
        new.black_rook_a_moved = self.black_rook_a_moved
        new.black_rook_h_moved = self.black_rook_h_moved
        new.turn = self.turn
        return new

    def get(self, rank: int, file: int) -> Tuple[PieceType, int]:
        if 0 <= rank < 8 and 0 <= file < 8:
            return self.squares[rank][file]
        return (PieceType.EMPTY, 0)

    def get_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Get all legal moves.
        Returns: (from_sq, to_sq, move_type)
        move_type: 'normal', 'castle_k', 'castle_q'
        """
        moves = []
        color = self.turn
        back_rank = 0 if color == 1 else 7

        for r in range(8):
            for f in range(8):
                piece_type, piece_color = self.get(r, f)
                if piece_color != color:
                    continue

                if piece_type == PieceType.KING:
                    # King moves
                    for dr in [-1, 0, 1]:
                        for df in [-1, 0, 1]:
                            if dr == 0 and df == 0:
                                continue
                            nr, nf = r + dr, f + df
                            if 0 <= nr < 8 and 0 <= nf < 8:
                                target_type, target_color = self.get(nr, nf)
                                if target_color != color:
                                    moves.append(((r, f), (nr, nf), 'normal'))

                    # Castling
                    if r == back_rank and f == 4:
                        # Kingside
                        if self._can_castle_kingside():
                            moves.append(((r, 4), (r, 6), 'castle_k'))
                        # Queenside
                        if self._can_castle_queenside():
                            moves.append(((r, 4), (r, 2), 'castle_q'))

                elif piece_type == PieceType.ROOK:
                    # Rook moves
                    for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nf = r + dr, f + df
                        while 0 <= nr < 8 and 0 <= nf < 8:
                            target_type, target_color = self.get(nr, nf)
                            if target_color == color:
                                break
                            moves.append(((r, f), (nr, nf), 'normal'))
                            if target_type != PieceType.EMPTY:
                                break
                            nr += dr
                            nf += df

        return moves

    def _can_castle_kingside(self) -> bool:
        """Check if kingside castling is legal."""
        if self.turn == 1:
            if not self.white_king_side:
                return False
            if self.white_king_moved or self.white_rook_h_moved:
                return False
            # Check squares between king and rook are empty
            if self.get(0, 5)[0] != PieceType.EMPTY:
                return False
            if self.get(0, 6)[0] != PieceType.EMPTY:
                return False
        else:
            if not self.black_king_side:
                return False
            if self.black_king_moved or self.black_rook_h_moved:
                return False
            if self.get(7, 5)[0] != PieceType.EMPTY:
                return False
            if self.get(7, 6)[0] != PieceType.EMPTY:
                return False
        return True

    def _can_castle_queenside(self) -> bool:
        """Check if queenside castling is legal."""
        if self.turn == 1:
            if not self.white_queen_side:
                return False
            if self.white_king_moved or self.white_rook_a_moved:
                return False
            if self.get(0, 1)[0] != PieceType.EMPTY:
                return False
            if self.get(0, 2)[0] != PieceType.EMPTY:
                return False
            if self.get(0, 3)[0] != PieceType.EMPTY:
                return False
        else:
            if not self.black_queen_side:
                return False
            if self.black_king_moved or self.black_rook_a_moved:
                return False
            if self.get(7, 1)[0] != PieceType.EMPTY:
                return False
            if self.get(7, 2)[0] != PieceType.EMPTY:
                return False
            if self.get(7, 3)[0] != PieceType.EMPTY:
                return False
        return True

    def make_move(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int], move_type: str):
        """Make a move and update castling rights."""
        r1, f1 = from_sq
        r2, f2 = to_sq

        piece_type, piece_color = self.get(r1, f1)

        # Track piece movement (persistent flags)
        if piece_type == PieceType.KING:
            if piece_color == 1:
                self.white_king_moved = True
                self.white_king_side = False
                self.white_queen_side = False
            else:
                self.black_king_moved = True
                self.black_king_side = False
                self.black_queen_side = False

        elif piece_type == PieceType.ROOK:
            if piece_color == 1:
                if f1 == 0:
                    self.white_rook_a_moved = True
                    self.white_queen_side = False
                elif f1 == 7:
                    self.white_rook_h_moved = True
                    self.white_king_side = False
            else:
                if f1 == 0:
                    self.black_rook_a_moved = True
                    self.black_queen_side = False
                elif f1 == 7:
                    self.black_rook_h_moved = True
                    self.black_king_side = False

        # Handle castling move
        if move_type == 'castle_k':
            # Move king
            self.squares[r2][f2] = self.squares[r1][f1]
            self.squares[r1][f1] = (PieceType.EMPTY, 0)
            # Move rook
            self.squares[r2][5] = self.squares[r2][7]
            self.squares[r2][7] = (PieceType.EMPTY, 0)
        elif move_type == 'castle_q':
            # Move king
            self.squares[r2][f2] = self.squares[r1][f1]
            self.squares[r1][f1] = (PieceType.EMPTY, 0)
            # Move rook
            self.squares[r2][3] = self.squares[r2][0]
            self.squares[r2][0] = (PieceType.EMPTY, 0)
        else:
            # Normal move
            self.squares[r2][f2] = self.squares[r1][f1]
            self.squares[r1][f1] = (PieceType.EMPTY, 0)

        self.turn *= -1


# =============================================================================
# CASTLING ENVIRONMENT
# =============================================================================

class CastlingEnvironment:
    """Environment for learning castling rights."""

    def __init__(self):
        self.board = CastlingBoard()
        self.move_count = 0

    def reset(self):
        self.board.reset()
        self.move_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            'board': self.board.copy(),
            'turn': self.board.turn,
        }

    def get_legal_moves(self):
        return self.board.get_moves()

    def step(self, move):
        from_sq, to_sq, move_type = move
        self.board.make_move(from_sq, to_sq, move_type)
        self.move_count += 1
        done = self.move_count > 50 or not self.board.get_moves()
        return self._get_obs(), 0.0, done, {}

    def can_castle(self) -> Dict[str, bool]:
        """Get current castling availability."""
        return {
            'white_king_side': self.board._can_castle_kingside() if self.board.turn == 1 else False,
            'white_queen_side': self.board._can_castle_queenside() if self.board.turn == 1 else False,
            'black_king_side': self.board._can_castle_kingside() if self.board.turn == -1 else False,
            'black_queen_side': self.board._can_castle_queenside() if self.board.turn == -1 else False,
        }

    def get_context(self) -> Dict:
        return {
            'board': self.board.copy(),
            'turn': self.board.turn,
            'white_king_moved': self.board.white_king_moved,
            'white_rook_a_moved': self.board.white_rook_a_moved,
            'white_rook_h_moved': self.board.white_rook_h_moved,
            'black_king_moved': self.board.black_king_moved,
            'black_rook_a_moved': self.board.black_rook_a_moved,
            'black_rook_h_moved': self.board.black_rook_h_moved,
        }


# =============================================================================
# CASTLING GENOME
# =============================================================================

@dataclass
class CastlingGenome:
    """
    Genome for learning castling rights detection.

    Key insight: We need PERSISTENT FLAGS that track:
    - Has king ever moved?
    - Has each rook ever moved?
    These flags persist forever once set.
    """
    # Feature flags
    track_king_moved: bool = False
    track_rook_a_moved: bool = False
    track_rook_h_moved: bool = False
    check_path_clear: bool = False

    # Weights
    w_king: float = 0.0
    w_rook_a: float = 0.0
    w_rook_h: float = 0.0
    w_path: float = 0.0
    bias: float = 0.0

    # Fitness
    fitness: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def copy(self):
        new = CastlingGenome(
            track_king_moved=self.track_king_moved,
            track_rook_a_moved=self.track_rook_a_moved,
            track_rook_h_moved=self.track_rook_h_moved,
            check_path_clear=self.check_path_clear,
            w_king=self.w_king,
            w_rook_a=self.w_rook_a,
            w_rook_h=self.w_rook_h,
            w_path=self.w_path,
            bias=self.bias,
        )
        new.fitness = self.fitness
        new.precision = self.precision
        new.recall = self.recall
        new.f1 = self.f1
        return new

    def predict_can_castle(self, ctx: Dict, side: str) -> bool:
        """Predict if castling is available on given side."""
        features = self._extract_features(ctx, side)

        logit = self.bias
        if self.track_king_moved:
            logit += self.w_king * features['king_unmoved']
        if side == 'kingside' and self.track_rook_h_moved:
            logit += self.w_rook_h * features['rook_unmoved']
        elif side == 'queenside' and self.track_rook_a_moved:
            logit += self.w_rook_a * features['rook_unmoved']
        if self.check_path_clear:
            logit += self.w_path * features['path_clear']

        return logit > 0

    def _extract_features(self, ctx: Dict, side: str) -> Dict[str, float]:
        """Extract features for castling prediction."""
        turn = ctx.get('turn', 1)

        if turn == 1:
            king_moved = ctx.get('white_king_moved', False)
            rook_moved = ctx.get('white_rook_h_moved' if side == 'kingside' else 'white_rook_a_moved', False)
        else:
            king_moved = ctx.get('black_king_moved', False)
            rook_moved = ctx.get('black_rook_h_moved' if side == 'kingside' else 'black_rook_a_moved', False)

        king_unmoved = 0.0 if king_moved else 1.0
        rook_unmoved = 0.0 if rook_moved else 1.0

        # Path clear (simplified - would check actual squares)
        path_clear = 1.0  # Assume clear for now

        return {
            'king_unmoved': king_unmoved,
            'rook_unmoved': rook_unmoved,
            'path_clear': path_clear,
        }


# =============================================================================
# CASTLING EVOLVER
# =============================================================================

class CastlingEvolver:
    """Evolves castling detection from self-play."""

    def __init__(self, population_size: int = 30, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[CastlingGenome] = []
        self.best_genome: Optional[CastlingGenome] = None
        self.generation = 0

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            genome = CastlingGenome(
                track_king_moved=random.random() > 0.5,
                track_rook_a_moved=random.random() > 0.5,
                track_rook_h_moved=random.random() > 0.5,
                check_path_clear=random.random() > 0.5,
                w_king=random.gauss(0, 3),
                w_rook_a=random.gauss(0, 3),
                w_rook_h=random.gauss(0, 3),
                w_path=random.gauss(0, 3),
                bias=random.gauss(0, 3),
            )
            self.population.append(genome)

    def evaluate_genome(self, genome: CastlingGenome) -> float:
        """Evaluate genome on castling detection."""
        env = CastlingEnvironment()

        tp, fp, tn, fn = 0, 0, 0, 0

        for _ in range(50):
            env.reset()

            for _ in range(30):
                moves = env.get_legal_moves()
                if not moves:
                    break

                ctx = env.get_context()

                # Check castling predictions
                for side in ['kingside', 'queenside']:
                    actual = env.can_castle().get(
                        f"{'white' if env.board.turn == 1 else 'black'}_{side.replace('side', '_side')}", False
                    )
                    pred = genome.predict_can_castle(ctx, side)

                    if actual and pred:
                        tp += 1
                    elif actual and not pred:
                        fn += 1
                    elif not actual and pred:
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

    def mutate(self, genome: CastlingGenome) -> CastlingGenome:
        new = genome.copy()
        if random.random() < self.mutation_rate:
            choice = random.randint(0, 4)
            if choice == 0:
                new.track_king_moved = not new.track_king_moved
            elif choice == 1:
                new.track_rook_a_moved = not new.track_rook_a_moved
            elif choice == 2:
                new.track_rook_h_moved = not new.track_rook_h_moved
            elif choice == 3:
                new.check_path_clear = not new.check_path_clear
            else:
                attr = random.choice(['w_king', 'w_rook_a', 'w_rook_h', 'w_path', 'bias'])
                setattr(new, attr, getattr(new, attr) + random.gauss(0, 1))
        return new

    def evolve(self, n_generations: int = 50, verbose: bool = True) -> CastlingGenome:
        if not self.population:
            self.initialize_population()

        for gen in range(n_generations):
            for g in self.population:
                self.evaluate_genome(g)

            self.population.sort(key=lambda g: g.fitness, reverse=True)

            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].copy()

            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                best = self.population[0]
                print(f"Gen {gen:3d}: F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
                print(f"         king={best.track_king_moved}, rook_a={best.track_rook_a_moved}, "
                      f"rook_h={best.track_rook_h_moved}")

            new_pop = [self.population[0].copy()]
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
    print("LEARNING CASTLING RIGHTS FROM SELF-PLAY")
    print("=" * 70)
    print()
    print("Goal: Evolve PERSISTENT FLAGS for piece movement tracking")
    print()
    print("Castling Rule:")
    print("- King can castle if neither king nor rook has EVER moved")
    print("- Once moved, castling right is lost FOREVER")
    print("- This requires persistent flags that never reset")
    print()

    evolver = CastlingEvolver(population_size=30, mutation_rate=0.4)

    print("-" * 40)
    print("EVOLVING...")
    print("-" * 40)

    best = evolver.evolve(n_generations=50, verbose=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
Learned Castling Detection:

Features Discovered:
  - Track king moved:    {best.track_king_moved}
  - Track rook A moved:  {best.track_rook_a_moved}
  - Track rook H moved:  {best.track_rook_h_moved}
  - Check path clear:    {best.check_path_clear}

Learned Weights:
  - w_king:   {best.w_king:+.2f}
  - w_rook_a: {best.w_rook_a:+.2f}
  - w_rook_h: {best.w_rook_h:+.2f}
  - w_path:   {best.w_path:+.2f}
  - bias:     {best.bias:+.2f}

Performance:
  - Precision: {best.precision:.1%}
  - Recall:    {best.recall:.1%}
  - F1 Score:  {best.f1:.1%}

Key Insight:
  Castling requires PERSISTENT FLAGS - state that never resets.
  Evolution should discover:
  1. Need to track king movement (permanent flag)
  2. Need to track each rook movement (separate permanent flags)
  3. These flags are "set once, never cleared" - true history state
""")


if __name__ == "__main__":
    main()
