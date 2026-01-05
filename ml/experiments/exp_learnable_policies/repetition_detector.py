"""
Learning Draw by Repetition from Self-Play

Goal: Evolve a RING BUFFER mechanism that detects position repetition.

Threefold Repetition Rule:
- Game is drawn if the same position occurs three times
- Position includes piece placement, turn, castling rights, en passant
- Requires tracking game history (not just current position)

What we're learning:
- Need a ring buffer to store position hashes
- Need to check for matches against buffer
- Need to count occurrences (threshold = 3)
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import deque


# =============================================================================
# SIMPLIFIED GAME FOR REPETITION LEARNING
# =============================================================================

@dataclass
class SimplePosition:
    """A hashable game position for repetition detection."""
    # Simplified: just track some state values
    state_hash: int
    turn: int

    def __hash__(self):
        return hash((self.state_hash, self.turn))

    def __eq__(self, other):
        return self.state_hash == other.state_hash and self.turn == other.turn


class RepetitionGame:
    """
    Simple game for testing repetition detection.

    The game can revisit positions, and we need to detect when
    the same position occurs three times.
    """

    def __init__(self, n_states: int = 20):
        self.n_states = n_states
        self.current_state = 0
        self.turn = 1
        self.position_history: List[SimplePosition] = []
        self.move_count = 0

    def reset(self):
        self.current_state = random.randint(0, self.n_states - 1)
        self.turn = 1
        self.position_history = []
        self.move_count = 0
        self._record_position()

    def _record_position(self):
        pos = SimplePosition(self.current_state, self.turn)
        self.position_history.append(pos)

    def get_position_hash(self) -> int:
        return hash(SimplePosition(self.current_state, self.turn))

    def get_legal_moves(self) -> List[int]:
        """Get possible next states (can revisit previous states)."""
        # Allow moving to adjacent states and back to start
        moves = []
        for delta in [-2, -1, 0, 1, 2]:
            new_state = (self.current_state + delta) % self.n_states
            moves.append(new_state)
        return moves

    def make_move(self, new_state: int):
        """Move to new state."""
        self.current_state = new_state
        self.turn *= -1
        self.move_count += 1
        self._record_position()

    def count_position_occurrences(self) -> int:
        """Count how many times current position has occurred."""
        current = SimplePosition(self.current_state, self.turn)
        return sum(1 for p in self.position_history if p == current)

    def is_threefold_repetition(self) -> bool:
        """Check if current position has occurred 3 times."""
        return self.count_position_occurrences() >= 3

    def get_context(self) -> Dict:
        return {
            'state': self.current_state,
            'turn': self.turn,
            'history': list(self.position_history),
            'history_hashes': [hash(p) % 10000 for p in self.position_history],
            'move_count': self.move_count,
        }


# =============================================================================
# REPETITION DETECTOR GENOME
# =============================================================================

@dataclass
class RepetitionGenome:
    """
    Genome for learning repetition detection.

    Key insight: We need a RING BUFFER to track position history.
    """
    # Buffer configuration
    buffer_size: int = 10
    use_hash_matching: bool = False
    count_threshold: int = 3

    # Weights for detection
    w_match_count: float = 0.0
    w_recent_match: float = 0.0
    w_buffer_depth: float = 0.0
    bias: float = 0.0

    # Internal state
    position_buffer: deque = None

    # Fitness
    fitness: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def __post_init__(self):
        if self.position_buffer is None:
            self.position_buffer = deque(maxlen=max(1, self.buffer_size))

    def copy(self):
        new = RepetitionGenome(
            buffer_size=self.buffer_size,
            use_hash_matching=self.use_hash_matching,
            count_threshold=self.count_threshold,
            w_match_count=self.w_match_count,
            w_recent_match=self.w_recent_match,
            w_buffer_depth=self.w_buffer_depth,
            bias=self.bias,
        )
        new.position_buffer = deque(self.position_buffer, maxlen=max(1, self.buffer_size))
        new.fitness = self.fitness
        new.precision = self.precision
        new.recall = self.recall
        new.f1 = self.f1
        return new

    def reset_buffer(self):
        """Reset the position buffer."""
        self.position_buffer = deque(maxlen=max(1, self.buffer_size))

    def record_position(self, pos_hash: int):
        """Record a position in the buffer."""
        self.position_buffer.append(pos_hash)

    def predict_repetition(self, ctx: Dict) -> bool:
        """Predict if current position is a threefold repetition."""
        current_hash = hash(SimplePosition(ctx['state'], ctx['turn'])) % 10000

        # Count matches in buffer
        match_count = sum(1 for h in self.position_buffer if h == current_hash)

        # Check for recent match
        recent_match = 0
        if len(self.position_buffer) > 0 and self.position_buffer[-1] == current_hash:
            recent_match = 1

        # Buffer depth feature
        buffer_depth = len(self.position_buffer) / max(1, self.buffer_size)

        # Compute prediction
        logit = self.bias
        if self.use_hash_matching:
            logit += self.w_match_count * match_count
            logit += self.w_recent_match * recent_match
            logit += self.w_buffer_depth * buffer_depth

        return logit > 0 or match_count >= self.count_threshold

    def _extract_features(self, ctx: Dict) -> Dict[str, float]:
        """Extract features for prediction."""
        current_hash = hash(SimplePosition(ctx['state'], ctx['turn'])) % 10000
        history_hashes = ctx.get('history_hashes', [])

        match_count = sum(1 for h in history_hashes if h == current_hash)
        recent_match = 1.0 if history_hashes and history_hashes[-1] == current_hash else 0.0
        buffer_depth = len(self.position_buffer) / max(1, self.buffer_size)

        return {
            'match_count': float(match_count),
            'recent_match': recent_match,
            'buffer_depth': buffer_depth,
        }


# =============================================================================
# REPETITION EVOLVER
# =============================================================================

class RepetitionEvolver:
    """Evolves repetition detection from self-play."""

    def __init__(self, population_size: int = 30, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[RepetitionGenome] = []
        self.best_genome: Optional[RepetitionGenome] = None
        self.generation = 0

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            genome = RepetitionGenome(
                buffer_size=random.randint(3, 20),
                use_hash_matching=random.random() > 0.5,
                count_threshold=random.randint(2, 4),
                w_match_count=random.gauss(0, 3),
                w_recent_match=random.gauss(0, 3),
                w_buffer_depth=random.gauss(0, 3),
                bias=random.gauss(0, 3),
            )
            self.population.append(genome)

    def evaluate_genome(self, genome: RepetitionGenome) -> float:
        """Evaluate genome on repetition detection."""
        tp, fp, tn, fn = 0, 0, 0, 0

        for _ in range(100):  # Games
            game = RepetitionGame(n_states=10)  # Smaller state space = more repetitions
            game.reset()
            genome.reset_buffer()

            for _ in range(50):  # Moves per game
                moves = game.get_legal_moves()
                if not moves:
                    break

                ctx = game.get_context()

                # Predict repetition
                actual_rep = game.is_threefold_repetition()
                pred_rep = genome.predict_repetition(ctx)

                if actual_rep and pred_rep:
                    tp += 1
                elif actual_rep and not pred_rep:
                    fn += 1
                elif not actual_rep and pred_rep:
                    fp += 1
                else:
                    tn += 1

                # Make move (biased toward revisiting)
                if random.random() < 0.3 and len(game.position_history) > 5:
                    # Try to revisit old position
                    old_state = random.choice([p.state_hash for p in game.position_history[-10:]])
                    if old_state in moves:
                        move = old_state
                    else:
                        move = random.choice(moves)
                else:
                    move = random.choice(moves)

                game.make_move(move)
                genome.record_position(game.get_position_hash() % 10000)

                if game.move_count > 100:
                    break

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        genome.precision = precision
        genome.recall = recall
        genome.f1 = f1
        genome.fitness = f1

        return f1

    def mutate(self, genome: RepetitionGenome) -> RepetitionGenome:
        new = genome.copy()
        if random.random() < self.mutation_rate:
            choice = random.randint(0, 5)
            if choice == 0:
                new.buffer_size = max(3, min(30, new.buffer_size + random.randint(-2, 2)))
            elif choice == 1:
                new.use_hash_matching = not new.use_hash_matching
            elif choice == 2:
                new.count_threshold = max(2, min(5, new.count_threshold + random.randint(-1, 1)))
            elif choice == 3:
                new.w_match_count += random.gauss(0, 1)
            elif choice == 4:
                new.w_recent_match += random.gauss(0, 1)
            else:
                new.bias += random.gauss(0, 1)
        return new

    def evolve(self, n_generations: int = 50, verbose: bool = True) -> RepetitionGenome:
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
                print(f"         buffer={best.buffer_size}, hash={best.use_hash_matching}, "
                      f"thresh={best.count_threshold}")

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
    print("LEARNING DRAW BY REPETITION FROM SELF-PLAY")
    print("=" * 70)
    print()
    print("Goal: Evolve a RING BUFFER for position history tracking")
    print()
    print("Threefold Repetition Rule:")
    print("- Game is drawn if same position occurs 3 times")
    print("- Requires tracking position history (ring buffer)")
    print("- Must hash positions and count occurrences")
    print()

    evolver = RepetitionEvolver(population_size=30, mutation_rate=0.4)

    print("-" * 40)
    print("EVOLVING...")
    print("-" * 40)

    best = evolver.evolve(n_generations=50, verbose=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
Learned Repetition Detection:

Configuration Discovered:
  - Buffer size:       {best.buffer_size}
  - Use hash matching: {best.use_hash_matching}
  - Count threshold:   {best.count_threshold}

Learned Weights:
  - w_match_count:  {best.w_match_count:+.2f}
  - w_recent_match: {best.w_recent_match:+.2f}
  - w_buffer_depth: {best.w_buffer_depth:+.2f}
  - bias:           {best.bias:+.2f}

Performance:
  - Precision: {best.precision:.1%}
  - Recall:    {best.recall:.1%}
  - F1 Score:  {best.f1:.1%}

Key Insight:
  Repetition detection requires a RING BUFFER - bounded memory
  for position history. Evolution should discover:
  1. Need to store position hashes
  2. Need to count matches against current position
  3. Threshold of 3 for threefold repetition
  4. Buffer size balances memory vs detection window
""")


if __name__ == "__main__":
    main()
