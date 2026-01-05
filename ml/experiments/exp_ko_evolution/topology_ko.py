"""
Ko Evolution via Topology Discovery

PIVOT: Use topology evolution framework, not preset selection.

Key innovation: History depth is EVOLVABLE from 0.
- Start with no history (depth=0)
- Evolution discovers depth=1 solves Ko
- The STRUCTURE emerges, not selected

Adds to StatechartGenome:
- history_depth[i]: 0-5 (evolvable per state)
- history_encoding[i]: 'none', 'raw', 'hash', 'diff' (evolvable)
- decay_rate[i]: 0.0-1.0 (evolvable)

Fitness:
- Games that finish: +1
- Infinite loops: -10
- Ko violations fixed: +5
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict
from enum import IntEnum


# =============================================================================
# History Encoding Types
# =============================================================================

class HistoryEncoding(IntEnum):
    NONE = 0      # No history (depth ignored)
    RAW = 1       # Store raw position
    HASH = 2      # Store position hash
    DIFF = 3      # Store delta from previous


# =============================================================================
# Extended Genome with Evolvable History
# =============================================================================

@dataclass
class HistoryGene:
    """Evolvable history parameters for a single state."""
    depth: int = 0                          # 0-5, how many steps to remember
    encoding: HistoryEncoding = HistoryEncoding.NONE
    decay_rate: float = 0.0                 # 0.0-1.0, how fast history decays

    def copy(self) -> 'HistoryGene':
        return HistoryGene(
            depth=self.depth,
            encoding=self.encoding,
            decay_rate=self.decay_rate
        )


@dataclass
class KoGenome:
    """
    Extended genome for Ko evolution.

    Key difference from preset selection:
    - history_genes[i] is EVOLVABLE per state
    - depth starts at 0 (no history)
    - evolution must DISCOVER that depth=1 solves Ko
    """
    n_states: int
    parent: List[int]
    state_type: List[int]  # 0=BASIC, 1=OR, 2=AND
    history_genes: List[HistoryGene]  # NEW: Evolvable history per state
    transitions: List[Tuple[int, int, int]]  # (src, tgt, event)

    # Fitness tracking
    fitness: float = 0.0
    games_completed: int = 0
    games_looped: int = 0
    ko_violations_prevented: int = 0

    def copy(self) -> 'KoGenome':
        return KoGenome(
            n_states=self.n_states,
            parent=self.parent.copy(),
            state_type=self.state_type.copy(),
            history_genes=[g.copy() for g in self.history_genes],
            transitions=[t for t in self.transitions],
            fitness=0.0
        )

    def is_valid(self) -> bool:
        """Check genome validity."""
        if len(self.parent) != self.n_states:
            return False
        if len(self.state_type) != self.n_states:
            return False
        if len(self.history_genes) != self.n_states:
            return False
        # Check for cycles
        for i in range(self.n_states):
            visited = set()
            curr = i
            while curr != -1:
                if curr in visited:
                    return False
                visited.add(curr)
                curr = self.parent[curr] if curr < len(self.parent) else -1
        return True

    def max_history_depth(self) -> int:
        """Get maximum history depth across all states."""
        return max(g.depth for g in self.history_genes)

    def active_history_states(self) -> int:
        """Count states with active history (depth > 0)."""
        return sum(1 for g in self.history_genes if g.depth > 0)


def create_random_ko_genome(n_states: int = 5) -> KoGenome:
    """
    Create random genome with history starting at depth=0.

    Key: Most history starts DISABLED (depth=0).
    Evolution must discover that enabling it helps.
    """
    parent = [-1] + [random.randint(0, i) for i in range(n_states - 1)]
    state_type = [random.randint(0, 2) for _ in range(n_states)]
    state_type[0] = 1  # Root is OR

    # CRITICAL: History starts mostly disabled (depth=0)
    history_genes = []
    for i in range(n_states):
        if random.random() < 0.1:  # Only 10% start with any history
            gene = HistoryGene(
                depth=random.randint(1, 3),
                encoding=HistoryEncoding(random.randint(1, 3)),
                decay_rate=random.uniform(0.0, 0.5)
            )
        else:
            gene = HistoryGene(depth=0)  # No history
        history_genes.append(gene)

    # Random transitions
    n_trans = random.randint(1, n_states)
    transitions = []
    for _ in range(n_trans):
        src = random.randint(0, n_states - 1)
        tgt = random.randint(0, n_states - 1)
        event = random.randint(0, 80)  # Go 9x9 events
        transitions.append((src, tgt, event))

    return KoGenome(
        n_states=n_states,
        parent=parent,
        state_type=state_type,
        history_genes=history_genes,
        transitions=transitions
    )


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_history_gene(gene: HistoryGene) -> HistoryGene:
    """Mutate a single history gene."""
    new = gene.copy()

    r = random.random()

    if r < 0.4:
        # Mutate depth (most important!)
        delta = random.choice([-1, 0, 0, 1, 1])  # Bias toward increasing
        new.depth = max(0, min(5, new.depth + delta))
    elif r < 0.6:
        # Mutate encoding
        if new.depth > 0:  # Only if history active
            new.encoding = HistoryEncoding(random.randint(0, 3))
    else:
        # Mutate decay
        new.decay_rate = max(0.0, min(1.0, new.decay_rate + random.gauss(0, 0.1)))

    # If depth becomes 0, reset encoding to NONE
    if new.depth == 0:
        new.encoding = HistoryEncoding.NONE
    elif new.encoding == HistoryEncoding.NONE and new.depth > 0:
        new.encoding = HistoryEncoding.RAW

    return new


def mutate_ko_genome(genome: KoGenome) -> KoGenome:
    """Mutate a Ko genome."""
    new = genome.copy()

    # Mutate history genes (high probability - this is key!)
    for i in range(new.n_states):
        if random.random() < 0.3:
            new.history_genes[i] = mutate_history_gene(new.history_genes[i])

    # Sometimes mutate structure
    if random.random() < 0.1:
        # Change a parent
        idx = random.randint(1, new.n_states - 1)
        new.parent[idx] = random.randint(0, idx - 1)

    if random.random() < 0.1:
        # Change state type
        idx = random.randint(1, new.n_states - 1)
        new.state_type[idx] = random.randint(0, 2)

    if random.random() < 0.15:
        # Add transition
        src = random.randint(0, new.n_states - 1)
        tgt = random.randint(0, new.n_states - 1)
        event = random.randint(0, 80)
        new.transitions.append((src, tgt, event))

    if random.random() < 0.1 and len(new.transitions) > 1:
        # Remove transition
        new.transitions.pop(random.randint(0, len(new.transitions) - 1))

    return new


def crossover_ko(g1: KoGenome, g2: KoGenome) -> KoGenome:
    """Crossover two Ko genomes."""
    # Take structure from one parent, history from the other
    if random.random() < 0.5:
        struct_parent, hist_parent = g1, g2
    else:
        struct_parent, hist_parent = g2, g1

    n_states = min(struct_parent.n_states, hist_parent.n_states)

    return KoGenome(
        n_states=n_states,
        parent=struct_parent.parent[:n_states],
        state_type=struct_parent.state_type[:n_states],
        history_genes=[hist_parent.history_genes[i].copy()
                       if i < len(hist_parent.history_genes)
                       else HistoryGene(depth=0)
                       for i in range(n_states)],
        transitions=random.choice([struct_parent.transitions, hist_parent.transitions])
    )


# =============================================================================
# Go Game (Simplified for Ko Testing)
# =============================================================================

class GoGame:
    """Simplified Go for Ko evolution testing."""

    SIZE = 9
    TOTAL = 81

    def __init__(self):
        self.board = [0] * self.TOTAL  # 0=empty, 1=black, 2=white
        self.current = 1  # Black starts
        self.history: List[Tuple[int, ...]] = []
        self.last_capture: Optional[int] = None
        self.passes = 0

    def copy(self) -> 'GoGame':
        g = GoGame()
        g.board = self.board.copy()
        g.current = self.current
        g.history = self.history.copy()
        g.last_capture = self.last_capture
        g.passes = self.passes
        return g

    def _neighbors(self, idx: int) -> List[int]:
        x, y = idx % self.SIZE, idx // self.SIZE
        n = []
        if x > 0: n.append(idx - 1)
        if x < 8: n.append(idx + 1)
        if y > 0: n.append(idx - 9)
        if y < 8: n.append(idx + 9)
        return n

    def _group_liberties(self, idx: int) -> Tuple[Set[int], int]:
        """Get group containing idx and its liberty count."""
        color = self.board[idx]
        if color == 0:
            return set(), 0

        group = set()
        liberties = set()
        stack = [idx]

        while stack:
            curr = stack.pop()
            if curr in group:
                continue
            if self.board[curr] != color:
                continue
            group.add(curr)
            for n in self._neighbors(curr):
                if self.board[n] == 0:
                    liberties.add(n)
                elif self.board[n] == color and n not in group:
                    stack.append(n)

        return group, len(liberties)

    def _would_suicide(self, idx: int) -> bool:
        """Check if move would be suicide."""
        self.board[idx] = self.current
        opp = 3 - self.current

        # Check if captures
        for n in self._neighbors(idx):
            if self.board[n] == opp:
                _, libs = self._group_liberties(n)
                if libs == 0:
                    self.board[idx] = 0
                    return False  # Captures, not suicide

        # Check own liberties
        _, libs = self._group_liberties(idx)
        self.board[idx] = 0
        return libs == 0

    def is_legal_no_ko(self, idx: int) -> bool:
        """Legal check WITHOUT Ko rule."""
        if self.board[idx] != 0:
            return False
        return not self._would_suicide(idx)

    def is_ko_violation(self, idx: int, genome: KoGenome) -> bool:
        """
        Check if move violates Ko based on EVOLVED history.

        This is where evolution discovers what history depth works!
        """
        # Find max history depth in genome
        max_depth = genome.max_history_depth()

        if max_depth == 0:
            return False  # No history = no Ko check

        # Simulate move
        test = self.copy()
        test.board[idx] = test.current

        # Remove captured stones
        opp = 3 - test.current
        for n in test._neighbors(idx):
            if test.board[n] == opp:
                group, libs = test._group_liberties(n)
                if libs == 0:
                    for s in group:
                        test.board[s] = 0

        new_pos = tuple(test.board)

        # Check against history up to max_depth
        for i, hist_pos in enumerate(reversed(self.history[-max_depth:])):
            if new_pos == hist_pos:
                return True  # Position repeated = Ko violation!

        return False

    def make_move(self, idx: int) -> Tuple[bool, int]:
        """Make move. Returns (success, captures_count)."""
        if self.board[idx] != 0:
            return False, 0

        self.board[idx] = self.current
        opp = 3 - self.current

        # Capture
        captured = 0
        captured_single = None
        for n in self._neighbors(idx):
            if self.board[n] == opp:
                group, libs = self._group_liberties(n)
                if libs == 0:
                    if len(group) == 1:
                        captured_single = list(group)[0]
                    for s in group:
                        self.board[s] = 0
                    captured += len(group)

        # Record position
        self.history.append(tuple(self.board))
        self.last_capture = captured_single if captured == 1 else None

        # Switch
        self.current = 3 - self.current
        self.passes = 0

        return True, captured

    def pass_move(self):
        self.current = 3 - self.current
        self.passes += 1
        self.last_capture = None

    def is_terminal(self) -> bool:
        return self.passes >= 2

    def position_repeated(self, lookback: int = 10) -> bool:
        """Check if current position repeats recent history."""
        if len(self.history) < 2:
            return False
        current = tuple(self.board)
        for pos in self.history[-lookback:-1]:
            if pos == current:
                return True
        return False


# =============================================================================
# Fitness Evaluation
# =============================================================================

def play_game_with_genome(genome: KoGenome, max_moves: int = 80) -> Dict:
    """
    Play a self-play game using evolved Ko genome.

    Returns game statistics for fitness.
    """
    game = GoGame()
    ko_prevented = 0

    for move_num in range(max_moves):
        # Get legal moves
        legal = []
        for idx in range(81):
            if game.is_legal_no_ko(idx):
                # Check Ko using EVOLVED history
                if not game.is_ko_violation(idx, genome):
                    legal.append(idx)
                else:
                    ko_prevented += 1

        if not legal:
            game.pass_move()
            if game.is_terminal():
                return {
                    "completed": True,
                    "length": move_num + 1,
                    "looped": False,
                    "ko_prevented": ko_prevented
                }
            continue

        # Random move
        move = random.choice(legal)
        game.make_move(move)

        # Check for infinite loop
        if game.position_repeated(lookback=8):
            return {
                "completed": False,
                "length": move_num + 1,
                "looped": True,
                "ko_prevented": ko_prevented
            }

    return {
        "completed": False,
        "length": max_moves,
        "looped": False,
        "ko_prevented": ko_prevented
    }


def evaluate_ko_genome(genome: KoGenome, n_games: int = 15) -> float:
    """
    Evaluate fitness of a Ko genome.

    Fitness:
    - Completed games: +1.0
    - Looped games: -10.0
    - Ko violations prevented: +0.1 each
    """
    total_fitness = 0.0
    games_completed = 0
    games_looped = 0
    total_ko_prevented = 0

    for _ in range(n_games):
        result = play_game_with_genome(genome)

        if result["completed"]:
            total_fitness += 1.0
            games_completed += 1
        elif result["looped"]:
            total_fitness -= 10.0
            games_looped += 1
        else:
            total_fitness += 0.3  # Didn't finish but didn't loop

        # Bonus for preventing Ko (only if it helped)
        if not result["looped"] and result["ko_prevented"] > 0:
            total_fitness += 0.1 * result["ko_prevented"]

        total_ko_prevented += result["ko_prevented"]

    genome.fitness = total_fitness / n_games
    genome.games_completed = games_completed
    genome.games_looped = games_looped
    genome.ko_violations_prevented = total_ko_prevented

    return genome.fitness


# =============================================================================
# Evolution Loop
# =============================================================================

def evolve_ko_topology(
    n_generations: int = 100,
    population_size: int = 30,
    games_per_eval: int = 15,
    elite_size: int = 3,
    verbose: bool = True
) -> KoGenome:
    """
    Evolve Ko handling through topology discovery.

    Key: history_depth starts at 0. Evolution must discover
    that depth=1 (or more) prevents loops.
    """

    if verbose:
        print("=" * 60)
        print("KO TOPOLOGY EVOLUTION")
        print("=" * 60)
        print(f"Population: {population_size}")
        print(f"Generations: {n_generations}")
        print("History depth starts at 0 - must be DISCOVERED")
        print("-" * 60)

    # Initialize population - history mostly disabled
    population = [create_random_ko_genome(n_states=5) for _ in range(population_size)]

    # Evaluate
    for genome in population:
        evaluate_ko_genome(genome, games_per_eval)

    best_ever = max(population, key=lambda g: g.fitness).copy()
    best_ever.fitness = max(population, key=lambda g: g.fitness).fitness

    for gen in range(n_generations):
        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Elite
        new_pop = [g.copy() for g in population[:elite_size]]

        # Generate offspring
        while len(new_pop) < population_size:
            if random.random() < 0.7:
                p1 = random.choice(population[:population_size//2])
                p2 = random.choice(population[:population_size//2])
                child = crossover_ko(p1, p2)
            else:
                parent = random.choice(population[:population_size//2])
                child = parent.copy()

            child = mutate_ko_genome(child)
            if child.is_valid():
                new_pop.append(child)

        # Evaluate
        for genome in new_pop:
            evaluate_ko_genome(genome, games_per_eval)

        population = new_pop

        # Track best
        gen_best = max(population, key=lambda g: g.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best.copy()
            best_ever.fitness = gen_best.fitness
            best_ever.games_completed = gen_best.games_completed
            best_ever.games_looped = gen_best.games_looped
            best_ever.ko_violations_prevented = gen_best.ko_violations_prevented

        # Progress
        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            # Analyze population history depths
            depths = [g.max_history_depth() for g in population]
            avg_depth = sum(depths) / len(depths)
            active = sum(1 for g in population if g.max_history_depth() > 0)

            print(f"Gen {gen:3d} | Best: {gen_best.fitness:+.2f} "
                  f"(completed={gen_best.games_completed}/{games_per_eval}, "
                  f"looped={gen_best.games_looped}) "
                  f"| Avg depth: {avg_depth:.2f} | Active history: {active}/{population_size}")

    if verbose:
        print("-" * 60)
        print(f"\nBEST EVOLVED TOPOLOGY:")
        print(f"  States: {best_ever.n_states}")
        print(f"  Max history depth: {best_ever.max_history_depth()}")
        print(f"  Active history states: {best_ever.active_history_states()}")
        print(f"  Fitness: {best_ever.fitness:+.2f}")
        print(f"  Games completed: {best_ever.games_completed}/{games_per_eval}")
        print(f"  Games looped: {best_ever.games_looped}/{games_per_eval}")
        print(f"  Ko prevented: {best_ever.ko_violations_prevented}")

        # Show history genes
        print(f"\n  History genes:")
        for i, gene in enumerate(best_ever.history_genes):
            if gene.depth > 0:
                print(f"    State {i}: depth={gene.depth}, encoding={gene.encoding.name}, decay={gene.decay_rate:.2f}")

        if best_ever.max_history_depth() >= 1 and best_ever.games_looped == 0:
            print(f"\n  *** HISTORY DEPTH DISCOVERED! depth={best_ever.max_history_depth()} solves Ko ***")

    return best_ever


# =============================================================================
# Test
# =============================================================================

def test_ko_topology():
    """Test Ko topology evolution."""
    print("Testing Ko Topology Evolution...\n")

    best = evolve_ko_topology(
        n_generations=50,
        population_size=25,
        games_per_eval=12,
        verbose=True
    )

    return best


if __name__ == "__main__":
    test_ko_topology()
