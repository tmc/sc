"""
Topology Evolution: Discover Statechart Structure Through Evolution

This experiment evolves the STRUCTURE of statecharts:
- How many states?
- Which states are parallel (AND) vs exclusive (OR)?
- Where should hierarchy be?
- Which states need history?

Genome Encoding:
- n_states: int
- parent[i]: which state is parent (-1 for root)
- state_type[i]: BASIC | OR | AND
- has_history[i]: bool
- transitions: list of (src, tgt, event, guard_idx)

Fitness = accuracy * (1 + parsimony_bonus) * interpretability

Key Question: Can evolution rediscover the hand-coded structure?

Now supports multiple games via environment abstraction!
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Type
from enum import IntEnum

import mlx.core as mx
import mlx.nn as nn

# Import environment abstraction
from .environments import (
    GameEnvironment, GameConfig,
    TicTacToeEnv, Connect4Env,
    generate_legal_move_dataset, get_environment
)


# =============================================================================
# State Types
# =============================================================================

class StateType(IntEnum):
    BASIC = 0   # Leaf state (no children)
    OR = 1      # Exclusive children (XOR semantics)
    AND = 2     # Parallel children (AND semantics)


# =============================================================================
# Genome Encoding
# =============================================================================

@dataclass
class Transition:
    """A transition in the statechart."""
    src: int          # Source state index
    tgt: int          # Target state index
    event: int        # Event index (game-specific: 0-8 for TicTacToe, 0-6 for Connect4)
    guard_idx: int    # Which guard function (0 = always true)


@dataclass
class StatechartGenome:
    """
    Genome encoding a statechart topology.

    State indices: 0 is always root, 1..n_states-1 are other states

    Now parametrized for different games:
    - n_events: number of possible events (moves) in the game
    - n_guards: number of guard functions available
    - max_states: upper limit for this game
    """
    n_states: int
    parent: List[int]           # parent[i] = -1 for root, else parent index
    state_type: List[StateType]
    has_history: List[bool]
    transitions: List[Transition]
    initial_state: int = 1      # Default initial state (first child of root)

    # Game-specific parameters
    n_events: int = 9           # Default for TicTacToe
    n_guards: int = 4           # Number of available guards
    max_states: int = 10        # Upper limit for evolution

    # Cached fitness
    fitness: float = 0.0
    accuracy: float = 0.0

    def __post_init__(self):
        """Validate genome structure."""
        assert len(self.parent) == self.n_states
        assert len(self.state_type) == self.n_states
        assert len(self.has_history) == self.n_states
        assert self.parent[0] == -1  # Root has no parent

    def get_children(self, state_idx: int) -> List[int]:
        """Get children of a state."""
        return [i for i in range(self.n_states) if self.parent[i] == state_idx]

    def get_leaves(self) -> List[int]:
        """Get leaf states (BASIC type)."""
        return [i for i in range(self.n_states)
                if self.state_type[i] == StateType.BASIC]

    def get_depth(self, state_idx: int) -> int:
        """Get depth of state in hierarchy."""
        depth = 0
        curr = state_idx
        while self.parent[curr] != -1:
            curr = self.parent[curr]
            depth += 1
        return depth

    def is_valid(self) -> bool:
        """Check if genome represents valid statechart."""
        # Check for cycles in parent hierarchy
        for i in range(self.n_states):
            visited = set()
            curr = i
            while curr != -1:
                if curr in visited:
                    return False  # Cycle detected
                visited.add(curr)
                curr = self.parent[curr]

        # Check that non-BASIC states have children
        for i in range(self.n_states):
            if self.state_type[i] != StateType.BASIC:
                if not self.get_children(i):
                    return False  # Composite state has no children

        # Check transitions reference valid states
        for t in self.transitions:
            if t.src < 0 or t.src >= self.n_states:
                return False
            if t.tgt < 0 or t.tgt >= self.n_states:
                return False

        return True

    def to_mermaid(self) -> str:
        """Generate Mermaid stateDiagram for visualization."""
        lines = ["stateDiagram-v2"]

        # State names
        state_names = [f"S{i}" for i in range(self.n_states)]
        state_names[0] = "Root"

        # Add states with types
        for i in range(1, self.n_states):  # Skip root
            type_suffix = ""
            if self.state_type[i] == StateType.AND:
                type_suffix = " [AND]"
            elif self.state_type[i] == StateType.OR and self.get_children(i):
                type_suffix = " [OR]"
            if self.has_history[i]:
                type_suffix += " [H]"

            if self.get_children(i):
                lines.append(f"    state \"{state_names[i]}{type_suffix}\" as {state_names[i]} {{")
                for child in self.get_children(i):
                    lines.append(f"        {state_names[child]}")
                lines.append("    }")

        # Add transitions
        for t in self.transitions:
            event_name = f"e{t.event}"
            guard_str = f" [g{t.guard_idx}]" if t.guard_idx > 0 else ""
            lines.append(f"    {state_names[t.src]} --> {state_names[t.tgt]} : {event_name}{guard_str}")

        # Initial state
        lines.append(f"    [*] --> {state_names[self.initial_state]}")

        return "\n".join(lines)


def create_random_genome(
    max_states: int = 10,
    n_events: int = 9,
    n_guards: int = 4
) -> StatechartGenome:
    """Create a random statechart genome."""
    # Random number of states (at least 3: root + 2 leaves)
    n_states = random.randint(3, max_states)

    # Build parent array (ensure valid tree)
    parent = [-1]  # Root
    for i in range(1, n_states):
        # Parent must be an existing state with index < i
        parent.append(random.randint(0, i - 1))

    # State types - leaves are BASIC, others are OR or AND
    state_type = []
    for i in range(n_states):
        children = [j for j in range(n_states) if parent[j] == i]
        if not children:
            state_type.append(StateType.BASIC)
        else:
            state_type.append(random.choice([StateType.OR, StateType.AND]))

    # History - random for non-root states
    has_history = [False] + [random.random() < 0.2 for _ in range(n_states - 1)]

    # Transitions - connect leaf states with events
    leaves = [i for i in range(n_states) if state_type[i] == StateType.BASIC]
    transitions = []

    # Add some random transitions
    n_transitions = random.randint(len(leaves), len(leaves) * 3)
    for _ in range(n_transitions):
        src = random.choice(leaves)
        tgt = random.choice(leaves)
        event = random.randint(0, n_events - 1)
        guard_idx = random.randint(0, n_guards - 1)
        transitions.append(Transition(src, tgt, event, guard_idx))

    # Initial state - first leaf
    initial = leaves[0] if leaves else 1

    genome = StatechartGenome(
        n_states=n_states,
        parent=parent,
        state_type=state_type,
        has_history=has_history,
        transitions=transitions,
        initial_state=initial,
        n_events=n_events,
        n_guards=n_guards,
        max_states=max_states
    )

    return genome


def create_genome_for_game(
    env_class: Type[GameEnvironment],
    max_states: int = 20
) -> StatechartGenome:
    """
    Create a random genome sized appropriately for a game.

    Args:
        env_class: GameEnvironment class (e.g., TicTacToeEnv)
        max_states: Maximum number of states to evolve
    """
    config = env_class.config
    n_events = config.n_positions  # One event per possible move
    n_guards = 4 + len(config.base_features)  # Base guards + feature-based

    return create_random_genome(
        max_states=max_states,
        n_events=n_events,
        n_guards=n_guards
    )


# =============================================================================
# Legacy TicTacToe (removed - use environments.py instead)
# =============================================================================

# TicTacToe and Connect4 are now in environments.py
# Use TicTacToeEnv and Connect4Env instead


def generate_legal_move_dataset(n_positions: int = 1000) -> List[Tuple[Tuple[int, ...], List[bool]]]:
    """
    DEPRECATED: Use generate_env_dataset() instead.

    Generate dataset of (position, legal_moves) pairs for TicTacToe.
    Returns list of (board_state, [is_legal_0, is_legal_1, ..., is_legal_8])
    """
    # Use new environment-based function
    env_dataset = generate_env_dataset(TicTacToeEnv, n_positions)
    # Convert to old format (state tuple instead of env)
    return [(env.get_state(), legal_mask) for env, legal_mask in env_dataset]


# =============================================================================
# Statechart Execution (Simplified)
# =============================================================================

class StatechartExecutor:
    """
    Execute a statechart genome on any game environment.

    Now supports multiple games via environment abstraction.
    """

    def __init__(
        self,
        genome: StatechartGenome,
        env_class: Type[GameEnvironment] = None
    ):
        self.genome = genome
        self.env_class = env_class or TicTacToeEnv
        self.n_events = genome.n_events
        self.n_guards = genome.n_guards

        # Build transition lookup: (state, event) -> [(target, guard)]
        self.transition_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for t in genome.transitions:
            key = (t.src, t.event)
            if key not in self.transition_map:
                self.transition_map[key] = []
            self.transition_map[key].append((t.tgt, t.guard_idx))

    def evaluate_guard(
        self,
        guard_idx: int,
        env: GameEnvironment,
        move: int
    ) -> bool:
        """
        Evaluate guard condition.

        Base guards (0-3):
            0: Always true
            1: Move is legal (cell empty / column not full)
            2: Early game (< half moves played)
            3: Strategic position (corner/center for TicTacToe, center columns for Connect4)

        Extended guards (4+) based on game features.
        """
        if guard_idx == 0:
            return True  # Always true

        elif guard_idx == 1:
            return env.is_legal(move)  # Move is legal

        elif guard_idx == 2:
            # Early game
            state = env.get_state()
            move_count = sum(1 for c in state if c != 0)
            max_moves = len(state)
            return move_count < max_moves / 2

        elif guard_idx == 3:
            # Strategic position - game-specific
            if self.env_class == TicTacToeEnv:
                return move in [0, 2, 4, 6, 8]  # Corner or center
            elif self.env_class == Connect4Env:
                return move in [2, 3, 4]  # Center columns
            return True

        else:
            # Extended guards based on features
            features = env.get_features()
            feature_list = list(features.values())
            feature_idx = guard_idx - 4
            if feature_idx < len(feature_list):
                # Guard is true if feature > 0.5
                return feature_list[feature_idx] > 0.5
            return True

    def predict_legal_moves(
        self,
        env: GameEnvironment,
        current_state: int
    ) -> List[bool]:
        """
        Predict which moves are legal from current state.
        A move is predicted legal if there's an enabled transition for it.
        """
        predictions = [False] * self.n_events

        # Check each possible move (event)
        for move in range(self.n_events):
            key = (current_state, move)
            if key in self.transition_map:
                # Check if any transition is enabled (guard evaluates to true)
                for tgt, guard_idx in self.transition_map[key]:
                    if self.evaluate_guard(guard_idx, env, move):
                        predictions[move] = True
                        break

        return predictions

    def step(
        self,
        current_state: int,
        env: GameEnvironment,
        event: int
    ) -> int:
        """Take a step in the statechart. Returns new state."""
        key = (current_state, event)
        if key in self.transition_map:
            for tgt, guard_idx in self.transition_map[key]:
                if self.evaluate_guard(guard_idx, env, event):
                    return tgt
        return current_state  # No transition, stay in current state


# =============================================================================
# Fitness Evaluation
# =============================================================================

def evaluate_fitness(
    genome: StatechartGenome,
    dataset: List[Tuple[GameEnvironment, List[bool]]],
    env_class: Type[GameEnvironment] = None,
    max_states: int = 10
) -> Tuple[float, float]:
    """
    Evaluate fitness of a genome on any game.
    Returns (total_fitness, accuracy).

    fitness = accuracy * (1 + parsimony_bonus)

    Args:
        genome: The statechart genome to evaluate
        dataset: List of (environment, legal_mask) pairs
        env_class: Game environment class
        max_states: Maximum states for parsimony calculation
    """
    if not genome.is_valid():
        return 0.0, 0.0

    env_class = env_class or TicTacToeEnv
    executor = StatechartExecutor(genome, env_class)

    # Evaluate accuracy on dataset
    correct = 0
    total = 0

    n_events = genome.n_events

    for env, legal_mask in dataset:
        # Use initial state for prediction
        predictions = executor.predict_legal_moves(env, genome.initial_state)

        for i in range(n_events):
            if predictions[i] == legal_mask[i]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Parsimony bonus: fewer states = better
    parsimony_bonus = 0.1 * (max_states - genome.n_states) / max_states

    # Total fitness
    fitness = accuracy * (1 + parsimony_bonus)

    return fitness, accuracy


def generate_env_dataset(
    env_class: Type[GameEnvironment],
    n_positions: int = 1000
) -> List[Tuple[GameEnvironment, List[bool]]]:
    """
    Generate dataset of (environment, legal_moves) pairs.

    Returns list of (env_instance, legal_mask) tuples.
    """
    dataset = []

    for _ in range(n_positions):
        env = env_class.generate_random_position()
        if env.is_terminal():
            continue

        legal_mask = env.get_legal_mask()
        dataset.append((env, legal_mask))

    return dataset


# =============================================================================
# Evolution Operators
# =============================================================================

def mutate(genome: StatechartGenome, mutation_rate: float = 0.3) -> StatechartGenome:
    """Apply random mutations to a genome."""
    g = copy.deepcopy(genome)

    # Mutation: Add state
    if random.random() < mutation_rate and g.n_states < 15:
        parent_idx = random.randint(0, g.n_states - 1)
        g.n_states += 1
        g.parent.append(parent_idx)
        g.state_type.append(StateType.BASIC)
        g.has_history.append(random.random() < 0.2)

        # Update parent's type if it was BASIC
        if g.state_type[parent_idx] == StateType.BASIC:
            g.state_type[parent_idx] = random.choice([StateType.OR, StateType.AND])

    # Mutation: Remove state (if not root and has no children)
    if random.random() < mutation_rate and g.n_states > 3:
        removable = [i for i in range(1, g.n_states)
                     if not g.get_children(i)]
        if removable:
            to_remove = random.choice(removable)
            # Remove transitions involving this state
            g.transitions = [t for t in g.transitions
                           if t.src != to_remove and t.tgt != to_remove]
            # Update indices
            g.parent = [p if p < to_remove else (p - 1 if p > to_remove else p)
                       for i, p in enumerate(g.parent) if i != to_remove]
            g.state_type = [g.state_type[i] for i in range(g.n_states) if i != to_remove]
            g.has_history = [g.has_history[i] for i in range(g.n_states) if i != to_remove]
            g.n_states -= 1

            # Update transition indices
            for t in g.transitions:
                if t.src > to_remove:
                    t.src -= 1
                if t.tgt > to_remove:
                    t.tgt -= 1

            # Update initial state
            if g.initial_state == to_remove:
                leaves = g.get_leaves()
                g.initial_state = leaves[0] if leaves else 1
            elif g.initial_state > to_remove:
                g.initial_state -= 1

    # Mutation: Change state type
    if random.random() < mutation_rate:
        idx = random.randint(1, g.n_states - 1)
        children = g.get_children(idx)
        if children:
            # Composite state - toggle OR/AND
            g.state_type[idx] = StateType.AND if g.state_type[idx] == StateType.OR else StateType.OR

    # Mutation: Toggle history
    if random.random() < mutation_rate:
        idx = random.randint(1, g.n_states - 1)
        g.has_history[idx] = not g.has_history[idx]

    # Mutation: Add transition
    if random.random() < mutation_rate:
        leaves = g.get_leaves()
        if len(leaves) >= 2:
            src = random.choice(leaves)
            tgt = random.choice(leaves)
            event = random.randint(0, 8)
            guard = random.randint(0, 3)
            g.transitions.append(Transition(src, tgt, event, guard))

    # Mutation: Remove transition
    if random.random() < mutation_rate and len(g.transitions) > 1:
        idx = random.randint(0, len(g.transitions) - 1)
        g.transitions.pop(idx)

    # Mutation: Modify transition
    if random.random() < mutation_rate and g.transitions:
        idx = random.randint(0, len(g.transitions) - 1)
        leaves = g.get_leaves()
        if leaves:
            g.transitions[idx].src = random.choice(leaves)
            g.transitions[idx].tgt = random.choice(leaves)
            g.transitions[idx].event = random.randint(0, 8)
            g.transitions[idx].guard_idx = random.randint(0, 3)

    return g


def crossover(parent1: StatechartGenome, parent2: StatechartGenome) -> StatechartGenome:
    """Create offspring by combining two parents."""
    # Simple crossover: take structure from p1, transitions from p2
    child = copy.deepcopy(parent1)

    # Mix transitions
    n_from_p2 = len(parent2.transitions) // 2
    child.transitions = child.transitions[:len(child.transitions)//2]

    # Add compatible transitions from p2
    for t in random.sample(parent2.transitions, min(n_from_p2, len(parent2.transitions))):
        # Remap state indices if within range
        if t.src < child.n_states and t.tgt < child.n_states:
            child.transitions.append(copy.deepcopy(t))

    return child


def tournament_select(population: List[StatechartGenome], k: int = 3) -> StatechartGenome:
    """Select individual via tournament selection."""
    tournament = random.sample(population, min(k, len(population)))
    return max(tournament, key=lambda g: g.fitness)


# =============================================================================
# Main Evolution Loop
# =============================================================================

def evolve_topology(
    env_class: Type[GameEnvironment] = None,
    n_generations: int = 100,
    population_size: int = 50,
    elite_size: int = 5,
    max_states: int = 10,
    dataset_size: int = 500,
    verbose: bool = True
) -> StatechartGenome:
    """
    Main evolution loop for any game environment.
    Returns best genome found.

    Args:
        env_class: GameEnvironment class (e.g., TicTacToeEnv, Connect4Env)
        n_generations: Number of generations to evolve
        population_size: Number of individuals in population
        elite_size: Number of best individuals to keep each generation
        max_states: Maximum states per genome
        dataset_size: Number of positions for fitness evaluation
        verbose: Print progress updates
    """
    env_class = env_class or TicTacToeEnv
    config = env_class.config

    print(f"Game: {config.name}")
    print(f"Board positions: {config.n_positions}")

    # Generate dataset using new environment-based function
    print("Generating dataset...")
    dataset = generate_env_dataset(env_class, dataset_size)
    print(f"Generated {len(dataset)} positions")

    # Initialize population with game-appropriate genomes
    print("Initializing population...")
    population = [
        create_genome_for_game(env_class, max_states)
        for _ in range(population_size)
    ]

    # Evaluate initial fitness
    for g in population:
        g.fitness, g.accuracy = evaluate_fitness(g, dataset, env_class, max_states)

    best_ever = max(population, key=lambda g: g.fitness)

    print(f"\nStarting evolution for {n_generations} generations...")
    print("=" * 60)

    for gen in range(n_generations):
        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Keep elite
        new_population = population[:elite_size]

        # Generate offspring
        while len(new_population) < population_size:
            if random.random() < 0.7:
                # Crossover
                p1 = tournament_select(population)
                p2 = tournament_select(population)
                child = crossover(p1, p2)
            else:
                # Mutation only
                parent = tournament_select(population)
                child = copy.deepcopy(parent)

            # Always mutate
            child = mutate(child)

            # Ensure validity
            if child.is_valid():
                child.fitness, child.accuracy = evaluate_fitness(
                    child, dataset, env_class, max_states
                )
                new_population.append(child)

        population = new_population

        # Track best
        gen_best = max(population, key=lambda g: g.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(gen_best)

        # Progress report
        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            avg_fitness = sum(g.fitness for g in population) / len(population)
            avg_states = sum(g.n_states for g in population) / len(population)
            print(f"Gen {gen:3d} | Best: {gen_best.fitness:.4f} (acc={gen_best.accuracy:.3f}, "
                  f"states={gen_best.n_states}) | Avg: {avg_fitness:.4f} | Avg states: {avg_states:.1f}")

    print("=" * 60)
    print(f"\nBest genome found:")
    print(f"  Fitness: {best_ever.fitness:.4f}")
    print(f"  Accuracy: {best_ever.accuracy:.3f}")
    print(f"  States: {best_ever.n_states}")
    print(f"  Transitions: {len(best_ever.transitions)}")
    print(f"  Events: {best_ever.n_events}")
    print(f"  Guards: {best_ever.n_guards}")

    return best_ever


def analyze_best(genome: StatechartGenome):
    """Analyze and visualize the best genome."""
    print("\n" + "=" * 60)
    print("TOPOLOGY ANALYSIS")
    print("=" * 60)

    # Structure analysis
    print(f"\nStructure:")
    print(f"  Total states: {genome.n_states}")
    print(f"  Leaf states: {len(genome.get_leaves())}")

    # Count state types
    n_or = sum(1 for t in genome.state_type if t == StateType.OR)
    n_and = sum(1 for t in genome.state_type if t == StateType.AND)
    n_basic = sum(1 for t in genome.state_type if t == StateType.BASIC)
    print(f"  OR states: {n_or}")
    print(f"  AND states: {n_and}")
    print(f"  BASIC states: {n_basic}")

    # History
    n_history = sum(1 for h in genome.has_history if h)
    print(f"  States with history: {n_history}")

    # Transitions
    print(f"\nTransitions: {len(genome.transitions)}")

    # Events used
    events_used = set(t.event for t in genome.transitions)
    print(f"  Events used: {sorted(events_used)}")

    # Guards used
    guards_used = set(t.guard_idx for t in genome.transitions)
    print(f"  Guards used: {sorted(guards_used)}")

    # Mermaid diagram
    print("\n" + "=" * 60)
    print("MERMAID DIAGRAM")
    print("=" * 60)
    print(genome.to_mermaid())

    # Key question
    print("\n" + "=" * 60)
    print("KEY QUESTION: Did evolution rediscover structure?")
    print("=" * 60)
    print("""
TicTacToe has natural structure:
- 9 positions (cells)
- Each position can be: Empty, X, O
- Legal moves = empty cells

Hand-coded statechart would have:
- 9 OR states (one per cell) OR
- Single state with 9 transitions (one per move)

What did evolution find?
""")
    print(f"Evolution found {genome.n_states} states with {len(genome.transitions)} transitions")

    if genome.n_states <= 3 and len(genome.transitions) >= 9:
        print("-> MINIMAL: Single composite state, many transitions")
        print("   This is the 'flat' representation")
    elif n_and > 0:
        print("-> PARALLEL: Uses AND states")
        print("   Evolution discovered concurrent regions!")
    elif genome.n_states >= 9:
        print("-> EXPANDED: Many states")
        print("   Could map to board positions or game phases")
    else:
        print("-> HYBRID: Mix of states and transitions")


def main(game: str = "tictactoe"):
    """
    Run the topology evolution experiment.

    Args:
        game: Game to evolve on ("tictactoe" or "connect4")
    """
    print("=" * 60)
    print("EXPERIMENT: TOPOLOGY EVOLUTION")
    print("Discovering statechart structure through evolution")
    print("=" * 60)

    # Get environment class
    env_class = get_environment(game)
    config = env_class.config

    print(f"\nGame: {config.name}")
    print(f"Board positions: {config.n_positions}")
    print(f"Max game length: {config.max_game_length}")
    print("Goal: Can evolution rediscover the 'natural' structure?")
    print()

    # Set seed for reproducibility
    random.seed(42)

    # Run evolution
    best = evolve_topology(
        env_class=env_class,
        n_generations=100,
        population_size=50,
        elite_size=5,
        max_states=12 if game == "tictactoe" else 20,
        dataset_size=500,
        verbose=True
    )

    # Analyze result
    analyze_best(best)

    return best


if __name__ == "__main__":
    import sys

    # Parse command line argument for game
    game = sys.argv[1] if len(sys.argv) > 1 else "tictactoe"

    print(f"Running topology evolution on: {game}")
    best_genome = main(game)
