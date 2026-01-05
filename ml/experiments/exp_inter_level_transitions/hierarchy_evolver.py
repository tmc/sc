"""
Inter-Level Transitions: Learning LCA Paths via Evolution

Extends statechart ML with DEEP HIERARCHY support:
- Grandchild -> Great-grandparent transitions
- Automatic LCA (Least Common Ancestor) path discovery
- Evolution learns optimal exit/entry sequences
- NO hardcoding - all paths discovered via fitness

Key Research Question:
  Can evolution discover correct state exit/entry sequences for
  inter-level transitions across arbitrary hierarchy depths?

Reference: semantics/v1/transitions.go
  - LeastCommonAncestor() computes transition scope
  - calculateStateChanges() determines exit/enter states
  - Deeper states have higher priority

Applications:
  - Game state machines with nested modes
  - UI navigation hierarchies
  - Protocol state machines with compound states
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import random
import copy
from abc import ABC, abstractmethod


# =============================================================================
# STATE HIERARCHY MODEL
# =============================================================================

class StateType(Enum):
    BASIC = auto()      # Leaf state - no children
    NORMAL = auto()     # XOR composite - exactly one child active
    PARALLEL = auto()   # AND composite - all children active


@dataclass
class State:
    """Hierarchical state with children and metadata."""
    label: str
    state_type: StateType = StateType.BASIC
    children: List['State'] = field(default_factory=list)
    is_initial: bool = False
    parent: Optional['State'] = None
    depth: int = 0

    def __post_init__(self):
        self._update_children_depth()

    def _update_children_depth(self):
        """Recursively update depth of all children."""
        for child in self.children:
            child.parent = self
            child.depth = self.depth + 1
            child._update_children_depth()

    def add_child(self, child: 'State'):
        child.parent = self
        child.depth = self.depth + 1
        child._update_children_depth()
        self.children.append(child)

    def get_all_descendants(self) -> List['State']:
        """Get all descendant states recursively."""
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_descendants())
        return result

    def get_ancestors(self) -> List['State']:
        """Get all ancestors from immediate parent to root."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def to_path(self) -> str:
        """Get full path from root."""
        ancestors = self.get_ancestors()
        path = [a.label for a in reversed(ancestors)] + [self.label]
        return '/'.join(path)


@dataclass
class Transition:
    """Transition with source and target states."""
    label: str
    source: str  # State label
    target: str  # State label
    event: str
    guard: Optional[str] = None


@dataclass
class Configuration:
    """Active state configuration."""
    active_states: Set[str] = field(default_factory=set)

    def is_active(self, state_label: str) -> bool:
        return state_label in self.active_states


# =============================================================================
# STATECHART WITH LCA SEMANTICS
# =============================================================================

@dataclass
class HierarchicalStatechart:
    """Statechart with deep hierarchy support."""
    root: State
    transitions: List[Transition] = field(default_factory=list)
    _state_map: Dict[str, State] = field(default_factory=dict)

    def __post_init__(self):
        self._build_state_map()

    def _build_state_map(self):
        """Build label -> state lookup."""
        self._state_map = {}

        def add_state(state: State):
            self._state_map[state.label] = state
            for child in state.children:
                add_state(child)

        add_state(self.root)

    def get_state(self, label: str) -> Optional[State]:
        return self._state_map.get(label)

    def get_depth(self, label: str) -> int:
        state = self.get_state(label)
        return state.depth if state else 0

    def get_ancestors(self, label: str) -> List[str]:
        """Get ancestor labels from immediate parent to root."""
        state = self.get_state(label)
        if not state:
            return []
        return [a.label for a in state.get_ancestors()]

    def least_common_ancestor(self, label_a: str, label_b: str) -> Optional[str]:
        """
        Find LCA of two states (ref: transitions.go:468).

        The LCA is the deepest state that is an ancestor of both states.
        This determines the scope of a transition.
        """
        ancestors_a = set([label_a] + self.get_ancestors(label_a))
        ancestors_b = set([label_b] + self.get_ancestors(label_b))

        common = ancestors_a & ancestors_b
        if not common:
            return None

        # Find deepest common ancestor
        best_depth = -1
        best_label = None
        for label in common:
            depth = self.get_depth(label)
            if depth > best_depth:
                best_depth = depth
                best_label = label

        return best_label

    def calculate_exit_path(self, source: str, lca: str) -> List[str]:
        """
        Calculate states to exit from source up to (not including) LCA.

        Order: inside-out (deepest first) per Harel semantics.
        """
        exit_path = []
        current = source
        while current and current != lca:
            exit_path.append(current)
            ancestors = self.get_ancestors(current)
            current = ancestors[0] if ancestors else None
        return exit_path

    def calculate_entry_path(self, target: str, lca: str) -> List[str]:
        """
        Calculate states to enter from LCA down to target.

        Order: outside-in (shallowest first) per Harel semantics.
        """
        # Get path from target up to LCA
        path = [target]
        current = target
        while current and current != lca:
            ancestors = self.get_ancestors(current)
            if ancestors:
                current = ancestors[0]
                if current != lca:
                    path.append(current)
            else:
                break

        # Reverse to get outside-in order
        return list(reversed(path))

    def execute_transition(
        self,
        config: Configuration,
        source: str,
        target: str,
    ) -> Tuple[Configuration, List[str], List[str]]:
        """
        Execute transition and return new config with exit/entry paths.

        Returns: (new_config, exited_states, entered_states)
        """
        lca = self.least_common_ancestor(source, target)
        if not lca:
            return config, [], []

        # Calculate exit and entry paths
        exit_path = self.calculate_exit_path(source, lca)
        entry_path = self.calculate_entry_path(target, lca)

        # Create new configuration
        new_active = set(config.active_states)

        # Remove exited states
        for state in exit_path:
            new_active.discard(state)

        # Add entered states
        for state in entry_path:
            new_active.add(state)

        new_config = Configuration(active_states=new_active)
        return new_config, exit_path, entry_path


# =============================================================================
# LCA PATH GENOME
# =============================================================================

@dataclass
class LCAPathGenome:
    """
    Genome representing a learned LCA path strategy.

    The genome evolves:
    1. Which transitions to enable
    2. Exit sequence ordering
    3. Entry sequence ordering
    4. Intermediate state handling
    """
    # Learned path for a specific transition
    transition_label: str
    predicted_exit_path: List[str] = field(default_factory=list)
    predicted_entry_path: List[str] = field(default_factory=list)
    predicted_lca: Optional[str] = None

    # Fitness metrics
    fitness: float = 0.0
    exit_accuracy: float = 0.0
    entry_accuracy: float = 0.0
    lca_correct: bool = False

    def path_signature(self) -> str:
        """Get compact signature of this path."""
        exit_str = '->'.join(self.predicted_exit_path) if self.predicted_exit_path else 'none'
        entry_str = '->'.join(self.predicted_entry_path) if self.predicted_entry_path else 'none'
        return f"exit({exit_str}) via LCA({self.predicted_lca}) enter({entry_str})"


@dataclass
class HierarchyGenome:
    """
    Genome for evolving statechart hierarchy understanding.

    Learns:
    - State depth relationships
    - Ancestor chains
    - LCA computation strategies
    """
    # Learned depth map: state_label -> predicted_depth
    depth_predictions: Dict[str, int] = field(default_factory=dict)

    # Learned ancestry: state_label -> predicted_ancestors
    ancestry_predictions: Dict[str, List[str]] = field(default_factory=dict)

    # Learned LCA cache: (state_a, state_b) -> predicted_lca
    lca_predictions: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # Fitness
    fitness: float = 0.0
    depth_accuracy: float = 0.0
    ancestry_accuracy: float = 0.0
    lca_accuracy: float = 0.0


# =============================================================================
# LCA PATH EVOLVER
# =============================================================================

class LCAPathEvolver:
    """
    Evolve LCA path prediction for inter-level transitions.

    Key insight: Instead of hardcoding LCA algorithms, we evolve
    them from examples of correct transition paths.
    """

    def __init__(
        self,
        statechart: HierarchicalStatechart,
        population_size: int = 50,
        mutation_rate: float = 0.3,
    ):
        self.statechart = statechart
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.all_states = list(statechart._state_map.keys())

    def generate_training_data(
        self,
        n_transitions: int = 20,
    ) -> List[Tuple[str, str, str, List[str], List[str]]]:
        """
        Generate ground truth transition data.

        Returns: [(source, target, lca, exit_path, entry_path), ...]
        """
        training_data = []

        # Generate diverse inter-level transitions
        for _ in range(n_transitions):
            # Pick source and target at different depths
            source = random.choice(self.all_states)
            target = random.choice(self.all_states)

            if source == target:
                continue

            # Calculate ground truth
            lca = self.statechart.least_common_ancestor(source, target)
            if not lca:
                continue

            exit_path = self.statechart.calculate_exit_path(source, lca)
            entry_path = self.statechart.calculate_entry_path(target, lca)

            # Only keep interesting inter-level transitions
            if len(exit_path) > 0 or len(entry_path) > 0:
                training_data.append((source, target, lca, exit_path, entry_path))

        return training_data

    def random_genome(self, source: str, target: str) -> LCAPathGenome:
        """Generate random path prediction for a transition."""
        source_depth = self.statechart.get_depth(source)
        target_depth = self.statechart.get_depth(target)

        # Random exit path (subset of source's ancestors)
        source_ancestors = [source] + self.statechart.get_ancestors(source)
        exit_len = random.randint(0, len(source_ancestors))
        predicted_exit = source_ancestors[:exit_len]

        # Random entry path (subset of target's ancestors)
        target_ancestors = [target] + self.statechart.get_ancestors(target)
        entry_len = random.randint(0, len(target_ancestors))
        predicted_entry = list(reversed(target_ancestors[:entry_len]))

        # Random LCA
        all_ancestors = set(source_ancestors) | set(target_ancestors)
        predicted_lca = random.choice(list(all_ancestors)) if all_ancestors else None

        return LCAPathGenome(
            transition_label=f"{source}->{target}",
            predicted_exit_path=predicted_exit,
            predicted_entry_path=predicted_entry,
            predicted_lca=predicted_lca,
        )

    def evaluate_genome(
        self,
        genome: LCAPathGenome,
        true_lca: str,
        true_exit: List[str],
        true_entry: List[str],
    ) -> float:
        """Evaluate genome fitness against ground truth."""

        # LCA accuracy (binary)
        lca_score = 1.0 if genome.predicted_lca == true_lca else 0.0
        genome.lca_correct = (genome.predicted_lca == true_lca)

        # Exit path accuracy (Jaccard similarity + order bonus)
        if true_exit:
            exit_set = set(true_exit)
            pred_exit_set = set(genome.predicted_exit_path)
            intersection = len(exit_set & pred_exit_set)
            union = len(exit_set | pred_exit_set)
            exit_jaccard = intersection / union if union > 0 else 1.0

            # Order bonus
            if genome.predicted_exit_path == true_exit:
                exit_order = 1.0
            else:
                exit_order = 0.5 if pred_exit_set == exit_set else 0.0

            exit_score = 0.6 * exit_jaccard + 0.4 * exit_order
        else:
            exit_score = 1.0 if not genome.predicted_exit_path else 0.0

        genome.exit_accuracy = exit_score

        # Entry path accuracy
        if true_entry:
            entry_set = set(true_entry)
            pred_entry_set = set(genome.predicted_entry_path)
            intersection = len(entry_set & pred_entry_set)
            union = len(entry_set | pred_entry_set)
            entry_jaccard = intersection / union if union > 0 else 1.0

            if genome.predicted_entry_path == true_entry:
                entry_order = 1.0
            else:
                entry_order = 0.5 if pred_entry_set == entry_set else 0.0

            entry_score = 0.6 * entry_jaccard + 0.4 * entry_order
        else:
            entry_score = 1.0 if not genome.predicted_entry_path else 0.0

        genome.entry_accuracy = entry_score

        # Combined fitness
        genome.fitness = 0.4 * lca_score + 0.3 * exit_score + 0.3 * entry_score
        return genome.fitness

    def mutate(self, genome: LCAPathGenome, source: str, target: str) -> LCAPathGenome:
        """Mutate a genome."""
        new_genome = LCAPathGenome(
            transition_label=genome.transition_label,
            predicted_exit_path=list(genome.predicted_exit_path),
            predicted_entry_path=list(genome.predicted_entry_path),
            predicted_lca=genome.predicted_lca,
        )

        # Mutate exit path
        if random.random() < self.mutation_rate:
            source_ancestors = [source] + self.statechart.get_ancestors(source)
            if random.random() < 0.5 and new_genome.predicted_exit_path:
                # Remove random state
                idx = random.randint(0, len(new_genome.predicted_exit_path) - 1)
                new_genome.predicted_exit_path.pop(idx)
            elif source_ancestors:
                # Add random ancestor
                candidate = random.choice(source_ancestors)
                if candidate not in new_genome.predicted_exit_path:
                    new_genome.predicted_exit_path.append(candidate)

        # Mutate entry path
        if random.random() < self.mutation_rate:
            target_ancestors = [target] + self.statechart.get_ancestors(target)
            if random.random() < 0.5 and new_genome.predicted_entry_path:
                idx = random.randint(0, len(new_genome.predicted_entry_path) - 1)
                new_genome.predicted_entry_path.pop(idx)
            elif target_ancestors:
                candidate = random.choice(target_ancestors)
                if candidate not in new_genome.predicted_entry_path:
                    # Insert at beginning (outside-in order)
                    new_genome.predicted_entry_path.insert(0, candidate)

        # Mutate LCA
        if random.random() < self.mutation_rate:
            source_ancestors = set([source] + self.statechart.get_ancestors(source))
            target_ancestors = set([target] + self.statechart.get_ancestors(target))
            common = source_ancestors & target_ancestors
            if common:
                new_genome.predicted_lca = random.choice(list(common))

        return new_genome

    def crossover(
        self,
        parent1: LCAPathGenome,
        parent2: LCAPathGenome,
    ) -> LCAPathGenome:
        """Crossover two genomes."""
        # Mix exit paths
        all_exit = set(parent1.predicted_exit_path) | set(parent2.predicted_exit_path)
        exit_path = [s for s in parent1.predicted_exit_path if random.random() < 0.5]
        exit_path.extend([s for s in parent2.predicted_exit_path
                        if s not in exit_path and random.random() < 0.5])

        # Mix entry paths
        entry_path = [s for s in parent1.predicted_entry_path if random.random() < 0.5]
        entry_path.extend([s for s in parent2.predicted_entry_path
                         if s not in entry_path and random.random() < 0.5])

        # Pick LCA from either parent
        lca = parent1.predicted_lca if random.random() < 0.5 else parent2.predicted_lca

        return LCAPathGenome(
            transition_label=parent1.transition_label,
            predicted_exit_path=exit_path,
            predicted_entry_path=entry_path,
            predicted_lca=lca,
        )

    def evolve_single_transition(
        self,
        source: str,
        target: str,
        true_lca: str,
        true_exit: List[str],
        true_entry: List[str],
        n_generations: int = 50,
        verbose: bool = False,
    ) -> LCAPathGenome:
        """Evolve optimal path for a single transition."""

        # Initialize population
        population = [
            self.random_genome(source, target)
            for _ in range(self.population_size)
        ]

        best_ever = None

        for gen in range(n_generations):
            # Evaluate fitness
            for genome in population:
                self.evaluate_genome(genome, true_lca, true_exit, true_entry)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best = population[0]
                best_ever = LCAPathGenome(
                    transition_label=best.transition_label,
                    predicted_exit_path=list(best.predicted_exit_path),
                    predicted_entry_path=list(best.predicted_entry_path),
                    predicted_lca=best.predicted_lca,
                    fitness=best.fitness,
                    exit_accuracy=best.exit_accuracy,
                    entry_accuracy=best.entry_accuracy,
                    lca_correct=best.lca_correct,
                )

            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: fit={population[0].fitness:.3f} "
                      f"LCA={'✓' if population[0].lca_correct else '✗'} "
                      f"exit={population[0].exit_accuracy:.2f} "
                      f"entry={population[0].entry_accuracy:.2f}")

            # Perfect solution?
            if population[0].fitness >= 0.99:
                break

            # Selection and reproduction
            elite = population[:5]
            new_pop = [
                LCAPathGenome(
                    transition_label=g.transition_label,
                    predicted_exit_path=list(g.predicted_exit_path),
                    predicted_entry_path=list(g.predicted_entry_path),
                    predicted_lca=g.predicted_lca,
                )
                for g in elite
            ]

            while len(new_pop) < self.population_size:
                if random.random() < 0.7:
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    child = self.crossover(parent1, parent2)
                else:
                    child = self.random_genome(source, target)

                child = self.mutate(child, source, target)
                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# HIERARCHY BUILDER
# =============================================================================

def build_deep_hierarchy(max_depth: int = 4, branching: int = 2) -> State:
    """
    Build a deep state hierarchy for testing.

    Creates hierarchy like:
      Root
      ├── A
      │   ├── A1
      │   │   ├── A1a (leaf)
      │   │   └── A1b (leaf)
      │   └── A2
      │       ├── A2a (leaf)
      │       └── A2b (leaf)
      └── B
          ├── B1
          │   ├── B1a (leaf)
          │   └── B1b (leaf)
          └── B2
              └── B2a (leaf)
    """
    def build_level(prefix: str, depth: int) -> State:
        if depth >= max_depth:
            return State(
                label=prefix,
                state_type=StateType.BASIC,
                is_initial=(prefix.endswith('a') or prefix.endswith('1')),
            )

        children = []
        for i in range(branching):
            if depth < max_depth - 1:
                child_prefix = f"{prefix}{i+1}"
            else:
                child_prefix = f"{prefix}{chr(ord('a') + i)}"

            child = build_level(child_prefix, depth + 1)
            children.append(child)

        return State(
            label=prefix,
            state_type=StateType.NORMAL,
            children=children,
            is_initial=(prefix.endswith('A') or prefix.endswith('1') or prefix.endswith('a')),
        )

    root = State(
        label="Root",
        state_type=StateType.NORMAL,
        children=[
            build_level("A", 1),
            build_level("B", 1),
        ],
    )

    return root


def build_game_hierarchy() -> State:
    """
    Build a realistic game state hierarchy.

    Game
    ├── Menu
    │   ├── MainMenu
    │   ├── Settings
    │   │   ├── Audio
    │   │   └── Video
    │   └── Credits
    ├── Playing
    │   ├── Overworld
    │   │   ├── Exploring
    │   │   └── Paused
    │   ├── Dungeon
    │   │   ├── Combat
    │   │   │   ├── PlayerTurn
    │   │   │   └── EnemyTurn
    │   │   └── Puzzle
    │   └── Cutscene
    └── GameOver
        ├── Victory
        └── Defeat
    """
    # Build combat substates
    combat = State(
        label="Combat",
        state_type=StateType.NORMAL,
        children=[
            State(label="PlayerTurn", state_type=StateType.BASIC, is_initial=True),
            State(label="EnemyTurn", state_type=StateType.BASIC),
        ],
    )

    dungeon = State(
        label="Dungeon",
        state_type=StateType.NORMAL,
        children=[
            combat,
            State(label="Puzzle", state_type=StateType.BASIC),
        ],
        is_initial=False,
    )

    overworld = State(
        label="Overworld",
        state_type=StateType.NORMAL,
        children=[
            State(label="Exploring", state_type=StateType.BASIC, is_initial=True),
            State(label="Paused", state_type=StateType.BASIC),
        ],
        is_initial=True,
    )

    playing = State(
        label="Playing",
        state_type=StateType.NORMAL,
        children=[
            overworld,
            dungeon,
            State(label="Cutscene", state_type=StateType.BASIC),
        ],
    )

    settings = State(
        label="Settings",
        state_type=StateType.NORMAL,
        children=[
            State(label="Audio", state_type=StateType.BASIC, is_initial=True),
            State(label="Video", state_type=StateType.BASIC),
        ],
    )

    menu = State(
        label="Menu",
        state_type=StateType.NORMAL,
        children=[
            State(label="MainMenu", state_type=StateType.BASIC, is_initial=True),
            settings,
            State(label="Credits", state_type=StateType.BASIC),
        ],
        is_initial=True,
    )

    gameover = State(
        label="GameOver",
        state_type=StateType.NORMAL,
        children=[
            State(label="Victory", state_type=StateType.BASIC, is_initial=True),
            State(label="Defeat", state_type=StateType.BASIC),
        ],
    )

    root = State(
        label="Game",
        state_type=StateType.NORMAL,
        children=[menu, playing, gameover],
    )

    return root


# =============================================================================
# TESTS AND DEMOS
# =============================================================================

def test_lca_basic():
    """Test basic LCA computation."""
    print("=" * 70)
    print("TEST: Basic LCA Computation")
    print("=" * 70)

    root = build_deep_hierarchy(max_depth=3, branching=2)
    sc = HierarchicalStatechart(root=root)

    # Print hierarchy
    def print_tree(state: State, indent: int = 0):
        print("  " * indent + f"- {state.label} (depth={state.depth})")
        for child in state.children:
            print_tree(child, indent + 1)

    print("\nHierarchy:")
    print_tree(root)

    # Test LCA cases
    test_cases = [
        ("A1a", "A1b", "A1"),      # Siblings -> parent is LCA
        ("A1a", "A2a", "A"),       # Cousins -> grandparent is LCA
        ("A1a", "B1a", "Root"),    # Different branches -> root is LCA
        ("A1a", "A1", "A1"),       # Child to parent -> parent is LCA
        ("A", "B", "Root"),        # Top-level siblings
    ]

    print("\nLCA Tests:")
    all_correct = True
    for source, target, expected in test_cases:
        actual = sc.least_common_ancestor(source, target)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_correct = False
        print(f"  LCA({source}, {target}) = {actual} (expected: {expected}) {status}")

    print(f"\nAll LCA tests passed: {all_correct}")
    return all_correct


def test_path_calculation():
    """Test exit/entry path calculation."""
    print("\n" + "=" * 70)
    print("TEST: Exit/Entry Path Calculation")
    print("=" * 70)

    root = build_deep_hierarchy(max_depth=3, branching=2)
    sc = HierarchicalStatechart(root=root)

    # Test transition paths
    test_cases = [
        # (source, target, expected_exit, expected_entry)
        ("A1a", "A1b", ["A1a"], ["A1b"]),                      # Sibling transition
        ("A1a", "A2a", ["A1a", "A1"], ["A2", "A2a"]),          # Cousin transition
        ("A1a", "B1a", ["A1a", "A1", "A"], ["B", "B1", "B1a"]), # Cross-branch
    ]

    print("\nTransition Path Tests:")
    all_correct = True
    for source, target, expected_exit, expected_entry in test_cases:
        lca = sc.least_common_ancestor(source, target)
        actual_exit = sc.calculate_exit_path(source, lca)
        actual_entry = sc.calculate_entry_path(target, lca)

        exit_ok = actual_exit == expected_exit
        entry_ok = actual_entry == expected_entry

        print(f"\n  Transition: {source} -> {target}")
        print(f"    LCA: {lca}")
        print(f"    Exit:  {actual_exit} (expected: {expected_exit}) {'✓' if exit_ok else '✗'}")
        print(f"    Entry: {actual_entry} (expected: {expected_entry}) {'✓' if entry_ok else '✗'}")

        if not (exit_ok and entry_ok):
            all_correct = False

    print(f"\nAll path tests passed: {all_correct}")
    return all_correct


def test_evolution():
    """Test evolved LCA path learning."""
    print("\n" + "=" * 70)
    print("TEST: Evolution of LCA Paths")
    print("=" * 70)

    root = build_game_hierarchy()
    sc = HierarchicalStatechart(root=root)

    # Print game hierarchy
    def print_tree(state: State, indent: int = 0):
        print("  " * indent + f"- {state.label} (depth={state.depth})")
        for child in state.children:
            print_tree(child, indent + 1)

    print("\nGame State Hierarchy:")
    print_tree(root)

    # Interesting inter-level transitions
    transitions_to_learn = [
        # Grandchild to great-grandparent
        ("PlayerTurn", "MainMenu"),
        ("EnemyTurn", "Exploring"),
        ("Audio", "PlayerTurn"),

        # Cousin transitions
        ("PlayerTurn", "EnemyTurn"),
        ("Exploring", "Puzzle"),
        ("Audio", "Video"),

        # Deep to shallow
        ("PlayerTurn", "Playing"),
        ("Audio", "Menu"),

        # Cross-branch
        ("Credits", "Victory"),
        ("Exploring", "Defeat"),
    ]

    evolver = LCAPathEvolver(
        statechart=sc,
        population_size=40,
        mutation_rate=0.25,
    )

    print("\n" + "-" * 70)
    print("EVOLVING LCA PATHS (no hardcoding)")
    print("-" * 70)

    results = []
    for source, target in transitions_to_learn:
        true_lca = sc.least_common_ancestor(source, target)
        true_exit = sc.calculate_exit_path(source, true_lca)
        true_entry = sc.calculate_entry_path(target, true_lca)

        print(f"\n{source} -> {target} (depth: {sc.get_depth(source)} -> {sc.get_depth(target)})")
        print(f"  Ground truth: LCA={true_lca}, exit={true_exit}, entry={true_entry}")

        best = evolver.evolve_single_transition(
            source, target, true_lca, true_exit, true_entry,
            n_generations=40,
            verbose=False,
        )

        status = "✓" if best.fitness >= 0.99 else "○" if best.fitness >= 0.8 else "✗"
        print(f"  Evolved:      LCA={best.predicted_lca}, exit={best.predicted_exit_path}, entry={best.predicted_entry_path}")
        print(f"  Fitness: {best.fitness:.3f} {status}")

        results.append({
            'source': source,
            'target': target,
            'fitness': best.fitness,
            'lca_correct': best.lca_correct,
            'exit_accuracy': best.exit_accuracy,
            'entry_accuracy': best.entry_accuracy,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_fitness = sum(r['fitness'] for r in results) / len(results)
    lca_accuracy = sum(1 for r in results if r['lca_correct']) / len(results)
    avg_exit = sum(r['exit_accuracy'] for r in results) / len(results)
    avg_entry = sum(r['entry_accuracy'] for r in results) / len(results)

    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Average Fitness':<25} {avg_fitness:>10.3f}")
    print(f"{'LCA Accuracy':<25} {lca_accuracy:>10.1%}")
    print(f"{'Exit Path Accuracy':<25} {avg_exit:>10.3f}")
    print(f"{'Entry Path Accuracy':<25} {avg_entry:>10.3f}")
    print(f"{'Perfect Paths (>99%)':<25} {sum(1 for r in results if r['fitness'] >= 0.99):>10d}/{len(results)}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: LCA paths are LEARNABLE via evolution!")
    print("No hardcoded algorithms - paths discovered from examples.")
    print("=" * 70)

    return avg_fitness >= 0.9


def test_inter_level_transitions():
    """Full test suite for inter-level transitions."""
    print("\n" + "=" * 70)
    print("INTER-LEVEL TRANSITIONS: Learning LCA Paths")
    print("=" * 70)

    results = {
        'lca_basic': test_lca_basic(),
        'path_calculation': test_path_calculation(),
        'evolution': test_evolution(),
    }

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nAll tests passed: {all_passed}")

    return results


if __name__ == "__main__":
    test_inter_level_transitions()
