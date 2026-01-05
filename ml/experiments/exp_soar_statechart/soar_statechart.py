"""
SOAR-Inspired Statechart Program Synthesis

Programs are represented as statechart traversals:
- States = program points / operations
- Transitions = control flow with guards
- Actions = grid transformations (for ARC)

Key insight: Statechart structure constrains the search space,
making evolutionary refinement more efficient.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import copy


# =============================================================================
# ARC Grid Operations (primitives for statechart actions)
# =============================================================================

class GridOp(Enum):
    """Primitive operations on ARC grids."""
    IDENTITY = auto()      # No change
    FILL = auto()          # Fill region with color
    COPY = auto()          # Copy region
    ROTATE = auto()        # Rotate 90/180/270
    FLIP_H = auto()        # Horizontal flip
    FLIP_V = auto()        # Vertical flip
    SCALE = auto()         # Scale up/down
    TRANSLATE = auto()     # Move region
    MASK = auto()          # Apply mask
    FLOOD_FILL = auto()    # Connected component fill
    EXTRACT = auto()       # Extract pattern
    TILE = auto()          # Tile pattern
    OVERLAY = auto()       # Overlay patterns


@dataclass
class GridAction:
    """An action that transforms an ARC grid."""
    op: GridOp
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, grid: mx.array) -> mx.array:
        """Apply this action to a grid."""
        if self.op == GridOp.IDENTITY:
            return grid
        elif self.op == GridOp.FILL:
            color = self.params.get("color", 0)
            mask = self.params.get("mask", None)
            if mask is not None:
                return mx.where(mask, color, grid)
            return mx.full_like(grid, color)
        elif self.op == GridOp.ROTATE:
            k = self.params.get("k", 1)  # 90 degrees * k
            # MLX doesn't have rot90, use transpose + flip
            for _ in range(k % 4):
                grid = mx.transpose(grid)[::-1, :]
            return grid
        elif self.op == GridOp.FLIP_H:
            return grid[:, ::-1]
        elif self.op == GridOp.FLIP_V:
            return grid[::-1, :]
        # Add more operations as needed
        return grid


# =============================================================================
# Statechart States for Programs
# =============================================================================

class StateType(Enum):
    """Types of states in program statechart."""
    BASIC = auto()      # Leaf state - executes action
    OR = auto()         # Choice - one child active
    AND = auto()        # Parallel - all children active
    INITIAL = auto()    # Start state
    FINAL = auto()      # Accepting state
    HISTORY = auto()    # Remembers last active child


@dataclass
class ProgramState:
    """A state in the program statechart."""
    label: str
    state_type: StateType = StateType.BASIC
    action: Optional[GridAction] = None
    children: List["ProgramState"] = field(default_factory=list)
    is_initial: bool = False
    is_final: bool = False

    # For learning
    embedding: Optional[mx.array] = None

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not isinstance(other, ProgramState):
            return False
        return self.label == other.label


@dataclass
class Guard:
    """Guard condition for transitions."""
    # Expression tree for guard
    predicate: str  # e.g., "grid[0,0] == 1", "has_pattern(grid, P)"
    params: Dict[str, Any] = field(default_factory=dict)

    # Learned guard embedding
    embedding: Optional[mx.array] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate guard in context."""
        # Simple evaluation - in practice use AST
        grid = context.get("grid")
        if self.predicate == "true":
            return True
        elif self.predicate == "has_nonzero":
            return bool(mx.any(grid != 0))
        elif self.predicate == "is_symmetric":
            return bool(mx.all(grid == grid[::-1, ::-1]))
        # Add more predicates
        return True


@dataclass
class Transition:
    """A transition between program states."""
    source: str  # State label
    target: str  # State label
    event: str = ""  # Trigger event (empty = completion)
    guard: Optional[Guard] = None
    priority: int = 0  # For conflict resolution

    def is_enabled(self, context: Dict[str, Any]) -> bool:
        """Check if transition is enabled."""
        if self.guard is None:
            return True
        return self.guard.evaluate(context)


# =============================================================================
# Statechart Program
# =============================================================================

@dataclass
class StatechartProgram:
    """A program represented as a statechart."""
    states: Dict[str, ProgramState] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    root_label: str = "root"

    # Execution state
    current_config: Set[str] = field(default_factory=set)
    history: Dict[str, str] = field(default_factory=dict)  # For history states

    # Fitness tracking (SOAR-style)
    fitness: float = 0.0
    solved_tasks: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.current_config and self.root_label in self.states:
            self._enter_initial()

    def _enter_initial(self):
        """Enter initial configuration."""
        self.current_config = set()
        self._enter_state(self.root_label)

    def _enter_state(self, label: str):
        """Enter a state and its initial children."""
        if label not in self.states:
            return

        state = self.states[label]
        self.current_config.add(label)

        # For OR states, enter initial child
        if state.state_type == StateType.OR:
            for child in state.children:
                if child.is_initial:
                    self._enter_state(child.label)
                    break
        # For AND states, enter all children
        elif state.state_type == StateType.AND:
            for child in state.children:
                self._enter_state(child.label)

    def step(self, context: Dict[str, Any]) -> Tuple[mx.array, bool]:
        """Execute one step of the program.

        Returns:
            (transformed_grid, is_done)
        """
        grid = context.get("grid")

        # Find enabled transitions from current config
        enabled = []
        for trans in self.transitions:
            if trans.source in self.current_config:
                if trans.is_enabled(context):
                    enabled.append(trans)

        if not enabled:
            # No transitions - check if in final state
            is_done = any(
                self.states[s].is_final
                for s in self.current_config
                if s in self.states
            )
            return grid, is_done

        # Select highest priority transition
        enabled.sort(key=lambda t: -t.priority)
        trans = enabled[0]

        # Exit source, enter target
        self.current_config.discard(trans.source)
        self._enter_state(trans.target)

        # Execute action at target state
        target_state = self.states.get(trans.target)
        if target_state and target_state.action:
            grid = target_state.action.apply(grid)

        context["grid"] = grid

        # Check if done
        is_done = any(
            self.states[s].is_final
            for s in self.current_config
            if s in self.states
        )

        return grid, is_done

    def execute(self, input_grid: mx.array, max_steps: int = 100) -> mx.array:
        """Execute the program on an input grid."""
        self._enter_initial()
        context = {"grid": input_grid}

        for _ in range(max_steps):
            grid, done = self.step(context)
            if done:
                break

        return context["grid"]

    def clone(self) -> "StatechartProgram":
        """Deep copy for mutation."""
        return copy.deepcopy(self)


# =============================================================================
# SOAR-Style Statechart Synthesizer
# =============================================================================

class SOARStatechart:
    """
    SOAR-inspired statechart program synthesis.

    Combines:
    1. Evolutionary refinement (REX-style)
    2. Gradient-based guard learning
    3. Hindsight relabeling for sample efficiency
    """

    def __init__(
        self,
        max_states: int = 20,
        max_transitions: int = 50,
        population_size: int = 50,
        archive_size: int = 1000,
    ):
        self.max_states = max_states
        self.max_transitions = max_transitions
        self.population_size = population_size
        self.archive_size = archive_size

        # Population and archive
        self.population: List[StatechartProgram] = []
        self.archive: List[StatechartProgram] = []

        # Primitive operations
        self.grid_ops = list(GridOp)
        self.predicates = [
            "true", "has_nonzero", "is_symmetric",
            "grid[0,0] == 1", "shape_match"
        ]

    def random_program(self) -> StatechartProgram:
        """Generate a random statechart program."""
        prog = StatechartProgram()

        # Create root state
        root = ProgramState(
            label="root",
            state_type=StateType.OR,
            is_initial=True,
        )
        prog.states["root"] = root

        # Add random number of states
        n_states = random.randint(3, self.max_states)
        for i in range(n_states):
            label = f"s{i}"
            action = GridAction(
                op=random.choice(self.grid_ops),
                params={"color": random.randint(0, 9)}
            )
            state = ProgramState(
                label=label,
                state_type=StateType.BASIC,
                action=action,
                is_initial=(i == 0),
                is_final=(i == n_states - 1),
            )
            prog.states[label] = state
            root.children.append(state)

        # Add random transitions
        state_labels = [f"s{i}" for i in range(n_states)]
        n_trans = random.randint(n_states, min(n_states * 2, self.max_transitions))

        for _ in range(n_trans):
            src = random.choice(state_labels)
            tgt = random.choice(state_labels)
            guard = Guard(predicate=random.choice(self.predicates))
            trans = Transition(
                source=src,
                target=tgt,
                guard=guard,
                priority=random.randint(0, 10),
            )
            prog.transitions.append(trans)

        return prog

    def mutate(self, program: StatechartProgram) -> StatechartProgram:
        """Apply mutation operator to program."""
        prog = program.clone()

        mutation_type = random.choice([
            "add_state", "remove_state", "modify_action",
            "add_transition", "remove_transition", "modify_guard"
        ])

        if mutation_type == "add_state" and len(prog.states) < self.max_states:
            label = f"s{len(prog.states)}"
            action = GridAction(
                op=random.choice(self.grid_ops),
                params={"color": random.randint(0, 9)}
            )
            state = ProgramState(label=label, action=action)
            prog.states[label] = state
            if "root" in prog.states:
                prog.states["root"].children.append(state)

        elif mutation_type == "remove_state" and len(prog.states) > 3:
            removable = [
                s for s in prog.states.values()
                if not s.is_initial and not s.is_final and s.label != "root"
            ]
            if removable:
                state = random.choice(removable)
                del prog.states[state.label]
                prog.transitions = [
                    t for t in prog.transitions
                    if t.source != state.label and t.target != state.label
                ]

        elif mutation_type == "modify_action":
            basic_states = [
                s for s in prog.states.values()
                if s.state_type == StateType.BASIC
            ]
            if basic_states:
                state = random.choice(basic_states)
                state.action = GridAction(
                    op=random.choice(self.grid_ops),
                    params={"color": random.randint(0, 9)}
                )

        elif mutation_type == "add_transition" and len(prog.transitions) < self.max_transitions:
            state_labels = list(prog.states.keys())
            if len(state_labels) >= 2:
                src = random.choice(state_labels)
                tgt = random.choice(state_labels)
                guard = Guard(predicate=random.choice(self.predicates))
                trans = Transition(source=src, target=tgt, guard=guard)
                prog.transitions.append(trans)

        elif mutation_type == "remove_transition" and len(prog.transitions) > 1:
            prog.transitions.pop(random.randint(0, len(prog.transitions) - 1))

        elif mutation_type == "modify_guard":
            if prog.transitions:
                trans = random.choice(prog.transitions)
                trans.guard = Guard(predicate=random.choice(self.predicates))

        return prog

    def crossover(
        self,
        parent1: StatechartProgram,
        parent2: StatechartProgram
    ) -> StatechartProgram:
        """Crossover two programs (subtree exchange)."""
        child = parent1.clone()

        # Exchange some states
        for label, state in parent2.states.items():
            if label != "root" and random.random() < 0.5:
                child.states[label] = copy.deepcopy(state)

        # Exchange some transitions
        for trans in parent2.transitions:
            if random.random() < 0.3:
                child.transitions.append(copy.deepcopy(trans))

        # Clean up invalid transitions
        valid_labels = set(child.states.keys())
        child.transitions = [
            t for t in child.transitions
            if t.source in valid_labels and t.target in valid_labels
        ]

        return child

    def evaluate(
        self,
        program: StatechartProgram,
        tasks: List[Tuple[mx.array, mx.array]]
    ) -> float:
        """Evaluate program fitness on tasks."""
        correct = 0
        for input_grid, expected_output in tasks:
            try:
                output = program.execute(input_grid)
                if mx.all(output == expected_output):
                    correct += 1
            except Exception:
                pass

        return correct / len(tasks) if tasks else 0.0

    def evolve(
        self,
        tasks: List[Tuple[mx.array, mx.array]],
        n_generations: int = 100,
        verbose: bool = True,
    ) -> StatechartProgram:
        """
        SOAR-style evolutionary refinement.

        REX algorithm:
        1. Generate initial population
        2. Evaluate fitness
        3. Select parents from archive + population
        4. Apply mutations and crossover
        5. Update archive with diverse solutions
        """
        # Initialize population
        if not self.population:
            self.population = [
                self.random_program()
                for _ in range(self.population_size)
            ]

        best_program = None
        best_fitness = 0.0

        for gen in range(n_generations):
            # Evaluate population
            for prog in self.population:
                prog.fitness = self.evaluate(prog, tasks)
                if prog.fitness > best_fitness:
                    best_fitness = prog.fitness
                    best_program = prog.clone()

            # Update archive
            for prog in self.population:
                if len(self.archive) < self.archive_size:
                    self.archive.append(prog.clone())
                elif prog.fitness > min(p.fitness for p in self.archive):
                    # Replace worst in archive
                    self.archive.sort(key=lambda p: p.fitness)
                    self.archive[0] = prog.clone()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best_fitness={best_fitness:.3f}, "
                      f"archive_size={len(self.archive)}")

            if best_fitness >= 1.0:
                break

            # Create next generation
            next_pop = []

            # Elitism: keep top 10%
            self.population.sort(key=lambda p: -p.fitness)
            elite_size = max(1, self.population_size // 10)
            next_pop.extend(p.clone() for p in self.population[:elite_size])

            # Fill rest with mutations and crossover
            while len(next_pop) < self.population_size:
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(
                        self.population[:self.population_size // 2]
                    )
                    child = self.mutate(parent)
                else:
                    # Crossover
                    pool = self.population + self.archive
                    p1 = random.choice(pool)
                    p2 = random.choice(pool)
                    child = self.crossover(p1, p2)

                next_pop.append(child)

            self.population = next_pop

        return best_program


# =============================================================================
# ARC Task Synthesizer
# =============================================================================

class ARCStatechartSynthesizer:
    """End-to-end synthesizer for ARC tasks using statecharts."""

    def __init__(
        self,
        soar: Optional[SOARStatechart] = None,
    ):
        self.soar = soar or SOARStatechart()
        self.solved_tasks: Dict[str, StatechartProgram] = {}

    def synthesize(
        self,
        task_id: str,
        train_examples: List[Tuple[mx.array, mx.array]],
        test_inputs: List[mx.array],
        n_generations: int = 100,
    ) -> List[mx.array]:
        """
        Synthesize a program for an ARC task.

        Args:
            task_id: Unique task identifier
            train_examples: List of (input, output) training pairs
            test_inputs: Test inputs to predict outputs for
            n_generations: Max evolution generations

        Returns:
            Predicted outputs for test inputs
        """
        # Evolve program on training examples
        program = self.soar.evolve(
            tasks=train_examples,
            n_generations=n_generations,
        )

        self.solved_tasks[task_id] = program

        # Apply to test inputs
        predictions = []
        for test_input in test_inputs:
            try:
                output = program.execute(test_input)
                predictions.append(output)
            except Exception:
                predictions.append(test_input)  # Fallback to identity

        return predictions

    def get_program(self, task_id: str) -> Optional[StatechartProgram]:
        """Get the synthesized program for a task."""
        return self.solved_tasks.get(task_id)


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demonstrate SOAR-style statechart synthesis."""
    print("=" * 60)
    print("SOAR-Inspired Statechart Program Synthesis")
    print("=" * 60)

    # Create simple ARC-like task: fill grid with specific color
    # Task: Input has some 1s, output should have all 1s become 2s

    tasks = []
    for _ in range(5):
        h, w = random.randint(3, 10), random.randint(3, 10)
        input_grid = mx.array([[random.randint(0, 1) for _ in range(w)] for _ in range(h)])
        output_grid = mx.where(input_grid == 1, 2, input_grid)
        tasks.append((input_grid, output_grid))

    print(f"\nTask: Replace 1s with 2s")
    print(f"Training examples: {len(tasks)}")
    print(f"Example input shape: {tasks[0][0].shape}")

    # Synthesize program
    synth = ARCStatechartSynthesizer()

    print("\nEvolving statechart program...")
    program = synth.soar.evolve(tasks, n_generations=50, verbose=True)

    print(f"\nBest program:")
    print(f"  States: {len(program.states)}")
    print(f"  Transitions: {len(program.transitions)}")
    print(f"  Fitness: {program.fitness:.3f}")

    # Test on new example
    test_input = mx.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    expected = mx.where(test_input == 1, 2, test_input)

    output = program.execute(test_input)
    correct = bool(mx.all(output == expected))

    print(f"\nTest:")
    print(f"  Input:\n{test_input}")
    print(f"  Expected:\n{expected}")
    print(f"  Output:\n{output}")
    print(f"  Correct: {correct}")

    return program


if __name__ == "__main__":
    demo()
