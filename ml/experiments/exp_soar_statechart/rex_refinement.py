"""
REX Refinement - Evolutionary Algorithm for Statechart Programs

Implements SOAR-style REX (Refine-EXplore) algorithm adapted for statecharts:
1. Thompson Sampling for parent selection (explore-exploit balance)
2. Coverage-based crossover for complementary solutions
3. Archive of diverse programs for long-term memory
4. Learned mutation operators that improve over time

Key insight: Statechart structure constrains mutations to valid programs,
making evolution more sample-efficient than unconstrained program synthesis.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
from collections import defaultdict
import random
import copy
import math

try:
    from .soar_statechart import (
        StatechartProgram, ProgramState, Transition, Guard,
        GridAction, GridOp, StateType
    )
except ImportError:
    from soar_statechart import (
        StatechartProgram, ProgramState, Transition, Guard,
        GridAction, GridOp, StateType
    )


# =============================================================================
# Refinement Operators
# =============================================================================

class MutationType(Enum):
    """Types of mutations for statechart programs."""
    ADD_STATE = auto()
    REMOVE_STATE = auto()
    MODIFY_ACTION = auto()
    ADD_TRANSITION = auto()
    REMOVE_TRANSITION = auto()
    MODIFY_GUARD = auto()
    CHANGE_PRIORITY = auto()
    SWAP_ACTIONS = auto()
    SPLIT_STATE = auto()
    MERGE_STATES = auto()


@dataclass
class RefinementOperator:
    """A learned refinement operator for statechart programs."""
    mutation_type: MutationType
    success_count: int = 0
    attempt_count: int = 0
    total_improvement: float = 0.0

    # Learned parameters
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Empirical success rate of this operator."""
        if self.attempt_count == 0:
            return 0.5  # Prior
        return self.success_count / self.attempt_count

    @property
    def avg_improvement(self) -> float:
        """Average fitness improvement when successful."""
        if self.success_count == 0:
            return 0.0
        return self.total_improvement / self.success_count

    def thompson_score(self, c: float = 10.0) -> float:
        """Thompson sampling score for operator selection."""
        alpha = 1 + c * self.success_rate
        beta = 1 + c * (1 - self.success_rate) + self.attempt_count * 0.1
        # Sample from Beta distribution
        # Approximate with mean + noise
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        noise = random.gauss(0, math.sqrt(var))
        return max(0, min(1, mean + noise))

    def record_outcome(self, success: bool, improvement: float = 0.0):
        """Record the outcome of applying this operator."""
        self.attempt_count += 1
        if success:
            self.success_count += 1
            self.total_improvement += improvement


@dataclass
class MutationArchive:
    """
    Archive of programs with diversity maintenance.

    Inspired by MAP-Elites: maintain diverse solutions across
    behavioral dimensions (e.g., program size, solved tasks).
    """
    max_size: int = 1000
    programs: List[StatechartProgram] = field(default_factory=list)

    # Behavioral descriptors for each program
    descriptors: List[Tuple[float, ...]] = field(default_factory=list)

    # Grid-based archive (MAP-Elites style)
    grid_resolution: int = 10
    grid: Dict[Tuple[int, ...], StatechartProgram] = field(default_factory=dict)

    def _compute_descriptor(self, program: StatechartProgram) -> Tuple[float, ...]:
        """Compute behavioral descriptor for a program."""
        # Dimensions: program complexity, transition density
        n_states = len(program.states)
        n_trans = len(program.transitions)

        # Normalize to [0, 1]
        complexity = min(1.0, n_states / 20.0)
        density = min(1.0, n_trans / (n_states + 1) / 3.0)
        fitness = program.fitness

        return (complexity, density, fitness)

    def _descriptor_to_cell(self, desc: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert continuous descriptor to grid cell."""
        return tuple(
            min(self.grid_resolution - 1, int(d * self.grid_resolution))
            for d in desc
        )

    def add(self, program: StatechartProgram) -> bool:
        """
        Add program to archive if it improves diversity.

        Returns True if program was added.
        """
        desc = self._compute_descriptor(program)
        cell = self._descriptor_to_cell(desc)

        # Check if cell is empty or new program is better
        if cell not in self.grid or program.fitness > self.grid[cell].fitness:
            self.grid[cell] = program.clone()

            # Also add to flat list (with size limit)
            self.programs.append(program.clone())
            self.descriptors.append(desc)

            if len(self.programs) > self.max_size:
                # Remove lowest fitness
                min_idx = min(range(len(self.programs)),
                              key=lambda i: self.programs[i].fitness)
                self.programs.pop(min_idx)
                self.descriptors.pop(min_idx)

            return True
        return False

    def sample(self, n: int = 1) -> List[StatechartProgram]:
        """Sample programs from archive."""
        if not self.programs:
            return []
        n = min(n, len(self.programs))
        return random.sample(self.programs, n)

    def sample_diverse(self, n: int = 2) -> List[StatechartProgram]:
        """Sample diverse programs (from different cells)."""
        if len(self.grid) < n:
            return self.sample(n)

        cells = random.sample(list(self.grid.keys()), n)
        return [self.grid[cell].clone() for cell in cells]

    def get_elite(self, n: int = 5) -> List[StatechartProgram]:
        """Get top-n programs by fitness."""
        sorted_progs = sorted(self.programs, key=lambda p: -p.fitness)
        return [p.clone() for p in sorted_progs[:n]]

    def coverage(self) -> float:
        """Fraction of grid cells filled."""
        total_cells = self.grid_resolution ** 3  # 3D grid
        return len(self.grid) / total_cells


# =============================================================================
# REX Evolver
# =============================================================================

class REXEvolver:
    """
    REX (Refine-EXplore) evolutionary algorithm for statecharts.

    Key features:
    1. Thompson sampling for parent selection
    2. Learned refinement operators
    3. Diversity-maintaining archive
    4. Coverage-based crossover
    """

    def __init__(
        self,
        max_states: int = 20,
        max_transitions: int = 50,
        population_size: int = 50,
        archive_size: int = 1000,
        c_thompson: float = 20.0,
    ):
        self.max_states = max_states
        self.max_transitions = max_transitions
        self.population_size = population_size
        self.c_thompson = c_thompson

        # Population
        self.population: List[StatechartProgram] = []

        # Archive
        self.archive = MutationArchive(max_size=archive_size)

        # Learned operators
        self.operators = {
            mt: RefinementOperator(mutation_type=mt)
            for mt in MutationType
        }

        # Primitive operations
        self.grid_ops = list(GridOp)
        self.predicates = [
            "true", "has_nonzero", "is_symmetric",
            "grid[0,0] == 1", "shape_match", "color_match"
        ]

        # Statistics
        self.generation = 0
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []

    def _select_operator(self) -> RefinementOperator:
        """Select operator using Thompson sampling."""
        scores = {
            mt: op.thompson_score(self.c_thompson)
            for mt, op in self.operators.items()
        }
        best_mt = max(scores.keys(), key=lambda mt: scores[mt])
        return self.operators[best_mt]

    def _thompson_select_parent(self, candidates: List[StatechartProgram]) -> StatechartProgram:
        """Select parent using Thompson sampling on fitness."""
        if not candidates:
            raise ValueError("No candidates for parent selection")

        scores = []
        for prog in candidates:
            # Thompson sampling: Beta(1 + C*fitness, 1 + C*(1-fitness) + N)
            h = prog.fitness
            n = getattr(prog, 'selection_count', 0)
            alpha = 1 + self.c_thompson * h
            beta = 1 + self.c_thompson * (1 - h) + n * 0.1

            # Approximate Beta sample
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            score = max(0, min(1, mean + random.gauss(0, math.sqrt(var))))
            scores.append(score)

        # Select best score
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        selected = candidates[best_idx]

        # Track selection count
        if not hasattr(selected, 'selection_count'):
            selected.selection_count = 0
        selected.selection_count += 1

        return selected

    def _coverage_select_pair(
        self,
        candidates: List[StatechartProgram],
        tasks: List[Tuple[mx.array, mx.array]]
    ) -> Tuple[StatechartProgram, StatechartProgram]:
        """Select pair of parents that maximize coverage."""
        if len(candidates) < 2:
            return candidates[0], candidates[0]

        best_pair = None
        best_coverage = -1

        # Sample subset of pairs for efficiency
        n_pairs = min(50, len(candidates) * (len(candidates) - 1) // 2)

        for _ in range(n_pairs):
            p1, p2 = random.sample(candidates, 2)

            # Compute combined coverage
            coverage = self._combined_coverage(p1, p2, tasks)

            # Weight by selection frequency (prefer less-selected)
            n1 = getattr(p1, 'selection_count', 0)
            n2 = getattr(p2, 'selection_count', 0)
            weight = 1.0 / (n1 + n2 + 1)

            score = coverage + weight * 0.1

            if score > best_coverage:
                best_coverage = score
                best_pair = (p1, p2)

        return best_pair

    def _combined_coverage(
        self,
        p1: StatechartProgram,
        p2: StatechartProgram,
        tasks: List[Tuple[mx.array, mx.array]]
    ) -> float:
        """Compute fraction of tasks solved by either program."""
        if not tasks:
            return 0.0

        solved = 0
        for input_grid, expected in tasks:
            try:
                out1 = p1.execute(input_grid)
                out2 = p2.execute(input_grid)
                if mx.all(out1 == expected) or mx.all(out2 == expected):
                    solved += 1
            except Exception:
                pass

        return solved / len(tasks)

    def mutate(self, program: StatechartProgram) -> Tuple[StatechartProgram, RefinementOperator]:
        """Apply mutation operator to program."""
        operator = self._select_operator()
        prog = program.clone()
        mt = operator.mutation_type

        if mt == MutationType.ADD_STATE and len(prog.states) < self.max_states:
            label = f"s{len(prog.states)}"
            action = GridAction(
                op=random.choice(self.grid_ops),
                params={"color": random.randint(0, 9)}
            )
            state = ProgramState(label=label, action=action)
            prog.states[label] = state
            if "root" in prog.states:
                prog.states["root"].children.append(state)

        elif mt == MutationType.REMOVE_STATE and len(prog.states) > 3:
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

        elif mt == MutationType.MODIFY_ACTION:
            basic_states = [
                s for s in prog.states.values()
                if s.state_type == StateType.BASIC and s.action is not None
            ]
            if basic_states:
                state = random.choice(basic_states)
                state.action = GridAction(
                    op=random.choice(self.grid_ops),
                    params={"color": random.randint(0, 9)}
                )

        elif mt == MutationType.ADD_TRANSITION and len(prog.transitions) < self.max_transitions:
            state_labels = [l for l in prog.states.keys() if l != "root"]
            if len(state_labels) >= 2:
                src = random.choice(state_labels)
                tgt = random.choice(state_labels)
                guard = Guard(predicate=random.choice(self.predicates))
                trans = Transition(source=src, target=tgt, guard=guard)
                prog.transitions.append(trans)

        elif mt == MutationType.REMOVE_TRANSITION and len(prog.transitions) > 1:
            prog.transitions.pop(random.randint(0, len(prog.transitions) - 1))

        elif mt == MutationType.MODIFY_GUARD:
            if prog.transitions:
                trans = random.choice(prog.transitions)
                trans.guard = Guard(predicate=random.choice(self.predicates))

        elif mt == MutationType.CHANGE_PRIORITY:
            if prog.transitions:
                trans = random.choice(prog.transitions)
                trans.priority = random.randint(0, 10)

        elif mt == MutationType.SWAP_ACTIONS:
            basic_states = [
                s for s in prog.states.values()
                if s.state_type == StateType.BASIC and s.action is not None
            ]
            if len(basic_states) >= 2:
                s1, s2 = random.sample(basic_states, 2)
                s1.action, s2.action = s2.action, s1.action

        elif mt == MutationType.SPLIT_STATE:
            # Split a state into two sequential states
            basic_states = [
                s for s in prog.states.values()
                if s.state_type == StateType.BASIC and s.label != "root"
            ]
            if basic_states and len(prog.states) < self.max_states:
                state = random.choice(basic_states)
                new_label = f"{state.label}_b"
                new_state = ProgramState(
                    label=new_label,
                    state_type=StateType.BASIC,
                    action=GridAction(op=GridOp.IDENTITY),
                )
                prog.states[new_label] = new_state

                # Redirect outgoing transitions
                for trans in prog.transitions:
                    if trans.source == state.label:
                        trans.source = new_label

                # Add transition from original to new
                prog.transitions.append(Transition(
                    source=state.label,
                    target=new_label,
                    guard=Guard(predicate="true"),
                ))

        elif mt == MutationType.MERGE_STATES:
            # Merge two states into one
            basic_states = [
                s for s in prog.states.values()
                if s.state_type == StateType.BASIC and s.label != "root"
                and not s.is_initial and not s.is_final
            ]
            if len(basic_states) >= 2:
                s1, s2 = random.sample(basic_states, 2)
                # Keep s1, remove s2
                for trans in prog.transitions:
                    if trans.source == s2.label:
                        trans.source = s1.label
                    if trans.target == s2.label:
                        trans.target = s1.label
                del prog.states[s2.label]

        return prog, operator

    def crossover(
        self,
        parent1: StatechartProgram,
        parent2: StatechartProgram
    ) -> StatechartProgram:
        """Crossover two programs (subtree exchange)."""
        child = parent1.clone()

        # Exchange some states from parent2
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

        # Remove duplicate transitions
        seen = set()
        unique_trans = []
        for t in child.transitions:
            key = (t.source, t.target, t.guard.predicate if t.guard else "")
            if key not in seen:
                seen.add(key)
                unique_trans.append(t)
        child.transitions = unique_trans

        return child

    def evaluate(
        self,
        program: StatechartProgram,
        tasks: List[Tuple[mx.array, mx.array]]
    ) -> float:
        """Evaluate program fitness on tasks."""
        if not tasks:
            return 0.0

        correct = 0
        for input_grid, expected_output in tasks:
            try:
                output = program.execute(input_grid)
                if mx.all(output == expected_output):
                    correct += 1
            except Exception:
                pass

        return correct / len(tasks)

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

        # Add random states
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

    def evolve(
        self,
        tasks: List[Tuple[mx.array, mx.array]],
        n_generations: int = 100,
        crossover_rate: float = 0.3,
        verbose: bool = True,
    ) -> StatechartProgram:
        """
        Run REX evolutionary algorithm.

        Args:
            tasks: List of (input, output) training pairs
            n_generations: Number of generations
            crossover_rate: Probability of crossover vs mutation
            verbose: Print progress

        Returns:
            Best program found
        """
        # Initialize population
        if not self.population:
            self.population = [
                self.random_program()
                for _ in range(self.population_size)
            ]

        best_program = None

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate population
            for prog in self.population:
                prog.fitness = self.evaluate(prog, tasks)
                if prog.fitness > self.best_fitness:
                    self.best_fitness = prog.fitness
                    best_program = prog.clone()

                # Add to archive
                self.archive.add(prog)

            self.fitness_history.append(self.best_fitness)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best={self.best_fitness:.3f}, "
                      f"archive={len(self.archive.programs)}, "
                      f"coverage={self.archive.coverage():.2%}")

            if self.best_fitness >= 1.0:
                break

            # Create next generation
            next_pop = []

            # Elitism: keep top 10%
            self.population.sort(key=lambda p: -p.fitness)
            elite_size = max(1, self.population_size // 10)
            next_pop.extend(p.clone() for p in self.population[:elite_size])

            # Fill rest with mutations and crossover
            pool = self.population + self.archive.programs

            while len(next_pop) < self.population_size:
                if random.random() < crossover_rate:
                    # Crossover with coverage-based selection
                    p1, p2 = self._coverage_select_pair(pool, tasks)
                    child = self.crossover(p1, p2)
                else:
                    # Mutation with Thompson sampling
                    parent = self._thompson_select_parent(pool)
                    child, operator = self.mutate(parent)

                    # Evaluate and record operator outcome
                    child.fitness = self.evaluate(child, tasks)
                    improvement = child.fitness - parent.fitness
                    operator.record_outcome(
                        success=(improvement > 0),
                        improvement=max(0, improvement)
                    )

                next_pop.append(child)

            self.population = next_pop

        return best_program

    def get_operator_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each operator."""
        return {
            mt.name: {
                "success_rate": op.success_rate,
                "avg_improvement": op.avg_improvement,
                "attempts": op.attempt_count,
            }
            for mt, op in self.operators.items()
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate REX evolutionary algorithm."""
    print("=" * 60)
    print("REX Evolutionary Algorithm for Statecharts")
    print("=" * 60)

    # Create simple task: replace 1s with 2s
    tasks = []
    for _ in range(5):
        h, w = random.randint(3, 8), random.randint(3, 8)
        input_grid = mx.array([
            [random.randint(0, 1) for _ in range(w)]
            for _ in range(h)
        ])
        output_grid = mx.where(input_grid == 1, 2, input_grid)
        tasks.append((input_grid, output_grid))

    print(f"\nTask: Replace 1s with 2s")
    print(f"Training examples: {len(tasks)}")

    # Create evolver
    evolver = REXEvolver(
        population_size=30,
        archive_size=100,
    )

    print("\nEvolving...")
    best = evolver.evolve(tasks, n_generations=50, verbose=True)

    print(f"\nBest program:")
    print(f"  States: {len(best.states)}")
    print(f"  Transitions: {len(best.transitions)}")
    print(f"  Fitness: {best.fitness:.3f}")

    print(f"\nOperator statistics:")
    for name, stats in evolver.get_operator_stats().items():
        if stats["attempts"] > 0:
            print(f"  {name}: success={stats['success_rate']:.2%}, "
                  f"attempts={stats['attempts']}")

    print(f"\nArchive coverage: {evolver.archive.coverage():.2%}")

    return evolver


if __name__ == "__main__":
    demo()
