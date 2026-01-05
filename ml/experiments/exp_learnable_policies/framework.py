"""
Framework for Learning State Machines from Self-Play

Core idea: Evolution discovers what state structure is needed to play correctly.
The fitness function rewards correct predictions and penalizes rule violations.
"""

import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
from collections import deque
import copy


class StateType(Enum):
    """Types of states in a statechart."""
    BASIC = 1      # Leaf state
    OR = 2         # XOR children (one active at a time)
    AND = 3        # Parallel children (all active)
    HISTORY = 4    # Remembers last active substate


@dataclass
class StateNode:
    """A state in the learned statechart."""
    id: int
    state_type: StateType = StateType.BASIC
    parent_id: Optional[int] = None
    is_initial: bool = False
    has_history: bool = False
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"S{self.id}"


@dataclass
class Transition:
    """A transition between states."""
    source_id: int
    target_id: int
    event: str = ""
    guard_expr: str = ""  # Expression evaluated on context
    priority: int = 0


@dataclass
class StateGenome:
    """
    Genome encoding a learnable statechart.

    The genome represents:
    - State topology (hierarchy, types)
    - Transitions (sources, targets, events, guards)
    - Memory structure (history, buffers)
    """
    states: List[StateNode] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)

    # Memory configuration
    history_depth: int = 1      # How many past states to remember
    buffer_size: int = 0        # Ring buffer for position tracking
    ephemeral_flags: int = 0    # Flags that reset each turn
    persistent_flags: int = 0   # Flags that persist until explicitly changed

    # Fitness tracking
    fitness: float = 0.0
    accuracy: float = 0.0
    violation_rate: float = 0.0

    def copy(self) -> 'StateGenome':
        """Deep copy the genome."""
        new = StateGenome(
            states=[copy.copy(s) for s in self.states],
            transitions=[copy.copy(t) for t in self.transitions],
            history_depth=self.history_depth,
            buffer_size=self.buffer_size,
            ephemeral_flags=self.ephemeral_flags,
            persistent_flags=self.persistent_flags,
        )
        return new

    def add_state(self, state_type: StateType = StateType.BASIC,
                  parent_id: Optional[int] = None) -> int:
        """Add a new state and return its ID."""
        new_id = len(self.states)
        self.states.append(StateNode(
            id=new_id,
            state_type=state_type,
            parent_id=parent_id,
        ))
        return new_id

    def add_transition(self, source: int, target: int,
                       event: str = "", guard: str = "") -> None:
        """Add a transition."""
        self.transitions.append(Transition(
            source_id=source,
            target_id=target,
            event=event,
            guard_expr=guard,
        ))

    def get_leaf_states(self) -> List[int]:
        """Get all leaf (BASIC) states."""
        parent_ids = {s.parent_id for s in self.states if s.parent_id is not None}
        return [s.id for s in self.states if s.id not in parent_ids]

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram."""
        lines = ["stateDiagram-v2"]

        # States
        for s in self.states:
            type_label = ""
            if s.state_type == StateType.AND:
                type_label = " [AND]"
            elif s.state_type == StateType.HISTORY:
                type_label = " [H]"
            elif s.has_history:
                type_label = " [H]"

            if s.parent_id is not None:
                lines.append(f"    state \"{s.label}{type_label}\" as {s.label}")

        # Initial state
        initial = [s for s in self.states if s.is_initial]
        if initial:
            lines.append(f"    [*] --> {initial[0].label}")

        # Transitions
        for t in self.transitions:
            src = self.states[t.source_id].label if t.source_id < len(self.states) else "?"
            tgt = self.states[t.target_id].label if t.target_id < len(self.states) else "?"
            label = t.event
            if t.guard_expr:
                label += f" [{t.guard_expr}]"
            if label:
                lines.append(f"    {src} --> {tgt} : {label}")
            else:
                lines.append(f"    {src} --> {tgt}")

        return "\n".join(lines)


class GameEnvironment(ABC):
    """
    Abstract base class for game environments.

    Subclasses implement specific games (Go, Chess variants).
    """

    @abstractmethod
    def reset(self) -> Any:
        """Reset to initial state, return observation."""
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Get list of legal moves in current state."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take action, return (obs, reward, done, info)."""
        pass

    @abstractmethod
    def check_special_rule_violation(self, action: Any) -> bool:
        """Check if action violates the special rule we're trying to learn."""
        pass

    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """Get context dict for guard evaluation."""
        pass

    @abstractmethod
    def get_special_rule_name(self) -> str:
        """Name of the special rule being learned."""
        pass


@dataclass
class EvolutionConfig:
    """Configuration for evolution."""
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elite_count: int = 5
    tournament_size: int = 3

    # State limits
    max_states: int = 20
    max_transitions: int = 50
    max_buffer_size: int = 10
    max_flags: int = 8

    # Fitness weights
    accuracy_weight: float = 1.0
    violation_weight: float = 10.0  # Heavy penalty for violations
    parsimony_weight: float = 0.1   # Reward simpler structures

    # Evaluation
    games_per_eval: int = 100
    moves_per_game: int = 200


class PolicyEvolver:
    """
    Evolves statechart policies from self-play.

    Process:
    1. Initialize random population of StateGenomes
    2. Evaluate each genome by playing games
    3. Select best genomes (tournament selection)
    4. Apply crossover and mutation
    5. Repeat until convergence
    """

    def __init__(self, env_class: type, config: EvolutionConfig = None):
        self.env_class = env_class
        self.config = config or EvolutionConfig()
        self.population: List[StateGenome] = []
        self.best_genome: Optional[StateGenome] = None
        self.generation = 0
        self.history: List[Dict] = []

    def initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            genome = self._create_random_genome()
            self.population.append(genome)

    def _create_random_genome(self) -> StateGenome:
        """Create a random genome."""
        genome = StateGenome()

        # Random number of states (3-10)
        n_states = random.randint(3, min(10, self.config.max_states))

        # Create root state
        root_id = genome.add_state(StateType.OR)
        genome.states[root_id].is_initial = True

        # Create child states
        for i in range(n_states - 1):
            state_type = random.choice([StateType.BASIC, StateType.BASIC, StateType.OR])
            parent = random.randint(0, len(genome.states) - 1) if i > 0 else root_id
            genome.add_state(state_type, parent)

        # Random transitions between leaf states
        leaves = genome.get_leaf_states()
        n_trans = random.randint(len(leaves), min(len(leaves) * 2, self.config.max_transitions))

        for _ in range(n_trans):
            src = random.choice(leaves)
            tgt = random.choice(leaves)
            event = f"e{random.randint(0, 5)}"
            guard = random.choice(["", "ctx.flag0", "ctx.flag1", "not ctx.flag0"])
            genome.add_transition(src, tgt, event, guard)

        # Random memory configuration
        genome.history_depth = random.randint(0, 3)
        genome.buffer_size = random.randint(0, 5)
        genome.ephemeral_flags = random.randint(0, 4)
        genome.persistent_flags = random.randint(0, 4)

        return genome

    def evaluate_genome(self, genome: StateGenome) -> float:
        """Evaluate a genome by playing games."""
        env = self.env_class()

        total_moves = 0
        correct_predictions = 0
        violations = 0

        for _ in range(self.config.games_per_eval):
            env.reset()

            # Simple state tracking based on genome
            current_state = 0
            history = deque(maxlen=max(1, genome.history_depth))
            buffer = deque(maxlen=max(1, genome.buffer_size))
            ephemeral = [False] * max(1, genome.ephemeral_flags)
            persistent = [False] * max(1, genome.persistent_flags)

            for move_num in range(self.config.moves_per_game):
                legal_moves = env.get_legal_moves()
                if not legal_moves:
                    break

                # Get context for guard evaluation
                ctx = env.get_context()
                ctx['history'] = list(history)
                ctx['buffer'] = list(buffer)
                ctx['ephemeral'] = ephemeral
                ctx['persistent'] = persistent
                ctx['state'] = current_state

                # Choose move (random for now - policy learning is separate)
                action = random.choice(legal_moves)

                # Check if we correctly predict legality
                is_violation = env.check_special_rule_violation(action)

                # Evaluate genome's prediction
                would_block = self._evaluate_guards(genome, ctx, action)

                if is_violation and would_block:
                    correct_predictions += 1  # Correctly blocked violation
                elif not is_violation and not would_block:
                    correct_predictions += 1  # Correctly allowed legal move

                if is_violation:
                    violations += 1

                total_moves += 1

                # Take step
                obs, reward, done, info = env.step(action)

                # Update state tracking
                history.append(current_state)
                buffer.append(hash(str(obs)) % 1000)  # Simple position hash
                ephemeral = [False] * len(ephemeral)  # Reset ephemeral

                if done:
                    break

        # Compute fitness
        accuracy = correct_predictions / max(1, total_moves)
        violation_rate = violations / max(1, total_moves)

        # Parsimony bonus (simpler is better)
        complexity = len(genome.states) + len(genome.transitions) / 5
        parsimony = max(0, 1 - complexity / self.config.max_states)

        fitness = (
            self.config.accuracy_weight * accuracy
            - self.config.violation_weight * violation_rate
            + self.config.parsimony_weight * parsimony
        )

        genome.fitness = fitness
        genome.accuracy = accuracy
        genome.violation_rate = violation_rate

        return fitness

    def _evaluate_guards(self, genome: StateGenome, ctx: Dict, action: Any) -> bool:
        """Evaluate if genome's guards would block this action."""
        # Simplified guard evaluation - in practice this would be more sophisticated
        for trans in genome.transitions:
            if trans.guard_expr:
                try:
                    # Very simplified evaluation
                    if 'history' in trans.guard_expr and len(ctx.get('history', [])) > 0:
                        return True
                    if 'buffer' in trans.guard_expr and len(ctx.get('buffer', [])) > 0:
                        return True
                except:
                    pass
        return False

    def mutate(self, genome: StateGenome) -> StateGenome:
        """Apply mutation operators."""
        new = genome.copy()

        if random.random() < self.config.mutation_rate:
            mutation = random.choice([
                'add_state', 'remove_state', 'add_transition',
                'remove_transition', 'mutate_guard', 'mutate_memory'
            ])

            if mutation == 'add_state' and len(new.states) < self.config.max_states:
                parent = random.choice([s.id for s in new.states])
                new.add_state(StateType.BASIC, parent)

            elif mutation == 'remove_state' and len(new.states) > 3:
                leaves = new.get_leaf_states()
                if leaves:
                    to_remove = random.choice(leaves)
                    new.states = [s for s in new.states if s.id != to_remove]
                    new.transitions = [t for t in new.transitions
                                       if t.source_id != to_remove and t.target_id != to_remove]

            elif mutation == 'add_transition' and len(new.transitions) < self.config.max_transitions:
                leaves = new.get_leaf_states() or [0]
                src = random.choice(leaves)
                tgt = random.choice(leaves)
                guards = ["", "ctx.history", "ctx.buffer", "ctx.ephemeral[0]", "ctx.persistent[0]"]
                new.add_transition(src, tgt, f"e{random.randint(0,5)}", random.choice(guards))

            elif mutation == 'remove_transition' and len(new.transitions) > 1:
                idx = random.randint(0, len(new.transitions) - 1)
                new.transitions.pop(idx)

            elif mutation == 'mutate_guard' and new.transitions:
                trans = random.choice(new.transitions)
                guards = ["", "ctx.history", "ctx.buffer", "len(ctx.history) > 0",
                          "ctx.ephemeral[0]", "ctx.persistent[0]", "ctx.last_capture"]
                trans.guard_expr = random.choice(guards)

            elif mutation == 'mutate_memory':
                choice = random.choice(['history', 'buffer', 'ephemeral', 'persistent'])
                if choice == 'history':
                    new.history_depth = max(0, min(5, new.history_depth + random.randint(-1, 1)))
                elif choice == 'buffer':
                    new.buffer_size = max(0, min(self.config.max_buffer_size,
                                                  new.buffer_size + random.randint(-1, 1)))
                elif choice == 'ephemeral':
                    new.ephemeral_flags = max(0, min(self.config.max_flags,
                                                      new.ephemeral_flags + random.randint(-1, 1)))
                elif choice == 'persistent':
                    new.persistent_flags = max(0, min(self.config.max_flags,
                                                       new.persistent_flags + random.randint(-1, 1)))

        return new

    def crossover(self, parent1: StateGenome, parent2: StateGenome) -> StateGenome:
        """Create offspring from two parents."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy()

        child = StateGenome()

        # Crossover states
        split = random.randint(1, min(len(parent1.states), len(parent2.states)) - 1)
        child.states = [copy.copy(s) for s in parent1.states[:split]]
        child.states.extend([copy.copy(s) for s in parent2.states[split:]])

        # Crossover transitions (take from both)
        child.transitions = []
        for t in parent1.transitions[:len(parent1.transitions)//2]:
            if t.source_id < len(child.states) and t.target_id < len(child.states):
                child.transitions.append(copy.copy(t))
        for t in parent2.transitions[len(parent2.transitions)//2:]:
            if t.source_id < len(child.states) and t.target_id < len(child.states):
                child.transitions.append(copy.copy(t))

        # Crossover memory config
        child.history_depth = random.choice([parent1.history_depth, parent2.history_depth])
        child.buffer_size = random.choice([parent1.buffer_size, parent2.buffer_size])
        child.ephemeral_flags = random.choice([parent1.ephemeral_flags, parent2.ephemeral_flags])
        child.persistent_flags = random.choice([parent1.persistent_flags, parent2.persistent_flags])

        return child

    def select_parent(self) -> StateGenome:
        """Tournament selection."""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)

    def evolve_generation(self) -> None:
        """Run one generation of evolution."""
        # Evaluate all genomes
        for genome in self.population:
            self.evaluate_genome(genome)

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Track best
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = self.population[0].copy()

        # Record history
        self.history.append({
            'generation': self.generation,
            'best_fitness': self.population[0].fitness,
            'best_accuracy': self.population[0].accuracy,
            'best_violation_rate': self.population[0].violation_rate,
            'avg_fitness': sum(g.fitness for g in self.population) / len(self.population),
        })

        # Create new population
        new_population = []

        # Elitism
        new_population.extend([g.copy() for g in self.population[:self.config.elite_count]])

        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population[:self.config.population_size]
        self.generation += 1

    def evolve(self, n_generations: int = None, verbose: bool = True) -> StateGenome:
        """Run full evolution."""
        n_gens = n_generations or self.config.n_generations

        if not self.population:
            self.initialize_population()

        for gen in range(n_gens):
            self.evolve_generation()

            if verbose and (gen % 10 == 0 or gen == n_gens - 1):
                h = self.history[-1]
                print(f"Gen {gen:3d}: fitness={h['best_fitness']:.4f}, "
                      f"acc={h['best_accuracy']:.1%}, "
                      f"violations={h['best_violation_rate']:.2%}")

        return self.best_genome


if __name__ == "__main__":
    print("Framework loaded. Use specific game modules to run experiments.")
