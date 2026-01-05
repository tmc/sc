"""
Evolutionary Statechart Discovery for Coverage Prediction

EVOLVE the optimal statechart topology for coverage prediction instead of
using hardcoded CFG-to-statechart conversion.

Key insight: LEARNED structure outperforms hand-designed structure.

Components evolved:
1. Number and types of states
2. Transition patterns (which states connect)
3. Guard expressions (branch conditions)
4. State-to-line mappings (which lines each state covers)

Fitness: Coverage prediction F1 score on held-out examples.
"""

import random
import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict
from enum import Enum, auto
import json

from .dataset import Dataset, DatasetExample, DatasetGenerator
from .benchmark import compute_metrics, aggregate_metrics, EvaluationMetrics


# =============================================================================
# GENOME REPRESENTATION
# =============================================================================

class EvolvedStateType(Enum):
    """Types of states in evolved statechart."""
    ENTRY = auto()
    EXIT = auto()
    BASIC = auto()
    BRANCH = auto()
    LOOP = auto()
    COMPOSITE = auto()


@dataclass
class EvolvedState:
    """A state in the evolved statechart."""
    id: str
    state_type: EvolvedStateType
    lines: Set[int]             # Source lines this state covers
    is_initial: bool = False
    is_final: bool = False

    def copy(self) -> 'EvolvedState':
        return EvolvedState(
            id=self.id,
            state_type=self.state_type,
            lines=self.lines.copy(),
            is_initial=self.is_initial,
            is_final=self.is_final,
        )


@dataclass
class EvolvedGuard:
    """An evolved guard expression."""
    expression: str
    variables: List[str] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Evaluate guard symbolically.
        Returns (can_be_true, can_be_false).
        """
        if not self.expression or self.expression == "true":
            return (True, False)
        if self.expression == "false":
            return (False, True)

        try:
            # Simple evaluation
            result = eval(self.expression, {"__builtins__": {}}, context)
            return (result, not result)
        except:
            # If can't evaluate, assume both possible
            return (True, True)


@dataclass
class EvolvedTransition:
    """A transition in the evolved statechart."""
    from_state: str
    to_state: str
    guard: Optional[EvolvedGuard] = None
    priority: int = 0           # For ordering when multiple transitions possible

    def copy(self) -> 'EvolvedTransition':
        return EvolvedTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            guard=EvolvedGuard(
                expression=self.guard.expression,
                variables=self.guard.variables.copy()
            ) if self.guard else None,
            priority=self.priority,
        )


@dataclass
class CoverageStatechartGenome:
    """
    Evolvable genome for coverage prediction statecharts.

    This is the unit of evolution - represents a complete statechart
    that predicts coverage.
    """
    states: Dict[str, EvolvedState]
    transitions: List[EvolvedTransition]
    program_id: str = ""        # ID of program this is for (or "" for generic)
    max_lines: int = 100        # Maximum line number

    # Fitness cache
    fitness: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    def __post_init__(self):
        self._id = None

    @property
    def id(self) -> str:
        if self._id is None:
            self._id = self._compute_id()
        return self._id

    def _compute_id(self) -> str:
        """Compute unique ID for caching."""
        parts = [self.program_id]
        for s in sorted(self.states.values(), key=lambda x: x.id):
            parts.append(f"{s.id}:{s.state_type.name}:{sorted(s.lines)}")
        for t in sorted(self.transitions, key=lambda x: (x.from_state, x.to_state)):
            guard = t.guard.expression if t.guard else ""
            parts.append(f"{t.from_state}->{t.to_state}:{guard}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def is_valid(self) -> bool:
        """Check if genome is structurally valid."""
        if not self.states:
            return False

        # Must have at least one initial and one final state
        has_initial = any(s.is_initial for s in self.states.values())
        has_final = any(s.is_final for s in self.states.values())
        if not has_initial or not has_final:
            return False

        # All transitions must reference valid states
        state_ids = set(self.states.keys())
        for t in self.transitions:
            if t.from_state not in state_ids or t.to_state not in state_ids:
                return False

        return True

    def copy(self) -> 'CoverageStatechartGenome':
        """Deep copy the genome, preserving fitness values."""
        new_genome = CoverageStatechartGenome(
            states={k: v.copy() for k, v in self.states.items()},
            transitions=[t.copy() for t in self.transitions],
            program_id=self.program_id,
            max_lines=self.max_lines,
        )
        # Preserve fitness values
        new_genome.fitness = self.fitness
        new_genome.f1_score = self.f1_score
        new_genome.precision = self.precision
        new_genome.recall = self.recall
        return new_genome

    def get_reachable_states(self, start: Optional[str] = None) -> Set[str]:
        """Get all states reachable from start (or initial state)."""
        if start is None:
            starts = [s.id for s in self.states.values() if s.is_initial]
        else:
            starts = [start]

        reachable = set()
        queue = list(starts)

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            for t in self.transitions:
                if t.from_state == current and t.to_state not in reachable:
                    queue.append(t.to_state)

        return reachable

    def predict_coverage(
        self,
        input_context: Dict[str, Any],
        max_steps: int = 100,
    ) -> Set[int]:
        """
        Predict covered lines by simulating statechart with input.

        Returns set of predicted covered lines.
        """
        # Start at initial states
        current_states = {s.id for s in self.states.values() if s.is_initial}
        visited_states = set()

        for _ in range(max_steps):
            if not current_states:
                break

            visited_states |= current_states
            next_states = set()

            for state_id in current_states:
                # Find enabled transitions
                for t in self.transitions:
                    if t.from_state != state_id:
                        continue

                    # Check guard
                    if t.guard:
                        can_true, can_false = t.guard.evaluate(input_context)
                        if can_true:
                            next_states.add(t.to_state)
                    else:
                        next_states.add(t.to_state)

            # Stop if no progress
            if next_states <= visited_states:
                break

            current_states = next_states - visited_states

        # Collect covered lines
        covered = set()
        for state_id in visited_states:
            if state_id in self.states:
                covered |= self.states[state_id].lines

        return covered


# =============================================================================
# GENOME FACTORY
# =============================================================================

class GenomeFactory:
    """Factory for creating and mutating genomes."""

    # Guard templates
    GUARD_TEMPLATES = [
        "x > {c}",
        "x < {c}",
        "x == {c}",
        "x >= {c}",
        "x <= {c}",
        "x != {c}",
        "len(items) > {c}",
        "len(items) == {c}",
        "x > 0",
        "x < 0",
        "x == 0",
        "x",
        "not x",
        "items",
        "not items",
        "True",
    ]

    def __init__(
        self,
        max_states: int = 15,
        min_states: int = 3,
        max_lines: int = 30,
        observed_line_patterns: Optional[List[Set[int]]] = None,
    ):
        self.max_states = max_states
        self.min_states = min_states
        self.max_lines = max_lines
        # Line patterns actually observed in training data
        self.observed_line_patterns = observed_line_patterns or []
        # Common line groups (lines that often appear together)
        self.common_line_groups = self._compute_common_groups()

    def _compute_common_groups(self) -> List[Set[int]]:
        """Compute groups of lines that frequently co-occur."""
        if not self.observed_line_patterns:
            return []

        # Count line co-occurrences
        cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        line_counts: Dict[int, int] = defaultdict(int)

        for pattern in self.observed_line_patterns:
            lines = sorted(pattern)
            for line in lines:
                line_counts[line] += 1
            for i, l1 in enumerate(lines):
                for l2 in lines[i+1:]:
                    cooccurrence[(l1, l2)] += 1

        # Find lines that always appear together (>90% co-occurrence)
        groups = []
        all_lines = set(line_counts.keys())
        visited = set()

        for line in sorted(all_lines):
            if line in visited:
                continue

            group = {line}
            for other in all_lines:
                if other == line or other in visited:
                    continue
                key = (min(line, other), max(line, other))
                together = cooccurrence.get(key, 0)
                either = max(line_counts[line], line_counts[other])
                if either > 0 and together / either > 0.8:
                    group.add(other)

            if len(group) > 1:
                groups.append(group)
                visited |= group

        # Also add individual frequent lines as singleton groups
        for line in all_lines:
            if line not in visited and line_counts[line] >= 2:
                groups.append({line})

        return groups

    def create_from_observations(self, program_id: str = "") -> CoverageStatechartGenome:
        """
        Create a genome directly from observed coverage patterns.

        This seeds evolution with a structure that reflects actual behavior.
        """
        if not self.common_line_groups:
            return self.create_random(program_id)

        states = {}

        # Entry state - first group or first few lines
        entry_lines = {1, 2}
        for group in self.common_line_groups:
            if 1 in group or 2 in group:
                entry_lines = group.copy()
                break

        entry = EvolvedState(
            id="entry",
            state_type=EvolvedStateType.ENTRY,
            lines=entry_lines,
            is_initial=True,
        )
        states["entry"] = entry

        # Exit state
        exit_state = EvolvedState(
            id="exit",
            state_type=EvolvedStateType.EXIT,
            lines=set(),
            is_final=True,
        )
        states["exit"] = exit_state

        # Create states for each observed line group
        used_groups = []
        for i, group in enumerate(self.common_line_groups):
            if group == entry_lines:
                continue
            if len(states) >= self.max_states:
                break

            # Determine state type based on line patterns
            sorted_lines = sorted(group)
            if len(sorted_lines) == 1:
                state_type = EvolvedStateType.BASIC
            elif sorted_lines[-1] - sorted_lines[0] < len(sorted_lines) * 2:
                state_type = EvolvedStateType.BASIC  # Consecutive lines = basic block
            else:
                state_type = EvolvedStateType.BRANCH  # Non-consecutive = branch

            state = EvolvedState(
                id=f"obs_{i}",
                state_type=state_type,
                lines=group.copy(),
            )
            states[state.id] = state
            used_groups.append(state.id)

        # Create transitions based on line order
        transitions = []

        # Entry transitions
        if used_groups:
            # Connect entry to states containing low line numbers
            for sid in used_groups:
                state = states[sid]
                if any(l < 5 for l in state.lines):
                    transitions.append(EvolvedTransition(from_state="entry", to_state=sid))
                    break
            else:
                transitions.append(EvolvedTransition(from_state="entry", to_state=used_groups[0]))

        # Connect states in line-order
        sorted_states = sorted(
            [(sid, min(states[sid].lines)) for sid in used_groups],
            key=lambda x: x[1]
        )

        for i, (sid, _) in enumerate(sorted_states):
            if i < len(sorted_states) - 1:
                next_sid = sorted_states[i+1][0]
                transitions.append(EvolvedTransition(from_state=sid, to_state=next_sid))

                # Add guard-based branches
                if random.random() < 0.3 and len(sorted_states) > 2:
                    # Skip transition with guard
                    skip_to = random.choice([s for s, _ in sorted_states if s != sid])
                    guard = self.create_random_guard()
                    transitions.append(EvolvedTransition(
                        from_state=sid,
                        to_state=skip_to,
                        guard=guard,
                    ))

        # Exit transitions
        if used_groups:
            # Connect states with high line numbers to exit
            for sid, _ in reversed(sorted_states):
                state = states[sid]
                if any(l > self.max_lines - 5 for l in state.lines):
                    transitions.append(EvolvedTransition(from_state=sid, to_state="exit"))
                    break
            else:
                transitions.append(EvolvedTransition(from_state=sorted_states[-1][0], to_state="exit"))
        else:
            transitions.append(EvolvedTransition(from_state="entry", to_state="exit"))

        return CoverageStatechartGenome(
            states=states,
            transitions=transitions,
            program_id=program_id,
            max_lines=self.max_lines,
        )

    def create_random(self, program_id: str = "") -> CoverageStatechartGenome:
        """Create a random valid genome using observed patterns when available."""
        n_states = random.randint(self.min_states, self.max_states)

        # Create states
        states = {}

        # Entry state - always covers first lines
        entry_lines = {1, 2}
        if self.common_line_groups:
            # Find group containing line 1 or 2
            for group in self.common_line_groups:
                if 1 in group or 2 in group:
                    entry_lines = group
                    break

        entry = EvolvedState(
            id="entry",
            state_type=EvolvedStateType.ENTRY,
            lines=entry_lines,
            is_initial=True,
        )
        states["entry"] = entry

        # Exit state
        exit_state = EvolvedState(
            id="exit",
            state_type=EvolvedStateType.EXIT,
            lines=set(),
            is_final=True,
        )
        states["exit"] = exit_state

        # Intermediate states - use observed line groups if available
        used_groups = set()
        for i in range(n_states - 2):
            state_type = random.choice([
                EvolvedStateType.BASIC,
                EvolvedStateType.BRANCH,
                EvolvedStateType.LOOP,
            ])

            # Assign lines from observed patterns
            if self.common_line_groups:
                # Pick from observed line groups
                available = [j for j, g in enumerate(self.common_line_groups)
                            if j not in used_groups and g not in [entry_lines]]
                if available:
                    idx = random.choice(available)
                    lines = self.common_line_groups[idx].copy()
                    used_groups.add(idx)
                else:
                    # Fall back to random from observed
                    if self.observed_line_patterns:
                        pattern = random.choice(self.observed_line_patterns)
                        n_to_pick = min(len(pattern), random.randint(2, 4))
                        lines = set(random.sample(list(pattern), n_to_pick))
                    else:
                        lines = {random.randint(1, self.max_lines)}
            else:
                # Random lines
                n_lines = random.randint(1, 4)
                start_line = random.randint(2, self.max_lines - n_lines)
                lines = set(range(start_line, start_line + n_lines))

            state = EvolvedState(
                id=f"s{i}",
                state_type=state_type,
                lines=lines,
            )
            states[state.id] = state

        # Create transitions
        transitions = []
        state_ids = list(states.keys())

        # Ensure path from entry to exit
        # Entry -> first state
        if len(state_ids) > 2:
            mid_states = [s for s in state_ids if s not in ("entry", "exit")]
            random.shuffle(mid_states)

            # Chain: entry -> s0 -> s1 -> ... -> exit
            prev = "entry"
            for s in mid_states:
                transitions.append(EvolvedTransition(
                    from_state=prev,
                    to_state=s,
                ))
                prev = s

            transitions.append(EvolvedTransition(
                from_state=prev,
                to_state="exit",
            ))

            # Add some random transitions with guards
            for _ in range(random.randint(1, n_states)):
                from_s = random.choice(mid_states)
                to_s = random.choice([s for s in state_ids if s != from_s])

                guard = None
                if random.random() < 0.5:
                    guard = self.create_random_guard()

                transitions.append(EvolvedTransition(
                    from_state=from_s,
                    to_state=to_s,
                    guard=guard,
                ))
        else:
            # Direct entry -> exit
            transitions.append(EvolvedTransition(
                from_state="entry",
                to_state="exit",
            ))

        genome = CoverageStatechartGenome(
            states=states,
            transitions=transitions,
            program_id=program_id,
            max_lines=self.max_lines,
        )

        return genome

    def create_random_guard(self) -> EvolvedGuard:
        """Create a random guard expression."""
        template = random.choice(self.GUARD_TEMPLATES)
        const = random.choice([0, 1, -1, 5, 10])
        expr = template.replace("{c}", str(const))
        return EvolvedGuard(expression=expr)

    def mutate(self, genome: CoverageStatechartGenome) -> CoverageStatechartGenome:
        """Mutate a genome."""
        mutated = genome.copy()

        # Choose mutation type
        mutation = random.choice([
            'add_state',
            'remove_state',
            'add_transition',
            'remove_transition',
            'mutate_guard',
            'mutate_lines',
            'toggle_type',
        ])

        if mutation == 'add_state' and len(mutated.states) < self.max_states:
            self._mutate_add_state(mutated)
        elif mutation == 'remove_state' and len(mutated.states) > self.min_states:
            self._mutate_remove_state(mutated)
        elif mutation == 'add_transition':
            self._mutate_add_transition(mutated)
        elif mutation == 'remove_transition' and len(mutated.transitions) > 1:
            self._mutate_remove_transition(mutated)
        elif mutation == 'mutate_guard':
            self._mutate_guard(mutated)
        elif mutation == 'mutate_lines':
            self._mutate_lines(mutated)
        elif mutation == 'toggle_type':
            self._mutate_toggle_type(mutated)

        mutated._id = None  # Reset cached ID
        return mutated

    def _mutate_add_state(self, genome: CoverageStatechartGenome):
        """Add a new state."""
        new_id = f"s{len(genome.states)}"
        state_type = random.choice([
            EvolvedStateType.BASIC,
            EvolvedStateType.BRANCH,
        ])

        n_lines = random.randint(1, 3)
        start_line = random.randint(2, self.max_lines - n_lines)
        lines = set(range(start_line, start_line + n_lines))

        new_state = EvolvedState(
            id=new_id,
            state_type=state_type,
            lines=lines,
        )
        genome.states[new_id] = new_state

        # Connect to existing states
        existing = [s for s in genome.states.keys() if s not in (new_id, "exit")]
        if existing:
            from_s = random.choice(existing)
            genome.transitions.append(EvolvedTransition(
                from_state=from_s,
                to_state=new_id,
            ))
            genome.transitions.append(EvolvedTransition(
                from_state=new_id,
                to_state=random.choice(list(genome.states.keys())),
            ))

    def _mutate_remove_state(self, genome: CoverageStatechartGenome):
        """Remove a state."""
        removable = [s for s in genome.states.keys()
                     if not genome.states[s].is_initial and not genome.states[s].is_final]
        if not removable:
            return

        to_remove = random.choice(removable)
        del genome.states[to_remove]

        # Remove transitions involving this state
        genome.transitions = [t for t in genome.transitions
                             if t.from_state != to_remove and t.to_state != to_remove]

    def _mutate_add_transition(self, genome: CoverageStatechartGenome):
        """Add a new transition."""
        state_ids = list(genome.states.keys())
        from_s = random.choice([s for s in state_ids if s != "exit"])
        to_s = random.choice([s for s in state_ids if s != from_s])

        guard = self.create_random_guard() if random.random() < 0.5 else None

        genome.transitions.append(EvolvedTransition(
            from_state=from_s,
            to_state=to_s,
            guard=guard,
        ))

    def _mutate_remove_transition(self, genome: CoverageStatechartGenome):
        """Remove a transition."""
        if genome.transitions:
            idx = random.randint(0, len(genome.transitions) - 1)
            genome.transitions.pop(idx)

    def _mutate_guard(self, genome: CoverageStatechartGenome):
        """Mutate a transition's guard."""
        if not genome.transitions:
            return

        trans = random.choice(genome.transitions)
        if random.random() < 0.3:
            trans.guard = None
        else:
            trans.guard = self.create_random_guard()

    def _mutate_lines(self, genome: CoverageStatechartGenome):
        """Mutate a state's line coverage."""
        modifiable = [s for s in genome.states.values()
                      if not s.is_initial and not s.is_final]
        if not modifiable:
            return

        state = random.choice(modifiable)

        if random.random() < 0.5 and state.lines:
            # Remove a line
            state.lines.discard(random.choice(list(state.lines)))
        else:
            # Add a line from observed patterns if available
            if self.observed_line_patterns:
                # Pick from observed patterns
                pattern = random.choice(self.observed_line_patterns)
                new_line = random.choice(list(pattern))
            else:
                new_line = random.randint(1, self.max_lines)
            state.lines.add(new_line)

    def _mutate_toggle_type(self, genome: CoverageStatechartGenome):
        """Toggle a state's type."""
        modifiable = [s for s in genome.states.values()
                      if s.state_type not in (EvolvedStateType.ENTRY, EvolvedStateType.EXIT)]
        if not modifiable:
            return

        state = random.choice(modifiable)
        types = [EvolvedStateType.BASIC, EvolvedStateType.BRANCH, EvolvedStateType.LOOP]
        state.state_type = random.choice([t for t in types if t != state.state_type])

    def crossover(
        self,
        parent1: CoverageStatechartGenome,
        parent2: CoverageStatechartGenome,
    ) -> CoverageStatechartGenome:
        """Crossover two genomes."""
        # Take states from both parents
        child_states = {}

        # Always include entry and exit
        if "entry" in parent1.states:
            child_states["entry"] = parent1.states["entry"].copy()
        if "exit" in parent1.states:
            child_states["exit"] = parent1.states["exit"].copy()

        # Mix other states
        other_states_1 = [s for s in parent1.states.values()
                         if not s.is_initial and not s.is_final]
        other_states_2 = [s for s in parent2.states.values()
                         if not s.is_initial and not s.is_final]

        # Take random subset from each
        n_from_1 = random.randint(0, len(other_states_1))
        n_from_2 = random.randint(0, len(other_states_2))

        for s in random.sample(other_states_1, min(n_from_1, len(other_states_1))):
            child_states[s.id] = s.copy()

        for s in random.sample(other_states_2, min(n_from_2, len(other_states_2))):
            new_id = f"p2_{s.id}"
            new_state = s.copy()
            new_state.id = new_id
            child_states[new_id] = new_state

        # Take transitions that connect valid states
        child_transitions = []
        state_ids = set(child_states.keys())

        for t in parent1.transitions + parent2.transitions:
            if t.from_state in state_ids and t.to_state in state_ids:
                child_transitions.append(t.copy())

        # Ensure connectivity
        if not any(t.from_state == "entry" for t in child_transitions):
            # Add entry transition
            targets = [s for s in state_ids if s != "entry"]
            if targets:
                child_transitions.append(EvolvedTransition(
                    from_state="entry",
                    to_state=random.choice(targets),
                ))

        if not any(t.to_state == "exit" for t in child_transitions):
            # Add exit transition
            sources = [s for s in state_ids if s != "exit"]
            if sources:
                child_transitions.append(EvolvedTransition(
                    from_state=random.choice(sources),
                    to_state="exit",
                ))

        child = CoverageStatechartGenome(
            states=child_states,
            transitions=child_transitions,
            program_id=parent1.program_id,
            max_lines=max(parent1.max_lines, parent2.max_lines),
        )

        return child


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

class CoverageEvaluator:
    """Evaluate genome fitness on coverage prediction task."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.cache: Dict[str, float] = {}

    def evaluate(
        self,
        genome: CoverageStatechartGenome,
        examples: Optional[List[DatasetExample]] = None,
    ) -> Tuple[float, EvaluationMetrics]:
        """
        Evaluate genome on coverage prediction.

        Returns (fitness, metrics).
        """
        # Check cache
        if genome.id in self.cache:
            # Return cached fitness (metrics not cached)
            return self.cache[genome.id], EvaluationMetrics()

        if examples is None:
            examples = self.dataset.examples

        metrics_list = []

        for example in examples:
            # Parse input
            try:
                input_val = eval(example.input_repr, {"__builtins__": {}})
            except:
                input_val = None

            # Build context
            context = {'x': input_val, 'items': input_val}
            if isinstance(input_val, (list, tuple)):
                context['items'] = input_val
                if input_val:
                    context['x'] = input_val[0]
            elif isinstance(input_val, tuple) and len(input_val) >= 1:
                context['x'] = input_val[0]
                if len(input_val) >= 2:
                    context['y'] = input_val[1]

            # Predict coverage
            predicted = genome.predict_coverage(context)

            # Actual coverage
            actual = set(example.covered_lines)
            all_lines = set(range(1, example.n_lines + 1))

            # Compute metrics
            m = compute_metrics(predicted, actual, all_lines)
            metrics_list.append(m)

        # Aggregate
        aggregated = aggregate_metrics(metrics_list)

        # Fitness = F1 score with parsimony penalty
        parsimony = 0.01 * genome.n_states  # Prefer simpler models
        fitness = aggregated.f1 - parsimony

        # Cache
        self.cache[genome.id] = fitness

        # Store in genome
        genome.fitness = fitness
        genome.f1_score = aggregated.f1
        genome.precision = aggregated.precision
        genome.recall = aggregated.recall

        return fitness, aggregated


# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for statechart evolution."""
    population_size: int = 50
    n_generations: int = 100
    elite_size: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_states: int = 15
    min_states: int = 3
    max_lines: int = 30
    verbose: bool = True
    log_every: int = 10


class StatechartEvolver:
    """
    Evolve statecharts for optimal coverage prediction.

    This is the main evolution engine that discovers the best
    statechart structure for predicting code coverage.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: EvolutionConfig = None,
    ):
        self.dataset = dataset
        self.config = config or EvolutionConfig()

        # Extract observed line patterns from dataset
        observed_patterns = [set(ex.covered_lines) for ex in dataset.examples]

        self.factory = GenomeFactory(
            max_states=self.config.max_states,
            min_states=self.config.min_states,
            max_lines=self.config.max_lines,
            observed_line_patterns=observed_patterns,
        )
        self.evaluator = CoverageEvaluator(dataset)

    def initialize_population(self) -> List[CoverageStatechartGenome]:
        """Create initial population seeded with observation-based genomes."""
        population = []

        # Seed with observation-based genomes (50% of population)
        n_seeded = self.config.population_size // 2
        for _ in range(n_seeded):
            genome = self.factory.create_from_observations()
            if genome.is_valid():
                population.append(genome)
                # Also add a mutated variant
                if len(population) < self.config.population_size:
                    mutated = self.factory.mutate(genome)
                    if mutated.is_valid():
                        population.append(mutated)

        # Fill rest with random genomes
        attempts = 0
        max_attempts = self.config.population_size * 3

        while len(population) < self.config.population_size and attempts < max_attempts:
            genome = self.factory.create_random()
            if genome.is_valid():
                population.append(genome)
            attempts += 1

        return population[:self.config.population_size]

    def evaluate_population(
        self,
        population: List[CoverageStatechartGenome],
    ) -> List[float]:
        """Evaluate fitness for all genomes."""
        fitness_list = []
        for genome in population:
            fitness, _ = self.evaluator.evaluate(genome)
            fitness_list.append(fitness)
        return fitness_list

    def select_parents(
        self,
        population: List[CoverageStatechartGenome],
        fitness_list: List[float],
    ) -> List[CoverageStatechartGenome]:
        """Select parents using tournament selection."""
        parents = []
        n_parents = self.config.population_size // 2

        for _ in range(n_parents):
            candidates = random.sample(
                list(range(len(population))),
                min(self.config.tournament_size, len(population))
            )
            winner = max(candidates, key=lambda i: fitness_list[i])
            parents.append(population[winner])

        return parents

    def create_offspring(
        self,
        parents: List[CoverageStatechartGenome],
    ) -> List[CoverageStatechartGenome]:
        """Create offspring from parents."""
        offspring = []

        while len(offspring) < self.config.population_size - self.config.elite_size:
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
                child = self.factory.crossover(p1, p2)
            else:
                parent = random.choice(parents)
                child = parent.copy()

            if random.random() < self.config.mutation_rate:
                child = self.factory.mutate(child)

            if child.is_valid():
                offspring.append(child)

        return offspring

    def evolve(self) -> CoverageStatechartGenome:
        """
        Run evolution to find optimal statechart.

        Returns best genome found.
        """
        if self.config.verbose:
            print("=" * 60)
            print("STATECHART EVOLUTION FOR COVERAGE PREDICTION")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print(f"Dataset: {len(self.dataset)} examples")
            print("-" * 60)

        start_time = time.time()

        # Initialize
        population = self.initialize_population()
        fitness_list = self.evaluate_population(population)

        # Track best
        best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
        best_ever = population[best_idx].copy()

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Sort by fitness
            sorted_pairs = sorted(
                zip(population, fitness_list),
                key=lambda x: x[1],
                reverse=True
            )
            population = [p for p, _ in sorted_pairs]
            fitness_list = [f for _, f in sorted_pairs]

            # Elitism
            elite = [g.copy() for g in population[:self.config.elite_size]]

            # Select and breed
            parents = self.select_parents(population, fitness_list)
            offspring = self.create_offspring(parents)

            # New population
            population = elite + offspring
            fitness_list = self.evaluate_population(population)

            # Track best
            best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
            if fitness_list[best_idx] > best_ever.fitness:
                best_ever = population[best_idx].copy()

            # Log
            if self.config.verbose and (gen % self.config.log_every == 0 or gen == self.config.n_generations - 1):
                best = population[best_idx]
                avg_fitness = sum(fitness_list) / len(fitness_list)
                avg_states = sum(g.n_states for g in population) / len(population)
                print(
                    f"Gen {gen:3d} | "
                    f"Best: F1={best.f1_score:.3f} (states={best.n_states}) | "
                    f"Avg: {avg_fitness:.3f} | "
                    f"States: {avg_states:.1f}"
                )

        elapsed = time.time() - start_time

        if self.config.verbose:
            print("-" * 60)
            print(f"Evolution complete in {elapsed:.1f}s")
            print(f"Best genome: {best_ever.n_states} states, F1={best_ever.f1_score:.3f}")
            print("=" * 60)

        return best_ever


# =============================================================================
# COMPARISON WITH HARDCODED
# =============================================================================

def compare_evolved_vs_hardcoded(
    dataset: Dataset,
    n_generations: int = 50,
    population_size: int = 30,
) -> Dict:
    """
    Compare evolved statechart vs hardcoded CFG conversion.

    Returns comparison results.
    """
    from .statechart_predictor import StatechartCoveragePredictor, StatechartConfig
    from .program_statechart import build_cfg

    print("=" * 60)
    print("EVOLVED vs HARDCODED COMPARISON")
    print("=" * 60)

    # Evaluate hardcoded approach
    print("\n1. Evaluating HARDCODED CFG approach...")
    config = StatechartConfig(max_iterations=50)
    hardcoded = StatechartCoveragePredictor(config)

    hardcoded_metrics = []
    for example in dataset.examples:
        try:
            input_val = eval(example.input_repr, {"__builtins__": {}})
        except:
            input_val = None

        predicted, _ = hardcoded.predict_coverage(example.source, input_val)
        actual = set(example.covered_lines)
        all_lines = set(range(1, example.n_lines + 1))

        m = compute_metrics(predicted, actual, all_lines)
        hardcoded_metrics.append(m)

    hardcoded_agg = aggregate_metrics(hardcoded_metrics)
    print(f"   Hardcoded F1: {hardcoded_agg.f1:.3f}")

    # Evolve optimized statechart
    print(f"\n2. EVOLVING optimized statechart ({n_generations} generations)...")
    evol_config = EvolutionConfig(
        population_size=population_size,
        n_generations=n_generations,
        verbose=True,
        log_every=10,
    )

    evolver = StatechartEvolver(dataset, evol_config)
    best_evolved = evolver.evolve()

    print(f"\n3. Final comparison:")
    print("-" * 40)
    print(f"   Hardcoded CFG:   F1={hardcoded_agg.f1:.3f}, Prec={hardcoded_agg.precision:.3f}, Rec={hardcoded_agg.recall:.3f}")
    print(f"   Evolved:         F1={best_evolved.f1_score:.3f}, Prec={best_evolved.precision:.3f}, Rec={best_evolved.recall:.3f}")
    print(f"   Evolved states:  {best_evolved.n_states}")

    improvement = best_evolved.f1_score - hardcoded_agg.f1
    print(f"\n   Improvement: {improvement:+.3f} ({improvement/hardcoded_agg.f1*100:+.1f}%)")

    if improvement > 0:
        print("\n   EVOLVED structure OUTPERFORMS hardcoded!")
    else:
        print("\n   Hardcoded structure performs better (may need more evolution)")

    return {
        'hardcoded_f1': hardcoded_agg.f1,
        'evolved_f1': best_evolved.f1_score,
        'improvement': improvement,
        'evolved_states': best_evolved.n_states,
        'best_genome': best_evolved,
    }


def demo():
    """Demonstrate statechart evolution."""
    print("=" * 60)
    print("STATECHART EVOLUTION DEMO")
    print("=" * 60)

    # Generate dataset
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate_dataset()

    # Use subset for faster demo
    subset = Dataset(examples=dataset.examples[:100])
    print(f"\nUsing {len(subset)} examples for demo")

    # Run comparison
    results = compare_evolved_vs_hardcoded(
        subset,
        n_generations=30,
        population_size=20,
    )

    return results


if __name__ == "__main__":
    demo()
