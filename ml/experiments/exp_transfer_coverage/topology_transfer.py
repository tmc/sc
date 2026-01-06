"""
Topology Transfer for Coverage Statecharts

Extract structural patterns from evolved coverage statecharts and transfer
them to new programs. Key insight: control flow PATTERNS generalize even
when specific line numbers don't.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum, auto

from ..exp_coverage_prediction.statechart_evolver import (
    CoverageStatechartGenome,
    EvolvedState,
    EvolvedTransition,
    EvolvedGuard,
    EvolvedStateType,
    GenomeFactory,
    CoverageEvaluator,
    EvolutionConfig,
)
from ..exp_coverage_prediction.dataset import Dataset, DatasetExample


class PatternType(Enum):
    SEQUENCE = auto()
    BRANCH = auto()
    MULTI_BRANCH = auto()
    LOOP = auto()
    NESTED_LOOP = auto()


@dataclass
class AbstractState:
    id: str
    pattern_role: str
    state_type: EvolvedStateType
    is_initial: bool = False
    is_final: bool = False
    relative_position: float = 0.0


@dataclass
class AbstractTransition:
    from_state: str
    to_state: str
    guard_type: str
    is_conditional: bool = False


@dataclass
class AbstractTopology:
    states: Dict[str, AbstractState]
    transitions: List[AbstractTransition]
    pattern_type: PatternType
    original_f1: float = 0.0
    n_training_examples: int = 0

    def n_states(self) -> int:
        return len(self.states)

    def n_transitions(self) -> int:
        return len(self.transitions)


def extract_topology(genome: CoverageStatechartGenome) -> AbstractTopology:
    """Extract abstract topology from trained coverage statechart."""
    abstract_states = {}
    all_lines = set()
    for state in genome.states.values():
        all_lines |= state.lines

    min_line = min(all_lines) if all_lines else 1
    max_line = max(all_lines) if all_lines else 1
    line_range = max_line - min_line if max_line > min_line else 1

    for state_id, state in genome.states.items():
        role = _infer_pattern_role(state, genome)
        if state.lines:
            avg_line = sum(state.lines) / len(state.lines)
            rel_pos = (avg_line - min_line) / line_range
        else:
            rel_pos = 1.0 if state.is_final else 0.0

        abstract_states[state_id] = AbstractState(
            id=state_id, pattern_role=role, state_type=state.state_type,
            is_initial=state.is_initial, is_final=state.is_final,
            relative_position=rel_pos,
        )

    abstract_transitions = []
    for trans in genome.transitions:
        guard_type = _infer_guard_type(trans)
        abstract_transitions.append(AbstractTransition(
            from_state=trans.from_state, to_state=trans.to_state,
            guard_type=guard_type, is_conditional=trans.guard is not None,
        ))

    pattern_type = _infer_pattern_type(abstract_states, abstract_transitions)
    return AbstractTopology(
        states=abstract_states, transitions=abstract_transitions,
        pattern_type=pattern_type, original_f1=genome.f1_score,
    )


def _infer_pattern_role(state: EvolvedState, genome: CoverageStatechartGenome) -> str:
    if state.is_initial:
        return "entry"
    if state.is_final:
        return "exit"
    incoming = sum(1 for t in genome.transitions if t.to_state == state.id)
    conditional_out = sum(1 for t in genome.transitions
                         if t.from_state == state.id and t.guard is not None)
    if conditional_out >= 2:
        return "branch_point"
    if incoming > 1:
        return "merge_point"
    if state.state_type == EvolvedStateType.LOOP:
        return "loop_body"
    return "sequence"


def _infer_guard_type(trans: EvolvedTransition) -> str:
    if trans.guard is None:
        return "unconditional"
    expr = trans.guard.expression.lower()
    if ">" in expr:
        return "greater_than"
    if "<" in expr:
        return "less_than"
    if "==" in expr:
        return "equals"
    if "len(" in expr:
        return "length_check"
    return "conditional"


def _infer_pattern_type(states: Dict[str, AbstractState], transitions: List[AbstractTransition]) -> PatternType:
    n_conditional = sum(1 for t in transitions if t.is_conditional)
    n_branch_points = sum(1 for s in states.values() if s.pattern_role == "branch_point")
    has_back_edge = any(
        states.get(t.to_state, AbstractState("", "", EvolvedStateType.BASIC)).relative_position <
        states.get(t.from_state, AbstractState("", "", EvolvedStateType.BASIC)).relative_position - 0.1
        for t in transitions
    )
    if has_back_edge and n_branch_points > 0:
        return PatternType.NESTED_LOOP
    if has_back_edge:
        return PatternType.LOOP
    if n_branch_points > 1:
        return PatternType.MULTI_BRANCH
    if n_conditional > 0:
        return PatternType.BRANCH
    return PatternType.SEQUENCE


def adapt_topology(topology: AbstractTopology, target_dataset: Dataset, max_line: int = 30) -> CoverageStatechartGenome:
    """Adapt abstract topology to target dataset."""
    line_groups = _discover_line_groups(target_dataset)
    sorted_groups = sorted(line_groups, key=lambda g: sum(g) / len(g) if g else 0)

    concrete_states = {}
    sorted_abstract = sorted(topology.states.values(), key=lambda s: s.relative_position)

    for i, abstract in enumerate(sorted_abstract):
        if abstract.is_initial:
            lines = {1, 2}
        elif abstract.is_final:
            lines = set()
        elif i < len(sorted_groups):
            lines = sorted_groups[i].copy()
        else:
            start = int(abstract.relative_position * (max_line - 2)) + 2
            lines = {start, start + 1}

        concrete_states[abstract.id] = EvolvedState(
            id=abstract.id, state_type=abstract.state_type, lines=lines,
            is_initial=abstract.is_initial, is_final=abstract.is_final,
        )

    concrete_transitions = []
    for abstract_trans in topology.transitions:
        guard = _create_guard_for_type(abstract_trans.guard_type) if abstract_trans.is_conditional else None
        concrete_transitions.append(EvolvedTransition(
            from_state=abstract_trans.from_state, to_state=abstract_trans.to_state, guard=guard,
        ))

    return CoverageStatechartGenome(states=concrete_states, transitions=concrete_transitions, max_lines=max_line)


def _discover_line_groups(dataset: Dataset) -> List[Set[int]]:
    cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
    line_counts: Dict[int, int] = defaultdict(int)
    for example in dataset.examples:
        lines = sorted(example.covered_lines)
        for line in lines:
            line_counts[line] += 1
        for i, l1 in enumerate(lines):
            for l2 in lines[i+1:]:
                cooccurrence[(l1, l2)] += 1

    all_lines = set(line_counts.keys())
    groups = []
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
            if either > 0 and together / either > 0.7:
                group.add(other)
        if group:
            groups.append(group)
            visited |= group
    return groups


def _create_guard_for_type(guard_type: str) -> EvolvedGuard:
    templates = {"greater_than": "x > 0", "less_than": "x < 0", "equals": "x == 0",
                 "length_check": "len(items) > 0", "conditional": "x", "unconditional": "True"}
    return EvolvedGuard(expression=templates.get(guard_type, "True"))


class TopologyTransferLearner:
    """Learn to transfer coverage topologies across programs."""

    def __init__(self, source_dataset: Dataset, target_dataset: Dataset,
                 evolution_config: Optional[EvolutionConfig] = None):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.config = evolution_config or EvolutionConfig(population_size=20, n_generations=30, verbose=False)
        self.source_genome: Optional[CoverageStatechartGenome] = None
        self.topology: Optional[AbstractTopology] = None
        self.adapted_genome: Optional[CoverageStatechartGenome] = None
        self.finetuned_genome: Optional[CoverageStatechartGenome] = None

    def train_on_source(self) -> CoverageStatechartGenome:
        from ..exp_coverage_prediction.statechart_evolver import StatechartEvolver
        evolver = StatechartEvolver(self.source_dataset, self.config)
        self.source_genome = evolver.evolve()
        return self.source_genome

    def extract_topology(self) -> AbstractTopology:
        if self.source_genome is None:
            raise ValueError("Must train on source first")
        self.topology = extract_topology(self.source_genome)
        self.topology.n_training_examples = len(self.source_dataset)
        return self.topology

    def adapt_to_target(self) -> CoverageStatechartGenome:
        if self.topology is None:
            raise ValueError("Must extract topology first")
        max_line = max(max(ex.covered_lines) if ex.covered_lines else 1
                      for ex in self.target_dataset.examples) if self.target_dataset.examples else 30
        self.adapted_genome = adapt_topology(self.topology, self.target_dataset, max_line=max_line)
        return self.adapted_genome

    def finetune_guards(self, n_generations: int = 20) -> CoverageStatechartGenome:
        if self.adapted_genome is None:
            raise ValueError("Must adapt to target first")

        population = [self.adapted_genome.copy()]
        factory = GenomeFactory(max_states=len(self.adapted_genome.states),
                               min_states=len(self.adapted_genome.states),
                               max_lines=self.adapted_genome.max_lines)

        for _ in range(self.config.population_size - 1):
            variant = self.adapted_genome.copy()
            for trans in variant.transitions:
                if trans.guard and random.random() < 0.3:
                    trans.guard = factory.create_random_guard()
            population.append(variant)

        evaluator = CoverageEvaluator(self.target_dataset)
        for gen in range(n_generations):
            fitness_list = [evaluator.evaluate(g)[0] for g in population]
            sorted_pairs = sorted(zip(population, fitness_list), key=lambda x: -x[1])
            elite = [g.copy() for g, _ in sorted_pairs[:5]]

            offspring = []
            while len(offspring) < self.config.population_size - len(elite):
                parent = random.choice(elite)
                child = parent.copy()
                for trans in child.transitions:
                    if random.random() < 0.2:
                        trans.guard = factory.create_random_guard() if random.random() < 0.5 else None
                child._id = None
                offspring.append(child)
            population = elite + offspring

        fitness_list = [evaluator.evaluate(g)[0] for g in population]
        best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
        self.finetuned_genome = population[best_idx]
        evaluator.evaluate(self.finetuned_genome)
        return self.finetuned_genome

    def full_transfer(self, verbose: bool = True) -> Dict[str, Any]:
        results = {}
        if verbose:
            print("=" * 60)
            print("COVERAGE TOPOLOGY TRANSFER")
            print("=" * 60)

        if verbose:
            print("\n[1/5] Training on source...")
        self.train_on_source()
        results['source_f1'] = self.source_genome.f1_score
        results['source_states'] = self.source_genome.n_states

        if verbose:
            print(f"      F1={results['source_f1']:.3f}, states={results['source_states']}")
            print("\n[2/5] Extracting topology...")
        self.extract_topology()
        results['pattern_type'] = self.topology.pattern_type.name

        if verbose:
            print(f"      Pattern: {results['pattern_type']}")
            print("\n[3/5] Adapting to target (zero-shot)...")
        self.adapt_to_target()
        evaluator = CoverageEvaluator(self.target_dataset)
        evaluator.evaluate(self.adapted_genome)
        results['zeroshot_f1'] = self.adapted_genome.f1_score

        if verbose:
            print(f"      Zero-shot F1={results['zeroshot_f1']:.3f}")
            print("\n[4/5] Fine-tuning guards...")
        self.finetune_guards(n_generations=20)
        results['finetuned_f1'] = self.finetuned_genome.f1_score

        if verbose:
            print(f"      Fine-tuned F1={results['finetuned_f1']:.3f}")
            print("\n[5/5] Training from scratch (baseline)...")
        from ..exp_coverage_prediction.statechart_evolver import StatechartEvolver
        scratch_evolver = StatechartEvolver(self.target_dataset, self.config)
        scratch_genome = scratch_evolver.evolve()
        results['scratch_f1'] = scratch_genome.f1_score

        results['transfer_benefit'] = results['finetuned_f1'] - results['scratch_f1']
        if verbose:
            print(f"      From-scratch F1={results['scratch_f1']:.3f}")
            print("\n" + "-" * 40)
            print(f"  Transfer benefit: {results['transfer_benefit']:+.3f}")
        return results
