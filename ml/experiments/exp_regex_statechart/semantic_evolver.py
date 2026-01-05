"""
Semantic Evolution for Extended Statecharts.

Evolves all aspects of statecharts:
- Topology: states and transitions
- Guards: character matching + counter/flag conditions
- Actions: increment, reset, set flags
- Variables: add/remove counters, flags
- Accept conditions: final state requirements

This enables learning patterns that require memory and counting.
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable
from enum import Enum, auto

from .regex_statechart import StateType
from .semantic_components import (
    StateVar, VarType, ExtendedState,
    GuardExpr, CharGuardExpr, CounterGuardExpr, FlagGuardExpr, CompositeGuardExpr,
    ActionExpr, NoOpAction, IncrementAction, ResetAction, SetFlagAction,
    CompositeAction, SemanticTransition, ComponentFactory
)
from .extended_statechart import ExtendedStatechart, AcceptCondition, StateActions


class MutationType(Enum):
    """Types of mutations for semantic evolution."""
    # Topology
    ADD_STATE = auto()
    REMOVE_STATE = auto()
    TOGGLE_ACCEPT = auto()

    # Transitions
    ADD_TRANSITION = auto()
    REMOVE_TRANSITION = auto()
    CHANGE_TARGET = auto()

    # Guards
    MUTATE_GUARD = auto()
    ADD_GUARD_CONDITION = auto()
    SIMPLIFY_GUARD = auto()

    # Actions
    MUTATE_ACTION = auto()
    ADD_ACTION = auto()
    REMOVE_ACTION = auto()

    # Variables
    ADD_VARIABLE = auto()
    REMOVE_VARIABLE = auto()

    # Accept conditions
    MUTATE_ACCEPT_CONDITION = auto()


@dataclass
class SemanticEvolutionConfig:
    """Configuration for semantic evolution."""
    # Population
    population_size: int = 50
    elite_size: int = 5
    n_generations: int = 100

    # Genome limits
    max_states: int = 15
    min_states: int = 2
    max_transitions: int = 40
    max_variables: int = 5

    # Mutation rates
    mutation_rate: float = 0.5
    crossover_rate: float = 0.5

    # Mutation weights (normalized)
    topology_weight: float = 0.25  # State add/remove, toggle accept
    transition_weight: float = 0.25  # Add/remove/change transitions
    guard_weight: float = 0.20  # Mutate guards
    action_weight: float = 0.15  # Mutate actions
    variable_weight: float = 0.10  # Add/remove variables
    accept_weight: float = 0.05  # Accept conditions

    # Fitness
    size_penalty: float = 0.005
    complexity_penalty: float = 0.001

    # Selection
    tournament_size: int = 3

    # Alphabet
    alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789"

    # Variables
    counter_names: List[str] = field(default_factory=lambda: ["count", "n", "reps"])
    flag_names: List[str] = field(default_factory=lambda: ["seen", "matched", "done"])

    # Logging
    log_every: int = 10
    verbose: bool = True

    # Early stopping
    target_f1: float = 0.99


@dataclass
class SemanticEvolutionStats:
    """Statistics from semantic evolution."""
    generations: int = 0
    best_f1: float = 0.0
    best_accuracy: float = 0.0
    avg_f1: float = 0.0
    elapsed_time: float = 0.0
    evaluations: int = 0


class SemanticEvolver:
    """
    Evolve extended statecharts with full semantic components.

    Key innovations:
    1. Co-evolves topology AND semantics
    2. Discovers counters/flags when needed
    3. Learns accept conditions for counting patterns
    4. Composite guard evolution (AND/OR combinations)
    """

    def __init__(self, config: SemanticEvolutionConfig = None):
        """Initialize semantic evolver."""
        self.config = config or SemanticEvolutionConfig()
        self.stats = SemanticEvolutionStats()

        # Component factory
        self.factory = ComponentFactory(
            alphabet=self.config.alphabet,
            counter_vars=self.config.counter_names,
            flag_vars=self.config.flag_names
        )

        # Normalize mutation weights
        total = (
            self.config.topology_weight +
            self.config.transition_weight +
            self.config.guard_weight +
            self.config.action_weight +
            self.config.variable_weight +
            self.config.accept_weight
        )
        self._mutation_probs = {
            'topology': self.config.topology_weight / total,
            'transition': self.config.transition_weight / total,
            'guard': self.config.guard_weight / total,
            'action': self.config.action_weight / total,
            'variable': self.config.variable_weight / total,
            'accept': self.config.accept_weight / total,
        }

    def create_random(self) -> ExtendedStatechart:
        """Create a random extended statechart."""
        n_states = random.randint(self.config.min_states, min(4, self.config.max_states))

        # Create states
        labels = ["START"] + [f"S{i}" for i in range(1, n_states - 1)] + ["ACCEPT"]
        types = [StateType.START]
        types += [StateType.INTERMEDIATE] * (n_states - 2)
        types.append(StateType.ACCEPT)

        # Maybe add variables
        variables = []
        if random.random() < 0.5:
            var_name = random.choice(self.config.counter_names)
            variables.append(StateVar(name=var_name, var_type=VarType.COUNTER, initial_value=0))
        if random.random() < 0.3:
            var_name = random.choice(self.config.flag_names)
            variables.append(StateVar(name=var_name, var_type=VarType.FLAG, initial_value=False))

        # Create random transitions
        transitions = []
        n_trans = random.randint(n_states - 1, n_states * 2)

        for _ in range(n_trans):
            trans = self._random_transition(n_states, variables)
            transitions.append(trans)

        # Ensure START has outgoing transition
        if not any(t.source == 0 for t in transitions):
            transitions.append(self._random_transition(n_states, variables, source=0))

        sc = ExtendedStatechart(
            state_labels=labels,
            state_types=types,
            variables=variables,
            transitions=transitions,
            initial_state=0
        )

        return sc

    def _random_transition(
        self,
        n_states: int,
        variables: List[StateVar],
        source: int = None
    ) -> SemanticTransition:
        """Create a random semantic transition."""
        src = source if source is not None else random.randint(0, n_states - 1)
        tgt = random.randint(0, n_states - 1)

        # Create guard
        guard = self._random_guard(variables)

        # Create action
        action = self._random_action(variables)

        return SemanticTransition(source=src, target=tgt, guard=guard, action=action)

    def _random_guard(self, variables: List[StateVar]) -> GuardExpr:
        """Create a random guard expression."""
        choice = random.random()

        # Simple character guard most common
        if choice < 0.6 or not variables:
            return self.factory.random_char_guard()

        # Counter guard if we have counters
        counters = [v for v in variables if v.var_type == VarType.COUNTER]
        flags = [v for v in variables if v.var_type == VarType.FLAG]

        if choice < 0.75 and counters:
            var = random.choice(counters)
            op = random.choice(['<', '<=', '==', '>=', '>'])
            val = random.randint(0, 5)
            return CounterGuardExpr(var_name=var.name, operator=op, value=val)

        if choice < 0.85 and flags:
            var = random.choice(flags)
            return FlagGuardExpr(var_name=var.name, expected=random.choice([True, False]))

        # Composite guard
        if choice < 0.95:
            left = self.factory.random_char_guard()
            if counters:
                var = random.choice(counters)
                right = CounterGuardExpr(var_name=var.name, operator=random.choice(['<', '==']), value=random.randint(1, 4))
            else:
                right = self.factory.random_char_guard()
            return CompositeGuardExpr(left=left, right=right, operator=random.choice(['and', 'or']))

        return self.factory.random_char_guard()

    def _random_action(self, variables: List[StateVar]) -> ActionExpr:
        """Create a random action."""
        if not variables or random.random() < 0.4:
            return NoOpAction()

        counters = [v for v in variables if v.var_type == VarType.COUNTER]
        flags = [v for v in variables if v.var_type == VarType.FLAG]

        choice = random.random()

        if choice < 0.5 and counters:
            var = random.choice(counters)
            return IncrementAction(var_name=var.name)

        if choice < 0.7 and counters:
            var = random.choice(counters)
            return ResetAction(var_name=var.name, value=0)

        if choice < 0.9 and flags:
            var = random.choice(flags)
            return SetFlagAction(var_name=var.name, value=random.choice([True, False]))

        return NoOpAction()

    def mutate(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Apply a mutation to an extended statechart."""
        sc = sc.copy()

        # Select mutation category
        r = random.random()
        cumulative = 0.0

        for category, prob in self._mutation_probs.items():
            cumulative += prob
            if r < cumulative:
                if category == 'topology':
                    sc = self._mutate_topology(sc)
                elif category == 'transition':
                    sc = self._mutate_transition(sc)
                elif category == 'guard':
                    sc = self._mutate_guard(sc)
                elif category == 'action':
                    sc = self._mutate_action(sc)
                elif category == 'variable':
                    sc = self._mutate_variable(sc)
                elif category == 'accept':
                    sc = self._mutate_accept_condition(sc)
                break

        return sc

    def _mutate_topology(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Mutate state topology."""
        mutation = random.choice(['add_state', 'remove_state', 'toggle_accept'])

        if mutation == 'add_state' and sc.n_states < self.config.max_states:
            new_idx = sc.n_states
            sc.state_labels.append(f"S{new_idx}")
            sc.state_types.append(StateType.INTERMEDIATE)

        elif mutation == 'remove_state' and sc.n_states > self.config.min_states:
            # Find removable state
            removable = []
            for i in range(1, sc.n_states):
                if sc.state_types[i] == StateType.ACCEPT:
                    if sum(1 for t in sc.state_types if t == StateType.ACCEPT) > 1:
                        removable.append(i)
                else:
                    removable.append(i)

            if removable:
                idx = random.choice(removable)
                sc.state_labels.pop(idx)
                sc.state_types.pop(idx)

                # Update transitions
                new_trans = []
                for t in sc.transitions:
                    if t.source == idx or t.target == idx:
                        continue
                    new_src = t.source if t.source < idx else t.source - 1
                    new_tgt = t.target if t.target < idx else t.target - 1
                    new_trans.append(SemanticTransition(
                        source=new_src, target=new_tgt, guard=t.guard, action=t.action
                    ))
                sc.transitions = new_trans

        elif mutation == 'toggle_accept':
            candidates = list(range(1, sc.n_states))
            if candidates:
                idx = random.choice(candidates)
                if sc.state_types[idx] == StateType.ACCEPT:
                    if sum(1 for t in sc.state_types if t == StateType.ACCEPT) > 1:
                        sc.state_types[idx] = StateType.INTERMEDIATE
                else:
                    sc.state_types[idx] = StateType.ACCEPT

        return sc

    def _mutate_transition(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Mutate transitions."""
        mutation = random.choice(['add', 'remove', 'change_target'])

        if mutation == 'add' and len(sc.transitions) < self.config.max_transitions:
            trans = self._random_transition(sc.n_states, sc.variables)
            sc.transitions.append(trans)

        elif mutation == 'remove' and sc.transitions:
            idx = random.randint(0, len(sc.transitions) - 1)
            sc.transitions.pop(idx)

        elif mutation == 'change_target' and sc.transitions:
            idx = random.randint(0, len(sc.transitions) - 1)
            t = sc.transitions[idx]
            new_tgt = random.randint(0, sc.n_states - 1)
            sc.transitions[idx] = SemanticTransition(
                source=t.source, target=new_tgt, guard=t.guard, action=t.action
            )

        return sc

    def _mutate_guard(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Mutate a transition guard."""
        if not sc.transitions:
            return sc

        idx = random.randint(0, len(sc.transitions) - 1)
        t = sc.transitions[idx]

        mutation = random.choice(['mutate', 'replace', 'wrap'])

        if mutation == 'mutate':
            new_guard = t.guard.mutate(self.config.alphabet)
        elif mutation == 'replace':
            new_guard = self._random_guard(sc.variables)
        else:  # wrap in composite
            other_guard = self._random_guard(sc.variables)
            new_guard = CompositeGuardExpr(
                left=t.guard,
                right=other_guard,
                operator=random.choice(['and', 'or'])
            )

        sc.transitions[idx] = SemanticTransition(
            source=t.source, target=t.target, guard=new_guard, action=t.action
        )

        return sc

    def _mutate_action(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Mutate a transition action."""
        if not sc.transitions:
            return sc

        idx = random.randint(0, len(sc.transitions) - 1)
        t = sc.transitions[idx]

        mutation = random.choice(['mutate', 'replace', 'add'])

        if mutation == 'mutate':
            new_action = t.action.mutate()
        elif mutation == 'replace':
            new_action = self._random_action(sc.variables)
        else:  # add to composite
            if isinstance(t.action, CompositeAction):
                actions = t.action.actions + [self._random_action(sc.variables)]
                new_action = CompositeAction(actions=actions)
            elif isinstance(t.action, NoOpAction):
                new_action = self._random_action(sc.variables)
            else:
                new_action = CompositeAction(actions=[t.action, self._random_action(sc.variables)])

        sc.transitions[idx] = SemanticTransition(
            source=t.source, target=t.target, guard=t.guard, action=new_action
        )

        return sc

    def _mutate_variable(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Add or remove a variable."""
        mutation = random.choice(['add', 'remove'])

        if mutation == 'add' and len(sc.variables) < self.config.max_variables:
            var_type = random.choice([VarType.COUNTER, VarType.FLAG])
            if var_type == VarType.COUNTER:
                name = random.choice(self.config.counter_names)
                sc.variables.append(StateVar(name=name, var_type=var_type, initial_value=0))
            else:
                name = random.choice(self.config.flag_names)
                sc.variables.append(StateVar(name=name, var_type=var_type, initial_value=False))

        elif mutation == 'remove' and sc.variables:
            idx = random.randint(0, len(sc.variables) - 1)
            sc.variables.pop(idx)

        return sc

    def _mutate_accept_condition(self, sc: ExtendedStatechart) -> ExtendedStatechart:
        """Mutate accept conditions."""
        accept_states = sc.accept_states
        if not accept_states:
            return sc

        state_idx = random.choice(accept_states)
        counters = [v for v in sc.variables if v.var_type == VarType.COUNTER]

        if counters:
            var = random.choice(counters)
            op = random.choice(['==', '>=', '<=', '>', '<'])
            val = random.randint(1, 5)
            sc.accept_conditions[state_idx] = AcceptCondition(
                var_name=var.name, operator=op, value=val
            )
        else:
            sc.accept_conditions[state_idx] = AcceptCondition(operator="any")

        return sc

    def crossover(self, parent1: ExtendedStatechart, parent2: ExtendedStatechart) -> ExtendedStatechart:
        """Crossover two extended statecharts."""
        # Take structure from one parent
        if random.random() < 0.5:
            child = parent1.copy()
            other = parent2
        else:
            child = parent2.copy()
            other = parent1

        # Maybe take transitions from other
        for t in other.transitions:
            if t.source < child.n_states and t.target < child.n_states:
                if random.random() < 0.3:
                    child.transitions.append(t)

        # Maybe take variables from other
        for var in other.variables:
            if var.name not in [v.name for v in child.variables]:
                if random.random() < 0.3:
                    child.variables.append(var)

        return child

    def evaluate_fitness(
        self,
        sc: ExtendedStatechart,
        positive: List[str],
        negative: List[str]
    ) -> Tuple[float, Dict]:
        """Evaluate fitness of an extended statechart."""
        if not sc.is_valid():
            return 0.0, {'f1': 0.0, 'accuracy': 0.0, 'valid': False}

        self.stats.evaluations += 1

        tp = fp = tn = fn = 0

        for s in positive:
            try:
                if sc.matches(s):
                    tp += 1
                else:
                    fn += 1
            except Exception:
                fn += 1

        for s in negative:
            try:
                if sc.matches(s):
                    fp += 1
                else:
                    tn += 1
            except Exception:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Penalties
        size_penalty = sc.n_states * self.config.size_penalty
        complexity_penalty = sc.complexity() * self.config.complexity_penalty

        fitness = f1 - size_penalty - complexity_penalty
        fitness = max(0.0, fitness)

        metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'n_states': sc.n_states,
            'n_transitions': sc.n_transitions,
            'n_variables': len(sc.variables),
            'complexity': sc.complexity(),
            'valid': True
        }

        return fitness, metrics

    def evolve(
        self,
        positive: List[str],
        negative: List[str],
        callback: Callable = None
    ) -> Tuple[ExtendedStatechart, SemanticEvolutionStats]:
        """
        Evolve an extended statechart to match examples.

        Returns (best_statechart, evolution_stats).
        """
        self.stats = SemanticEvolutionStats()
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print("SEMANTIC STATECHART EVOLUTION")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print(f"Positive: {len(positive)}, Negative: {len(negative)}")
            print("-" * 60)

        # Initialize population
        population = [self.create_random() for _ in range(self.config.population_size)]

        # Evaluate initial population
        fitness_list = []
        metrics_list = []
        for sc in population:
            fitness, metrics = self.evaluate_fitness(sc, positive, negative)
            sc.fitness = fitness
            fitness_list.append(fitness)
            metrics_list.append(metrics)

        # Track best
        best_idx = max(range(len(population)), key=lambda i: fitness_list[i])
        best_ever = population[best_idx].copy()
        best_ever_metrics = metrics_list[best_idx]

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Sort by fitness
            sorted_pairs = sorted(
                zip(population, fitness_list, metrics_list),
                key=lambda x: x[1],
                reverse=True
            )
            population = [p for p, _, _ in sorted_pairs]
            fitness_list = [f for _, f, _ in sorted_pairs]
            metrics_list = [m for _, _, m in sorted_pairs]

            # Update best
            if fitness_list[0] > best_ever.fitness:
                best_ever = population[0].copy()
                best_ever_metrics = metrics_list[0]

            # Logging
            if self.config.verbose and (gen % self.config.log_every == 0 or gen == self.config.n_generations - 1):
                avg_f1 = sum(m['f1'] for m in metrics_list) / len(metrics_list)
                print(
                    f"Gen {gen:3d} | "
                    f"Best F1: {metrics_list[0]['f1']:.4f} | "
                    f"Avg F1: {avg_f1:.4f} | "
                    f"States: {population[0].n_states}, Vars: {len(population[0].variables)}"
                )

            # Callback
            if callback:
                callback(gen, population, best_ever)

            # Early stopping
            if best_ever_metrics['f1'] >= self.config.target_f1:
                if self.config.verbose:
                    print(f"Target F1 reached at generation {gen}!")
                break

            # Selection and reproduction
            elite = [p.copy() for p in population[:self.config.elite_size]]
            new_pop = elite.copy()

            while len(new_pop) < self.config.population_size:
                # Tournament selection
                tournament = random.sample(
                    list(range(len(population))),
                    min(self.config.tournament_size, len(population))
                )
                parent1_idx = max(tournament, key=lambda i: fitness_list[i])
                parent1 = population[parent1_idx]

                if random.random() < self.config.crossover_rate:
                    tournament = random.sample(
                        list(range(len(population))),
                        min(self.config.tournament_size, len(population))
                    )
                    parent2_idx = max(tournament, key=lambda i: fitness_list[i])
                    parent2 = population[parent2_idx]
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < self.config.mutation_rate:
                    child = self.mutate(child)

                new_pop.append(child)

            population = new_pop

            # Evaluate
            fitness_list = []
            metrics_list = []
            for sc in population:
                fitness, metrics = self.evaluate_fitness(sc, positive, negative)
                sc.fitness = fitness
                fitness_list.append(fitness)
                metrics_list.append(metrics)

        # Final stats
        self.stats.generations = gen + 1
        self.stats.best_f1 = best_ever_metrics['f1']
        self.stats.best_accuracy = best_ever_metrics['accuracy']
        self.stats.avg_f1 = sum(m['f1'] for m in metrics_list) / len(metrics_list)
        self.stats.elapsed_time = time.time() - start_time

        if self.config.verbose:
            print("-" * 60)
            print(f"Evolution complete!")
            print(f"  Best F1: {self.stats.best_f1:.4f}")
            print(f"  States: {best_ever.n_states}, Variables: {len(best_ever.variables)}")
            print(f"  Time: {self.stats.elapsed_time:.2f}s")
            print("=" * 60)

        return best_ever, self.stats


def test_semantic_evolver():
    """Test semantic evolution."""
    print("=" * 60)
    print("SEMANTIC EVOLVER TESTS")
    print("=" * 60)

    # Test 1: Simple pattern (a+)
    print("\n1. Evolving for pattern 'a+':")
    positive = ["a", "aa", "aaa", "aaaa"]
    negative = ["", "b", "ab", "ba", "bb"]

    config = SemanticEvolutionConfig(
        population_size=30,
        n_generations=40,
        max_states=5,
        verbose=True,
        log_every=10
    )

    evolver = SemanticEvolver(config)
    best, stats = evolver.evolve(positive, negative)

    print(f"\nBest statechart:")
    print(best.to_string())

    # Test
    print("\nTest results:")
    for s in positive + negative:
        expected = s in positive
        actual = best.matches(s)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  '{s}' -> {actual} (expected {expected}) [{status}]")

    # Test 2: Counting pattern (need 2-4 a's)
    print("\n" + "=" * 60)
    print("2. Evolving for counting pattern (2-4 'a's):")
    positive = ["aa", "aaa", "aaaa"]
    negative = ["", "a", "aaaaa", "aaaaaa", "b", "ab"]

    best2, stats2 = evolver.evolve(positive, negative)

    print(f"\nBest statechart:")
    print(best2.to_string())

    print("\nTest results:")
    for s in positive + negative:
        expected = s in positive
        actual = best2.matches(s)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  '{s}' -> {actual} (expected {expected}) [{status}]")

    print("\n" + "=" * 60)
    print("Semantic evolver tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_evolver()
