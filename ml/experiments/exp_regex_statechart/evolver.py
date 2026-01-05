"""
Evolution framework for regex statecharts.

Evolves statecharts to match regex behavior using:
- State mutations: add/remove states, mark as accepting
- Transition mutations: add/remove, change guards, change targets
- Guard mutations: add/remove chars, widen/narrow classes

Fitness: F1 score on positive/negative examples + size penalty (Occam's razor)
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable
from enum import Enum

from .regex_statechart import (
    RegexStatechart, StateType, Transition, CharGuard
)
from .re2_oracle import RE2Oracle, RegexExamples


@dataclass
class EvolutionConfig:
    """Configuration for regex statechart evolution."""
    # Population
    population_size: int = 50
    elite_size: int = 5
    n_generations: int = 100

    # Genome limits
    max_states: int = 15
    min_states: int = 2
    max_transitions: int = 50

    # Mutation rates
    mutation_rate: float = 0.4
    crossover_rate: float = 0.6

    # Mutation operator weights (normalized internally)
    add_state_weight: float = 0.10
    remove_state_weight: float = 0.08
    toggle_accept_weight: float = 0.12
    add_transition_weight: float = 0.20
    remove_transition_weight: float = 0.10
    change_target_weight: float = 0.15
    mutate_guard_weight: float = 0.25

    # Fitness
    size_penalty: float = 0.01  # Penalty per state
    transition_penalty: float = 0.002  # Penalty per transition

    # Selection
    tournament_size: int = 3

    # Alphabet for guards
    alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789"

    # Logging
    log_every: int = 10
    verbose: bool = True

    # Early stopping
    target_f1: float = 0.99  # Stop if F1 reaches this


@dataclass
class EvolutionStats:
    """Statistics from evolution run."""
    generations: int = 0
    best_f1: float = 0.0
    best_accuracy: float = 0.0
    avg_f1: float = 0.0
    elapsed_time: float = 0.0
    evaluations: int = 0


class RegexEvolver:
    """
    Evolve regex statecharts from examples.

    Given positive (matching) and negative (non-matching) examples,
    evolves a statechart that correctly classifies all examples.
    """

    def __init__(self, config: EvolutionConfig = None):
        """Initialize evolver with configuration."""
        self.config = config or EvolutionConfig()
        self.stats = EvolutionStats()

        # Normalize mutation weights
        total_weight = (
            self.config.add_state_weight +
            self.config.remove_state_weight +
            self.config.toggle_accept_weight +
            self.config.add_transition_weight +
            self.config.remove_transition_weight +
            self.config.change_target_weight +
            self.config.mutate_guard_weight
        )
        self._mutation_probs = {
            'add_state': self.config.add_state_weight / total_weight,
            'remove_state': self.config.remove_state_weight / total_weight,
            'toggle_accept': self.config.toggle_accept_weight / total_weight,
            'add_transition': self.config.add_transition_weight / total_weight,
            'remove_transition': self.config.remove_transition_weight / total_weight,
            'change_target': self.config.change_target_weight / total_weight,
            'mutate_guard': self.config.mutate_guard_weight / total_weight,
        }

    def create_random(self, alphabet: str = None) -> RegexStatechart:
        """Create a random statechart."""
        alphabet = alphabet or self.config.alphabet

        n_states = random.randint(self.config.min_states, min(5, self.config.max_states))

        # Create states
        labels = ["START"] + [f"S{i}" for i in range(1, n_states - 1)] + ["ACCEPT"]
        types = [StateType.START]
        types += [StateType.INTERMEDIATE] * (n_states - 2)
        types.append(StateType.ACCEPT)

        # Create some random transitions
        transitions = []
        n_transitions = random.randint(n_states - 1, n_states * 2)

        for _ in range(n_transitions):
            src = random.randint(0, n_states - 1)
            tgt = random.randint(0, n_states - 1)
            guard = self._random_guard(alphabet)
            transitions.append(Transition(source=src, target=tgt, guard=guard))

        # Ensure at least one transition from START
        if not any(t.source == 0 for t in transitions):
            tgt = random.randint(1, n_states - 1)
            guard = self._random_guard(alphabet)
            transitions.append(Transition(source=0, target=tgt, guard=guard))

        sc = RegexStatechart(
            state_labels=labels,
            state_types=types,
            transitions=transitions,
            initial_state=0
        )

        return sc

    def _random_guard(self, alphabet: str) -> CharGuard:
        """Create a random character guard."""
        choice = random.random()

        if choice < 0.4:
            # Single character
            return CharGuard.single(random.choice(alphabet))
        elif choice < 0.7:
            # Small character class (2-4 chars)
            n_chars = random.randint(2, 4)
            chars = ''.join(random.sample(alphabet, min(n_chars, len(alphabet))))
            return CharGuard.char_class(chars)
        elif choice < 0.85:
            # Character range
            if 'a' in alphabet and 'z' in alphabet:
                start = random.choice('abc')
                end = random.choice('xyz')
                if start <= end:
                    return CharGuard.range(start, end)
            return CharGuard.single(random.choice(alphabet))
        else:
            # Any char
            return CharGuard.any_char()

    def mutate(self, sc: RegexStatechart, alphabet: str = None) -> RegexStatechart:
        """Apply a mutation to a statechart."""
        alphabet = alphabet or self.config.alphabet
        sc = sc.copy()

        # Select mutation operator
        r = random.random()
        cumulative = 0.0

        for op, prob in self._mutation_probs.items():
            cumulative += prob
            if r < cumulative:
                if op == 'add_state':
                    sc = self._mutate_add_state(sc)
                elif op == 'remove_state':
                    sc = self._mutate_remove_state(sc)
                elif op == 'toggle_accept':
                    sc = self._mutate_toggle_accept(sc)
                elif op == 'add_transition':
                    sc = self._mutate_add_transition(sc, alphabet)
                elif op == 'remove_transition':
                    sc = self._mutate_remove_transition(sc)
                elif op == 'change_target':
                    sc = self._mutate_change_target(sc)
                elif op == 'mutate_guard':
                    sc = self._mutate_guard(sc, alphabet)
                break

        return sc

    def _mutate_add_state(self, sc: RegexStatechart) -> RegexStatechart:
        """Add a new intermediate state."""
        if sc.n_states >= self.config.max_states:
            return sc

        new_idx = sc.n_states
        sc.state_labels.append(f"S{new_idx}")
        sc.state_types.append(StateType.INTERMEDIATE)

        return sc

    def _mutate_remove_state(self, sc: RegexStatechart) -> RegexStatechart:
        """Remove a state (not START or last ACCEPT)."""
        if sc.n_states <= self.config.min_states:
            return sc

        # Find removable states (not START, not the only ACCEPT)
        removable = []
        for i in range(1, sc.n_states):
            if sc.state_types[i] == StateType.ACCEPT:
                # Can only remove if there's another ACCEPT
                if sum(1 for t in sc.state_types if t == StateType.ACCEPT) > 1:
                    removable.append(i)
            else:
                removable.append(i)

        if not removable:
            return sc

        idx = random.choice(removable)

        # Remove state
        sc.state_labels.pop(idx)
        sc.state_types.pop(idx)

        # Update transitions (remove references, adjust indices)
        new_transitions = []
        for t in sc.transitions:
            if t.source == idx or t.target == idx:
                continue
            new_src = t.source if t.source < idx else t.source - 1
            new_tgt = t.target if t.target < idx else t.target - 1
            new_transitions.append(Transition(source=new_src, target=new_tgt, guard=t.guard))

        sc.transitions = new_transitions

        return sc

    def _mutate_toggle_accept(self, sc: RegexStatechart) -> RegexStatechart:
        """Toggle a state's accepting status."""
        # Don't toggle START
        candidates = list(range(1, sc.n_states))
        if not candidates:
            return sc

        idx = random.choice(candidates)

        if sc.state_types[idx] == StateType.ACCEPT:
            # Can only un-accept if there's another ACCEPT
            if sum(1 for t in sc.state_types if t == StateType.ACCEPT) > 1:
                sc.state_types[idx] = StateType.INTERMEDIATE
        else:
            sc.state_types[idx] = StateType.ACCEPT

        return sc

    def _mutate_add_transition(self, sc: RegexStatechart, alphabet: str) -> RegexStatechart:
        """Add a new transition."""
        if len(sc.transitions) >= self.config.max_transitions:
            return sc

        src = random.randint(0, sc.n_states - 1)
        tgt = random.randint(0, sc.n_states - 1)
        guard = self._random_guard(alphabet)

        sc.transitions.append(Transition(source=src, target=tgt, guard=guard))
        return sc

    def _mutate_remove_transition(self, sc: RegexStatechart) -> RegexStatechart:
        """Remove a transition."""
        if not sc.transitions:
            return sc

        idx = random.randint(0, len(sc.transitions) - 1)
        sc.transitions.pop(idx)
        return sc

    def _mutate_change_target(self, sc: RegexStatechart) -> RegexStatechart:
        """Change the target of a transition."""
        if not sc.transitions:
            return sc

        idx = random.randint(0, len(sc.transitions) - 1)
        t = sc.transitions[idx]
        new_tgt = random.randint(0, sc.n_states - 1)

        sc.transitions[idx] = Transition(source=t.source, target=new_tgt, guard=t.guard)
        return sc

    def _mutate_guard(self, sc: RegexStatechart, alphabet: str) -> RegexStatechart:
        """Mutate a transition's guard."""
        if not sc.transitions:
            return sc

        idx = random.randint(0, len(sc.transitions) - 1)
        t = sc.transitions[idx]
        old_guard = t.guard

        # Different guard mutations
        mutation = random.choice(['add_char', 'remove_char', 'replace', 'toggle_any', 'toggle_negate'])

        if mutation == 'add_char' and not old_guard.is_any:
            # Add a character to the class
            new_char = random.choice(alphabet)
            new_chars = old_guard.chars | {new_char}
            new_guard = CharGuard(chars=new_chars, is_negated=old_guard.is_negated)
        elif mutation == 'remove_char' and len(old_guard.chars) > 1:
            # Remove a character from the class
            chars_list = list(old_guard.chars)
            chars_list.remove(random.choice(chars_list))
            new_guard = CharGuard(chars=frozenset(chars_list), is_negated=old_guard.is_negated)
        elif mutation == 'replace':
            # Replace with new random guard
            new_guard = self._random_guard(alphabet)
        elif mutation == 'toggle_any':
            # Toggle between any and specific
            if old_guard.is_any:
                new_guard = self._random_guard(alphabet)
            else:
                new_guard = CharGuard.any_char()
        elif mutation == 'toggle_negate':
            # Toggle negation
            new_guard = CharGuard(chars=old_guard.chars, is_negated=not old_guard.is_negated)
        else:
            new_guard = old_guard

        sc.transitions[idx] = Transition(source=t.source, target=t.target, guard=new_guard)
        return sc

    def crossover(self, parent1: RegexStatechart, parent2: RegexStatechart) -> RegexStatechart:
        """Crossover two statecharts."""
        # Simple crossover: take states from one parent, transitions from both
        if random.random() < 0.5:
            child = parent1.copy()
            # Add some transitions from parent2
            for t in parent2.transitions:
                if t.source < child.n_states and t.target < child.n_states:
                    if random.random() < 0.3:
                        child.transitions.append(t)
        else:
            child = parent2.copy()
            for t in parent1.transitions:
                if t.source < child.n_states and t.target < child.n_states:
                    if random.random() < 0.3:
                        child.transitions.append(t)

        return child

    def evaluate_fitness(
        self,
        sc: RegexStatechart,
        positive: List[str],
        negative: List[str]
    ) -> Tuple[float, Dict]:
        """
        Evaluate fitness of a statechart.

        Returns (fitness, metrics_dict).
        """
        if not sc.is_valid():
            return 0.0, {'f1': 0.0, 'accuracy': 0.0, 'valid': False}

        self.stats.evaluations += 1

        tp = fp = tn = fn = 0

        for s in positive:
            if sc.matches(s):
                tp += 1
            else:
                fn += 1

        for s in negative:
            if sc.matches(s):
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Size penalty (Occam's razor)
        size_penalty = (
            sc.n_states * self.config.size_penalty +
            sc.n_transitions * self.config.transition_penalty
        )

        fitness = f1 - size_penalty
        fitness = max(0.0, fitness)

        metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'n_states': sc.n_states,
            'n_transitions': sc.n_transitions,
            'valid': True
        }

        return fitness, metrics

    def evolve(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: str = None,
        callback: Callable = None
    ) -> Tuple[RegexStatechart, EvolutionStats]:
        """
        Evolve a statechart to match the given examples.

        Args:
            positive: Strings that should match
            negative: Strings that should not match
            alphabet: Characters to use in guards
            callback: Called each generation with (gen, population, best)

        Returns:
            (best_statechart, evolution_stats)
        """
        alphabet = alphabet or self._infer_alphabet(positive + negative)
        self.stats = EvolutionStats()
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print("REGEX STATECHART EVOLUTION")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print(f"Positive examples: {len(positive)}")
            print(f"Negative examples: {len(negative)}")
            print(f"Alphabet: {alphabet[:20]}{'...' if len(alphabet) > 20 else ''}")
            print("-" * 60)

        # Initialize population
        population = [self.create_random(alphabet) for _ in range(self.config.population_size)]

        # Evaluate initial population
        fitness_list = []
        metrics_list = []
        for sc in population:
            fitness, metrics = self.evaluate_fitness(sc, positive, negative)
            sc.fitness = fitness
            fitness_list.append(fitness)
            metrics_list.append(metrics)

        # Track best ever
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

            # Update best ever
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
                    f"States: {population[0].n_states}"
                )

            # Callback
            if callback:
                callback(gen, population, best_ever)

            # Early stopping
            if best_ever_metrics['f1'] >= self.config.target_f1:
                if self.config.verbose:
                    print(f"Target F1 reached at generation {gen}!")
                break

            # Elitism
            elite = [p.copy() for p in population[:self.config.elite_size]]

            # Selection and reproduction
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
                    child = self.mutate(child, alphabet)

                new_pop.append(child)

            population = new_pop

            # Evaluate new population
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
            print(f"  Accuracy: {self.stats.best_accuracy:.4f}")
            print(f"  States: {best_ever.n_states}, Transitions: {best_ever.n_transitions}")
            print(f"  Time: {self.stats.elapsed_time:.2f}s")
            print(f"  Evaluations: {self.stats.evaluations}")
            print("=" * 60)

        return best_ever, self.stats

    def _infer_alphabet(self, strings: List[str]) -> str:
        """Infer alphabet from example strings."""
        chars = set()
        for s in strings:
            chars.update(s)

        # Add some extras for mutations
        if any(c.islower() for c in chars):
            chars.update('abc')
        if any(c.isdigit() for c in chars):
            chars.update('012')

        return ''.join(sorted(chars))


def test_evolver():
    """Test the regex evolver."""
    print("=" * 60)
    print("REGEX EVOLVER TESTS")
    print("=" * 60)

    # Test 1: Simple pattern a+
    print("\n1. Evolving statechart for pattern 'a+':")
    oracle = RE2Oracle("a+")
    examples = oracle.generate_examples(n_positive=30, n_negative=30)

    config = EvolutionConfig(
        population_size=30,
        n_generations=50,
        max_states=5,
        verbose=True,
        log_every=10
    )

    evolver = RegexEvolver(config)
    best, stats = evolver.evolve(examples.positive, examples.negative)

    print(f"\nBest statechart:")
    print(best.to_string())

    # Validate against oracle
    score = oracle.score_statechart(best)
    print(f"\nValidation against oracle:")
    print(f"  F1: {score['f1']:.4f}")

    # Test 2: More complex pattern ab+
    print("\n" + "=" * 60)
    print("2. Evolving statechart for pattern 'ab+':")
    oracle2 = RE2Oracle("ab+")
    examples2 = oracle2.generate_examples(n_positive=40, n_negative=40)

    best2, stats2 = evolver.evolve(examples2.positive, examples2.negative)

    print(f"\nBest statechart:")
    print(best2.to_string())

    score2 = oracle2.score_statechart(best2)
    print(f"\nValidation: F1 = {score2['f1']:.4f}")

    print("\n" + "=" * 60)
    print("Evolver tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_evolver()
