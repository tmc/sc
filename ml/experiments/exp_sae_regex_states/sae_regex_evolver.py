"""
SAE Regex Evolver.

Evolves statecharts using SAE-discovered states.

Key difference from exp_regex_statechart:
- States come from SAE features (learned)
- Transitions are evolved to match SAE dynamics
- Accept conditions evolved based on feature patterns

The evolution focuses on:
1. Transition refinement (which state goes where on which char)
2. Accept condition tuning (which feature combinations accept)
3. Guard synthesis (for extended patterns like counting)
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
from collections import defaultdict

from .sae_regex import CharSequenceSAE, SAEConfig, train_char_sae
from .feature_to_state import (
    FeatureStateMapper, SAEState, extract_states_from_sae
)


@dataclass
class SAEEvolutionConfig:
    """Configuration for SAE-based evolution."""
    # SAE config
    n_features: int = 32
    k_active: int = 4
    sae_epochs: int = 50

    # Evolution config
    population_size: int = 30
    n_generations: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5

    # Fitness
    f1_weight: float = 1.0
    complexity_penalty: float = 0.01

    # Alphabet
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class SAETransition:
    """A transition in the SAE-based statechart."""
    source_state: int       # State ID
    char: str               # Character trigger
    target_state: int       # Target state ID
    priority: int = 0       # For conflict resolution


@dataclass
class SAEStatechart:
    """
    A statechart built from SAE states.

    States are defined by SAE feature activations.
    Transitions are evolved to match observed dynamics.
    """
    # Core components
    states: Dict[int, SAEState]           # state_id -> state
    transitions: List[SAETransition]       # All transitions
    accepting_states: Set[int]             # Set of accepting state IDs
    initial_state: int                     # Initial state ID

    # SAE reference (for state identification)
    sae: Optional[CharSequenceSAE] = None
    mapper: Optional[FeatureStateMapper] = None

    # Fitness
    fitness: float = 0.0

    def get_next_state(self, current: int, char: str) -> Optional[int]:
        """Get next state given current state and input character."""
        for t in self.transitions:
            if t.source_state == current and t.char == char:
                return t.target_state
        return None

    def matches(self, s: str) -> bool:
        """Check if string matches (ends in accepting state)."""
        current = self.initial_state

        for char in s:
            next_state = self.get_next_state(current, char)
            if next_state is None:
                return False
            current = next_state

        return current in self.accepting_states

    def matches_via_sae(self, s: str) -> bool:
        """
        Check if string matches using SAE state identification.

        This uses the SAE directly to identify states, rather than
        following evolved transitions.
        """
        if self.sae is None or self.mapper is None:
            return self.matches(s)

        # Get final state from SAE
        features = self.sae.get_state_for_prefix(s)
        state = self.mapper.get_or_create_state(features)

        return state.is_accepting

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def to_string(self) -> str:
        """String representation."""
        lines = [
            f"SAEStatechart:",
            f"  States: {self.n_states}",
            f"  Transitions: {self.n_transitions}",
            f"  Accepting: {self.accepting_states}",
            f"  Initial: {self.initial_state}",
        ]

        lines.append("  Transitions:")
        for t in sorted(self.transitions, key=lambda x: (x.source_state, x.char)):
            acc_mark = "*" if t.target_state in self.accepting_states else ""
            lines.append(f"    {t.source_state} --'{t.char}'--> {t.target_state}{acc_mark}")

        return "\n".join(lines)


class SAERegexEvolver:
    """
    Evolves statecharts using SAE-discovered states.

    Evolution process:
    1. Train SAE on all strings
    2. Extract states from SAE features
    3. Initialize transitions from observed dynamics
    4. Evolve transitions and accept conditions
    """

    def __init__(self, config: SAEEvolutionConfig = None):
        self.config = config or SAEEvolutionConfig()
        self.sae: Optional[CharSequenceSAE] = None
        self.mapper: Optional[FeatureStateMapper] = None
        self.population: List[SAEStatechart] = []

    def train_sae(
        self,
        positive: List[str],
        negative: List[str],
        verbose: bool = False
    ) -> CharSequenceSAE:
        """Train SAE on all examples."""
        sae_config = SAEConfig(
            n_features=self.config.n_features,
            k_active=self.config.k_active,
            n_epochs=self.config.sae_epochs,
            alphabet=self.config.alphabet
        )

        all_strings = positive + negative
        self.sae = train_char_sae(all_strings, sae_config, verbose=verbose)
        return self.sae

    def extract_states(
        self,
        positive: List[str],
        negative: List[str],
        verbose: bool = False
    ) -> Tuple[List[SAEState], Set[int]]:
        """Extract states from trained SAE."""
        if self.sae is None:
            raise ValueError("Must train SAE first")

        self.mapper, states, accepting = extract_states_from_sae(
            self.sae, positive, negative, verbose=verbose
        )

        return states, accepting

    def _create_initial_statechart(
        self,
        states: List[SAEState],
        accepting: Set[int],
        positive: List[str],
        negative: List[str]
    ) -> SAEStatechart:
        """Create initial statechart from SAE states and observed transitions."""
        # Get observed transitions
        observed = self.mapper.get_state_transitions(positive + negative)

        transitions = []
        for (src, char), tgt in observed.items():
            transitions.append(SAETransition(
                source_state=src,
                char=char,
                target_state=tgt
            ))

        # Find initial state
        initial = None
        for state in states:
            if state.is_initial:
                initial = state.id
                break

        if initial is None:
            initial = states[0].id if states else 0

        return SAEStatechart(
            states={s.id: s for s in states},
            transitions=transitions,
            accepting_states=accepting,
            initial_state=initial,
            sae=self.sae,
            mapper=self.mapper
        )

    def _mutate(self, sc: SAEStatechart) -> SAEStatechart:
        """Mutate a statechart."""
        # Deep copy
        new_states = {k: SAEState(
            id=v.id,
            active_features=v.active_features,
            label=v.label,
            is_accepting=v.is_accepting,
            is_initial=v.is_initial
        ) for k, v in sc.states.items()}

        new_transitions = [SAETransition(
            source_state=t.source_state,
            char=t.char,
            target_state=t.target_state,
            priority=t.priority
        ) for t in sc.transitions]

        new_accepting = sc.accepting_states.copy()

        # Apply mutations
        if random.random() < self.config.mutation_rate:
            mutation_type = random.choice([
                'change_target', 'add_transition', 'remove_transition',
                'toggle_accepting'
            ])

            state_ids = list(new_states.keys())

            if mutation_type == 'change_target' and new_transitions:
                # Change a transition's target
                idx = random.randrange(len(new_transitions))
                new_transitions[idx].target_state = random.choice(state_ids)

            elif mutation_type == 'add_transition' and state_ids:
                # Add a new transition
                src = random.choice(state_ids)
                tgt = random.choice(state_ids)
                char = random.choice(self.config.alphabet)
                new_transitions.append(SAETransition(
                    source_state=src,
                    char=char,
                    target_state=tgt
                ))

            elif mutation_type == 'remove_transition' and len(new_transitions) > 1:
                # Remove a transition
                idx = random.randrange(len(new_transitions))
                new_transitions.pop(idx)

            elif mutation_type == 'toggle_accepting' and state_ids:
                # Toggle a state's accepting status
                state_id = random.choice(state_ids)
                if state_id in new_accepting:
                    new_accepting.remove(state_id)
                else:
                    new_accepting.add(state_id)

        return SAEStatechart(
            states=new_states,
            transitions=new_transitions,
            accepting_states=new_accepting,
            initial_state=sc.initial_state,
            sae=sc.sae,
            mapper=sc.mapper
        )

    def _crossover(self, sc1: SAEStatechart, sc2: SAEStatechart) -> SAEStatechart:
        """Crossover two statecharts."""
        # Take states from first parent
        new_states = {k: SAEState(
            id=v.id,
            active_features=v.active_features,
            label=v.label,
            is_accepting=v.is_accepting,
            is_initial=v.is_initial
        ) for k, v in sc1.states.items()}

        # Mix transitions from both parents
        all_transitions = sc1.transitions + sc2.transitions
        n_keep = max(1, len(all_transitions) // 2)
        new_transitions = random.sample(all_transitions, min(n_keep, len(all_transitions)))

        # Copy transitions to avoid mutation issues
        new_transitions = [SAETransition(
            source_state=t.source_state,
            char=t.char,
            target_state=t.target_state,
            priority=t.priority
        ) for t in new_transitions]

        # Mix accepting states
        new_accepting = sc1.accepting_states.copy()
        for state_id in sc2.accepting_states:
            if random.random() < 0.5:
                new_accepting.add(state_id)

        return SAEStatechart(
            states=new_states,
            transitions=new_transitions,
            accepting_states=new_accepting,
            initial_state=sc1.initial_state,
            sae=sc1.sae,
            mapper=sc1.mapper
        )

    def _evaluate(
        self,
        sc: SAEStatechart,
        positive: List[str],
        negative: List[str]
    ) -> float:
        """Evaluate statechart fitness."""
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

        # F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Complexity penalty
        complexity = (sc.n_states + sc.n_transitions) * self.config.complexity_penalty

        fitness = self.config.f1_weight * f1 - complexity
        sc.fitness = fitness

        return fitness

    def evolve(
        self,
        positive: List[str],
        negative: List[str],
        verbose: bool = False
    ) -> Tuple[SAEStatechart, Dict[str, float]]:
        """
        Evolve a statechart for the given examples.

        Returns (best_statechart, stats)
        """
        # Step 1: Train SAE
        if verbose:
            print("Training SAE...")
        self.train_sae(positive, negative, verbose=verbose)

        # Step 2: Extract states
        if verbose:
            print("Extracting states...")
        states, accepting = self.extract_states(positive, negative, verbose=verbose)

        if not states:
            raise ValueError("No states extracted from SAE")

        # Step 3: Initialize population
        base = self._create_initial_statechart(states, accepting, positive, negative)

        self.population = [base]
        for _ in range(self.config.population_size - 1):
            mutant = self._mutate(base)
            self.population.append(mutant)

        # Step 4: Evolution loop
        best_fitness = float('-inf')
        best_statechart = base
        stagnation = 0

        for gen in range(self.config.n_generations):
            # Evaluate population
            for sc in self.population:
                self._evaluate(sc, positive, negative)

            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            if self.population[0].fitness > best_fitness:
                best_fitness = self.population[0].fitness
                best_statechart = self.population[0]
                stagnation = 0
            else:
                stagnation += 1

            if verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen + 1}: best_fitness = {best_fitness:.4f}, "
                      f"states = {best_statechart.n_states}")

            # Early stopping
            if best_fitness >= 0.99:
                break

            # Selection and reproduction
            n_elite = max(2, self.config.population_size // 5)
            new_pop = self.population[:n_elite]  # Elitism

            while len(new_pop) < self.config.population_size:
                # Tournament selection
                candidates = random.sample(self.population, min(3, len(self.population)))
                parent1 = max(candidates, key=lambda x: x.fitness)

                candidates = random.sample(self.population, min(3, len(self.population)))
                parent2 = max(candidates, key=lambda x: x.fitness)

                # Crossover or mutation
                if random.random() < self.config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._mutate(parent1)

                new_pop.append(child)

            self.population = new_pop

        # Final evaluation
        self._evaluate(best_statechart, positive, negative)

        # Compute final stats
        tp = sum(1 for s in positive if best_statechart.matches(s))
        tn = sum(1 for s in negative if not best_statechart.matches(s))
        precision = tp / (tp + (len(negative) - tn)) if (tp + len(negative) - tn) > 0 else 0
        recall = tp / len(positive) if positive else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': (tp + tn) / (len(positive) + len(negative)),
            'n_states': best_statechart.n_states,
            'n_transitions': best_statechart.n_transitions,
            'n_generations': gen + 1,
        }

        return best_statechart, stats


def evolve_sae_regex(
    positive: List[str],
    negative: List[str],
    config: SAEEvolutionConfig = None,
    verbose: bool = False
) -> Tuple[SAEStatechart, Dict[str, float]]:
    """
    Evolve a regex statechart using SAE states.

    Convenience function.
    """
    evolver = SAERegexEvolver(config)
    return evolver.evolve(positive, negative, verbose=verbose)


def test_sae_evolver():
    """Test SAE regex evolution."""
    print("=" * 60)
    print("Testing SAE Regex Evolver")
    print("=" * 60)

    # Test pattern: a+b (one or more 'a' followed by 'b')
    positive = ["ab", "aab", "aaab", "aaaab"]
    negative = ["", "a", "b", "ba", "bb", "baa", "aba"]

    print("\n1. Pattern: a+b")
    print(f"   Positive: {positive}")
    print(f"   Negative: {negative}")

    # Evolve
    config = SAEEvolutionConfig(
        n_features=16,
        k_active=3,
        population_size=20,
        n_generations=30
    )

    print("\n2. Evolving statechart...")
    sc, stats = evolve_sae_regex(positive, negative, config, verbose=True)

    print("\n3. Results:")
    print(f"   F1: {stats['f1']:.4f}")
    print(f"   Accuracy: {stats['accuracy']:.4f}")
    print(f"   States: {stats['n_states']}")
    print(f"   Transitions: {stats['n_transitions']}")

    print("\n4. Statechart:")
    print(sc.to_string())

    print("\n5. Test matches:")
    all_examples = positive + negative
    for s in all_examples:
        result = sc.matches(s)
        expected = s in positive
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}': {result} (expected {expected}) [{status}]")

    # Test pattern: (ab)+
    print("\n" + "-" * 60)
    print("6. Pattern: (ab)+")

    positive2 = ["ab", "abab", "ababab"]
    negative2 = ["", "a", "b", "aba", "abba", "ba"]

    sc2, stats2 = evolve_sae_regex(positive2, negative2, config, verbose=False)

    print(f"   F1: {stats2['f1']:.4f}")
    print(f"   States: {stats2['n_states']}")

    for s in positive2 + negative2[:3]:
        result = sc2.matches(s)
        expected = s in positive2
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}': {result} (expected {expected}) [{status}]")

    print("\n" + "=" * 60)
    print("SAE Regex Evolver tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_sae_evolver()
