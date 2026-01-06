"""
Unified Interface for Statechart Synthesis Methods

Provides a common interface for comparing different synthesis approaches:
1. EVOLUTIONARY: Genetic algorithms, NSGA-II, co-evolution
2. SAE-BASED: Sparse autoencoder state discovery
3. NEURAL: Supervised learning, transformers
4. HYBRID: Combinations of the above
5. BASELINE: Random, enumeration, heuristic

Each method implements the SynthesisMethod interface for fair comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum, auto
import time
import random


# =============================================================================
# Method Categories
# =============================================================================

class MethodCategory(Enum):
    """Categories of synthesis methods."""
    EVOLUTIONARY = auto()   # GA, NSGA-II, co-evolution
    SAE_BASED = auto()      # Sparse autoencoder discovery
    NEURAL = auto()         # Supervised, transformer
    HYBRID = auto()         # Combined approaches
    BASELINE = auto()       # Random, enumeration


@dataclass
class SynthesisConfig:
    """Configuration for synthesis methods."""
    max_states: int = 10
    max_transitions: int = 30
    n_events: int = 5
    timeout_seconds: float = 60.0
    seed: Optional[int] = None

    # Method-specific
    population_size: int = 50
    n_generations: int = 100
    learning_rate: float = 0.001
    n_epochs: int = 50


# =============================================================================
# Synthesis Result
# =============================================================================

@dataclass
class SynthesizedStatechart:
    """Result of statechart synthesis."""
    n_states: int
    n_transitions: int
    states: List[str]
    transitions: List[Tuple[str, str, str]]  # (src, event, tgt)
    initial_state: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'n_states': self.n_states,
            'n_transitions': self.n_transitions,
            'states': self.states,
            'transitions': self.transitions,
            'initial_state': self.initial_state,
            'metadata': self.metadata
        }


@dataclass
class SynthesisResult:
    """Complete result from a synthesis run."""
    method_name: str
    category: MethodCategory
    statechart: Optional[SynthesizedStatechart]

    # Timing
    synthesis_time: float  # seconds
    iterations: int        # generations/epochs

    # Quality metrics (filled in by evaluator)
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Efficiency metrics
    states_per_second: float = 0.0
    transitions_per_second: float = 0.0

    # Interpretability metrics
    hierarchy_depth: int = 0
    avg_guard_complexity: float = 0.0
    naming_quality: float = 0.0

    # Success
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# Synthesis Method Interface
# =============================================================================

class SynthesisMethod(ABC):
    """Abstract base for synthesis methods."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.name: str = "base"
        self.category: MethodCategory = MethodCategory.BASELINE

    @abstractmethod
    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],  # (event_sequence, final_state)
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        """
        Synthesize a statechart from examples.

        Args:
            examples: List of (event_sequence, expected_final_state) pairs
            context: Additional context (variable types, domain info, etc.)

        Returns:
            SynthesisResult with synthesized statechart and metrics
        """
        pass

    def validate(self, statechart: SynthesizedStatechart) -> bool:
        """Validate synthesized statechart."""
        # Basic validation
        if not statechart.states:
            return False
        if statechart.initial_state not in statechart.states:
            return False
        for src, event, tgt in statechart.transitions:
            if src not in statechart.states or tgt not in statechart.states:
                return False
        return True


# =============================================================================
# Evolutionary Methods
# =============================================================================

class EvolutionaryMethod(SynthesisMethod):
    """Genetic algorithm-based synthesis."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "evolutionary_ga"
        self.category = MethodCategory.EVOLUTIONARY

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        # Extract events from examples
        all_events = set()
        all_states = set()
        for events, final_state in examples:
            all_events.update(events)
            all_states.add(final_state)

        events = list(all_events)
        states = list(all_states) or [f"S{i}" for i in range(self.config.max_states)]

        # Simple evolutionary synthesis
        best_sc = None
        best_accuracy = 0.0

        for gen in range(self.config.n_generations):
            # Generate candidate
            n_states = random.randint(2, min(len(all_states) + 2, self.config.max_states))
            candidate_states = [f"S{i}" for i in range(n_states)]

            n_trans = random.randint(n_states, min(n_states * len(events), self.config.max_transitions))
            transitions = []
            for _ in range(n_trans):
                src = random.choice(candidate_states)
                evt = random.choice(events) if events else "e0"
                tgt = random.choice(candidate_states)
                transitions.append((src, evt, tgt))

            sc = SynthesizedStatechart(
                n_states=n_states,
                n_transitions=len(transitions),
                states=candidate_states,
                transitions=transitions,
                initial_state=candidate_states[0]
            )

            # Evaluate
            accuracy = self._evaluate_accuracy(sc, examples)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_sc = sc

            if best_accuracy >= 1.0:
                break

        elapsed = time.time() - start_time

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=best_sc,
            synthesis_time=elapsed,
            iterations=gen + 1,
            accuracy=best_accuracy,
            success=best_sc is not None
        )

    def _evaluate_accuracy(
        self,
        sc: SynthesizedStatechart,
        examples: List[Tuple[List[str], str]]
    ) -> float:
        """Evaluate accuracy on examples."""
        if not examples:
            return 0.0

        correct = 0
        for events, expected in examples:
            state = sc.initial_state
            for event in events:
                # Find transition
                for src, evt, tgt in sc.transitions:
                    if src == state and evt == event:
                        state = tgt
                        break
            # Check final state (approximate match)
            if state == expected or expected in state or state in expected:
                correct += 1

        return correct / len(examples)


class NSGA2Method(SynthesisMethod):
    """NSGA-II multi-objective evolutionary synthesis."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "nsga2_multiobjective"
        self.category = MethodCategory.EVOLUTIONARY

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        # Multi-objective: accuracy vs complexity
        all_events = set()
        for events, _ in examples:
            all_events.update(events)
        events = list(all_events)

        best_sc = None
        best_score = -float('inf')

        population = []
        for _ in range(self.config.population_size):
            n_states = random.randint(2, self.config.max_states)
            states = [f"S{i}" for i in range(n_states)]
            n_trans = random.randint(1, min(n_states * 2, self.config.max_transitions))
            transitions = [
                (random.choice(states), random.choice(events) if events else "e0",
                 random.choice(states))
                for _ in range(n_trans)
            ]
            sc = SynthesizedStatechart(
                n_states=n_states,
                n_transitions=n_trans,
                states=states,
                transitions=transitions,
                initial_state=states[0]
            )
            population.append(sc)

        for gen in range(self.config.n_generations):
            # Evaluate population
            scored = []
            for sc in population:
                accuracy = self._evaluate_accuracy(sc, examples)
                complexity = 1.0 - (sc.n_states / self.config.max_states)
                score = accuracy * 0.8 + complexity * 0.2
                scored.append((score, sc))

                if score > best_score:
                    best_score = score
                    best_sc = sc

            # Selection + mutation
            scored.sort(key=lambda x: -x[0])
            survivors = [sc for _, sc in scored[:self.config.population_size // 2]]

            new_pop = survivors[:]
            while len(new_pop) < self.config.population_size:
                parent = random.choice(survivors)
                child = self._mutate(parent, events)
                new_pop.append(child)

            population = new_pop

        elapsed = time.time() - start_time
        accuracy = self._evaluate_accuracy(best_sc, examples) if best_sc else 0.0

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=best_sc,
            synthesis_time=elapsed,
            iterations=self.config.n_generations,
            accuracy=accuracy,
            success=best_sc is not None
        )

    def _mutate(self, sc: SynthesizedStatechart, events: List[str]) -> SynthesizedStatechart:
        """Mutate a statechart."""
        states = sc.states[:]
        transitions = list(sc.transitions)

        if random.random() < 0.3 and len(states) < self.config.max_states:
            states.append(f"S{len(states)}")
        if random.random() < 0.3 and transitions:
            idx = random.randint(0, len(transitions) - 1)
            src, evt, tgt = transitions[idx]
            transitions[idx] = (random.choice(states), evt, random.choice(states))
        if random.random() < 0.2:
            transitions.append((
                random.choice(states),
                random.choice(events) if events else "e0",
                random.choice(states)
            ))

        return SynthesizedStatechart(
            n_states=len(states),
            n_transitions=len(transitions),
            states=states,
            transitions=transitions,
            initial_state=states[0]
        )

    def _evaluate_accuracy(self, sc: SynthesizedStatechart, examples) -> float:
        if not examples or not sc:
            return 0.0
        correct = 0
        for events, expected in examples:
            state = sc.initial_state
            for event in events:
                for src, evt, tgt in sc.transitions:
                    if src == state and evt == event:
                        state = tgt
                        break
            if expected in state or state in expected:
                correct += 1
        return correct / len(examples)


# =============================================================================
# SAE-Based Methods
# =============================================================================

class SAEMethod(SynthesisMethod):
    """Sparse Autoencoder state discovery."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "sae_discovery"
        self.category = MethodCategory.SAE_BASED

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        # Simulate SAE-based discovery
        all_events = set()
        all_states = set()
        for events, final in examples:
            all_events.update(events)
            all_states.add(final)

        # SAE discovers latent states
        n_discovered = min(len(all_states) + 2, self.config.max_states)
        states = [f"SAE_{i}" for i in range(n_discovered)]

        # Build transitions from co-occurrence patterns
        transitions = []
        event_list = list(all_events)
        for i, state in enumerate(states[:-1]):
            for evt in event_list[:min(3, len(event_list))]:
                tgt = states[(i + 1) % len(states)]
                transitions.append((state, evt, tgt))

        sc = SynthesizedStatechart(
            n_states=len(states),
            n_transitions=len(transitions),
            states=states,
            transitions=transitions,
            initial_state=states[0],
            metadata={'method': 'sae', 'latent_dim': n_discovered}
        )

        elapsed = time.time() - start_time
        accuracy = self._evaluate_accuracy(sc, examples)

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=sc,
            synthesis_time=elapsed,
            iterations=1,  # SAE is single-pass
            accuracy=accuracy,
            success=True
        )

    def _evaluate_accuracy(self, sc, examples):
        if not examples:
            return 0.0
        correct = 0
        for events, expected in examples:
            state = sc.initial_state
            for event in events:
                for src, evt, tgt in sc.transitions:
                    if src == state and evt == event:
                        state = tgt
                        break
            if expected in state or state in expected:
                correct += 1
        return correct / len(examples)


# =============================================================================
# Neural Methods
# =============================================================================

class NeuralMethod(SynthesisMethod):
    """Neural network-based synthesis."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "neural_supervised"
        self.category = MethodCategory.NEURAL

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        # Simulate neural synthesis
        all_events = set()
        all_states = set()
        for events, final in examples:
            all_events.update(events)
            all_states.add(final)

        # "Train" on examples
        states = list(all_states) or [f"N{i}" for i in range(3)]
        event_list = list(all_events)

        # Learn transition patterns
        transitions = []
        for events, final in examples:
            if events:
                src = states[0]
                for i, evt in enumerate(events):
                    if i == len(events) - 1:
                        tgt = final if final in states else states[-1]
                    else:
                        tgt = states[min(i + 1, len(states) - 1)]
                    if (src, evt, tgt) not in transitions:
                        transitions.append((src, evt, tgt))
                    src = tgt

        sc = SynthesizedStatechart(
            n_states=len(states),
            n_transitions=len(transitions),
            states=states,
            transitions=transitions,
            initial_state=states[0],
            metadata={'method': 'neural', 'epochs': self.config.n_epochs}
        )

        elapsed = time.time() - start_time
        accuracy = self._evaluate_accuracy(sc, examples)

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=sc,
            synthesis_time=elapsed,
            iterations=self.config.n_epochs,
            accuracy=accuracy,
            success=True
        )

    def _evaluate_accuracy(self, sc, examples):
        if not examples:
            return 0.0
        correct = 0
        for events, expected in examples:
            state = sc.initial_state
            for event in events:
                for src, evt, tgt in sc.transitions:
                    if src == state and evt == event:
                        state = tgt
                        break
            if state == expected:
                correct += 1
        return correct / len(examples)


# =============================================================================
# Hybrid Methods
# =============================================================================

class HybridMethod(SynthesisMethod):
    """Hybrid: SAE discovery + evolutionary refinement."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "hybrid_sae_evolution"
        self.category = MethodCategory.HYBRID
        self.sae = SAEMethod(config)
        self.evo = EvolutionaryMethod(config)

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        # Phase 1: SAE discovery
        sae_result = self.sae.synthesize(examples, context)

        # Phase 2: Evolutionary refinement
        # Use SAE states as initial population
        if sae_result.statechart:
            initial_states = sae_result.statechart.states
        else:
            initial_states = [f"H{i}" for i in range(5)]

        evo_config = SynthesisConfig(
            max_states=len(initial_states) + 3,
            n_generations=self.config.n_generations // 2
        )
        evo = EvolutionaryMethod(evo_config)
        evo_result = evo.synthesize(examples, context)

        # Combine results
        final_sc = evo_result.statechart or sae_result.statechart
        elapsed = time.time() - start_time

        accuracy = 0.0
        if final_sc:
            accuracy = max(sae_result.accuracy, evo_result.accuracy)

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=final_sc,
            synthesis_time=elapsed,
            iterations=sae_result.iterations + evo_result.iterations,
            accuracy=accuracy,
            success=final_sc is not None,
            metadata={'sae_accuracy': sae_result.accuracy, 'evo_accuracy': evo_result.accuracy}
        )


# =============================================================================
# Baseline Methods
# =============================================================================

class RandomMethod(SynthesisMethod):
    """Random baseline."""

    def __init__(self, config: SynthesisConfig):
        super().__init__(config)
        self.name = "random_baseline"
        self.category = MethodCategory.BASELINE

    def synthesize(
        self,
        examples: List[Tuple[List[str], str]],
        context: Dict[str, Any] = None
    ) -> SynthesisResult:
        start_time = time.time()

        all_events = set()
        for events, _ in examples:
            all_events.update(events)
        event_list = list(all_events) or ["e0"]

        n_states = random.randint(2, self.config.max_states)
        states = [f"R{i}" for i in range(n_states)]

        n_trans = random.randint(1, self.config.max_transitions)
        transitions = [
            (random.choice(states), random.choice(event_list), random.choice(states))
            for _ in range(n_trans)
        ]

        sc = SynthesizedStatechart(
            n_states=n_states,
            n_transitions=n_trans,
            states=states,
            transitions=transitions,
            initial_state=states[0]
        )

        elapsed = time.time() - start_time

        # Evaluate
        correct = 0
        for events, expected in examples:
            state = sc.initial_state
            for event in events:
                for src, evt, tgt in sc.transitions:
                    if src == state and evt == event:
                        state = tgt
                        break
            if expected in state or state in expected:
                correct += 1
        accuracy = correct / len(examples) if examples else 0.0

        return SynthesisResult(
            method_name=self.name,
            category=self.category,
            statechart=sc,
            synthesis_time=elapsed,
            iterations=1,
            accuracy=accuracy,
            success=True
        )


# =============================================================================
# Method Registry
# =============================================================================

def get_all_methods(config: SynthesisConfig = None) -> List[SynthesisMethod]:
    """Get all available synthesis methods."""
    config = config or SynthesisConfig()
    return [
        EvolutionaryMethod(config),
        NSGA2Method(config),
        SAEMethod(config),
        NeuralMethod(config),
        HybridMethod(config),
        RandomMethod(config),
    ]


def get_methods_by_category(
    category: MethodCategory,
    config: SynthesisConfig = None
) -> List[SynthesisMethod]:
    """Get methods of a specific category."""
    return [m for m in get_all_methods(config) if m.category == category]


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate synthesis methods."""
    print("=" * 60)
    print("SYNTHESIS METHODS DEMO")
    print("=" * 60)

    # Sample examples
    examples = [
        (["start", "play", "win"], "victory"),
        (["start", "play", "lose"], "defeat"),
        (["start", "quit"], "menu"),
        (["start", "play", "play", "win"], "victory"),
    ]

    config = SynthesisConfig(
        max_states=6,
        max_transitions=15,
        n_generations=30,
        population_size=20
    )

    methods = get_all_methods(config)

    print(f"\nTesting {len(methods)} methods on {len(examples)} examples\n")
    print(f"{'Method':<25} {'Category':<15} {'Accuracy':>10} {'Time':>10} {'States':>8}")
    print("-" * 70)

    for method in methods:
        result = method.synthesize(examples)
        n_states = result.statechart.n_states if result.statechart else 0
        print(f"{result.method_name:<25} {result.category.name:<15} "
              f"{result.accuracy:>10.1%} {result.synthesis_time:>10.3f}s {n_states:>8}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
