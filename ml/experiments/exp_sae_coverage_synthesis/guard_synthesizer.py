"""
Guard Synthesizer: Synthesize Guards from Feature Activations

Key insight: Guards are patterns over SAE features.
A guard is true when certain features are active/inactive.

From exp_guard_synthesis:
- Guards are boolean expressions over context
- We evolve guards from positive/negative examples

Here, the "context" is the SAE feature activation pattern.
Guards become: "feature 42 active AND feature 17 inactive"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet, Any
from enum import Enum, auto
from collections import defaultdict
import random


class GuardOp(Enum):
    """Guard operations."""
    FEATURE_ACTIVE = auto()      # feature_i > threshold
    FEATURE_INACTIVE = auto()    # feature_i < threshold
    FEATURE_ABOVE = auto()       # feature_i > value
    FEATURE_BELOW = auto()       # feature_i < value
    AND = auto()                 # guard1 AND guard2
    OR = auto()                  # guard1 OR guard2
    NOT = auto()                 # NOT guard


@dataclass
class ActivationPattern:
    """
    A pattern of feature activations.

    Used as positive/negative examples for guard synthesis.
    """
    active_features: FrozenSet[int]
    activations: Dict[int, float]
    label: bool  # True = transition should fire, False = shouldn't

    def __hash__(self):
        return hash((self.active_features, self.label))


@dataclass
class FeatureGuard:
    """
    A guard condition based on SAE feature activations.

    Guards are boolean expressions over features.
    """
    op: GuardOp
    feature_idx: Optional[int] = None
    threshold: float = 0.1
    children: List["FeatureGuard"] = field(default_factory=list)

    # Evolution tracking
    fitness: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def evaluate(self, pattern: ActivationPattern) -> bool:
        """Evaluate guard on activation pattern."""
        if self.op == GuardOp.FEATURE_ACTIVE:
            return self.feature_idx in pattern.active_features

        elif self.op == GuardOp.FEATURE_INACTIVE:
            return self.feature_idx not in pattern.active_features

        elif self.op == GuardOp.FEATURE_ABOVE:
            act = pattern.activations.get(self.feature_idx, 0.0)
            return act > self.threshold

        elif self.op == GuardOp.FEATURE_BELOW:
            act = pattern.activations.get(self.feature_idx, 0.0)
            return act < self.threshold

        elif self.op == GuardOp.AND:
            return all(child.evaluate(pattern) for child in self.children)

        elif self.op == GuardOp.OR:
            return any(child.evaluate(pattern) for child in self.children)

        elif self.op == GuardOp.NOT:
            return not self.children[0].evaluate(pattern) if self.children else True

        return True

    def to_string(self) -> str:
        """Convert guard to readable string."""
        if self.op == GuardOp.FEATURE_ACTIVE:
            return f"F{self.feature_idx}_active"

        elif self.op == GuardOp.FEATURE_INACTIVE:
            return f"F{self.feature_idx}_inactive"

        elif self.op == GuardOp.FEATURE_ABOVE:
            return f"F{self.feature_idx}>{self.threshold:.2f}"

        elif self.op == GuardOp.FEATURE_BELOW:
            return f"F{self.feature_idx}<{self.threshold:.2f}"

        elif self.op == GuardOp.AND:
            parts = [child.to_string() for child in self.children]
            return f"({' AND '.join(parts)})"

        elif self.op == GuardOp.OR:
            parts = [child.to_string() for child in self.children]
            return f"({' OR '.join(parts)})"

        elif self.op == GuardOp.NOT:
            return f"NOT({self.children[0].to_string()})" if self.children else "NOT(?)"

        return "True"

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def clone(self) -> "FeatureGuard":
        """Deep copy."""
        return FeatureGuard(
            op=self.op,
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            children=[c.clone() for c in self.children],
        )


class GuardSynthesizer:
    """
    Synthesize guards from feature activation patterns.

    Given positive examples (when transition should fire) and
    negative examples (when it shouldn't), evolve a guard.
    """

    def __init__(
        self,
        feature_indices: List[int],
        max_depth: int = 3,
        population_size: int = 50,
    ):
        self.feature_indices = feature_indices
        self.max_depth = max_depth
        self.population_size = population_size

        self.population: List[FeatureGuard] = []
        self.best_guard: Optional[FeatureGuard] = None

    def random_guard(self, depth: int = 0) -> FeatureGuard:
        """Generate random guard."""
        if depth >= self.max_depth or random.random() < 0.4:
            # Terminal: feature check
            feature = random.choice(self.feature_indices)
            op = random.choice([
                GuardOp.FEATURE_ACTIVE,
                GuardOp.FEATURE_INACTIVE,
                GuardOp.FEATURE_ABOVE,
                GuardOp.FEATURE_BELOW,
            ])
            threshold = random.uniform(0.05, 0.5)
            return FeatureGuard(op=op, feature_idx=feature, threshold=threshold)

        # Non-terminal: AND, OR, NOT
        op = random.choice([GuardOp.AND, GuardOp.OR, GuardOp.NOT])

        if op == GuardOp.NOT:
            return FeatureGuard(
                op=op,
                children=[self.random_guard(depth + 1)],
            )
        else:
            n_children = random.randint(2, 3)
            return FeatureGuard(
                op=op,
                children=[self.random_guard(depth + 1) for _ in range(n_children)],
            )

    def mutate(self, guard: FeatureGuard) -> FeatureGuard:
        """Mutate a guard."""
        new = guard.clone()

        mutation = random.choice([
            "change_feature", "change_op", "change_threshold",
            "add_child", "remove_child", "replace_subtree",
        ])

        if mutation == "change_feature" and new.feature_idx is not None:
            new.feature_idx = random.choice(self.feature_indices)

        elif mutation == "change_op":
            if new.op in [GuardOp.FEATURE_ACTIVE, GuardOp.FEATURE_INACTIVE]:
                new.op = random.choice([GuardOp.FEATURE_ACTIVE, GuardOp.FEATURE_INACTIVE])
            elif new.op in [GuardOp.FEATURE_ABOVE, GuardOp.FEATURE_BELOW]:
                new.op = random.choice([GuardOp.FEATURE_ABOVE, GuardOp.FEATURE_BELOW])
            elif new.op in [GuardOp.AND, GuardOp.OR]:
                new.op = GuardOp.OR if new.op == GuardOp.AND else GuardOp.AND

        elif mutation == "change_threshold":
            new.threshold = max(0.01, new.threshold + random.uniform(-0.1, 0.1))

        elif mutation == "add_child" and new.op in [GuardOp.AND, GuardOp.OR]:
            new.children.append(self.random_guard(2))

        elif mutation == "remove_child" and new.op in [GuardOp.AND, GuardOp.OR] and len(new.children) > 2:
            new.children.pop(random.randint(0, len(new.children) - 1))

        elif mutation == "replace_subtree":
            new = self.random_guard(1)

        return new

    def crossover(self, parent1: FeatureGuard, parent2: FeatureGuard) -> FeatureGuard:
        """Crossover two guards."""
        # Simple: take structure from parent1, parameters from parent2
        child = parent1.clone()

        if child.feature_idx is not None and parent2.feature_idx is not None:
            if random.random() < 0.5:
                child.feature_idx = parent2.feature_idx

        if random.random() < 0.5:
            child.threshold = parent2.threshold

        # Mix children
        if child.children and parent2.children:
            for i in range(min(len(child.children), len(parent2.children))):
                if random.random() < 0.5:
                    child.children[i] = parent2.children[i].clone()

        return child

    def evaluate(
        self,
        guard: FeatureGuard,
        positive: List[ActivationPattern],
        negative: List[ActivationPattern],
    ) -> float:
        """Evaluate guard fitness."""
        tp, fp, tn, fn = 0, 0, 0, 0

        for pattern in positive:
            if guard.evaluate(pattern):
                tp += 1
            else:
                fn += 1

        for pattern in negative:
            if guard.evaluate(pattern):
                fp += 1
            else:
                tn += 1

        guard.true_positives = tp
        guard.false_positives = fp
        guard.true_negatives = tn
        guard.false_negatives = fn
        guard.fitness = guard.f1

        return guard.fitness

    def synthesize(
        self,
        positive: List[ActivationPattern],
        negative: List[ActivationPattern],
        n_generations: int = 50,
        verbose: bool = True,
    ) -> FeatureGuard:
        """
        Synthesize a guard from positive/negative examples.

        Args:
            positive: Patterns where guard should return True
            negative: Patterns where guard should return False
            n_generations: Evolution generations
            verbose: Print progress

        Returns:
            Best guard found
        """
        # Initialize population
        self.population = [self.random_guard() for _ in range(self.population_size)]

        # Also add simple feature-based guards
        for feature in self.feature_indices[:min(10, len(self.feature_indices))]:
            self.population.append(FeatureGuard(
                op=GuardOp.FEATURE_ACTIVE, feature_idx=feature
            ))
            self.population.append(FeatureGuard(
                op=GuardOp.FEATURE_INACTIVE, feature_idx=feature
            ))

        for gen in range(n_generations):
            # Evaluate
            for guard in self.population:
                self.evaluate(guard, positive, negative)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if self.best_guard is None or self.population[0].fitness > self.best_guard.fitness:
                self.best_guard = self.population[0].clone()
                self.best_guard.fitness = self.population[0].fitness
                self.best_guard.true_positives = self.population[0].true_positives
                self.best_guard.false_positives = self.population[0].false_positives
                self.best_guard.true_negatives = self.population[0].true_negatives
                self.best_guard.false_negatives = self.population[0].false_negatives

            if verbose and gen % 10 == 0:
                best = self.population[0]
                print(f"Gen {gen:3d}: F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
                print(f"         {best.to_string()[:50]}...")

            # Perfect?
            if self.population[0].f1 >= 0.99:
                break

            # Selection and reproduction
            elite = self.population[:5]
            new_pop = [g.clone() for g in elite]

            while len(new_pop) < self.population_size:
                if random.random() < 0.7:
                    parent = random.choice(self.population[:self.population_size // 2])
                    child = self.mutate(parent)
                else:
                    p1 = random.choice(self.population[:self.population_size // 2])
                    p2 = random.choice(self.population[:self.population_size // 2])
                    child = self.crossover(p1, p2)
                new_pop.append(child)

            self.population = new_pop

        return self.best_guard

    def synthesize_from_transitions(
        self,
        positive_transitions: List[Tuple[FrozenSet[int], FrozenSet[int]]],
        negative_transitions: List[Tuple[FrozenSet[int], FrozenSet[int]]],
        n_generations: int = 50,
    ) -> FeatureGuard:
        """
        Synthesize guard from transition examples.

        Args:
            positive_transitions: (src_config, tgt_config) where transition fires
            negative_transitions: (src_config, tgt_config) where transition doesn't fire

        Returns:
            Guard that distinguishes positive from negative
        """
        # Convert to activation patterns (use source config)
        positive = []
        for src, tgt in positive_transitions:
            positive.append(ActivationPattern(
                active_features=src,
                activations={f: 1.0 for f in src},
                label=True,
            ))

        negative = []
        for src, tgt in negative_transitions:
            negative.append(ActivationPattern(
                active_features=src,
                activations={f: 1.0 for f in src},
                label=False,
            ))

        return self.synthesize(positive, negative, n_generations)


def demo():
    """Demonstrate guard synthesis."""
    print("=" * 60)
    print("GUARD SYNTHESIZER: Guards from Feature Activations")
    print("=" * 60)

    # Feature indices
    features = list(range(100))

    synthesizer = GuardSynthesizer(
        feature_indices=features,
        max_depth=3,
        population_size=30,
    )

    # Create synthetic examples
    # Positive: feature 5 active AND feature 10 inactive
    positive = []
    for _ in range(20):
        active = {5} | set(random.sample([i for i in range(100) if i != 10], 5))
        positive.append(ActivationPattern(
            active_features=frozenset(active),
            activations={f: random.uniform(0.2, 0.8) for f in active},
            label=True,
        ))

    # Negative: feature 5 inactive OR feature 10 active
    negative = []
    for _ in range(20):
        # Case 1: 5 inactive
        active = set(random.sample([i for i in range(100) if i != 5], 6))
        negative.append(ActivationPattern(
            active_features=frozenset(active),
            activations={f: random.uniform(0.2, 0.8) for f in active},
            label=False,
        ))
        # Case 2: 10 active
        active = {10} | set(random.sample(range(100), 5))
        negative.append(ActivationPattern(
            active_features=frozenset(active),
            activations={f: random.uniform(0.2, 0.8) for f in active},
            label=False,
        ))

    print(f"\nPositive examples: {len(positive)}")
    print(f"Negative examples: {len(negative)}")
    print("\nGround truth: F5_active AND F10_inactive")

    print("\nSynthesizing guard...")
    best = synthesizer.synthesize(positive, negative, n_generations=30, verbose=True)

    print(f"\n{'='*60}")
    print(f"BEST GUARD: {best.to_string()}")
    print(f"F1: {best.f1:.3f}, Precision: {best.precision:.3f}, Recall: {best.recall:.3f}")
    print(f"{'='*60}")

    return synthesizer


if __name__ == "__main__":
    demo()
