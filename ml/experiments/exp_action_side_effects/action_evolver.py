"""
Action Evolver: Learn Action->Guard Dependencies Through Evolution

Core insight: Actions mutate context variables. Guards check context variables.
The causal relationship (which actions enable which guards) can be EVOLVED
rather than hardcoded.

Approach:
1. Observe (action, context_before, context_after, guard_before, guard_after) tuples
2. Evolve hypotheses about which context variables each action affects
3. Evolve hypotheses about which context variables each guard depends on
4. Compute action->guard dependencies from variable intersection

This is UNSUPERVISED - no labeled action->guard relationships needed.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import copy
from collections import defaultdict


# =============================================================================
# Action Effects
# =============================================================================

class EffectType(Enum):
    """Types of effects actions can have on context."""
    SET = auto()       # action sets variable to specific value
    INCREMENT = auto() # action increments variable
    DECREMENT = auto() # action decrements variable
    TOGGLE = auto()    # action toggles boolean variable
    APPEND = auto()    # action appends to list variable
    CLEAR = auto()     # action clears/resets variable
    CONDITIONAL = auto()  # action conditionally modifies variable


@dataclass
class ActionEffect:
    """
    Hypothesis about what effect an action has on context.

    NOT hardcoded - this is EVOLVED from observations.
    """
    action_name: str
    variable: str
    effect_type: EffectType
    confidence: float = 0.0  # How confident we are in this effect

    # Optional parameters for the effect
    value_expr: Optional[str] = None  # Expression for the new value
    condition_expr: Optional[str] = None  # Condition for conditional effects

    def __hash__(self):
        return hash((self.action_name, self.variable, self.effect_type))

    def __eq__(self, other):
        if not isinstance(other, ActionEffect):
            return False
        return (self.action_name == other.action_name and
                self.variable == other.variable and
                self.effect_type == other.effect_type)


# =============================================================================
# Guard Dependencies
# =============================================================================

class DependencyType(Enum):
    """Types of dependencies guards can have on context."""
    READS = auto()      # guard reads variable
    COMPARES = auto()   # guard compares variable to value
    CHECKS_TRUE = auto()  # guard checks if variable is truthy
    CHECKS_FALSE = auto() # guard checks if variable is falsy
    RANGE_CHECK = auto()  # guard checks if variable in range


@dataclass
class GuardDependency:
    """
    Hypothesis about what context a guard depends on.

    NOT hardcoded - this is EVOLVED from observations.
    """
    guard_name: str
    variable: str
    dependency_type: DependencyType
    confidence: float = 0.0

    # Optional parameters
    compare_value: Optional[Any] = None
    compare_op: Optional[str] = None  # '==', '!=', '<', '>', '<=', '>='

    def __hash__(self):
        return hash((self.guard_name, self.variable, self.dependency_type))

    def __eq__(self, other):
        if not isinstance(other, GuardDependency):
            return False
        return (self.guard_name == other.guard_name and
                self.variable == other.variable and
                self.dependency_type == other.dependency_type)


# =============================================================================
# Dependency Genome - Evolvable Representation
# =============================================================================

@dataclass
class DependencyGenome:
    """
    Evolvable genome representing action->guard dependency hypotheses.

    This is what we EVOLVE to learn the relationships.
    """
    # Hypothesized action effects
    action_effects: Dict[str, Set[ActionEffect]] = field(default_factory=dict)

    # Hypothesized guard dependencies
    guard_deps: Dict[str, Set[GuardDependency]] = field(default_factory=dict)

    # Fitness metrics
    fitness: float = 0.0
    prediction_accuracy: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0

    def get_action_variables(self, action_name: str) -> Set[str]:
        """Get variables that action is hypothesized to modify."""
        if action_name not in self.action_effects:
            return set()
        return {e.variable for e in self.action_effects[action_name]}

    def get_guard_variables(self, guard_name: str) -> Set[str]:
        """Get variables that guard is hypothesized to depend on."""
        if guard_name not in self.guard_deps:
            return set()
        return {d.variable for d in self.guard_deps[guard_name]}

    def predict_enables(self, action_name: str, guard_name: str) -> float:
        """
        Predict probability that action enables guard.

        Based on variable intersection between action effects and guard deps.
        """
        action_vars = self.get_action_variables(action_name)
        guard_vars = self.get_guard_variables(guard_name)

        if not action_vars or not guard_vars:
            return 0.0

        intersection = action_vars & guard_vars
        if not intersection:
            return 0.0

        # Weight by confidence scores
        action_conf = sum(
            e.confidence for e in self.action_effects.get(action_name, set())
            if e.variable in intersection
        )
        guard_conf = sum(
            d.confidence for d in self.guard_deps.get(guard_name, set())
            if d.variable in intersection
        )

        n_effects = len(self.action_effects.get(action_name, set()))
        n_deps = len(self.guard_deps.get(guard_name, set()))

        if n_effects == 0 or n_deps == 0:
            return 0.0

        # Jaccard-like score weighted by confidence
        score = (len(intersection) / (len(action_vars) + len(guard_vars) - len(intersection)))
        confidence = (action_conf / n_effects + guard_conf / n_deps) / 2

        return score * confidence

    def clone(self) -> "DependencyGenome":
        """Deep copy for mutation."""
        new = DependencyGenome()
        new.action_effects = {
            k: set(e for e in v) for k, v in self.action_effects.items()
        }
        new.guard_deps = {
            k: set(d for d in v) for k, v in self.guard_deps.items()
        }
        return new


# =============================================================================
# Observation Collection
# =============================================================================

@dataclass
class ActionObservation:
    """Observation of an action's effect on context and guards."""
    action_name: str
    context_before: Dict[str, Any]
    context_after: Dict[str, Any]
    guards_before: Dict[str, bool]  # guard_name -> was_enabled
    guards_after: Dict[str, bool]   # guard_name -> is_enabled

    @property
    def changed_variables(self) -> Set[str]:
        """Variables that changed."""
        changed = set()
        all_vars = set(self.context_before.keys()) | set(self.context_after.keys())
        for var in all_vars:
            if self.context_before.get(var) != self.context_after.get(var):
                changed.add(var)
        return changed

    @property
    def enabled_guards(self) -> Set[str]:
        """Guards that became enabled (False -> True)."""
        enabled = set()
        for guard, after in self.guards_after.items():
            before = self.guards_before.get(guard, False)
            if not before and after:
                enabled.add(guard)
        return enabled

    @property
    def disabled_guards(self) -> Set[str]:
        """Guards that became disabled (True -> False)."""
        disabled = set()
        for guard, after in self.guards_after.items():
            before = self.guards_before.get(guard, True)
            if before and not after:
                disabled.add(guard)
        return disabled


# =============================================================================
# Action Evolver
# =============================================================================

class ActionEvolver:
    """
    Evolve action->guard dependencies from observations.

    NO HARDCODING. Everything is learned through evolution.
    """

    def __init__(
        self,
        known_actions: List[str],
        known_guards: List[str],
        known_variables: List[str],
        population_size: int = 50,
    ):
        self.known_actions = known_actions
        self.known_guards = known_guards
        self.known_variables = known_variables
        self.population_size = population_size

        # Observations collected
        self.observations: List[ActionObservation] = []

        # Population of genomes
        self.population: List[DependencyGenome] = []

        # Best genome found
        self.best_genome: Optional[DependencyGenome] = None

        # Tracking
        self.generation = 0

    def collect_observation(self, obs: ActionObservation):
        """Collect an observation of action effects."""
        self.observations.append(obs)

    def random_genome(self) -> DependencyGenome:
        """Generate a random dependency genome."""
        genome = DependencyGenome()

        # Random action effects
        for action in self.known_actions:
            n_effects = random.randint(0, len(self.known_variables))
            effects = set()
            for _ in range(n_effects):
                var = random.choice(self.known_variables)
                effect_type = random.choice(list(EffectType))
                effects.add(ActionEffect(
                    action_name=action,
                    variable=var,
                    effect_type=effect_type,
                    confidence=random.random(),
                ))
            if effects:
                genome.action_effects[action] = effects

        # Random guard dependencies
        for guard in self.known_guards:
            n_deps = random.randint(0, len(self.known_variables))
            deps = set()
            for _ in range(n_deps):
                var = random.choice(self.known_variables)
                dep_type = random.choice(list(DependencyType))
                deps.add(GuardDependency(
                    guard_name=guard,
                    variable=var,
                    dependency_type=dep_type,
                    confidence=random.random(),
                ))
            if deps:
                genome.guard_deps[guard] = deps

        return genome

    def _seed_from_observations(self) -> DependencyGenome:
        """Create a genome seeded from actual observations."""
        genome = DependencyGenome()

        # Collect observed variable changes per action
        action_var_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        guard_var_corr: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        action_counts: Dict[str, int] = defaultdict(int)

        for obs in self.observations:
            action_counts[obs.action_name] += 1
            for var in obs.changed_variables:
                action_var_counts[obs.action_name][var] += 1

            # Track correlation: which variables change when guards change
            for guard in obs.enabled_guards | obs.disabled_guards:
                for var in obs.changed_variables:
                    guard_var_corr[guard][var] += 1

        # Create effects with confidence based on observation frequency
        for action, var_counts in action_var_counts.items():
            effects = set()
            total = action_counts[action]
            for var, count in var_counts.items():
                confidence = count / total if total > 0 else 0.0
                if confidence > 0.1:  # Only include if seen >10% of the time
                    effects.add(ActionEffect(
                        action_name=action,
                        variable=var,
                        effect_type=EffectType.SET,  # Default, will be refined
                        confidence=confidence,
                    ))
            if effects:
                genome.action_effects[action] = effects

        # Create guard deps with confidence based on correlation
        for guard, var_counts in guard_var_corr.items():
            deps = set()
            total = sum(var_counts.values())
            for var, count in var_counts.items():
                confidence = count / total if total > 0 else 0.0
                if confidence > 0.1:
                    deps.add(GuardDependency(
                        guard_name=guard,
                        variable=var,
                        dependency_type=DependencyType.READS,
                        confidence=confidence,
                    ))
            if deps:
                genome.guard_deps[guard] = deps

        return genome

    def mutate(self, genome: DependencyGenome, mutation_rate: float = 0.3) -> DependencyGenome:
        """Mutate a genome."""
        new = genome.clone()

        mutation_type = random.choice([
            "add_effect", "remove_effect", "modify_effect_confidence",
            "add_dep", "remove_dep", "modify_dep_confidence",
            "change_effect_type", "change_dep_type",
        ])

        if mutation_type == "add_effect":
            action = random.choice(self.known_actions)
            var = random.choice(self.known_variables)
            effect_type = random.choice(list(EffectType))
            effect = ActionEffect(
                action_name=action,
                variable=var,
                effect_type=effect_type,
                confidence=random.random(),
            )
            if action not in new.action_effects:
                new.action_effects[action] = set()
            new.action_effects[action].add(effect)

        elif mutation_type == "remove_effect":
            if new.action_effects:
                action = random.choice(list(new.action_effects.keys()))
                if new.action_effects[action]:
                    effect = random.choice(list(new.action_effects[action]))
                    new.action_effects[action].discard(effect)

        elif mutation_type == "modify_effect_confidence":
            if new.action_effects:
                action = random.choice(list(new.action_effects.keys()))
                if new.action_effects[action]:
                    effects = list(new.action_effects[action])
                    old_effect = random.choice(effects)
                    new.action_effects[action].discard(old_effect)
                    new_effect = ActionEffect(
                        action_name=old_effect.action_name,
                        variable=old_effect.variable,
                        effect_type=old_effect.effect_type,
                        confidence=max(0.0, min(1.0, old_effect.confidence + random.uniform(-0.2, 0.2))),
                    )
                    new.action_effects[action].add(new_effect)

        elif mutation_type == "add_dep":
            guard = random.choice(self.known_guards)
            var = random.choice(self.known_variables)
            dep_type = random.choice(list(DependencyType))
            dep = GuardDependency(
                guard_name=guard,
                variable=var,
                dependency_type=dep_type,
                confidence=random.random(),
            )
            if guard not in new.guard_deps:
                new.guard_deps[guard] = set()
            new.guard_deps[guard].add(dep)

        elif mutation_type == "remove_dep":
            if new.guard_deps:
                guard = random.choice(list(new.guard_deps.keys()))
                if new.guard_deps[guard]:
                    dep = random.choice(list(new.guard_deps[guard]))
                    new.guard_deps[guard].discard(dep)

        elif mutation_type == "modify_dep_confidence":
            if new.guard_deps:
                guard = random.choice(list(new.guard_deps.keys()))
                if new.guard_deps[guard]:
                    deps = list(new.guard_deps[guard])
                    old_dep = random.choice(deps)
                    new.guard_deps[guard].discard(old_dep)
                    new_dep = GuardDependency(
                        guard_name=old_dep.guard_name,
                        variable=old_dep.variable,
                        dependency_type=old_dep.dependency_type,
                        confidence=max(0.0, min(1.0, old_dep.confidence + random.uniform(-0.2, 0.2))),
                    )
                    new.guard_deps[guard].add(new_dep)

        elif mutation_type == "change_effect_type":
            if new.action_effects:
                action = random.choice(list(new.action_effects.keys()))
                if new.action_effects[action]:
                    effects = list(new.action_effects[action])
                    old_effect = random.choice(effects)
                    new.action_effects[action].discard(old_effect)
                    new_effect = ActionEffect(
                        action_name=old_effect.action_name,
                        variable=old_effect.variable,
                        effect_type=random.choice(list(EffectType)),
                        confidence=old_effect.confidence,
                    )
                    new.action_effects[action].add(new_effect)

        elif mutation_type == "change_dep_type":
            if new.guard_deps:
                guard = random.choice(list(new.guard_deps.keys()))
                if new.guard_deps[guard]:
                    deps = list(new.guard_deps[guard])
                    old_dep = random.choice(deps)
                    new.guard_deps[guard].discard(old_dep)
                    new_dep = GuardDependency(
                        guard_name=old_dep.guard_name,
                        variable=old_dep.variable,
                        dependency_type=random.choice(list(DependencyType)),
                        confidence=old_dep.confidence,
                    )
                    new.guard_deps[guard].add(new_dep)

        return new

    def crossover(self, parent1: DependencyGenome, parent2: DependencyGenome) -> DependencyGenome:
        """Crossover two genomes."""
        child = DependencyGenome()

        # Mix action effects
        all_actions = set(parent1.action_effects.keys()) | set(parent2.action_effects.keys())
        for action in all_actions:
            effects1 = parent1.action_effects.get(action, set())
            effects2 = parent2.action_effects.get(action, set())

            # Take effects from both parents
            child_effects = set()
            for e in effects1:
                if random.random() < 0.5:
                    child_effects.add(e)
            for e in effects2:
                if random.random() < 0.5:
                    child_effects.add(e)

            if child_effects:
                child.action_effects[action] = child_effects

        # Mix guard deps
        all_guards = set(parent1.guard_deps.keys()) | set(parent2.guard_deps.keys())
        for guard in all_guards:
            deps1 = parent1.guard_deps.get(guard, set())
            deps2 = parent2.guard_deps.get(guard, set())

            child_deps = set()
            for d in deps1:
                if random.random() < 0.5:
                    child_deps.add(d)
            for d in deps2:
                if random.random() < 0.5:
                    child_deps.add(d)

            if child_deps:
                child.guard_deps[guard] = child_deps

        return child

    def evaluate_fitness(self, genome: DependencyGenome) -> float:
        """
        Evaluate genome fitness against observations.

        Fitness = accuracy of predicting guard enablement changes.
        """
        if not self.observations:
            return 0.0

        tp, fp, tn, fn = 0, 0, 0, 0

        for obs in self.observations:
            for guard in self.known_guards:
                # Predict: does this action enable this guard?
                prediction_score = genome.predict_enables(obs.action_name, guard)
                predicted_enables = prediction_score > 0.5

                # Ground truth: did guard become enabled?
                actual_enabled = guard in obs.enabled_guards

                if predicted_enables and actual_enabled:
                    tp += 1
                elif predicted_enables and not actual_enabled:
                    fp += 1
                elif not predicted_enables and actual_enabled:
                    fn += 1
                else:
                    tn += 1

        genome.false_positives = fp
        genome.false_negatives = fn

        # F1-style fitness
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        genome.prediction_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        genome.fitness = f1

        return f1

    def evolve(
        self,
        n_generations: int = 100,
        verbose: bool = True,
    ) -> DependencyGenome:
        """
        Evolve action->guard dependencies from observations.

        Returns the best genome found.
        """
        if not self.observations:
            raise ValueError("No observations collected. Call collect_observation first.")

        # Initialize population
        self.population = []

        # Seed some genomes from observations
        for _ in range(self.population_size // 4):
            self.population.append(self._seed_from_observations())

        # Fill rest with random genomes
        while len(self.population) < self.population_size:
            self.population.append(self.random_genome())

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate
            for genome in self.population:
                self.evaluate_fitness(genome)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].clone()

            if verbose and gen % 10 == 0:
                best = self.population[0]
                print(f"Gen {gen:3d}: fitness={best.fitness:.3f}, "
                      f"accuracy={best.prediction_accuracy:.3f}, "
                      f"FP={best.false_positives}, FN={best.false_negatives}")

            # Perfect?
            if self.population[0].fitness >= 0.99:
                break

            # Selection and reproduction
            elite_size = max(1, self.population_size // 10)
            new_pop = [g.clone() for g in self.population[:elite_size]]

            while len(new_pop) < self.population_size:
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(self.population[:self.population_size // 2])
                    child = self.mutate(parent)
                else:
                    # Crossover
                    p1 = random.choice(self.population[:self.population_size // 2])
                    p2 = random.choice(self.population[:self.population_size // 2])
                    child = self.crossover(p1, p2)

                new_pop.append(child)

            self.population = new_pop

        return self.best_genome

    def get_learned_dependencies(self) -> Dict[str, Dict[str, float]]:
        """
        Get the learned action->guard dependencies.

        Returns:
            Dict mapping action_name -> {guard_name -> probability}
        """
        if self.best_genome is None:
            return {}

        deps = {}
        for action in self.known_actions:
            deps[action] = {}
            for guard in self.known_guards:
                prob = self.best_genome.predict_enables(action, guard)
                if prob > 0.1:  # Only include non-trivial dependencies
                    deps[action][guard] = prob

        return deps

    def print_learned_dependencies(self):
        """Print learned dependencies in readable format."""
        deps = self.get_learned_dependencies()

        print("\n" + "=" * 60)
        print("LEARNED ACTION -> GUARD DEPENDENCIES")
        print("=" * 60)

        for action, guard_probs in deps.items():
            if guard_probs:
                print(f"\n{action}:")
                for guard, prob in sorted(guard_probs.items(), key=lambda x: -x[1]):
                    print(f"  -> {guard}: {prob:.3f}")

        print("\n" + "=" * 60)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate action->guard dependency evolution."""
    print("=" * 60)
    print("ACTION EVOLVER: Learning Action->Guard Dependencies")
    print("=" * 60)

    # Define a simple domain: traffic light controller
    actions = ["turn_green", "turn_yellow", "turn_red", "start_timer", "stop_timer"]
    guards = ["is_green", "is_yellow", "is_red", "timer_expired", "safe_to_change"]
    variables = ["light_state", "timer_value", "timer_running", "pedestrian_waiting"]

    evolver = ActionEvolver(
        known_actions=actions,
        known_guards=guards,
        known_variables=variables,
        population_size=30,
    )

    # Generate synthetic observations
    # Key relationships we want to LEARN (not hardcode):
    # - turn_green modifies light_state, which is_green checks
    # - turn_yellow modifies light_state, which is_yellow checks
    # - turn_red modifies light_state, which is_red checks
    # - start_timer modifies timer_running, which timer_expired depends on

    observations = [
        # turn_green enables is_green guard
        ActionObservation(
            action_name="turn_green",
            context_before={"light_state": "red", "timer_value": 0, "timer_running": False},
            context_after={"light_state": "green", "timer_value": 0, "timer_running": False},
            guards_before={"is_green": False, "is_yellow": False, "is_red": True, "timer_expired": False, "safe_to_change": True},
            guards_after={"is_green": True, "is_yellow": False, "is_red": False, "timer_expired": False, "safe_to_change": True},
        ),
        # turn_yellow enables is_yellow guard
        ActionObservation(
            action_name="turn_yellow",
            context_before={"light_state": "green", "timer_value": 30, "timer_running": True},
            context_after={"light_state": "yellow", "timer_value": 30, "timer_running": True},
            guards_before={"is_green": True, "is_yellow": False, "is_red": False, "timer_expired": True, "safe_to_change": False},
            guards_after={"is_green": False, "is_yellow": True, "is_red": False, "timer_expired": True, "safe_to_change": False},
        ),
        # turn_red enables is_red guard
        ActionObservation(
            action_name="turn_red",
            context_before={"light_state": "yellow", "timer_value": 5, "timer_running": True},
            context_after={"light_state": "red", "timer_value": 5, "timer_running": True},
            guards_before={"is_green": False, "is_yellow": True, "is_red": False, "timer_expired": True, "safe_to_change": False},
            guards_after={"is_green": False, "is_yellow": False, "is_red": True, "timer_expired": True, "safe_to_change": True},
        ),
        # start_timer affects timer_expired
        ActionObservation(
            action_name="start_timer",
            context_before={"light_state": "green", "timer_value": 0, "timer_running": False},
            context_after={"light_state": "green", "timer_value": 30, "timer_running": True},
            guards_before={"is_green": True, "is_yellow": False, "is_red": False, "timer_expired": True, "safe_to_change": True},
            guards_after={"is_green": True, "is_yellow": False, "is_red": False, "timer_expired": False, "safe_to_change": False},
        ),
    ]

    # Add more variation
    for _ in range(20):
        for obs in observations:
            evolver.collect_observation(obs)

    print(f"\nCollected {len(evolver.observations)} observations")
    print(f"Actions: {actions}")
    print(f"Guards: {guards}")
    print(f"Variables: {variables}")

    print("\nEvolving dependencies...")
    best = evolver.evolve(n_generations=50, verbose=True)

    print(f"\nBest genome fitness: {best.fitness:.3f}")
    evolver.print_learned_dependencies()

    # Verify learned relationships
    print("\nKey learned relationships:")
    learned = evolver.get_learned_dependencies()
    expected = [
        ("turn_green", "is_green"),
        ("turn_yellow", "is_yellow"),
        ("turn_red", "is_red"),
    ]
    for action, guard in expected:
        prob = learned.get(action, {}).get(guard, 0.0)
        status = "LEARNED" if prob > 0.5 else "MISSED"
        print(f"  {action} -> {guard}: {prob:.3f} [{status}]")

    return evolver


if __name__ == "__main__":
    demo()
