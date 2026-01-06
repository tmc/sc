"""
Verified Evolution: Evolution with Formal Verification Fitness

Integrates formal verification into the evolutionary process:
1. Generate candidate statecharts via mutation/crossover
2. Verify each candidate against properties
3. Reject candidates that violate properties
4. Fitness combines accuracy + verification status

Key insight: Formal verification as fitness component ensures
evolved statecharts are not just accurate but CORRECT.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
import random
import copy
import time

from .smt_encoder import (
    SMTEncoder, StatechartFormula, StateVariable, TransitionFormula,
    GuardFormula, StateType, HAS_Z3,
)
from .property_checker import (
    PropertyChecker, Property, PropertyResult, PropertyStatus,
    IllegalTransitionProperty, DeadlockFreedomProperty,
    DeterminismProperty, ReachabilityProperty, LivenessProperty,
)


@dataclass
class VerificationFitness:
    """
    Fitness that combines accuracy and verification.

    Total fitness = accuracy_weight * accuracy + verification_weight * verification_score
    """
    accuracy: float = 0.0           # Task accuracy (0-1)
    verification_score: float = 0.0  # Verification score (0-1)
    properties_passed: int = 0       # Number of properties passed
    properties_total: int = 0        # Total properties checked
    verification_time: float = 0.0   # Time spent verifying

    # Weights
    accuracy_weight: float = 0.7
    verification_weight: float = 0.3

    @property
    def total_fitness(self) -> float:
        """Combined fitness score."""
        return (
            self.accuracy_weight * self.accuracy +
            self.verification_weight * self.verification_score
        )

    @property
    def is_valid(self) -> bool:
        """Is statechart valid (all properties pass)?"""
        return self.properties_passed == self.properties_total and self.properties_total > 0


@dataclass
class VerifiedGenome:
    """
    A statechart genome with verification status.

    Extends basic statechart with:
    - Verification results
    - Fitness decomposition
    - Validity flag
    """
    states: List[StateVariable] = field(default_factory=list)
    transitions: List[TransitionFormula] = field(default_factory=list)

    # Fitness
    fitness: VerificationFitness = field(default_factory=VerificationFitness)

    # Verification
    verification_results: List[PropertyResult] = field(default_factory=list)
    is_verified: bool = False

    # Generation tracking
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    def clone(self) -> "VerifiedGenome":
        """Deep copy."""
        new = VerifiedGenome()
        new.states = [copy.deepcopy(s) for s in self.states]
        new.transitions = [copy.deepcopy(t) for t in self.transitions]
        new.generation = self.generation + 1
        return new


class VerifiedEvolver:
    """
    Evolve statecharts with formal verification.

    Evolution loop:
    1. Generate population
    2. Evaluate accuracy (task-specific)
    3. Verify each candidate
    4. Compute combined fitness
    5. Select parents
    6. Generate next generation
    7. Repeat until convergence or valid solution found
    """

    def __init__(
        self,
        state_names: List[str],
        event_names: List[str] = None,
        context_vars: List[str] = None,
        population_size: int = 50,
        verification_weight: float = 0.3,
        reject_invalid: bool = True,
    ):
        self.state_names = state_names
        self.event_names = event_names or []
        self.context_vars = context_vars or []
        self.population_size = population_size
        self.verification_weight = verification_weight
        self.reject_invalid = reject_invalid

        self.encoder = SMTEncoder()
        self.checker = PropertyChecker(self.encoder)

        self.population: List[VerifiedGenome] = []
        self.best_genome: Optional[VerifiedGenome] = None
        self.generation = 0

        # Statistics
        self.total_verified = 0
        self.total_rejected = 0
        self.verification_time = 0.0

    def random_genome(self) -> VerifiedGenome:
        """Generate random statechart genome."""
        genome = VerifiedGenome()

        # Create states
        n_states = random.randint(2, len(self.state_names))
        selected_states = random.sample(self.state_names, n_states)

        for i, name in enumerate(selected_states):
            state = StateVariable(
                name=name,
                state_type=StateType.BASIC,
                is_initial=(i == 0),
                is_final=(i == n_states - 1),
            )
            genome.states.append(state)

        # Create random transitions
        n_trans = random.randint(n_states - 1, n_states * 2)
        for _ in range(n_trans):
            src = random.choice(selected_states)
            tgt = random.choice(selected_states)

            # Random guard
            if self.context_vars and random.random() < 0.3:
                var = random.choice(self.context_vars)
                op = random.choice(["<", ">", "==", ">=", "<="])
                val = random.randint(0, 10)
                guard = GuardFormula(f"{var} {op} {val}")
            else:
                guard = GuardFormula("true")

            trans = TransitionFormula(
                source=src,
                target=tgt,
                guard=guard,
            )
            genome.transitions.append(trans)

        return genome

    def mutate(self, genome: VerifiedGenome) -> VerifiedGenome:
        """Mutate a genome."""
        new = genome.clone()

        mutation = random.choice([
            "add_state", "remove_state",
            "add_transition", "remove_transition",
            "change_guard", "change_target",
        ])

        if mutation == "add_state" and len(new.states) < len(self.state_names):
            used = {s.name for s in new.states}
            available = [n for n in self.state_names if n not in used]
            if available:
                name = random.choice(available)
                new.states.append(StateVariable(name=name, state_type=StateType.BASIC))

        elif mutation == "remove_state" and len(new.states) > 2:
            # Don't remove initial or final
            removable = [s for s in new.states if not s.is_initial and not s.is_final]
            if removable:
                state = random.choice(removable)
                new.states.remove(state)
                # Remove transitions involving this state
                new.transitions = [
                    t for t in new.transitions
                    if t.source != state.name and t.target != state.name
                ]

        elif mutation == "add_transition":
            state_names = [s.name for s in new.states]
            if len(state_names) >= 2:
                src = random.choice(state_names)
                tgt = random.choice(state_names)
                new.transitions.append(TransitionFormula(
                    source=src, target=tgt, guard=GuardFormula("true")
                ))

        elif mutation == "remove_transition" and len(new.transitions) > 1:
            new.transitions.pop(random.randint(0, len(new.transitions) - 1))

        elif mutation == "change_guard" and new.transitions:
            trans = random.choice(new.transitions)
            if self.context_vars:
                var = random.choice(self.context_vars)
                op = random.choice(["<", ">", "=="])
                val = random.randint(0, 10)
                trans.guard = GuardFormula(f"{var} {op} {val}")
            else:
                trans.guard = GuardFormula("true")

        elif mutation == "change_target" and new.transitions:
            trans = random.choice(new.transitions)
            state_names = [s.name for s in new.states]
            trans.target = random.choice(state_names)

        return new

    def crossover(self, parent1: VerifiedGenome, parent2: VerifiedGenome) -> VerifiedGenome:
        """Crossover two genomes."""
        child = VerifiedGenome()

        # Take states from both parents
        all_states = {}
        for s in parent1.states + parent2.states:
            all_states[s.name] = copy.deepcopy(s)

        # Select subset
        n_states = random.randint(2, len(all_states))
        selected = random.sample(list(all_states.keys()), min(n_states, len(all_states)))

        for name in selected:
            child.states.append(all_states[name])

        # Ensure initial and final
        if not any(s.is_initial for s in child.states):
            child.states[0].is_initial = True
        if not any(s.is_final for s in child.states):
            child.states[-1].is_final = True

        # Take transitions that connect selected states
        for trans in parent1.transitions + parent2.transitions:
            if trans.source in selected and trans.target in selected:
                if random.random() < 0.5:
                    child.transitions.append(copy.deepcopy(trans))

        child.generation = max(parent1.generation, parent2.generation) + 1
        return child

    def verify(self, genome: VerifiedGenome) -> VerificationFitness:
        """
        Verify genome and compute verification fitness.

        Returns:
            VerificationFitness with verification results
        """
        t0 = time.time()

        # Encode to SMT
        formula = self.encoder.encode_statechart(genome.states, genome.transitions)

        # Check properties
        allowed = {(t.source, t.target) for t in genome.transitions}
        targets = {s.name for s in genome.states if s.is_final}

        properties = [
            IllegalTransitionProperty(allowed),
            DeadlockFreedomProperty(),
            DeterminismProperty(),
            ReachabilityProperty(),
        ]

        if targets:
            properties.append(LivenessProperty(targets))

        results = []
        passed = 0
        for prop in properties:
            result = prop.check(formula, timeout_ms=1000)
            results.append(result)
            if result.passed:
                passed += 1

        elapsed = time.time() - t0
        self.verification_time += elapsed
        self.total_verified += 1

        genome.verification_results = results
        genome.is_verified = True

        fitness = VerificationFitness(
            verification_score=passed / len(properties) if properties else 0.0,
            properties_passed=passed,
            properties_total=len(properties),
            verification_time=elapsed,
            verification_weight=self.verification_weight,
        )

        return fitness

    def evaluate(
        self,
        genome: VerifiedGenome,
        accuracy_fn: Callable[[VerifiedGenome], float] = None,
    ) -> VerificationFitness:
        """
        Evaluate genome with accuracy and verification.

        Args:
            genome: Genome to evaluate
            accuracy_fn: Function to compute task accuracy

        Returns:
            Combined fitness
        """
        # Verify
        fitness = self.verify(genome)

        # Compute accuracy if provided
        if accuracy_fn:
            fitness.accuracy = accuracy_fn(genome)
        else:
            fitness.accuracy = 0.5  # Default

        genome.fitness = fitness
        return fitness

    def evolve(
        self,
        n_generations: int = 50,
        accuracy_fn: Callable[[VerifiedGenome], float] = None,
        target_fitness: float = 0.95,
        verbose: bool = True,
    ) -> VerifiedGenome:
        """
        Evolve statecharts with verification.

        Args:
            n_generations: Max generations
            accuracy_fn: Accuracy evaluation function
            target_fitness: Stop when reached
            verbose: Print progress

        Returns:
            Best verified genome
        """
        # Initialize population
        self.population = [self.random_genome() for _ in range(self.population_size)]

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate all genomes
            for genome in self.population:
                self.evaluate(genome, accuracy_fn)

            # Reject invalid if configured
            if self.reject_invalid:
                valid = [g for g in self.population if g.fitness.is_valid]
                if valid:
                    self.population = valid
                    self.total_rejected += self.population_size - len(valid)

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness.total_fitness, reverse=True)

            # Track best
            if self.best_genome is None or self.population[0].fitness.total_fitness > self.best_genome.fitness.total_fitness:
                self.best_genome = self.population[0].clone()
                self.best_genome.fitness = self.population[0].fitness

            if verbose and gen % 5 == 0:
                best = self.population[0]
                print(f"Gen {gen:3d}: fitness={best.fitness.total_fitness:.3f} "
                      f"(acc={best.fitness.accuracy:.3f}, ver={best.fitness.verification_score:.3f}) "
                      f"valid={sum(1 for g in self.population if g.fitness.is_valid)}/{len(self.population)}")

            # Check termination
            if self.population[0].fitness.total_fitness >= target_fitness and self.population[0].fitness.is_valid:
                if verbose:
                    print(f"\nTarget fitness reached at generation {gen}")
                break

            # Selection and reproduction
            elite_size = max(1, self.population_size // 10)
            new_pop = [g.clone() for g in self.population[:elite_size]]

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

        return self.best_genome

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'generations': self.generation,
            'total_verified': self.total_verified,
            'total_rejected': self.total_rejected,
            'verification_time': self.verification_time,
            'best_fitness': self.best_genome.fitness.total_fitness if self.best_genome else 0.0,
            'best_is_valid': self.best_genome.fitness.is_valid if self.best_genome else False,
        }


def demo():
    """Demonstrate verified evolution."""
    print("=" * 60)
    print("VERIFIED EVOLUTION: Evolution with Formal Verification")
    print("=" * 60)

    if not HAS_Z3:
        print("\nZ3 not available. Install with: pip install z3-solver")
        print("Running with mock verification.")

    # Define state space
    state_names = ["idle", "running", "paused", "stopped", "error"]
    context_vars = ["counter", "timer"]

    evolver = VerifiedEvolver(
        state_names=state_names,
        context_vars=context_vars,
        population_size=20,
        verification_weight=0.4,
        reject_invalid=True,
    )

    # Simple accuracy function: more states = better (for demo)
    def accuracy_fn(genome: VerifiedGenome) -> float:
        return len(genome.states) / len(state_names)

    print("\nEvolving verified statecharts...")
    best = evolver.evolve(
        n_generations=30,
        accuracy_fn=accuracy_fn,
        target_fitness=0.9,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)

    print(f"\nStatistics: {evolver.get_statistics()}")

    if best:
        print(f"\nBest genome:")
        print(f"  States: {[s.name for s in best.states]}")
        print(f"  Transitions: {len(best.transitions)}")
        print(f"  Total fitness: {best.fitness.total_fitness:.3f}")
        print(f"  Accuracy: {best.fitness.accuracy:.3f}")
        print(f"  Verification: {best.fitness.verification_score:.3f}")
        print(f"  Is valid: {best.fitness.is_valid}")

        print(f"\nVerification results:")
        for result in best.verification_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.property_name}")

    return evolver


if __name__ == "__main__":
    demo()
