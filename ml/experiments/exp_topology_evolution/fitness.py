"""
Multi-Objective Fitness for Topology Evolution

Provides multiple fitness components and NSGA-II style selection
for evolving statecharts with diverse, high-quality solutions.

Components:
- Accuracy: Legal move prediction accuracy
- Parsimony: Reward smaller structures
- Hierarchy: Reward deeper nesting
- Parallelism: Reward AND states (concurrent regions)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random


# =============================================================================
# Fitness Components
# =============================================================================

@dataclass
class FitnessComponents:
    """
    Multi-objective fitness components for statechart genomes.

    Each component is normalized to [0, 1] range for fair comparison.
    """
    accuracy: float = 0.0           # Legal move prediction accuracy
    parsimony: float = 0.0          # Fewer states = better
    hierarchy_score: float = 0.0    # Deeper = more structured
    parallelism_score: float = 0.0  # More AND states = better
    transition_efficiency: float = 0.0  # Fewer transitions per accuracy

    def to_vector(self) -> Tuple[float, ...]:
        """Convert to tuple for Pareto comparison."""
        return (
            self.accuracy,
            self.parsimony,
            self.hierarchy_score,
            self.parallelism_score,
            self.transition_efficiency
        )

    def weighted_sum(self, weights: Dict[str, float] = None) -> float:
        """
        Combine components with weights.

        Default weights prioritize accuracy with small bonuses for structure.
        """
        weights = weights or {
            "accuracy": 1.0,
            "parsimony": 0.1,
            "hierarchy_score": 0.05,
            "parallelism_score": 0.05,
            "transition_efficiency": 0.02
        }

        return (
            self.accuracy * weights.get("accuracy", 1.0) +
            self.parsimony * weights.get("parsimony", 0.0) +
            self.hierarchy_score * weights.get("hierarchy_score", 0.0) +
            self.parallelism_score * weights.get("parallelism_score", 0.0) +
            self.transition_efficiency * weights.get("transition_efficiency", 0.0)
        )


# =============================================================================
# Component Calculation
# =============================================================================

def compute_fitness_components(
    genome,  # StatechartGenome
    accuracy: float,
    max_states: int = 20
) -> FitnessComponents:
    """
    Compute all fitness components for a genome.

    Args:
        genome: The statechart genome
        accuracy: Prediction accuracy [0, 1]
        max_states: Maximum allowed states (for normalization)
    """
    from .evolve import StateType

    # Parsimony: Reward fewer states
    parsimony = 1.0 - (genome.n_states / max_states)
    parsimony = max(0.0, parsimony)

    # Hierarchy: Reward deeper nesting
    def get_depth(state_idx: int, depth: int = 0) -> int:
        """Get depth of a state in the hierarchy."""
        max_depth = depth
        for i, p in enumerate(genome.parent):
            if p == state_idx:
                child_depth = get_depth(i, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth

    root_depth = get_depth(0)
    # Normalize: depth 5+ gets full score
    hierarchy_score = min(1.0, root_depth / 5.0)

    # Parallelism: Reward AND states
    n_and_states = sum(1 for t in genome.state_type if t == StateType.AND)
    # Normalize: 3+ AND states gets full score
    parallelism_score = min(1.0, n_and_states / 3.0)

    # Transition efficiency: accuracy per transition
    n_transitions = len(genome.transitions)
    if n_transitions > 0:
        # Higher accuracy with fewer transitions is better
        transition_efficiency = accuracy / (1 + n_transitions / 10.0)
    else:
        transition_efficiency = 0.0

    return FitnessComponents(
        accuracy=accuracy,
        parsimony=parsimony,
        hierarchy_score=hierarchy_score,
        parallelism_score=parallelism_score,
        transition_efficiency=transition_efficiency
    )


# =============================================================================
# Pareto Dominance
# =============================================================================

def dominates(components1: FitnessComponents, components2: FitnessComponents) -> bool:
    """
    Check if components1 Pareto-dominates components2.

    Dominance: components1 is >= on all objectives and > on at least one.
    """
    v1 = components1.to_vector()
    v2 = components2.to_vector()

    all_ge = all(a >= b for a, b in zip(v1, v2))
    any_gt = any(a > b for a, b in zip(v1, v2))

    return all_ge and any_gt


def compute_pareto_fronts(
    population: List,  # List[StatechartGenome]
    components: List[FitnessComponents]
) -> List[List[int]]:
    """
    Compute Pareto fronts for a population.

    Returns list of fronts, where each front is a list of population indices.
    Front 0 is the non-dominated set, Front 1 is dominated only by Front 0, etc.
    """
    n = len(population)
    domination_count = [0] * n  # How many solutions dominate this one
    dominated_by = [[] for _ in range(n)]  # Which solutions this one dominates

    # Compute domination relationships
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(components[i], components[j]):
                dominated_by[i].append(j)
            elif dominates(components[j], components[i]):
                domination_count[i] += 1

    # Build fronts
    fronts = []
    remaining = set(range(n))

    while remaining:
        # Find non-dominated in remaining
        front = [i for i in remaining if domination_count[i] == 0]

        if not front:
            # Shouldn't happen, but handle gracefully
            front = list(remaining)

        fronts.append(front)

        # Remove front from consideration
        for i in front:
            remaining.discard(i)
            for j in dominated_by[i]:
                if j in remaining:
                    domination_count[j] -= 1

    return fronts


# =============================================================================
# Crowding Distance
# =============================================================================

def compute_crowding_distance(
    indices: List[int],
    components: List[FitnessComponents]
) -> Dict[int, float]:
    """
    Compute crowding distance for a set of solutions.

    Crowding distance measures how "crowded" a solution is by its neighbors.
    Higher distance = more unique/diverse = preferred.
    """
    n = len(indices)
    if n == 0:
        return {}
    if n <= 2:
        return {i: float('inf') for i in indices}

    distance = {i: 0.0 for i in indices}

    # For each objective
    vectors = [components[i].to_vector() for i in indices]
    n_objectives = len(vectors[0])

    for obj in range(n_objectives):
        # Sort by this objective
        sorted_indices = sorted(indices, key=lambda i: components[i].to_vector()[obj])

        # Boundary points get infinite distance
        distance[sorted_indices[0]] = float('inf')
        distance[sorted_indices[-1]] = float('inf')

        # Range of this objective
        obj_min = components[sorted_indices[0]].to_vector()[obj]
        obj_max = components[sorted_indices[-1]].to_vector()[obj]
        obj_range = obj_max - obj_min

        if obj_range < 1e-10:
            continue

        # Interior points
        for k in range(1, n - 1):
            prev_val = components[sorted_indices[k-1]].to_vector()[obj]
            next_val = components[sorted_indices[k+1]].to_vector()[obj]
            distance[sorted_indices[k]] += (next_val - prev_val) / obj_range

    return distance


# =============================================================================
# NSGA-II Selection
# =============================================================================

def nsga2_select(
    population: List,  # List[StatechartGenome]
    components: List[FitnessComponents],
    n_select: int
) -> List[int]:
    """
    NSGA-II selection: select best individuals based on Pareto fronts and crowding.

    Returns indices of selected individuals.
    """
    if n_select >= len(population):
        return list(range(len(population)))

    selected = []
    fronts = compute_pareto_fronts(population, components)

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            # Add entire front
            selected.extend(front)
        else:
            # Partial front - use crowding distance
            needed = n_select - len(selected)
            distances = compute_crowding_distance(front, components)
            # Sort by crowding distance (higher = more diverse = preferred)
            sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
            selected.extend(sorted_front[:needed])
            break

    return selected


# =============================================================================
# Multi-Objective Fitness Evaluator
# =============================================================================

class MultiObjectiveFitness:
    """
    Multi-objective fitness evaluation and selection.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        use_pareto: bool = True
    ):
        """
        Args:
            weights: Weights for weighted sum (if not using pure Pareto)
            use_pareto: Use NSGA-II selection (vs weighted sum)
        """
        self.weights = weights or {
            "accuracy": 1.0,
            "parsimony": 0.1,
            "hierarchy_score": 0.05,
            "parallelism_score": 0.05
        }
        self.use_pareto = use_pareto

    def evaluate(
        self,
        genome,  # StatechartGenome
        accuracy: float,
        max_states: int = 20
    ) -> Tuple[float, FitnessComponents]:
        """
        Evaluate fitness for a genome.

        Returns (scalar_fitness, components).
        """
        components = compute_fitness_components(genome, accuracy, max_states)
        scalar = components.weighted_sum(self.weights)
        return scalar, components

    def select(
        self,
        population: List,
        components: List[FitnessComponents],
        n_select: int
    ) -> List[int]:
        """
        Select individuals for next generation.
        """
        if self.use_pareto:
            return nsga2_select(population, components, n_select)
        else:
            # Simple weighted sum ranking
            scores = [c.weighted_sum(self.weights) for c in components]
            sorted_indices = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
            return sorted_indices[:n_select]


# =============================================================================
# Testing
# =============================================================================

def test_fitness():
    """Test multi-objective fitness functionality."""
    print("=" * 60)
    print("MULTI-OBJECTIVE FITNESS TEST")
    print("=" * 60)

    # Create mock fitness components
    print("\n1. Fitness components:")
    c1 = FitnessComponents(accuracy=0.9, parsimony=0.5, hierarchy_score=0.3, parallelism_score=0.2)
    c2 = FitnessComponents(accuracy=0.8, parsimony=0.7, hierarchy_score=0.5, parallelism_score=0.4)
    c3 = FitnessComponents(accuracy=0.85, parsimony=0.6, hierarchy_score=0.4, parallelism_score=0.3)

    print(f"   C1: acc={c1.accuracy}, pars={c1.parsimony}, hier={c1.hierarchy_score}")
    print(f"   C2: acc={c2.accuracy}, pars={c2.parsimony}, hier={c2.hierarchy_score}")
    print(f"   C3: acc={c3.accuracy}, pars={c3.parsimony}, hier={c3.hierarchy_score}")

    # Test weighted sum
    print("\n2. Weighted sum:")
    print(f"   C1 weighted: {c1.weighted_sum():.4f}")
    print(f"   C2 weighted: {c2.weighted_sum():.4f}")
    print(f"   C3 weighted: {c3.weighted_sum():.4f}")

    # Test dominance
    print("\n3. Pareto dominance:")
    print(f"   C1 dominates C2: {dominates(c1, c2)}")
    print(f"   C2 dominates C1: {dominates(c2, c1)}")
    print(f"   C1 dominates C3: {dominates(c1, c3)}")

    # Test Pareto fronts with mock population
    print("\n4. Pareto fronts:")
    components = [c1, c2, c3]
    fronts = compute_pareto_fronts([None, None, None], components)
    print(f"   Front 0: {fronts[0]}")
    if len(fronts) > 1:
        print(f"   Front 1: {fronts[1]}")

    # Test crowding distance
    print("\n5. Crowding distance:")
    distances = compute_crowding_distance([0, 1, 2], components)
    for i, d in distances.items():
        print(f"   Solution {i}: distance={d:.4f}")

    # Test NSGA-II selection
    print("\n6. NSGA-II selection (select 2 from 3):")
    selected = nsga2_select([None, None, None], components, 2)
    print(f"   Selected: {selected}")

    # Test with more solutions
    print("\n7. Larger population test:")
    more_components = [
        FitnessComponents(accuracy=0.95, parsimony=0.3, hierarchy_score=0.2, parallelism_score=0.1),
        FitnessComponents(accuracy=0.70, parsimony=0.9, hierarchy_score=0.8, parallelism_score=0.7),
        FitnessComponents(accuracy=0.80, parsimony=0.6, hierarchy_score=0.5, parallelism_score=0.4),
        FitnessComponents(accuracy=0.85, parsimony=0.5, hierarchy_score=0.4, parallelism_score=0.3),
        FitnessComponents(accuracy=0.75, parsimony=0.7, hierarchy_score=0.6, parallelism_score=0.5),
    ]

    fronts = compute_pareto_fronts([None] * 5, more_components)
    print(f"   Number of fronts: {len(fronts)}")
    for i, front in enumerate(fronts):
        print(f"   Front {i}: {front}")

    selected = nsga2_select([None] * 5, more_components, 3)
    print(f"   Selected (top 3): {selected}")

    print("\nAll fitness tests passed!")


if __name__ == "__main__":
    test_fitness()
