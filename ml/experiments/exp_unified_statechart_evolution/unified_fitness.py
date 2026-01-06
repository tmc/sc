"""
Unified Fitness Evaluation with NSGA-II

Multi-objective optimization for statecharts:
1. CORRECTNESS: Accuracy on test scenarios
2. MINIMALITY: Fewer states, transitions, actions
3. INTERPRETABILITY: Structure quality, naming clarity

NSGA-II selection finds the Pareto front of solutions
trading off these objectives.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .unified_genome import UnifiedGenome, StateType, HistoryType, GuardOp
from .unified_executor import execute_scenarios


# =============================================================================
# Fitness Components
# =============================================================================

@dataclass
class UnifiedFitnessComponents:
    """
    Multi-objective fitness components.

    All normalized to [0, 1] where higher is better.
    """
    # Primary objectives
    correctness: float = 0.0     # Accuracy on scenarios
    minimality: float = 0.0      # Inverse of complexity
    interpretability: float = 0.0  # Structure quality

    # Sub-components (for analysis)
    accuracy: float = 0.0
    state_parsimony: float = 0.0
    transition_parsimony: float = 0.0
    action_parsimony: float = 0.0
    hierarchy_score: float = 0.0
    history_usage: float = 0.0
    guard_clarity: float = 0.0

    def to_vector(self) -> Tuple[float, float, float]:
        """Primary objectives as tuple for Pareto comparison."""
        return (self.correctness, self.minimality, self.interpretability)

    def weighted_sum(self, weights: Dict[str, float] = None) -> float:
        """Combine objectives with weights."""
        weights = weights or {
            'correctness': 1.0,
            'minimality': 0.2,
            'interpretability': 0.1
        }
        return (
            self.correctness * weights.get('correctness', 1.0) +
            self.minimality * weights.get('minimality', 0.0) +
            self.interpretability * weights.get('interpretability', 0.0)
        )


# =============================================================================
# Fitness Computation
# =============================================================================

def compute_unified_fitness(
    genome: UnifiedGenome,
    scenarios: List[Tuple[List[int], int]],
    max_states: int = 20,
    max_transitions: int = 50
) -> UnifiedFitnessComponents:
    """
    Compute all fitness components for a genome.

    Args:
        genome: The unified genome to evaluate
        scenarios: Test scenarios as (events, expected_state) pairs
        max_states: Maximum states for normalization
        max_transitions: Maximum transitions for normalization

    Returns:
        UnifiedFitnessComponents with all objectives
    """
    # === CORRECTNESS ===
    accuracy, _ = execute_scenarios(genome, scenarios)
    correctness = accuracy

    # === MINIMALITY ===
    # State parsimony
    state_parsimony = 1.0 - (genome.n_states / max_states)
    state_parsimony = max(0.0, state_parsimony)

    # Transition parsimony
    n_trans = len(genome.transitions)
    transition_parsimony = 1.0 - (n_trans / max_transitions)
    transition_parsimony = max(0.0, transition_parsimony)

    # Action parsimony
    n_actions = sum(len(t.actions) for t in genome.transitions)
    max_actions = max_transitions * 2
    action_parsimony = 1.0 - (n_actions / max_actions)
    action_parsimony = max(0.0, action_parsimony)

    # Combined minimality
    minimality = (state_parsimony * 0.4 +
                  transition_parsimony * 0.4 +
                  action_parsimony * 0.2)

    # === INTERPRETABILITY ===

    # Hierarchy score: deeper = more structured
    max_depth = genome.get_max_depth()
    hierarchy_score = min(1.0, max_depth / 4.0)

    # History usage: appropriate use of history
    n_composite = sum(1 for i in range(genome.n_states) if genome.is_composite(i))
    n_history = sum(1 for h in genome.history_type if h != HistoryType.NONE)
    if n_composite > 0:
        history_usage = n_history / n_composite
    else:
        history_usage = 0.0

    # Guard clarity: specific guards (not all TRUE)
    n_guards = sum(1 for t in genome.transitions if t.guard.op != GuardOp.TRUE)
    n_trans = len(genome.transitions)
    if n_trans > 0:
        guard_ratio = n_guards / n_trans
        # Sweet spot: ~30-50% of transitions should have guards
        guard_clarity = 1.0 - abs(guard_ratio - 0.4) * 2
        guard_clarity = max(0.0, guard_clarity)
    else:
        guard_clarity = 0.0

    # Combined interpretability
    interpretability = (
        hierarchy_score * 0.3 +
        history_usage * 0.3 +
        guard_clarity * 0.4
    )

    return UnifiedFitnessComponents(
        correctness=correctness,
        minimality=minimality,
        interpretability=interpretability,
        accuracy=accuracy,
        state_parsimony=state_parsimony,
        transition_parsimony=transition_parsimony,
        action_parsimony=action_parsimony,
        hierarchy_score=hierarchy_score,
        history_usage=history_usage,
        guard_clarity=guard_clarity
    )


# =============================================================================
# NSGA-II Selection
# =============================================================================

def dominates(c1: UnifiedFitnessComponents, c2: UnifiedFitnessComponents) -> bool:
    """
    Check if c1 Pareto-dominates c2.

    Dominance: c1 is >= on all objectives and > on at least one.
    """
    v1 = c1.to_vector()
    v2 = c2.to_vector()

    all_ge = all(a >= b for a, b in zip(v1, v2))
    any_gt = any(a > b for a, b in zip(v1, v2))

    return all_ge and any_gt


def compute_pareto_fronts(
    components_list: List[UnifiedFitnessComponents]
) -> List[List[int]]:
    """
    Compute non-dominated fronts using NSGA-II.

    Returns list of fronts, each front is list of indices.
    Front 0 = Pareto front (best), Front 1 = second best, etc.
    """
    n = len(components_list)
    if n == 0:
        return []

    # domination_count[i] = number of solutions dominating i
    domination_count = [0] * n
    # dominated_by[i] = solutions dominated by i
    dominated_by = [[] for _ in range(n)]

    # Compute domination relationships
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(components_list[i], components_list[j]):
                dominated_by[i].append(j)
            elif dominates(components_list[j], components_list[i]):
                domination_count[i] += 1

    # Build fronts
    fronts = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []

        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front = next_front

    return fronts


def crowding_distance(
    indices: List[int],
    components_list: List[UnifiedFitnessComponents]
) -> Dict[int, float]:
    """
    Compute crowding distance for solutions in a front.

    Higher distance = more isolated = more diverse = prefer to keep.
    """
    if len(indices) <= 2:
        return {i: float('inf') for i in indices}

    distance = {i: 0.0 for i in indices}
    n_objectives = 3  # correctness, minimality, interpretability

    for m in range(n_objectives):
        # Sort by objective m
        sorted_indices = sorted(indices, key=lambda i: components_list[i].to_vector()[m])

        # Boundary solutions get infinite distance
        distance[sorted_indices[0]] = float('inf')
        distance[sorted_indices[-1]] = float('inf')

        # Range for normalization
        f_max = components_list[sorted_indices[-1]].to_vector()[m]
        f_min = components_list[sorted_indices[0]].to_vector()[m]

        if f_max - f_min < 1e-10:
            continue

        # Interior solutions
        for i in range(1, len(sorted_indices) - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]

            f_prev = components_list[prev_idx].to_vector()[m]
            f_next = components_list[next_idx].to_vector()[m]

            distance[idx] += (f_next - f_prev) / (f_max - f_min)

    return distance


def nsga2_select(
    population: List[UnifiedGenome],
    components_list: List[UnifiedFitnessComponents],
    n_select: int
) -> List[int]:
    """
    NSGA-II selection: select top solutions based on Pareto ranking and crowding.

    Args:
        population: List of genomes
        components_list: Fitness components for each genome
        n_select: Number to select

    Returns:
        Indices of selected individuals
    """
    if n_select >= len(population):
        return list(range(len(population)))

    # Compute Pareto fronts
    fronts = compute_pareto_fronts(components_list)

    selected = []

    # Add complete fronts until we'd exceed n_select
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            # Need to select some from this front based on crowding
            remaining = n_select - len(selected)
            distances = crowding_distance(front, components_list)

            # Sort by crowding distance (descending) and take top remaining
            sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    return selected


def tournament_select_nsga2(
    population: List[UnifiedGenome],
    components_list: List[UnifiedFitnessComponents],
    n_parents: int,
    tournament_size: int = 3
) -> List[int]:
    """
    Tournament selection using NSGA-II dominance and crowding.
    """
    selected = []
    fronts = compute_pareto_fronts(components_list)

    # Create rank lookup
    rank = {}
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r

    # Compute crowding for all
    all_crowding = {}
    for front in fronts:
        all_crowding.update(crowding_distance(front, components_list))

    for _ in range(n_parents):
        candidates = random.sample(range(len(population)), min(tournament_size, len(population)))

        # Compare by rank first, then crowding
        def compare_key(i):
            return (rank[i], -all_crowding.get(i, 0))

        winner = min(candidates, key=compare_key)
        selected.append(winner)

    return selected


# =============================================================================
# Testing
# =============================================================================

def test_nsga2():
    """Test NSGA-II selection."""
    print("=" * 60)
    print("NSGA-II SELECTION TEST")
    print("=" * 60)

    # Create mock fitness components
    components = [
        UnifiedFitnessComponents(correctness=0.9, minimality=0.3, interpretability=0.5),
        UnifiedFitnessComponents(correctness=0.8, minimality=0.7, interpretability=0.4),
        UnifiedFitnessComponents(correctness=0.7, minimality=0.9, interpretability=0.6),
        UnifiedFitnessComponents(correctness=0.85, minimality=0.5, interpretability=0.8),
        UnifiedFitnessComponents(correctness=0.6, minimality=0.4, interpretability=0.7),
    ]

    # Test Pareto fronts
    fronts = compute_pareto_fronts(components)
    print(f"Pareto fronts: {fronts}")

    # Test selection
    selected = nsga2_select(None, components, 3)
    print(f"Selected (top 3): {selected}")

    # Check dominance
    print("\nDominance matrix:")
    for i in range(len(components)):
        row = []
        for j in range(len(components)):
            if i == j:
                row.append("-")
            elif dominates(components[i], components[j]):
                row.append("1")
            else:
                row.append("0")
        print(f"  {i}: {' '.join(row)}")

    print("\nNSGA-II test complete!")


if __name__ == "__main__":
    test_nsga2()
