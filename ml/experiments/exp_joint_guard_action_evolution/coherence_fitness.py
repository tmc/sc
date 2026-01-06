"""
Coherence Fitness for Joint Guard-Action Evolution

Measures how well guards and actions work together:
1. CAUSAL COHERENCE: Guards read what actions write
2. TEMPORAL COHERENCE: Cause precedes effect in execution
3. BEHAVIORAL COHERENCE: Transitions produce correct behavior
4. STRUCTURAL COHERENCE: No dead guards or useless actions

The key insight: random guards and actions are incoherent.
Evolution should favor coherent pairs where:
  action(t).write(X) enables guard(t+1).read(X)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

from .joint_genome import JointGenome, GuardActionPair, Guard, Action
from .causal_graph import build_causal_graph, analyze_causal_graph, CausalAnalysis


# =============================================================================
# Fitness Components
# =============================================================================

@dataclass
class CoherenceFitness:
    """Multi-objective fitness for guard-action coherence."""

    # Primary objectives
    causal_coherence: float = 0.0      # Guards read what actions write
    temporal_coherence: float = 0.0     # Execution order respects causality
    behavioral_coherence: float = 0.0   # Correct behavior on scenarios
    structural_coherence: float = 0.0   # No dead code

    # Sub-components
    variable_coverage: float = 0.0     # Fraction of guard vars written
    dependency_satisfaction: float = 0.0  # Fraction of dependencies met
    guard_satisfiability: float = 0.0   # Fraction of guards that can fire
    action_usefulness: float = 0.0      # Fraction of actions that matter
    scenario_accuracy: float = 0.0      # Correct on test scenarios

    # Penalties
    dead_guard_penalty: float = 0.0    # Guards that can never fire
    unused_action_penalty: float = 0.0  # Actions with no observable effect
    inconsistency_penalty: float = 0.0  # Contradictory guard-action pairs

    def total_fitness(self, weights: Dict[str, float] = None) -> float:
        """Weighted sum of objectives."""
        weights = weights or {
            'causal_coherence': 0.3,
            'temporal_coherence': 0.2,
            'behavioral_coherence': 0.3,
            'structural_coherence': 0.2
        }

        base = (
            self.causal_coherence * weights.get('causal_coherence', 0.3) +
            self.temporal_coherence * weights.get('temporal_coherence', 0.2) +
            self.behavioral_coherence * weights.get('behavioral_coherence', 0.3) +
            self.structural_coherence * weights.get('structural_coherence', 0.2)
        )

        penalties = (self.dead_guard_penalty +
                     self.unused_action_penalty +
                     self.inconsistency_penalty) * 0.1

        return max(0.0, base - penalties)

    def to_vector(self) -> Tuple[float, float, float, float]:
        """For Pareto comparison."""
        return (self.causal_coherence, self.temporal_coherence,
                self.behavioral_coherence, self.structural_coherence)


# =============================================================================
# Causal Coherence
# =============================================================================

def compute_causal_coherence(
    genome: JointGenome,
    analysis: CausalAnalysis
) -> Tuple[float, float, float]:
    """Compute causal coherence metrics.

    Returns: (variable_coverage, dependency_satisfaction, causal_coherence)
    """
    # Variable coverage: what fraction of guard-read variables are action-written?
    guard_vars = genome.all_variables_read()
    action_vars = genome.all_variables_written()

    if not guard_vars:
        variable_coverage = 1.0
    else:
        covered = guard_vars & action_vars
        variable_coverage = len(covered) / len(guard_vars)

    # Dependency satisfaction: what fraction of pairs have incoming edges?
    dependency_satisfaction = analysis.causal_coverage

    # Combined causal coherence
    causal_coherence = (variable_coverage * 0.5 + dependency_satisfaction * 0.5)

    return variable_coverage, dependency_satisfaction, causal_coherence


# =============================================================================
# Temporal Coherence
# =============================================================================

def compute_temporal_coherence(
    genome: JointGenome,
    execution_traces: List[List[int]] = None
) -> float:
    """Compute temporal coherence: cause precedes effect in execution.

    A genome is temporally coherent if:
    - When guard G reads X, there exists a prior action A that wrote X
    - The action A is reachable before G in state space

    This requires actual execution traces to measure properly.
    Without traces, we estimate based on state ordering.
    """
    if not genome.pairs:
        return 1.0

    # Simple heuristic: pairs from lower states should write
    # variables that pairs from higher states read

    coherent_pairs = 0
    total_pairs = 0

    for i, pair_i in enumerate(genome.pairs):
        for j, pair_j in enumerate(genome.pairs):
            if i == j:
                continue

            # Does pair_i write what pair_j reads?
            writes_i = pair_i.action_writes()
            reads_j = pair_j.guard_reads()
            shared = writes_i & reads_j

            if shared:
                total_pairs += 1
                # Temporal coherence: writer should be from lower/equal state
                if pair_i.source_state <= pair_j.source_state:
                    coherent_pairs += 1

    if total_pairs == 0:
        return 0.5  # No dependencies = neutral

    return coherent_pairs / total_pairs


def compute_temporal_coherence_from_traces(
    genome: JointGenome,
    traces: List[List[Tuple[int, Dict[str, Any]]]]  # [(pair_idx, context)]
) -> float:
    """Compute temporal coherence from actual execution traces.

    Checks: when pair fires, were its guard variables set by prior actions?
    """
    if not traces:
        return compute_temporal_coherence(genome)

    coherent = 0
    total = 0

    for trace in traces:
        written_vars: Set[str] = set()

        for step_idx, (pair_idx, context) in enumerate(trace):
            pair = genome.pairs[pair_idx]
            guard_vars = pair.guard_reads()

            if guard_vars:
                total += 1
                # Were all guard vars written by prior actions?
                if guard_vars <= written_vars:
                    coherent += 1

            # Update written vars
            written_vars |= pair.action_writes()

    if total == 0:
        return 0.5

    return coherent / total


# =============================================================================
# Behavioral Coherence
# =============================================================================

@dataclass
class Scenario:
    """Test scenario for behavioral coherence."""
    initial_context: Dict[str, Any]
    events: List[int]
    expected_final_context: Dict[str, Any]
    expected_states: Optional[List[int]] = None


def execute_scenario(
    genome: JointGenome,
    scenario: Scenario
) -> Tuple[bool, Dict[str, Any], List[int]]:
    """Execute scenario and check correctness.

    Returns: (correct, final_context, state_trace)
    """
    context = dict(scenario.initial_context)
    current_state = 0
    state_trace = [current_state]

    for event in scenario.events:
        # Find enabled transition
        transitions = [p for p in genome.pairs
                       if p.source_state == current_state and p.event == event]

        # Sort by priority, take highest priority enabled
        transitions.sort(key=lambda p: -p.priority)

        fired = False
        for trans in transitions:
            if trans.is_enabled(context):
                context = trans.action.apply(context)
                current_state = trans.target_state
                fired = True
                break

        state_trace.append(current_state)

    # Check correctness
    correct = True
    for key, expected in scenario.expected_final_context.items():
        if context.get(key) != expected:
            correct = False
            break

    if scenario.expected_states is not None:
        if state_trace != scenario.expected_states:
            correct = False

    return correct, context, state_trace


def compute_behavioral_coherence(
    genome: JointGenome,
    scenarios: List[Scenario]
) -> Tuple[float, float]:
    """Compute behavioral coherence on test scenarios.

    Returns: (scenario_accuracy, behavioral_coherence)
    """
    if not scenarios:
        return 0.5, 0.5

    correct = 0
    for scenario in scenarios:
        is_correct, _, _ = execute_scenario(genome, scenario)
        if is_correct:
            correct += 1

    accuracy = correct / len(scenarios)
    return accuracy, accuracy


# =============================================================================
# Structural Coherence
# =============================================================================

def compute_structural_coherence(
    genome: JointGenome,
    analysis: CausalAnalysis
) -> Tuple[float, float, float, float]:
    """Compute structural coherence metrics.

    Returns: (guard_satisfiability, action_usefulness,
              dead_guard_penalty, unused_action_penalty)
    """
    n_pairs = len(genome.pairs)
    if n_pairs == 0:
        return 1.0, 1.0, 0.0, 0.0

    # Dead guards: guards that can never be satisfied
    # (we approximate: guards with no incoming causal edges)
    unsatisfied = len(analysis.unsatisfied_guards)
    guard_satisfiability = 1.0 - (unsatisfied / n_pairs)
    dead_guard_penalty = unsatisfied / n_pairs

    # Unused actions: actions that write to variables nothing reads
    unused = len(analysis.unused_actions)
    action_usefulness = 1.0 - (unused / n_pairs)
    unused_action_penalty = unused / n_pairs

    return guard_satisfiability, action_usefulness, dead_guard_penalty, unused_action_penalty


# =============================================================================
# Full Fitness Computation
# =============================================================================

def compute_coherence_fitness(
    genome: JointGenome,
    scenarios: List[Scenario] = None
) -> CoherenceFitness:
    """Compute all coherence fitness components."""

    # Build causal graph
    graph = build_causal_graph(genome)
    analysis = analyze_causal_graph(graph, genome)

    # Causal coherence
    var_cov, dep_sat, causal_coh = compute_causal_coherence(genome, analysis)

    # Temporal coherence
    temporal_coh = compute_temporal_coherence(genome)

    # Behavioral coherence
    if scenarios:
        scenario_acc, behavioral_coh = compute_behavioral_coherence(genome, scenarios)
    else:
        scenario_acc, behavioral_coh = 0.5, 0.5

    # Structural coherence
    guard_sat, action_use, dead_pen, unused_pen = compute_structural_coherence(
        genome, analysis)

    structural_coh = (guard_sat + action_use) / 2

    # Check for inconsistencies
    inconsistency_pen = compute_inconsistency_penalty(genome)

    return CoherenceFitness(
        causal_coherence=causal_coh,
        temporal_coherence=temporal_coh,
        behavioral_coherence=behavioral_coh,
        structural_coherence=structural_coh,
        variable_coverage=var_cov,
        dependency_satisfaction=dep_sat,
        guard_satisfiability=guard_sat,
        action_usefulness=action_use,
        scenario_accuracy=scenario_acc,
        dead_guard_penalty=dead_pen,
        unused_action_penalty=unused_pen,
        inconsistency_penalty=inconsistency_pen
    )


def compute_inconsistency_penalty(genome: JointGenome) -> float:
    """Detect guard-action inconsistencies.

    An inconsistency: guard checks X == v, action sets X := not v.
    """
    inconsistencies = 0
    total_checks = 0

    for pair in genome.pairs:
        for clause in pair.guard.clauses:
            for effect in pair.action.effects:
                if clause.variable == effect.variable:
                    total_checks += 1
                    # Simple check: SET to different value than guard expects
                    if effect.effect_op.name == "SET":
                        if clause.op.name == "EQ" and effect.value != clause.value:
                            inconsistencies += 1

    if total_checks == 0:
        return 0.0

    return inconsistencies / total_checks


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate coherence fitness."""
    print("=" * 60)
    print("COHERENCE FITNESS DEMO")
    print("=" * 60)

    from .joint_genome import random_joint_genome

    variables = ["turn", "phase", "score", "ready"]
    possible_values = {
        "turn": [0, 1, 2],
        "phase": ["init", "play", "end"],
        "score": [0, 1, 2, 3],
        "ready": [True, False]
    }

    # Generate genome
    genome = random_joint_genome(
        n_states=4,
        n_events=3,
        n_pairs=8,
        variables=variables,
        possible_values=possible_values
    )

    # Create test scenarios
    scenarios = [
        Scenario(
            initial_context={"turn": 0, "phase": "init", "score": 0, "ready": False},
            events=[0, 1, 2],
            expected_final_context={"phase": "play"}  # Just check phase
        ),
        Scenario(
            initial_context={"turn": 1, "phase": "play", "score": 2, "ready": True},
            events=[1, 1, 0],
            expected_final_context={"score": 2}  # Score unchanged
        ),
    ]

    # Compute fitness
    fitness = compute_coherence_fitness(genome, scenarios)

    print(f"\n--- Fitness Components ---")
    print(f"Causal coherence:     {fitness.causal_coherence:.3f}")
    print(f"  Variable coverage:  {fitness.variable_coverage:.3f}")
    print(f"  Dependency sat.:    {fitness.dependency_satisfaction:.3f}")
    print(f"Temporal coherence:   {fitness.temporal_coherence:.3f}")
    print(f"Behavioral coherence: {fitness.behavioral_coherence:.3f}")
    print(f"  Scenario accuracy:  {fitness.scenario_accuracy:.3f}")
    print(f"Structural coherence: {fitness.structural_coherence:.3f}")
    print(f"  Guard satisfiability: {fitness.guard_satisfiability:.3f}")
    print(f"  Action usefulness:    {fitness.action_usefulness:.3f}")

    print(f"\n--- Penalties ---")
    print(f"Dead guard penalty:   {fitness.dead_guard_penalty:.3f}")
    print(f"Unused action penalty: {fitness.unused_action_penalty:.3f}")
    print(f"Inconsistency penalty: {fitness.inconsistency_penalty:.3f}")

    print(f"\n--- Total Fitness ---")
    print(f"Total: {fitness.total_fitness():.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
