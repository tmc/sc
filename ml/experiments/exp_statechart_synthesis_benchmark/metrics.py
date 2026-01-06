"""
Metrics for Statechart Synthesis Benchmark

Three categories of metrics:
1. ACCURACY: How well does the synthesized statechart match expected behavior?
2. EFFICIENCY: How fast and resource-efficient is synthesis?
3. INTERPRETABILITY: How human-readable is the result?

Each metric is normalized to [0, 1] for fair comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum, auto
import math
import time

from .synthesis_methods import SynthesizedStatechart, SynthesisResult
from .domain_suite import Domain, DomainExample


# =============================================================================
# Metric Categories
# =============================================================================

class MetricCategory(Enum):
    """Categories of metrics."""
    ACCURACY = auto()
    EFFICIENCY = auto()
    INTERPRETABILITY = auto()


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    name: str
    category: MetricCategory
    value: float          # Normalized to [0, 1]
    raw_value: Any        # Original value
    description: str = ""


# =============================================================================
# Accuracy Metrics
# =============================================================================

def compute_accuracy(
    statechart: SynthesizedStatechart,
    examples: List[DomainExample]
) -> float:
    """Compute raw accuracy on examples."""
    if not examples or not statechart:
        return 0.0

    correct = 0
    for ex in examples:
        predicted = simulate_statechart(statechart, ex.events)
        if predicted == ex.final_state:
            correct += 1
        # Also accept partial matches
        elif ex.final_state in predicted or predicted in ex.final_state:
            correct += 0.5

    return correct / len(examples)


def simulate_statechart(
    statechart: SynthesizedStatechart,
    events: List[str]
) -> str:
    """Simulate statechart on event sequence."""
    state = statechart.initial_state
    for event in events:
        for src, evt, tgt in statechart.transitions:
            if src == state and evt == event:
                state = tgt
                break
    return state


def compute_precision_recall_f1(
    statechart: SynthesizedStatechart,
    examples: List[DomainExample]
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for state predictions."""
    if not examples or not statechart:
        return 0.0, 0.0, 0.0

    # Collect predictions and actuals
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    predicted_states: Set[str] = set()
    actual_states: Set[str] = set()

    for ex in examples:
        predicted = simulate_statechart(statechart, ex.events)
        actual = ex.final_state

        predicted_states.add(predicted)
        actual_states.add(actual)

        if predicted == actual:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_state_coverage(
    statechart: SynthesizedStatechart,
    domain: Domain
) -> float:
    """How many domain states are represented in synthesized statechart?"""
    if not domain.state_names or not statechart:
        return 0.0

    covered = 0
    for domain_state in domain.state_names:
        for sc_state in statechart.states:
            if domain_state in sc_state or sc_state in domain_state:
                covered += 1
                break

    return covered / len(domain.state_names)


def compute_transition_coverage(
    statechart: SynthesizedStatechart,
    examples: List[DomainExample]
) -> float:
    """How many transitions are actually used on examples?"""
    if not statechart or not examples:
        return 0.0

    used_transitions: Set[Tuple[str, str, str]] = set()

    for ex in examples:
        state = statechart.initial_state
        for event in ex.events:
            for src, evt, tgt in statechart.transitions:
                if src == state and evt == event:
                    used_transitions.add((src, evt, tgt))
                    state = tgt
                    break

    if not statechart.transitions:
        return 0.0

    return len(used_transitions) / len(statechart.transitions)


# =============================================================================
# Efficiency Metrics
# =============================================================================

def compute_synthesis_speed(
    result: SynthesisResult,
    n_examples: int
) -> float:
    """Compute examples processed per second."""
    if result.synthesis_time <= 0 or n_examples <= 0:
        return 0.0
    return n_examples / result.synthesis_time


def compute_iteration_efficiency(
    result: SynthesisResult
) -> float:
    """Compute accuracy gain per iteration."""
    if result.iterations <= 0:
        return 0.0
    return result.accuracy / result.iterations


def compute_size_efficiency(
    statechart: SynthesizedStatechart,
    domain: Domain
) -> float:
    """Compute how efficiently states are used (vs domain size)."""
    if not statechart or not domain.state_names:
        return 0.0

    optimal_size = len(domain.state_names)
    actual_size = statechart.n_states

    if actual_size <= optimal_size:
        return 1.0
    else:
        # Penalize excess states
        return optimal_size / actual_size


def normalize_time(time_seconds: float, max_time: float = 60.0) -> float:
    """Normalize time to [0, 1] where lower is better."""
    if time_seconds <= 0:
        return 1.0
    if time_seconds >= max_time:
        return 0.0
    return 1.0 - (time_seconds / max_time)


# =============================================================================
# Interpretability Metrics
# =============================================================================

def compute_hierarchy_depth(statechart: SynthesizedStatechart) -> int:
    """Compute depth of state hierarchy (if present)."""
    # Basic statecharts are flat
    # Check for hierarchical naming (e.g., "parent.child")
    max_depth = 1
    for state in statechart.states:
        depth = state.count('.') + 1
        max_depth = max(max_depth, depth)
    return max_depth


def compute_naming_quality(
    statechart: SynthesizedStatechart,
    domain: Domain
) -> float:
    """Assess quality of state names (meaningful vs generic)."""
    if not statechart or not statechart.states:
        return 0.0

    meaningful_names = 0
    for state in statechart.states:
        # Check if name is meaningful (not just S0, S1, etc.)
        if not state[0].isdigit() and not state.startswith('S') and len(state) > 2:
            meaningful_names += 1
        # Check if matches domain vocabulary
        for domain_state in domain.state_names:
            if domain_state.lower() in state.lower() or state.lower() in domain_state.lower():
                meaningful_names += 0.5
                break

    return min(1.0, meaningful_names / len(statechart.states))


def compute_transition_clarity(statechart: SynthesizedStatechart) -> float:
    """Assess clarity of transitions (distinct events, no ambiguity)."""
    if not statechart or not statechart.transitions:
        return 0.0

    # Check for ambiguous transitions (same src, event, different targets)
    transition_map: Dict[Tuple[str, str], Set[str]] = {}
    for src, evt, tgt in statechart.transitions:
        key = (src, evt)
        if key not in transition_map:
            transition_map[key] = set()
        transition_map[key].add(tgt)

    ambiguous = sum(1 for targets in transition_map.values() if len(targets) > 1)
    total = len(transition_map)

    if total == 0:
        return 0.0

    return 1.0 - (ambiguous / total)


def compute_minimality(
    statechart: SynthesizedStatechart,
    max_states: int = 20,
    max_transitions: int = 50
) -> float:
    """Penalize overly complex statecharts."""
    if not statechart:
        return 0.0

    state_penalty = statechart.n_states / max_states
    trans_penalty = statechart.n_transitions / max_transitions

    # Lower complexity = higher score
    return max(0.0, 1.0 - (state_penalty + trans_penalty) / 2)


# =============================================================================
# Comprehensive Metrics
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Complete set of benchmark metrics."""

    # Accuracy
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    state_coverage: float = 0.0
    transition_coverage: float = 0.0

    # Efficiency
    synthesis_time: float = 0.0
    iterations: int = 0
    examples_per_second: float = 0.0
    time_normalized: float = 0.0
    size_efficiency: float = 0.0

    # Interpretability
    hierarchy_depth: int = 1
    naming_quality: float = 0.0
    transition_clarity: float = 0.0
    minimality: float = 0.0

    def accuracy_score(self) -> float:
        """Aggregate accuracy score."""
        return (self.accuracy * 0.4 +
                self.f1_score * 0.3 +
                self.state_coverage * 0.15 +
                self.transition_coverage * 0.15)

    def efficiency_score(self) -> float:
        """Aggregate efficiency score."""
        return (self.time_normalized * 0.5 +
                self.size_efficiency * 0.3 +
                min(1.0, self.examples_per_second / 100) * 0.2)

    def interpretability_score(self) -> float:
        """Aggregate interpretability score."""
        return (self.naming_quality * 0.3 +
                self.transition_clarity * 0.3 +
                self.minimality * 0.4)

    def overall_score(self, weights: Dict[str, float] = None) -> float:
        """Compute overall weighted score."""
        weights = weights or {
            'accuracy': 0.5,
            'efficiency': 0.25,
            'interpretability': 0.25
        }
        return (self.accuracy_score() * weights.get('accuracy', 0.5) +
                self.efficiency_score() * weights.get('efficiency', 0.25) +
                self.interpretability_score() * weights.get('interpretability', 0.25))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'state_coverage': self.state_coverage,
                'transition_coverage': self.transition_coverage,
                'score': self.accuracy_score()
            },
            'efficiency': {
                'synthesis_time': self.synthesis_time,
                'iterations': self.iterations,
                'examples_per_second': self.examples_per_second,
                'time_normalized': self.time_normalized,
                'size_efficiency': self.size_efficiency,
                'score': self.efficiency_score()
            },
            'interpretability': {
                'hierarchy_depth': self.hierarchy_depth,
                'naming_quality': self.naming_quality,
                'transition_clarity': self.transition_clarity,
                'minimality': self.minimality,
                'score': self.interpretability_score()
            },
            'overall': self.overall_score()
        }


def compute_all_metrics(
    result: SynthesisResult,
    domain: Domain
) -> BenchmarkMetrics:
    """Compute all metrics for a synthesis result."""

    if not result.statechart:
        return BenchmarkMetrics(synthesis_time=result.synthesis_time)

    sc = result.statechart
    all_examples = domain.train_examples + domain.test_examples

    # Accuracy metrics
    accuracy = compute_accuracy(sc, all_examples)
    precision, recall, f1 = compute_precision_recall_f1(sc, all_examples)
    state_cov = compute_state_coverage(sc, domain)
    trans_cov = compute_transition_coverage(sc, all_examples)

    # Efficiency metrics
    n_examples = len(all_examples)
    examples_per_sec = compute_synthesis_speed(result, n_examples)
    time_norm = normalize_time(result.synthesis_time)
    size_eff = compute_size_efficiency(sc, domain)

    # Interpretability metrics
    hierarchy = compute_hierarchy_depth(sc)
    naming = compute_naming_quality(sc, domain)
    clarity = compute_transition_clarity(sc)
    minimality = compute_minimality(sc)

    return BenchmarkMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        state_coverage=state_cov,
        transition_coverage=trans_cov,
        synthesis_time=result.synthesis_time,
        iterations=result.iterations,
        examples_per_second=examples_per_sec,
        time_normalized=time_norm,
        size_efficiency=size_eff,
        hierarchy_depth=hierarchy,
        naming_quality=naming,
        transition_clarity=clarity,
        minimality=minimality
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate metrics computation."""
    print("=" * 60)
    print("METRICS DEMO")
    print("=" * 60)

    from .synthesis_methods import EvolutionaryMethod, SynthesisConfig
    from .domain_suite import create_tictactoe_domain

    domain = create_tictactoe_domain()
    config = SynthesisConfig(n_generations=50)
    method = EvolutionaryMethod(config)

    print(f"\nDomain: {domain.name}")
    print(f"Examples: {len(domain.train_examples)} train, {len(domain.test_examples)} test")

    result = method.synthesize(domain.get_train_pairs())
    metrics = compute_all_metrics(result, domain)

    print(f"\n--- Accuracy Metrics ---")
    print(f"Accuracy:           {metrics.accuracy:.1%}")
    print(f"Precision:          {metrics.precision:.1%}")
    print(f"Recall:             {metrics.recall:.1%}")
    print(f"F1 Score:           {metrics.f1_score:.1%}")
    print(f"State Coverage:     {metrics.state_coverage:.1%}")
    print(f"Transition Coverage:{metrics.transition_coverage:.1%}")
    print(f"Accuracy Score:     {metrics.accuracy_score():.3f}")

    print(f"\n--- Efficiency Metrics ---")
    print(f"Synthesis Time:     {metrics.synthesis_time:.3f}s")
    print(f"Iterations:         {metrics.iterations}")
    print(f"Examples/sec:       {metrics.examples_per_second:.1f}")
    print(f"Size Efficiency:    {metrics.size_efficiency:.1%}")
    print(f"Efficiency Score:   {metrics.efficiency_score():.3f}")

    print(f"\n--- Interpretability Metrics ---")
    print(f"Hierarchy Depth:    {metrics.hierarchy_depth}")
    print(f"Naming Quality:     {metrics.naming_quality:.1%}")
    print(f"Transition Clarity: {metrics.transition_clarity:.1%}")
    print(f"Minimality:         {metrics.minimality:.1%}")
    print(f"Interpret. Score:   {metrics.interpretability_score():.3f}")

    print(f"\n--- Overall ---")
    print(f"Overall Score:      {metrics.overall_score():.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
