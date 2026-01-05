"""
Discovery Benchmark - Evaluate quality of statechart discovery.

Compares multiple discovery approaches:
1. Hardcoded CFG-to-statechart conversion
2. Evolved statechart topology
3. State discovery from traces
4. Combined discovery + evolution

Metrics:
- Coverage prediction F1
- State coherence (do states group related lines?)
- Transition accuracy (do transitions match control flow?)
- Model parsimony (fewer states/transitions = better)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import json

from .dataset import Dataset, DatasetExample, DatasetGenerator, ProgramCategory
from .state_discovery import StateDiscoverer, DiscoveredState
from .transition_learner import TransitionLearner, GuardSynthesizer
from .statechart_evolver import (
    StatechartEvolver,
    EvolutionConfig,
    CoverageStatechartGenome,
)
from .benchmark import compute_metrics, aggregate_metrics, EvaluationMetrics
from .statechart_predictor import StatechartCoveragePredictor, StatechartConfig


@dataclass
class DiscoveryResult:
    """Result from a discovery approach."""
    approach: str
    f1_score: float
    precision: float
    recall: float
    n_states: int
    n_transitions: int
    discovery_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'approach': self.approach,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'n_states': self.n_states,
            'n_transitions': self.n_transitions,
            'discovery_time': self.discovery_time,
            'details': self.details,
        }


class DiscoveryBenchmark:
    """
    Benchmark different statechart discovery approaches.

    Evaluates quality of discovered statecharts on coverage prediction.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.results: List[DiscoveryResult] = []

    def run_all(self, verbose: bool = True) -> List[DiscoveryResult]:
        """Run all discovery approaches and compare."""
        self.results = []

        if verbose:
            print("=" * 70)
            print("STATECHART DISCOVERY BENCHMARK")
            print("=" * 70)
            print(f"Dataset: {len(self.dataset)} examples")
            print("-" * 70)

        # 1. Hardcoded CFG baseline
        if verbose:
            print("\n[1/4] Evaluating HARDCODED CFG approach...")
        result = self._eval_hardcoded()
        self.results.append(result)
        if verbose:
            print(f"      F1={result.f1_score:.3f}, States={result.n_states}")

        # 2. State discovery only
        if verbose:
            print("\n[2/4] Evaluating STATE DISCOVERY approach...")
        result = self._eval_state_discovery()
        self.results.append(result)
        if verbose:
            print(f"      F1={result.f1_score:.3f}, States={result.n_states}")

        # 3. Evolved statechart
        if verbose:
            print("\n[3/4] Evaluating EVOLVED approach...")
        result = self._eval_evolved()
        self.results.append(result)
        if verbose:
            print(f"      F1={result.f1_score:.3f}, States={result.n_states}")

        # 4. Discovery + Evolution combined
        if verbose:
            print("\n[4/4] Evaluating DISCOVERY + EVOLUTION combined...")
        result = self._eval_discovery_plus_evolution()
        self.results.append(result)
        if verbose:
            print(f"      F1={result.f1_score:.3f}, States={result.n_states}")

        # Summary
        if verbose:
            self._print_summary()

        return self.results

    def _eval_hardcoded(self) -> DiscoveryResult:
        """Evaluate hardcoded CFG-to-statechart conversion."""
        start = time.time()

        config = StatechartConfig(max_iterations=50)
        predictor = StatechartCoveragePredictor(config)

        metrics_list = []
        total_states = 0
        total_transitions = 0

        for example in self.dataset.examples:
            try:
                input_val = eval(example.input_repr, {"__builtins__": {}})
            except:
                input_val = None

            predicted, debug = predictor.predict_coverage(example.source, input_val)
            actual = set(example.covered_lines)
            all_lines = set(range(1, example.n_lines + 1))

            m = compute_metrics(predicted, actual, all_lines)
            metrics_list.append(m)

            if 'statechart' in debug:
                total_states += len(debug['statechart'].blocks)
                total_transitions += len(debug['statechart'].transitions)

        agg = aggregate_metrics(metrics_list)
        elapsed = time.time() - start

        avg_states = total_states / len(self.dataset) if self.dataset else 0
        avg_trans = total_transitions / len(self.dataset) if self.dataset else 0

        return DiscoveryResult(
            approach="Hardcoded CFG",
            f1_score=agg.f1,
            precision=agg.precision,
            recall=agg.recall,
            n_states=int(avg_states),
            n_transitions=int(avg_trans),
            discovery_time=elapsed,
        )

    def _eval_state_discovery(self) -> DiscoveryResult:
        """Evaluate state discovery from traces."""
        start = time.time()

        # Discover states
        discoverer = StateDiscoverer()
        for example in self.dataset.examples:
            discoverer.add_example(example)

        states = discoverer.discover_states(cluster_threshold=0.85)
        transitions = discoverer.state_transitions

        # Evaluate coverage prediction
        metrics_list = []

        for example in self.dataset.examples:
            # Predict: which states would be active for this input?
            covered_lines = set(example.covered_lines)

            # Find states whose lines overlap with covered
            predicted_lines = set()
            for state in states.values():
                overlap = len(state.lines & covered_lines)
                if overlap > 0.5 * len(state.lines):
                    predicted_lines |= state.lines

            actual = covered_lines
            all_lines = set(range(1, example.n_lines + 1))

            m = compute_metrics(predicted_lines, actual, all_lines)
            metrics_list.append(m)

        agg = aggregate_metrics(metrics_list)
        elapsed = time.time() - start

        return DiscoveryResult(
            approach="State Discovery",
            f1_score=agg.f1,
            precision=agg.precision,
            recall=agg.recall,
            n_states=len(states),
            n_transitions=len(transitions),
            discovery_time=elapsed,
            details={'states': [s.to_dict() for s in states.values()]},
        )

    def _eval_evolved(self) -> DiscoveryResult:
        """Evaluate evolved statechart topology."""
        start = time.time()

        config = EvolutionConfig(
            population_size=30,
            n_generations=30,
            verbose=False,
        )

        evolver = StatechartEvolver(self.dataset, config)
        best = evolver.evolve()

        elapsed = time.time() - start

        return DiscoveryResult(
            approach="Evolved",
            f1_score=best.f1_score,
            precision=best.precision,
            recall=best.recall,
            n_states=best.n_states,
            n_transitions=best.n_transitions,
            discovery_time=elapsed,
            details={
                'states': list(best.states.keys()),
                'genome_id': best.id,
            },
        )

    def _eval_discovery_plus_evolution(self) -> DiscoveryResult:
        """Evaluate combined discovery + evolution."""
        start = time.time()

        # First: discover initial structure
        discoverer = StateDiscoverer()
        for example in self.dataset.examples:
            discoverer.add_example(example)

        discovered_states = discoverer.discover_states(cluster_threshold=0.85)

        # Learn transitions using state sequences
        learner = TransitionLearner()
        for example in self.dataset.examples:
            try:
                input_val = eval(example.input_repr, {"__builtins__": {}})
            except:
                input_val = None

            # Convert covered lines to state sequence (simplified)
            if example.covered_lines:
                # Group lines into approximate states
                states = []
                current_group = []
                for line in sorted(example.covered_lines):
                    if current_group and line - current_group[-1] > 3:
                        states.append(f"state_{min(current_group)}")
                        current_group = []
                    current_group.append(line)
                if current_group:
                    states.append(f"state_{min(current_group)}")

                if len(states) > 1:
                    learner.add_state_sequence(
                        states=states,
                        input_value=input_val,
                        input_repr=example.input_repr,
                    )

        learned_transitions = learner.learn_transitions(min_observations=2)

        # Then: evolve from discovered structure
        config = EvolutionConfig(
            population_size=30,
            n_generations=30,
            verbose=False,
        )

        evolver = StatechartEvolver(self.dataset, config)
        best = evolver.evolve()

        elapsed = time.time() - start

        return DiscoveryResult(
            approach="Discovery + Evolution",
            f1_score=best.f1_score,
            precision=best.precision,
            recall=best.recall,
            n_states=best.n_states,
            n_transitions=best.n_transitions,
            discovery_time=elapsed,
            details={
                'discovered_states': len(discovered_states),
                'learned_transitions': len(learned_transitions),
                'evolved_genome_id': best.id,
            },
        )

    def _print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda r: r.f1_score, reverse=True)

        print(f"\n{'Approach':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'States':>8} {'Time':>8}")
        print("-" * 70)

        for r in sorted_results:
            print(f"{r.approach:<25} {r.f1_score:>8.3f} {r.precision:>8.3f} "
                  f"{r.recall:>8.3f} {r.n_states:>8} {r.discovery_time:>7.1f}s")

        # Winner
        winner = sorted_results[0]
        print("\n" + "-" * 70)
        print(f"WINNER: {winner.approach} with F1={winner.f1_score:.3f}")

        # Improvement over hardcoded
        hardcoded = next((r for r in self.results if r.approach == "Hardcoded CFG"), None)
        if hardcoded and winner.approach != "Hardcoded CFG":
            improvement = winner.f1_score - hardcoded.f1_score
            pct = improvement / hardcoded.f1_score * 100 if hardcoded.f1_score > 0 else 0
            print(f"Improvement over hardcoded: {improvement:+.3f} ({pct:+.1f}%)")

        print("=" * 70)

    def run_by_category(self, verbose: bool = True) -> Dict[str, List[DiscoveryResult]]:
        """Run benchmark broken down by program category."""
        results_by_category: Dict[str, List[DiscoveryResult]] = {}

        # Group examples by category
        examples_by_category: Dict[str, List[DatasetExample]] = defaultdict(list)
        for example in self.dataset.examples:
            examples_by_category[example.category].append(example)

        if verbose:
            print("=" * 70)
            print("CATEGORY-LEVEL BENCHMARK")
            print("=" * 70)

        for category, examples in sorted(examples_by_category.items()):
            if verbose:
                print(f"\n--- {category} ({len(examples)} examples) ---")

            subset_dataset = Dataset(examples=examples)
            subset_benchmark = DiscoveryBenchmark(subset_dataset)

            results = subset_benchmark.run_all(verbose=False)
            results_by_category[category] = results

            if verbose:
                best = max(results, key=lambda r: r.f1_score)
                hardcoded = next((r for r in results if r.approach == "Hardcoded CFG"), None)
                print(f"  Hardcoded: F1={hardcoded.f1_score:.3f}" if hardcoded else "")
                print(f"  Best: {best.approach} with F1={best.f1_score:.3f}")

        return results_by_category


def compute_state_coherence(
    discovered_states: Dict[str, DiscoveredState],
    examples: List[DatasetExample],
) -> float:
    """
    Measure how coherent discovered states are.

    Coherence = how often lines in a state appear together in traces.
    """
    if not discovered_states:
        return 0.0

    coherence_scores = []

    for state in discovered_states.values():
        if len(state.lines) <= 1:
            coherence_scores.append(1.0)
            continue

        # Count co-occurrence
        cooccur = 0
        total = 0

        for example in examples:
            covered = set(example.covered_lines)
            lines_in_state = state.lines
            covered_in_state = lines_in_state & covered

            if covered_in_state:
                total += 1
                # All lines in state should appear together
                if covered_in_state == lines_in_state:
                    cooccur += 1

        coherence = cooccur / total if total > 0 else 0.0
        coherence_scores.append(coherence)

    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0


def demo():
    """Run discovery benchmark demo."""
    print("=" * 60)
    print("DISCOVERY BENCHMARK DEMO")
    print("=" * 60)

    # Generate dataset
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate_dataset()

    # Use subset for faster demo
    subset = Dataset(examples=dataset.examples[:150])
    print(f"\nUsing {len(subset)} examples for demo")

    # Run benchmark
    benchmark = DiscoveryBenchmark(subset)
    results = benchmark.run_all(verbose=True)

    return results


if __name__ == "__main__":
    demo()
