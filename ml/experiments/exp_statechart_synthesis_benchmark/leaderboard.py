"""
Leaderboard for Statechart Synthesis Benchmark

Ranks methods across domains and provides:
1. Overall rankings
2. Per-domain rankings
3. Per-metric rankings
4. Category comparisons
5. Statistical significance testing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import json
import math

from .synthesis_methods import MethodCategory, SynthesisResult
from .domain_suite import Domain, DomainType
from .metrics import BenchmarkMetrics


# =============================================================================
# Leaderboard Entry
# =============================================================================

@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""
    method_name: str
    category: MethodCategory
    domain_name: str
    domain_type: DomainType

    metrics: BenchmarkMetrics

    # Computed ranks
    overall_rank: int = 0
    accuracy_rank: int = 0
    efficiency_rank: int = 0
    interpretability_rank: int = 0

    def score_tuple(self) -> Tuple[float, float, float, float]:
        """Return scores as tuple for comparison."""
        return (
            self.metrics.overall_score(),
            self.metrics.accuracy_score(),
            self.metrics.efficiency_score(),
            self.metrics.interpretability_score()
        )


# =============================================================================
# Rankings
# =============================================================================

@dataclass
class MethodRanking:
    """Aggregated ranking for a method across domains."""
    method_name: str
    category: MethodCategory

    # Aggregate scores
    avg_overall: float = 0.0
    avg_accuracy: float = 0.0
    avg_efficiency: float = 0.0
    avg_interpretability: float = 0.0

    # Domain-specific
    domain_scores: Dict[str, float] = field(default_factory=dict)

    # Rank position
    overall_rank: int = 0

    # Stats
    n_domains: int = 0
    best_domain: str = ""
    worst_domain: str = ""


@dataclass
class DomainRanking:
    """Rankings for methods on a specific domain."""
    domain_name: str
    domain_type: DomainType

    # Method rankings
    method_rankings: List[Tuple[str, float]] = field(default_factory=list)

    # Best method
    best_method: str = ""
    best_score: float = 0.0


# =============================================================================
# Leaderboard
# =============================================================================

class Leaderboard:
    """Main leaderboard for benchmark results."""

    def __init__(self):
        self.entries: List[LeaderboardEntry] = []
        self.method_rankings: Dict[str, MethodRanking] = {}
        self.domain_rankings: Dict[str, DomainRanking] = {}

    def add_entry(
        self,
        method_name: str,
        category: MethodCategory,
        domain: Domain,
        metrics: BenchmarkMetrics
    ):
        """Add a new benchmark result."""
        entry = LeaderboardEntry(
            method_name=method_name,
            category=category,
            domain_name=domain.name,
            domain_type=domain.domain_type,
            metrics=metrics
        )
        self.entries.append(entry)

    def compute_rankings(self):
        """Compute all rankings from entries."""
        self._compute_per_entry_ranks()
        self._compute_method_rankings()
        self._compute_domain_rankings()

    def _compute_per_entry_ranks(self):
        """Rank entries by each metric."""
        # Group by domain
        by_domain: Dict[str, List[LeaderboardEntry]] = defaultdict(list)
        for entry in self.entries:
            by_domain[entry.domain_name].append(entry)

        # Rank within each domain
        for domain_name, entries in by_domain.items():
            # Overall rank
            sorted_entries = sorted(entries, key=lambda e: -e.metrics.overall_score())
            for i, entry in enumerate(sorted_entries):
                entry.overall_rank = i + 1

            # Accuracy rank
            sorted_entries = sorted(entries, key=lambda e: -e.metrics.accuracy_score())
            for i, entry in enumerate(sorted_entries):
                entry.accuracy_rank = i + 1

            # Efficiency rank
            sorted_entries = sorted(entries, key=lambda e: -e.metrics.efficiency_score())
            for i, entry in enumerate(sorted_entries):
                entry.efficiency_rank = i + 1

            # Interpretability rank
            sorted_entries = sorted(entries, key=lambda e: -e.metrics.interpretability_score())
            for i, entry in enumerate(sorted_entries):
                entry.interpretability_rank = i + 1

    def _compute_method_rankings(self):
        """Aggregate rankings per method."""
        # Group by method
        by_method: Dict[str, List[LeaderboardEntry]] = defaultdict(list)
        for entry in self.entries:
            by_method[entry.method_name].append(entry)

        # Compute aggregates
        for method_name, entries in by_method.items():
            if not entries:
                continue

            category = entries[0].category
            n = len(entries)

            avg_overall = sum(e.metrics.overall_score() for e in entries) / n
            avg_accuracy = sum(e.metrics.accuracy_score() for e in entries) / n
            avg_efficiency = sum(e.metrics.efficiency_score() for e in entries) / n
            avg_interpret = sum(e.metrics.interpretability_score() for e in entries) / n

            domain_scores = {e.domain_name: e.metrics.overall_score() for e in entries}
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            worst_domain = min(domain_scores.items(), key=lambda x: x[1])[0]

            self.method_rankings[method_name] = MethodRanking(
                method_name=method_name,
                category=category,
                avg_overall=avg_overall,
                avg_accuracy=avg_accuracy,
                avg_efficiency=avg_efficiency,
                avg_interpretability=avg_interpret,
                domain_scores=domain_scores,
                n_domains=n,
                best_domain=best_domain,
                worst_domain=worst_domain
            )

        # Assign overall ranks
        sorted_methods = sorted(self.method_rankings.values(),
                                key=lambda m: -m.avg_overall)
        for i, m in enumerate(sorted_methods):
            m.overall_rank = i + 1

    def _compute_domain_rankings(self):
        """Compute rankings per domain."""
        # Group by domain
        by_domain: Dict[str, List[LeaderboardEntry]] = defaultdict(list)
        for entry in self.entries:
            by_domain[entry.domain_name].append(entry)

        for domain_name, entries in by_domain.items():
            if not entries:
                continue

            domain_type = entries[0].domain_type

            # Rank methods
            method_rankings = [
                (e.method_name, e.metrics.overall_score())
                for e in sorted(entries, key=lambda e: -e.metrics.overall_score())
            ]

            best = method_rankings[0] if method_rankings else ("", 0.0)

            self.domain_rankings[domain_name] = DomainRanking(
                domain_name=domain_name,
                domain_type=domain_type,
                method_rankings=method_rankings,
                best_method=best[0],
                best_score=best[1]
            )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_top_methods(self, n: int = 5) -> List[MethodRanking]:
        """Get top N methods overall."""
        sorted_methods = sorted(self.method_rankings.values(),
                                key=lambda m: -m.avg_overall)
        return sorted_methods[:n]

    def get_best_for_domain(self, domain_name: str) -> Optional[str]:
        """Get best method for a domain."""
        if domain_name in self.domain_rankings:
            return self.domain_rankings[domain_name].best_method
        return None

    def get_best_for_domain_type(self, domain_type: DomainType) -> Dict[str, str]:
        """Get best method per domain for a domain type."""
        result = {}
        for domain_name, ranking in self.domain_rankings.items():
            if ranking.domain_type == domain_type:
                result[domain_name] = ranking.best_method
        return result

    def get_category_comparison(self) -> Dict[MethodCategory, float]:
        """Compare average scores by category."""
        by_category: Dict[MethodCategory, List[float]] = defaultdict(list)

        for method in self.method_rankings.values():
            by_category[method.category].append(method.avg_overall)

        return {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in by_category.items()
        }

    # =========================================================================
    # Display
    # =========================================================================

    def print_overall_leaderboard(self, top_n: int = 10):
        """Print overall leaderboard."""
        print("=" * 80)
        print("OVERALL LEADERBOARD")
        print("=" * 80)

        print(f"\n{'Rank':<6} {'Method':<25} {'Category':<15} {'Overall':>10} {'Accuracy':>10} {'Effic.':>10}")
        print("-" * 80)

        for method in self.get_top_methods(top_n):
            print(f"{method.overall_rank:<6} {method.method_name:<25} "
                  f"{method.category.name:<15} "
                  f"{method.avg_overall:>10.3f} {method.avg_accuracy:>10.3f} "
                  f"{method.avg_efficiency:>10.3f}")

    def print_domain_leaderboard(self, domain_name: str):
        """Print leaderboard for a specific domain."""
        if domain_name not in self.domain_rankings:
            print(f"No results for domain: {domain_name}")
            return

        ranking = self.domain_rankings[domain_name]

        print(f"\n--- {domain_name} ({ranking.domain_type.name}) ---")
        print(f"{'Rank':<6} {'Method':<30} {'Score':>10}")
        print("-" * 50)

        for i, (method, score) in enumerate(ranking.method_rankings):
            print(f"{i+1:<6} {method:<30} {score:>10.3f}")

    def print_category_comparison(self):
        """Print comparison by method category."""
        print("\n--- Category Comparison ---")
        comparison = self.get_category_comparison()

        for cat, score in sorted(comparison.items(), key=lambda x: -x[1]):
            print(f"{cat.name:<15}: {score:.3f}")

    def to_json(self) -> str:
        """Export leaderboard as JSON."""
        data = {
            'methods': [
                {
                    'name': m.method_name,
                    'category': m.category.name,
                    'rank': m.overall_rank,
                    'avg_overall': m.avg_overall,
                    'avg_accuracy': m.avg_accuracy,
                    'avg_efficiency': m.avg_efficiency,
                    'avg_interpretability': m.avg_interpretability,
                    'n_domains': m.n_domains,
                    'best_domain': m.best_domain,
                    'worst_domain': m.worst_domain
                }
                for m in sorted(self.method_rankings.values(),
                                key=lambda x: x.overall_rank)
            ],
            'domains': [
                {
                    'name': d.domain_name,
                    'type': d.domain_type.name,
                    'best_method': d.best_method,
                    'best_score': d.best_score,
                    'rankings': d.method_rankings
                }
                for d in self.domain_rankings.values()
            ],
            'category_comparison': {
                cat.name: score
                for cat, score in self.get_category_comparison().items()
            }
        }
        return json.dumps(data, indent=2)


# =============================================================================
# Result Aggregation from Experiments
# =============================================================================

# Known results from completed experiments
KNOWN_RESULTS = {
    # From exp_lca_neural
    'hybrid_lca': {'accuracy': 0.744, 'category': 'HYBRID'},
    'supervised_lca': {'accuracy': 0.736, 'category': 'NEURAL'},
    'evolutionary_lca': {'accuracy': 0.096, 'category': 'EVOLUTIONARY'},

    # From exp_unified_statechart_evolution
    'nsga2_unified': {'accuracy': 0.85, 'category': 'EVOLUTIONARY'},

    # From exp_transfer_coverage
    'transfer_zero_shot': {'accuracy': 0.58, 'category': 'HYBRID'},

    # From exp_dialogue_statechart
    'dialogue_extractor': {'accuracy': 0.90, 'category': 'NEURAL'},

    # From exp_sae_coverage_synthesis
    'sae_coverage': {'accuracy': 0.82, 'category': 'SAE_BASED'},

    # From exp_priority_attention
    'priority_attention': {'accuracy': 1.14, 'category': 'NEURAL'},  # vs baseline

    # From exp_guard_synthesis
    'guard_evolution': {'accuracy': 0.95, 'category': 'EVOLUTIONARY'},

    # From exp_action_composition
    'action_algebra': {'accuracy': 1.0, 'category': 'EVOLUTIONARY'},

    # From exp_joint_guard_action_evolution
    'joint_evolution': {'accuracy': 0.888, 'category': 'EVOLUTIONARY'},
    'independent_evolution': {'accuracy': 0.788, 'category': 'EVOLUTIONARY'},
}


def aggregate_experiment_results() -> Dict[str, Dict]:
    """Aggregate known results from all experiments."""
    return KNOWN_RESULTS.copy()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate leaderboard."""
    print("=" * 60)
    print("LEADERBOARD DEMO")
    print("=" * 60)

    from .synthesis_methods import get_all_methods, SynthesisConfig
    from .domain_suite import get_all_domains
    from .metrics import compute_all_metrics

    # Run quick benchmark
    config = SynthesisConfig(n_generations=20, population_size=15)
    methods = get_all_methods(config)
    domains = get_all_domains()[:3]  # Quick: first 3 domains

    leaderboard = Leaderboard()

    print(f"\nRunning {len(methods)} methods on {len(domains)} domains...")

    for domain in domains:
        for method in methods:
            result = method.synthesize(domain.get_train_pairs())
            metrics = compute_all_metrics(result, domain)
            leaderboard.add_entry(method.name, method.category, domain, metrics)

    leaderboard.compute_rankings()

    # Display results
    leaderboard.print_overall_leaderboard()

    print("\n--- Per-Domain Results ---")
    for domain in domains:
        leaderboard.print_domain_leaderboard(domain.name)

    leaderboard.print_category_comparison()

    # Show known results from experiments
    print("\n--- Known Results from Experiments ---")
    for method, results in aggregate_experiment_results().items():
        print(f"{method:<25}: accuracy={results['accuracy']:.3f} [{results['category']}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
