"""
Benchmark: SAE States vs Clustering States.

Compares two approaches to state discovery for regex synthesis:

1. CLUSTERING (exp_regex_statechart):
   - Embed prefixes
   - Cluster embeddings
   - Clusters = states

2. SAE (this experiment):
   - Embed prefixes
   - SAE encodes to sparse features
   - Features = states

The hypothesis is that SAE states are:
- More interpretable (monosemantic)
- More compositional (feature combinations)
- Better for transfer (shared features)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .sae_regex_evolver import (
    SAERegexEvolver, SAEEvolutionConfig, SAEStatechart
)


@dataclass
class BenchmarkPattern:
    """A pattern to benchmark."""
    name: str
    description: str
    positive: List[str]
    negative: List[str]
    difficulty: str = "medium"


# Standard benchmark patterns
BENCHMARK_PATTERNS = [
    # Simple patterns
    BenchmarkPattern(
        name="a_plus",
        description="One or more 'a' (a+)",
        positive=["a", "aa", "aaa", "aaaa"],
        negative=["", "b", "ab", "ba"],
        difficulty="easy"
    ),
    BenchmarkPattern(
        name="ab_star",
        description="Zero or more 'ab' ((ab)*)",
        positive=["", "ab", "abab", "ababab"],
        negative=["a", "b", "aba", "abba"],
        difficulty="easy"
    ),
    BenchmarkPattern(
        name="a_plus_b",
        description="One or more 'a' then 'b' (a+b)",
        positive=["ab", "aab", "aaab", "aaaab"],
        negative=["", "a", "b", "ba", "aba"],
        difficulty="easy"
    ),

    # Medium patterns
    BenchmarkPattern(
        name="ab_plus",
        description="One or more 'ab' ((ab)+)",
        positive=["ab", "abab", "ababab"],
        negative=["", "a", "b", "aba", "abba", "ba"],
        difficulty="medium"
    ),
    BenchmarkPattern(
        name="a_or_b",
        description="Single 'a' or 'b' (a|b)",
        positive=["a", "b"],
        negative=["", "aa", "bb", "ab", "ba"],
        difficulty="medium"
    ),
    BenchmarkPattern(
        name="starts_a_ends_b",
        description="Starts with 'a', ends with 'b' (a.*b)",
        positive=["ab", "aab", "abb", "axyzb"],
        negative=["", "a", "b", "ba", "abc"],
        difficulty="medium"
    ),

    # Hard patterns
    BenchmarkPattern(
        name="even_a",
        description="Even number of 'a' ((aa)*)",
        positive=["", "aa", "aaaa", "aaaaaa"],
        negative=["a", "aaa", "aaaaa"],
        difficulty="hard"
    ),
    BenchmarkPattern(
        name="abc_plus",
        description="One or more 'abc' ((abc)+)",
        positive=["abc", "abcabc", "abcabcabc"],
        negative=["", "ab", "abca", "abcab"],
        difficulty="hard"
    ),
    BenchmarkPattern(
        name="alternating",
        description="Alternating a,b starting with a (a(ba)*)",
        positive=["a", "aba", "ababa", "abababa"],
        negative=["", "b", "ab", "aa", "bb"],
        difficulty="hard"
    ),
]


@dataclass
class SAEBenchmarkResult:
    """Result of benchmarking on a pattern."""
    pattern: BenchmarkPattern
    # SAE approach results
    sae_f1: float
    sae_accuracy: float
    sae_n_states: int
    sae_n_transitions: int
    sae_time: float
    sae_success: bool
    # Clustering approach results (if available)
    cluster_f1: Optional[float] = None
    cluster_accuracy: Optional[float] = None
    cluster_n_states: Optional[int] = None
    cluster_time: Optional[float] = None
    cluster_success: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            'pattern': self.pattern.name,
            'difficulty': self.pattern.difficulty,
            'sae_f1': self.sae_f1,
            'sae_accuracy': self.sae_accuracy,
            'sae_states': self.sae_n_states,
            'sae_time': self.sae_time,
            'sae_success': self.sae_success,
            'cluster_f1': self.cluster_f1,
            'cluster_states': self.cluster_n_states,
            'cluster_success': self.cluster_success,
        }


class SAERegexBenchmark:
    """Benchmark SAE state discovery for regex synthesis."""

    def __init__(
        self,
        target_f1: float = 0.90,
        n_generations: int = 40,
        n_features: int = 24,
        k_active: int = 3
    ):
        self.target_f1 = target_f1
        self.n_generations = n_generations
        self.n_features = n_features
        self.k_active = k_active

    def run_pattern(self, pattern: BenchmarkPattern) -> SAEBenchmarkResult:
        """Run benchmark on a single pattern."""
        start = time.time()

        config = SAEEvolutionConfig(
            n_features=self.n_features,
            k_active=self.k_active,
            population_size=25,
            n_generations=self.n_generations
        )

        try:
            evolver = SAERegexEvolver(config)
            sc, stats = evolver.evolve(
                pattern.positive,
                pattern.negative,
                verbose=False
            )

            elapsed = time.time() - start

            return SAEBenchmarkResult(
                pattern=pattern,
                sae_f1=stats['f1'],
                sae_accuracy=stats['accuracy'],
                sae_n_states=stats['n_states'],
                sae_n_transitions=stats['n_transitions'],
                sae_time=elapsed,
                sae_success=stats['f1'] >= self.target_f1
            )

        except Exception as e:
            elapsed = time.time() - start
            return SAEBenchmarkResult(
                pattern=pattern,
                sae_f1=0.0,
                sae_accuracy=0.0,
                sae_n_states=0,
                sae_n_transitions=0,
                sae_time=elapsed,
                sae_success=False
            )

    def run_all(
        self,
        patterns: List[BenchmarkPattern] = None,
        verbose: bool = True
    ) -> List[SAEBenchmarkResult]:
        """Run benchmark on all patterns."""
        patterns = patterns or BENCHMARK_PATTERNS

        if verbose:
            print("=" * 80)
            print("SAE REGEX STATES BENCHMARK")
            print("=" * 80)
            print(f"Target F1: {self.target_f1}")
            print(f"Generations: {self.n_generations}")
            print(f"Features: {self.n_features}, K: {self.k_active}")
            print("-" * 80)
            header = f"{'Pattern':<20} {'Diff':<8} {'F1':<8} {'Acc':<8} {'States':<8} {'Time':<8} {'Status'}"
            print(header)
            print("-" * 80)

        results = []
        for pattern in patterns:
            result = self.run_pattern(pattern)
            results.append(result)

            if verbose:
                status = "PASS" if result.sae_success else "FAIL"
                print(
                    f"{pattern.name:<20} "
                    f"{pattern.difficulty:<8} "
                    f"{result.sae_f1:<8.4f} "
                    f"{result.sae_accuracy:<8.4f} "
                    f"{result.sae_n_states:<8} "
                    f"{result.sae_time:<8.2f} "
                    f"{status}"
                )

        if verbose:
            print("-" * 80)
            successes = sum(1 for r in results if r.sae_success)
            avg_f1 = sum(r.sae_f1 for r in results) / len(results)
            total_time = sum(r.sae_time for r in results)

            # By difficulty
            print("\nResults by difficulty:")
            for diff in ["easy", "medium", "hard"]:
                diff_results = [r for r in results if r.pattern.difficulty == diff]
                if diff_results:
                    diff_success = sum(1 for r in diff_results if r.sae_success)
                    diff_f1 = sum(r.sae_f1 for r in diff_results) / len(diff_results)
                    print(f"  {diff:<10}: {diff_success}/{len(diff_results)} passed, Avg F1: {diff_f1:.4f}")

            print(f"\nSummary: {successes}/{len(results)} passed, Avg F1: {avg_f1:.4f}, Total: {total_time:.1f}s")
            print("=" * 80)

        return results


def compare_sae_vs_clustering(
    patterns: List[BenchmarkPattern] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare SAE approach to clustering approach.

    Note: This imports from exp_regex_statechart if available.
    """
    patterns = patterns or BENCHMARK_PATTERNS[:5]  # Use subset for speed

    if verbose:
        print("=" * 80)
        print("SAE vs CLUSTERING COMPARISON")
        print("=" * 80)

    # Run SAE benchmark
    sae_benchmark = SAERegexBenchmark(
        target_f1=0.85,
        n_generations=30,
        n_features=20,
        k_active=3
    )

    sae_results = []
    cluster_results = []

    # Try to import clustering approach
    try:
        from ..exp_regex_statechart import ComponentSynthesizer
        has_clustering = True
    except ImportError:
        has_clustering = False
        if verbose:
            print("Note: exp_regex_statechart not available for comparison")

    for pattern in patterns:
        if verbose:
            print(f"\nPattern: {pattern.name}")

        # SAE approach
        sae_result = sae_benchmark.run_pattern(pattern)
        sae_results.append(sae_result)

        if verbose:
            print(f"  SAE: F1={sae_result.sae_f1:.4f}, states={sae_result.sae_n_states}, "
                  f"time={sae_result.sae_time:.2f}s")

        # Clustering approach (if available)
        if has_clustering:
            start = time.time()
            try:
                synth = ComponentSynthesizer()
                sc = synth.synthesize_and_evolve(
                    pattern.positive,
                    pattern.negative,
                    n_generations=30,
                    verbose=False
                )

                # Evaluate
                tp = sum(1 for s in pattern.positive if sc.matches(s))
                tn = sum(1 for s in pattern.negative if not sc.matches(s))
                total = len(pattern.positive) + len(pattern.negative)
                accuracy = (tp + tn) / total

                precision = tp / (tp + (len(pattern.negative) - tn)) if (tp + len(pattern.negative) - tn) > 0 else 0
                recall = tp / len(pattern.positive) if pattern.positive else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                elapsed = time.time() - start

                cluster_results.append({
                    'f1': f1,
                    'states': sc.n_states,
                    'time': elapsed
                })

                if verbose:
                    print(f"  Cluster: F1={f1:.4f}, states={sc.n_states}, time={elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - start
                cluster_results.append({
                    'f1': 0.0,
                    'states': 0,
                    'time': elapsed
                })
                if verbose:
                    print(f"  Cluster: FAILED ({e})")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        avg_sae_f1 = sum(r.sae_f1 for r in sae_results) / len(sae_results)
        avg_sae_states = sum(r.sae_n_states for r in sae_results) / len(sae_results)
        avg_sae_time = sum(r.sae_time for r in sae_results) / len(sae_results)

        print(f"SAE Approach:")
        print(f"  Avg F1: {avg_sae_f1:.4f}")
        print(f"  Avg States: {avg_sae_states:.1f}")
        print(f"  Avg Time: {avg_sae_time:.2f}s")

        if cluster_results:
            avg_cluster_f1 = sum(r['f1'] for r in cluster_results) / len(cluster_results)
            avg_cluster_states = sum(r['states'] for r in cluster_results) / len(cluster_results)
            avg_cluster_time = sum(r['time'] for r in cluster_results) / len(cluster_results)

            print(f"\nClustering Approach:")
            print(f"  Avg F1: {avg_cluster_f1:.4f}")
            print(f"  Avg States: {avg_cluster_states:.1f}")
            print(f"  Avg Time: {avg_cluster_time:.2f}s")

            print(f"\nDifference (SAE - Clustering):")
            print(f"  F1: {avg_sae_f1 - avg_cluster_f1:+.4f}")
            print(f"  States: {avg_sae_states - avg_cluster_states:+.1f}")

    return {
        'sae_f1': sum(r.sae_f1 for r in sae_results) / len(sae_results),
        'sae_states': sum(r.sae_n_states for r in sae_results) / len(sae_results),
        'n_patterns': len(patterns),
    }


def run_sae_benchmark(verbose: bool = True) -> List[SAEBenchmarkResult]:
    """Run the full SAE regex benchmark."""
    benchmark = SAERegexBenchmark(
        target_f1=0.90,
        n_generations=40,
        n_features=24,
        k_active=3
    )
    return benchmark.run_all(verbose=verbose)


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Quick test with one pattern
    print("\n1. Single pattern test:")
    benchmark = SAERegexBenchmark(
        target_f1=0.80,
        n_generations=20,
        n_features=16,
        k_active=3
    )

    pattern = BENCHMARK_PATTERNS[0]  # a_plus
    result = benchmark.run_pattern(pattern)

    print(f"  Pattern: {pattern.name}")
    print(f"  F1: {result.sae_f1:.4f}")
    print(f"  States: {result.sae_n_states}")
    print(f"  Time: {result.sae_time:.2f}s")
    print(f"  Success: {result.sae_success}")

    # Test subset benchmark
    print("\n2. Subset benchmark (easy patterns):")
    easy_patterns = [p for p in BENCHMARK_PATTERNS if p.difficulty == "easy"]
    results = benchmark.run_all(easy_patterns)

    success_rate = sum(1 for r in results if r.sae_success) / len(results)
    print(f"\n  Success rate: {success_rate:.1%}")

    # Test comparison function
    print("\n3. Comparison test (2 patterns):")
    compare_sae_vs_clustering(BENCHMARK_PATTERNS[:2], verbose=True)

    print("\n" + "=" * 60)
    print("Benchmark tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
