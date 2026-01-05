"""
Advanced Benchmark: Patterns Requiring Extended Semantics.

Tests patterns that go beyond simple DFA:
- Counting: a{2,4}, exact repetitions
- Context: matching based on what came before
- Repetition: (ab)+, repeating patterns
- Length constraints: strings of specific lengths

These patterns require:
- Extended state (counters, flags)
- Semantic guards (counter comparisons)
- Actions (increment, reset)
- Accept conditions (final state requirements)

This benchmark tests the FULL power of semantic evolution.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .extended_statechart import ExtendedStatechart
from .semantic_evolver import SemanticEvolver, SemanticEvolutionConfig
from .component_synthesis import ComponentSynthesizer


@dataclass
class AdvancedPattern:
    """An advanced benchmark pattern requiring extended semantics."""
    name: str
    description: str
    category: str  # "counting", "repetition", "context", "length"
    positive: List[str]
    negative: List[str]
    requires_counter: bool = False
    requires_flag: bool = False
    difficulty: str = "medium"  # "easy", "medium", "hard"


# =============================================================================
# Benchmark Patterns
# =============================================================================

ADVANCED_PATTERNS = [
    # === Counting Patterns ===
    AdvancedPattern(
        name="exactly_3_a",
        description="Exactly 3 'a's",
        category="counting",
        positive=["aaa"],
        negative=["", "a", "aa", "aaaa", "aaaaa", "b", "aab"],
        requires_counter=True,
        difficulty="easy"
    ),
    AdvancedPattern(
        name="at_least_2_a",
        description="At least 2 'a's",
        category="counting",
        positive=["aa", "aaa", "aaaa", "aaaaa"],
        negative=["", "a", "b", "ab", "ba"],
        requires_counter=True,
        difficulty="easy"
    ),
    AdvancedPattern(
        name="between_2_and_4_a",
        description="Between 2 and 4 'a's (a{2,4})",
        category="counting",
        positive=["aa", "aaa", "aaaa"],
        negative=["", "a", "aaaaa", "aaaaaa", "b"],
        requires_counter=True,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="even_number_of_a",
        description="Even number of 'a's",
        category="counting",
        positive=["", "aa", "aaaa", "aaaaaa"],
        negative=["a", "aaa", "aaaaa"],
        requires_counter=True,
        difficulty="hard"
    ),

    # === Repetition Patterns ===
    AdvancedPattern(
        name="ab_repeat",
        description="One or more 'ab' ((ab)+)",
        category="repetition",
        positive=["ab", "abab", "ababab"],
        negative=["", "a", "b", "aba", "abba", "ba"],
        requires_counter=False,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="abc_repeat",
        description="One or more 'abc' ((abc)+)",
        category="repetition",
        positive=["abc", "abcabc", "abcabcabc"],
        negative=["", "ab", "a", "abca", "abcab"],
        requires_counter=False,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="repeat_2_times",
        description="Exactly 2 repetitions of 'ab' ((ab){2})",
        category="repetition",
        positive=["abab"],
        negative=["", "ab", "ababab", "aba", "ababa"],
        requires_counter=True,
        difficulty="hard"
    ),

    # === Context Patterns ===
    AdvancedPattern(
        name="a_then_b_or_c",
        description="'a' followed by 'b' or 'c'",
        category="context",
        positive=["ab", "ac"],
        negative=["", "a", "b", "c", "ba", "ca", "aa"],
        requires_flag=False,
        difficulty="easy"
    ),
    AdvancedPattern(
        name="starts_with_a",
        description="Starts with 'a', any content after",
        category="context",
        positive=["a", "ab", "abc", "aaa", "axyz"],
        negative=["", "b", "ba", "xab"],
        requires_flag=False,
        difficulty="easy"
    ),
    AdvancedPattern(
        name="ends_with_b",
        description="Ends with 'b'",
        category="context",
        positive=["b", "ab", "aab", "xyzb"],
        negative=["", "a", "ba", "abc"],
        requires_flag=False,
        difficulty="easy"
    ),
    AdvancedPattern(
        name="a_before_b",
        description="Contains 'a' before 'b'",
        category="context",
        positive=["ab", "aab", "axb", "axxb"],
        negative=["", "b", "ba", "a", "bxa"],
        requires_flag=True,
        difficulty="medium"
    ),

    # === Length Patterns ===
    AdvancedPattern(
        name="length_exactly_3",
        description="Exactly 3 characters",
        category="length",
        positive=["abc", "aaa", "xyz", "123"],
        negative=["", "a", "ab", "abcd", "abcde"],
        requires_counter=True,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="length_2_to_4",
        description="Length between 2 and 4",
        category="length",
        positive=["ab", "abc", "abcd", "aa", "aaa"],
        negative=["", "a", "abcde", "abcdef"],
        requires_counter=True,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="odd_length",
        description="Odd length strings",
        category="length",
        positive=["a", "abc", "abcde"],
        negative=["", "ab", "abcd", "abcdef"],
        requires_counter=True,
        difficulty="hard"
    ),

    # === Combined Patterns ===
    AdvancedPattern(
        name="a_then_2_b",
        description="'a' followed by exactly 2 'b's",
        category="counting",
        positive=["abb"],
        negative=["", "a", "ab", "abbb", "bb", "bba"],
        requires_counter=True,
        difficulty="medium"
    ),
    AdvancedPattern(
        name="alternating_ab",
        description="Alternating 'a' and 'b' starting with 'a'",
        category="repetition",
        positive=["a", "ab", "aba", "abab", "ababa"],
        negative=["", "b", "ba", "aa", "bb", "aab"],
        requires_flag=True,
        difficulty="hard"
    ),
]


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class AdvancedBenchmarkResult:
    """Result for a single advanced benchmark pattern."""
    pattern: AdvancedPattern
    statechart: Optional[ExtendedStatechart]
    test_f1: float
    test_accuracy: float
    elapsed_time: float
    success: bool
    n_states: int = 0
    n_variables: int = 0
    n_transitions: int = 0

    def to_dict(self) -> Dict:
        return {
            'name': self.pattern.name,
            'category': self.pattern.category,
            'difficulty': self.pattern.difficulty,
            'test_f1': self.test_f1,
            'test_accuracy': self.test_accuracy,
            'success': self.success,
            'n_states': self.n_states,
            'n_variables': self.n_variables,
            'elapsed_time': self.elapsed_time,
        }


# =============================================================================
# Advanced Benchmark Runner
# =============================================================================

class AdvancedBenchmark:
    """Run benchmarks on advanced patterns requiring extended semantics."""

    def __init__(
        self,
        target_f1: float = 0.95,
        n_generations: int = 75,
        use_synthesis: bool = True  # Use ML synthesis before evolution
    ):
        """
        Initialize benchmark.

        Args:
            target_f1: Target F1 score for success
            n_generations: Evolution generations
            use_synthesis: Use component synthesis for initial structure
        """
        self.target_f1 = target_f1
        self.n_generations = n_generations
        self.use_synthesis = use_synthesis

    def run_pattern(self, pattern: AdvancedPattern) -> AdvancedBenchmarkResult:
        """Run benchmark on a single pattern."""
        start_time = time.time()

        try:
            if self.use_synthesis:
                # Use ML synthesis + evolution
                synth = ComponentSynthesizer(alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
                statechart = synth.synthesize_and_evolve(
                    pattern.positive,
                    pattern.negative,
                    n_generations=self.n_generations,
                    verbose=False
                )
            else:
                # Pure evolution
                config = SemanticEvolutionConfig(
                    population_size=40,
                    n_generations=self.n_generations,
                    max_states=10,
                    verbose=False
                )
                evolver = SemanticEvolver(config)
                statechart, _ = evolver.evolve(pattern.positive, pattern.negative)

            # Evaluate on test set
            tp = fp = tn = fn = 0

            for s in pattern.positive:
                if statechart.matches(s):
                    tp += 1
                else:
                    fn += 1

            for s in pattern.negative:
                if statechart.matches(s):
                    fp += 1
                else:
                    tn += 1

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            elapsed = time.time() - start_time

            return AdvancedBenchmarkResult(
                pattern=pattern,
                statechart=statechart,
                test_f1=f1,
                test_accuracy=accuracy,
                elapsed_time=elapsed,
                success=f1 >= self.target_f1,
                n_states=statechart.n_states,
                n_variables=len(statechart.variables),
                n_transitions=statechart.n_transitions
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return AdvancedBenchmarkResult(
                pattern=pattern,
                statechart=None,
                test_f1=0.0,
                test_accuracy=0.0,
                elapsed_time=elapsed,
                success=False
            )

    def run_all(
        self,
        patterns: List[AdvancedPattern] = None,
        verbose: bool = True
    ) -> List[AdvancedBenchmarkResult]:
        """Run benchmark on all patterns."""
        patterns = patterns or ADVANCED_PATTERNS

        if verbose:
            print("=" * 80)
            print("ADVANCED SEMANTIC BENCHMARK")
            print("=" * 80)
            print(f"Target F1: {self.target_f1}")
            print(f"Generations: {self.n_generations}")
            print(f"Use Synthesis: {self.use_synthesis}")
            print(f"Patterns: {len(patterns)}")
            print("-" * 80)
            header = f"{'Pattern':<25} {'Category':<12} {'Diff':<8} {'F1':<8} {'States':<8} {'Vars':<6} {'Time':<8} {'Status'}"
            print(header)
            print("-" * 80)

        results = []
        for pattern in patterns:
            result = self.run_pattern(pattern)
            results.append(result)

            if verbose:
                status = "PASS" if result.success else "FAIL"
                print(
                    f"{pattern.name:<25} "
                    f"{pattern.category:<12} "
                    f"{pattern.difficulty:<8} "
                    f"{result.test_f1:<8.4f} "
                    f"{result.n_states:<8} "
                    f"{result.n_variables:<6} "
                    f"{result.elapsed_time:<8.2f} "
                    f"{status}"
                )

        if verbose:
            print("-" * 80)
            successes = sum(1 for r in results if r.success)
            avg_f1 = sum(r.test_f1 for r in results) / len(results)
            total_time = sum(r.elapsed_time for r in results)

            # By category
            print("\nResults by category:")
            categories = set(p.category for p in patterns)
            for cat in sorted(categories):
                cat_results = [r for r in results if r.pattern.category == cat]
                cat_success = sum(1 for r in cat_results if r.success)
                cat_f1 = sum(r.test_f1 for r in cat_results) / len(cat_results)
                print(f"  {cat:<15}: {cat_success}/{len(cat_results)} passed, Avg F1: {cat_f1:.4f}")

            # By difficulty
            print("\nResults by difficulty:")
            difficulties = ["easy", "medium", "hard"]
            for diff in difficulties:
                diff_results = [r for r in results if r.pattern.difficulty == diff]
                if diff_results:
                    diff_success = sum(1 for r in diff_results if r.success)
                    diff_f1 = sum(r.test_f1 for r in diff_results) / len(diff_results)
                    print(f"  {diff:<10}: {diff_success}/{len(diff_results)} passed, Avg F1: {diff_f1:.4f}")

            print(f"\nSummary: {successes}/{len(results)} passed, Avg F1: {avg_f1:.4f}, Total time: {total_time:.2f}s")
            print("=" * 80)

        return results

    def run_by_category(
        self,
        category: str,
        verbose: bool = True
    ) -> List[AdvancedBenchmarkResult]:
        """Run benchmark on patterns of a specific category."""
        patterns = [p for p in ADVANCED_PATTERNS if p.category == category]
        return self.run_all(patterns, verbose)

    def run_by_difficulty(
        self,
        difficulty: str,
        verbose: bool = True
    ) -> List[AdvancedBenchmarkResult]:
        """Run benchmark on patterns of a specific difficulty."""
        patterns = [p for p in ADVANCED_PATTERNS if p.difficulty == difficulty]
        return self.run_all(patterns, verbose)


def run_quick_advanced_benchmark():
    """Run quick benchmark on easy patterns."""
    print("Quick Advanced Benchmark (Easy Patterns Only)")
    benchmark = AdvancedBenchmark(
        target_f1=0.90,
        n_generations=30,
        use_synthesis=True
    )

    easy_patterns = [p for p in ADVANCED_PATTERNS if p.difficulty == "easy"]
    return benchmark.run_all(easy_patterns)


def run_full_advanced_benchmark():
    """Run full benchmark on all advanced patterns."""
    print("Full Advanced Benchmark")
    benchmark = AdvancedBenchmark(
        target_f1=0.95,
        n_generations=75,
        use_synthesis=True
    )

    return benchmark.run_all()


def test_advanced_benchmark():
    """Test the advanced benchmark system."""
    print("=" * 60)
    print("ADVANCED BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Test single pattern
    print("\n1. Testing single advanced pattern:")
    benchmark = AdvancedBenchmark(
        target_f1=0.80,
        n_generations=30,
        use_synthesis=True
    )

    pattern = ADVANCED_PATTERNS[0]  # exactly_3_a
    result = benchmark.run_pattern(pattern)

    print(f"Pattern: {pattern.name} ({pattern.description})")
    print(f"Category: {pattern.category}, Difficulty: {pattern.difficulty}")
    print(f"Test F1: {result.test_f1:.4f}")
    print(f"Success: {result.success}")
    print(f"States: {result.n_states}, Variables: {result.n_variables}")

    if result.statechart:
        print(f"\nSynthesized statechart:")
        print(result.statechart.to_string())

        print("\nTest cases:")
        all_examples = pattern.positive + pattern.negative
        for s in all_examples:
            expected = s in pattern.positive
            actual = result.statechart.matches(s)
            status = "PASS" if actual == expected else "FAIL"
            print(f"  '{s}' -> {actual} (expected {expected}) [{status}]")

    # Test quick benchmark
    print("\n" + "=" * 60)
    print("2. Running quick advanced benchmark:")
    easy_patterns = [p for p in ADVANCED_PATTERNS if p.difficulty == "easy"][:3]
    results = benchmark.run_all(easy_patterns)

    print("\n" + "=" * 60)
    print("Advanced benchmark tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_advanced_benchmark()
