"""
Benchmark suite for regex statechart evolution.

Tests evolution on patterns of varying complexity:
- Simple: a*, (ab)+
- Medium: [a-z]+, digit patterns
- Complex: alternations, wildcards

Target metrics:
- F1 >= 0.99 on test examples
- Evolution generations < 100
- Statechart size <= DFA size
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

from .regex_statechart import RegexStatechart
from .re2_oracle import RE2Oracle, RegexExamples
from .evolver import RegexEvolver, EvolutionConfig
from .synthesis import RegexSynthesizer, SynthesisResult


@dataclass
class BenchmarkPattern:
    """A benchmark regex pattern."""
    name: str
    pattern: str
    complexity: str  # "simple", "medium", "complex"
    description: str
    expected_states: int  # Approximate expected state count

    # Specific test cases (beyond generated examples)
    test_positive: List[str] = field(default_factory=list)
    test_negative: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark pattern."""
    pattern: BenchmarkPattern
    synthesis_result: Optional[SynthesisResult]
    test_f1: float
    test_accuracy: float
    oracle_match_rate: float  # Agreement with RE2 oracle
    elapsed_time: float
    success: bool  # F1 >= target

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'pattern_name': self.pattern.name,
            'pattern': self.pattern.pattern,
            'complexity': self.pattern.complexity,
            'test_f1': self.test_f1,
            'test_accuracy': self.test_accuracy,
            'oracle_match_rate': self.oracle_match_rate,
            'n_states': self.synthesis_result.n_states if self.synthesis_result else 0,
            'n_transitions': self.synthesis_result.n_transitions if self.synthesis_result else 0,
            'elapsed_time': self.elapsed_time,
            'success': self.success,
        }


# =============================================================================
# Benchmark Patterns
# =============================================================================

BENCHMARK_PATTERNS = [
    # Simple patterns
    BenchmarkPattern(
        name="star_a",
        pattern="a*",
        complexity="simple",
        description="Zero or more 'a'",
        expected_states=1,
        test_positive=["", "a", "aa", "aaa", "aaaa"],
        test_negative=["b", "ab", "ba", "aab"],
    ),
    BenchmarkPattern(
        name="plus_a",
        pattern="a+",
        complexity="simple",
        description="One or more 'a'",
        expected_states=2,
        test_positive=["a", "aa", "aaa", "aaaa"],
        test_negative=["", "b", "ab", "ba"],
    ),
    BenchmarkPattern(
        name="seq_ab",
        pattern="ab",
        complexity="simple",
        description="Literal 'ab'",
        expected_states=3,
        test_positive=["ab"],
        test_negative=["", "a", "b", "ba", "abc", "aab"],
    ),
    BenchmarkPattern(
        name="concat_ab_plus",
        pattern="ab+",
        complexity="simple",
        description="'a' followed by one or more 'b'",
        expected_states=3,
        test_positive=["ab", "abb", "abbb"],
        test_negative=["", "a", "b", "ba", "aa"],
    ),
    BenchmarkPattern(
        name="repeat_ab",
        pattern="(ab)+",
        complexity="simple",
        description="One or more 'ab' sequences",
        expected_states=3,
        test_positive=["ab", "abab", "ababab"],
        test_negative=["", "a", "b", "aba", "abba"],
    ),

    # Medium patterns
    BenchmarkPattern(
        name="alpha_lower",
        pattern="[a-z]+",
        complexity="medium",
        description="One or more lowercase letters",
        expected_states=2,
        test_positive=["a", "abc", "hello", "xyz"],
        test_negative=["", "123", "ABC", "a1", "hello!"],
    ),
    BenchmarkPattern(
        name="digit_3",
        pattern="[0-9]{3}",
        complexity="medium",
        description="Exactly 3 digits",
        expected_states=4,
        test_positive=["123", "000", "999"],
        test_negative=["", "12", "1234", "abc", "12a"],
    ),
    BenchmarkPattern(
        name="word",
        pattern="[a-z]+[0-9]*",
        complexity="medium",
        description="Word optionally followed by digits",
        expected_states=3,
        test_positive=["abc", "abc123", "x", "x1"],
        test_negative=["", "123", "123abc", "abc-def"],
    ),
    BenchmarkPattern(
        name="phone_simple",
        pattern="[0-9]{3}-[0-9]{4}",
        complexity="medium",
        description="Simple phone format xxx-xxxx",
        expected_states=8,
        test_positive=["123-4567", "000-0000", "999-9999"],
        test_negative=["", "1234567", "12-3456", "123-456"],
    ),

    # Complex patterns
    BenchmarkPattern(
        name="alternation",
        pattern="(foo|bar)",
        complexity="complex",
        description="Match 'foo' or 'bar'",
        expected_states=7,
        test_positive=["foo", "bar"],
        test_negative=["", "fo", "ba", "foobar", "baz"],
    ),
    BenchmarkPattern(
        name="alt_suffix",
        pattern="(foo|bar)baz",
        complexity="complex",
        description="'foo' or 'bar' followed by 'baz'",
        expected_states=10,
        test_positive=["foobaz", "barbaz"],
        test_negative=["", "foo", "bar", "baz", "foobar"],
    ),
    BenchmarkPattern(
        name="wildcard",
        pattern="a.*b",
        complexity="complex",
        description="'a' then anything then 'b'",
        expected_states=3,
        test_positive=["ab", "aXb", "aXXb", "a123b"],
        test_negative=["", "a", "b", "ba", "axbx"],
    ),
    BenchmarkPattern(
        name="optional",
        pattern="colou?r",
        complexity="complex",
        description="'color' or 'colour'",
        expected_states=7,
        test_positive=["color", "colour"],
        test_negative=["", "colr", "colouur", "colors"],
    ),
]


# =============================================================================
# Benchmark Runner
# =============================================================================

class RegexBenchmark:
    """Run benchmarks on regex patterns."""

    def __init__(
        self,
        target_f1: float = 0.99,
        n_examples: int = 100,
        n_test: int = 50,
        config: EvolutionConfig = None
    ):
        """
        Initialize benchmark.

        Args:
            target_f1: Target F1 score for success
            n_examples: Examples for training
            n_test: Examples for testing
            config: Evolution configuration
        """
        self.target_f1 = target_f1
        self.n_examples = n_examples
        self.n_test = n_test
        self.config = config or EvolutionConfig(
            population_size=40,
            n_generations=75,
            max_states=15,
            verbose=False
        )

    def run_pattern(self, pattern: BenchmarkPattern) -> BenchmarkResult:
        """Run benchmark on a single pattern."""
        start_time = time.time()

        try:
            # Create oracle
            oracle = RE2Oracle(pattern.pattern)

            # Generate examples
            examples = oracle.generate_examples(
                n_positive=self.n_examples // 2,
                n_negative=self.n_examples // 2
            )

            # Synthesize
            synthesizer = RegexSynthesizer(self.config)
            result = synthesizer.synthesize(
                examples.positive,
                examples.negative,
                original_pattern=pattern.pattern,
                verbose=False
            )

            # Generate test set
            test_examples = oracle.generate_examples(
                n_positive=self.n_test // 2,
                n_negative=self.n_test // 2
            )

            # Add specific test cases
            all_test = test_examples.all_examples
            for s in pattern.test_positive:
                all_test.append((s, True))
            for s in pattern.test_negative:
                all_test.append((s, False))

            # Evaluate on test set
            tp = fp = tn = fn = 0
            oracle_agree = 0

            for s, expected in all_test:
                predicted = result.statechart.matches(s)
                oracle_result = oracle.matches(s)

                if predicted == oracle_result:
                    oracle_agree += 1

                if expected and predicted:
                    tp += 1
                elif expected and not predicted:
                    fn += 1
                elif not expected and predicted:
                    fp += 1
                else:
                    tn += 1

            total = tp + fp + tn + fn
            test_accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            test_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            oracle_match_rate = oracle_agree / total if total > 0 else 0.0

            elapsed = time.time() - start_time

            return BenchmarkResult(
                pattern=pattern,
                synthesis_result=result,
                test_f1=test_f1,
                test_accuracy=test_accuracy,
                oracle_match_rate=oracle_match_rate,
                elapsed_time=elapsed,
                success=test_f1 >= self.target_f1
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return BenchmarkResult(
                pattern=pattern,
                synthesis_result=None,
                test_f1=0.0,
                test_accuracy=0.0,
                oracle_match_rate=0.0,
                elapsed_time=elapsed,
                success=False
            )

    def run_all(
        self,
        patterns: List[BenchmarkPattern] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """Run benchmark on all patterns."""
        patterns = patterns or BENCHMARK_PATTERNS

        if verbose:
            print("=" * 70)
            print("REGEX STATECHART BENCHMARK")
            print("=" * 70)
            print(f"Target F1: {self.target_f1}")
            print(f"Training examples: {self.n_examples}")
            print(f"Test examples: {self.n_test}")
            print(f"Patterns: {len(patterns)}")
            print("-" * 70)
            print(f"{'Pattern':<20} {'Complexity':<10} {'F1':<8} {'States':<8} {'Time':<8} {'Status'}")
            print("-" * 70)

        results = []
        for pattern in patterns:
            result = self.run_pattern(pattern)
            results.append(result)

            if verbose:
                status = "PASS" if result.success else "FAIL"
                n_states = result.synthesis_result.n_states if result.synthesis_result else 0
                print(
                    f"{pattern.name:<20} "
                    f"{pattern.complexity:<10} "
                    f"{result.test_f1:<8.4f} "
                    f"{n_states:<8} "
                    f"{result.elapsed_time:<8.2f} "
                    f"{status}"
                )

        if verbose:
            print("-" * 70)
            successes = sum(1 for r in results if r.success)
            avg_f1 = sum(r.test_f1 for r in results) / len(results)
            total_time = sum(r.elapsed_time for r in results)
            print(f"Summary: {successes}/{len(results)} passed, Avg F1: {avg_f1:.4f}, Total time: {total_time:.2f}s")
            print("=" * 70)

        return results

    def run_by_complexity(
        self,
        complexity: str,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """Run benchmark on patterns of a specific complexity."""
        patterns = [p for p in BENCHMARK_PATTERNS if p.complexity == complexity]
        return self.run_all(patterns, verbose)


def run_quick_benchmark():
    """Run a quick benchmark on simple patterns."""
    print("Quick Benchmark (Simple Patterns Only)")
    benchmark = RegexBenchmark(
        target_f1=0.95,
        n_examples=60,
        n_test=30,
        config=EvolutionConfig(
            population_size=25,
            n_generations=40,
            max_states=8,
            verbose=False
        )
    )

    simple_patterns = [p for p in BENCHMARK_PATTERNS if p.complexity == "simple"]
    return benchmark.run_all(simple_patterns[:3])


def run_full_benchmark():
    """Run full benchmark on all patterns."""
    print("Full Benchmark (All Patterns)")
    benchmark = RegexBenchmark(
        target_f1=0.99,
        n_examples=100,
        n_test=50,
        config=EvolutionConfig(
            population_size=50,
            n_generations=100,
            max_states=15,
            verbose=False
        )
    )

    return benchmark.run_all()


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Test single pattern
    print("\n1. Testing single pattern benchmark:")
    benchmark = RegexBenchmark(
        target_f1=0.90,
        n_examples=40,
        n_test=20,
        config=EvolutionConfig(
            population_size=20,
            n_generations=30,
            max_states=6,
            verbose=False
        )
    )

    pattern = BENCHMARK_PATTERNS[0]  # a*
    result = benchmark.run_pattern(pattern)

    print(f"Pattern: {pattern.name} ({pattern.pattern})")
    print(f"Test F1: {result.test_f1:.4f}")
    print(f"Oracle Match: {result.oracle_match_rate:.4f}")
    print(f"Success: {result.success}")

    if result.synthesis_result:
        print(f"\nSynthesized statechart:")
        print(result.synthesis_result.statechart.to_string())

    # Test quick benchmark
    print("\n" + "-" * 60)
    print("2. Running quick benchmark:")
    results = run_quick_benchmark()

    print("\n" + "=" * 60)
    print("Benchmark tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
