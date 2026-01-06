"""
Comprehensive Benchmark Suite for Formal Language Learning

Benchmarks:
1. Algorithm comparison (RPNI vs L* vs Genetic)
2. Scaling behavior (examples vs states)
3. Language class detection accuracy
4. Statechart conversion quality
5. Cross-algorithm consistency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
import time
import random
import string

from .formal_base import DFA, NFA, Alphabet, State, Transition, Symbol
from .regex_inducer import (
    DFAInducer,
    InductionAlgorithm,
    InducedDFA,
    RPNIInducer,
    LStarInducer,
)
from .grammar_inducer import CFGInducer, InducedCFG
from .unified_learner import (
    UnifiedLearner,
    LanguageType,
    LanguageComplexity,
    LearningResult,
    learn_from_examples,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    algorithm: str
    n_positive: int
    n_negative: int
    n_states_learned: int
    accuracy: float
    time_ms: float
    memory_estimate_kb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def get_by_algorithm(self, algorithm: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.algorithm == algorithm]

    def get_by_name(self, name: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.name == name]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        algorithms = set(r.algorithm for r in self.results)
        summary = {}

        for algo in algorithms:
            algo_results = self.get_by_algorithm(algo)
            summary[algo] = {
                "count": len(algo_results),
                "avg_accuracy": sum(r.accuracy for r in algo_results) / len(algo_results),
                "avg_time_ms": sum(r.time_ms for r in algo_results) / len(algo_results),
                "avg_states": sum(r.n_states_learned for r in algo_results) / len(algo_results),
            }

        return summary


class LanguageGenerator:
    """Generate example strings for various language types."""

    @staticmethod
    def regular_ab_star(n_pos: int, n_neg: int) -> Tuple[List[str], List[str]]:
        """Generate examples for (ab)* language."""
        positive = [""] + ["ab" * i for i in range(1, n_pos)]
        negative = ["a", "b", "ba", "aab", "abb"] + [
            "".join(random.choices("ab", k=random.randint(1, 5)))
            for _ in range(n_neg - 5)
        ]
        negative = [s for s in negative if s not in positive][:n_neg]
        return positive, negative

    @staticmethod
    def regular_ends_ab(n_pos: int, n_neg: int) -> Tuple[List[str], List[str]]:
        """Generate examples for strings ending in 'ab'."""
        positive = ["ab"] + [
            "".join(random.choices("ab", k=random.randint(0, 5))) + "ab"
            for _ in range(n_pos - 1)
        ]
        negative = ["", "a", "b", "ba", "aa", "bb"] + [
            "".join(random.choices("ab", k=random.randint(1, 6)))
            for _ in range(n_neg - 6)
        ]
        negative = [s for s in negative if not s.endswith("ab")][:n_neg]
        return positive, negative

    @staticmethod
    def regular_even_a(n_pos: int, n_neg: int) -> Tuple[List[str], List[str]]:
        """Generate examples for strings with even number of 'a's."""
        positive = []
        for _ in range(n_pos):
            n_a = random.randint(0, 4) * 2  # Even number
            n_b = random.randint(0, 4)
            s = list("a" * n_a + "b" * n_b)
            random.shuffle(s)
            positive.append("".join(s))

        negative = []
        for _ in range(n_neg):
            n_a = random.randint(0, 4) * 2 + 1  # Odd number
            n_b = random.randint(0, 4)
            s = list("a" * n_a + "b" * n_b)
            random.shuffle(s)
            negative.append("".join(s))

        return positive, negative

    @staticmethod
    def cfg_balanced_parens(n_pos: int) -> List[str]:
        """Generate balanced parentheses examples."""

        def generate(depth: int, max_depth: int) -> str:
            if depth >= max_depth or random.random() < 0.3:
                return ""
            inner = generate(depth + 1, max_depth)
            if random.random() < 0.5:
                return f"({inner})"
            else:
                left = generate(depth + 1, max_depth)
                right = generate(depth + 1, max_depth)
                return f"({left}){right}"

        examples = set()
        for _ in range(n_pos * 3):
            ex = generate(0, 4)
            if ex:
                examples.add(ex)
            if len(examples) >= n_pos:
                break

        return list(examples)

    @staticmethod
    def cfg_anbn(n_pos: int) -> List[str]:
        """Generate a^n b^n examples."""
        return ["a" * i + "b" * i for i in range(1, n_pos + 1)]


class AlgorithmBenchmark:
    """Benchmark different induction algorithms."""

    def __init__(self):
        self.suite = BenchmarkSuite()

    def run_dfa_comparison(
        self,
        positive: List[str],
        negative: List[str],
        language_name: str,
    ) -> Dict[str, BenchmarkResult]:
        """Compare DFA induction algorithms on same data."""
        results = {}

        algorithms = [
            (InductionAlgorithm.RPNI, "rpni"),
            (InductionAlgorithm.L_STAR, "lstar"),
        ]

        for algo_enum, algo_name in algorithms:
            t0 = time.time()

            try:
                inducer = DFAInducer(algorithm=algo_enum)
                induced = inducer.induce(positive, negative)

                result = BenchmarkResult(
                    name=language_name,
                    algorithm=algo_name,
                    n_positive=len(positive),
                    n_negative=len(negative),
                    n_states_learned=induced.n_states,
                    accuracy=induced.accuracy,
                    time_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                result = BenchmarkResult(
                    name=language_name,
                    algorithm=algo_name,
                    n_positive=len(positive),
                    n_negative=len(negative),
                    n_states_learned=0,
                    accuracy=0.0,
                    time_ms=(time.time() - t0) * 1000,
                    metadata={"error": str(e)},
                )

            results[algo_name] = result
            self.suite.add_result(result)

        return results

    def run_scaling_benchmark(
        self,
        generator_fn,
        language_name: str,
        sizes: List[int] = None,
    ) -> List[BenchmarkResult]:
        """Test how algorithms scale with example count."""
        sizes = sizes or [5, 10, 20, 50, 100]
        results = []

        for size in sizes:
            positive, negative = generator_fn(size, size)

            t0 = time.time()
            inducer = DFAInducer(algorithm=InductionAlgorithm.RPNI)
            induced = inducer.induce(positive, negative)

            result = BenchmarkResult(
                name=f"{language_name}_scale_{size}",
                algorithm="rpni",
                n_positive=len(positive),
                n_negative=len(negative),
                n_states_learned=induced.n_states,
                accuracy=induced.accuracy,
                time_ms=(time.time() - t0) * 1000,
                metadata={"scale": size},
            )
            results.append(result)
            self.suite.add_result(result)

        return results


class ClassDetectionBenchmark:
    """Benchmark language class detection accuracy."""

    def __init__(self):
        self.suite = BenchmarkSuite()

    def run(self) -> List[BenchmarkResult]:
        """Run language class detection benchmark."""
        results = []

        # Test cases with known classifications
        test_cases = [
            # (positive, negative, expected_type, name)
            (
                ["ab", "aab", "bab", "aaab"],
                ["", "a", "ba"],
                LanguageType.REGULAR,
                "ends_ab",
            ),
            (
                ["()", "(())", "((()))", "(()())"],
                None,
                LanguageType.CONTEXT_FREE,
                "balanced_parens",
            ),
            (
                ["a", "aa", "aaa", "aaaa"],
                ["", "b", "ab"],
                LanguageType.REGULAR,
                "a_plus",
            ),
            (
                ["ab", "aabb", "aaabbb"],
                None,
                LanguageType.CONTEXT_FREE,
                "anbn",
            ),
        ]

        learner = UnifiedLearner()

        for positive, negative, expected, name in test_cases:
            t0 = time.time()
            result = learner.learn(positive, negative)

            correct = result.language_type == expected
            accuracy = 1.0 if correct else 0.0

            bench_result = BenchmarkResult(
                name=f"detection_{name}",
                algorithm="unified",
                n_positive=len(positive),
                n_negative=len(negative) if negative else 0,
                n_states_learned=result.n_states or result.n_productions,
                accuracy=accuracy,
                time_ms=(time.time() - t0) * 1000,
                metadata={
                    "expected": expected.value,
                    "detected": result.language_type.value,
                    "correct": correct,
                },
            )
            results.append(bench_result)
            self.suite.add_result(bench_result)

        return results


class StatechartConversionBenchmark:
    """Benchmark quality of statechart conversion."""

    def __init__(self):
        self.suite = BenchmarkSuite()

    def run(self) -> List[BenchmarkResult]:
        """Run statechart conversion benchmark."""
        results = []

        # Learn some languages and convert to statecharts
        test_cases = [
            (["ab", "aab", "bab"], ["", "a", "ba"], "regular"),
            (["()", "(())"], None, "cfg"),
        ]

        for positive, negative, case_name in test_cases:
            t0 = time.time()

            result = learn_from_examples(positive, negative)
            sc = result.to_statechart()

            # Quality metrics
            has_root = "root_state" in sc
            has_transitions = len(sc.get("transitions", [])) > 0 or result.cfg is not None
            has_states = len(sc.get("root_state", {}).get("children", [])) > 0

            quality_score = (
                (0.4 if has_root else 0) +
                (0.3 if has_transitions else 0) +
                (0.3 if has_states else 0)
            )

            bench_result = BenchmarkResult(
                name=f"conversion_{case_name}",
                algorithm="unified",
                n_positive=len(positive),
                n_negative=len(negative) if negative else 0,
                n_states_learned=len(sc.get("root_state", {}).get("children", [])),
                accuracy=quality_score,
                time_ms=(time.time() - t0) * 1000,
                metadata={
                    "has_root": has_root,
                    "has_transitions": has_transitions,
                    "has_states": has_states,
                },
            )
            results.append(bench_result)
            self.suite.add_result(bench_result)

        return results


def run_benchmark(verbose: bool = True) -> BenchmarkSuite:
    """Run all benchmarks."""
    master_suite = BenchmarkSuite()

    if verbose:
        print("=" * 70)
        print("FORMAL LANGUAGE SUITE: Comprehensive Benchmarks")
        print("=" * 70)

    # 1. Algorithm comparison
    if verbose:
        print("\n--- Algorithm Comparison ---")

    algo_bench = AlgorithmBenchmark()

    pos, neg = LanguageGenerator.regular_ends_ab(20, 20)
    results = algo_bench.run_dfa_comparison(pos, neg, "ends_ab")

    if verbose:
        for name, r in results.items():
            print(f"  {name}: {r.n_states_learned} states, "
                  f"{r.accuracy:.1%} accuracy, {r.time_ms:.1f}ms")

    master_suite.results.extend(algo_bench.suite.results)

    # 2. Scaling benchmark
    if verbose:
        print("\n--- Scaling Benchmark ---")

    scaling_results = algo_bench.run_scaling_benchmark(
        LanguageGenerator.regular_even_a,
        "even_a",
        sizes=[5, 10, 20, 50],
    )

    if verbose:
        for r in scaling_results:
            print(f"  n={r.metadata['scale']}: {r.n_states_learned} states, "
                  f"{r.time_ms:.1f}ms")

    # 3. Class detection
    if verbose:
        print("\n--- Language Class Detection ---")

    detect_bench = ClassDetectionBenchmark()
    detect_results = detect_bench.run()

    if verbose:
        correct = sum(1 for r in detect_results if r.metadata.get("correct"))
        print(f"  Detection accuracy: {correct}/{len(detect_results)}")
        for r in detect_results:
            status = "✓" if r.metadata.get("correct") else "✗"
            print(f"    {status} {r.name}: expected={r.metadata['expected']}, "
                  f"detected={r.metadata['detected']}")

    master_suite.results.extend(detect_bench.suite.results)

    # 4. Statechart conversion
    if verbose:
        print("\n--- Statechart Conversion ---")

    sc_bench = StatechartConversionBenchmark()
    sc_results = sc_bench.run()

    if verbose:
        for r in sc_results:
            print(f"  {r.name}: quality={r.accuracy:.1%}")

    master_suite.results.extend(sc_bench.suite.results)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        summary = master_suite.summary()
        for algo, stats in summary.items():
            print(f"\n{algo}:")
            print(f"  Benchmarks: {stats['count']}")
            print(f"  Avg Accuracy: {stats['avg_accuracy']:.1%}")
            print(f"  Avg Time: {stats['avg_time_ms']:.1f}ms")
            print(f"  Avg States: {stats['avg_states']:.1f}")

    return master_suite


def demo():
    """Quick benchmark demo."""
    print("Running benchmark demo (subset)...")
    return run_benchmark(verbose=True)


if __name__ == "__main__":
    demo()
