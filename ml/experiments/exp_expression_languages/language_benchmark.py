"""
Language Benchmark - Compare expression languages across multiple dimensions.

Benchmarks:
1. Parse performance: Time to parse expressions
2. Evaluation performance: Time to evaluate expressions
3. Expressiveness: What can be expressed in each language
4. Safety: Security characteristics
5. Equivalence: Cross-language compatibility

Goal: Help users choose the right expression language for their use case.
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

from .expression_parser import (
    ExpressionLanguage,
    ParsedExpression,
    MultiLanguageParser,
)
from .guard_equivalence import (
    GuardEquivalenceChecker,
    EquivalenceLevel,
)
from .security_tester import (
    ExpressionSecurityTester,
    SecurityConfig,
)


@dataclass
class LanguageMetrics:
    """Performance and capability metrics for a language."""
    language: ExpressionLanguage
    parse_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    n_parsed: int = 0
    n_evaluated: int = 0
    n_parse_errors: int = 0
    n_eval_errors: int = 0
    avg_depth: float = 0.0
    security_pass_rate: float = 0.0
    equivalence_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'language': self.language.name,
            'parse_time_ms': self.parse_time_ms,
            'eval_time_ms': self.eval_time_ms,
            'n_parsed': self.n_parsed,
            'parse_error_rate': self.n_parse_errors / max(self.n_parsed, 1),
            'eval_error_rate': self.n_eval_errors / max(self.n_evaluated, 1),
            'avg_depth': self.avg_depth,
            'security_pass_rate': self.security_pass_rate,
            'equivalence_rate': self.equivalence_rate,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    metrics: Dict[ExpressionLanguage, LanguageMetrics]
    equivalence_matrix: Dict[Tuple[ExpressionLanguage, ExpressionLanguage], float]
    recommendations: List[str]

    def get_ranking(self, metric: str) -> List[ExpressionLanguage]:
        """Rank languages by a metric (higher is better, except for time)."""
        items = [(lang, getattr(m, metric)) for lang, m in self.metrics.items()]

        # For time metrics, lower is better
        if 'time' in metric:
            items.sort(key=lambda x: x[1])
        else:
            items.sort(key=lambda x: -x[1])

        return [lang for lang, _ in items]


class ExpressionGenerator:
    """Generate benchmark expressions."""

    TEMPLATES = {
        'simple_compare': [
            ('x > 0', 'x > 0', 'x > 0', 'x > 0', 'x > 0'),
            ('y == 10', 'y == 10', 'y == 10', 'y === 10', 'y == 10'),
            ('z != 5', 'z != 5', 'z != 5', 'z !== 5', 'z != 5'),
        ],
        'logical_and': [
            ('x > 0 and y < 10', 'x > 0 && y < 10', 'x > 0 and y < 10', 'x > 0 && y < 10', 'x > 0 && y < 10'),
        ],
        'logical_or': [
            ('x == 0 or y == 0', 'x == 0 || y == 0', 'x == 0 or y == 0', 'x === 0 || y === 0', 'x == 0 || y == 0'),
        ],
        'negation': [
            ('not flag', '!flag', 'not flag', '!flag', '!flag'),
        ],
        'complex': [
            ('(x > 0 and y > 0) or z == 0', '(x > 0 && y > 0) || z == 0',
             '(x > 0 and y > 0) or z == 0', '(x > 0 && y > 0) || z === 0', '(x > 0 && y > 0) || z == 0'),
        ],
        'nested': [
            ('(a or (b and (c or d)))', '(a || (b && (c || d)))',
             '(a or (b and (c or d)))', '(a || (b && (c || d)))', '(a || (b && (c || d)))'),
        ],
        'function_call': [
            ('len(items) > 0', 'size(items) > 0', 'len(items) > 0',
             'items.length > 0', 'len(items) > 0'),
        ],
    }

    LANGUAGES = [
        ExpressionLanguage.RAW,
        ExpressionLanguage.CEL,
        ExpressionLanguage.STARLARK,
        ExpressionLanguage.JAVASCRIPT,
        ExpressionLanguage.GO,
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_suite(self) -> Dict[str, Dict[ExpressionLanguage, str]]:
        """Generate full benchmark suite."""
        suite = {}

        for category, templates in self.TEMPLATES.items():
            for i, exprs in enumerate(templates):
                name = f"{category}_{i}"
                suite[name] = dict(zip(self.LANGUAGES, exprs))

        return suite

    def generate_random_context(self) -> Dict[str, Any]:
        """Generate random evaluation context."""
        return {
            'x': self.rng.randint(-10, 10),
            'y': self.rng.randint(-10, 10),
            'z': self.rng.randint(-10, 10),
            'a': self.rng.choice([True, False]),
            'b': self.rng.choice([True, False]),
            'c': self.rng.choice([True, False]),
            'd': self.rng.choice([True, False]),
            'flag': self.rng.choice([True, False]),
            'items': [1, 2, 3] if self.rng.random() > 0.5 else [],
        }


class ParseBenchmark:
    """Benchmark parsing performance."""

    def __init__(self, parser: MultiLanguageParser):
        self.parser = parser

    def benchmark(
        self,
        expressions: Dict[str, Dict[ExpressionLanguage, str]],
        n_iterations: int = 100,
    ) -> Dict[ExpressionLanguage, LanguageMetrics]:
        """Benchmark parsing across languages."""
        metrics = {lang: LanguageMetrics(language=lang) for lang in ExpressionLanguage}

        for lang in ExpressionLanguage:
            total_time = 0.0
            total_depth = 0.0
            n_parsed = 0
            n_errors = 0

            for _ in range(n_iterations):
                for name, exprs in expressions.items():
                    if lang not in exprs:
                        continue

                    expr = exprs[lang]
                    start = time.perf_counter()
                    parsed = self.parser.parse(expr, lang)
                    elapsed = time.perf_counter() - start

                    total_time += elapsed * 1000  # Convert to ms
                    n_parsed += 1

                    if parsed.is_valid:
                        total_depth += parsed.depth
                    else:
                        n_errors += 1

            metrics[lang].parse_time_ms = total_time
            metrics[lang].n_parsed = n_parsed
            metrics[lang].n_parse_errors = n_errors
            metrics[lang].avg_depth = total_depth / max(n_parsed - n_errors, 1)

        return metrics


class EvalBenchmark:
    """Benchmark evaluation performance."""

    def __init__(self, parser: MultiLanguageParser, seed: int = 42):
        self.parser = parser
        self.gen = ExpressionGenerator(seed)

    def benchmark(
        self,
        expressions: Dict[str, Dict[ExpressionLanguage, str]],
        n_iterations: int = 100,
    ) -> Dict[ExpressionLanguage, LanguageMetrics]:
        """Benchmark evaluation across languages."""
        metrics = {lang: LanguageMetrics(language=lang) for lang in ExpressionLanguage}

        for lang in ExpressionLanguage:
            total_time = 0.0
            n_evaluated = 0
            n_errors = 0

            for _ in range(n_iterations):
                context = self.gen.generate_random_context()

                for name, exprs in expressions.items():
                    if lang not in exprs:
                        continue

                    expr = exprs[lang]
                    start = time.perf_counter()
                    result, error = self.parser.evaluate(expr, lang, context)
                    elapsed = time.perf_counter() - start

                    total_time += elapsed * 1000
                    n_evaluated += 1

                    if error:
                        n_errors += 1

            metrics[lang].eval_time_ms = total_time
            metrics[lang].n_evaluated = n_evaluated
            metrics[lang].n_eval_errors = n_errors

        return metrics


class EquivalenceBenchmark:
    """Benchmark cross-language equivalence."""

    def __init__(self, seed: int = 42):
        self.checker = GuardEquivalenceChecker(seed)

    def benchmark(
        self,
        expressions: Dict[str, Dict[ExpressionLanguage, str]],
    ) -> Tuple[Dict[ExpressionLanguage, float], Dict[Tuple[ExpressionLanguage, ExpressionLanguage], float]]:
        """
        Benchmark equivalence across languages.
        Returns (per-language rates, pairwise matrix).
        """
        pairwise_matches = defaultdict(list)
        per_language = defaultdict(list)

        for name, exprs in expressions.items():
            langs = list(exprs.keys())

            for i, lang1 in enumerate(langs):
                for lang2 in langs[i+1:]:
                    result = self.checker.check_equivalence(
                        exprs[lang1], lang1,
                        exprs[lang2], lang2,
                        n_test_samples=50,
                    )

                    is_equiv = result.is_equivalent()
                    pairwise_matches[(lang1, lang2)].append(1.0 if is_equiv else 0.0)

                    per_language[lang1].append(1.0 if is_equiv else 0.0)
                    per_language[lang2].append(1.0 if is_equiv else 0.0)

        # Compute averages
        per_lang_rates = {lang: sum(v) / len(v) if v else 0.0
                        for lang, v in per_language.items()}
        pairwise_rates = {pair: sum(v) / len(v) if v else 0.0
                        for pair, v in pairwise_matches.items()}

        return per_lang_rates, pairwise_rates


class SecurityBenchmark:
    """Benchmark security characteristics."""

    def __init__(self, config: SecurityConfig = None):
        self.tester = ExpressionSecurityTester(config)

    def benchmark(
        self,
        expressions: Dict[str, Dict[ExpressionLanguage, str]],
    ) -> Dict[ExpressionLanguage, float]:
        """Benchmark security pass rates."""
        results = defaultdict(list)

        for name, exprs in expressions.items():
            for lang, expr in exprs.items():
                result = self.tester.test_expression(expr, lang)
                results[lang].append(1.0 if result.passed else 0.0)

        return {lang: sum(v) / len(v) if v else 0.0
               for lang, v in results.items()}


class LanguageBenchmark:
    """Main benchmark orchestrator."""

    def __init__(self, seed: int = 42):
        self.parser = MultiLanguageParser()
        self.gen = ExpressionGenerator(seed)
        self.parse_bench = ParseBenchmark(self.parser)
        self.eval_bench = EvalBenchmark(self.parser, seed)
        self.equiv_bench = EquivalenceBenchmark(seed)
        self.security_bench = SecurityBenchmark()

    def run(
        self,
        n_parse_iterations: int = 100,
        n_eval_iterations: int = 100,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run complete benchmark suite."""
        expressions = self.gen.generate_suite()

        if verbose:
            print("=" * 70)
            print("EXPRESSION LANGUAGE BENCHMARK")
            print("=" * 70)
            print(f"\nBenchmark suite: {len(expressions)} expression templates")

        # Parse benchmark
        if verbose:
            print("\n[1/4] Parse performance...")
        parse_metrics = self.parse_bench.benchmark(expressions, n_parse_iterations)

        # Eval benchmark
        if verbose:
            print("[2/4] Evaluation performance...")
        eval_metrics = self.eval_bench.benchmark(expressions, n_eval_iterations)

        # Equivalence benchmark
        if verbose:
            print("[3/4] Cross-language equivalence...")
        equiv_rates, equiv_matrix = self.equiv_bench.benchmark(expressions)

        # Security benchmark
        if verbose:
            print("[4/4] Security characteristics...")
        security_rates = self.security_bench.benchmark(expressions)

        # Combine metrics
        combined = {}
        for lang in ExpressionLanguage:
            m = LanguageMetrics(language=lang)
            if lang in parse_metrics:
                m.parse_time_ms = parse_metrics[lang].parse_time_ms
                m.n_parsed = parse_metrics[lang].n_parsed
                m.n_parse_errors = parse_metrics[lang].n_parse_errors
                m.avg_depth = parse_metrics[lang].avg_depth
            if lang in eval_metrics:
                m.eval_time_ms = eval_metrics[lang].eval_time_ms
                m.n_evaluated = eval_metrics[lang].n_evaluated
                m.n_eval_errors = eval_metrics[lang].n_eval_errors
            m.equivalence_rate = equiv_rates.get(lang, 0.0)
            m.security_pass_rate = security_rates.get(lang, 0.0)
            combined[lang] = m

        # Generate recommendations
        recommendations = self._generate_recommendations(combined, equiv_matrix)

        result = BenchmarkResult(
            metrics=combined,
            equivalence_matrix=equiv_matrix,
            recommendations=recommendations,
        )

        if verbose:
            self._print_results(result)

        return result

    def _generate_recommendations(
        self,
        metrics: Dict[ExpressionLanguage, LanguageMetrics],
        equiv_matrix: Dict,
    ) -> List[str]:
        """Generate recommendations based on results."""
        recs = []

        # Find best performers
        fastest_parse = min(metrics.items(), key=lambda x: x[1].parse_time_ms)[0]
        fastest_eval = min(metrics.items(), key=lambda x: x[1].eval_time_ms)[0]
        most_secure = max(metrics.items(), key=lambda x: x[1].security_pass_rate)[0]
        most_compatible = max(metrics.items(), key=lambda x: x[1].equivalence_rate)[0]

        recs.append(f"Fastest parsing: {fastest_parse.name}")
        recs.append(f"Fastest evaluation: {fastest_eval.name}")
        recs.append(f"Best security: {most_secure.name}")
        recs.append(f"Best cross-language compatibility: {most_compatible.name}")

        # Use case recommendations
        recs.append("")
        recs.append("Use case recommendations:")
        recs.append("  - Performance-critical: Use RAW (Python-like) syntax")
        recs.append("  - Multi-platform: Use CEL for best standardization")
        recs.append("  - Configuration: Use Starlark for safety + expressiveness")
        recs.append("  - Web integration: JavaScript for browser compatibility")

        return recs

    def _print_results(self, result: BenchmarkResult):
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        # Performance table
        print(f"\n{'Language':<12} {'Parse(ms)':<12} {'Eval(ms)':<12} {'Errors':<10} {'Security':<10} {'Compat':<10}")
        print("-" * 70)

        for lang, m in result.metrics.items():
            errors = f"{m.n_parse_errors + m.n_eval_errors}"
            print(f"{lang.name:<12} {m.parse_time_ms:<12.2f} {m.eval_time_ms:<12.2f} "
                  f"{errors:<10} {m.security_pass_rate*100:<10.0f}% {m.equivalence_rate*100:<10.0f}%")

        # Equivalence matrix
        print("\n" + "-" * 70)
        print("EQUIVALENCE MATRIX")
        print("-" * 70)

        langs = list(result.metrics.keys())
        header = "          " + " ".join(f"{l.name[:6]:<8}" for l in langs)
        print(header)

        for lang1 in langs:
            row = f"{lang1.name[:8]:<10}"
            for lang2 in langs:
                if lang1 == lang2:
                    row += f"{'---':<8}"
                else:
                    key = (lang1, lang2) if (lang1, lang2) in result.equivalence_matrix else (lang2, lang1)
                    rate = result.equivalence_matrix.get(key, 0.0)
                    row += f"{rate*100:<8.0f}"
            print(row)

        # Recommendations
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        for rec in result.recommendations:
            print(f"  {rec}")

        print("=" * 70)


def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("LANGUAGE BENCHMARK DEMO")
    print("=" * 60)

    benchmark = LanguageBenchmark(seed=42)
    result = benchmark.run(
        n_parse_iterations=50,
        n_eval_iterations=50,
        verbose=True,
    )

    return result


if __name__ == "__main__":
    demo()
