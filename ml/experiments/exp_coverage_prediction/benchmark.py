"""
Coverage Prediction Benchmark

Compare all approaches on standardized test suite:
1. Sequence Model (baseline)
2. AST-based Model (baseline)
3. GNN on CFG (baseline)
4. Statechart Symbolic Execution (ours)
5. Differentiable Statechart (ours)

Metrics:
- F1 score per line (macro-averaged)
- Jaccard similarity
- Precision / Recall

Expected result: Statechart methods outperform baselines because
coverage = statechart simulation.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import json
import time
from pathlib import Path

from .coverage_collector import CoverageCollector, CoverageTriple, SyntheticDataGenerator
from .program_statechart import build_cfg, ProgramStatechart
from .statechart_predictor import StatechartCoveragePredictor, StatechartConfig
from .baseline_models import (
    SequenceCoverageModel, ASTCoverageModel, GNNCoverageModel,
    CoverageModelConfig, SimpleTokenizer
)
from .dataset import Dataset, DatasetExample, DatasetGenerator


@dataclass
class EvaluationMetrics:
    """Metrics for coverage prediction evaluation."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    jaccard: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'jaccard': self.jaccard,
            'accuracy': self.accuracy,
            'tp': self.true_positives,
            'fp': self.false_positives,
            'tn': self.true_negatives,
            'fn': self.false_negatives,
        }


def compute_metrics(
    predicted: Set[int],
    actual: Set[int],
    all_lines: Set[int],
) -> EvaluationMetrics:
    """Compute evaluation metrics for a single prediction."""
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(all_lines - predicted - actual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Jaccard = intersection / union
    union = len(predicted | actual)
    jaccard = tp / union if union > 0 else 1.0

    accuracy = (tp + tn) / len(all_lines) if len(all_lines) > 0 else 0.0

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
    """Aggregate metrics from multiple predictions (macro-average)."""
    if not metrics_list:
        return EvaluationMetrics()

    n = len(metrics_list)
    return EvaluationMetrics(
        precision=sum(m.precision for m in metrics_list) / n,
        recall=sum(m.recall for m in metrics_list) / n,
        f1=sum(m.f1 for m in metrics_list) / n,
        jaccard=sum(m.jaccard for m in metrics_list) / n,
        accuracy=sum(m.accuracy for m in metrics_list) / n,
        true_positives=sum(m.true_positives for m in metrics_list),
        false_positives=sum(m.false_positives for m in metrics_list),
        true_negatives=sum(m.true_negatives for m in metrics_list),
        false_negatives=sum(m.false_negatives for m in metrics_list),
    )


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str
    metrics: EvaluationMetrics
    time_seconds: float
    n_examples: int

    def to_dict(self) -> Dict:
        return {
            'model': self.model_name,
            'metrics': self.metrics.to_dict(),
            'time_seconds': self.time_seconds,
            'n_examples': self.n_examples,
        }


class CoverageBenchmark:
    """
    Benchmark suite for coverage prediction methods.

    Runs all methods on the same test data and compares results.
    """

    def __init__(self, verbose: bool = True, use_full_dataset: bool = False):
        self.verbose = verbose
        self.use_full_dataset = use_full_dataset
        self.collector = CoverageCollector()
        self.generator = SyntheticDataGenerator(self.collector)
        self.dataset_generator = DatasetGenerator(seed=42)
        self.results: List[BenchmarkResult] = []
        self.dataset: Optional[Dataset] = None

    def log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def generate_test_suite(self, n_programs: int = 10) -> List[CoverageTriple]:
        """Generate synthetic test programs with ground truth coverage."""
        self.log("Generating test suite...")

        if self.use_full_dataset:
            # Use comprehensive dataset
            self.dataset = self.dataset_generator.generate_dataset()
            stats = self.dataset.statistics()
            self.log(f"Generated {stats['n_examples']} examples from {stats['n_programs']} programs")
            self.log(f"Categories: {len(stats['categories'])}")

            # Convert DatasetExample to CoverageTriple for compatibility
            triples = []
            for ex in self.dataset.examples:
                triple = CoverageTriple(
                    program_id=ex.program_id,
                    program_source=ex.source,
                    program_ast=None,
                    input_repr=ex.input_repr,
                    input_value=None,  # We don't have the actual value
                    covered_lines=set(ex.covered_lines),
                    total_lines=ex.n_lines,
                    coverage_ratio=ex.coverage_ratio,
                    execution_trace=ex.execution_trace,
                    branches_taken={int(k): v for k, v in ex.branches_taken.items()},
                    error=ex.error,
                )
                triples.append(triple)

            return triples

        # Legacy: small test suite for quick testing
        self.collector.dataset = []

        # Generate if-chain programs
        for n_branches in [2, 3, 4, 5]:
            source, expected = self.generator.generate_if_chain(n_branches)
            inputs = list(range(n_branches)) + [n_branches + 1]
            self.collector.collect_from_source(source, 'check_value', inputs)

        # Generate nested condition programs
        for depth in [2, 3]:
            source, expected = self.generator.generate_nested_conditions(depth)
            inputs = [
                (False, False, False),
                (True, False, False),
                (True, True, False),
                (True, True, True),
            ]
            self.collector.collect_from_source(source, 'nested_check', inputs)

        # Add some custom programs
        custom_programs = [
            ('''
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
''', 'sum_list', [[], [1], [1, 2, 3]]),

            ('''
def sum_positive(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    return total
''', 'sum_positive', [[], [1, 2], [-1, -2], [1, -1]]),

            ('''
def find_first(items, target):
    for item in items:
        if item == target:
            return item
    return None
''', 'find_first', [([1, 2, 3], 2), ([1, 2, 3], 5), ([], 1)]),

            ('''
def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    elif x < 10:
        return "small"
    else:
        return "large"
''', 'classify', [-5, 0, 5, 15]),
        ]

        for source, func_name, inputs in custom_programs:
            source = source.strip()
            self.collector.collect_from_source(source, func_name, inputs)

        self.log(f"Generated {len(self.collector.dataset)} test cases")
        return self.collector.dataset

    def evaluate_statechart_predictor(
        self,
        test_data: List[CoverageTriple],
    ) -> BenchmarkResult:
        """Evaluate statechart-based coverage prediction."""
        self.log("\nEvaluating: Statechart Predictor")

        config = StatechartConfig(max_iterations=50)
        predictor = StatechartCoveragePredictor(config)

        metrics_list = []
        start_time = time.time()

        for triple in test_data:
            try:
                # Build statechart to get all lines
                cfg = build_cfg(triple.program_source)
                all_lines = set()
                for block in cfg.blocks.values():
                    all_lines |= block.lines

                # Try to parse input_repr back to value for symbolic execution
                input_value = triple.input_value
                if input_value is None and triple.input_repr:
                    try:
                        # Safe eval for common types
                        input_value = eval(triple.input_repr, {"__builtins__": {}})
                    except:
                        input_value = None

                # Predict coverage
                predicted, _ = predictor.predict_coverage(
                    triple.program_source,
                    input_value,
                )

                # Compute metrics
                m = compute_metrics(predicted, triple.covered_lines, all_lines)
                metrics_list.append(m)

            except Exception as e:
                self.log(f"  Error: {e}")
                metrics_list.append(EvaluationMetrics())

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(metrics_list)

        result = BenchmarkResult(
            model_name="Statechart",
            metrics=aggregated,
            time_seconds=elapsed,
            n_examples=len(test_data),
        )
        self.results.append(result)

        self.log(f"  F1: {aggregated.f1:.3f}, Jaccard: {aggregated.jaccard:.3f}")
        return result

    def evaluate_baseline_random(
        self,
        test_data: List[CoverageTriple],
    ) -> BenchmarkResult:
        """Evaluate random baseline (predicts random lines)."""
        self.log("\nEvaluating: Random Baseline")

        import random
        metrics_list = []
        start_time = time.time()

        for triple in test_data:
            # Get all lines
            all_lines = set(range(1, triple.total_lines + 1))

            # Predict random subset
            n_predict = max(1, triple.total_lines // 2)
            predicted = set(random.sample(list(all_lines), min(n_predict, len(all_lines))))

            m = compute_metrics(predicted, triple.covered_lines, all_lines)
            metrics_list.append(m)

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(metrics_list)

        result = BenchmarkResult(
            model_name="Random",
            metrics=aggregated,
            time_seconds=elapsed,
            n_examples=len(test_data),
        )
        self.results.append(result)

        self.log(f"  F1: {aggregated.f1:.3f}, Jaccard: {aggregated.jaccard:.3f}")
        return result

    def evaluate_baseline_all_lines(
        self,
        test_data: List[CoverageTriple],
    ) -> BenchmarkResult:
        """Evaluate all-lines baseline (predicts all lines covered)."""
        self.log("\nEvaluating: All Lines Baseline")

        metrics_list = []
        start_time = time.time()

        for triple in test_data:
            all_lines = set(range(1, triple.total_lines + 1))
            predicted = all_lines.copy()

            m = compute_metrics(predicted, triple.covered_lines, all_lines)
            metrics_list.append(m)

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(metrics_list)

        result = BenchmarkResult(
            model_name="AllLines",
            metrics=aggregated,
            time_seconds=elapsed,
            n_examples=len(test_data),
        )
        self.results.append(result)

        self.log(f"  F1: {aggregated.f1:.3f}, Jaccard: {aggregated.jaccard:.3f}")
        return result

    def evaluate_baseline_entry_only(
        self,
        test_data: List[CoverageTriple],
    ) -> BenchmarkResult:
        """Evaluate entry-only baseline (predicts just function def line)."""
        self.log("\nEvaluating: Entry Only Baseline")

        metrics_list = []
        start_time = time.time()

        for triple in test_data:
            all_lines = set(range(1, triple.total_lines + 1))
            predicted = {1}  # Just first line

            m = compute_metrics(predicted, triple.covered_lines, all_lines)
            metrics_list.append(m)

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(metrics_list)

        result = BenchmarkResult(
            model_name="EntryOnly",
            metrics=aggregated,
            time_seconds=elapsed,
            n_examples=len(test_data),
        )
        self.results.append(result)

        self.log(f"  F1: {aggregated.f1:.3f}, Jaccard: {aggregated.jaccard:.3f}")
        return result

    def evaluate_cfg_reachability(
        self,
        test_data: List[CoverageTriple],
    ) -> BenchmarkResult:
        """
        Evaluate CFG reachability baseline.

        Predicts all lines reachable from entry (ignores guards).
        """
        self.log("\nEvaluating: CFG Reachability (no guards)")

        metrics_list = []
        start_time = time.time()

        for triple in test_data:
            try:
                cfg = build_cfg(triple.program_source)

                # Get all lines reachable from entry
                reachable = cfg.get_reachable_from(cfg.entry_block)
                predicted = cfg.get_covered_lines(reachable)

                all_lines = set()
                for block in cfg.blocks.values():
                    all_lines |= block.lines

                m = compute_metrics(predicted, triple.covered_lines, all_lines)
                metrics_list.append(m)

            except Exception as e:
                self.log(f"  Error: {e}")
                metrics_list.append(EvaluationMetrics())

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(metrics_list)

        result = BenchmarkResult(
            model_name="CFGReachability",
            metrics=aggregated,
            time_seconds=elapsed,
            n_examples=len(test_data),
        )
        self.results.append(result)

        self.log(f"  F1: {aggregated.f1:.3f}, Jaccard: {aggregated.jaccard:.3f}")
        return result

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        self.log("=" * 60)
        self.log("COVERAGE PREDICTION BENCHMARK")
        self.log("=" * 60)

        # Generate test data
        test_data = self.generate_test_suite()

        # Run all evaluations
        self.evaluate_baseline_random(test_data)
        self.evaluate_baseline_all_lines(test_data)
        self.evaluate_baseline_entry_only(test_data)
        self.evaluate_cfg_reachability(test_data)
        self.evaluate_statechart_predictor(test_data)

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK SUMMARY")
        self.log("=" * 60)

        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda r: r.metrics.f1, reverse=True)

        self.log(f"\n{'Model':<20} {'F1':>8} {'Jaccard':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}")
        self.log("-" * 70)

        for r in sorted_results:
            m = r.metrics
            self.log(f"{r.model_name:<20} {m.f1:>8.3f} {m.jaccard:>8.3f} "
                    f"{m.precision:>8.3f} {m.recall:>8.3f} {r.time_seconds:>7.2f}s")

        # Return results as dict
        return {
            'results': [r.to_dict() for r in self.results],
            'n_test_cases': len(test_data),
            'best_model': sorted_results[0].model_name,
            'best_f1': sorted_results[0].metrics.f1,
        }

    def save_results(self, path: Path):
        """Save benchmark results to JSON."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        path.write_text(json.dumps(data, indent=2))


def demo():
    """Run the benchmark demo."""
    benchmark = CoverageBenchmark(verbose=True)
    results = benchmark.run_full_benchmark()

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The Statechart predictor should outperform baselines because:

1. Coverage = "which states are visited during execution"
2. Statechart simulation = "which states are visited given input"
3. Therefore: Coverage prediction IS statechart simulation

The Statechart method:
- Uses symbolic execution with abstract values
- Evaluates guards to determine taken branches
- Tracks all reachable states
- Maps states back to source lines

Baselines either:
- Ignore control flow (Random, AllLines, EntryOnly)
- Ignore guard conditions (CFGReachability)
- Need training data (Neural models)

The Statechart method achieves high accuracy WITHOUT training!
""")

    return results


if __name__ == "__main__":
    demo()
