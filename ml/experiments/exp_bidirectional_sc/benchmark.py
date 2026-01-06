"""
Bidirectional Benchmark: Run full evaluation and report results.

Report format: BIDIRECTIONAL: gen_validity=X%, explain_clarity=X%
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import time

from .dataset import create_dataset, BidirectionalDataset
from .evaluator import BidirectionalEvaluator, EvaluationResult


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmark run."""
    n_test_samples: int = 20  # Number of test samples
    max_gen_tokens: int = 400
    max_explain_tokens: int = 200
    temperature: float = 0.7


def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run full bidirectional benchmark.
    
    Args:
        config: Benchmark configuration
        verbose: Print progress
        
    Returns:
        EvaluationResult with gen_validity and explain_clarity
    """
    config = config or BenchmarkConfig()
    
    if verbose:
        print("=" * 60)
        print("BIDIRECTIONAL SC BENCHMARK")
        print("=" * 60)
        print(f"Test samples: {config.n_test_samples}")
    
    # Load dataset from benchmark
    dataset = create_dataset()
    
    if verbose:
        print(f"Loaded {len(dataset)} pairs from benchmark")
    
    # Select test samples
    test_pairs = [(p.description, p.statechart) 
                  for p in dataset.pairs[:config.n_test_samples]]
    
    if verbose:
        print(f"\nRunning evaluation on {len(test_pairs)} pairs...")
        print("-" * 60)
    
    # Run evaluation
    start_time = time.time()
    evaluator = BidirectionalEvaluator()
    result = evaluator.evaluate(test_pairs, verbose=verbose)
    eval_time = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(result.summary())
        print(f"Evaluation time: {eval_time:.1f}s")
    
    return result


def format_report(result: EvaluationResult) -> str:
    """Format result for orchestrator report."""
    return (
        f"BIDIRECTIONAL: "
        f"gen_validity={result.gen_validity:.0%}, "
        f"explain_clarity={result.explain_clarity:.0%}"
    )


def quick_benchmark(n_samples: int = 10, verbose: bool = True) -> EvaluationResult:
    """Run quick benchmark with fewer samples."""
    config = BenchmarkConfig(n_test_samples=n_samples)
    return run_benchmark(config, verbose)


def full_benchmark(verbose: bool = True) -> EvaluationResult:
    """Run full benchmark on all 100 tasks."""
    config = BenchmarkConfig(n_test_samples=100)
    return run_benchmark(config, verbose)


def demo():
    """Run quick demo."""
    print("Running quick bidirectional benchmark...")
    result = quick_benchmark(n_samples=5)
    report = format_report(result)
    print(f"\nReport: {report}")
    return result, report


if __name__ == "__main__":
    result, report = demo()
    print(f"\n{report}")
