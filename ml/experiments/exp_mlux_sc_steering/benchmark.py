"""
Steering Benchmark for Statechart Validity

Finds optimal layer and alpha for validity steering.

Grid search over:
- Layers: [6, 12, 18, 24] (or model-appropriate)
- Alpha: [0.25, 0.5, 1.0, 1.5, 2.0]

Target: +10% validity improvement with steering.

Outputs:
- Best (layer, alpha) configuration
- Validity improvement matrix
- Recommendations for production use
"""

import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    MLUX_AVAILABLE,
)

from .steering_vectors import (
    SteeringVectorComputer,
    SteeringVector,
    create_contrastive_pairs,
)

from .validity_tester import (
    ValidityTester,
    ValidityTestResult,
    SC_GENERATION_PROMPTS,
)


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for steering benchmark."""
    layers: List[int] = field(default_factory=lambda: [6, 12, 18])
    alphas: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    num_prompts: int = 5
    improvement_target: float = 0.10  # +10%
    verbose: bool = True


# =============================================================================
# BENCHMARK RESULT
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of benchmark run."""
    config: BenchmarkConfig
    results: List[ValidityTestResult]
    best_layer: int
    best_alpha: float
    best_improvement: float
    target_met: bool
    avg_baseline_rate: float
    avg_steered_rate: float
    elapsed_time: float

    def __repr__(self):
        return (f"BenchmarkResult(best=layer{self.best_layer}@{self.best_alpha}, "
                f"improvement={self.best_improvement:+.1%}, "
                f"target_met={self.target_met})")


# =============================================================================
# STEERING BENCHMARK
# =============================================================================

class SteeringBenchmark:
    """
    Benchmark for finding optimal steering configuration.

    Grid searches layer/alpha combinations to find best
    validity improvement for statechart generation.
    """

    IMPROVEMENT_TARGET = 0.10

    def __init__(
        self,
        model: HookedModelWrapper = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        config: BenchmarkConfig = None,
    ):
        self.model = model or load_model(model_name)
        self.config = config or BenchmarkConfig()
        self.tester = ValidityTester(model=self.model)
        self.results: List[ValidityTestResult] = []
        self.benchmark_result: Optional[BenchmarkResult] = None

    @property
    def has_mlux(self) -> bool:
        return self.model.has_interpretability

    def run(self, prompts: List[str] = None) -> BenchmarkResult:
        """
        Run benchmark grid search.

        Args:
            prompts: Generation prompts (default: first N from SC_GENERATION_PROMPTS)

        Returns:
            BenchmarkResult with best configuration
        """
        if prompts is None:
            prompts = SC_GENERATION_PROMPTS[:self.config.num_prompts]

        if self.config.verbose:
            print("=" * 70)
            print("STEERING BENCHMARK")
            print("=" * 70)
            print(f"Layers: {self.config.layers}")
            print(f"Alphas: {self.config.alphas}")
            print(f"Prompts: {len(prompts)}")
            print(f"Target improvement: +{self.config.improvement_target:.0%}")
            print(f"MLUX available: {self.has_mlux}")
            print()

        start_time = time.time()
        self.results = []

        total_configs = len(self.config.layers) * len(self.config.alphas)
        current = 0

        for layer in self.config.layers:
            for alpha in self.config.alphas:
                current += 1
                if self.config.verbose:
                    print(f"\n[{current}/{total_configs}] Testing layer={layer}, alpha={alpha}")

                result = self.tester.test_validity(prompts, layer, alpha)
                self.results.append(result)

        elapsed = time.time() - start_time

        # Find best configuration
        best = max(self.results, key=lambda r: r.improvement)

        # Compute averages
        avg_baseline = sum(r.baseline_validity_rate for r in self.results) / len(self.results)
        avg_steered = sum(r.steered_validity_rate for r in self.results) / len(self.results)

        self.benchmark_result = BenchmarkResult(
            config=self.config,
            results=self.results,
            best_layer=best.layer,
            best_alpha=best.alpha,
            best_improvement=best.improvement,
            target_met=best.improvement >= self.IMPROVEMENT_TARGET,
            avg_baseline_rate=avg_baseline,
            avg_steered_rate=avg_steered,
            elapsed_time=elapsed,
        )

        if self.config.verbose:
            self._print_summary()

        return self.benchmark_result

    def run_quick(self) -> BenchmarkResult:
        """Run quick benchmark with fewer configurations."""
        quick_config = BenchmarkConfig(
            layers=[12],
            alphas=[1.0],
            num_prompts=3,
            verbose=self.config.verbose,
        )
        self.config = quick_config
        return self.run()

    def _print_summary(self):
        """Print benchmark summary."""
        if not self.benchmark_result:
            return

        r = self.benchmark_result

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        # Results matrix
        print("\nImprovement Matrix (Layer x Alpha):")
        print("-" * 50)

        # Header
        header = "Layer\\Alpha"
        for alpha in self.config.alphas:
            header += f"  {alpha:>6}"
        print(header)
        print("-" * 50)

        # Rows
        for layer in self.config.layers:
            row = f"    {layer:>2}"
            for alpha in self.config.alphas:
                result = next(
                    (res for res in self.results
                     if res.layer == layer and res.alpha == alpha),
                    None
                )
                if result:
                    imp = result.improvement * 100
                    marker = "*" if result.target_met else ""
                    row += f"  {imp:>+5.1f}%{marker}"
                else:
                    row += "     N/A"
            print(row)

        print("-" * 50)
        print("* = meets +10% target")

        # Best configuration
        print(f"\nBest Configuration:")
        print(f"  Layer: {r.best_layer}")
        print(f"  Alpha: {r.best_alpha}")
        print(f"  Improvement: {r.best_improvement:+.1%}")

        # Averages
        print(f"\nAverage Rates:")
        print(f"  Baseline: {r.avg_baseline_rate:.1%}")
        print(f"  Steered: {r.avg_steered_rate:.1%}")
        print(f"  Overall improvement: {r.avg_steered_rate - r.avg_baseline_rate:+.1%}")

        # Target check
        print(f"\nTarget (+10% improvement):")
        if r.target_met:
            print(f"  [PASS] Best config achieves {r.best_improvement:+.1%}")
        else:
            print(f"  [FAIL] Best improvement is {r.best_improvement:+.1%}")
            if not self.has_mlux:
                print("  Note: MLUX not available, steering has no effect in mock mode")

        print(f"\nTime: {r.elapsed_time:.1f}s")
        print("=" * 70)

    @property
    def passed(self) -> bool:
        """Check if benchmark target is met."""
        if not self.benchmark_result:
            return False
        return self.benchmark_result.target_met

    @property
    def best_config(self) -> Tuple[int, float]:
        """Get best (layer, alpha) configuration."""
        if not self.benchmark_result:
            return (12, 1.0)  # Default
        return (self.benchmark_result.best_layer, self.benchmark_result.best_alpha)

    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.benchmark_result:
            return {"completed": False}

        r = self.benchmark_result
        return {
            "completed": True,
            "total_configs": len(self.results),
            "best_layer": r.best_layer,
            "best_alpha": r.best_alpha,
            "best_improvement": r.best_improvement,
            "target_met": r.target_met,
            "avg_baseline_rate": r.avg_baseline_rate,
            "avg_steered_rate": r.avg_steered_rate,
            "elapsed_time": r.elapsed_time,
        }


# =============================================================================
# TESTING
# =============================================================================

def test_steering_benchmark():
    """Test steering benchmark."""
    print("=" * 60)
    print("STEERING BENCHMARK TEST")
    print("=" * 60)

    # Quick benchmark
    print("\nRunning quick benchmark...")
    benchmark = SteeringBenchmark()
    result = benchmark.run_quick()

    print(f"\nResult: {result}")
    print(f"Passed: {benchmark.passed}")
    print(f"Best config: {benchmark.best_config}")

    # Summary
    print("\nSummary:")
    for k, v in benchmark.summary().items():
        print(f"  {k}: {v}")

    print("\n[PASS] Steering benchmark test complete")
    return True


def run_full_benchmark():
    """Run full benchmark grid search."""
    config = BenchmarkConfig(
        layers=[6, 12, 18],
        alphas=[0.5, 1.0, 1.5],
        num_prompts=5,
        verbose=True,
    )

    benchmark = SteeringBenchmark(config=config)
    result = benchmark.run()

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        test_steering_benchmark()
    else:
        run_full_benchmark()
