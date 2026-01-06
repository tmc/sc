#!/usr/bin/env python3
"""
Benchmark: Test if head steering improves hierarchy validity.

Measures:
1. JSON validity rate
2. Hierarchy presence (nested states)
3. State count accuracy
4. Comparison across steering configs
"""

import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .head_amplifier import HeadAmplifier, AmplificationConfig
from .targeted_steering import (
    TargetedSteering,
    TargetedSteeringConfig,
    SteeringResult,
    CONFIGS,
)

from utils.mlux_loader import MLUX_AVAILABLE


@dataclass
class BenchmarkMetrics:
    """Metrics for a single configuration."""
    config_name: str
    num_samples: int = 0
    valid_json_count: int = 0
    has_hierarchy_count: int = 0
    total_states: int = 0
    total_nested: int = 0
    avg_generation_time: float = 0.0

    @property
    def validity_rate(self) -> float:
        return self.valid_json_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def hierarchy_rate(self) -> float:
        return self.has_hierarchy_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_states(self) -> float:
        return self.total_states / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_nested(self) -> float:
        return self.total_nested / self.num_samples if self.num_samples > 0 else 0


@dataclass
class BenchmarkResult:
    """Full benchmark results."""
    metrics: Dict[str, BenchmarkMetrics] = field(default_factory=dict)
    total_time: float = 0.0
    mlux_available: bool = False

    def get_best_config(self, metric: str = "hierarchy_rate") -> str:
        """Get config with best metric."""
        if not self.metrics:
            return "none"

        best_name = "baseline"
        best_value = 0.0

        for name, m in self.metrics.items():
            value = getattr(m, metric, 0)
            if value > best_value:
                best_value = value
                best_name = name

        return best_name


# Test prompts designed to encourage hierarchy
TEST_PROMPTS = [
    # Explicit hierarchy request
    '''Generate a hierarchical statechart for a media player with Playing containing Normal and FastForward substates:
```json
{"root_state": {"label": "__root__", "type": 2, "children": [''',

    # Nested state machine
    '''Generate a statechart for user authentication with LoggedIn containing Active and Idle substates:
```json
{"root_state": {"label": "__root__", "type": 2, "children": [''',

    # Game state with substates
    '''Generate a game statechart with Playing containing Exploring and Combat nested states:
```json
{"root_state": {"label": "__root__", "type": 2, "children": [''',

    # UI navigation with hierarchy
    '''Generate a UI navigation statechart with MainMenu containing Settings and Profile substates:
```json
{"root_state": {"label": "__root__", "type": 2, "children": [''',

    # Order processing with nested states
    '''Generate an order processing statechart with Processing containing Payment and Shipping substates:
```json
{"root_state": {"label": "__root__", "type": 2, "children": [''',
]


class SteeringBenchmark:
    """Benchmarks steering effectiveness for hierarchy generation."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("STEERING BENCHMARK")
        print("=" * 60)

        self.steering = TargetedSteering(model_name)

        print("=" * 60)

    def run_benchmark(
        self,
        prompts: Optional[List[str]] = None,
        configs: Optional[Dict[str, TargetedSteeringConfig]] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run full benchmark.

        Args:
            prompts: Test prompts (default: TEST_PROMPTS)
            configs: Steering configs (default: CONFIGS)
            verbose: Print progress

        Returns:
            BenchmarkResult with metrics
        """
        prompts = prompts or TEST_PROMPTS
        configs = configs or CONFIGS

        result = BenchmarkResult(mlux_available=MLUX_AVAILABLE)
        start_time = time.time()

        # Initialize metrics for baseline + all configs
        result.metrics["baseline"] = BenchmarkMetrics(config_name="baseline")
        for name in configs:
            result.metrics[name] = BenchmarkMetrics(config_name=name)

        if verbose:
            print(f"\nRunning benchmark:")
            print(f"  Prompts: {len(prompts)}")
            print(f"  Configs: {len(configs) + 1} (including baseline)")
            print(f"  Total generations: {len(prompts) * (len(configs) + 1)}")

        for prompt_idx, prompt in enumerate(prompts):
            if verbose:
                print(f"\n--- Prompt {prompt_idx + 1}/{len(prompts)} ---")

            # Run all configs for this prompt
            prompt_results = self.steering.compare_steering_configs(prompt, configs)

            # Update metrics
            for config_name, steering_result in prompt_results.items():
                metrics = result.metrics[config_name]
                metrics.num_samples += 1

                if steering_result.valid_json:
                    metrics.valid_json_count += 1
                if steering_result.has_hierarchy:
                    metrics.has_hierarchy_count += 1
                metrics.total_states += steering_result.num_states
                metrics.total_nested += steering_result.num_nested

                if verbose:
                    status = "H" if steering_result.has_hierarchy else "F" if steering_result.valid_json else "X"
                    print(f"  [{status}] {config_name}: {steering_result.num_states} states, "
                          f"{steering_result.num_nested} nested")

        result.total_time = time.time() - start_time

        return result

    def print_summary(self, result: BenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nMLUX available: {result.mlux_available}")
        print(f"Total time: {result.total_time:.1f}s")

        print("\n--- Metrics by Configuration ---")
        print(f"{'Config':<25} {'Valid%':>8} {'Hier%':>8} {'AvgStates':>10} {'AvgNested':>10}")
        print("-" * 65)

        for name, metrics in sorted(result.metrics.items()):
            print(f"{name:<25} {metrics.validity_rate:>7.1%} {metrics.hierarchy_rate:>7.1%} "
                  f"{metrics.avg_states:>10.1f} {metrics.avg_nested:>10.1f}")

        # Find best config
        best_hier = result.get_best_config("hierarchy_rate")
        best_valid = result.get_best_config("validity_rate")

        print(f"\n--- Best Configurations ---")
        print(f"  Highest hierarchy rate: {best_hier} ({result.metrics[best_hier].hierarchy_rate:.1%})")
        print(f"  Highest validity rate: {best_valid} ({result.metrics[best_valid].validity_rate:.1%})")

        # Calculate improvement over baseline
        baseline = result.metrics.get("baseline")
        if baseline and best_hier != "baseline":
            best_metrics = result.metrics[best_hier]
            hier_improvement = best_metrics.hierarchy_rate - baseline.hierarchy_rate
            print(f"\n  Hierarchy improvement over baseline: +{hier_improvement:.1%}")

    def quick_test(self) -> BenchmarkResult:
        """Quick test with fewer prompts."""
        return self.run_benchmark(
            prompts=TEST_PROMPTS[:2],
            configs={"hierarchy_boost": CONFIGS["hierarchy_boost"]},
            verbose=True,
        )


def demo():
    """Quick demo."""
    print("=" * 60)
    print("STEERING BENCHMARK DEMO")
    print("=" * 60)

    benchmark = SteeringBenchmark()
    result = benchmark.quick_test()
    benchmark.print_summary(result)

    return result


def run_full_benchmark():
    """Run full benchmark."""
    benchmark = SteeringBenchmark()
    result = benchmark.run_benchmark(verbose=True)
    benchmark.print_summary(result)

    # Check if L23H1 amplification helps
    print("\n" + "=" * 60)
    print("L23H1 HIERARCHY HEAD ANALYSIS")
    print("=" * 60)

    baseline_hier = result.metrics.get("baseline", BenchmarkMetrics("baseline")).hierarchy_rate
    hier_boost = result.metrics.get("hierarchy_boost", BenchmarkMetrics("hierarchy_boost")).hierarchy_rate

    if hier_boost > baseline_hier:
        improvement = hier_boost - baseline_hier
        print(f"[SUCCESS] L23H1 amplification improves hierarchy: +{improvement:.1%}")
        print(f"  Baseline hierarchy rate: {baseline_hier:.1%}")
        print(f"  With L23H1 boost: {hier_boost:.1%}")
    else:
        print(f"[NEUTRAL] L23H1 amplification did not improve hierarchy")
        print(f"  Baseline: {baseline_hier:.1%}, Boosted: {hier_boost:.1%}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        demo()
