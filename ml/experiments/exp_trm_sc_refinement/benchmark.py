"""
Benchmark: Evaluate TRM-style statechart refinement.

Target metrics:
- Turn 60% valid initial SCs into 95%+ valid after refinement
- Track validity improvement across iterations
- Compare with baseline approaches
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .sc_refiner import SCRefiner, SCRefinerConfig
from .refinement_loop import RefinementLoop, RefinementResult, run_refinement
from .validity_tracker import ValidityTracker, ValidityMetrics, track_validity


@dataclass
class BenchmarkConfig:
    """Configuration for refinement benchmark."""
    # Dataset
    num_statecharts: int = 100
    max_states: int = 16
    max_transitions: int = 32

    # Model
    hidden_dim: int = 128
    H_cycles: int = 3
    L_cycles: int = 6

    # Targets
    initial_validity_target: float = 0.60  # Start with ~60% valid
    final_validity_target: float = 0.95    # Target 95%+ valid

    # Output
    save_dir: str = "refinement_results"


@dataclass
class BenchmarkResult:
    """Results from refinement benchmark."""
    config: BenchmarkConfig
    runtime_seconds: float

    # Dataset stats
    num_statecharts: int
    initial_valid_count: int
    initial_validity_rate: float

    # Refinement results
    final_valid_count: int
    final_validity_rate: float
    validity_improvement: float

    # Per-SC metrics
    mean_initial_validity: float
    mean_final_validity: float
    mean_improvement: float

    # Convergence
    num_converged: int
    mean_convergence_iter: float

    # Target achievement
    achieved_target: bool

    # Trajectory summary
    validity_trajectory: List[float]

    # Individual results
    individual_results: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "runtime_seconds": self.runtime_seconds,
            "num_statecharts": self.num_statecharts,
            "initial_validity_rate": self.initial_validity_rate,
            "final_validity_rate": self.final_validity_rate,
            "validity_improvement": self.validity_improvement,
            "mean_initial_validity": self.mean_initial_validity,
            "mean_final_validity": self.mean_final_validity,
            "mean_improvement": self.mean_improvement,
            "num_converged": self.num_converged,
            "mean_convergence_iter": self.mean_convergence_iter,
            "achieved_target": self.achieved_target,
            "validity_trajectory": self.validity_trajectory,
        }


class RefinementBenchmark:
    """
    Benchmark for statechart refinement.

    Generates partially-valid statecharts, refines them,
    and measures improvement.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

        # Initialize model
        model_config = SCRefinerConfig(
            hidden_dim=self.config.hidden_dim,
            max_states=self.config.max_states,
            max_transitions=self.config.max_transitions,
            H_cycles=self.config.H_cycles,
            L_cycles=self.config.L_cycles,
        )
        self.model = SCRefiner(model_config)
        self.validity_tracker = ValidityTracker(
            max_states=self.config.max_states,
            max_transitions=self.config.max_transitions,
        )

    def generate_partial_statechart(
        self,
        target_validity: float = 0.6,
    ) -> Dict[str, mx.array]:
        """
        Generate a partially-valid statechart.

        Args:
            target_validity: Target initial validity (0-1)

        Returns:
            Encoded statechart data
        """
        S = self.config.max_states
        T = self.config.max_transitions
        max_label_len = 16

        # Determine how many issues to inject
        # Lower target_validity = more issues
        num_issues = int((1 - target_validity) * 5)

        # Generate base valid structure
        # State types: mostly BASIC with some composites
        state_types = mx.zeros((1, S), dtype=mx.int32)
        state_types = state_types.at[0, 0].add(1)  # Root is NORMAL
        state_types = state_types.at[0, 4].add(1)  # Another composite

        # Depths
        state_depths = mx.zeros((1, S), dtype=mx.int32)
        for i in range(1, 4):
            state_depths = state_depths.at[0, i].add(1)
        for i in range(5, 8):
            state_depths = state_depths.at[0, i].add(1)

        # Label chars (random)
        label_chars = mx.random.randint(97, 123, (1, S, max_label_len))  # a-z

        # Transitions (some valid, some potentially invalid)
        source_indices = mx.random.randint(0, S, (1, T))
        target_indices = mx.random.randint(0, S, (1, T))
        event_ids = mx.random.randint(0, 10, (1, T))

        # Inject issues based on target_validity
        if num_issues > 0:
            # Issue 1: Invalid transition source
            source_indices = source_indices.at[0, 0].add(S + 10)

        if num_issues > 1:
            # Issue 2: Invalid transition target
            target_indices = target_indices.at[0, 1].add(S + 10)

        if num_issues > 2:
            # Issue 3: Non-determinism (duplicate source+event)
            source_indices = source_indices.at[0, 2].add(0)
            source_indices = source_indices.at[0, 3].add(0)
            event_ids = event_ids.at[0, 2].add(0)
            event_ids = event_ids.at[0, 3].add(0)

        # Masks
        state_mask = mx.ones((1, S))
        trans_mask = mx.ones((1, T))

        # Only use first 8 states and 12 transitions
        state_mask = state_mask.at[0, 8:].add(-1)
        trans_mask = trans_mask.at[0, 12:].add(-1)

        return {
            "state_types": state_types,
            "state_depths": state_depths,
            "label_chars": label_chars,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "event_ids": event_ids,
            "state_mask": state_mask,
            "trans_mask": trans_mask,
        }

    def evaluate_statechart(
        self,
        sc_data: Dict[str, mx.array],
    ) -> ValidityMetrics:
        """Evaluate a statechart's validity."""
        # Use model to get predictions
        predictions = self.model.refine(**sc_data)

        return track_validity(
            predictions,
            sc_data["state_mask"],
            sc_data["trans_mask"],
        )

    def run(self) -> BenchmarkResult:
        """
        Run the full benchmark.

        Returns:
            BenchmarkResult with all metrics
        """
        print("=" * 60)
        print("STATECHART REFINEMENT BENCHMARK")
        print("=" * 60)
        print(f"Statecharts: {self.config.num_statecharts}")
        print(f"H_cycles: {self.config.H_cycles}, L_cycles: {self.config.L_cycles}")
        print(f"Target: {self.config.initial_validity_target:.0%} → "
              f"{self.config.final_validity_target:.0%}")

        start_time = time.time()

        individual_results = []
        validity_trajectories = []

        for i in range(self.config.num_statecharts):
            # Generate partial statechart
            target_val = self.config.initial_validity_target + \
                         np.random.uniform(-0.2, 0.2)
            target_val = np.clip(target_val, 0.3, 0.8)

            sc_data = self.generate_partial_statechart(target_val)

            # Run refinement
            result = run_refinement(
                self.model,
                sc_data,
                H_cycles=self.config.H_cycles,
                L_cycles=self.config.L_cycles,
                validity_threshold=self.config.final_validity_target,
            )

            individual_results.append({
                "id": i,
                "initial_validity": result.input_validity,
                "final_validity": result.output_validity,
                "improvement": result.validity_improvement,
                "converged": result.converged,
                "convergence_iter": result.convergence_iteration,
            })

            validity_trajectories.append(result.get_validity_trajectory())

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{self.config.num_statecharts}")

        runtime = time.time() - start_time

        # Aggregate metrics
        initial_vals = [r["initial_validity"] for r in individual_results]
        final_vals = [r["final_validity"] for r in individual_results]
        improvements = [r["improvement"] for r in individual_results]

        initial_valid_count = sum(1 for v in initial_vals if v >= 0.95)
        final_valid_count = sum(1 for v in final_vals if v >= 0.95)

        num_converged = sum(1 for r in individual_results if r["converged"])
        conv_iters = [
            r["convergence_iter"]
            for r in individual_results
            if r["convergence_iter"] is not None
        ]

        # Aggregate trajectory
        if validity_trajectories:
            max_len = max(len(t) for t in validity_trajectories)
            padded = []
            for t in validity_trajectories:
                if len(t) < max_len:
                    t = np.concatenate([t, [t[-1]] * (max_len - len(t))])
                padded.append(t)
            mean_trajectory = np.mean(padded, axis=0).tolist()
        else:
            mean_trajectory = []

        result = BenchmarkResult(
            config=self.config,
            runtime_seconds=runtime,
            num_statecharts=self.config.num_statecharts,
            initial_valid_count=initial_valid_count,
            initial_validity_rate=initial_valid_count / self.config.num_statecharts,
            final_valid_count=final_valid_count,
            final_validity_rate=final_valid_count / self.config.num_statecharts,
            validity_improvement=(final_valid_count - initial_valid_count) /
                                 self.config.num_statecharts,
            mean_initial_validity=float(np.mean(initial_vals)),
            mean_final_validity=float(np.mean(final_vals)),
            mean_improvement=float(np.mean(improvements)),
            num_converged=num_converged,
            mean_convergence_iter=float(np.mean(conv_iters)) if conv_iters else -1,
            achieved_target=final_valid_count / self.config.num_statecharts >=
                           self.config.final_validity_target,
            validity_trajectory=mean_trajectory,
            individual_results=individual_results,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Runtime: {runtime:.1f}s")
        print(f"\nValidity Improvement:")
        print(f"  Initial: {result.mean_initial_validity:.2%}")
        print(f"  Final:   {result.mean_final_validity:.2%}")
        print(f"  Improvement: {result.mean_improvement:.2%}")
        print(f"\nConvergence:")
        print(f"  Converged: {result.num_converged}/{self.config.num_statecharts}")
        print(f"  Avg iter: {result.mean_convergence_iter:.1f}")
        print(f"\nTarget Achievement: {'YES' if result.achieved_target else 'NO'}")

        # Save results
        if self.config.save_dir:
            self._save_results(result)

        return result

    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Save JSON summary
        summary_path = os.path.join(self.config.save_dir, "benchmark_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nSaved results to {summary_path}")

        # Save detailed report
        report = self._generate_report(result)
        report_path = os.path.join(self.config.save_dir, "refinement_report.md")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Saved report to {report_path}")

    def _generate_report(self, result: BenchmarkResult) -> str:
        """Generate markdown report."""
        lines = [
            "# Statechart Refinement Benchmark Report",
            "",
            "## Overview",
            f"- Statecharts tested: {result.num_statecharts}",
            f"- Runtime: {result.runtime_seconds:.1f}s",
            f"- Target: {self.config.initial_validity_target:.0%} → "
            f"{self.config.final_validity_target:.0%}",
            "",
            "## Validity Improvement",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Initial Validity | {result.mean_initial_validity:.2%} |",
            f"| Mean Final Validity | {result.mean_final_validity:.2%} |",
            f"| Mean Improvement | {result.mean_improvement:.2%} |",
            "",
            "## Convergence Statistics",
            "",
            f"- Converged: {result.num_converged}/{result.num_statecharts} "
            f"({result.num_converged/result.num_statecharts:.1%})",
            f"- Mean convergence iteration: {result.mean_convergence_iter:.1f}",
            "",
            "## Validity Trajectory",
            "",
        ]

        if result.validity_trajectory:
            lines.append("| Iteration | Mean Validity |")
            lines.append("|-----------|---------------|")
            for i, v in enumerate(result.validity_trajectory[:10]):
                lines.append(f"| {i} | {v:.3f} |")
            if len(result.validity_trajectory) > 10:
                lines.append(f"| ... | ... |")
                lines.append(f"| {len(result.validity_trajectory)-1} | "
                           f"{result.validity_trajectory[-1]:.3f} |")

        lines.extend([
            "",
            "## Target Achievement",
            "",
            f"**Target Achieved: {'YES' if result.achieved_target else 'NO'}**",
            "",
            f"Goal was to achieve {self.config.final_validity_target:.0%} "
            f"validity rate. Achieved {result.final_validity_rate:.2%}.",
        ])

        return "\n".join(lines)


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
    """Convenience function to run benchmark."""
    benchmark = RefinementBenchmark(config)
    return benchmark.run()


def test_benchmark():
    """Test the benchmark."""
    print("=" * 60)
    print("Testing Refinement Benchmark")
    print("=" * 60)

    config = BenchmarkConfig(
        num_statecharts=10,
        max_states=16,
        max_transitions=32,
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
        save_dir="/tmp/test_refinement",
    )

    result = run_benchmark(config)

    print("\n" + "=" * 60)
    print(f"Test complete: {result.num_statecharts} SCs processed")
    print(f"Validity: {result.mean_initial_validity:.2%} → "
          f"{result.mean_final_validity:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
