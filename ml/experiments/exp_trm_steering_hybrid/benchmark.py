"""
Benchmark: TRM + Steering Hybrid Refinement

Compares methods for achieving 60% → 99% validity:
1. Baseline: No refinement
2. TRM only: Test-time refinement without steering
3. Steering only: Steered generation without TRM
4. Hybrid: TRM + Steering combined

TARGET: 60% initial → 99% final validity
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .hybrid_refiner import (
    HybridRefiner,
    HybridConfig,
    SteeringConfig,
    TRMConfig,
    ValidityChecker,
    RefinementResult,
)
from .steering_guided_iteration import (
    SteeringGuidedIterator,
    GuidedIterationConfig,
    AspectAnalyzer,
)


@dataclass
class BenchmarkCase:
    """A single benchmark case."""
    name: str
    prompt: str
    initial_chart: Dict[str, Any]
    expected_validity: float = 0.6  # Starting validity


@dataclass
class MethodResult:
    """Result for a single method on a single case."""
    case_name: str
    method_name: str
    initial_validity: float
    final_validity: float
    iterations: int
    duration: float
    success: bool

    @property
    def improvement(self) -> float:
        return self.final_validity - self.initial_validity

    @property
    def reached_target(self) -> bool:
        return self.final_validity >= 0.99


@dataclass
class MethodSummary:
    """Summary for a method across all cases."""
    method_name: str
    n_cases: int
    mean_initial: float
    mean_final: float
    mean_improvement: float
    mean_iterations: float
    success_rate: float
    target_rate: float  # Reached 99%
    total_duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "cases": self.n_cases,
            "mean_initial": self.mean_initial,
            "mean_final": self.mean_final,
            "improvement": self.mean_improvement,
            "iterations": self.mean_iterations,
            "success_rate": self.success_rate,
            "target_rate": self.target_rate,
            "duration": self.total_duration,
        }


# =============================================================================
# Test cases with ~60% initial validity
# =============================================================================

BENCHMARK_CASES = [
    BenchmarkCase(
        name="missing_initial",
        prompt="Generate a traffic light:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "red", "type": 1},  # Missing is_initial
                    {"label": "yellow", "type": 1},
                    {"label": "green", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["red"], "to": ["green"], "event": "TIMER"},
                {"from": ["green"], "to": ["yellow"], "event": "TIMER"},
                {"from": ["yellow"], "to": ["red"], "event": "TIMER"},
            ]
        },
    ),
    BenchmarkCase(
        name="bad_refs",
        prompt="Generate a login flow:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "logged_out", "type": 1, "is_initial": True},
                    {"label": "logging_in", "type": 1},
                    {"label": "logged_in", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["logged_out"], "to": ["TYPO_logging_in"], "event": "LOGIN"},
                {"from": ["logging_in"], "to": ["logged_in"], "event": "SUCCESS"},
                {"from": ["nonexistent"], "to": ["logged_out"], "event": "LOGOUT"},
            ]
        },
    ),
    BenchmarkCase(
        name="empty_compound",
        prompt="Generate an order process:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "pending", "type": 1, "is_initial": True},
                    {
                        "label": "processing",
                        "type": 2,  # Compound but no children
                        "children": []
                    },
                    {"label": "complete", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["pending"], "to": ["processing"], "event": "START"},
                {"from": ["processing"], "to": ["complete"], "event": "DONE"},
            ]
        },
    ),
    BenchmarkCase(
        name="missing_events",
        prompt="Generate a player state:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "idle", "type": 1, "is_initial": True},
                    {"label": "walking", "type": 1},
                    {"label": "running", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["idle"], "to": ["walking"]},  # Missing event
                {"from": ["walking"], "to": ["running"]},  # Missing event
                {"from": ["running"], "to": ["idle"], "event": "STOP"},
            ]
        },
    ),
    BenchmarkCase(
        name="duplicates",
        prompt="Generate a media player:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "stopped", "type": 1, "is_initial": True},
                    {"label": "playing", "type": 1},
                    {"label": "stopped", "type": 1},  # Duplicate
                ]
            },
            "transitions": [
                {"from": ["stopped"], "to": ["playing"], "event": "PLAY"},
                {"from": ["playing"], "to": ["stopped"], "event": "STOP"},
            ]
        },
    ),
    BenchmarkCase(
        name="mixed_issues",
        prompt="Generate a door lock:",
        initial_chart={
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "locked", "type": 1},  # No initial
                    {"label": "unlocked", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["locked"], "to": ["TYPO_unlocked"]},  # Bad ref, no event
                {"from": ["unlocked"], "to": ["locked"], "event": "LOCK"},
            ]
        },
    ),
]


# =============================================================================
# Refinement methods
# =============================================================================

class BaselineMethod:
    """No refinement - just validate."""

    def __init__(self):
        self.checker = ValidityChecker()

    def refine(self, chart: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], int, float]:
        """Return chart unchanged."""
        start = time.time()
        return chart, 0, time.time() - start


class TRMOnlyMethod:
    """TRM refinement without steering."""

    def __init__(self, max_iterations: int = 5):
        from .hybrid_refiner import TRMRefiner
        self.refiner = TRMRefiner(
            model=None,
            config=TRMConfig(max_iterations=max_iterations),
            steering_config=None,  # No steering
        )
        self.checker = ValidityChecker()

    def refine(self, chart: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], int, float]:
        start = time.time()
        refined, steps = self.refiner.refine(chart, prompt)
        return refined, len(steps), time.time() - start


class SteeringOnlyMethod:
    """Steering without TRM refinement."""

    def __init__(self, strength: float = 1.5):
        from .hybrid_refiner import SteeringGenerator
        self.generator = SteeringGenerator(
            model=None,
            config=SteeringConfig(strength=strength),
        )
        self.checker = ValidityChecker()

    def refine(self, chart: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], int, float]:
        start = time.time()
        # Re-generate with steering (ignores chart)
        output = self.generator.generate(prompt)
        try:
            refined = json.loads(output)
        except json.JSONDecodeError:
            refined = chart
        return refined, 1, time.time() - start


class HybridMethod:
    """TRM + Steering combined."""

    def __init__(
        self,
        steering_strength: float = 1.2,
        max_iterations: int = 5,
    ):
        config = HybridConfig(
            steering=SteeringConfig(strength=steering_strength),
            trm=TRMConfig(max_iterations=max_iterations),
            use_steering_in_trm=True,
        )
        self.refiner = HybridRefiner(model=None, config=config, verbose=False)

    def refine(self, chart: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], int, float]:
        # Note: HybridRefiner generates fresh, so we use the iterator for fixing
        start = time.time()

        iterator = SteeringGuidedIterator(model=None, verbose=False)
        refined, results = iterator.iterate(chart, prompt)

        return refined, len(results), time.time() - start


class HybridPlusMethod:
    """Enhanced hybrid with multiple passes."""

    def __init__(self):
        self.iterator = SteeringGuidedIterator(
            model=None,
            config=GuidedIterationConfig(
                max_h_iterations=4,
                max_l_iterations=4,
                strength_schedule=[1.0, 1.3, 1.6, 2.0],
                early_stop_threshold=0.99,
            ),
            verbose=False,
        )
        self.checker = ValidityChecker()

    def refine(self, chart: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], int, float]:
        start = time.time()

        # Multiple passes
        current = chart
        total_iterations = 0

        for pass_num in range(3):  # Up to 3 passes
            score = self.checker.check(current)["total"]
            if score >= 0.99:
                break

            refined, results = self.iterator.iterate(current, prompt)
            total_iterations += len(results)

            new_score = self.checker.check(refined)["total"]
            if new_score > score:
                current = refined

        return current, total_iterations, time.time() - start


# =============================================================================
# Benchmark runner
# =============================================================================

class HybridBenchmark:
    """Benchmark TRM + Steering hybrid refinement."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checker = ValidityChecker()

        self.methods = {
            "baseline": BaselineMethod(),
            "trm_only": TRMOnlyMethod(),
            "steering_only": SteeringOnlyMethod(),
            "hybrid": HybridMethod(),
            "hybrid_plus": HybridPlusMethod(),
        }

    def run_case(
        self,
        case: BenchmarkCase,
        method_name: str,
    ) -> MethodResult:
        """Run a single case with a method."""
        method = self.methods[method_name]

        initial_scores = self.checker.check(case.initial_chart)
        initial_validity = initial_scores["total"]

        refined, iterations, duration = method.refine(
            case.initial_chart, case.prompt
        )

        final_scores = self.checker.check(refined)
        final_validity = final_scores["total"]

        success = final_validity > initial_validity + 0.05 or final_validity >= 0.9

        return MethodResult(
            case_name=case.name,
            method_name=method_name,
            initial_validity=initial_validity,
            final_validity=final_validity,
            iterations=iterations,
            duration=duration,
            success=success,
        )

    def run_benchmark(
        self,
        cases: Optional[List[BenchmarkCase]] = None,
    ) -> Dict[str, MethodSummary]:
        """Run full benchmark."""
        if cases is None:
            cases = BENCHMARK_CASES

        results_by_method: Dict[str, List[MethodResult]] = defaultdict(list)

        total = len(cases) * len(self.methods)
        current = 0

        for case in cases:
            for method_name in self.methods:
                current += 1
                if self.verbose and current % 5 == 0:
                    print(f"  Progress: {current}/{total}")

                result = self.run_case(case, method_name)
                results_by_method[method_name].append(result)

        # Compute summaries
        summaries = {}
        for method_name, results in results_by_method.items():
            n = len(results)
            summaries[method_name] = MethodSummary(
                method_name=method_name,
                n_cases=n,
                mean_initial=sum(r.initial_validity for r in results) / n,
                mean_final=sum(r.final_validity for r in results) / n,
                mean_improvement=sum(r.improvement for r in results) / n,
                mean_iterations=sum(r.iterations for r in results) / n,
                success_rate=sum(1 for r in results if r.success) / n,
                target_rate=sum(1 for r in results if r.reached_target) / n,
                total_duration=sum(r.duration for r in results),
            )

        return summaries


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate hybrid refinement benchmark."""
    print("=" * 70)
    print("TRM + Steering Hybrid Refinement Benchmark")
    print("TARGET: 60% → 99% validity")
    print("=" * 70)

    benchmark = HybridBenchmark(verbose=True)

    print("\nRunning benchmark...")
    summaries = benchmark.run_benchmark()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<16} {'Initial':<10} {'Final':<10} {'Δ':<10} "
          f"{'Iters':<8} {'Success':<10} {'99% Rate':<10}")
    print("-" * 80)

    for name in ["baseline", "trm_only", "steering_only", "hybrid", "hybrid_plus"]:
        s = summaries[name]
        print(f"{name:<16} {s.mean_initial:>8.1%} {s.mean_final:>8.1%} "
              f"{s.mean_improvement:>+8.1%} {s.mean_iterations:>6.1f} "
              f"{s.success_rate:>8.0%} {s.target_rate:>8.0%}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")

    baseline = summaries["baseline"]
    hybrid_plus = summaries["hybrid_plus"]

    improvement = hybrid_plus.mean_final - baseline.mean_final
    print(f"\n  Hybrid+ achieves {hybrid_plus.mean_final:.1%} validity "
          f"(+{improvement:.1%} vs baseline)")

    if hybrid_plus.target_rate > 0:
        print(f"  Reached 99% target in {hybrid_plus.target_rate:.0%} of cases")

    # Compare methods
    best = max(summaries.values(), key=lambda s: s.mean_final)
    most_efficient = max(
        summaries.values(),
        key=lambda s: s.mean_improvement / (s.mean_iterations + 1)
    )

    print(f"\n  Best validity: {best.method_name} ({best.mean_final:.1%})")
    print(f"  Most efficient: {most_efficient.method_name} "
          f"({most_efficient.mean_improvement/(most_efficient.mean_iterations+1):.3f} per iter)")

    return summaries


if __name__ == "__main__":
    demo()
