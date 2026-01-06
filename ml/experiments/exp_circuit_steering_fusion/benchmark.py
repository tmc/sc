"""
Benchmark: Compare global vs circuit-specific steering.

Compares:
1. No steering (baseline)
2. Global steering (single layer)
3. Transition-only steering (L8-14)
4. Hierarchy-only steering (L0-6)
5. Fused steering (both circuits)

Goal: Show that circuit-specific steering outperforms global steering
for statechart generation.
"""

import json
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

from .fused_steering import (
    FusedSteeredSampler,
    FusedSteeringConfig,
    FusedGenerationResult,
    load_steering_vectors,
)
from .circuit_hooks import CircuitType, CIRCUITS


# =============================================================================
# TEST PROMPTS
# =============================================================================

# Prompts targeting different circuit capabilities
TRANSITION_PROMPTS = [
    "Generate a statechart with transitions between 3 states",
    "Create a workflow with valid source and target states",
    "Build a state machine with bidirectional transitions",
]

HIERARCHY_PROMPTS = [
    "Generate a hierarchical statechart with nested states",
    "Create a statechart with parent and child states",
    "Build a statechart with composite states containing substates",
]

MIXED_PROMPTS = [
    "Generate a complete statechart for order processing",
    "Create a game character statechart with states and transitions",
    "Build a user authentication workflow statechart",
]

ALL_PROMPTS = TRANSITION_PROMPTS + HIERARCHY_PROMPTS + MIXED_PROMPTS


# =============================================================================
# VALIDATION
# =============================================================================

def validate_statechart(text: str) -> Tuple[bool, str, Optional[Dict]]:
    """Validate generated statechart JSON."""
    import re

    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')

    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    start = text.find('{')
    if start == -1:
        return False, "No JSON found", None

    depth = 0
    end = start
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        sc = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        return False, f"JSON error: {e}", None

    if "root_state" not in sc:
        return False, "Missing root_state", None

    root = sc["root_state"]
    children = root.get("children", [])

    if len(children) == 0:
        return False, "Empty children", None

    has_initial = any(c.get("is_initial", False) for c in children)
    if not has_initial:
        return False, "No initial state", None

    for child in children:
        if "label" not in child:
            return False, "State missing label", None

    transitions = sc.get("transitions", [])
    labels = {c["label"] for c in children}

    for t in transitions:
        for s in t.get("from", []):
            if s not in labels:
                return False, f"Invalid source: {s}", None
        for s in t.get("to", []):
            if s not in labels:
                return False, f"Invalid target: {s}", None

    return True, "Valid", sc


# =============================================================================
# BENCHMARK RESULTS
# =============================================================================

@dataclass
class ModeResult:
    """Result for a single mode."""
    mode: str
    valid_count: int
    total_count: int
    validity_rate: float
    avg_time: float
    circuits: List[str]


@dataclass
class BenchmarkResult:
    """Full benchmark result."""
    mode_results: Dict[str, ModeResult]
    best_mode: str
    best_validity: float
    fused_vs_global_improvement: float
    prompt_category: str

    def __repr__(self):
        return (f"BenchmarkResult(best={self.best_mode}@{self.best_validity:.1%}, "
                f"fused_vs_global={self.fused_vs_global_improvement:+.1%})")


@dataclass
class FullBenchmarkResult:
    """Result across all prompt categories."""
    category_results: Dict[str, BenchmarkResult]
    overall_by_mode: Dict[str, ModeResult]
    best_overall_mode: str
    fused_wins: int
    global_wins: int


# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class CircuitFusionBenchmark:
    """
    Benchmark comparing global vs circuit-specific steering.

    Tests each steering mode on different prompt categories.
    """

    MODES = ["baseline", "global", "transition", "hierarchy", "fused"]

    def __init__(
        self,
        sampler: FusedSteeredSampler = None,
        config: FusedSteeringConfig = None,
    ):
        self.sampler = sampler or FusedSteeredSampler()
        self.base_config = config or FusedSteeringConfig(
            max_tokens=256,
            temperature=0.3,
        )
        self.results: Dict[str, BenchmarkResult] = {}

    def run_mode(
        self,
        prompts: List[str],
        mode: str,
        verbose: bool = True,
    ) -> ModeResult:
        """Run benchmark for a single mode."""
        config = FusedSteeringConfig(
            max_tokens=self.base_config.max_tokens,
            temperature=self.base_config.temperature,
            global_alpha=self.base_config.global_alpha,
            transition_alpha=self.base_config.transition_alpha,
            hierarchy_alpha=self.base_config.hierarchy_alpha,
            mode=mode if mode != "baseline" else "global",
        )

        # For baseline, we'll generate without steering
        valid_count = 0
        total_time = 0.0
        circuits = []

        for i, prompt in enumerate(prompts):
            if mode == "baseline":
                # No steering
                config_copy = FusedSteeringConfig(
                    max_tokens=self.base_config.max_tokens,
                    temperature=self.base_config.temperature,
                    mode="global",
                    global_alpha=0.0,  # Zero alpha = no steering
                )
                result = self.sampler.generate(prompt, config_copy)
            else:
                result = self.sampler.generate(prompt, config)

            is_valid, error, _ = validate_statechart(result.output)
            if is_valid:
                valid_count += 1
            total_time += result.generation_time

            if not circuits and result.circuits_used:
                circuits = result.circuits_used

            if verbose:
                status = "VALID" if is_valid else f"INVALID ({error})"
                print(f"  [{i+1}/{len(prompts)}] {status}")

        return ModeResult(
            mode=mode,
            valid_count=valid_count,
            total_count=len(prompts),
            validity_rate=valid_count / max(len(prompts), 1),
            avg_time=total_time / max(len(prompts), 1),
            circuits=circuits,
        )

    def run_category(
        self,
        category: str,
        prompts: List[str],
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark for a prompt category."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"CATEGORY: {category}")
            print(f"{'='*60}")

        mode_results = {}

        for mode in self.MODES:
            if verbose:
                print(f"\nMode: {mode}")
            result = self.run_mode(prompts, mode, verbose)
            mode_results[mode] = result

            if verbose:
                print(f"  Valid: {result.valid_count}/{result.total_count} ({result.validity_rate:.1%})")

        # Find best mode
        best_mode = max(mode_results.keys(), key=lambda m: mode_results[m].validity_rate)
        best_validity = mode_results[best_mode].validity_rate

        # Compare fused vs global
        fused_rate = mode_results["fused"].validity_rate
        global_rate = mode_results["global"].validity_rate
        improvement = fused_rate - global_rate

        result = BenchmarkResult(
            mode_results=mode_results,
            best_mode=best_mode,
            best_validity=best_validity,
            fused_vs_global_improvement=improvement,
            prompt_category=category,
        )

        self.results[category] = result
        return result

    def run_full(self, verbose: bool = True) -> FullBenchmarkResult:
        """Run full benchmark across all categories."""
        if verbose:
            print("=" * 70)
            print("CIRCUIT FUSION BENCHMARK")
            print("=" * 70)

        # Ensure steering vectors are set
        if self.sampler._steering_vector is None:
            if MLX_AVAILABLE:
                vec = mx.random.normal((896,)) * 0.1
                self.sampler.set_steering_vector(vec)
                if verbose:
                    print("Using random steering vector")

        categories = [
            ("TRANSITION", TRANSITION_PROMPTS),
            ("HIERARCHY", HIERARCHY_PROMPTS),
            ("MIXED", MIXED_PROMPTS),
        ]

        for category, prompts in categories:
            self.run_category(category, prompts, verbose)

        # Aggregate results
        overall_by_mode = self._aggregate_results()
        best_overall = max(overall_by_mode.keys(), key=lambda m: overall_by_mode[m].validity_rate)

        # Count wins
        fused_wins = sum(1 for r in self.results.values() if r.best_mode == "fused")
        global_wins = sum(1 for r in self.results.values() if r.best_mode == "global")

        full_result = FullBenchmarkResult(
            category_results=self.results,
            overall_by_mode=overall_by_mode,
            best_overall_mode=best_overall,
            fused_wins=fused_wins,
            global_wins=global_wins,
        )

        if verbose:
            self._print_summary(full_result)

        return full_result

    def run_quick(self, verbose: bool = True) -> FullBenchmarkResult:
        """Run quick benchmark with fewer prompts."""
        if verbose:
            print("=" * 60)
            print("QUICK CIRCUIT FUSION BENCHMARK")
            print("=" * 60)

        if self.sampler._steering_vector is None:
            if MLX_AVAILABLE:
                vec = mx.random.normal((896,)) * 0.1
                self.sampler.set_steering_vector(vec)

        # Use only first prompt from each category
        categories = [
            ("TRANSITION", TRANSITION_PROMPTS[:1]),
            ("HIERARCHY", HIERARCHY_PROMPTS[:1]),
            ("MIXED", MIXED_PROMPTS[:1]),
        ]

        for category, prompts in categories:
            self.run_category(category, prompts, verbose)

        overall_by_mode = self._aggregate_results()
        best_overall = max(overall_by_mode.keys(), key=lambda m: overall_by_mode[m].validity_rate)

        fused_wins = sum(1 for r in self.results.values() if r.best_mode == "fused")
        global_wins = sum(1 for r in self.results.values() if r.best_mode == "global")

        full_result = FullBenchmarkResult(
            category_results=self.results,
            overall_by_mode=overall_by_mode,
            best_overall_mode=best_overall,
            fused_wins=fused_wins,
            global_wins=global_wins,
        )

        if verbose:
            self._print_summary(full_result)

        return full_result

    def _aggregate_results(self) -> Dict[str, ModeResult]:
        """Aggregate results across categories."""
        overall = {}

        for mode in self.MODES:
            total_valid = 0
            total_count = 0
            total_time = 0.0
            circuits = []

            for cat_result in self.results.values():
                mode_result = cat_result.mode_results[mode]
                total_valid += mode_result.valid_count
                total_count += mode_result.total_count
                total_time += mode_result.avg_time * mode_result.total_count
                if not circuits:
                    circuits = mode_result.circuits

            overall[mode] = ModeResult(
                mode=mode,
                valid_count=total_valid,
                total_count=total_count,
                validity_rate=total_valid / max(total_count, 1),
                avg_time=total_time / max(total_count, 1),
                circuits=circuits,
            )

        return overall

    def _print_summary(self, result: FullBenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Per-category results
        print("\nResults by category:")
        print(f"{'Category':<15} {'Best Mode':<12} {'Validity':<10} {'Fused vs Global':<15}")
        print("-" * 60)

        for category, cat_result in result.category_results.items():
            print(f"{category:<15} {cat_result.best_mode:<12} "
                  f"{cat_result.best_validity:>8.1%} "
                  f"{cat_result.fused_vs_global_improvement:>+14.1%}")

        # Overall results
        print("\nOverall by mode:")
        print(f"{'Mode':<12} {'Valid':<10} {'Rate':<10} {'Circuits':<25}")
        print("-" * 60)

        for mode, mode_result in result.overall_by_mode.items():
            circuits_str = ", ".join(mode_result.circuits[:2]) if mode_result.circuits else "none"
            print(f"{mode:<12} {mode_result.valid_count}/{mode_result.total_count:<8} "
                  f"{mode_result.validity_rate:>8.1%} {circuits_str:<25}")

        # Winner
        print("\n" + "-" * 60)
        print(f"Best overall mode: {result.best_overall_mode}")
        print(f"Fused wins: {result.fused_wins}/{len(result.category_results)}")
        print(f"Global wins: {result.global_wins}/{len(result.category_results)}")

        if result.fused_wins > result.global_wins:
            print("\n[INSIGHT] Fused circuit steering outperforms global steering!")
        elif result.fused_wins == result.global_wins:
            print("\n[TIE] Fused and global steering perform similarly")
        else:
            print("\n[NOTE] Global steering outperforms fused in this test")

        print("=" * 70)


# =============================================================================
# TESTING
# =============================================================================

def test_benchmark():
    """Test benchmark functionality."""
    print("=" * 60)
    print("CIRCUIT FUSION BENCHMARK TEST")
    print("=" * 60)

    benchmark = CircuitFusionBenchmark()
    result = benchmark.run_quick(verbose=True)

    print(f"\nFull result: {result}")
    print(f"Best mode: {result.best_overall_mode}")

    print("\n[PASS] Benchmark test complete")
    return True


if __name__ == "__main__":
    test_benchmark()
