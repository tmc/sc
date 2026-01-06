"""
Benchmark: Compare steered vs baseline generation on edge cases.

Tests statechart generation quality with and without steering vectors.
Focuses on edge cases where the model typically produces invalid output.

Integrates with exp_mlux_sc_steering for steering vector computation.
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

from .steered_sampler import SteeredSampler, SamplerConfig, GenerationResult
from .hook_injection import create_steering_hook


# =============================================================================
# EDGE CASE PROMPTS
# =============================================================================

# These are prompts where models often produce invalid statecharts
EDGE_CASE_PROMPTS = [
    # Minimal - might forget initial state
    "Generate a minimal statechart with exactly 2 states",

    # Complex - might produce syntax errors
    "Generate a statechart for a complex workflow with 5+ states and guards",

    # Nested - might break hierarchy
    "Generate a hierarchical statechart with nested states",

    # Parallel - might confuse AND/OR states
    "Generate a parallel statechart with orthogonal regions",

    # History - might misuse history states
    "Generate a statechart with history state",

    # Ambiguous - might produce incomplete
    "Generate a statechart JSON",

    # Specific but tricky
    "Generate a statechart for user authentication with timeout handling",

    # Game logic - complex transitions
    "Generate a statechart for a game character with combat and inventory states",
]

# Prompts that are easier (for baseline comparison)
STANDARD_PROMPTS = [
    "Generate a simple on/off toggle statechart",
    "Generate a traffic light statechart with 3 states",
    "Generate a basic order processing statechart",
]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_statechart(text: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate generated statechart JSON.

    Returns:
        (is_valid, error_message, parsed_sc)
    """
    import re

    # Handle bytes
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')

    # Extract JSON
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find JSON object
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

    # Structural validation
    if "root_state" not in sc:
        return False, "Missing root_state", None

    root = sc["root_state"]
    children = root.get("children", [])

    if len(children) == 0:
        return False, "Empty children", None

    # Initial state check
    has_initial = any(c.get("is_initial", False) for c in children)
    if not has_initial:
        return False, "No initial state", None

    # State labels check
    for child in children:
        if "label" not in child:
            return False, "State missing label", None

    # Transition validation
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
# BENCHMARK RESULT
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    prompt: str
    baseline_valid: bool
    steered_valid: bool
    baseline_error: str
    steered_error: str
    baseline_output: str
    steered_output: str
    improved: bool  # Steered valid when baseline was not


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_prompts: int
    baseline_valid: int
    steered_valid: int
    baseline_rate: float
    steered_rate: float
    improvement: float
    improvements_count: int  # Cases where steering fixed invalid output
    regressions_count: int  # Cases where steering broke valid output
    results: List[BenchmarkResult]

    def __repr__(self):
        return (f"BenchmarkSummary(baseline={self.baseline_rate:.1%}, "
                f"steered={self.steered_rate:.1%}, "
                f"improvement={self.improvement:+.1%}, "
                f"fixes={self.improvements_count})")


# =============================================================================
# BENCHMARK
# =============================================================================

class SteeredDecodingBenchmark:
    """
    Benchmark comparing steered vs baseline statechart generation.

    Integrates with exp_mlux_sc_steering for steering vectors.
    """

    IMPROVEMENT_TARGET = 0.10  # +10%

    def __init__(
        self,
        sampler: SteeredSampler = None,
        steering_vector: Any = None,
        config: SamplerConfig = None,
    ):
        self.sampler = sampler or SteeredSampler()
        self.steering_vector = steering_vector
        self.config = config or SamplerConfig(
            max_tokens=512,
            temperature=0.3,
            steering_alpha=1.0,
            steering_layer=12,
        )
        self.results: List[BenchmarkResult] = []
        self.summary: Optional[BenchmarkSummary] = None

    def load_steering_vector(self, layer: int = 12) -> Any:
        """Load steering vector from exp_mlux_sc_steering."""
        try:
            from experiments.exp_mlux_sc_steering import (
                SteeringVectorComputer,
                create_contrastive_pairs,
            )
            # SteeringVectorComputer will create its own model wrapper
            computer = SteeringVectorComputer()
            vec = computer.compute_averaged(layer=layer)
            self.steering_vector = vec.vector
            print(f"Loaded steering vector: shape={vec.vector.shape if hasattr(vec.vector, 'shape') else 'mock'}")
            return self.steering_vector
        except ImportError as e:
            print(f"Warning: exp_mlux_sc_steering not available: {e}")
            if MLX_AVAILABLE:
                # Create dummy vector with small random values
                self.steering_vector = mx.random.normal((896,)) * 0.1
                print(f"Created dummy steering vector: shape={self.steering_vector.shape}")
            return self.steering_vector
        except Exception as e:
            print(f"Warning: Failed to load steering vector: {e}")
            if MLX_AVAILABLE:
                self.steering_vector = mx.random.normal((896,)) * 0.1
                print(f"Created dummy steering vector: shape={self.steering_vector.shape}")
            return self.steering_vector

    def run_single(self, prompt: str) -> BenchmarkResult:
        """Run benchmark on single prompt."""
        # Baseline generation
        baseline_result = self.sampler.generate(
            prompt,
            steering_vector=None,
            config=self.config,
        )
        baseline_valid, baseline_error, _ = validate_statechart(baseline_result.output)

        # Steered generation
        steered_result = self.sampler.generate(
            prompt,
            steering_vector=self.steering_vector,
            config=self.config,
        )
        steered_valid, steered_error, _ = validate_statechart(steered_result.output)

        improved = steered_valid and not baseline_valid

        return BenchmarkResult(
            prompt=prompt,
            baseline_valid=baseline_valid,
            steered_valid=steered_valid,
            baseline_error=baseline_error,
            steered_error=steered_error,
            baseline_output=baseline_result.output,
            steered_output=steered_result.output,
            improved=improved,
        )

    def run(
        self,
        prompts: List[str] = None,
        verbose: bool = True,
    ) -> BenchmarkSummary:
        """
        Run full benchmark.

        Args:
            prompts: Prompts to test (default: EDGE_CASE_PROMPTS)
            verbose: Print progress

        Returns:
            BenchmarkSummary
        """
        if prompts is None:
            prompts = EDGE_CASE_PROMPTS

        # Load steering vector if not set
        if self.steering_vector is None:
            if verbose:
                print("Loading steering vector...")
            self.load_steering_vector(layer=self.config.steering_layer)

        if verbose:
            print("=" * 70)
            print("STEERED DECODING BENCHMARK")
            print("=" * 70)
            print(f"Prompts: {len(prompts)}")
            print(f"Layer: {self.config.steering_layer}")
            print(f"Alpha: {self.config.steering_alpha}")
            print()

        self.results = []

        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")

            result = self.run_single(prompt)
            self.results.append(result)

            if verbose:
                b_status = "VALID" if result.baseline_valid else f"INVALID ({result.baseline_error})"
                s_status = "VALID" if result.steered_valid else f"INVALID ({result.steered_error})"
                improved = " [IMPROVED]" if result.improved else ""
                print(f"  Baseline: {b_status}")
                print(f"  Steered:  {s_status}{improved}")

        # Compute summary
        baseline_valid = sum(1 for r in self.results if r.baseline_valid)
        steered_valid = sum(1 for r in self.results if r.steered_valid)
        improvements = sum(1 for r in self.results if r.improved)
        regressions = sum(1 for r in self.results
                         if r.baseline_valid and not r.steered_valid)

        baseline_rate = baseline_valid / len(prompts)
        steered_rate = steered_valid / len(prompts)
        improvement = steered_rate - baseline_rate

        self.summary = BenchmarkSummary(
            total_prompts=len(prompts),
            baseline_valid=baseline_valid,
            steered_valid=steered_valid,
            baseline_rate=baseline_rate,
            steered_rate=steered_rate,
            improvement=improvement,
            improvements_count=improvements,
            regressions_count=regressions,
            results=self.results,
        )

        if verbose:
            self._print_summary()

        return self.summary

    def run_edge_cases(self, verbose: bool = True) -> BenchmarkSummary:
        """Run benchmark on edge case prompts only."""
        return self.run(EDGE_CASE_PROMPTS, verbose)

    def run_standard(self, verbose: bool = True) -> BenchmarkSummary:
        """Run benchmark on standard (easier) prompts."""
        return self.run(STANDARD_PROMPTS, verbose)

    def run_quick(self, verbose: bool = True) -> BenchmarkSummary:
        """Run quick benchmark with fewer prompts."""
        quick_prompts = EDGE_CASE_PROMPTS[:3] + STANDARD_PROMPTS[:2]
        return self.run(quick_prompts, verbose)

    def _print_summary(self):
        """Print benchmark summary."""
        if not self.summary:
            return

        s = self.summary
        print()
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Total prompts: {s.total_prompts}")
        print(f"Baseline valid: {s.baseline_valid}/{s.total_prompts} ({s.baseline_rate:.1%})")
        print(f"Steered valid:  {s.steered_valid}/{s.total_prompts} ({s.steered_rate:.1%})")
        print(f"Improvement:    {s.improvement:+.1%}")
        print()
        print(f"Cases improved (steering fixed): {s.improvements_count}")
        print(f"Cases regressed (steering broke): {s.regressions_count}")
        print()

        if s.improvement >= self.IMPROVEMENT_TARGET:
            print(f"[PASS] Target (+10%) achieved: {s.improvement:+.1%}")
        else:
            print(f"[WORK NEEDED] Target (+10%) not met: {s.improvement:+.1%}")

        print("=" * 70)

    @property
    def passed(self) -> bool:
        """Check if benchmark meets target."""
        if not self.summary:
            return False
        return self.summary.improvement >= self.IMPROVEMENT_TARGET


# =============================================================================
# LAYER/ALPHA SEARCH
# =============================================================================

def grid_search(
    sampler: SteeredSampler,
    prompts: List[str] = None,
    layers: List[int] = None,
    alphas: List[float] = None,
    verbose: bool = True,
) -> Dict[Tuple[int, float], BenchmarkSummary]:
    """
    Grid search over layers and alphas.

    Returns:
        {(layer, alpha): BenchmarkSummary}
    """
    if prompts is None:
        prompts = EDGE_CASE_PROMPTS[:3]  # Quick search
    if layers is None:
        layers = [6, 12, 18]
    if alphas is None:
        alphas = [0.5, 1.0, 1.5]

    results = {}

    for layer in layers:
        for alpha in alphas:
            if verbose:
                print(f"\nTesting layer={layer}, alpha={alpha}")

            config = SamplerConfig(
                max_tokens=512,
                temperature=0.3,
                steering_layer=layer,
                steering_alpha=alpha,
            )

            benchmark = SteeredDecodingBenchmark(sampler=sampler, config=config)
            summary = benchmark.run(prompts, verbose=False)
            results[(layer, alpha)] = summary

            if verbose:
                print(f"  Result: {summary}")

    # Find best
    if verbose and results:
        best_key = max(results.keys(), key=lambda k: results[k].improvement)
        best = results[best_key]
        print(f"\nBest config: layer={best_key[0]}, alpha={best_key[1]}")
        print(f"  Improvement: {best.improvement:+.1%}")

    return results


# =============================================================================
# TESTING
# =============================================================================

def test_benchmark():
    """Test benchmark functionality."""
    print("=" * 60)
    print("STEERED DECODING BENCHMARK TEST")
    print("=" * 60)

    # Create benchmark
    print("\nCreating benchmark...")
    benchmark = SteeredDecodingBenchmark()

    # Run quick benchmark
    print("\nRunning quick benchmark...")
    summary = benchmark.run_quick(verbose=True)

    print(f"\nSummary: {summary}")
    print(f"Passed: {benchmark.passed}")

    print("\n[PASS] Benchmark test complete")
    return True


if __name__ == "__main__":
    test_benchmark()
