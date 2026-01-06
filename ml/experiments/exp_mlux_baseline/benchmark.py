#!/usr/bin/env python3
"""
Benchmark: Performance comparison across mlux/mlx_lm backends.

Measures:
1. Generation speed (tokens/second)
2. Memory usage with/without caching
3. Overhead of interpretability features
4. Statechart validity rates
"""

import json
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Add utils to path
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    CacheConfig,
    MLUX_AVAILABLE,
    MLX_LM_AVAILABLE,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    backend: str
    prompt_tokens: int
    output_tokens: int
    generation_time: float
    tokens_per_second: float
    with_cache: bool
    valid_json: bool
    has_statechart_structure: bool


@dataclass
class BenchmarkSuite:
    """Results of full benchmark suite."""
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tokens_per_second for r in self.results) / len(self.results)

    @property
    def validity_rate(self) -> float:
        if not self.results:
            return 0.0
        valid = sum(1 for r in self.results if r.valid_json)
        return valid / len(self.results)

    @property
    def cache_overhead(self) -> float:
        """Calculate overhead percentage of caching."""
        cached = [r for r in self.results if r.with_cache]
        uncached = [r for r in self.results if not r.with_cache]

        if not cached or not uncached:
            return 0.0

        cached_avg = sum(r.generation_time for r in cached) / len(cached)
        uncached_avg = sum(r.generation_time for r in uncached) / len(uncached)

        if uncached_avg == 0:
            return 0.0

        return ((cached_avg - uncached_avg) / uncached_avg) * 100


BENCHMARK_PROMPTS = [
    # Simple statechart
    """Generate a JSON statechart for a toggle switch with On/Off states:
```json
""",

    # Medium complexity
    """Generate a JSON statechart for a traffic light with Red/Yellow/Green states:
```json
""",

    # More complex
    """Generate a JSON statechart for user authentication with Login/Authenticated/Error states:
```json
""",

    # Hierarchical hint
    """Generate a JSON statechart for a media player with Playing (containing Normal/Fast) and Stopped states:
```json
""",
]


class MLUXBenchmark:
    """Benchmark suite for mlux integration."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        self.model_name = model_name
        self._model: Optional[HookedModelWrapper] = None

    def _load_model(self, backend: Optional[ModelBackend] = None):
        """Load model with specified backend."""
        self._model = load_model(self.model_name, backend=backend)

    def benchmark_generation(
        self,
        prompt: str,
        with_cache: bool = False,
        config: Optional[GenerationConfig] = None,
    ) -> BenchmarkResult:
        """Benchmark a single generation."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        config = config or GenerationConfig(max_tokens=256, temperature=0.3)

        start_time = time.time()

        if with_cache and self._model.has_interpretability:
            cache_config = CacheConfig(
                hooks=["model.layers.*.mlp"],
                include_attention=True,
            )
            output, cache = self._model.generate_with_cache(prompt, config, cache_config)
        else:
            output = self._model.generate(prompt, config)

        generation_time = time.time() - start_time

        # Estimate tokens
        prompt_tokens = len(prompt.split())
        output_tokens = len(output.split())

        # Check validity
        valid_json, has_structure = self._check_output(output)

        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        return BenchmarkResult(
            backend=self._model.backend.name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            with_cache=with_cache,
            valid_json=valid_json,
            has_statechart_structure=has_structure,
        )

    def _check_output(self, output: str) -> tuple:
        """Check if output is valid JSON statechart."""
        import re

        # Clean output
        text = re.sub(r'```json\s*', '', output)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Find JSON
        start = text.find('{')
        if start == -1:
            return False, False

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
            data = json.loads(text[start:end])
            has_structure = "root_state" in data or "states" in data
            return True, has_structure
        except json.JSONDecodeError:
            return False, False

    def run_suite(
        self,
        num_iterations: int = 3,
        backends: Optional[List[ModelBackend]] = None,
    ) -> BenchmarkSuite:
        """Run full benchmark suite."""
        suite = BenchmarkSuite()
        start_time = time.time()

        # Determine backends to test
        if backends is None:
            backends = []
            if MLUX_AVAILABLE:
                backends.append(ModelBackend.MLUX)
            if MLX_LM_AVAILABLE:
                backends.append(ModelBackend.MLX_LM)
            if not backends:
                backends.append(ModelBackend.MOCK)

        print("=" * 60)
        print("MLUX BENCHMARK SUITE")
        print("=" * 60)
        print(f"Backends to test: {[b.name for b in backends]}")
        print(f"Prompts: {len(BENCHMARK_PROMPTS)}")
        print(f"Iterations: {num_iterations}")
        print(f"Total runs: {len(backends) * len(BENCHMARK_PROMPTS) * num_iterations * 2}")
        print("=" * 60)

        for backend in backends:
            print(f"\n--- Backend: {backend.name} ---")
            self._load_model(backend)

            for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS):
                for iteration in range(num_iterations):
                    # Without cache
                    result = self.benchmark_generation(prompt, with_cache=False)
                    suite.results.append(result)

                    status = "OK" if result.valid_json else "FAIL"
                    print(f"  [{status}] Prompt {prompt_idx+1}, iter {iteration+1}: "
                          f"{result.tokens_per_second:.1f} tok/s (no cache)")

                    # With cache (if available)
                    if self._model.has_interpretability:
                        result = self.benchmark_generation(prompt, with_cache=True)
                        suite.results.append(result)

                        status = "OK" if result.valid_json else "FAIL"
                        print(f"  [{status}] Prompt {prompt_idx+1}, iter {iteration+1}: "
                              f"{result.tokens_per_second:.1f} tok/s (with cache)")

        suite.total_time = time.time() - start_time
        return suite

    def print_summary(self, suite: BenchmarkSuite):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Group by backend
        by_backend: Dict[str, List[BenchmarkResult]] = {}
        for r in suite.results:
            if r.backend not in by_backend:
                by_backend[r.backend] = []
            by_backend[r.backend].append(r)

        for backend, results in by_backend.items():
            print(f"\n--- {backend} ---")

            cached = [r for r in results if r.with_cache]
            uncached = [r for r in results if not r.with_cache]

            if uncached:
                avg_speed = sum(r.tokens_per_second for r in uncached) / len(uncached)
                avg_time = sum(r.generation_time for r in uncached) / len(uncached)
                validity = sum(1 for r in uncached if r.valid_json) / len(uncached)
                print(f"  Without cache:")
                print(f"    Speed: {avg_speed:.1f} tokens/sec")
                print(f"    Avg time: {avg_time:.2f}s")
                print(f"    Validity: {validity:.1%}")

            if cached:
                avg_speed = sum(r.tokens_per_second for r in cached) / len(cached)
                avg_time = sum(r.generation_time for r in cached) / len(cached)
                validity = sum(1 for r in cached if r.valid_json) / len(cached)
                print(f"  With cache:")
                print(f"    Speed: {avg_speed:.1f} tokens/sec")
                print(f"    Avg time: {avg_time:.2f}s")
                print(f"    Validity: {validity:.1%}")

        print(f"\n--- Overall ---")
        print(f"  Total runs: {len(suite.results)}")
        print(f"  Total time: {suite.total_time:.1f}s")
        print(f"  Avg tokens/sec: {suite.avg_tokens_per_second:.1f}")
        print(f"  Validity rate: {suite.validity_rate:.1%}")
        print(f"  Cache overhead: {suite.cache_overhead:.1f}%")


def demo():
    """Quick demo with fewer iterations."""
    print("=" * 60)
    print("MLUX BENCHMARK DEMO")
    print("=" * 60)
    print(f"MLUX available: {MLUX_AVAILABLE}")
    print(f"MLX_LM available: {MLX_LM_AVAILABLE}")

    benchmark = MLUXBenchmark()
    suite = benchmark.run_suite(num_iterations=1)
    benchmark.print_summary(suite)

    return suite


def run_full_benchmark():
    """Run full benchmark with multiple iterations."""
    benchmark = MLUXBenchmark()
    suite = benchmark.run_suite(num_iterations=3)
    benchmark.print_summary(suite)

    # Check targets
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    targets_met = 0

    # Target 1: >80% validity
    validity_met = suite.validity_rate >= 0.80
    print(f"  Validity >=80%: {'PASS' if validity_met else 'FAIL'} ({suite.validity_rate:.1%})")
    if validity_met:
        targets_met += 1

    # Target 2: <50% cache overhead
    overhead_ok = suite.cache_overhead < 50
    print(f"  Cache overhead <50%: {'PASS' if overhead_ok else 'FAIL'} ({suite.cache_overhead:.1f}%)")
    if overhead_ok:
        targets_met += 1

    print(f"\nTargets met: {targets_met}/2")

    return suite


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        demo()
