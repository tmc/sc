"""
Benchmark: SC-Guided vs Generic JSON Guidance.

Compares:
1. SC-Guided: Strict SC grammar enforcement
2. Generic JSON: Only enforces valid JSON syntax
3. Unguided: No constraints

Metrics:
- JSON validity rate
- SC validity rate
- Average token count
- Generation time
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .sc_grammar import SCGrammar, tokenize_sc_json
from .sc_guided_sampler import SCGuidedSampler, GenericJSONSampler, SamplerConfig, GenerationResult


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_samples: int = 100
    max_tokens: int = 200
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method: str
    num_samples: int
    json_valid_count: int = 0
    sc_valid_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    string_limit_hits: int = 0  # Total times string limit was enforced

    @property
    def json_validity(self) -> float:
        return self.json_valid_count / self.num_samples if self.num_samples > 0 else 0.0

    @property
    def sc_validity(self) -> float:
        return self.sc_valid_count / self.num_samples if self.num_samples > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.num_samples if self.num_samples > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.num_samples if self.num_samples > 0 else 0.0

    def summary(self) -> str:
        return (f"{self.method}: json={self.json_validity:.1%}, "
                f"sc={self.sc_validity:.1%}, avg_tokens={self.avg_tokens:.1f}, "
                f"time={self.avg_time*1000:.2f}ms, string_limits={self.string_limit_hits}")


class UnconstrainedGenerator:
    """Generates random JSON-like output without constraints."""

    TOKENS = [
        "{", "}", "[", "]", ":", ",",
        '"root_state"', '"label"', '"type"', '"children"',
        '"transitions"', '"from"', '"to"', '"event"',
        '"A"', '"B"', '"On"', '"Off"',
        "1", "2", "3", "true", "false",
    ]

    def generate(self) -> GenerationResult:
        """Generate unconstrained output."""
        result = GenerationResult()

        # Random token sequence
        tokens = []
        for _ in range(random.randint(20, 100)):
            tokens.append(random.choice(self.TOKENS))

        result.tokens = tokens
        result.text = "".join(tokens)
        result.num_tokens = len(tokens)

        # Check validity
        try:
            json.loads(result.text)
            result.is_valid_json = True
        except:
            result.is_valid_json = False

        # Check SC validity
        if result.is_valid_json:
            try:
                data = json.loads(result.text)
                if isinstance(data, dict) and "root_state" in data:
                    root = data["root_state"]
                    if isinstance(root, dict) and "label" in root:
                        result.is_valid_sc = True
            except:
                pass

        return result


def run_benchmark(config: BenchmarkConfig = None) -> Dict[str, BenchmarkResult]:
    """
    Run full benchmark comparing all methods.

    Returns dict of method -> BenchmarkResult.
    """
    config = config or BenchmarkConfig()
    random.seed(config.seed)

    results = {}

    # 1. SC-Guided
    print("Running SC-Guided generation...")
    sc_result = BenchmarkResult(method="SC-Guided", num_samples=config.num_samples)
    sampler = SCGuidedSampler(config=SamplerConfig(max_tokens=config.max_tokens))
    grammar = SCGrammar()

    start = time.time()
    for _ in range(config.num_samples):
        gen = sampler.generate()
        sc_result.total_tokens += gen.num_tokens
        sc_result.string_limit_hits += gen.string_limit_hits
        if gen.is_valid_json:
            sc_result.json_valid_count += 1
        if gen.is_valid_sc:
            sc_result.sc_valid_count += 1
    sc_result.total_time = time.time() - start
    results["SC-Guided"] = sc_result

    # 2. Generic JSON
    print("Running Generic JSON generation...")
    json_result = BenchmarkResult(method="Generic JSON", num_samples=config.num_samples)
    json_sampler = GenericJSONSampler()

    start = time.time()
    for _ in range(config.num_samples):
        gen = json_sampler.generate()
        json_result.total_tokens += gen.num_tokens

        if gen.is_valid_json:
            json_result.json_valid_count += 1

            # Check SC validity
            try:
                data = json.loads(gen.text)
                if isinstance(data, dict) and "root_state" in data:
                    root = data.get("root_state", {})
                    if isinstance(root, dict) and "label" in root:
                        json_result.sc_valid_count += 1
            except:
                pass
    json_result.total_time = time.time() - start
    results["Generic JSON"] = json_result

    # 3. Unconstrained
    print("Running Unconstrained generation...")
    uncon_result = BenchmarkResult(method="Unconstrained", num_samples=config.num_samples)
    uncon_gen = UnconstrainedGenerator()

    start = time.time()
    for _ in range(config.num_samples):
        gen = uncon_gen.generate()
        uncon_result.total_tokens += gen.num_tokens
        if gen.is_valid_json:
            uncon_result.json_valid_count += 1
        if gen.is_valid_sc:
            uncon_result.sc_valid_count += 1
    uncon_result.total_time = time.time() - start
    results["Unconstrained"] = uncon_result

    return results


def print_benchmark_results(results: Dict[str, BenchmarkResult]):
    """Print formatted benchmark results."""
    print("\n" + "=" * 85)
    print("BENCHMARK RESULTS")
    print("=" * 85)
    print(f"{'Method':<20} {'JSON Valid':>12} {'SC Valid':>12} {'Avg Tokens':>12} {'Time/Gen':>12} {'StrLimits':>12}")
    print("-" * 85)

    for method, result in results.items():
        print(f"{method:<20} {result.json_validity:>11.1%} {result.sc_validity:>11.1%} "
              f"{result.avg_tokens:>11.1f} {result.avg_time*1000:>10.2f}ms {result.string_limit_hits:>11}")

    print("=" * 85)

    # Key insights
    sc_guided = results.get("SC-Guided")
    generic = results.get("Generic JSON")
    uncon = results.get("Unconstrained")

    print("\nKey Insights:")
    if sc_guided and generic:
        sc_improvement = sc_guided.sc_validity - generic.sc_validity
        print(f"  SC-Guided SC validity improvement: +{sc_improvement:.1%} over Generic JSON")

    if sc_guided and uncon:
        sc_improvement = sc_guided.sc_validity - uncon.sc_validity
        print(f"  SC-Guided SC validity improvement: +{sc_improvement:.1%} over Unconstrained")


def test_benchmark():
    """Test benchmark."""
    print("=" * 60)
    print("Testing SC Guided Decoding Benchmark")
    print("=" * 60)

    # Run smaller benchmark for testing
    config = BenchmarkConfig(num_samples=50, seed=42)
    results = run_benchmark(config)

    print_benchmark_results(results)

    # Extract key metrics
    sc_guided = results["SC-Guided"]

    print(f"\nFinal metrics for reporting:")
    print(f"  validity={sc_guided.json_validity:.1%}")
    print(f"  sc_valid={sc_guided.sc_validity:.1%}")
    print(f"  avg_tokens={sc_guided.avg_tokens:.1f}")
    print(f"  string_limit_hits={sc_guided.string_limit_hits}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_benchmark()
