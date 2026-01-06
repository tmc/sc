"""
Benchmark for Live Generation Visualization.

Measures:
1. PARSE LATENCY: Time to parse partial JSON at each token
2. UPDATE RATE: Tokens per second with visualization
3. PARSE SUCCESS: % of partial parses that succeed
4. END-TO-END: Total generation + visualization time

Target metrics:
- Parse latency: <10ms per token
- Update rate: >20 tokens/sec
- Parse success: >80% during generation
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from statistics import mean, stdev

from .partial_parser import PartialParser, ParseState
from .generation_demo import (
    MockLLM, LiveGenerator, GenerationDemo,
    DemoConfig, GenerationSpeed, LocalTokenUpdate
)


@dataclass
class LatencyTest:
    """Results from a latency test."""
    name: str
    samples: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return mean(self.samples) * 1000 if self.samples else 0

    @property
    def std_ms(self) -> float:
        return stdev(self.samples) * 1000 if len(self.samples) > 1 else 0

    @property
    def max_ms(self) -> float:
        return max(self.samples) * 1000 if self.samples else 0

    @property
    def min_ms(self) -> float:
        return min(self.samples) * 1000 if self.samples else 0


@dataclass
class BenchmarkResult:
    """Results from running the benchmark."""
    total_tokens: int
    total_time: float
    tokens_per_second: float
    parse_success_rate: float
    parse_latency: LatencyTest
    update_latency: LatencyTest
    states_discovered: List[str]
    final_valid: bool
    details: Dict = field(default_factory=dict)


class VizBenchmark:
    """
    Benchmark for live visualization performance.

    Tests the integration of:
    - Token generation
    - Partial parsing
    - Visualization updates
    """

    def __init__(self):
        self.parser = PartialParser()
        self.results: List[BenchmarkResult] = []

    def benchmark_parse_latency(
        self,
        num_samples: int = 100,
        verbose: bool = False
    ) -> LatencyTest:
        """
        Benchmark partial JSON parsing latency.

        Args:
            num_samples: Number of parse operations to time
            verbose: Print progress

        Returns:
            LatencyTest with timing results
        """
        test = LatencyTest(name="parse_latency")

        # Generate test strings of increasing length
        llm = MockLLM("traffic_light")
        full_output = llm.get_full_output()

        # Sample at different positions
        positions = [
            int(i * len(full_output) / num_samples)
            for i in range(1, num_samples + 1)
        ]

        for pos in positions:
            partial = full_output[:pos]

            start = time.perf_counter()
            self.parser.parse(partial)
            elapsed = time.perf_counter() - start

            test.samples.append(elapsed)

        if verbose:
            print(f"Parse latency: {test.mean_ms:.2f}ms (std: {test.std_ms:.2f}ms)")

        return test

    def benchmark_update_rate(
        self,
        template: str = "traffic_light",
        verbose: bool = False
    ) -> LatencyTest:
        """
        Benchmark update rate during generation.

        Args:
            template: Template to use
            verbose: Print progress

        Returns:
            LatencyTest with update timings
        """
        test = LatencyTest(name="update_rate")

        llm = MockLLM(template)
        accumulated = ""

        async def run():
            nonlocal accumulated
            async for token in llm.generate_tokens("", delay=0):
                start = time.perf_counter()

                accumulated += token
                self.parser.parse(accumulated)

                elapsed = time.perf_counter() - start
                test.samples.append(elapsed)

        asyncio.run(run())

        if verbose:
            tokens_per_sec = 1.0 / test.mean_ms * 1000 if test.mean_ms > 0 else float('inf')
            print(f"Update rate: {tokens_per_sec:.0f} tokens/sec")

        return test

    def benchmark_full_generation(
        self,
        prompt: str = "Create a traffic light",
        speed: GenerationSpeed = GenerationSpeed.INSTANT,
        verbose: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark full generation with visualization.

        Args:
            prompt: Generation prompt
            speed: Generation speed
            verbose: Print progress

        Returns:
            BenchmarkResult with all metrics
        """
        config = DemoConfig(speed=speed)
        generator = LiveGenerator(config)

        parse_latency = LatencyTest(name="parse_latency")
        update_latency = LatencyTest(name="update_latency")
        states_seen: List[str] = []
        successful_parses = 0
        total_parses = 0

        def on_token(update: LocalTokenUpdate):
            nonlocal successful_parses, total_parses
            total_parses += 1
            if update.parse_success:
                successful_parses += 1
            for state in update.states_found:
                if state not in states_seen:
                    states_seen.append(state)

        start = time.perf_counter()
        result = asyncio.run(generator.generate(prompt, on_token=on_token))
        total_time = time.perf_counter() - start

        metrics = generator.get_metrics()

        benchmark_result = BenchmarkResult(
            total_tokens=metrics['tokens_generated'],
            total_time=total_time,
            tokens_per_second=metrics['tokens_generated'] / total_time if total_time > 0 else 0,
            parse_success_rate=successful_parses / max(1, total_parses),
            parse_latency=parse_latency,
            update_latency=update_latency,
            states_discovered=states_seen,
            final_valid='root_state' in result and result.get('transitions'),
            details={
                'prompt': prompt,
                'speed': speed.name,
                'metrics': metrics,
            }
        )

        self.results.append(benchmark_result)

        if verbose:
            print(f"Tokens: {benchmark_result.total_tokens}")
            print(f"Time: {benchmark_result.total_time:.3f}s")
            print(f"Rate: {benchmark_result.tokens_per_second:.0f} tok/s")
            print(f"Parse success: {benchmark_result.parse_success_rate:.1%}")

        return benchmark_result

    def run(self, verbose: bool = True) -> Dict:
        """
        Run full benchmark suite.

        Returns:
            Summary of all benchmark results
        """
        if verbose:
            print("=" * 70)
            print("LIVE GENERATION VISUALIZATION BENCHMARK")
            print("=" * 70)
            print("Targets:")
            print("  - Parse latency: <10ms per token")
            print("  - Update rate: >20 tokens/sec")
            print("  - Parse success: >80% during generation")
            print("-" * 70)

        # Test 1: Parse latency
        if verbose:
            print("\n1. Parse Latency Test:")
        parse_test = self.benchmark_parse_latency(num_samples=50, verbose=verbose)
        parse_pass = parse_test.mean_ms < 10

        # Test 2: Update rate
        if verbose:
            print("\n2. Update Rate Test:")
        update_test = self.benchmark_update_rate(verbose=verbose)
        tokens_per_sec = 1.0 / (update_test.mean_ms / 1000) if update_test.mean_ms > 0 else float('inf')
        rate_pass = tokens_per_sec > 20

        # Test 3: Full generation (multiple templates)
        if verbose:
            print("\n3. Full Generation Tests:")

        templates = [
            ("traffic_light", "Create a traffic light"),
            ("toggle", "Create a toggle switch"),
            ("player", "Create a media player"),
        ]

        gen_results = []
        for template, prompt in templates:
            if verbose:
                print(f"\n   {template}:")
            result = self.benchmark_full_generation(
                prompt=prompt,
                speed=GenerationSpeed.INSTANT,
                verbose=verbose
            )
            gen_results.append(result)

        # Calculate averages
        avg_parse_success = mean(r.parse_success_rate for r in gen_results)
        avg_tokens_per_sec = mean(r.tokens_per_second for r in gen_results)
        all_valid = all(r.final_valid for r in gen_results)

        parse_success_pass = avg_parse_success >= 0.80

        # Summary
        if verbose:
            print("\n" + "-" * 70)
            print("RESULTS SUMMARY:")
            print(f"  Parse latency:   {parse_test.mean_ms:.2f}ms {'PASS' if parse_pass else 'FAIL'} (target: <10ms)")
            print(f"  Update rate:     {tokens_per_sec:.0f} tok/s {'PASS' if rate_pass else 'FAIL'} (target: >20)")
            print(f"  Parse success:   {avg_parse_success:.1%} {'PASS' if parse_success_pass else 'FAIL'} (target: >80%)")
            print(f"  All valid:       {'YES' if all_valid else 'NO'}")

            all_pass = parse_pass and rate_pass and parse_success_pass and all_valid
            print("\n" + "=" * 70)
            if all_pass:
                print("ALL BENCHMARKS PASSED!")
            else:
                print("SOME BENCHMARKS FAILED")
            print("=" * 70)

        return {
            'parse_latency_ms': parse_test.mean_ms,
            'parse_latency_pass': parse_pass,
            'update_rate_tokens_per_sec': tokens_per_sec,
            'update_rate_pass': rate_pass,
            'parse_success_rate': avg_parse_success,
            'parse_success_pass': parse_success_pass,
            'all_valid': all_valid,
            'all_pass': parse_pass and rate_pass and parse_success_pass and all_valid,
            'generation_results': [
                {
                    'prompt': r.details['prompt'],
                    'tokens': r.total_tokens,
                    'time': r.total_time,
                    'rate': r.tokens_per_second,
                    'parse_success': r.parse_success_rate,
                    'states': r.states_discovered,
                    'valid': r.final_valid,
                }
                for r in gen_results
            ]
        }


def run_viz_benchmark(verbose: bool = True) -> Dict:
    """Convenience function to run benchmark."""
    benchmark = VizBenchmark()
    return benchmark.run(verbose=verbose)


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    benchmark = VizBenchmark()

    # Quick tests
    print("\n1. Quick parse latency test:")
    parse_test = benchmark.benchmark_parse_latency(num_samples=10, verbose=True)
    assert parse_test.mean_ms >= 0

    print("\n2. Quick update rate test:")
    update_test = benchmark.benchmark_update_rate(verbose=True)
    assert len(update_test.samples) > 0

    print("\n3. Single generation benchmark:")
    gen_result = benchmark.benchmark_full_generation(
        prompt="Create a toggle",
        speed=GenerationSpeed.INSTANT,
        verbose=True
    )
    assert gen_result.total_tokens > 0
    assert gen_result.final_valid

    print("\n" + "=" * 60)
    print("Benchmark system tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run full benchmark
    results = run_viz_benchmark(verbose=True)

    print("\n\nJSON Results:")
    import json
    print(json.dumps(results, indent=2, default=str))
