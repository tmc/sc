#!/usr/bin/env python3
"""
Unified Model Comparison Benchmark

Compare Qwen-Coder model sizes on statechart generation quality.

Usage:
    python benchmark.py --all-models
    python benchmark.py --model 0.5B
    python benchmark.py --model 0.5B 3B --compare
    python benchmark.py --quick  # Fast test with fewer prompts
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from .model_configs import (
    MODEL_CONFIGS,
    ModelConfig,
    SC_TEST_PROMPTS,
    COMPLEXITY_WEIGHTS,
    get_model_config,
)
from .metrics import (
    SCMetrics,
    AggregatedResults,
    compute_metrics,
    aggregate_results,
)

# Try to import model loading utilities
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.mlux_loader import load_model, ModelBackend, HookedModelWrapper
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False

try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    models: List[str] = field(default_factory=lambda: ["0.5B"])
    prompts: List[Dict] = field(default_factory=lambda: SC_TEST_PROMPTS)
    max_tokens: int = 512
    temperature: float = 0.1
    num_runs: int = 1  # Runs per prompt
    output_dir: str = "./results"
    verbose: bool = True


@dataclass
class ModelResult:
    """Results for a single model."""
    model_size: str
    config: ModelConfig
    metrics: List[SCMetrics]
    aggregated: AggregatedResults
    load_time_ms: float = 0.0
    total_time_ms: float = 0.0


class ModelBenchmark:
    """Benchmark runner for a single model."""

    def __init__(self, model_size: str, config: BenchmarkConfig):
        self.model_size = model_size
        self.model_config = get_model_config(model_size)
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> float:
        """Load model, return load time in ms."""
        start = time.time()

        if HAS_LOADER:
            wrapper = load_model(self.model_config.mlx_name)
            self.model = wrapper
            self.tokenizer = wrapper.tokenizer
        elif HAS_MLX_LM:
            self.model, self.tokenizer = load(self.model_config.mlx_name)
        else:
            print(f"[MOCK] Would load {self.model_config.mlx_name}")
            self.model = None
            self.tokenizer = None

        load_time = (time.time() - start) * 1000
        if self.config.verbose:
            print(f"Loaded {self.model_config.display_name} in {load_time:.0f}ms")
        return load_time

    def generate(self, prompt: str) -> tuple[str, float, int]:
        """Generate SC from prompt. Returns (output, time_ms, token_count)."""
        system_prompt = """You are an expert at generating statechart definitions in JSON format.
Generate valid statechart JSON following this structure:
{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [...]
  },
  "transitions": [...]
}

State types: 1=BASIC, 2=NORMAL (OR), 3=PARALLEL (AND)
Output ONLY valid JSON, no explanations."""

        full_prompt = f"{system_prompt}\n\nTask: {prompt}\n\nJSON:"

        start = time.time()

        if self.model is None:
            # Mock generation
            output = self._mock_generate(prompt)
            token_count = len(output.split())
        elif HAS_LOADER and isinstance(self.model, HookedModelWrapper):
            # Check if wrapper is in mock mode
            if self.model.backend == ModelBackend.MOCK:
                output = self._mock_generate(prompt)
                token_count = len(output.split())
            else:
                output = self.model.generate(full_prompt)
                token_count = len(self.tokenizer.encode(output)) if self.tokenizer else len(output.split())
        elif HAS_MLX_LM:
            output = generate(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature,
            )
            token_count = len(self.tokenizer.encode(output))
        else:
            output = self._mock_generate(prompt)
            token_count = len(output.split())

        gen_time = (time.time() - start) * 1000
        return output, gen_time, token_count

    def _mock_generate(self, prompt: str) -> str:
        """Generate mock SC for testing without model."""
        # Simple mock based on prompt keywords
        if "toggle" in prompt.lower():
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Off", "type": 1, "is_initial": True},
                        {"label": "On", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
                    {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"}
                ]
            })
        elif "traffic" in prompt.lower():
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Red", "type": 1, "is_initial": True},
                        {"label": "Yellow", "type": 1},
                        {"label": "Green", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                    {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                    {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"}
                ]
            })
        else:
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Initial", "type": 1, "is_initial": True},
                        {"label": "Final", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Initial"], "to": ["Final"], "event": "COMPLETE"}
                ]
            })

    def run(self) -> ModelResult:
        """Run benchmark for this model."""
        load_time = self.load_model()

        metrics_list = []
        start = time.time()

        for prompt_config in self.config.prompts:
            for run in range(self.config.num_runs):
                if self.config.verbose:
                    print(f"  [{prompt_config['name']}] Run {run + 1}/{self.config.num_runs}...", end=" ")

                output, gen_time, token_count = self.generate(prompt_config["prompt"])

                metrics = compute_metrics(
                    output=output,
                    prompt_name=prompt_config["name"],
                    model_size=self.model_size,
                    expected_states=prompt_config.get("expected_states", 0),
                    expected_transitions=prompt_config.get("expected_transitions", 0),
                    generation_time_ms=gen_time,
                    token_count=token_count,
                )

                metrics_list.append(metrics)

                if self.config.verbose:
                    status = "✓" if metrics.is_valid_semantic else "✗"
                    print(f"{status} ({gen_time:.0f}ms, score={metrics.overall_score:.2f})")

        total_time = (time.time() - start) * 1000

        return ModelResult(
            model_size=self.model_size,
            config=self.model_config,
            metrics=metrics_list,
            aggregated=aggregate_results(metrics_list),
            load_time_ms=load_time,
            total_time_ms=total_time,
        )


class ComparisonRunner:
    """Run comparison across multiple models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, ModelResult] = {}

    def run(self) -> Dict[str, ModelResult]:
        """Run benchmark for all configured models."""
        for model_size in self.config.models:
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking {model_size}")
                print(f"{'='*60}")

            benchmark = ModelBenchmark(model_size, self.config)
            result = benchmark.run()
            self.results[model_size] = result

        return self.results

    def print_comparison(self):
        """Print comparison table."""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        # Header
        print(f"\n{'Model':<12} {'JSON%':>8} {'Struct%':>8} {'Semantic%':>10} {'Overall':>8} {'Time(ms)':>10} {'Tok/s':>8}")
        print("-" * 80)

        for size in sorted(self.results.keys(), key=lambda x: float(x.replace("B", ""))):
            r = self.results[size]
            agg = r.aggregated
            n = agg.total_prompts

            json_rate = agg.valid_json_count / n * 100 if n > 0 else 0
            struct_rate = agg.valid_structure_count / n * 100 if n > 0 else 0
            sem_rate = agg.valid_semantic_count / n * 100 if n > 0 else 0

            print(f"{size:<12} {json_rate:>7.1f}% {struct_rate:>7.1f}% {sem_rate:>9.1f}% "
                  f"{agg.avg_overall_score:>7.2f} {agg.avg_generation_time_ms:>9.0f} "
                  f"{agg.avg_tokens_per_second:>7.1f}")

        print("-" * 80)

        # Best model
        best = max(self.results.items(), key=lambda x: x[1].aggregated.avg_overall_score)
        print(f"\nBest overall: {best[0]} (score: {best[1].aggregated.avg_overall_score:.3f})")

        # Fastest model
        fastest = min(self.results.items(), key=lambda x: x[1].aggregated.avg_generation_time_ms)
        print(f"Fastest: {fastest[0]} ({fastest[1].aggregated.avg_generation_time_ms:.0f}ms avg)")

    def save_results(self, output_path: str):
        """Save results to JSON."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "models": self.config.models,
                "num_prompts": len(self.config.prompts),
                "num_runs": self.config.num_runs,
            },
            "results": {
                size: {
                    "model_config": {
                        "name": r.config.name,
                        "size": r.config.size,
                        "params_billions": r.config.params_billions,
                    },
                    "aggregated": r.aggregated.to_dict(),
                    "load_time_ms": r.load_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "individual_metrics": [m.to_dict() for m in r.metrics],
                }
                for size, r in self.results.items()
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_path}")


def run_all_models(config: Optional[BenchmarkConfig] = None) -> Dict[str, ModelResult]:
    """Run benchmark for all available models."""
    if config is None:
        config = BenchmarkConfig(models=list(MODEL_CONFIGS.keys()))

    runner = ComparisonRunner(config)
    results = runner.run()
    runner.print_comparison()
    return results


def run_comparison(models: List[str], config: Optional[BenchmarkConfig] = None) -> Dict[str, ModelResult]:
    """Run comparison for specific models."""
    if config is None:
        config = BenchmarkConfig(models=models)
    else:
        config.models = models

    runner = ComparisonRunner(config)
    results = runner.run()
    runner.print_comparison()
    return results


def generate_report(results: Dict[str, ModelResult], output_path: str = "./results/comparison_report.json"):
    """Generate comprehensive report."""
    runner = ComparisonRunner(BenchmarkConfig())
    runner.results = results
    runner.save_results(output_path)


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument("--all-models", action="store_true", help="Test all available models")
    parser.add_argument("--model", nargs="+", default=["0.5B"], help="Model sizes to test")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer prompts")
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--output", default="./results/comparison.json", help="Output file")
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Configure
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = args.model

    prompts = SC_TEST_PROMPTS[:3] if args.quick else SC_TEST_PROMPTS

    config = BenchmarkConfig(
        models=models,
        prompts=prompts,
        num_runs=args.runs,
        output_dir=str(Path(args.output).parent),
        verbose=not args.quiet,
    )

    # Run
    print(f"Benchmarking models: {', '.join(models)}")
    print(f"Prompts: {len(prompts)}, Runs: {args.runs}")

    runner = ComparisonRunner(config)
    results = runner.run()

    if args.compare or len(models) > 1:
        runner.print_comparison()

    runner.save_results(args.output)


if __name__ == "__main__":
    main()
