"""
Benchmark: SC-Guided Sampling on 1.5B Models

Compares Base vs Instruct Qwen2.5-Coder-1.5B with guided sampling.

MODELS:
- Qwen2.5-Coder-1.5B-Instruct-4bit (expected baseline: 50%)
- Qwen2.5-Coder-1.5B-4bit (expected baseline: 20%)

CONFIGURATIONS:
1. Unguided (baseline)
2. SC-guided sampling

HYPOTHESIS: Guided sampling achieves 100% validity on BOTH models.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    MLX_LM_AVAILABLE,
)

from .guided_sampler import (
    GuidedSampler,
    UnguidedSampler,
    SamplingConfig,
    SamplingResult,
)


# Model configurations
MODELS = {
    "instruct": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "base": "mlx-community/Qwen2.5-Coder-1.5B-4bit",
}

# Test prompts
TEST_PROMPTS = [
    "Generate a JSON statechart for a traffic light with red, yellow, green states:",
    "Create a JSON statechart for user login authentication:",
    "Build a JSON statechart for an order processing workflow:",
    "Design a JSON statechart for a media player:",
    "Generate a JSON statechart for a door lock system:",
]


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    model_name: str
    sampling_type: str
    prompt: str
    is_valid: bool
    output: str
    duration: float
    violations: int = 0


@dataclass
class BenchmarkSummary:
    """Summary statistics for a model/sampling combination."""
    model_name: str
    sampling_type: str
    n_samples: int
    validity_rate: float
    mean_duration: float
    mean_violations: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "sampling": self.sampling_type,
            "validity": f"{self.validity_rate:.1%}",
            "duration": f"{self.mean_duration:.2f}s",
            "violations": f"{self.mean_violations:.1f}",
        }


class GuidedSamplingBenchmark:
    """
    Benchmark for SC-guided sampling on 1.5B models.
    """

    def __init__(
        self,
        verbose: bool = True,
        use_real_models: bool = True,
    ):
        self.verbose = verbose
        self.use_real_models = use_real_models and MLX_LM_AVAILABLE
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

    def load_models(self):
        """Load both models."""
        if not self.use_real_models:
            if self.verbose:
                print("Using mock models for testing")
            return

        for name, model_path in MODELS.items():
            if self.verbose:
                print(f"Loading {name}: {model_path}")

            try:
                from mlx_lm import load
                model, tokenizer = load(model_path)
                self.models[name] = model
                self.tokenizers[name] = tokenizer
            except Exception as e:
                print(f"  Error loading {name}: {e}")

    def run_single(
        self,
        model_name: str,
        sampling_type: str,
        prompt: str,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)

        config = SamplingConfig(
            max_tokens=512,
            temperature=0.7,
            use_grammar=(sampling_type == "guided"),
        )

        if sampling_type == "guided":
            sampler = GuidedSampler(model, tokenizer, config)
        else:
            sampler = UnguidedSampler(model, tokenizer, config)

        # Build full prompt
        full_prompt = f"""You are a statechart expert. Generate valid JSON.

{prompt}

Output format:
{{"root_state": {{"label": "__root__", "type": 2, "children": [...]}}, "transitions": [...]}}

JSON:"""

        result = sampler.sample(full_prompt)

        return BenchmarkResult(
            model_name=model_name,
            sampling_type=sampling_type,
            prompt=prompt,
            is_valid=result.is_valid_sc,
            output=result.output,
            duration=result.duration,
            violations=result.grammar_violations,
        )

    def run_benchmark(
        self,
        n_samples: int = 10,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkSummary]:
        """
        Run full benchmark comparing models and sampling types.

        Args:
            n_samples: Samples per configuration
            prompts: Test prompts

        Returns:
            Summaries for each configuration
        """
        if prompts is None:
            prompts = TEST_PROMPTS

        self.load_models()

        results_by_config: Dict[str, List[BenchmarkResult]] = defaultdict(list)

        total = len(MODELS) * 2 * n_samples
        current = 0

        for model_name in MODELS:
            for sampling_type in ["unguided", "guided"]:
                config_key = f"{model_name}_{sampling_type}"

                if self.verbose:
                    print(f"\nTesting {config_key}...")

                for i in range(n_samples):
                    prompt = prompts[i % len(prompts)]
                    result = self.run_single(model_name, sampling_type, prompt)
                    results_by_config[config_key].append(result)

                    current += 1
                    if self.verbose and current % 5 == 0:
                        print(f"  Progress: {current}/{total}")

        # Compute summaries
        summaries = {}
        for config_key, results in results_by_config.items():
            model_name, sampling_type = config_key.rsplit("_", 1)
            n = len(results)

            summary = BenchmarkSummary(
                model_name=model_name,
                sampling_type=sampling_type,
                n_samples=n,
                validity_rate=sum(1 for r in results if r.is_valid) / n if n > 0 else 0,
                mean_duration=sum(r.duration for r in results) / n if n > 0 else 0,
                mean_violations=sum(r.violations for r in results) / n if n > 0 else 0,
            )
            summaries[config_key] = summary

        return summaries


def run_experiment(
    n_samples: int = 10,
    use_real_models: bool = True,
) -> Tuple[Dict[str, BenchmarkSummary], str]:
    """
    Run the guided sampling experiment.

    Returns:
        (summaries, report_string)
    """
    print("=" * 60)
    print("SC-Guided Sampling Benchmark: 1.5B Models")
    print("=" * 60)

    benchmark = GuidedSamplingBenchmark(
        verbose=True,
        use_real_models=use_real_models,
    )

    summaries = benchmark.run_benchmark(n_samples=n_samples)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Config':<25} {'Validity':<12} {'Duration':<12} {'Violations':<12}")
    print("-" * 60)

    for key in ["instruct_unguided", "instruct_guided", "base_unguided", "base_guided"]:
        if key in summaries:
            s = summaries[key]
            print(f"{key:<25} {s.validity_rate:>10.1%} {s.mean_duration:>10.2f}s {s.mean_violations:>10.1f}")

    # Build report
    instruct_unguided = summaries.get("instruct_unguided", BenchmarkSummary("", "", 0, 0, 0, 0))
    instruct_guided = summaries.get("instruct_guided", BenchmarkSummary("", "", 0, 0, 0, 0))
    base_unguided = summaries.get("base_unguided", BenchmarkSummary("", "", 0, 0, 0, 0))
    base_guided = summaries.get("base_guided", BenchmarkSummary("", "", 0, 0, 0, 0))

    report = (
        f"instruct_unguided={instruct_unguided.validity_rate:.0%}, "
        f"instruct_guided={instruct_guided.validity_rate:.0%}, "
        f"base_unguided={base_unguided.validity_rate:.0%}, "
        f"base_guided={base_guided.validity_rate:.0%}"
    )

    print("\n" + "-" * 60)
    print("ANALYSIS:")

    # Check hypothesis
    if instruct_guided.validity_rate >= 0.95 and base_guided.validity_rate >= 0.95:
        print("  HYPOTHESIS CONFIRMED: Guided sampling achieves ~100% on both models")
    else:
        print("  HYPOTHESIS PARTIAL: Guided sampling improves but not 100%")

    # Improvement deltas
    instruct_delta = instruct_guided.validity_rate - instruct_unguided.validity_rate
    base_delta = base_guided.validity_rate - base_unguided.validity_rate

    print(f"\n  Instruct improvement: {instruct_unguided.validity_rate:.0%} -> {instruct_guided.validity_rate:.0%} ({instruct_delta:+.0%})")
    print(f"  Base improvement: {base_unguided.validity_rate:.0%} -> {base_guided.validity_rate:.0%} ({base_delta:+.0%})")

    return summaries, report


def demo():
    """Demo with mock models."""
    return run_experiment(n_samples=10, use_real_models=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real models")
    parser.add_argument("--samples", type=int, default=10, help="Samples per config")
    args = parser.parse_args()

    summaries, report = run_experiment(
        n_samples=args.samples,
        use_real_models=args.real,
    )
    print(f"\nReport: GUIDED_1.5B: {report}")
