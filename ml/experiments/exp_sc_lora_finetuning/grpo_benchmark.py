"""
GRPO Benchmark: Run training and report results

Executes full GRPO training loop and outputs:
GRPO_LORA: epoch=X, validity=X%, reward_mean=X, baseline_validity=X%

Target: +15% validity improvement over baseline.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json
import time

from .lora_config import LoRAConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig, GRPOResult, SC_PROMPTS, train_grpo
from .grpo_reward import SCRewardFunction


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None

    # GRPO
    n_samples: int = 8
    epochs: int = 10
    batch_size: int = 4
    temperature: float = 0.8

    # Data
    n_prompts: int = 100  # Use 100 prompts

    # Target
    target_improvement: float = 15.0  # +15% validity

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> GRPOResult:
    """
    Run full GRPO benchmark.

    Args:
        config: Benchmark configuration
        output_dir: Output directory
        verbose: Print progress

    Returns:
        GRPOResult with all metrics
    """
    config = config or BenchmarkConfig()
    output_dir = output_dir or Path("./grpo_benchmark_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("GRPO BENCHMARK: SC-Conditioned LoRA Training")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  LoRA rank: {config.lora_rank}")
        print(f"  LoRA alpha: {config.lora_alpha}")
        print(f"  Target modules: {config.target_modules}")
        print(f"  Samples/prompt: {config.n_samples}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Prompts: {config.n_prompts}")
        print(f"  Target improvement: +{config.target_improvement}%")

    # Setup configs
    lora_config = LoRAConfig(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_modules=config.target_modules,
    )

    grpo_config = GRPOConfig(
        n_samples=config.n_samples,
        epochs=config.epochs,
        batch_size=config.batch_size,
        temperature=config.temperature,
        log_interval=5,
    )

    # Expand prompts to target count
    prompts = _generate_prompts(config.n_prompts)

    if verbose:
        print(f"\nStarting training with {len(prompts)} prompts...")
        print("-" * 70)

    # Run training
    result = train_grpo(
        prompts=prompts,
        lora_config=lora_config,
        grpo_config=grpo_config,
        output_dir=output_dir,
    )

    # Check if target met
    target_met = result.improvement >= config.target_improvement

    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(result.summary())
        print(f"\nTarget: +{config.target_improvement}% improvement")
        print(f"Achieved: {result.improvement:+.1f}%")
        print(f"TARGET MET: {'YES' if target_met else 'NO'}")

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "config": {
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "n_samples": config.n_samples,
                "epochs": config.epochs,
                "n_prompts": config.n_prompts,
            },
            "results": {
                "baseline_validity": result.baseline_validity,
                "final_validity": result.final_validity,
                "improvement": result.improvement,
                "best_reward": result.best_reward,
                "training_time": result.training_time_seconds,
            },
            "target_met": target_met,
        }, f, indent=2)

    return result


def _generate_prompts(n: int) -> List[str]:
    """Generate n prompts from templates."""
    base_prompts = SC_PROMPTS.copy()

    # Domain variations
    domains = [
        "smart home", "e-commerce", "gaming", "IoT",
        "banking", "healthcare", "logistics", "education"
    ]

    # Entity variations
    entities = [
        "controller", "manager", "handler", "processor",
        "monitor", "tracker", "scheduler", "coordinator"
    ]

    # Expand with variations
    expanded = list(base_prompts)

    while len(expanded) < n:
        for base in base_prompts:
            if len(expanded) >= n:
                break
            # Add domain variation
            domain = domains[len(expanded) % len(domains)]
            entity = entities[len(expanded) % len(entities)]
            variation = base.replace("statechart", f"{domain} statechart")
            expanded.append(variation)

    return expanded[:n]


def format_report(result: GRPOResult, epoch: int = None) -> str:
    """Format result for reporting."""
    if epoch is None:
        epoch = result.total_epochs

    # Get metrics for specific epoch
    if result.metrics_history:
        epoch_metrics = [m for m in result.metrics_history if m.epoch == epoch - 1]
        if epoch_metrics:
            last_metric = epoch_metrics[-1]
            return (f"GRPO_LORA: epoch={epoch}, "
                    f"validity={last_metric.validity_rate:.0%}, "
                    f"reward_mean={last_metric.reward_mean:.3f}, "
                    f"baseline_validity={result.baseline_validity:.0%}")

    return (f"GRPO_LORA: epoch={epoch}, "
            f"validity={result.final_validity:.0%}, "
            f"reward_mean={result.best_reward:.3f}, "
            f"baseline_validity={result.baseline_validity:.0%}")


def quick_benchmark(verbose: bool = True) -> GRPOResult:
    """Run quick benchmark with reduced settings."""
    config = BenchmarkConfig(
        lora_rank=8,
        n_samples=4,
        epochs=3,
        batch_size=2,
        n_prompts=20,
    )
    return run_benchmark(config, verbose=verbose)


def full_benchmark(verbose: bool = True) -> GRPOResult:
    """Run full benchmark with recommended settings."""
    config = BenchmarkConfig(
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        n_samples=8,
        epochs=10,
        batch_size=4,
        n_prompts=100,
        target_improvement=15.0,
    )
    return run_benchmark(config, verbose=verbose)


def demo():
    """Run quick demo benchmark."""
    print("Running quick GRPO benchmark...")
    result = quick_benchmark()
    report = format_report(result)
    print(f"\nFinal report: {report}")
    return result, report


if __name__ == "__main__":
    result, report = demo()
    print(f"\n{report}")
