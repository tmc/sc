#!/usr/bin/env python3
"""
Guard Outcome Prediction Benchmark.

Evaluates LLM ability to predict which guard fires given context values.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .guard_predictor import (
    GuardLevel,
    generate_guard_set,
    generate_context_for_guards,
    compute_expected_outcome,
    create_guard_prediction_prompt_few_shot,
    parse_prediction,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    samples_per_level: int = 10
    max_tokens: int = 50
    temperature: float = 0.1  # Low temp for deterministic reasoning


@dataclass
class BenchmarkResult:
    """Result for a single prediction."""
    level: GuardLevel
    guards: List[str]
    context: Dict[str, Any]
    expected: int
    predicted: int
    correct: bool
    raw_output: str


def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run the guard outcome prediction benchmark."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("GUARD OUTCOME PREDICTION BENCHMARK")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Samples per level: {config.samples_per_level}")

    # Load model
    print("\nLoading model...")
    from mlx_lm import load, generate as mlx_generate

    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    # Results tracking
    results: List[BenchmarkResult] = []
    level_correct = {level: 0 for level in GuardLevel}
    level_total = {level: 0 for level in GuardLevel}

    # Run benchmark for each level
    print("\n" + "-" * 60)
    print("RUNNING BENCHMARK")
    print("-" * 60)

    for level in GuardLevel:
        print(f"\n{level.name}:")

        for sample_idx in range(config.samples_per_level):
            # Generate guard set and context
            guard_set = generate_guard_set(level)

            # Randomly pick which guard should fire
            target_idx = sample_idx % len(guard_set.guards)
            context = generate_context_for_guards(guard_set, target_idx)

            # Compute expected outcome
            expected = compute_expected_outcome(guard_set, context)

            # Create prompt - simple completion format without chat template
            prompt = create_guard_prediction_prompt_few_shot(guard_set, context)

            # Generate prediction (simple completion, not chat)
            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
            )

            # Parse prediction
            predicted = parse_prediction(output)

            # Check correctness
            correct = (predicted == expected)

            result = BenchmarkResult(
                level=level,
                guards=guard_set.guards,
                context=context,
                expected=expected,
                predicted=predicted,
                correct=correct,
                raw_output=output,
            )
            results.append(result)

            level_total[level] += 1
            if correct:
                level_correct[level] += 1

            status = "✓" if correct else "✗"
            print(f"  {status} Sample {sample_idx + 1}: expected={expected}, predicted={predicted}")

    # Compute accuracies
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    level_accuracy = {}
    for level in GuardLevel:
        acc = level_correct[level] / level_total[level] if level_total[level] > 0 else 0
        level_accuracy[level] = acc
        print(f"{level.name}: {acc:.1%} ({level_correct[level]}/{level_total[level]})")

    total_correct = sum(level_correct.values())
    total_samples = sum(level_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"\nOverall: {overall_accuracy:.1%} ({total_correct}/{total_samples})")

    # Format for report
    by_level_str = ", ".join(
        f"L{level.value}:{level_accuracy[level]:.0%}"
        for level in GuardLevel
    )
    print(f"\nGUARD_OUTCOME accuracy={overall_accuracy:.0%}, by_level=[{by_level_str}]")

    return {
        'overall_accuracy': overall_accuracy,
        'level_accuracy': {level.name: level_accuracy[level] for level in GuardLevel},
        'results': results,
        'total_correct': total_correct,
        'total_samples': total_samples,
    }


if __name__ == "__main__":
    result = run_benchmark()
