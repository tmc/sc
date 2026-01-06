#!/usr/bin/env python3
"""
Benchmark: Reachability Steps Prediction.

Compares baseline (0%) vs scratchpad approach for min_steps accuracy.
Target: 60%+ steps accuracy while maintaining F1.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from .steps_predictor import (
    ReachabilityCase,
    StepsPredictor,
    create_bfs_scratchpad_prompt,
    create_simple_trace_prompt,
    parse_steps_response,
    get_test_cases,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    max_tokens: int = 200


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    case_name: str
    expected_reachable: bool
    expected_steps: int
    predicted_reachable: bool
    predicted_steps: int
    reachable_correct: bool
    steps_correct: bool
    steps_within_one: bool
    raw_output: str


def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run the steps prediction benchmark."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("REACHABILITY STEPS BENCHMARK")
    print("=" * 60)
    print(f"Model: {config.model_path}")

    # Load model
    print("\nLoading model...")
    from mlx_lm import load, generate as mlx_generate

    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    # Get test cases
    test_cases = get_test_cases()
    print(f"\nTest cases: {len(test_cases)}")

    # Results tracking
    results: List[PredictionResult] = []

    # Baseline metrics (from NOTES.md)
    print("\n" + "-" * 60)
    print("BASELINE (from original experiment)")
    print("-" * 60)
    print("  Steps accuracy: 0% (always predicts 2)")
    print("  F1: 71%")

    # Scratchpad approach
    print("\n" + "-" * 60)
    print("SCRATCHPAD APPROACH (BFS expansion)")
    print("-" * 60)

    reachable_correct = 0
    steps_correct = 0
    steps_within_one = 0

    # TP, FP, TN, FN for F1
    tp = fp = tn = fn = 0

    for case in test_cases:
        case_name = f"{case.start}→{case.target}"

        # Create prompt
        prompt = create_bfs_scratchpad_prompt(case)

        # Generate
        output = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
        )

        # Parse
        pred_reachable, pred_steps = parse_steps_response(output)

        # Check reachability
        reachable_match = (pred_reachable == case.expected_reachable)
        if reachable_match:
            reachable_correct += 1

        # Update confusion matrix
        if case.expected_reachable and pred_reachable:
            tp += 1
        elif case.expected_reachable and not pred_reachable:
            fn += 1
        elif not case.expected_reachable and pred_reachable:
            fp += 1
        else:
            tn += 1

        # Check steps
        steps_match = (pred_steps == case.expected_steps)
        if steps_match:
            steps_correct += 1

        within_one = (
            abs(pred_steps - case.expected_steps) <= 1
            if pred_steps >= 0 and case.expected_steps >= 0
            else (pred_steps == case.expected_steps)
        )
        if within_one:
            steps_within_one += 1

        result = PredictionResult(
            case_name=case_name,
            expected_reachable=case.expected_reachable,
            expected_steps=case.expected_steps,
            predicted_reachable=pred_reachable,
            predicted_steps=pred_steps,
            reachable_correct=reachable_match,
            steps_correct=steps_match,
            steps_within_one=within_one,
            raw_output=output,
        )
        results.append(result)

        status = "✓" if steps_match else "✗"
        reach_status = "✓" if reachable_match else "✗"
        print(f"  {status} {case_name}: expected={case.expected_steps}, predicted={pred_steps} (reach: {reach_status})")

    # Compute metrics
    n = len(test_cases)
    reachability_acc = reachable_correct / n * 100
    steps_acc = steps_correct / n * 100
    within_one_acc = steps_within_one / n * 100

    # F1 calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Reachability accuracy: {reachability_acc:.0f}% ({reachable_correct}/{n})")
    print(f"Steps accuracy:        {steps_acc:.0f}% ({steps_correct}/{n})")
    print(f"Steps within ±1:       {within_one_acc:.0f}% ({steps_within_one}/{n})")
    print(f"\nF1 Score:   {f1*100:.0f}%")
    print(f"Precision:  {precision*100:.0f}%")
    print(f"Recall:     {recall*100:.0f}%")
    print(f"\nConfusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Format report
    print(f"\nREACHABILITY_STEPS baseline=0%, scratchpad={steps_acc:.0f}%, f1_maintained={f1*100:.0f}%")

    return {
        'reachability_accuracy': reachability_acc,
        'steps_accuracy': steps_acc,
        'steps_within_one': within_one_acc,
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'results': results,
    }


if __name__ == "__main__":
    result = run_benchmark()
