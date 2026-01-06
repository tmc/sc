#!/usr/bin/env python3
"""
Benchmark: Context Guards Comprehensive.

Tests LLM's ability to:
1. Evaluate boolean guards
2. Handle numeric comparisons
3. Process compound (AND/OR/NOT) expressions
4. Trace guard-action chains
5. Resolve transition priority
6. Handle in-state guards
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from .context_guards import (
    GuardCategory,
    GuardTestCase,
    create_guard_eval_prompt,
    parse_state_response,
    simulate_machine,
    get_test_cases,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    max_tokens: int = 400


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    category: GuardCategory
    description: str
    expected_state: str
    predicted_state: str
    correct: bool
    raw_output: str


def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run context guards comprehensive benchmark."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("CONTEXT GUARDS COMPREHENSIVE BENCHMARK")
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

    # Per-category metrics
    category_correct = {c: 0 for c in GuardCategory}
    category_total = {c: 0 for c in GuardCategory}

    print("\n" + "-" * 60)
    print("RUNNING TESTS")
    print("-" * 60)

    for case in test_cases:
        category_total[case.category] += 1

        # Verify expected result with ground truth simulation
        state, _ = simulate_machine(
            case.transitions,
            case.transitions[0].source,
            case.initial_context,
            case.event_sequence,
            case.active_states,
        )
        if state != case.expected_final_state:
            print(f"  WARNING: Ground truth mismatch for {case.description}")
            print(f"    Expected: {case.expected_final_state}, Simulated: {state}")

        # Create prompt
        prompt = create_guard_eval_prompt(case)

        # Generate
        output = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
        )

        # Parse response
        predicted = parse_state_response(output)

        # Evaluate
        correct = predicted.lower() == case.expected_final_state.lower()
        if correct:
            category_correct[case.category] += 1

        result = PredictionResult(
            category=case.category,
            description=case.description,
            expected_state=case.expected_final_state,
            predicted_state=predicted,
            correct=correct,
            raw_output=output,
        )
        results.append(result)

        status = "✓" if correct else "✗"
        print(f"  {status} [{case.category.name}] {case.description}")
        print(f"      Expected: {case.expected_final_state}, Predicted: {predicted}")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)

    category_accuracies = {}
    for cat in GuardCategory:
        if category_total[cat] > 0:
            acc = category_correct[cat] / category_total[cat] * 100
            category_accuracies[cat.name] = acc
            print(f"  {cat.name}: {acc:.0f}% ({category_correct[cat]}/{category_total[cat]})")

    total_correct = sum(category_correct.values())
    total = sum(category_total.values())
    overall_acc = total_correct / total * 100 if total > 0 else 0

    print(f"\nOverall accuracy: {overall_acc:.0f}% ({total_correct}/{total})")

    # Simplified categories for report
    bool_acc = category_accuracies.get("BOOLEAN", 0)
    compare_acc = category_accuracies.get("COMPARISON", 0)
    compound_acc = category_accuracies.get("COMPOUND", 0)
    chain_acc = category_accuracies.get("CHAIN", 0)
    priority_acc = category_accuracies.get("PRIORITY", 0)

    report = f"CONTEXT_GUARDS bool={bool_acc:.0f}%, compare={compare_acc:.0f}%, compound={compound_acc:.0f}%, chain={chain_acc:.0f}%, priority={priority_acc:.0f}%, overall={overall_acc:.0f}%"
    print(f"\n{report}")

    return {
        'overall_accuracy': overall_acc,
        'category_accuracies': category_accuracies,
        'results': results,
        'report': report,
    }


if __name__ == "__main__":
    result = run_benchmark()
