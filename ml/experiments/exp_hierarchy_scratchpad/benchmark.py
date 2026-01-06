#!/usr/bin/env python3
"""
Benchmark: Hierarchy Scratchpad.

Tests scratchpad approach for hierarchy reasoning across 4 task types:
- Containment: Is X inside Y?
- Cascade exit: If X exits, what else exits?
- Default entry: Entering X, what's the configuration?
- Depth: How deep is state X?
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set

from .hierarchy_scratchpad import (
    HierarchyCase,
    HierarchyTask,
    create_prompt,
    parse_response,
    get_test_cases,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    max_tokens: int = 300


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    task: HierarchyTask
    query: str
    expected: Any
    predicted: Any
    correct: bool
    raw_output: str


def evaluate_result(case: HierarchyCase, predicted: Any) -> bool:
    """Evaluate if prediction is correct."""
    if case.task == HierarchyTask.CONTAINMENT:
        return predicted == case.expected
    elif case.task in (HierarchyTask.CASCADE_EXIT, HierarchyTask.DEFAULT_ENTRY):
        # Set comparison
        if isinstance(predicted, set) and isinstance(case.expected, set):
            return predicted == case.expected
        return False
    elif case.task == HierarchyTask.DEPTH:
        return predicted == case.expected
    return False


def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """Run hierarchy scratchpad benchmark."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("HIERARCHY SCRATCHPAD BENCHMARK")
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

    # Per-task metrics
    task_correct = {t: 0 for t in HierarchyTask}
    task_total = {t: 0 for t in HierarchyTask}

    print("\n" + "-" * 60)
    print("SCRATCHPAD APPROACH")
    print("-" * 60)

    for case in test_cases:
        task_total[case.task] += 1

        # Create prompt
        prompt = create_prompt(case)

        # Generate
        output = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
        )

        # Parse response
        predicted = parse_response(case, output)

        # Evaluate
        correct = evaluate_result(case, predicted)
        if correct:
            task_correct[case.task] += 1

        # Format query description
        if case.task == HierarchyTask.CONTAINMENT:
            query_desc = f"{case.query_state} inside {case.target_state}?"
        elif case.task == HierarchyTask.CASCADE_EXIT:
            query_desc = f"exit {case.query_state}"
        elif case.task == HierarchyTask.DEFAULT_ENTRY:
            query_desc = f"enter {case.query_state}"
        else:
            query_desc = f"depth({case.query_state})"

        result = PredictionResult(
            task=case.task,
            query=query_desc,
            expected=case.expected,
            predicted=predicted,
            correct=correct,
            raw_output=output,
        )
        results.append(result)

        status = "✓" if correct else "✗"
        print(f"  {status} [{case.task.name}] {query_desc}: expected={case.expected}, predicted={predicted}")

    # Compute metrics
    total_correct = sum(task_correct.values())
    total = sum(task_total.values())
    overall_acc = total_correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nPer-task accuracy:")
    task_accuracies = {}
    for task in HierarchyTask:
        if task_total[task] > 0:
            acc = task_correct[task] / task_total[task] * 100
            task_accuracies[task.name] = acc
            print(f"  {task.name}: {acc:.0f}% ({task_correct[task]}/{task_total[task]})")

    print(f"\nOverall accuracy: {overall_acc:.0f}% ({total_correct}/{total})")

    # Baseline comparison
    print("\n" + "-" * 60)
    print("BASELINE vs SCRATCHPAD")
    print("-" * 60)
    print(f"  Baseline (from prior exp): 27%")
    print(f"  Scratchpad:                {overall_acc:.0f}%")
    improvement = overall_acc - 27
    print(f"  Improvement:               {improvement:+.0f}%")

    # Format report
    contain_acc = task_accuracies.get("CONTAINMENT", 0)
    cascade_acc = task_accuracies.get("CASCADE_EXIT", 0)
    entry_acc = task_accuracies.get("DEFAULT_ENTRY", 0)
    depth_acc = task_accuracies.get("DEPTH", 0)

    report = f"HIERARCHY_SCRATCHPAD baseline=27%, scratchpad={overall_acc:.0f}%, by_task=[contain:{contain_acc:.0f}%, cascade:{cascade_acc:.0f}%, entry:{entry_acc:.0f}%, depth:{depth_acc:.0f}%]"
    print(f"\n{report}")

    return {
        'overall_accuracy': overall_acc,
        'task_accuracies': task_accuracies,
        'results': results,
        'report': report,
    }


if __name__ == "__main__":
    result = run_benchmark()
