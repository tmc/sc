#!/usr/bin/env python3
"""
ARC-AGI Benchmark using SC Discovery

Tests TRUE color mapping tasks where SC discovery applies.
"""

import json
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_few_shot_sc_discovery.sc_discoverer import (
    SCDiscoverer,
    IOExample,
)
from experiments.exp_few_shot_sc_discovery.predictor import SCPredictor
from experiments.exp_few_shot_sc_discovery.arc_adapter import (
    ARCTask,
    load_arc_task,
    extract_color_mapping,
)


# TRUE color mapping tasks (cell-independent)
COLOR_MAPPING_TASKS = [
    "0d3d703e.json",  # 4 train, 8 mappings
    "aabf363d.json",  # 2 train, 5 mappings
    "b1948b0a.json",  # 3 train, 2 mappings
    "c8f0f002.json",  # 3 train, 3 mappings
    "d511f180.json",  # 3 train, 9 mappings
]

ARC_DATA_DIR = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training"


@dataclass
class TaskResult:
    task_id: str
    applicable: bool
    num_train_mappings: int
    test_accuracy: float
    details: str


def test_arc_task(task_path: str, model=None, tokenizer=None) -> TaskResult:
    """Test SC discovery on single ARC task."""
    task = load_arc_task(task_path)

    # Extract color mappings from training
    examples = extract_color_mapping(task)

    if not examples:
        return TaskResult(
            task_id=task.task_id,
            applicable=False,
            num_train_mappings=0,
            test_accuracy=0,
            details="No mappings extracted",
        )

    # Discover SC from training examples
    discoverer = SCDiscoverer(model, tokenizer)
    discovered = discoverer.discover(examples)

    # Test on test cases
    predictor = SCPredictor(model, tokenizer)

    total_correct = 0
    total_cells = 0

    for test_case in task.test:
        inp = test_case["input"]
        out = test_case["output"]

        # Check dimension compatibility
        if len(inp) != len(out):
            continue
        if inp and out and len(inp[0]) != len(out[0]):
            continue

        for i in range(len(inp)):
            for j in range(len(inp[0])):
                in_color = inp[i][j]
                expected_color = out[i][j]

                if in_color == expected_color:
                    # No change needed - still count as correct
                    total_correct += 1
                    total_cells += 1
                    continue

                total_cells += 1

                # Predict using discovered SC
                result = predictor.predict(
                    discovered,
                    "TRANSFORM",
                    f"c{in_color}",
                    expected_state=f"c{expected_color}",
                )

                if result.correct:
                    total_correct += 1

    accuracy = total_correct / total_cells * 100 if total_cells > 0 else 0

    return TaskResult(
        task_id=task.task_id,
        applicable=True,
        num_train_mappings=len(examples),
        test_accuracy=accuracy,
        details=f"States: {len(discovered.states)}, Trans: {len(discovered.transitions)}",
    )


def run_arc_benchmark(model=None, tokenizer=None) -> List[TaskResult]:
    """Run benchmark on TRUE color mapping ARC tasks."""
    results = []

    print("\n" + "=" * 70)
    print("ARC-AGI SC DISCOVERY BENCHMARK (Color Mapping Tasks Only)")
    print("=" * 70)
    print(f"\nTesting {len(COLOR_MAPPING_TASKS)} true color mapping tasks")

    for task_file in COLOR_MAPPING_TASKS:
        task_path = os.path.join(ARC_DATA_DIR, task_file)

        if not os.path.exists(task_path):
            print(f"\n[SKIP] {task_file} - not found")
            continue

        print(f"\n--- {task_file} ---")

        result = test_arc_task(task_path, model, tokenizer)
        results.append(result)

        if result.applicable:
            status = "OK" if result.test_accuracy >= 80 else "FAIL"
            print(f"  [{status}] Mappings: {result.num_train_mappings}, Accuracy: {result.test_accuracy:.0f}%")
            print(f"  {result.details}")
        else:
            print(f"  {result.details}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    applicable = [r for r in results if r.applicable]
    if applicable:
        avg_accuracy = sum(r.test_accuracy for r in applicable) / len(applicable)
        perfect = sum(1 for r in applicable if r.test_accuracy == 100)

        print(f"\nTasks tested: {len(applicable)}")
        print(f"Perfect (100%): {perfect}/{len(applicable)}")
        print(f"Average accuracy: {avg_accuracy:.0f}%")

        print("\nPer-task:")
        for r in applicable:
            status = "OK" if r.test_accuracy == 100 else "PARTIAL" if r.test_accuracy >= 80 else "FAIL"
            print(f"  [{status}] {r.task_id}: {r.test_accuracy:.0f}%")

    return results


if __name__ == "__main__":
    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-1.5B-Instruct-4bit...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None

    results = run_arc_benchmark(model, tokenizer)

    # Report
    applicable = [r for r in results if r.applicable]
    if applicable:
        avg = sum(r.test_accuracy for r in applicable) / len(applicable)
        perfect = sum(1 for r in applicable if r.test_accuracy == 100)
        print(f"\n[DDB5]: ARC_COLOR_MAPPING tasks={len(applicable)}, perfect={perfect}, avg_acc={avg:.0f}%")
