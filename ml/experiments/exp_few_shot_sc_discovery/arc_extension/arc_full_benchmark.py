#!/usr/bin/env python3
"""
Full ARC-AGI Benchmark using Spatial SC Discovery

Tests ALL 400 training tasks to see how many can be solved
with statechart-based rule discovery.
"""

import json
import os
import sys
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_few_shot_sc_discovery.spatial_sc import (
    extract_spatial_examples,
    discover_spatial_rules,
    apply_spatial_rules,
)


ARC_DIR = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training"


@dataclass
class TaskResult:
    task_id: str
    status: str  # "perfect", "partial", "failed", "skipped"
    accuracy: float
    rule_type: str
    reason: str
    num_examples: int


def test_task(task_path: str) -> TaskResult:
    """Test spatial SC on a single task."""
    with open(task_path) as f:
        task_data = json.load(f)

    task_id = task_path.split("/")[-1].replace(".json", "")

    # Check dimensions match between input/output
    valid = True
    for ex in task_data['train'] + task_data['test']:
        inp, out = ex['input'], ex['output']
        if len(inp) != len(out):
            valid = False
            break
        if inp and out and len(inp[0]) != len(out[0]):
            valid = False
            break

    if not valid:
        return TaskResult(
            task_id=task_id,
            status="skipped",
            accuracy=0,
            rule_type="N/A",
            reason="dimension_mismatch",
            num_examples=0,
        )

    # Extract spatial examples
    examples = extract_spatial_examples(task_data)
    if not examples:
        # Check if it's an identity transform (no changes)
        is_identity = True
        for ex in task_data['train']:
            if ex['input'] != ex['output']:
                is_identity = False
                break

        if is_identity:
            return TaskResult(
                task_id=task_id,
                status="perfect",
                accuracy=100,
                rule_type="identity",
                reason="no_changes_needed",
                num_examples=0,
            )

        return TaskResult(
            task_id=task_id,
            status="skipped",
            accuracy=0,
            rule_type="N/A",
            reason="no_changing_cells",
            num_examples=0,
        )

    # Discover rules
    rules = discover_spatial_rules(examples)

    if rules["type"] == "unknown":
        return TaskResult(
            task_id=task_id,
            status="failed",
            accuracy=0,
            rule_type="unknown",
            reason="no_consistent_rule",
            num_examples=len(examples),
        )

    # Test on test cases
    correct = 0
    total = 0

    for test_ex in task_data['test']:
        inp = test_ex['input']
        expected = test_ex['output']
        predicted = apply_spatial_rules(rules, inp)

        for r in range(len(inp)):
            for c in range(len(inp[0])):
                total += 1
                if predicted[r][c] == expected[r][c]:
                    correct += 1

    accuracy = correct / total * 100 if total > 0 else 0

    if accuracy == 100:
        status = "perfect"
    elif accuracy >= 80:
        status = "partial"
    else:
        status = "failed"

    return TaskResult(
        task_id=task_id,
        status=status,
        accuracy=accuracy,
        rule_type=rules["type"],
        reason="tested",
        num_examples=len(examples),
    )


def run_full_benchmark():
    """Run benchmark on ALL ARC training tasks."""
    print("\n" + "=" * 70)
    print("FULL ARC-AGI SPATIAL SC BENCHMARK (400 tasks)")
    print("=" * 70)

    results: List[TaskResult] = []

    task_files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith(".json")])
    total_tasks = len(task_files)

    print(f"\nProcessing {total_tasks} tasks...")

    for i, task_file in enumerate(task_files):
        task_path = os.path.join(ARC_DIR, task_file)
        result = test_task(task_path)
        results.append(result)

        # Progress indicator
        if (i + 1) % 50 == 0:
            perfect_so_far = sum(1 for r in results if r.status == "perfect")
            print(f"  [{i+1}/{total_tasks}] Perfect so far: {perfect_so_far}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # By status
    by_status = Counter(r.status for r in results)
    print(f"\nBy status:")
    for status in ["perfect", "partial", "failed", "skipped"]:
        count = by_status.get(status, 0)
        pct = count / total_tasks * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    # By rule type (for non-skipped)
    tested = [r for r in results if r.status != "skipped"]
    by_rule = Counter(r.rule_type for r in tested)
    print(f"\nBy rule type (tested tasks):")
    for rule_type, count in by_rule.most_common():
        perfect = sum(1 for r in tested if r.rule_type == rule_type and r.status == "perfect")
        print(f"  {rule_type}: {count} tasks, {perfect} perfect")

    # Perfect tasks breakdown
    perfect = [r for r in results if r.status == "perfect"]
    print(f"\nPerfect tasks ({len(perfect)}):")
    by_perfect_rule = Counter(r.rule_type for r in perfect)
    for rule_type, count in by_perfect_rule.most_common():
        print(f"  {rule_type}: {count}")

    # Skip reasons
    skipped = [r for r in results if r.status == "skipped"]
    by_skip_reason = Counter(r.reason for r in skipped)
    print(f"\nSkip reasons ({len(skipped)}):")
    for reason, count in by_skip_reason.most_common():
        print(f"  {reason}: {count}")

    # Show some perfect task IDs
    print(f"\nSample perfect tasks:")
    for r in perfect[:10]:
        print(f"  {r.task_id}: {r.rule_type}")

    # Failed tasks with high example counts (might be discoverable with more rules)
    failed = [r for r in results if r.status == "failed" and r.num_examples > 10]
    if failed:
        print(f"\nFailed with many examples (potential for new rules):")
        for r in sorted(failed, key=lambda x: -x.num_examples)[:5]:
            print(f"  {r.task_id}: {r.num_examples} examples, {r.accuracy:.0f}%")

    # Overall metrics
    total_tested = len(tested)
    total_perfect = len(perfect)
    total_partial = sum(1 for r in results if r.status == "partial")

    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    print(f"Total tasks: {total_tasks}")
    print(f"Tested (same dimensions): {total_tested} ({total_tested/total_tasks*100:.1f}%)")
    print(f"Perfect (100%): {total_perfect} ({total_perfect/total_tasks*100:.1f}%)")
    print(f"High (>=80%): {total_perfect + total_partial} ({(total_perfect+total_partial)/total_tasks*100:.1f}%)")

    if tested:
        avg_accuracy = sum(r.accuracy for r in tested) / len(tested)
        print(f"Avg accuracy (tested): {avg_accuracy:.1f}%")

    return results


if __name__ == "__main__":
    results = run_full_benchmark()

    # Report
    perfect = sum(1 for r in results if r.status == "perfect")
    tested = sum(1 for r in results if r.status != "skipped")
    total = len(results)

    print(f"\n[DDB5]: ARC_FULL_BENCHMARK total={total}, tested={tested}, perfect={perfect} ({perfect/total*100:.1f}%)")
