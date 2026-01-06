#!/usr/bin/env python3
"""
Spatial SC Discovery Benchmark on ARC-AGI

Tests how well spatial statechart rules handle ARC patterns.
"""

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_few_shot_sc_discovery.spatial_sc import (
    extract_spatial_examples,
    discover_spatial_rules,
    apply_spatial_rules,
)


ARC_DIR = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training"


def test_task(task_path: str) -> Dict:
    """Test spatial SC on a single task."""
    with open(task_path) as f:
        task_data = json.load(f)

    task_id = task_path.split("/")[-1].replace(".json", "")

    # Check dimensions match
    valid = True
    for ex in task_data['train'] + task_data['test']:
        inp, out = ex['input'], ex['output']
        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            valid = False
            break

    if not valid:
        return {"task_id": task_id, "applicable": False, "reason": "dimension_mismatch"}

    # Extract and discover
    examples = extract_spatial_examples(task_data)
    if not examples:
        return {"task_id": task_id, "applicable": False, "reason": "no_changes"}

    rules = discover_spatial_rules(examples)

    if rules["type"] == "unknown":
        return {"task_id": task_id, "applicable": False, "reason": "no_rule_found"}

    # Test
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

    return {
        "task_id": task_id,
        "applicable": True,
        "rule_type": rules["type"],
        "accuracy": accuracy,
        "num_examples": len(examples),
    }


def run_benchmark():
    """Test on various small ARC tasks."""
    # Small tasks (3x3 to 5x5)
    test_tasks = [
        # Previously tested
        "0d3d703e.json",  # color mapping (was 100%)
        "25d8a9c8.json",  # row uniformity (was 44%, now should be 100%)
        # New spatial tasks
        "25ff71a9.json",
        "3c9b0459.json",
        "5582e5ca.json",
        "44f52bb0.json",
        "27a28665.json",
        "b1948b0a.json",
        "c8f0f002.json",
        "d511f180.json",
    ]

    print("\n" + "=" * 70)
    print("SPATIAL SC DISCOVERY BENCHMARK")
    print("=" * 70)

    results = []
    for task_file in test_tasks:
        task_path = os.path.join(ARC_DIR, task_file)
        if not os.path.exists(task_path):
            continue

        result = test_task(task_path)
        results.append(result)

        if result.get("applicable"):
            status = "OK" if result["accuracy"] >= 80 else "FAIL"
            print(f"  [{status}] {result['task_id']}: {result['accuracy']:.0f}% ({result['rule_type']}, {result['num_examples']} ex)")
        else:
            print(f"  [SKIP] {result['task_id']}: {result.get('reason', 'N/A')}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    applicable = [r for r in results if r.get("applicable")]
    if applicable:
        avg = sum(r["accuracy"] for r in applicable) / len(applicable)
        perfect = sum(1 for r in applicable if r["accuracy"] == 100)
        high = sum(1 for r in applicable if r["accuracy"] >= 80)

        print(f"\nApplicable tasks: {len(applicable)}/{len(results)}")
        print(f"Perfect (100%): {perfect}")
        print(f"High (>=80%): {high}")
        print(f"Average accuracy: {avg:.0f}%")

        # By rule type
        by_type = {}
        for r in applicable:
            rt = r["rule_type"]
            if rt not in by_type:
                by_type[rt] = []
            by_type[rt].append(r["accuracy"])

        print("\nBy rule type:")
        for rt, accs in by_type.items():
            avg_type = sum(accs) / len(accs)
            print(f"  {rt}: {len(accs)} tasks, {avg_type:.0f}% avg")

        return avg, perfect, len(applicable)

    return 0, 0, 0


if __name__ == "__main__":
    avg, perfect, total = run_benchmark()
    print(f"\n[DDB5]: SPATIAL_SC tasks={total}, perfect={perfect}, avg_acc={avg:.0f}%")
