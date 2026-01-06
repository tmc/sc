#!/usr/bin/env python3
"""
ARC-AGI to Statechart Adapter

Frames ARC tasks as statechart discovery problems:
- Color mapping: each color is a state, TRANSFORM event triggers change
- Grid as context: position can influence transition

Tests if SC discovery generalizes to ARC-style pattern recognition.
"""

import json
import sys
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_few_shot_sc_discovery.sc_discoverer import (
    SCDiscoverer,
    IOExample,
)
from experiments.exp_few_shot_sc_discovery.predictor import SCPredictor


@dataclass
class ARCTask:
    """An ARC task with train/test examples."""
    task_id: str
    train: List[Dict]  # [{"input": [[]], "output": [[]]}]
    test: List[Dict]


def load_arc_task(path: str) -> ARCTask:
    """Load an ARC task from JSON."""
    with open(path) as f:
        data = json.load(f)
    task_id = path.split("/")[-1].replace(".json", "")
    return ARCTask(task_id=task_id, train=data["train"], test=data["test"])


def extract_color_mapping(task: ARCTask) -> List[IOExample]:
    """
    Extract color mappings from ARC task as I/O examples.

    Assumes: Each cell transforms independently (color mapping task).
    Returns: List of (TRANSFORM, input_color) -> output_color examples
    """
    mappings: Dict[int, int] = {}

    for example in task.train:
        inp = example["input"]
        out = example["output"]

        # If same dimensions, extract cell-by-cell mapping
        if len(inp) == len(out) and len(inp[0]) == len(out[0]):
            for i in range(len(inp)):
                for j in range(len(inp[0])):
                    in_color = inp[i][j]
                    out_color = out[i][j]
                    if in_color != out_color:
                        mappings[in_color] = out_color

    # Convert to IOExamples
    examples = []
    for in_color, out_color in mappings.items():
        examples.append(IOExample(
            event="TRANSFORM",
            current_state=f"c{in_color}",
            next_state=f"c{out_color}",
        ))

    return examples


def test_color_mapping_task(
    task: ARCTask,
    model=None,
    tokenizer=None,
) -> Dict:
    """Test SC discovery on a color mapping ARC task."""
    print(f"\n{'='*60}")
    print(f"ARC Task: {task.task_id}")
    print("="*60)

    # Extract training examples
    examples = extract_color_mapping(task)

    if not examples:
        print("  No color mappings found (not a simple mapping task)")
        return {"task_id": task.task_id, "applicable": False}

    print(f"\nExtracted {len(examples)} color mappings:")
    for ex in examples:
        print(f"  {ex}")

    # Phase 1: Discover SC
    discoverer = SCDiscoverer(model, tokenizer)
    discovered = discoverer.discover(examples)

    print(f"\nDiscovered SC:")
    print(f"  States: {sorted(discovered.states)}")
    print(f"  Pattern: {discovered.pattern}")
    print(f"  Transitions: {len(discovered.transitions)}")

    # Phase 2: Test on held-out test case
    predictor = SCPredictor(model, tokenizer)

    test_results = []
    for test_case in task.test:
        inp = test_case["input"]
        out = test_case["output"]

        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            continue

        correct = 0
        total = 0

        for i in range(len(inp)):
            for j in range(len(inp[0])):
                in_color = inp[i][j]
                expected_color = out[i][j]

                if in_color == expected_color:
                    continue  # No transformation needed

                total += 1

                # Predict using discovered SC
                result = predictor.predict(
                    discovered,
                    "TRANSFORM",
                    f"c{in_color}",
                    expected_state=f"c{expected_color}",
                )

                if result.correct:
                    correct += 1
                else:
                    print(f"  FAIL: c{in_color} -> {result.predicted_state} (expected c{expected_color})")

        if total > 0:
            accuracy = correct / total * 100
            test_results.append({"correct": correct, "total": total, "accuracy": accuracy})
            print(f"\nTest accuracy: {correct}/{total} ({accuracy:.0f}%)")

    return {
        "task_id": task.task_id,
        "applicable": True,
        "num_mappings": len(examples),
        "test_results": test_results,
    }


def find_color_mapping_tasks(data_dir: str) -> List[str]:
    """Find ARC tasks that are likely color mappings."""
    import os

    candidates = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(data_dir, filename)
        try:
            task = load_arc_task(path)

            # Check if input/output have same dimensions (cell-wise transform)
            valid = True
            for ex in task.train:
                inp, out = ex["input"], ex["output"]
                if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                    valid = False
                    break

            if valid and task.train:
                # Check for actual color changes
                has_changes = False
                for ex in task.train:
                    inp, out = ex["input"], ex["output"]
                    for i in range(len(inp)):
                        for j in range(len(inp[0])):
                            if inp[i][j] != out[i][j]:
                                has_changes = True
                                break

                if has_changes:
                    candidates.append(path)
        except Exception:
            pass

    return candidates


if __name__ == "__main__":
    # Test on 0d3d703e (color swap task)
    task_path = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training/0d3d703e.json"

    try:
        from mlx_lm import load
        print("Loading model...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None

    task = load_arc_task(task_path)
    result = test_color_mapping_task(task, model, tokenizer)

    print(f"\n{'='*60}")
    print("RESULT:", result)
