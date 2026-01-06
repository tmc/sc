#!/usr/bin/env python3
"""
Spatial Statechart Discovery

Extends SC discovery to handle spatial/positional ARC patterns by:
1. Encoding position/neighbor context in state
2. Using guards for spatial conditions
3. Discovering LOCAL rules (like cellular automata)

Key insight: Spatial patterns can be modeled as:
  (color, context) → new_color

Where context includes:
  - Position (row, col, corner, edge, center)
  - Neighbors (adjacent colors)
  - Global pattern (uniform row, surrounded, etc.)
"""

import json
import sys
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class SpatialExample:
    """I/O example with spatial context."""
    color: int
    position_type: str  # corner, edge, center
    neighbor_pattern: str  # encoded neighbor colors
    row_uniform: bool  # is entire row same color?
    col_uniform: bool  # is entire col same color?
    output_color: int


def get_position_type(row: int, col: int, nrows: int, ncols: int) -> str:
    """Classify position as corner/edge/center."""
    is_top = row == 0
    is_bottom = row == nrows - 1
    is_left = col == 0
    is_right = col == ncols - 1

    if (is_top or is_bottom) and (is_left or is_right):
        return "corner"
    elif is_top or is_bottom or is_left or is_right:
        return "edge"
    else:
        return "center"


def get_neighbor_pattern(grid: List[List[int]], row: int, col: int) -> str:
    """Encode 4-neighbor pattern (NESW)."""
    nrows, ncols = len(grid), len(grid[0])
    neighbors = []

    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N E S W
        nr, nc = row + dr, col + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            neighbors.append(str(grid[nr][nc]))
        else:
            neighbors.append("X")  # boundary

    return "".join(neighbors)


def is_row_uniform(grid: List[List[int]], row: int) -> bool:
    """Check if row has all same values."""
    return len(set(grid[row])) == 1


def is_col_uniform(grid: List[List[int]], col: int) -> bool:
    """Check if column has all same values."""
    return len(set(grid[r][col] for r in range(len(grid)))) == 1


def extract_spatial_examples(task_data: Dict) -> List[SpatialExample]:
    """Extract spatial I/O examples from ARC task."""
    examples = []

    for train_ex in task_data['train']:
        inp = train_ex['input']
        out = train_ex['output']

        # Must have same dimensions
        if len(inp) != len(out):
            continue
        if not inp or len(inp[0]) != len(out[0]):
            continue

        nrows, ncols = len(inp), len(inp[0])

        for r in range(nrows):
            for c in range(ncols):
                in_color = inp[r][c]
                out_color = out[r][c]

                # Only care about cells that change
                if in_color == out_color:
                    continue

                examples.append(SpatialExample(
                    color=in_color,
                    position_type=get_position_type(r, c, nrows, ncols),
                    neighbor_pattern=get_neighbor_pattern(inp, r, c),
                    row_uniform=is_row_uniform(inp, r),
                    col_uniform=is_col_uniform(inp, c),
                    output_color=out_color,
                ))

    return examples


def discover_spatial_rules(examples: List[SpatialExample]) -> Dict:
    """
    Discover spatial transformation rules.

    Priority order:
    1. Simple color mapping (most common)
    2. Row uniformity
    3. Column uniformity
    4. Position-based

    Returns rules like:
      - "color_mapping": color → color (cell-independent)
      - "row_uniformity": uniform row → X, non-uniform → Y
      - "position": corner → X, edge → Y, center → Z
    """
    rules = {}

    # Group by various contexts
    by_color = {}
    by_row_uniform = {"uniform": [], "non_uniform": []}
    by_col_uniform = {"uniform": [], "non_uniform": []}
    by_position = {"corner": [], "edge": [], "center": []}

    for ex in examples:
        if ex.color not in by_color:
            by_color[ex.color] = []
        by_color[ex.color].append(ex)

        if ex.row_uniform:
            by_row_uniform["uniform"].append(ex)
        else:
            by_row_uniform["non_uniform"].append(ex)

        if ex.col_uniform:
            by_col_uniform["uniform"].append(ex)
        else:
            by_col_uniform["non_uniform"].append(ex)

        by_position[ex.position_type].append(ex)

    # Rule 1: Simple color mapping (FIRST priority - most common)
    color_consistent = True
    for color, color_examples in by_color.items():
        outputs = set(ex.output_color for ex in color_examples)
        if len(outputs) > 1:
            color_consistent = False
            break

    if color_consistent and by_color:
        rules["type"] = "color_mapping"
        for color, color_examples in by_color.items():
            if color_examples:
                rules[f"c{color}"] = color_examples[0].output_color
        return rules

    # Rule 2: Row uniformity
    if by_row_uniform["uniform"] and by_row_uniform["non_uniform"]:
        uniform_outputs = set(ex.output_color for ex in by_row_uniform["uniform"])
        non_uniform_outputs = set(ex.output_color for ex in by_row_uniform["non_uniform"])

        if len(uniform_outputs) == 1 and len(non_uniform_outputs) == 1:
            rules["type"] = "row_uniformity"
            rules["uniform_row"] = list(uniform_outputs)[0]
            rules["non_uniform_row"] = list(non_uniform_outputs)[0]
            return rules

    # Rule 3: Column uniformity
    if by_col_uniform["uniform"] and by_col_uniform["non_uniform"]:
        uniform_outputs = set(ex.output_color for ex in by_col_uniform["uniform"])
        non_uniform_outputs = set(ex.output_color for ex in by_col_uniform["non_uniform"])

        if len(uniform_outputs) == 1 and len(non_uniform_outputs) == 1:
            rules["type"] = "col_uniformity"
            rules["uniform_col"] = list(uniform_outputs)[0]
            rules["non_uniform_col"] = list(non_uniform_outputs)[0]
            return rules

    # Rule 4: Position-based
    position_consistent = True
    for pos_type, pos_examples in by_position.items():
        if pos_examples:
            outputs = set(ex.output_color for ex in pos_examples)
            if len(outputs) > 1:
                position_consistent = False
                break

    if position_consistent and any(by_position.values()):
        rules["type"] = "position"
        for pos_type, pos_examples in by_position.items():
            if pos_examples:
                rules[pos_type] = pos_examples[0].output_color
        return rules

    # No consistent rule found
    rules["type"] = "unknown"
    rules["examples"] = len(examples)

    return rules


def apply_spatial_rules(rules: Dict, grid: List[List[int]]) -> List[List[int]]:
    """Apply discovered rules to a grid."""
    nrows, ncols = len(grid), len(grid[0])
    output = [[0] * ncols for _ in range(nrows)]

    rule_type = rules.get("type", "unknown")

    for r in range(nrows):
        for c in range(ncols):
            in_color = grid[r][c]

            if rule_type == "color_mapping":
                output[r][c] = rules.get(f"c{in_color}", in_color)

            elif rule_type == "row_uniformity":
                if is_row_uniform(grid, r):
                    output[r][c] = rules.get("uniform_row", in_color)
                else:
                    output[r][c] = rules.get("non_uniform_row", in_color)

            elif rule_type == "col_uniformity":
                if is_col_uniform(grid, c):
                    output[r][c] = rules.get("uniform_col", in_color)
                else:
                    output[r][c] = rules.get("non_uniform_col", in_color)

            elif rule_type == "position":
                pos_type = get_position_type(r, c, nrows, ncols)
                output[r][c] = rules.get(pos_type, in_color)

            else:
                output[r][c] = in_color

    return output


def test_spatial_sc(task_path: str) -> Dict:
    """Test spatial SC discovery on an ARC task."""
    with open(task_path) as f:
        task_data = json.load(f)

    task_id = task_path.split("/")[-1].replace(".json", "")

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print("="*60)

    # Extract spatial examples
    examples = extract_spatial_examples(task_data)
    print(f"\nExtracted {len(examples)} spatial examples")

    if not examples:
        print("No changing cells found")
        return {"task_id": task_id, "applicable": False}

    # Show sample examples
    for ex in examples[:3]:
        print(f"  c{ex.color} @ {ex.position_type}, row_uni={ex.row_uniform} → c{ex.output_color}")

    # Discover rules
    rules = discover_spatial_rules(examples)
    print(f"\nDiscovered rule type: {rules['type']}")
    print(f"Rules: {rules}")

    # Test on test cases
    correct_cells = 0
    total_cells = 0

    for test_ex in task_data['test']:
        inp = test_ex['input']
        expected = test_ex['output']

        if len(inp) != len(expected):
            continue

        predicted = apply_spatial_rules(rules, inp)

        for r in range(len(inp)):
            for c in range(len(inp[0])):
                total_cells += 1
                if predicted[r][c] == expected[r][c]:
                    correct_cells += 1

    accuracy = correct_cells / total_cells * 100 if total_cells > 0 else 0
    print(f"\nTest accuracy: {correct_cells}/{total_cells} ({accuracy:.0f}%)")

    return {
        "task_id": task_id,
        "applicable": True,
        "rule_type": rules["type"],
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    ARC_DIR = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training"

    # Test on tasks that failed with simple color mapping
    test_tasks = [
        "25d8a9c8.json",  # Row uniformity task (was 44%)
        "0d3d703e.json",  # Color mapping (was 100%)
    ]

    results = []
    for task_file in test_tasks:
        task_path = f"{ARC_DIR}/{task_file}"
        result = test_spatial_sc(task_path)
        results.append(result)

    print("\n" + "=" * 60)
    print("SPATIAL SC SUMMARY")
    print("=" * 60)

    for r in results:
        if r.get("applicable"):
            print(f"  {r['task_id']}: {r['accuracy']:.0f}% ({r['rule_type']})")
        else:
            print(f"  {r['task_id']}: N/A")
