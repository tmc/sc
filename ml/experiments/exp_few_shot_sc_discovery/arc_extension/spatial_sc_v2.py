#!/usr/bin/env python3
"""
Spatial Statechart Discovery v2

Extended with more rule types:
1. Color mapping (cell-independent)
2. Row/Col uniformity
3. Neighbor-based rules (cellular automata style)
4. Majority color rules
5. Border/interior rules
"""

import json
import sys
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class SpatialExampleV2:
    """Extended spatial example with neighbor info."""
    color: int
    output_color: int
    position_type: str  # corner, edge, center
    row_uniform: bool
    col_uniform: bool
    neighbor_colors: Tuple[int, ...]  # 4-neighbors (N,E,S,W) or -1 for boundary
    neighbor_same_count: int  # how many neighbors have same color
    is_surrounded: bool  # all 4 neighbors exist and are same color


def get_neighbors(grid: List[List[int]], row: int, col: int) -> Tuple[int, ...]:
    """Get 4-neighbor colors (-1 for boundary)."""
    nrows, ncols = len(grid), len(grid[0])
    neighbors = []
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N E S W
        nr, nc = row + dr, col + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            neighbors.append(grid[nr][nc])
        else:
            neighbors.append(-1)  # boundary
    return tuple(neighbors)


def get_position_type(row: int, col: int, nrows: int, ncols: int) -> str:
    """Classify position."""
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


def is_row_uniform(grid: List[List[int]], row: int) -> bool:
    return len(set(grid[row])) == 1


def is_col_uniform(grid: List[List[int]], col: int) -> bool:
    return len(set(grid[r][col] for r in range(len(grid)))) == 1


def extract_examples_v2(task_data: Dict) -> List[SpatialExampleV2]:
    """Extract extended spatial examples."""
    examples = []

    for train_ex in task_data['train']:
        inp = train_ex['input']
        out = train_ex['output']

        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            continue

        nrows, ncols = len(inp), len(inp[0])

        for r in range(nrows):
            for c in range(ncols):
                in_color = inp[r][c]
                out_color = out[r][c]

                if in_color == out_color:
                    continue

                neighbors = get_neighbors(inp, r, c)
                neighbor_same = sum(1 for n in neighbors if n == in_color)
                is_surrounded = all(n == in_color or n == -1 for n in neighbors) and neighbor_same >= 3

                examples.append(SpatialExampleV2(
                    color=in_color,
                    output_color=out_color,
                    position_type=get_position_type(r, c, nrows, ncols),
                    row_uniform=is_row_uniform(inp, r),
                    col_uniform=is_col_uniform(inp, c),
                    neighbor_colors=neighbors,
                    neighbor_same_count=neighbor_same,
                    is_surrounded=is_surrounded,
                ))

    return examples


def discover_rules_v2(examples: List[SpatialExampleV2]) -> Dict:
    """Discover rules with extended contexts."""
    if not examples:
        return {"type": "unknown"}

    rules = {}

    # Group by contexts
    by_color = {}
    by_surrounded = {"surrounded": [], "not_surrounded": []}
    by_neighbor_count = {0: [], 1: [], 2: [], 3: [], 4: []}

    for ex in examples:
        if ex.color not in by_color:
            by_color[ex.color] = []
        by_color[ex.color].append(ex)

        if ex.is_surrounded:
            by_surrounded["surrounded"].append(ex)
        else:
            by_surrounded["not_surrounded"].append(ex)

        by_neighbor_count[ex.neighbor_same_count].append(ex)

    # Rule 1: Simple color mapping
    color_consistent = all(
        len(set(ex.output_color for ex in exs)) == 1
        for exs in by_color.values()
    )
    if color_consistent:
        rules["type"] = "color_mapping"
        for color, exs in by_color.items():
            rules[f"c{color}"] = exs[0].output_color
        return rules

    # Rule 2: Surrounded vs not (fill interior)
    if by_surrounded["surrounded"] and by_surrounded["not_surrounded"]:
        surr_outputs = set(ex.output_color for ex in by_surrounded["surrounded"])
        not_surr_outputs = set(ex.output_color for ex in by_surrounded["not_surrounded"])

        if len(surr_outputs) == 1 and len(not_surr_outputs) == 1:
            rules["type"] = "surrounded"
            rules["surrounded"] = list(surr_outputs)[0]
            rules["not_surrounded"] = list(not_surr_outputs)[0]
            return rules

    # Rule 3: Neighbor count based
    # Check if output depends on number of same-color neighbors
    neighbor_consistent = {}
    all_consistent = True
    for count, exs in by_neighbor_count.items():
        if exs:
            outputs = set(ex.output_color for ex in exs)
            if len(outputs) == 1:
                neighbor_consistent[count] = list(outputs)[0]
            else:
                all_consistent = False
                break

    if all_consistent and len(neighbor_consistent) > 1:
        rules["type"] = "neighbor_count"
        rules["by_count"] = neighbor_consistent
        return rules

    # Rule 4: Color + position
    by_color_pos = {}
    for ex in examples:
        key = (ex.color, ex.position_type)
        if key not in by_color_pos:
            by_color_pos[key] = []
        by_color_pos[key].append(ex)

    color_pos_consistent = all(
        len(set(ex.output_color for ex in exs)) == 1
        for exs in by_color_pos.values()
    )
    if color_pos_consistent:
        rules["type"] = "color_position"
        for (color, pos), exs in by_color_pos.items():
            rules[f"c{color}_{pos}"] = exs[0].output_color
        return rules

    # Rule 5: Specific neighbor pattern
    # Group by exact neighbor pattern
    by_neighbor_pattern = {}
    for ex in examples:
        key = (ex.color, ex.neighbor_colors)
        if key not in by_neighbor_pattern:
            by_neighbor_pattern[key] = []
        by_neighbor_pattern[key].append(ex)

    # Check if limited number of patterns
    if len(by_neighbor_pattern) <= 20:
        pattern_consistent = all(
            len(set(ex.output_color for ex in exs)) == 1
            for exs in by_neighbor_pattern.values()
        )
        if pattern_consistent:
            rules["type"] = "neighbor_pattern"
            rules["patterns"] = {
                f"c{color}_n{neighbors}": exs[0].output_color
                for (color, neighbors), exs in by_neighbor_pattern.items()
            }
            return rules

    return {"type": "unknown", "examples": len(examples)}


def apply_rules_v2(rules: Dict, grid: List[List[int]]) -> List[List[int]]:
    """Apply v2 rules to a grid."""
    nrows, ncols = len(grid), len(grid[0])
    output = [[grid[r][c] for c in range(ncols)] for r in range(nrows)]

    rule_type = rules.get("type", "unknown")

    for r in range(nrows):
        for c in range(ncols):
            in_color = grid[r][c]

            if rule_type == "color_mapping":
                if f"c{in_color}" in rules:
                    output[r][c] = rules[f"c{in_color}"]

            elif rule_type == "surrounded":
                neighbors = get_neighbors(grid, r, c)
                neighbor_same = sum(1 for n in neighbors if n == in_color)
                is_surr = all(n == in_color or n == -1 for n in neighbors) and neighbor_same >= 3
                if is_surr:
                    output[r][c] = rules.get("surrounded", in_color)
                else:
                    output[r][c] = rules.get("not_surrounded", in_color)

            elif rule_type == "neighbor_count":
                neighbors = get_neighbors(grid, r, c)
                count = sum(1 for n in neighbors if n == in_color)
                by_count = rules.get("by_count", {})
                if count in by_count:
                    output[r][c] = by_count[count]

            elif rule_type == "color_position":
                pos = get_position_type(r, c, nrows, ncols)
                key = f"c{in_color}_{pos}"
                if key in rules:
                    output[r][c] = rules[key]

            elif rule_type == "neighbor_pattern":
                neighbors = get_neighbors(grid, r, c)
                key = f"c{in_color}_n{neighbors}"
                patterns = rules.get("patterns", {})
                if key in patterns:
                    output[r][c] = patterns[key]

    return output


def test_task_v2(task_path: str) -> Dict:
    """Test v2 rules on a task."""
    with open(task_path) as f:
        task_data = json.load(f)

    task_id = task_path.split("/")[-1].replace(".json", "")

    # Check dimensions
    for ex in task_data['train'] + task_data['test']:
        if len(ex['input']) != len(ex['output']):
            return {"task_id": task_id, "status": "skipped", "reason": "dim_mismatch"}
        if ex['input'] and len(ex['input'][0]) != len(ex['output'][0]):
            return {"task_id": task_id, "status": "skipped", "reason": "dim_mismatch"}

    # Extract and discover
    examples = extract_examples_v2(task_data)
    if not examples:
        # Check identity
        is_identity = all(ex['input'] == ex['output'] for ex in task_data['train'])
        if is_identity:
            return {"task_id": task_id, "status": "perfect", "rule_type": "identity", "accuracy": 100}
        return {"task_id": task_id, "status": "skipped", "reason": "no_changes"}

    rules = discover_rules_v2(examples)

    if rules["type"] == "unknown":
        return {"task_id": task_id, "status": "failed", "rule_type": "unknown", "accuracy": 0}

    # Test
    correct = 0
    total = 0
    for test_ex in task_data['test']:
        predicted = apply_rules_v2(rules, test_ex['input'])
        expected = test_ex['output']
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                total += 1
                if predicted[r][c] == expected[r][c]:
                    correct += 1

    accuracy = correct / total * 100 if total > 0 else 0
    status = "perfect" if accuracy == 100 else "partial" if accuracy >= 80 else "failed"

    return {
        "task_id": task_id,
        "status": status,
        "rule_type": rules["type"],
        "accuracy": accuracy,
        "num_examples": len(examples),
    }


if __name__ == "__main__":
    import os
    from collections import Counter

    ARC_DIR = "/Users/tmc/go/src/github.com/tmc/arc/docs/repos/fchollet/ARC-AGI/data/training"

    print("=" * 70)
    print("SPATIAL SC v2 FULL BENCHMARK")
    print("=" * 70)

    results = []
    task_files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith(".json")])

    for i, task_file in enumerate(task_files):
        result = test_task_v2(os.path.join(ARC_DIR, task_file))
        results.append(result)
        if (i + 1) % 100 == 0:
            perfect = sum(1 for r in results if r.get("status") == "perfect")
            print(f"  [{i+1}/400] Perfect: {perfect}")

    # Summary
    by_status = Counter(r.get("status") for r in results)
    perfect = [r for r in results if r.get("status") == "perfect"]

    print(f"\nBy status:")
    for s in ["perfect", "partial", "failed", "skipped"]:
        print(f"  {s}: {by_status.get(s, 0)}")

    print(f"\nPerfect by rule type:")
    by_rule = Counter(r.get("rule_type") for r in perfect)
    for rule, count in by_rule.most_common():
        print(f"  {rule}: {count}")

    print(f"\n[DDB5]: ARC_V2 perfect={len(perfect)}/400 ({len(perfect)/4:.1f}%)")
