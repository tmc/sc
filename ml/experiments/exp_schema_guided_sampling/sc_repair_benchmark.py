#!/usr/bin/env python3
"""
SC Repair Benchmark - Real MLX Model Testing

Tests repairing broken statecharts with various bugs:
- Wrong transition targets (non-existent states)
- Missing required fields
- Invalid state references
- Structural issues

This tests model UNDERSTANDING, not just completion.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate


@dataclass
class RepairResult:
    """Result of repairing a broken SC."""
    bug_type: str
    broken_sc: str
    repaired_sc: str
    valid_json: bool
    valid_structure: bool
    bug_fixed: bool
    gen_time_s: float


# Bug types and broken statecharts
BROKEN_STATECHARTS = [
    # 1. Wrong transition target - "OnState" doesn't exist, should be "On"
    {
        "bug_type": "wrong_target",
        "broken": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["OnState"], "event": "TOGGLE"}  # BUG: OnState doesn't exist
            ]
        },
        "fix_hint": "Transition target 'OnState' does not exist. Valid states are: Off, On",
        "expected_fix": "Change 'OnState' to 'On'"
    },

    # 2. Missing 'to' field in transition
    {
        "bug_type": "missing_field",
        "broken": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Active", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "event": "START"}  # BUG: missing 'to' field
            ]
        },
        "fix_hint": "Transition is missing 'to' field",
        "expected_fix": "Add 'to': ['Active']"
    },

    # 3. Wrong source state - "Closed" doesn't exist, should be "Close"
    {
        "bug_type": "wrong_source",
        "broken": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Open", "type": 1, "is_initial": True},
                    {"label": "Close", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"}  # BUG: Closed doesn't exist
            ]
        },
        "fix_hint": "Transition source 'Closed' does not exist. Valid states are: Open, Close",
        "expected_fix": "Change 'Closed' to 'Close'"
    },

    # 4. No initial state marked
    {
        "bug_type": "no_initial",
        "broken": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1},  # BUG: no is_initial
                    {"label": "Green", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"}
            ]
        },
        "fix_hint": "No initial state marked. One child must have is_initial: true",
        "expected_fix": "Add is_initial: true to Red"
    },

    # 5. Invalid state type
    {
        "bug_type": "invalid_type",
        "broken": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Start", "type": 99, "is_initial": True},  # BUG: type 99 invalid
                    {"label": "End", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Start"], "to": ["End"], "event": "FINISH"}
            ]
        },
        "fix_hint": "State type 99 is invalid. Valid types: 1 (basic), 2 (normal), 3 (parallel)",
        "expected_fix": "Change type 99 to type 1"
    },
]


def validate_statechart(sc: Dict) -> Tuple[bool, List[str]]:
    """Validate statechart and return list of issues."""
    issues = []

    if "root_state" not in sc:
        issues.append("Missing root_state")
        return False, issues

    root = sc["root_state"]

    # Collect all state labels
    all_labels = set()
    has_initial = False

    def check_state(state: Dict, path: str = ""):
        nonlocal has_initial
        label = state.get("label", "")
        all_labels.add(label)

        # Check type
        state_type = state.get("type", 0)
        if state_type not in [0, 1, 2, 3]:
            issues.append(f"Invalid state type {state_type} at {path}{label}")

        # Check initial
        if state.get("is_initial"):
            has_initial = True

        for child in state.get("children", []):
            check_state(child, f"{path}{label}.")

    check_state(root)

    if not has_initial and root.get("children"):
        issues.append("No initial state marked")

    # Check transitions
    for i, trans in enumerate(sc.get("transitions", [])):
        if "from" not in trans:
            issues.append(f"Transition {i}: missing 'from' field")
        elif isinstance(trans["from"], list):
            for src in trans["from"]:
                if src not in all_labels:
                    issues.append(f"Transition {i}: source '{src}' not found")

        if "to" not in trans:
            issues.append(f"Transition {i}: missing 'to' field")
        elif isinstance(trans["to"], list):
            for tgt in trans["to"]:
                if tgt not in all_labels:
                    issues.append(f"Transition {i}: target '{tgt}' not found")

    return len(issues) == 0, issues


def repair_statechart(
    model,
    tokenizer,
    broken: Dict,
    fix_hint: str,
    max_tokens: int = 300,
) -> RepairResult:
    """Use model to repair a broken statechart."""

    broken_json = json.dumps(broken, indent=2)

    # Construct repair prompt
    prompt = f"""The following statechart JSON has a bug:

```json
{broken_json}
```

Issue: {fix_hint}

Please output the CORRECTED statechart JSON only, with the bug fixed:
```json
"""

    t0 = time.time()

    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )

    gen_time = time.time() - t0

    # Extract JSON from output
    repaired_json = None
    try:
        # Try to find JSON in output
        output_clean = output.strip()

        # Remove markdown code blocks if present
        if "```" in output_clean:
            parts = output_clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    output_clean = part
                    break

        # Find JSON object
        start = output_clean.find("{")
        if start >= 0:
            # Find matching closing brace
            depth = 0
            end = start
            for i in range(start, len(output_clean)):
                if output_clean[i] == "{":
                    depth += 1
                elif output_clean[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            output_clean = output_clean[start:end]

        repaired = json.loads(output_clean)
        valid_json = True
    except json.JSONDecodeError:
        repaired = None
        valid_json = False
        repaired_json = output[:200]

    # Validate repaired SC
    if repaired:
        valid_structure, issues = validate_statechart(repaired)
        repaired_json = json.dumps(repaired)
    else:
        valid_structure = False
        issues = ["Could not parse JSON"]

    # Check if original bug was fixed
    _, original_issues = validate_statechart(broken)
    bug_fixed = valid_structure and len(issues) < len(original_issues)

    return RepairResult(
        bug_type=fix_hint[:30],
        broken_sc=broken_json[:100] + "...",
        repaired_sc=repaired_json[:200] if repaired_json else output[:200],
        valid_json=valid_json,
        valid_structure=valid_structure,
        bug_fixed=bug_fixed,
        gen_time_s=gen_time,
    )


def run_benchmark(model_id: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit") -> Dict:
    """Run SC repair benchmark."""
    print("=" * 70)
    print("SC REPAIR BENCHMARK - Real MLX Model")
    print("=" * 70)
    print(f"Model: {model_id}")

    print("\nLoading model...")
    model, tokenizer = load(model_id)
    print("Model loaded.\n")

    results = []

    for i, test in enumerate(BROKEN_STATECHARTS, 1):
        print(f"\n{i}. Bug type: {test['bug_type']}")
        print(f"   Hint: {test['fix_hint'][:60]}...")

        result = repair_statechart(
            model=model,
            tokenizer=tokenizer,
            broken=test["broken"],
            fix_hint=test["fix_hint"],
            max_tokens=300,
        )

        status = "FIXED" if result.bug_fixed else "FAIL"
        print(f"   Status: {status}")
        print(f"   Valid JSON: {result.valid_json}")
        print(f"   Valid Structure: {result.valid_structure}")
        print(f"   Bug Fixed: {result.bug_fixed}")
        print(f"   Time: {result.gen_time_s:.2f}s")

        results.append({
            "bug_type": test["bug_type"],
            "valid_json": result.valid_json,
            "valid_structure": result.valid_structure,
            "bug_fixed": result.bug_fixed,
        })

    # Summary
    fixed_count = sum(1 for r in results if r["bug_fixed"])
    valid_json_count = sum(1 for r in results if r["valid_json"])
    accuracy = fixed_count / len(results) * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Valid JSON: {valid_json_count}/{len(results)}")
    print(f"Bugs Fixed: {fixed_count}/{len(results)}")
    print(f"Repair Accuracy: {accuracy:.0f}%")

    report = f"SC_REPAIR accuracy={accuracy:.0f}%"
    print(f"\nReport: {report}")

    return {
        "accuracy": accuracy,
        "fixed_count": fixed_count,
        "valid_json_count": valid_json_count,
        "total": len(results),
        "results": results,
        "report": report,
    }


if __name__ == "__main__":
    run_benchmark()
