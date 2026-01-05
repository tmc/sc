#!/usr/bin/env python3
"""
Validity Tester: Measure unconstrained SC generation validity.

Tests whether the model can generate valid statechart JSON
without grammar constraints.
"""

import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate


@dataclass
class ValidityResult:
    """Result of validity test."""
    prompt: str
    generated: str
    is_valid_json: bool
    is_valid_structure: bool
    error: Optional[str] = None


@dataclass
class TestResults:
    """Aggregate test results."""
    total: int
    valid_json: int
    valid_structure: int
    json_validity_rate: float
    structure_validity_rate: float
    avg_gen_time: float
    results: List[ValidityResult]


# Test prompts for unconstrained generation
TEST_PROMPTS = [
    "Generate a valid statechart JSON for a toggle switch:",
    "Create a statechart for a traffic light system:",
    "Write the JSON for a door state machine:",
    "Define a statechart for a media player:",
    "Generate JSON for an on/off switch:",
    "Create a state machine for loading states:",
    "Write a statechart for user authentication:",
    "Generate a valid SC for a simple counter:",
    "Create JSON for a connection state machine:",
    "Define a statechart for a game pause system:",
]


def validate_json(text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Check if text is valid JSON."""
    try:
        # Try to extract JSON from response
        # Look for JSON block
        start = text.find('{')
        if start == -1:
            return False, None, "No JSON object found"

        # Find matching closing brace
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = text[start:end]
        data = json.loads(json_str)
        return True, data, None

    except json.JSONDecodeError as e:
        return False, None, str(e)


def validate_structure(data: Dict) -> Tuple[bool, Optional[str]]:
    """Check if JSON has valid statechart structure."""
    try:
        # Must have root_state
        if "root_state" not in data:
            return False, "Missing root_state"

        root = data["root_state"]

        # root_state must have label, type, children
        if "label" not in root:
            return False, "root_state missing label"
        if "type" not in root:
            return False, "root_state missing type"
        if "children" not in root:
            return False, "root_state missing children"

        # Check children structure
        children = root.get("children", [])
        if not isinstance(children, list):
            return False, "children must be array"
        if len(children) == 0:
            return False, "root_state has no children"

        # At least one child must have is_initial or be first (default initial)
        has_initial = False
        for child in children:
            if not isinstance(child, dict):
                return False, "child must be object"
            if "label" not in child:
                return False, "child missing label"
            if "type" not in child:
                return False, "child missing type"
            if child.get("is_initial"):
                has_initial = True

        # Check transitions if present
        transitions = data.get("transitions", [])
        if not isinstance(transitions, list):
            return False, "transitions must be array"

        for trans in transitions:
            if not isinstance(trans, dict):
                return False, "transition must be object"
            if "from" not in trans or "to" not in trans:
                return False, "transition missing from/to"
            if not isinstance(trans.get("from"), list):
                return False, "transition from must be array"
            if not isinstance(trans.get("to"), list):
                return False, "transition to must be array"

        return True, None

    except Exception as e:
        return False, str(e)


def test_single_generation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.3,
) -> ValidityResult:
    """Test a single unconstrained generation."""
    # Add instruction format
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Generate without constraints
    output = generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )

    # Validate JSON
    is_valid_json, data, json_error = validate_json(output)

    # Validate structure if JSON is valid
    is_valid_structure = False
    struct_error = None
    if is_valid_json and data:
        is_valid_structure, struct_error = validate_structure(data)

    error = json_error or struct_error

    return ValidityResult(
        prompt=prompt,
        generated=output,
        is_valid_json=is_valid_json,
        is_valid_structure=is_valid_structure,
        error=error,
    )


def test_validity(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    prompts: List[str] = None,
    max_tokens: int = 300,
    model=None,
    tokenizer=None,
) -> TestResults:
    """
    Test unconstrained generation validity.

    Args:
        model_id: Model to test
        prompts: Test prompts (default: TEST_PROMPTS)
        max_tokens: Max tokens per generation
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        TestResults with validity metrics
    """
    if prompts is None:
        prompts = TEST_PROMPTS

    # Load model if not provided
    if model is None:
        print(f"Loading model: {model_id}")
        model, tokenizer = load(model_id)
        print("Model loaded.")

    print(f"\nTesting {len(prompts)} prompts...")

    results = []
    total_time = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] Testing...")

        t0 = time.time()
        result = test_single_generation(model, tokenizer, prompt, max_tokens)
        total_time += time.time() - t0

        status = "VALID" if result.is_valid_structure else ("JSON_OK" if result.is_valid_json else "INVALID")
        print(f"    {status}: {result.prompt[:40]}...")

        results.append(result)

    # Calculate metrics
    valid_json = sum(1 for r in results if r.is_valid_json)
    valid_structure = sum(1 for r in results if r.is_valid_structure)
    total = len(results)

    return TestResults(
        total=total,
        valid_json=valid_json,
        valid_structure=valid_structure,
        json_validity_rate=valid_json / total * 100 if total > 0 else 0,
        structure_validity_rate=valid_structure / total * 100 if total > 0 else 0,
        avg_gen_time=total_time / total if total > 0 else 0,
        results=results,
    )


def print_results(results: TestResults):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("VALIDITY TEST RESULTS")
    print("=" * 60)
    print(f"Total prompts: {results.total}")
    print(f"Valid JSON: {results.valid_json}/{results.total} ({results.json_validity_rate:.0f}%)")
    print(f"Valid Structure: {results.valid_structure}/{results.total} ({results.structure_validity_rate:.0f}%)")
    print(f"Avg generation time: {results.avg_gen_time:.2f}s")

    print("\n" + "-" * 60)
    print("Per-prompt results:")
    print("-" * 60)

    for r in results.results:
        status = "VALID" if r.is_valid_structure else ("JSON" if r.is_valid_json else "FAIL")
        print(f"  [{status}] {r.prompt[:50]}...")
        if r.error:
            print(f"         Error: {r.error[:60]}...")


def demo():
    """Demo the validity tester."""
    print("=" * 60)
    print("Validity Tester Demo")
    print("=" * 60)

    # Test with first 5 prompts
    results = test_validity(prompts=TEST_PROMPTS[:5])
    print_results(results)


if __name__ == "__main__":
    demo()
