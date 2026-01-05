#!/usr/bin/env python3
"""
Guard Outcome Predictor.

Given a set of guard expressions and context values, predict which guard fires.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import IntEnum


class GuardLevel(IntEnum):
    """Guard complexity levels."""
    L1_SIMPLE = 1      # x > 5
    L2_COMPOUND = 2    # x > 5 && y < 10
    L3_ARITHMETIC = 3  # (x + y) > threshold
    L4_COMPLEX = 4     # items.length > 0 && items[0].valid


@dataclass
class GuardSet:
    """A set of competing guards."""
    guards: List[str]
    level: GuardLevel
    variables: List[str]


def generate_l1_guards(num_guards: int = 3) -> GuardSet:
    """Generate L1 guards: simple comparisons with fixed thresholds."""
    vars_pool = ['x', 'y', 'count', 'value', 'score', 'level']
    var = random.choice(vars_pool)

    # Use fixed thresholds for reliable testing
    guards = [
        f"{var} < 5",
        f"{var} >= 5 && {var} < 15",
        f"{var} >= 15",
    ]

    return GuardSet(guards=guards, level=GuardLevel.L1_SIMPLE, variables=[var])


def generate_l2_guards(num_guards: int = 3) -> GuardSet:
    """Generate L2 guards: compound comparisons."""
    var1, var2 = random.sample(['x', 'y', 'a', 'b', 'score', 'health'], 2)

    guards = [
        f"{var1} > 10 && {var2} > 10",
        f"{var1} > 10 && {var2} <= 10",
        f"{var1} <= 10",
    ]

    return GuardSet(guards=guards, level=GuardLevel.L2_COMPOUND, variables=[var1, var2])


def generate_l3_guards(num_guards: int = 3) -> GuardSet:
    """Generate L3 guards: arithmetic comparisons with fixed threshold."""
    var1, var2 = random.sample(['x', 'y', 'a', 'b'], 2)

    # Fixed thresholds: 20 and 40
    guards = [
        f"({var1} + {var2}) < 20",
        f"({var1} + {var2}) >= 20 && ({var1} + {var2}) < 40",
        f"({var1} + {var2}) >= 40",
    ]

    return GuardSet(guards=guards, level=GuardLevel.L3_ARITHMETIC, variables=[var1, var2])


def generate_l4_guards(num_guards: int = 3) -> GuardSet:
    """Generate L4 guards: list/property access (simplified to Python syntax)."""
    # Note: Using Python-compatible syntax for eval
    guards = [
        "len(items) == 0",
        "len(items) > 0 and not items[0]['valid']",
        "len(items) > 0 and items[0]['valid']",
    ]

    return GuardSet(guards=guards, level=GuardLevel.L4_COMPLEX, variables=['items'])


def generate_guard_set(level: GuardLevel, num_guards: int = 3) -> GuardSet:
    """Generate a guard set at the specified complexity level."""
    generators = {
        GuardLevel.L1_SIMPLE: generate_l1_guards,
        GuardLevel.L2_COMPOUND: generate_l2_guards,
        GuardLevel.L3_ARITHMETIC: generate_l3_guards,
        GuardLevel.L4_COMPLEX: generate_l4_guards,
    }
    return generators[level](num_guards)


def generate_context_for_guards(guard_set: GuardSet, target_index: int = -1) -> Dict[str, Any]:
    """
    Generate context values that make a specific guard fire.

    Args:
        guard_set: The set of guards
        target_index: Which guard should fire (-1 for none, though this is rare)

    Returns:
        Context dict that makes the target guard (and only that guard) fire
    """
    level = guard_set.level

    if level == GuardLevel.L1_SIMPLE:
        var = guard_set.variables[0]
        # L1 guards: x < 5, x >= 5 && x < 15, x >= 15
        if target_index == 0:
            value = random.randint(0, 4)      # x < 5
        elif target_index == 1:
            value = random.randint(5, 14)     # x >= 5 && x < 15
        else:
            value = random.randint(15, 25)    # x >= 15
        return {var: value}

    elif level == GuardLevel.L2_COMPOUND:
        var1, var2 = guard_set.variables
        if target_index == 0:
            # var1 > 10 && var2 > 10
            return {var1: random.randint(11, 20), var2: random.randint(11, 20)}
        elif target_index == 1:
            # var1 > 10 && var2 <= 10
            return {var1: random.randint(11, 20), var2: random.randint(0, 10)}
        else:
            # var1 <= 10
            return {var1: random.randint(0, 10), var2: random.randint(0, 20)}

    elif level == GuardLevel.L3_ARITHMETIC:
        var1, var2 = guard_set.variables
        # L3 guards: sum < 20, sum >= 20 && sum < 40, sum >= 40
        if target_index == 0:
            # Sum < 20
            return {var1: random.randint(0, 9), var2: random.randint(0, 9)}
        elif target_index == 1:
            # Sum >= 20 && sum < 40
            return {var1: random.randint(10, 19), var2: random.randint(10, 19)}
        else:
            # Sum >= 40
            return {var1: random.randint(20, 30), var2: random.randint(20, 30)}

    elif level == GuardLevel.L4_COMPLEX:
        if target_index == 0:
            # Empty list
            return {'items': []}
        elif target_index == 1:
            # Non-empty, first item invalid
            return {'items': [{'valid': False, 'value': 1}]}
        else:
            # Non-empty, first item valid
            return {'items': [{'valid': True, 'value': 42}]}

    return {}


def eval_guard_safe(guard: str, context: Dict[str, Any]) -> bool:
    """Safely evaluate a guard expression."""
    try:
        # Replace && and || with Python operators
        py_guard = guard.replace("&&", " and ").replace("||", " or ")
        result = eval(py_guard, {"__builtins__": {"len": len}}, context)
        return bool(result)
    except Exception:
        return False


def compute_expected_outcome(guard_set: GuardSet, context: Dict[str, Any]) -> int:
    """
    Compute which guard fires given the context.

    Returns:
        Index of firing guard (0..N-1) or -1 if none fire
    """
    for i, guard in enumerate(guard_set.guards):
        if eval_guard_safe(guard, context):
            return i
    return -1


def create_guard_prediction_prompt(guard_set: GuardSet, context: Dict[str, Any]) -> str:
    """Create a prompt for guard prediction."""
    guards_str = "\n".join(f"  [{i}] {g}" for i, g in enumerate(guard_set.guards))
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())

    prompt = f"""Given these guard expressions and context values, determine which guard evaluates to TRUE.

Guards:
{guards_str}

Context: {context_str}

Rules:
- Guards are evaluated in order (0, 1, 2, ...)
- Return the INDEX of the FIRST guard that evaluates to TRUE
- If no guard evaluates to TRUE, return -1

Think step by step:
1. Substitute context values into each guard
2. Evaluate each expression
3. Return the index of the first TRUE guard

Answer (just the number):"""

    return prompt


def parse_prediction(output: str) -> int:
    """Parse model output to get predicted guard index."""
    # Look for a number in the output
    import re

    # Clean output
    output = output.strip()

    # Try to find a standalone number
    numbers = re.findall(r'-?\d+', output)
    if numbers:
        # Return the first number found
        return int(numbers[0])

    return -1


def create_guard_prediction_prompt_few_shot(guard_set: GuardSet, context: Dict[str, Any]) -> str:
    """Create a few-shot prompt for guard prediction."""
    # Include examples for all three outcomes (0, 1, 2) to avoid bias
    examples = """Determine which guard evaluates to TRUE first. Return ONLY the index number.

Example 1:
Guards: [0] x < 5, [1] x >= 5 && x < 15, [2] x >= 15
Context: x=3
Answer: 0

Example 2:
Guards: [0] x < 5, [1] x >= 5 && x < 15, [2] x >= 15
Context: x=10
Answer: 1

Example 3:
Guards: [0] x < 5, [1] x >= 5 && x < 15, [2] x >= 15
Context: x=20
Answer: 2

Example 4:
Guards: [0] a > 10 && b > 10, [1] a > 10 && b <= 10, [2] a <= 10
Context: a=15, b=15
Answer: 0

Example 5:
Guards: [0] a > 10 && b > 10, [1] a > 10 && b <= 10, [2] a <= 10
Context: a=15, b=5
Answer: 1

Example 6:
Guards: [0] a > 10 && b > 10, [1] a > 10 && b <= 10, [2] a <= 10
Context: a=5, b=15
Answer: 2

"""
    guards_str = ", ".join(f"[{i}] {g}" for i, g in enumerate(guard_set.guards))
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())

    prompt = f"""{examples}Now solve:
Guards: {guards_str}
Context: {context_str}
Answer:"""

    return prompt


# Test
if __name__ == "__main__":
    print("Guard Predictor Test")
    print("=" * 60)

    for level in GuardLevel:
        print(f"\n{level.name}:")
        guard_set = generate_guard_set(level)

        for target_idx in range(len(guard_set.guards)):
            context = generate_context_for_guards(guard_set, target_idx)
            actual = compute_expected_outcome(guard_set, context)

            print(f"  Target: {target_idx}, Context: {context}")
            print(f"  Guards: {guard_set.guards}")
            print(f"  Actual fires: {actual} {'✓' if actual == target_idx else '✗'}")
