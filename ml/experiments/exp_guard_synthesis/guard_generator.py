#!/usr/bin/env python3
"""
Guard Expression Generator for Statecharts.

Generates guards at 4 complexity levels:
- L1: Single variable (is_ready)
- L2: Boolean operators (a && b, a || b)
- L3: Comparisons (x > 5, count < max)
- L4: Nested expressions ((a || b) && c)
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import IntEnum


class GuardLevel(IntEnum):
    """Guard complexity levels."""
    L1_SINGLE = 1
    L2_BOOLEAN = 2
    L3_COMPARISON = 3
    L4_NESTED = 4


@dataclass
class GuardTemplate:
    """Template for generating guards."""
    level: GuardLevel
    pattern: str  # Pattern with placeholders
    variables: List[str]  # Variable names used
    description: str


# Variable name pools
BOOL_VARS = [
    "is_ready", "is_enabled", "is_active", "is_locked", "is_connected",
    "is_authenticated", "is_valid", "is_complete", "is_open", "is_closed",
    "has_permission", "has_key", "has_data", "can_proceed", "should_retry",
]

COUNT_VARS = [
    "count", "retry_count", "attempt_count", "error_count", "item_count",
    "level", "score", "health", "energy", "time_elapsed",
]

THRESHOLD_VARS = [
    "max_retries", "max_attempts", "threshold", "limit", "max_count",
    "min_level", "timeout", "capacity", "buffer_size",
]


def generate_l1_guard() -> Tuple[str, List[str], str]:
    """Generate L1: Single variable guard."""
    var = random.choice(BOOL_VARS)
    # Optionally negate
    if random.random() < 0.3:
        return f"!{var}", [var], f"NOT {var}"
    return var, [var], var


def generate_l2_guard() -> Tuple[str, List[str], str]:
    """Generate L2: Boolean operator guard."""
    var1 = random.choice(BOOL_VARS)
    var2 = random.choice([v for v in BOOL_VARS if v != var1])

    op = random.choice(["&&", "||"])
    op_name = "AND" if op == "&&" else "OR"

    # Sometimes add negation to one operand
    if random.random() < 0.2:
        return f"!{var1} {op} {var2}", [var1, var2], f"NOT {var1} {op_name} {var2}"

    return f"{var1} {op} {var2}", [var1, var2], f"{var1} {op_name} {var2}"


def generate_l3_guard() -> Tuple[str, List[str], str]:
    """Generate L3: Comparison guard."""
    count_var = random.choice(COUNT_VARS)

    pattern_type = random.choice(["literal", "variable"])

    if pattern_type == "literal":
        # Compare to literal value
        value = random.choice([0, 1, 3, 5, 10, 100])
        op = random.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"{count_var} {op} {value}", [count_var], f"{count_var} {op} {value}"
    else:
        # Compare to another variable
        threshold = random.choice(THRESHOLD_VARS)
        op = random.choice(["<", "<=", ">", ">="])
        return f"{count_var} {op} {threshold}", [count_var, threshold], f"{count_var} {op} {threshold}"


def generate_l4_guard() -> Tuple[str, List[str], str]:
    """Generate L4: Nested expression guard."""
    patterns = [
        # (A || B) && C
        lambda: _make_nested_or_and(),
        # (A && B) || C
        lambda: _make_nested_and_or(),
        # !A && (B || C)
        lambda: _make_nested_not_and_or(),
        # (A < X) && (B || C)
        lambda: _make_comparison_and_bool(),
    ]
    return random.choice(patterns)()


def _make_nested_or_and() -> Tuple[str, List[str], str]:
    """(A || B) && C pattern."""
    vars = random.sample(BOOL_VARS, 3)
    expr = f"({vars[0]} || {vars[1]}) && {vars[2]}"
    desc = f"({vars[0]} OR {vars[1]}) AND {vars[2]}"
    return expr, vars, desc


def _make_nested_and_or() -> Tuple[str, List[str], str]:
    """(A && B) || C pattern."""
    vars = random.sample(BOOL_VARS, 3)
    expr = f"({vars[0]} && {vars[1]}) || {vars[2]}"
    desc = f"({vars[0]} AND {vars[1]}) OR {vars[2]}"
    return expr, vars, desc


def _make_nested_not_and_or() -> Tuple[str, List[str], str]:
    """!A && (B || C) pattern."""
    vars = random.sample(BOOL_VARS, 3)
    expr = f"!{vars[0]} && ({vars[1]} || {vars[2]})"
    desc = f"NOT {vars[0]} AND ({vars[1]} OR {vars[2]})"
    return expr, vars, desc


def _make_comparison_and_bool() -> Tuple[str, List[str], str]:
    """(count < X) && (A || B) pattern."""
    count_var = random.choice(COUNT_VARS)
    value = random.choice([3, 5, 10])
    bool_vars = random.sample(BOOL_VARS, 2)

    expr = f"({count_var} < {value}) && ({bool_vars[0]} || {bool_vars[1]})"
    desc = f"({count_var} < {value}) AND ({bool_vars[0]} OR {bool_vars[1]})"
    return expr, [count_var] + bool_vars, desc


def generate_guard(level: GuardLevel) -> Tuple[str, List[str], str]:
    """Generate a guard at the specified complexity level."""
    generators = {
        GuardLevel.L1_SINGLE: generate_l1_guard,
        GuardLevel.L2_BOOLEAN: generate_l2_guard,
        GuardLevel.L3_COMPARISON: generate_l3_guard,
        GuardLevel.L4_NESTED: generate_l4_guard,
    }
    return generators[level]()


def generate_statechart_with_guard(
    level: GuardLevel,
    label: str = "Machine",
) -> Dict:
    """Generate a complete statechart with a guarded transition."""
    guard_expr, variables, description = generate_guard(level)

    # Generate appropriate state names
    states = _get_states_for_context(variables)

    sc = {
        "root_state": {
            "label": label,
            "type": 2,  # Normal (OR)
            "children": [
                {
                    "label": states[0],
                    "type": 1,  # Basic
                    "is_initial": True,
                },
                {
                    "label": states[1],
                    "type": 1,  # Basic
                },
            ],
        },
        "transitions": [
            {
                "from": [states[0]],
                "to": [states[1]],
                "event": _get_event_for_context(variables),
                "guard": {
                    "expression": guard_expr,
                    "language": "go",
                },
            },
        ],
    }

    return sc, guard_expr, variables, description


def _get_states_for_context(variables: List[str]) -> List[str]:
    """Get state names appropriate for the guard context."""
    # Map variable contexts to state names
    if any("retry" in v or "attempt" in v for v in variables):
        return ["Waiting", "Retrying"]
    if any("auth" in v or "permission" in v for v in variables):
        return ["Unauthorized", "Authorized"]
    if any("connect" in v for v in variables):
        return ["Disconnected", "Connected"]
    if any("lock" in v for v in variables):
        return ["Unlocked", "Locked"]
    if any("ready" in v or "valid" in v for v in variables):
        return ["Pending", "Active"]
    if any("health" in v or "energy" in v for v in variables):
        return ["Idle", "Running"]
    return ["StateA", "StateB"]


def _get_event_for_context(variables: List[str]) -> str:
    """Get event name appropriate for the guard context."""
    if any("retry" in v or "attempt" in v for v in variables):
        return "RETRY"
    if any("auth" in v or "permission" in v for v in variables):
        return "LOGIN"
    if any("connect" in v for v in variables):
        return "CONNECT"
    if any("lock" in v for v in variables):
        return "TOGGLE_LOCK"
    if any("ready" in v for v in variables):
        return "ACTIVATE"
    return "TRIGGER"


def generate_training_prompt(level: GuardLevel) -> str:
    """Generate a training prompt for guard synthesis."""
    level_descriptions = {
        GuardLevel.L1_SINGLE: "a single boolean variable",
        GuardLevel.L2_BOOLEAN: "two boolean variables with AND or OR",
        GuardLevel.L3_COMPARISON: "a numeric comparison",
        GuardLevel.L4_NESTED: "nested boolean expressions with parentheses",
    }

    desc = level_descriptions[level]

    # Generate a specific context
    contexts = [
        ("retry mechanism", "retry_count", "max_retries"),
        ("authentication", "is_authenticated", "has_permission"),
        ("connection handler", "is_connected", "is_ready"),
        ("state validator", "is_valid", "is_complete"),
        ("resource manager", "count", "limit"),
    ]
    context_name, var1, var2 = random.choice(contexts)

    return f"Create a statechart for a {context_name} with a guarded transition using {desc}."


def validate_guard_syntax(guard_expr: str) -> Tuple[bool, Optional[str]]:
    """
    Validate guard expression syntax.

    Returns (is_valid, error_message).
    """
    if not guard_expr or not guard_expr.strip():
        return False, "Empty guard expression"

    # Check balanced parentheses
    paren_count = 0
    for char in guard_expr:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses"
    if paren_count != 0:
        return False, "Unbalanced parentheses"

    # Check for valid operators
    valid_ops = ['&&', '||', '!', '<', '>', '<=', '>=', '==', '!=']

    # Check for tautologies/contradictions (semantic check)
    if guard_expr in ['true', 'false']:
        return False, "Trivial guard (always true/false)"

    # Check for double operators
    if '&&&&' in guard_expr or '||||' in guard_expr:
        return False, "Invalid double operator"

    # Check variable names start with letter
    import re
    vars_in_expr = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', guard_expr)
    keywords = {'true', 'false', 'null', 'nil'}
    for var in vars_in_expr:
        if var in keywords:
            continue
        if not var[0].isalpha() and var[0] != '_':
            return False, f"Invalid variable name: {var}"

    return True, None


def validate_guard_semantics(guard_expr: str, level: GuardLevel) -> Tuple[bool, Optional[str]]:
    """
    Validate guard expression semantics for the expected level.

    Returns (is_valid, error_message).
    """
    # Count operators to verify complexity
    and_count = guard_expr.count('&&')
    or_count = guard_expr.count('||')
    not_count = guard_expr.count('!')
    comparison_ops = sum(guard_expr.count(op) for op in ['<', '>', '<=', '>=', '==', '!='])
    paren_count = guard_expr.count('(')

    if level == GuardLevel.L1_SINGLE:
        # Should have no binary operators
        if and_count > 0 or or_count > 0:
            return False, "L1 should not have AND/OR operators"
        if comparison_ops > 0:
            return False, "L1 should not have comparison operators"

    elif level == GuardLevel.L2_BOOLEAN:
        # Should have at least one AND or OR
        if and_count == 0 and or_count == 0:
            return False, "L2 should have AND or OR operator"
        if comparison_ops > 0:
            return False, "L2 should use boolean operators, not comparisons"

    elif level == GuardLevel.L3_COMPARISON:
        # Should have at least one comparison
        if comparison_ops == 0:
            return False, "L3 should have comparison operator"

    elif level == GuardLevel.L4_NESTED:
        # Should have parentheses
        if paren_count == 0:
            return False, "L4 should have parentheses for nesting"
        # Should have multiple operators
        if (and_count + or_count + comparison_ops) < 2:
            return False, "L4 should have multiple operators"

    return True, None


def extract_guard_from_statechart(sc_json: str) -> Optional[str]:
    """Extract guard expression from generated statechart JSON."""
    try:
        sc = json.loads(sc_json)

        # Look for guard in transitions
        transitions = sc.get("transitions", [])
        for t in transitions:
            guard = t.get("guard")
            if guard is None:
                continue
            if isinstance(guard, dict):
                expr = guard.get("expression", "")
                if expr and isinstance(expr, str):
                    return expr
            elif isinstance(guard, str):
                return guard

        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


# Test function
if __name__ == "__main__":
    print("Guard Generator Test")
    print("=" * 60)

    for level in GuardLevel:
        print(f"\n{level.name}:")
        for _ in range(3):
            expr, vars, desc = generate_guard(level)
            is_valid, error = validate_guard_syntax(expr)
            semantic_valid, sem_error = validate_guard_semantics(expr, level)
            status = "✓" if (is_valid and semantic_valid) else "✗"
            print(f"  {status} {expr}")
            if not is_valid:
                print(f"      Syntax error: {error}")
            if not semantic_valid:
                print(f"      Semantic error: {sem_error}")

    print("\n" + "=" * 60)
    print("Example Statechart with L3 Guard:")
    sc, guard, vars, desc = generate_statechart_with_guard(GuardLevel.L3_COMPARISON)
    print(json.dumps(sc, indent=2))
