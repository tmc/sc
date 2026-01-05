"""
Comparison Operator Guard Generator.

Generates guards specifically using comparison operators (<, >, <=, >=, ==, !=)
to address the L3 weakness in guard synthesis.
"""

import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional


class ComparisonOperator(Enum):
    """Comparison operators."""
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "=="
    NE = "!="


class ComparisonContext(Enum):
    """Semantic contexts for comparisons."""
    RETRY = "retry"
    HEALTH = "health"
    SCORE = "score"
    LEVEL = "level"
    TIME = "time"
    COUNT = "count"
    BUFFER = "buffer"
    CAPACITY = "capacity"


# Context-specific variable mappings
CONTEXT_VARS = {
    ComparisonContext.RETRY: {
        "vars": ["retry_count", "attempt_count", "error_count"],
        "thresholds": ["max_retries", "max_attempts", "retry_limit"],
        "literals": [3, 5, 10],
        "states": ("Waiting", "Retrying"),
        "event": "RETRY",
    },
    ComparisonContext.HEALTH: {
        "vars": ["health", "hp", "health_points"],
        "thresholds": ["max_health", "critical_threshold", "heal_threshold"],
        "literals": [0, 10, 20, 50, 100],
        "states": ("Alive", "Dead"),
        "event": "DAMAGE",
    },
    ComparisonContext.SCORE: {
        "vars": ["score", "points", "total_score"],
        "thresholds": ["winning_score", "target_score", "high_score"],
        "literals": [100, 500, 1000, 10000],
        "states": ("Playing", "Won"),
        "event": "SCORE_UPDATE",
    },
    ComparisonContext.LEVEL: {
        "vars": ["level", "current_level", "player_level"],
        "thresholds": ["required_level", "min_level", "unlock_level"],
        "literals": [1, 5, 10, 50, 100],
        "states": ("Locked", "Unlocked"),
        "event": "CHECK_LEVEL",
    },
    ComparisonContext.TIME: {
        "vars": ["elapsed_time", "time_remaining", "duration"],
        "thresholds": ["timeout", "max_duration", "time_limit"],
        "literals": [0, 30, 60, 300, 3600],
        "states": ("Running", "Expired"),
        "event": "TICK",
    },
    ComparisonContext.COUNT: {
        "vars": ["count", "item_count", "total_count"],
        "thresholds": ["max_count", "limit", "capacity"],
        "literals": [0, 1, 5, 10, 100],
        "states": ("Available", "Full"),
        "event": "ADD_ITEM",
    },
    ComparisonContext.BUFFER: {
        "vars": ["buffer_size", "queue_length", "pending_count"],
        "thresholds": ["buffer_capacity", "max_queue", "threshold"],
        "literals": [0, 10, 100, 1024],
        "states": ("Accepting", "Blocked"),
        "event": "ENQUEUE",
    },
    ComparisonContext.CAPACITY: {
        "vars": ["used_capacity", "current_load", "memory_used"],
        "thresholds": ["max_capacity", "soft_limit", "hard_limit"],
        "literals": [0, 50, 80, 90, 100],
        "states": ("Normal", "Overloaded"),
        "event": "CHECK_CAPACITY",
    },
}

# Natural language descriptions per operator
OPERATOR_DESCRIPTIONS = {
    ComparisonOperator.LT: [
        "{var} is less than {threshold}",
        "{var} hasn't reached {threshold}",
        "{var} is below {threshold}",
        "{var} is under {threshold}",
    ],
    ComparisonOperator.GT: [
        "{var} is greater than {threshold}",
        "{var} exceeds {threshold}",
        "{var} is above {threshold}",
        "{var} is over {threshold}",
    ],
    ComparisonOperator.LE: [
        "{var} is at most {threshold}",
        "{var} doesn't exceed {threshold}",
        "{var} is {threshold} or less",
        "{var} is no more than {threshold}",
    ],
    ComparisonOperator.GE: [
        "{var} is at least {threshold}",
        "{var} has reached {threshold}",
        "{var} is {threshold} or more",
        "{var} meets or exceeds {threshold}",
    ],
    ComparisonOperator.EQ: [
        "{var} equals {threshold}",
        "{var} is exactly {threshold}",
        "{var} matches {threshold}",
        "{var} is {threshold}",
    ],
    ComparisonOperator.NE: [
        "{var} is not equal to {threshold}",
        "{var} differs from {threshold}",
        "{var} is not {threshold}",
        "{var} doesn't equal {threshold}",
    ],
}


@dataclass
class ComparisonGuard:
    """A generated comparison guard."""
    expression: str
    operator: ComparisonOperator
    context: ComparisonContext
    var_name: str
    threshold: str
    description: str
    statechart: Dict


def generate_comparison_guard(
    operator: Optional[ComparisonOperator] = None,
    context: Optional[ComparisonContext] = None,
    use_literal: Optional[bool] = None,
) -> ComparisonGuard:
    """Generate a comparison guard with specified or random parameters."""
    if operator is None:
        operator = random.choice(list(ComparisonOperator))
    if context is None:
        context = random.choice(list(ComparisonContext))
    if use_literal is None:
        use_literal = random.random() < 0.5

    ctx = CONTEXT_VARS[context]
    var_name = random.choice(ctx["vars"])

    if use_literal:
        threshold = str(random.choice(ctx["literals"]))
    else:
        threshold = random.choice(ctx["thresholds"])

    expression = f"{var_name} {operator.value} {threshold}"

    # Generate natural language description
    desc_template = random.choice(OPERATOR_DESCRIPTIONS[operator])
    description = desc_template.format(var=var_name, threshold=threshold)

    # Generate statechart
    states = ctx["states"]
    event = ctx["event"]

    statechart = {
        "root_state": {
            "label": f"{context.value.title()}Machine",
            "type": 2,
            "children": [
                {"label": states[0], "type": 1, "is_initial": True},
                {"label": states[1], "type": 1},
            ],
        },
        "transitions": [
            {
                "from": [states[0]],
                "to": [states[1]],
                "event": event,
                "guard": {
                    "expression": expression,
                    "language": "go",
                },
            },
        ],
    }

    return ComparisonGuard(
        expression=expression,
        operator=operator,
        context=context,
        var_name=var_name,
        threshold=threshold,
        description=description,
        statechart=statechart,
    )


def create_comparison_prompt(
    operator: Optional[ComparisonOperator] = None,
    context: Optional[ComparisonContext] = None,
) -> Tuple[str, ComparisonGuard]:
    """Create a prompt specifically for comparison guard generation."""
    # Generate target guard
    target = generate_comparison_guard(operator, context)

    ctx = CONTEXT_VARS[target.context]
    states = ctx["states"]
    event = ctx["event"]

    # Operator descriptions with explicit symbols
    op_desc = {
        ComparisonOperator.LT: "strictly less than, symbol: <",
        ComparisonOperator.GT: "strictly greater than, symbol: >",
        ComparisonOperator.LE: "less than or equal to, symbol: <=",
        ComparisonOperator.GE: "greater than or equal to, symbol: >=",
        ComparisonOperator.EQ: "equal to, symbol: ==",
        ComparisonOperator.NE: "not equal to, symbol: !=",
    }

    prompt = f"""Generate a statechart JSON for a {target.context.value} check.

Requirement: Create a guard that checks if {target.var_name} is {op_desc[target.operator]}.
The guard expression must be: "{target.var_name} {target.operator.value} {target.threshold}"

States: {states[0]} (initial) -> {states[1]}
Event: {event}

Output valid JSON with root_state and transitions. The guard.expression field must contain exactly: {target.var_name} {target.operator.value} {target.threshold}

JSON:
"""

    return prompt, target


def generate_training_dataset(n_samples: int = 100) -> List[Tuple[str, ComparisonGuard]]:
    """Generate a training dataset of comparison guards."""
    dataset = []

    # Ensure coverage of all operators and contexts
    for op in ComparisonOperator:
        for ctx in ComparisonContext:
            # One with literal
            prompt, guard = create_comparison_prompt(op, ctx)
            dataset.append((prompt, guard))

    # Fill remaining with random samples
    while len(dataset) < n_samples:
        prompt, guard = create_comparison_prompt()
        dataset.append((prompt, guard))

    random.shuffle(dataset)
    return dataset[:n_samples]


def validate_comparison_guard(expression: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an expression contains a comparison operator.

    Returns (is_valid, operator_used).
    """
    for op in ComparisonOperator:
        if op.value in expression:
            return True, op.value

    return False, None


if __name__ == "__main__":
    print("Comparison Guard Generator Test")
    print("=" * 60)

    # Test each operator
    for op in ComparisonOperator:
        print(f"\n{op.name} ({op.value}):")
        guard = generate_comparison_guard(operator=op)
        print(f"  Expression: {guard.expression}")
        print(f"  Description: {guard.description}")
        print(f"  Context: {guard.context.value}")

    print("\n" + "=" * 60)
    print("Sample Prompt:")
    prompt, target = create_comparison_prompt()
    print(prompt[:500] + "...")
    print(f"\nTarget: {target.expression}")
