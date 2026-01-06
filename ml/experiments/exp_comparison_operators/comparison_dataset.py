"""
Comparison Dataset: 100+ L3 examples for few-shot prompting.

Covers all operators (<, >, <=, >=, ==, !=) with diverse contexts.
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ComparisonExample:
    """A single comparison example."""
    natural_language: str
    guard_expression: str
    operator: str
    variable: str
    value: str


# Comprehensive examples for each operator
COMPARISON_EXAMPLES: List[ComparisonExample] = [
    # LESS THAN (<)
    ComparisonExample("count less than 5", "count < 5", "<", "count", "5"),
    ComparisonExample("health below 20", "health < 20", "<", "health", "20"),
    ComparisonExample("score under 100", "score < 100", "<", "score", "100"),
    ComparisonExample("age is less than 18", "age < 18", "<", "age", "18"),
    ComparisonExample("temperature under freezing (32)", "temperature < 32", "<", "temperature", "32"),
    ComparisonExample("retries fewer than max", "retries < max_retries", "<", "retries", "max_retries"),
    ComparisonExample("buffer size below capacity", "buffer_size < capacity", "<", "buffer_size", "capacity"),
    ComparisonExample("level less than required", "level < required_level", "<", "level", "required_level"),
    ComparisonExample("time remaining under 60", "time_remaining < 60", "<", "time_remaining", "60"),
    ComparisonExample("attempts below limit", "attempts < limit", "<", "attempts", "limit"),
    ComparisonExample("x smaller than y", "x < y", "<", "x", "y"),
    ComparisonExample("index less than length", "index < length", "<", "index", "length"),
    ComparisonExample("price under budget", "price < budget", "<", "price", "budget"),
    ComparisonExample("speed below maximum", "speed < max_speed", "<", "speed", "max_speed"),
    ComparisonExample("errors fewer than threshold", "errors < threshold", "<", "errors", "threshold"),
    ComparisonExample("stock under minimum", "stock < min_stock", "<", "stock", "min_stock"),
    ComparisonExample("distance less than 10", "distance < 10", "<", "distance", "10"),

    # GREATER THAN (>)
    ComparisonExample("count greater than 5", "count > 5", ">", "count", "5"),
    ComparisonExample("health above 50", "health > 50", ">", "health", "50"),
    ComparisonExample("score over 1000", "score > 1000", ">", "score", "1000"),
    ComparisonExample("age exceeds 21", "age > 21", ">", "age", "21"),
    ComparisonExample("temperature above boiling", "temperature > 100", ">", "temperature", "100"),
    ComparisonExample("retries more than allowed", "retries > max_allowed", ">", "retries", "max_allowed"),
    ComparisonExample("memory exceeds limit", "memory > limit", ">", "memory", "limit"),
    ComparisonExample("level higher than 10", "level > 10", ">", "level", "10"),
    ComparisonExample("elapsed time over timeout", "elapsed > timeout", ">", "elapsed", "timeout"),
    ComparisonExample("value greater than zero", "value > 0", ">", "value", "0"),
    ComparisonExample("x larger than y", "x > y", ">", "x", "y"),
    ComparisonExample("count more than expected", "count > expected", ">", "count", "expected"),
    ComparisonExample("priority higher than normal", "priority > normal_priority", ">", "priority", "normal_priority"),
    ComparisonExample("load above capacity", "load > capacity", ">", "load", "capacity"),
    ComparisonExample("balance over minimum", "balance > minimum", ">", "balance", "minimum"),
    ComparisonExample("speed exceeds limit", "speed > speed_limit", ">", "speed", "speed_limit"),
    ComparisonExample("quantity more than 100", "quantity > 100", ">", "quantity", "100"),

    # LESS THAN OR EQUAL (<=)
    ComparisonExample("count at most 5", "count <= 5", "<=", "count", "5"),
    ComparisonExample("health no more than max", "health <= max_health", "<=", "health", "max_health"),
    ComparisonExample("score 100 or less", "score <= 100", "<=", "score", "100"),
    ComparisonExample("age is at most 65", "age <= 65", "<=", "age", "65"),
    ComparisonExample("temperature at or below 100", "temperature <= 100", "<=", "temperature", "100"),
    ComparisonExample("retries within limit", "retries <= max_retries", "<=", "retries", "max_retries"),
    ComparisonExample("size doesn't exceed capacity", "size <= capacity", "<=", "size", "capacity"),
    ComparisonExample("level at most required", "level <= required", "<=", "level", "required"),
    ComparisonExample("time remaining 60 or less", "time <= 60", "<=", "time", "60"),
    ComparisonExample("value no greater than max", "value <= max_value", "<=", "value", "max_value"),
    ComparisonExample("x not more than y", "x <= y", "<=", "x", "y"),
    ComparisonExample("index at most length minus 1", "index <= length - 1", "<=", "index", "length - 1"),
    ComparisonExample("price within budget", "price <= budget", "<=", "price", "budget"),
    ComparisonExample("speed at or under limit", "speed <= limit", "<=", "speed", "limit"),
    ComparisonExample("errors 5 or fewer", "errors <= 5", "<=", "errors", "5"),
    ComparisonExample("volume at most 100", "volume <= 100", "<=", "volume", "100"),
    ComparisonExample("depth no more than 10", "depth <= 10", "<=", "depth", "10"),

    # GREATER THAN OR EQUAL (>=)
    ComparisonExample("count at least 5", "count >= 5", ">=", "count", "5"),
    ComparisonExample("health 50 or above", "health >= 50", ">=", "health", "50"),
    ComparisonExample("score at least 100", "score >= 100", ">=", "score", "100"),
    ComparisonExample("age is 18 or older", "age >= 18", ">=", "age", "18"),
    ComparisonExample("temperature at or above freezing", "temperature >= 32", ">=", "temperature", "32"),
    ComparisonExample("retries meet minimum", "retries >= min_retries", ">=", "retries", "min_retries"),
    ComparisonExample("balance sufficient", "balance >= required", ">=", "balance", "required"),
    ComparisonExample("level meets requirement", "level >= required_level", ">=", "level", "required_level"),
    ComparisonExample("time at least 60", "time >= 60", ">=", "time", "60"),
    ComparisonExample("value not less than minimum", "value >= minimum", ">=", "value", "minimum"),
    ComparisonExample("x at least as large as y", "x >= y", ">=", "x", "y"),
    ComparisonExample("progress at least 50 percent", "progress >= 50", ">=", "progress", "50"),
    ComparisonExample("stock meets threshold", "stock >= threshold", ">=", "stock", "threshold"),
    ComparisonExample("score qualifying (70+)", "score >= 70", ">=", "score", "70"),
    ComparisonExample("capacity enough for demand", "capacity >= demand", ">=", "capacity", "demand"),
    ComparisonExample("strength sufficient", "strength >= required", ">=", "strength", "required"),
    ComparisonExample("votes reach quorum", "votes >= quorum", ">=", "votes", "quorum"),

    # EQUAL (==)
    ComparisonExample("count equals 5", "count == 5", "==", "count", "5"),
    ComparisonExample("health exactly zero", "health == 0", "==", "health", "0"),
    ComparisonExample("score is exactly 100", "score == 100", "==", "score", "100"),
    ComparisonExample("state equals ready", "state == ready", "==", "state", "ready"),
    ComparisonExample("status is complete", "status == complete", "==", "status", "complete"),
    ComparisonExample("retries equal max", "retries == max_retries", "==", "retries", "max_retries"),
    ComparisonExample("value matches expected", "value == expected", "==", "value", "expected"),
    ComparisonExample("level is exactly 10", "level == 10", "==", "level", "10"),
    ComparisonExample("count same as target", "count == target", "==", "count", "target"),
    ComparisonExample("x identical to y", "x == y", "==", "x", "y"),
    ComparisonExample("index at position 0", "index == 0", "==", "index", "0"),
    ComparisonExample("mode equals debug", "mode == debug", "==", "mode", "debug"),
    ComparisonExample("type is error", "type == error", "==", "type", "error"),
    ComparisonExample("result matches success", "result == success", "==", "result", "success"),
    ComparisonExample("phase exactly 3", "phase == 3", "==", "phase", "3"),
    ComparisonExample("step is final", "step == final_step", "==", "step", "final_step"),
    ComparisonExample("id matches target", "id == target_id", "==", "id", "target_id"),

    # NOT EQUAL (!=)
    ComparisonExample("count not equal to zero", "count != 0", "!=", "count", "0"),
    ComparisonExample("health not zero", "health != 0", "!=", "health", "0"),
    ComparisonExample("score differs from 100", "score != 100", "!=", "score", "100"),
    ComparisonExample("state is not idle", "state != idle", "!=", "state", "idle"),
    ComparisonExample("status not complete", "status != complete", "!=", "status", "complete"),
    ComparisonExample("error code different from OK", "error != OK", "!=", "error", "OK"),
    ComparisonExample("value doesn't match expected", "value != expected", "!=", "value", "expected"),
    ComparisonExample("level is not zero", "level != 0", "!=", "level", "0"),
    ComparisonExample("result not null", "result != null", "!=", "result", "null"),
    ComparisonExample("x different from y", "x != y", "!=", "x", "y"),
    ComparisonExample("index not at end", "index != end", "!=", "index", "end"),
    ComparisonExample("mode not production", "mode != production", "!=", "mode", "production"),
    ComparisonExample("type is not none", "type != none", "!=", "type", "none"),
    ComparisonExample("response differs from cached", "response != cached", "!=", "response", "cached"),
    ComparisonExample("current not equal to previous", "current != previous", "!=", "current", "previous"),
    ComparisonExample("input not empty", "input != empty", "!=", "input", "empty"),
    ComparisonExample("flag is not false", "flag != false", "!=", "flag", "false"),
]


def get_examples_by_operator(operator: str) -> List[ComparisonExample]:
    """Get all examples for a specific operator."""
    return [ex for ex in COMPARISON_EXAMPLES if ex.operator == operator]


def get_few_shot_examples(n: int = 10, operator: str = None) -> List[ComparisonExample]:
    """Get n examples for few-shot prompting."""
    import random
    if operator:
        pool = get_examples_by_operator(operator)
    else:
        pool = COMPARISON_EXAMPLES
    return random.sample(pool, min(n, len(pool)))


def format_examples_for_prompt(examples: List[ComparisonExample]) -> str:
    """Format examples as few-shot prompt text."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{i}. \"{ex.natural_language}\" → {ex.guard_expression}")
    return "\n".join(lines)


# Test data for benchmark
TEST_CASES = [
    # Required test cases from task
    ("count greater than 5", "count > 5", ">"),
    ("temperature less than or equal to 100", "temperature <= 100", "<="),
    ("health not equal to zero", "health != 0", "!="),
    ("score at least 50", "score >= 50", ">="),
    ("age below 18", "age < 18", "<"),

    # Additional diversity
    ("retries exceed maximum", "retries > max", ">"),
    ("level at most 10", "level <= 10", "<="),
    ("status equals ready", "status == ready", "=="),
    ("balance insufficient (under 100)", "balance < 100", "<"),
    ("attempts at least 3", "attempts >= 3", ">="),
    ("error not OK", "error != OK", "!="),
    ("time over limit", "time > limit", ">"),
    ("index less than length", "index < length", "<"),
    ("value exactly 42", "value == 42", "=="),
    ("priority higher than normal", "priority > normal", ">"),
]


if __name__ == "__main__":
    print(f"Total examples: {len(COMPARISON_EXAMPLES)}")

    for op in ["<", ">", "<=", ">=", "==", "!="]:
        examples = get_examples_by_operator(op)
        print(f"  {op}: {len(examples)} examples")

    print("\nSample few-shot prompt:")
    print(format_examples_for_prompt(get_few_shot_examples(5)))
