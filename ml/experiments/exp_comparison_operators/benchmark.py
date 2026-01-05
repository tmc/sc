"""
Comparison Operators Benchmark.

Tests LLM ability to generate guards with comparison operators.
Focus on L3 guards that were weak (20%) in exp_guard_synthesis.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .comparison_generator import (
    ComparisonOperator,
    ComparisonContext,
    generate_comparison_guard,
    create_comparison_prompt,
    validate_comparison_guard,
    CONTEXT_VARS,
)

try:
    import mlx.core as mx
    from mlx_lm import load, generate as mlx_generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    max_tokens: int = 300
    temperature: float = 0.3  # Lower temp for more precise comparisons
    samples_per_operator: int = 3
    samples_per_context: int = 2


@dataclass
class ComparisonResult:
    """Result for a single comparison guard generation."""
    target_operator: ComparisonOperator
    target_context: ComparisonContext
    target_expression: str
    prompt: str
    output: str
    generated_guard: Optional[str]
    json_valid: bool
    has_comparison: bool
    correct_operator: bool
    error: Optional[str] = None
    time: float = 0.0


def extract_guard_expression(json_str: str) -> Optional[str]:
    """Extract guard expression from generated JSON."""
    try:
        sc = json.loads(json_str)
        transitions = sc.get("transitions", [])
        for t in transitions:
            guard = t.get("guard")
            if guard:
                if isinstance(guard, dict):
                    return guard.get("expression")
                elif isinstance(guard, str):
                    return guard
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clean_json_output(output: str, prefix: str = "") -> str:
    """Clean generated JSON output, optionally prepending a prefix."""
    # For completion-style prompts, prepend the prefix
    if prefix:
        output = prefix + output

    # Handle prefix if prompt included it
    if not output.strip().startswith("{"):
        # Find the first {
        idx = output.find("{")
        if idx >= 0:
            output = output[idx:]

    # Remove trailing content after balanced JSON
    brace_count = 0
    last_valid = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(output):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                last_valid = i + 1
                break

    if last_valid > 0:
        return output[:last_valid]
    return output


def run_mock_benchmark(config: BenchmarkConfig) -> Dict:
    """Run benchmark in mock mode."""
    print("Running in MOCK mode (MLX not available)")
    print("-" * 60)

    results_by_operator = {op: [] for op in ComparisonOperator}
    results_by_context = {ctx: [] for ctx in ComparisonContext}

    # Test each operator
    for op in ComparisonOperator:
        print(f"\nOperator {op.name} ({op.value}):")
        for i in range(config.samples_per_operator):
            # Generate mock result with correct operator
            guard = generate_comparison_guard(operator=op)

            result = ComparisonResult(
                target_operator=op,
                target_context=guard.context,
                target_expression=guard.expression,
                prompt="mock prompt",
                output=json.dumps(guard.statechart),
                generated_guard=guard.expression,
                json_valid=True,
                has_comparison=True,
                correct_operator=True,
            )
            results_by_operator[op].append(result)
            results_by_context[guard.context].append(result)
            print(f"  [OK] {guard.expression}")

    return compute_summary(results_by_operator, results_by_context)


def run_real_benchmark(config: BenchmarkConfig) -> Dict:
    """Run benchmark with real LLM inference."""
    print(f"Loading model: {config.model_path}")
    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    print("\n" + "-" * 60)
    print("COMPARISON OPERATORS BENCHMARK")
    print("-" * 60)

    results_by_operator = {op: [] for op in ComparisonOperator}
    results_by_context = {ctx: [] for ctx in ComparisonContext}

    # Test each operator
    for op in ComparisonOperator:
        print(f"\nOperator {op.name} ({op.value}):")

        for i in range(config.samples_per_operator):
            # Pick random context
            ctx = random.choice(list(ComparisonContext))
            prompt, target = create_comparison_prompt(op, ctx)

            t0 = time.time()
            output = mlx_generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
            )
            elapsed = time.time() - t0

            # Clean output
            output = clean_json_output(output)

            # Analyze result
            result = ComparisonResult(
                target_operator=op,
                target_context=ctx,
                target_expression=target.expression,
                prompt=prompt[:100] + "...",
                output=output,
                generated_guard=None,
                json_valid=False,
                has_comparison=False,
                correct_operator=False,
                time=elapsed,
            )

            # Check JSON validity
            try:
                json.loads(output)
                result.json_valid = True
            except json.JSONDecodeError as e:
                result.error = f"Invalid JSON: {str(e)[:50]}"

            # Extract and validate guard
            if result.json_valid:
                guard_expr = extract_guard_expression(output)
                result.generated_guard = guard_expr

                if guard_expr:
                    is_comparison, used_op = validate_comparison_guard(guard_expr)
                    result.has_comparison = is_comparison

                    if is_comparison and used_op == op.value:
                        result.correct_operator = True
                    elif is_comparison:
                        result.error = f"Wrong operator: expected {op.value}, got {used_op}"
                else:
                    result.error = "No guard expression found"

            results_by_operator[op].append(result)
            results_by_context[ctx].append(result)

            # Print result
            status = "OK" if result.correct_operator else "FAIL"
            guard_display = result.generated_guard[:30] if result.generated_guard else "None"
            print(f"  [{status}] {guard_display} ({elapsed:.2f}s)")
            if result.error:
                print(f"        Error: {result.error}")

    return compute_summary(results_by_operator, results_by_context)


def compute_summary(
    results_by_operator: Dict[ComparisonOperator, List[ComparisonResult]],
    results_by_context: Dict[ComparisonContext, List[ComparisonResult]],
) -> Dict:
    """Compute summary statistics."""
    summary = {"by_operator": {}, "by_context": {}}

    # By operator
    for op, results in results_by_operator.items():
        if not results:
            continue
        n = len(results)
        json_valid = sum(1 for r in results if r.json_valid)
        has_comparison = sum(1 for r in results if r.has_comparison)
        correct_operator = sum(1 for r in results if r.correct_operator)

        summary["by_operator"][op.name] = {
            "operator": op.value,
            "total": n,
            "json_valid": json_valid,
            "json_rate": json_valid / n,
            "has_comparison": has_comparison,
            "comparison_rate": has_comparison / n,
            "correct_operator": correct_operator,
            "correct_rate": correct_operator / n,
        }

    # By context
    for ctx, results in results_by_context.items():
        if not results:
            continue
        n = len(results)
        correct = sum(1 for r in results if r.correct_operator)

        summary["by_context"][ctx.name] = {
            "total": n,
            "correct": correct,
            "rate": correct / n,
        }

    # Overall
    all_results = [r for results in results_by_operator.values() for r in results]
    total = len(all_results)
    if total > 0:
        json_valid = sum(1 for r in all_results if r.json_valid)
        has_comparison = sum(1 for r in all_results if r.has_comparison)
        correct = sum(1 for r in all_results if r.correct_operator)

        summary["overall"] = {
            "total": total,
            "json_valid": json_valid,
            "json_rate": json_valid / total,
            "has_comparison": has_comparison,
            "comparison_rate": has_comparison / total,
            "correct": correct,
            "correct_rate": correct / total,
        }
    else:
        summary["overall"] = {"total": 0, "correct": 0, "correct_rate": 0}

    return summary


def print_summary(summary: Dict):
    """Print summary in readable format."""
    print("\n" + "=" * 60)
    print("SUMMARY BY OPERATOR")
    print("=" * 60)

    for op_name, stats in summary["by_operator"].items():
        print(f"\n{op_name} ({stats['operator']}):")
        print(f"  JSON valid:      {stats['json_rate']:.1%} ({stats['json_valid']}/{stats['total']})")
        print(f"  Has comparison:  {stats['comparison_rate']:.1%} ({stats['has_comparison']}/{stats['total']})")
        print(f"  Correct operator:{stats['correct_rate']:.1%} ({stats['correct_operator']}/{stats['total']})")

    print("\n" + "=" * 60)
    print("SUMMARY BY CONTEXT")
    print("=" * 60)

    for ctx_name, stats in summary["by_context"].items():
        print(f"  {ctx_name}: {stats['rate']:.1%} ({stats['correct']}/{stats['total']})")

    overall = summary["overall"]
    print("\n" + "=" * 60)
    print(f"OVERALL L3 ACCURACY: {overall['correct_rate']:.1%} ({overall['correct']}/{overall['total']})")
    print("=" * 60)


def run_comparison_benchmark():
    """Run the comparison operators benchmark."""
    print("=" * 60)
    print("COMPARISON OPERATORS BENCHMARK")
    print("Target: Fix L3 guard weakness (20% -> 80%+)")
    print("=" * 60)

    config = BenchmarkConfig()

    if HAS_MLX:
        summary = run_real_benchmark(config)
    else:
        summary = run_mock_benchmark(config)

    print_summary(summary)

    # Generate confusion matrix for operators
    print("\n" + "=" * 60)
    print("OPERATOR CONFUSION (expected vs generated)")
    print("=" * 60)

    for op_name, stats in summary["by_operator"].items():
        rate = stats["correct_rate"]
        bar = "#" * int(rate * 20)
        print(f"  {op_name:5s}: {bar:20s} {rate:.1%}")

    return summary


if __name__ == "__main__":
    run_comparison_benchmark()
