"""
Path Length Prediction Benchmark with REAL MLX Inference.

Tests model's ability to predict steps until terminal state for:
- Fixed path (deterministic)
- Variable path (depends on guards/context)
- Unbounded (may not terminate)

Uses Qwen2.5-Coder-1.5B-Instruct-4bit for inference.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from .length_predictor import (
    PathLengthExample,
    generate_fixed_path_examples,
    generate_variable_path_examples,
    generate_unbounded_examples,
    compute_actual_path_length,
    create_path_length_prompt,
    parse_path_length_response,
    format_examples_for_few_shot,
)


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    example: PathLengthExample
    predicted: int  # -1 for infinite
    actual: int     # -1 for infinite
    is_terminal: bool
    error: Optional[str] = None

    @property
    def exact_match(self) -> bool:
        return self.predicted == self.actual

    @property
    def within_1(self) -> bool:
        if self.predicted == -1 and self.actual == -1:
            return True
        if self.predicted == -1 or self.actual == -1:
            return False
        return abs(self.predicted - self.actual) <= 1

    @property
    def within_2(self) -> bool:
        if self.predicted == -1 and self.actual == -1:
            return True
        if self.predicted == -1 or self.actual == -1:
            return False
        return abs(self.predicted - self.actual) <= 2

    @property
    def absolute_error(self) -> Optional[int]:
        if self.predicted == -1 or self.actual == -1:
            return None
        return abs(self.predicted - self.actual)


@dataclass
class BenchmarkResult:
    """Full benchmark results."""
    fixed_exact: int
    fixed_total: int
    variable_exact: int
    variable_total: int
    unbounded_exact: int
    unbounded_total: int
    within_1_count: int
    within_2_count: int
    total_count: int
    total_mae: float
    all_results: List[PredictionResult] = field(default_factory=list)

    @property
    def fixed_pct(self) -> float:
        return self.fixed_exact / self.fixed_total * 100 if self.fixed_total > 0 else 0

    @property
    def variable_pct(self) -> float:
        return self.variable_exact / self.variable_total * 100 if self.variable_total > 0 else 0

    @property
    def unbounded_pct(self) -> float:
        return self.unbounded_exact / self.unbounded_total * 100 if self.unbounded_total > 0 else 0

    @property
    def exact_pct(self) -> float:
        total_exact = self.fixed_exact + self.variable_exact + self.unbounded_exact
        return total_exact / self.total_count * 100 if self.total_count > 0 else 0

    @property
    def within_1_pct(self) -> float:
        return self.within_1_count / self.total_count * 100 if self.total_count > 0 else 0


def create_full_prompt(
    sc_json: Dict,
    initial_context: Dict[str, Any],
    events: List[str],
    few_shot_examples: str,
) -> str:
    """Create full prompt with few-shot examples."""
    # Count states and transitions for simpler reasoning
    root = sc_json.get("root_state", {})
    states = []
    def collect_states(s):
        if s.get("label"):
            states.append(s["label"])
        for c in s.get("children", []):
            collect_states(c)
    collect_states(root)

    transitions = sc_json.get("transitions", [])

    # Find terminal states
    terminal_states = []
    def find_terminal(s):
        if s.get("is_final"):
            terminal_states.append(s.get("label", "unknown"))
        for c in s.get("children", []):
            find_terminal(c)
    find_terminal(root)

    # Find initial state
    initial_state = None
    def find_initial(s):
        nonlocal initial_state
        if s.get("is_initial"):
            initial_state = s.get("label")
        for c in s.get("children", []):
            find_initial(c)
    find_initial(root)

    # Build transition list
    trans_list = []
    loop_bounds = []
    for t in transitions:
        frm = t.get("from", ["?"])[0]
        to = t.get("to", ["?"])[0]
        guard = t.get("guard", {})
        guard_str = ""
        if isinstance(guard, dict):
            expr = guard.get("expression", "")
            if expr:
                guard_str = f" if {expr}"
                # Extract loop bounds
                import re
                match = re.search(r'(\w+)\s*<\s*(\d+)', expr)
                if match and frm == to:
                    var, bound = match.groups()
                    loop_bounds.append((frm, var, int(bound)))
        trans_list.append(f"{frm}->{to}{guard_str}")

    # Calculate expected steps based on structure
    # Provide explicit calculation hint
    has_terminal = len(terminal_states) > 0

    # Count steps: initial -> ... -> terminal
    step_hint = ""

    # Check for unbounded (very high loop bounds or no terminal)
    is_unbounded = False
    if not has_terminal:
        is_unbounded = True
        step_hint = "No terminal state reachable = infinite"
    elif loop_bounds:
        for state, var, bound in loop_bounds:
            ctx_val = initial_context.get(var, 0)
            iterations = bound - ctx_val
            if iterations > 100:  # Very long loop = effectively unbounded
                is_unbounded = True
                step_hint = "Loop iterations > 100 = infinite"
            else:
                step_hint = f"Calculation: 1 (start) + {iterations} (loops) + 1 (exit) = {iterations + 2} steps"
    elif has_terminal:
        # No loops, count shortest path to terminal
        # Simple heuristic: count unique state hops needed
        # For branching paths (A->B1/B2->C->D): path is 3 regardless of branch
        non_loop_trans = [t for t in transitions if t.get("from") != t.get("to")]
        # Estimate path length: (total non-loop transitions) / 2 since branching doubles
        path_len = len(non_loop_trans)
        if path_len > 3:  # Likely has branching
            path_len = (path_len + 1) // 2  # Rough heuristic
        step_hint = f"Calculation: {path_len} transitions to terminal"

    # Very simple prompt with varied examples
    return f"""Count path length to terminal state.
Examples: A->B->C=2, Start->Loop(x3)->End=5, X<->Y(no terminal)=infinite

States: {states}
Terminal: {terminal_states if terminal_states else 'NONE'}
Transitions: {', '.join(trans_list[:5])}{'...' if len(trans_list) > 5 else ''}
Answer:"""


def run_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    samples_per_type: int = 3,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run full benchmark with REAL MLX inference."""
    from mlx_lm import load, generate

    print("=" * 60)
    print("PATH LENGTH PREDICTION BENCHMARK (REAL MLX INFERENCE)")
    print("=" * 60)

    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    print("Model loaded.")

    result = BenchmarkResult(
        fixed_exact=0, fixed_total=0,
        variable_exact=0, variable_total=0,
        unbounded_exact=0, unbounded_total=0,
        within_1_count=0, within_2_count=0,
        total_count=0, total_mae=0.0,
    )

    # Get examples
    fixed_examples = generate_fixed_path_examples()[:samples_per_type]
    variable_examples = generate_variable_path_examples()[:samples_per_type]
    unbounded_examples = generate_unbounded_examples()[:samples_per_type]

    # Create few-shot examples (use 2 from each category)
    few_shot_str = format_examples_for_few_shot(
        fixed_examples[:1] + variable_examples[:1] + unbounded_examples[:1],
        n=3
    )

    all_examples = [
        ("fixed", fixed_examples),
        ("variable", variable_examples),
        ("unbounded", unbounded_examples),
    ]

    mae_sum = 0
    mae_count = 0

    for path_type, examples in all_examples:
        print(f"\n--- Testing {path_type.upper()} path ---")

        for ex in examples:
            # Compute actual path length
            actual_len, is_terminal = compute_actual_path_length(
                ex.sc_json, ex.events, ex.initial_context
            )

            # For unbounded, actual should be -1 if not terminal
            if path_type == "unbounded" and not is_terminal:
                actual_len = -1

            # Create prompt
            prompt = create_full_prompt(
                ex.sc_json,
                ex.initial_context,
                ex.events,
                few_shot_str,
            )

            # Apply chat template with system prompt
            messages = [
                {"role": "system", "content": "You are a calculator. Output ONLY a number or 'infinite'. No explanation."},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate
            try:
                output = generate(
                    model, tokenizer,
                    prompt=formatted,
                    max_tokens=20,
                    verbose=False,
                )

                # Parse response
                predicted = parse_path_length_response(output)

                pred_result = PredictionResult(
                    example=ex,
                    predicted=predicted,
                    actual=actual_len,
                    is_terminal=is_terminal,
                )

            except Exception as e:
                pred_result = PredictionResult(
                    example=ex,
                    predicted=-1,
                    actual=actual_len,
                    is_terminal=is_terminal,
                    error=str(e),
                )

            result.all_results.append(pred_result)
            result.total_count += 1

            # Update counts by type
            if path_type == "fixed":
                result.fixed_total += 1
                if pred_result.exact_match:
                    result.fixed_exact += 1
            elif path_type == "variable":
                result.variable_total += 1
                if pred_result.exact_match:
                    result.variable_exact += 1
            elif path_type == "unbounded":
                result.unbounded_total += 1
                if pred_result.exact_match:
                    result.unbounded_exact += 1

            # Within tolerance
            if pred_result.within_1:
                result.within_1_count += 1
            if pred_result.within_2:
                result.within_2_count += 1

            # MAE (only for bounded cases)
            if pred_result.absolute_error is not None:
                mae_sum += pred_result.absolute_error
                mae_count += 1

            if verbose:
                status = "OK" if pred_result.exact_match else ("~1" if pred_result.within_1 else "FAIL")
                pred_str = str(predicted) if predicted >= 0 else "infinite"
                actual_str = str(actual_len) if actual_len >= 0 else "infinite"
                print(f"  {ex.name}: pred={pred_str}, actual={actual_str} [{status}]")
                if pred_result.error:
                    print(f"    Error: {pred_result.error}")

    # Calculate MAE
    result.total_mae = mae_sum / mae_count if mae_count > 0 else 0.0

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Fixed path:    {result.fixed_exact}/{result.fixed_total} ({result.fixed_pct:.0f}%)")
    print(f"Variable path: {result.variable_exact}/{result.variable_total} ({result.variable_pct:.0f}%)")
    print(f"Unbounded:     {result.unbounded_exact}/{result.unbounded_total} ({result.unbounded_pct:.0f}%)")
    print(f"Overall exact: {result.exact_pct:.0f}%")
    print(f"Within ±1:     {result.within_1_pct:.0f}%")
    print(f"MAE:           {result.total_mae:.2f}")

    return result


def format_report(result: BenchmarkResult) -> str:
    """Format result for orchestrator report."""
    return (
        f"PATH_LENGTH exact={result.exact_pct:.0f}%, "
        f"within_1={result.within_1_pct:.0f}%, mae={result.total_mae:.1f}\n\n"
        f"Results:\n"
        f"- Fixed path: {result.fixed_exact}/{result.fixed_total} ({result.fixed_pct:.0f}%)\n"
        f"- Variable path: {result.variable_exact}/{result.variable_total} ({result.variable_pct:.0f}%)\n"
        f"- Unbounded: {result.unbounded_exact}/{result.unbounded_total} ({result.unbounded_pct:.0f}%)\n"
        f"- Within ±1: {result.within_1_count}/{result.total_count} ({result.within_1_pct:.0f}%)\n"
        f"- MAE: {result.total_mae:.2f}"
    )


if __name__ == "__main__":
    result = run_benchmark(samples_per_type=3)
    print("\n" + format_report(result))
