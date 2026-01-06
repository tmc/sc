"""
Improved Guard Generator: Multiple prompting strategies for L3 comparisons.

Strategies:
1. Baseline - Simple prompt (current 20% accuracy)
2. Chain-of-Thought (CoT) - Step-by-step reasoning
3. Template - Explicit structure with placeholders
4. Augmented - 10+ few-shot examples
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .comparison_dataset import (
    COMPARISON_EXAMPLES,
    get_few_shot_examples,
    format_examples_for_prompt,
)


class PromptStrategy(Enum):
    """Available prompting strategies."""
    BASELINE = "baseline"
    COT = "cot"
    TEMPLATE = "template"
    AUGMENTED = "augmented"


@dataclass
class GenerationResult:
    """Result of guard generation."""
    strategy: PromptStrategy
    natural_language: str
    expected_guard: str
    expected_operator: str
    generated_guard: Optional[str]
    is_correct: bool
    has_comparison: bool
    raw_output: str
    generation_time_ms: float


class ImprovedGenerator:
    """Guard generator with multiple prompting strategies."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        natural_language: str,
        expected_guard: str,
        expected_operator: str,
        strategy: PromptStrategy,
    ) -> GenerationResult:
        """Generate guard using specified strategy."""
        prompt = self._build_prompt(natural_language, expected_operator, strategy)

        start = time.time()
        output = self._generate(prompt)
        elapsed_ms = (time.time() - start) * 1000

        # Extract guard from output
        generated = self._extract_guard(output, strategy)

        # Check correctness
        has_comparison = any(op in (generated or "") for op in ["<", ">", "<=", ">=", "==", "!="])
        is_correct = self._check_correct(generated, expected_guard, expected_operator)

        return GenerationResult(
            strategy=strategy,
            natural_language=natural_language,
            expected_guard=expected_guard,
            expected_operator=expected_operator,
            generated_guard=generated,
            is_correct=is_correct,
            has_comparison=has_comparison,
            raw_output=output,
            generation_time_ms=elapsed_ms,
        )

    def _build_prompt(
        self,
        natural_language: str,
        expected_operator: str,
        strategy: PromptStrategy,
    ) -> str:
        """Build prompt based on strategy."""
        if strategy == PromptStrategy.BASELINE:
            return self._baseline_prompt(natural_language)
        elif strategy == PromptStrategy.COT:
            return self._cot_prompt(natural_language)
        elif strategy == PromptStrategy.TEMPLATE:
            return self._template_prompt(natural_language, expected_operator)
        elif strategy == PromptStrategy.AUGMENTED:
            return self._augmented_prompt(natural_language, expected_operator)
        else:
            return self._baseline_prompt(natural_language)

    def _baseline_prompt(self, natural_language: str) -> str:
        """Simple baseline prompt (current approach)."""
        return f"""Convert this condition to a guard expression:

"{natural_language}"

Guard expression:"""

    def _cot_prompt(self, natural_language: str) -> str:
        """Chain-of-thought prompt with step-by-step reasoning."""
        return f"""Convert this condition to a guard expression using step-by-step reasoning.

Condition: "{natural_language}"

Step 1: Identify the variable being compared.
Step 2: Identify the comparison operator (one of: <, >, <=, >=, ==, !=).
Step 3: Identify the value or threshold being compared against.
Step 4: Combine into guard expression: variable operator value

Example:
Condition: "count greater than 5"
Step 1: Variable = count
Step 2: Operator = > (greater than)
Step 3: Value = 5
Step 4: Guard = count > 5

Now apply to: "{natural_language}"
Step 1: Variable ="""

    def _template_prompt(self, natural_language: str, operator: str) -> str:
        """Template prompt with explicit structure."""
        operator_map = {
            "<": "less than",
            ">": "greater than",
            "<=": "less than or equal to",
            ">=": "greater than or equal to",
            "==": "equal to",
            "!=": "not equal to",
        }
        op_desc = operator_map.get(operator, "comparison with")

        return f"""Generate a guard expression using this EXACT format:

VARIABLE {operator} VALUE

Where:
- VARIABLE is the thing being measured (a single word like count, health, score)
- {operator} is the {op_desc} operator
- VALUE is the number or variable being compared to

Examples:
- "count greater than 5" → count > 5
- "health below 20" → health < 20
- "score at least 100" → score >= 100
- "level equals 10" → level == 10
- "retries not zero" → retries != 0

Now convert:
"{natural_language}"

Guard expression (format: VARIABLE {operator} VALUE):"""

    def _augmented_prompt(self, natural_language: str, operator: str) -> str:
        """Augmented prompt with 10+ few-shot examples."""
        # Get diverse examples with emphasis on target operator
        examples = get_few_shot_examples(n=12, operator=operator)
        examples_text = format_examples_for_prompt(examples)

        return f"""Convert natural language conditions to guard expressions.

A guard expression has format: variable operator value
Operators: < (less than), > (greater than), <= (at most), >= (at least), == (equals), != (not equal)

Examples:
{examples_text}

Now convert:
"{natural_language}"

Guard expression:"""

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return ""

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=100,
                sampler=sampler,
            )
            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return ""

    def _extract_guard(self, output: str, strategy: PromptStrategy) -> Optional[str]:
        """Extract guard expression from output."""
        if not output:
            return None

        output = output.strip()

        # For CoT, look for the final Step 4 result or last line with operator
        if strategy == PromptStrategy.COT:
            lines = output.split("\n")
            for line in reversed(lines):
                line = line.strip()
                # Look for "Guard = X" or "Step 4: X" patterns
                if "Guard =" in line or "Guard:" in line:
                    parts = line.split("=", 1) if "=" in line else line.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                # Look for line with operator
                for op in ["<=", ">=", "!=", "==", "<", ">"]:
                    if op in line and not line.startswith("Step"):
                        return line
            # Fallback: look for any line with comparison
            for line in lines:
                for op in ["<=", ">=", "!=", "==", "<", ">"]:
                    if op in line:
                        # Extract just the expression part
                        line = line.strip()
                        if ":" in line:
                            line = line.split(":")[-1].strip()
                        return line

        # For other strategies, take first line with operator
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            # Skip empty or prompt-like lines
            if not line or line.startswith('"') or "convert" in line.lower():
                continue
            # Check for comparison operator
            for op in ["<=", ">=", "!=", "==", "<", ">"]:
                if op in line:
                    # Clean up common prefixes
                    if line.startswith("Guard:"):
                        line = line[6:].strip()
                    if line.startswith("Guard ="):
                        line = line[7:].strip()
                    return line

        # Fallback: return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()

        return None

    def _check_correct(
        self,
        generated: Optional[str],
        expected: str,
        expected_operator: str,
    ) -> bool:
        """Check if generated guard is correct."""
        if not generated:
            return False

        generated = generated.strip().lower()
        expected = expected.strip().lower()

        # Exact match
        if generated == expected:
            return True

        # Normalize and check
        # Remove extra spaces
        generated = " ".join(generated.split())
        expected = " ".join(expected.split())

        if generated == expected:
            return True

        # Check operator is present and correct
        if expected_operator not in generated:
            return False

        # Check variable and value are present (order-independent)
        expected_parts = expected.replace(expected_operator, " ").split()
        generated_parts = generated.replace(expected_operator, " ").split()

        if len(expected_parts) >= 2 and len(generated_parts) >= 2:
            # Both parts should be in generated
            return all(p in generated for p in expected_parts)

        return False


def run_strategy_comparison(
    test_cases: list,
    model=None,
    tokenizer=None,
) -> dict:
    """Run all strategies on test cases and compare."""
    generator = ImprovedGenerator(model, tokenizer)
    results = {s: [] for s in PromptStrategy}

    for nl, expected_guard, expected_op in test_cases:
        for strategy in PromptStrategy:
            result = generator.generate(nl, expected_guard, expected_op, strategy)
            results[strategy].append(result)

    # Compute accuracy per strategy
    summary = {}
    for strategy, strategy_results in results.items():
        n = len(strategy_results)
        correct = sum(1 for r in strategy_results if r.is_correct)
        has_comparison = sum(1 for r in strategy_results if r.has_comparison)

        summary[strategy.value] = {
            "total": n,
            "correct": correct,
            "accuracy": correct / n * 100 if n > 0 else 0,
            "has_comparison": has_comparison,
            "comparison_rate": has_comparison / n * 100 if n > 0 else 0,
        }

    return summary, results


if __name__ == "__main__":
    print("Improved Generator Test")
    print("=" * 60)

    # Test prompts for each strategy
    test_nl = "count greater than 5"
    generator = ImprovedGenerator()

    for strategy in PromptStrategy:
        print(f"\n{strategy.value.upper()} PROMPT:")
        print("-" * 40)
        prompt = generator._build_prompt(test_nl, ">", strategy)
        print(prompt[:500])
        print("...")
