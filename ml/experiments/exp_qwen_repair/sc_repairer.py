#!/usr/bin/env python3
"""
SC Repairer: LLM-based statechart repair using QwenCoder.

Uses Qwen2.5-Coder-0.5B-Instruct to fix invalid statecharts based on
validation errors and repair hints.

Strategy:
1. Try deterministic repairs first (fast, reliable)
2. Fall back to LLM for complex/novel errors
3. Validate repaired chart
4. Iterate if needed
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

from .error_analyzer import (
    ErrorType, ValidationError, ValidationResult, ErrorAnalyzer
)
from .repair_strategies import RepairStrategies, RepairPlan, RepairResult


@dataclass
class RepairConfig:
    """Configuration for LLM repair."""
    max_tokens: int = 512
    temperature: float = 0.3  # Lower for more deterministic
    top_p: float = 0.9
    max_iterations: int = 3
    use_deterministic_first: bool = True
    verbose: bool = True


@dataclass
class RepairAttempt:
    """Record of a single repair attempt."""
    iteration: int
    method: str  # "deterministic" or "llm"
    input_chart: Dict[str, Any]
    output_chart: Optional[Dict[str, Any]]
    errors_before: int
    errors_after: int
    success: bool
    duration: float
    raw_response: str = ""


@dataclass
class RepairSession:
    """Complete repair session for a chart."""
    original_chart: Dict[str, Any]
    final_chart: Optional[Dict[str, Any]] = None
    attempts: List[RepairAttempt] = field(default_factory=list)
    total_errors_fixed: int = 0
    final_valid: bool = False
    total_duration: float = 0.0

    @property
    def success(self) -> bool:
        return self.final_valid


class SCRepairer:
    """
    Statechart repair using QwenCoder LLM.

    Combines deterministic strategies with LLM for complex repairs.
    """

    SYSTEM_PROMPT = """You are an expert at fixing statechart definitions.
Given an invalid statechart JSON and validation errors, output ONLY the corrected JSON.

Statechart structure:
- root_state: Contains all states with label, type, children, is_initial
- transitions: Array of {from: [states], to: [states], event: string}
- State types: 1=BASIC, 2=NORMAL (compound), 3=PARALLEL

Common fixes:
- Add is_initial: true to first child of compound states
- Fix typos in state names to match existing states
- Remove duplicate state labels
- Add missing event fields to transitions

Output ONLY valid JSON. No explanations."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        sc_path: str = "./sc",
    ):
        """Initialize with QwenCoder model."""
        if MLX_AVAILABLE:
            print(f"Loading model: {model_name}")
            self.model, self.tokenizer = load(model_name)
            print(f"Model loaded. Vocab size: {len(self.tokenizer.get_vocab())}")
        else:
            print("MLX not available - using deterministic repair only")
            self.model = None
            self.tokenizer = None

        self.analyzer = ErrorAnalyzer(sc_path)
        self.strategies = RepairStrategies()

    def repair(
        self,
        chart: Dict[str, Any],
        config: Optional[RepairConfig] = None,
    ) -> RepairSession:
        """
        Repair an invalid statechart.

        Args:
            chart: Invalid statechart to repair
            config: Repair configuration

        Returns:
            RepairSession with repair history and result
        """
        config = config or RepairConfig()
        session = RepairSession(original_chart=chart)
        start_time = time.time()

        current_chart = chart
        current_result = self.analyzer.validate_chart(current_chart)

        if current_result.is_valid:
            session.final_chart = current_chart
            session.final_valid = True
            session.total_duration = time.time() - start_time
            return session

        for iteration in range(config.max_iterations):
            iter_start = time.time()

            if config.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
                print(f"Errors: {len(current_result.errors)}")

            # Try deterministic repair first
            if config.use_deterministic_first and iteration == 0:
                attempt = self._deterministic_repair(
                    current_chart, current_result, iteration
                )
            else:
                attempt = self._llm_repair(
                    current_chart, current_result, iteration, config
                )

            attempt.duration = time.time() - iter_start
            session.attempts.append(attempt)

            if attempt.output_chart:
                current_chart = attempt.output_chart
                current_result = self.analyzer.validate_chart(current_chart)

                if current_result.is_valid:
                    session.final_chart = current_chart
                    session.final_valid = True
                    session.total_errors_fixed = len(
                        self.analyzer.validate_chart(chart).errors
                    )
                    break

                if config.verbose:
                    print(f"Remaining errors: {len(current_result.errors)}")
                    for err in current_result.errors[:3]:
                        print(f"  - {err}")
            else:
                if config.verbose:
                    print("Repair failed to produce valid output")

        session.final_chart = current_chart
        session.total_duration = time.time() - start_time

        return session

    def _deterministic_repair(
        self,
        chart: Dict[str, Any],
        result: ValidationResult,
        iteration: int,
    ) -> RepairAttempt:
        """Apply deterministic repair strategies."""
        plan = self.strategies.apply_all(chart, result.errors)

        # Validate repaired chart
        repaired_result = self.analyzer.validate_chart(plan.repaired_chart)

        return RepairAttempt(
            iteration=iteration,
            method="deterministic",
            input_chart=chart,
            output_chart=plan.repaired_chart,
            errors_before=len(result.errors),
            errors_after=len(repaired_result.errors),
            success=repaired_result.is_valid,
            duration=0.0,
        )

    def _llm_repair(
        self,
        chart: Dict[str, Any],
        result: ValidationResult,
        iteration: int,
        config: RepairConfig,
    ) -> RepairAttempt:
        """Use LLM to repair the chart."""
        # Build prompt
        prompt = self._build_repair_prompt(chart, result)

        if config.verbose:
            print(f"LLM prompt length: {len(prompt)} chars")

        # Generate repair
        try:
            response = self._generate(prompt, config)
            repaired_chart = self._extract_json(response)

            if repaired_chart:
                repaired_result = self.analyzer.validate_chart(repaired_chart)
                return RepairAttempt(
                    iteration=iteration,
                    method="llm",
                    input_chart=chart,
                    output_chart=repaired_chart,
                    errors_before=len(result.errors),
                    errors_after=len(repaired_result.errors),
                    success=repaired_result.is_valid,
                    duration=0.0,
                    raw_response=response[:500],
                )
        except Exception as e:
            if config.verbose:
                print(f"LLM repair error: {e}")

        return RepairAttempt(
            iteration=iteration,
            method="llm",
            input_chart=chart,
            output_chart=None,
            errors_before=len(result.errors),
            errors_after=len(result.errors),
            success=False,
            duration=0.0,
            raw_response=str(e) if 'e' in dir() else "Unknown error",
        )

    def _build_repair_prompt(
        self,
        chart: Dict[str, Any],
        result: ValidationResult,
    ) -> str:
        """Build prompt for LLM repair."""
        # Get repair hints
        hints = self.analyzer.get_repair_hints(result.errors, chart)

        prompt = f"""{self.SYSTEM_PROMPT}

## Invalid Statechart
```json
{json.dumps(chart, indent=2)}
```

## Validation Errors
{chr(10).join(f'- {err}' for err in result.errors)}

## Repair Hints
{chr(10).join(f'- {hint}' for hint in hints)}

## Fixed Statechart (JSON only)
```json
"""
        return prompt

    def _generate(self, prompt: str, config: RepairConfig) -> str:
        """Generate response from LLM."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)

        # Generate
        generated_tokens = []

        for step in range(config.max_tokens):
            x = mx.array([input_ids + generated_tokens])
            logits = self.model(x)[:, -1, :]

            # Temperature
            if config.temperature > 0:
                logits = logits / config.temperature

            # Sample
            probs = mx.softmax(logits, axis=-1)

            # Top-p sampling
            if config.top_p < 1.0:
                sorted_indices = mx.argsort(-probs, axis=-1)
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
                cumsum = mx.cumsum(sorted_probs, axis=-1)

                cutoff_mask = cumsum <= config.top_p
                cutoff_mask = mx.concatenate([
                    mx.array([[True]]),
                    cutoff_mask[:, :-1]
                ], axis=-1)

                probs = mx.where(
                    mx.take_along_axis(cutoff_mask, mx.argsort(sorted_indices, axis=-1), axis=-1),
                    probs,
                    0.0
                )
                probs = probs / mx.sum(probs, axis=-1, keepdims=True)

            next_token = int(mx.random.categorical(mx.log(probs + 1e-10)))

            if next_token == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)

            # Check for end of JSON
            current_text = self.tokenizer.decode(generated_tokens)
            if current_text.count('{') > 0 and current_text.count('{') == current_text.count('}'):
                # Balanced braces - might be complete
                if current_text.rstrip().endswith('}'):
                    break

        return self.tokenizer.decode(generated_tokens)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Find JSON object
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object boundaries
        start = text.find('{')
        if start == -1:
            return None

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

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None


def repair_chart(
    chart: Dict[str, Any],
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    verbose: bool = True,
) -> RepairSession:
    """Convenience function to repair a chart."""
    repairer = SCRepairer(model_name)
    config = RepairConfig(verbose=verbose)
    return repairer.repair(chart, config)


def demo():
    """Demo statechart repair."""
    print("=" * 60)
    print("SC REPAIRER DEMO")
    print("=" * 60)

    repairer = SCRepairer()
    config = RepairConfig(verbose=True)

    # Test cases
    test_cases = [
        {
            "name": "Simple typo fix",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Idle", "is_initial": True},
                        {"label": "Running"},
                        {"label": "Done"},
                    ]
                },
                "transitions": [
                    {"from": ["Idel"], "to": ["Running"], "event": "START"},  # Typo
                    {"from": ["Running"], "to": ["Done"], "event": "FINISH"},
                ]
            }
        },
        {
            "name": "Missing initial + invalid target",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Start"},
                        {"label": "Middle"},
                        {"label": "End"},
                    ]
                },
                "transitions": [
                    {"from": ["Start"], "to": ["Midle"], "event": "GO"},  # Typo
                ]
            }
        },
        {
            "name": "Complex: multiple errors",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "A"},
                        {"label": "A"},  # Duplicate
                        {"label": "B"},
                    ]
                },
                "transitions": [
                    {"from": ["X"], "to": ["Y"], "event": "GO"},  # Both invalid
                    {"from": ["A"], "to": ["B"]},  # Missing event
                ]
            }
        },
    ]

    results = []

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print("=" * 60)

        # Initial validation
        initial_result = repairer.analyzer.validate_chart(tc["chart"])
        print(f"Initial errors: {len(initial_result.errors)}")
        for err in initial_result.errors:
            print(f"  - {err}")

        # Repair
        session = repairer.repair(tc["chart"], config)

        print(f"\nRepair result: {'SUCCESS' if session.success else 'FAILED'}")
        print(f"Iterations: {len(session.attempts)}")
        print(f"Duration: {session.total_duration:.2f}s")

        if session.final_chart:
            final_result = repairer.analyzer.validate_chart(session.final_chart)
            print(f"Final errors: {len(final_result.errors)}")

            if session.success:
                print("\nRepaired chart:")
                print(json.dumps(session.final_chart, indent=2)[:500])

        results.append({
            "name": tc["name"],
            "success": session.success,
            "initial_errors": len(initial_result.errors),
            "final_errors": len(final_result.errors) if session.final_chart else -1,
            "iterations": len(session.attempts),
            "duration": session.total_duration,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r["success"])
    print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['initial_errors']}→{r['final_errors']} errors, {r['duration']:.2f}s")

    print("\n" + "=" * 60)
    print("SC REPAIRER DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    demo()
