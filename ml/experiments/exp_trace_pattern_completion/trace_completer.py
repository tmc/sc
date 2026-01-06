"""
Trace Completer: Complete partial traces using learned patterns.

Given:
- Example traces showing a pattern
- A partial trace to complete

Predicts the next event(s) in the sequence.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .pattern_learner import PatternLearner, PatternAnalysis, Pattern


@dataclass
class CompletionResult:
    """Result of trace completion."""
    partial_trace: List[str] = field(default_factory=list)
    predicted_next: List[str] = field(default_factory=list)  # Next N events
    confidence: float = 0.0
    pattern_used: Optional[Pattern] = None
    scratchpad: str = ""
    generation_time_ms: float = 0.0


class TraceCompleter:
    """
    Completes partial traces using learned patterns.
    """

    def __init__(self, model=None, tokenizer=None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.learner = PatternLearner(verbose=verbose)

    def complete(
        self,
        example_traces: List[List[str]],
        partial_trace: List[str],
        num_predictions: int = 1,
    ) -> CompletionResult:
        """
        Complete a partial trace based on example patterns.

        Args:
            example_traces: Example traces showing the pattern
            partial_trace: Partial trace to complete
            num_predictions: Number of next events to predict

        Returns:
            CompletionResult with predictions
        """
        result = CompletionResult(partial_trace=partial_trace)
        start = time.time()
        scratchpad_lines = []

        # Learn pattern from examples
        analysis = self.learner.learn(example_traces)
        pattern = analysis.primary_pattern

        if not pattern:
            result.generation_time_ms = (time.time() - start) * 1000
            return result

        result.pattern_used = pattern
        scratchpad_lines.append(f"Pattern type: {pattern.pattern_type}")
        scratchpad_lines.append(f"Pattern sequence: {pattern.sequence}")
        scratchpad_lines.append(f"Transitions: {pattern.transitions}")

        # Complete using pattern
        if pattern.pattern_type in ("cycle", "alternating"):
            predictions = self._complete_cycle(partial_trace, pattern, num_predictions)
            scratchpad_lines.append(f"\nCycle completion:")
            scratchpad_lines.append(f"  Partial: {partial_trace}")
            scratchpad_lines.append(f"  Position in cycle: {self._find_position(partial_trace, pattern)}")
            scratchpad_lines.append(f"  Predicted: {predictions}")

        elif pattern.pattern_type == "growth":
            predictions = self._complete_growth(partial_trace, pattern, num_predictions)
            scratchpad_lines.append(f"\nGrowth completion:")
            scratchpad_lines.append(f"  Predicted: {predictions}")

        else:
            predictions = self._complete_transition_map(partial_trace, pattern, num_predictions)
            scratchpad_lines.append(f"\nTransition map completion:")
            scratchpad_lines.append(f"  Predicted: {predictions}")

        result.predicted_next = predictions
        result.confidence = pattern.confidence if predictions else 0.0
        result.scratchpad = "\n".join(scratchpad_lines)
        result.generation_time_ms = (time.time() - start) * 1000

        return result

    def complete_with_llm(
        self,
        example_traces: List[List[str]],
        partial_trace: List[str],
        num_predictions: int = 1,
    ) -> CompletionResult:
        """Complete using LLM with CoT prompting."""
        result = CompletionResult(partial_trace=partial_trace)
        start = time.time()

        # First learn pattern algorithmically
        analysis = self.learner.learn(example_traces)

        # Build prompt
        prompt = self._build_prompt(example_traces, partial_trace, analysis, num_predictions)

        # Generate
        output = self._generate(prompt)

        # Parse predictions
        predictions = self._parse_predictions(output, num_predictions)
        result.predicted_next = predictions
        result.pattern_used = analysis.primary_pattern
        result.confidence = 0.8 if predictions else 0.0
        result.scratchpad = output[:500]
        result.generation_time_ms = (time.time() - start) * 1000

        return result

    def _find_position(self, partial: List[str], pattern: Pattern) -> int:
        """Find position in cycle for last element."""
        if not partial or not pattern.sequence:
            return -1

        last = partial[-1]
        if last in pattern.sequence:
            return pattern.sequence.index(last)
        return -1

    def _complete_cycle(
        self,
        partial: List[str],
        pattern: Pattern,
        num_predictions: int
    ) -> List[str]:
        """Complete using cycle pattern."""
        if not partial:
            return pattern.sequence[:num_predictions] if pattern.sequence else []

        predictions = []
        current = partial[-1]

        for _ in range(num_predictions):
            if current in pattern.transitions:
                next_event = pattern.transitions[current]
                predictions.append(next_event)
                current = next_event
            else:
                # Try to find position in sequence
                pos = self._find_position([current], pattern)
                if pos >= 0:
                    next_pos = (pos + 1) % len(pattern.sequence)
                    next_event = pattern.sequence[next_pos]
                    predictions.append(next_event)
                    current = next_event
                else:
                    break

        return predictions

    def _complete_growth(
        self,
        partial: List[str],
        pattern: Pattern,
        num_predictions: int
    ) -> List[str]:
        """Complete using growth pattern."""
        # Growth: next element extends the sequence
        predictions = []
        current = partial[-1] if partial else None

        if current and current in pattern.transitions:
            for _ in range(num_predictions):
                if current in pattern.transitions:
                    next_event = pattern.transitions[current]
                    predictions.append(next_event)
                    current = next_event
                else:
                    break

        return predictions

    def _complete_transition_map(
        self,
        partial: List[str],
        pattern: Pattern,
        num_predictions: int
    ) -> List[str]:
        """Complete using transition map."""
        predictions = []
        current = partial[-1] if partial else None

        for _ in range(num_predictions):
            if current and current in pattern.transitions:
                next_event = pattern.transitions[current]
                predictions.append(next_event)
                current = next_event
            else:
                break

        return predictions

    def _build_prompt(
        self,
        examples: List[List[str]],
        partial: List[str],
        analysis: PatternAnalysis,
        num_predictions: int
    ) -> str:
        """Build CoT prompt for LLM completion."""
        examples_str = "\n".join([
            f"  {i+1}. [{', '.join(t)}]"
            for i, t in enumerate(examples[:5])
        ])

        partial_str = ", ".join(partial)
        pattern = analysis.primary_pattern

        pattern_hint = ""
        if pattern:
            pattern_hint = f"""
Pattern detected: {pattern.pattern_type}
Sequence: {pattern.sequence}
Transitions: {pattern.transitions}
"""

        prompt = f"""Complete the partial trace based on the pattern in example traces.

EXAMPLE TRACES:
{examples_str}

{pattern_hint}

PARTIAL TRACE TO COMPLETE:
[{partial_str}, ?]

SCRATCHPAD:
1. Identify the pattern type
2. Find position of last element in pattern
3. Predict next {num_predictions} element(s)

ANSWER (just the next {num_predictions} event(s), comma-separated):"""

        return prompt

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

            if self.verbose:
                print(f"[Completer] Generated: {output[:100]}...")

            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return ""

    def _parse_predictions(self, output: str, num_predictions: int) -> List[str]:
        """Parse predictions from LLM output."""
        predictions = []

        # Clean output
        output = output.strip()

        # Try to extract events
        # Look for comma-separated values
        parts = output.replace("[", "").replace("]", "").split(",")
        for part in parts[:num_predictions]:
            event = part.strip().strip("'\"")
            if event and len(event) < 20:  # Reasonable event name
                predictions.append(event)

        return predictions


def complete_trace(
    examples: List[List[str]],
    partial: List[str],
    num_predictions: int = 1
) -> CompletionResult:
    """Convenience function for trace completion."""
    completer = TraceCompleter()
    return completer.complete(examples, partial, num_predictions)


if __name__ == "__main__":
    print("Trace Completer Test")
    print("=" * 60)

    # Test 1: Complete cycle
    examples1 = [
        ["A", "B", "C", "A", "B", "C"],
        ["A", "B", "C", "A", "B", "C", "A"],
        ["A", "B", "C", "A"],
    ]
    partial1 = ["A", "B", "C", "A", "B"]

    print("\n1. Cycle completion:")
    print(f"  Examples: {examples1}")
    print(f"  Partial: {partial1}")
    result1 = complete_trace(examples1, partial1, num_predictions=1)
    print(f"  Predicted: {result1.predicted_next}")
    print(f"  Expected: ['C']")

    # Test 2: Complete alternating
    examples2 = [
        ["ON", "OFF", "ON", "OFF", "ON"],
        ["ON", "OFF", "ON", "OFF"],
    ]
    partial2 = ["ON", "OFF", "ON"]

    print("\n2. Alternating completion:")
    print(f"  Partial: {partial2}")
    result2 = complete_trace(examples2, partial2, num_predictions=1)
    print(f"  Predicted: {result2.predicted_next}")
    print(f"  Expected: ['OFF']")

    # Test 3: Multi-step prediction
    print("\n3. Multi-step prediction (3 steps):")
    result3 = complete_trace(examples1, ["A"], num_predictions=3)
    print(f"  Partial: ['A']")
    print(f"  Predicted: {result3.predicted_next}")
    print(f"  Expected: ['B', 'C', 'A']")
