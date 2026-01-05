"""
Context Predictor: LLM few-shot prediction of final context values.

Given a statechart, initial context, and event sequence,
predict the final context values after execution.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class PredictionResult:
    """Result of context prediction."""
    predicted_context: Dict[str, Any]
    ground_truth: Dict[str, Any]
    is_valid_json: bool
    generation_time_ms: float
    raw_output: str = ""

    # Computed metrics
    exact_match: bool = False
    partial_matches: int = 0  # Number of variables correctly predicted
    total_variables: int = 0
    mae: float = 0.0  # Mean absolute error for numeric values


class ContextPredictor:
    """
    Predicts final context values using LLM few-shot prompting.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def predict(
        self,
        sc_json: Dict,
        initial_context: Dict[str, Any],
        events: List[str],
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        Predict final context after executing events.

        Args:
            sc_json: Statechart definition
            initial_context: Starting context values
            events: Event sequence to execute
            ground_truth: Expected final context (for evaluation)

        Returns:
            PredictionResult with predicted context
        """
        result = PredictionResult(
            predicted_context={},
            ground_truth=ground_truth or {},
            is_valid_json=False,
            generation_time_ms=0.0,
        )

        # Build prompt
        prompt = self._build_prompt(sc_json, initial_context, events)

        # Generate prediction
        start = time.time()
        output = self._generate(prompt)
        result.generation_time_ms = (time.time() - start) * 1000
        result.raw_output = output

        # Parse predicted context
        predicted = self._parse_context(output)
        if predicted is not None:
            result.predicted_context = predicted
            result.is_valid_json = True

        # Compute metrics if ground truth provided
        if ground_truth:
            result.total_variables = len(ground_truth)
            result = self._compute_metrics(result)

        return result

    def _build_prompt(
        self,
        sc_json: Dict,
        initial_context: Dict[str, Any],
        events: List[str],
    ) -> str:
        """Build few-shot prompt for context prediction."""

        # Extract transition info
        transitions_str = self._format_transitions(sc_json)

        # Format events
        events_str = " -> ".join(events)

        # Format initial context
        ctx_str = json.dumps(initial_context, separators=(',', ':'))

        # Few-shot examples
        few_shot = """Example 1:
Transitions: Start --(BEGIN, count=0)--> Counting, Counting --(TICK, count<5, count=count+1)--> Counting
Initial context: {"count":0,"target":5}
Events: BEGIN -> TICK -> TICK -> TICK
Final context: {"count":3,"target":5}

Example 2:
Transitions: Check --(EVALUATE, score>=50, result='pass')--> PassPath, PassPath --(CONTINUE, bonus=10)--> Merge
Initial context: {"score":75,"result":null,"bonus":0}
Events: EVALUATE -> CONTINUE
Final context: {"score":75,"result":"pass","bonus":10}

Example 3:
Transitions: Idle --(START, sum=0, index=0)--> Processing, Processing --(NEXT, index<3, sum=sum+values[index], index=index+1)--> Processing
Initial context: {"sum":0,"index":0,"values":[10,20,30]}
Events: START -> NEXT -> NEXT -> NEXT
Final context: {"sum":60,"index":3,"values":[10,20,30]}
"""

        prompt = f"""{few_shot}
Now predict the final context:

Transitions: {transitions_str}
Initial context: {ctx_str}
Events: {events_str}
Final context:"""

        return prompt

    def _format_transitions(self, sc_json: Dict) -> str:
        """Format transitions for prompt."""
        parts = []
        for t in sc_json.get("transitions", [])[:10]:  # Limit to 10
            from_s = ",".join(t.get("from", []))
            to_s = ",".join(t.get("to", []))
            event = t.get("event", "")

            details = []
            if "guard" in t:
                guard = t["guard"]
                expr = guard.get("expression", "") if isinstance(guard, dict) else guard
                if expr:
                    details.append(expr)

            for a in t.get("actions", []):
                expr = a.get("expression", "") if isinstance(a, dict) else a
                if expr:
                    details.append(expr)

            detail_str = ", ".join(details) if details else ""
            if detail_str:
                parts.append(f"{from_s} --({event}, {detail_str})--> {to_s}")
            else:
                parts.append(f"{from_s} --({event})--> {to_s}")

        return ", ".join(parts)

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)  # Low temp for deterministic output
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=200,
                sampler=sampler,
            )
            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return "{}"

    def _parse_context(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse context JSON from LLM output."""
        try:
            # Clean output - find JSON object
            text = output.strip()
            if '{' in text:
                start = text.find('{')
                depth = 0
                end = len(text)
                for i, c in enumerate(text[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                json_str = text[start:end]
                return json.loads(json_str)
            return None
        except json.JSONDecodeError:
            return None

    def _compute_metrics(self, result: PredictionResult) -> PredictionResult:
        """Compute accuracy metrics."""
        pred = result.predicted_context
        truth = result.ground_truth

        if not truth:
            return result

        # Exact match
        result.exact_match = pred == truth

        # Partial matches and MAE
        numeric_errors = []
        matches = 0

        for key, expected in truth.items():
            if key in pred:
                actual = pred[key]
                if actual == expected:
                    matches += 1
                elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    numeric_errors.append(abs(expected - actual))

        result.partial_matches = matches
        if numeric_errors:
            result.mae = sum(numeric_errors) / len(numeric_errors)

        return result


def predict_context(
    sc_json: Dict,
    initial_context: Dict[str, Any],
    events: List[str],
    model=None,
    tokenizer=None,
) -> PredictionResult:
    """Convenience function for context prediction."""
    predictor = ContextPredictor(model, tokenizer)
    return predictor.predict(sc_json, initial_context, events)
