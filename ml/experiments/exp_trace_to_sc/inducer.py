"""
Statechart Inducer from Traces

Given observed event traces, induce the underlying statechart structure.
Uses LLM to infer states and transitions from trace patterns.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class InductionResult:
    """Result of statechart induction."""
    induced_sc: Optional[Dict] = None
    is_valid_json: bool = False
    num_states: int = 0
    num_transitions: int = 0
    events_covered: Set[str] = field(default_factory=set)
    generation_time_ms: float = 0.0

    # Accuracy metrics (if ground truth available)
    states_correct: float = 0.0  # Fraction of states correctly identified
    transitions_correct: float = 0.0  # Fraction of transitions correct
    overall_accuracy: float = 0.0


class SCInducer:
    """
    Induces statecharts from observed traces using LLM.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def induce(
        self,
        traces: List[List[str]],
        hints: Optional[Dict] = None,
    ) -> InductionResult:
        """
        Induce statechart from traces.

        Args:
            traces: List of event sequences
            hints: Optional hints (e.g., expected state count)

        Returns:
            InductionResult with induced statechart
        """
        result = InductionResult()

        # Build prompt
        prompt = self._build_prompt(traces, hints)

        # Generate with LLM
        start = time.time()
        output = self._generate(prompt)
        result.generation_time_ms = (time.time() - start) * 1000

        # Parse output
        sc = self._parse_statechart(output)
        if sc:
            result.induced_sc = sc
            result.is_valid_json = True
            result.num_states = self._count_states(sc)
            result.num_transitions = len(sc.get("transitions", []))
            result.events_covered = self._extract_events(sc)

        return result

    def _build_prompt(
        self,
        traces: List[List[str]],
        hints: Optional[Dict],
    ) -> str:
        """Build prompt for induction."""
        # Format traces
        trace_strs = []
        for i, trace in enumerate(traces[:5]):  # Limit to 5 traces
            trace_strs.append(f"  Trace {i+1}: {' -> '.join(trace)}")
        traces_formatted = "\n".join(trace_strs)

        # Extract unique events
        all_events = set()
        for trace in traces:
            all_events.update(trace)
        events_str = ", ".join(sorted(all_events))

        # Hints
        hint_str = ""
        if hints:
            if "num_states" in hints:
                hint_str += f"\nHint: approximately {hints['num_states']} states."

        prompt = f"""Analyze these event traces and induce the statechart:

{traces_formatted}

Events: {events_str}
{hint_str}

Infer meaningful state names based on events. Output valid JSON:
{{"root_state": {{"label": "__root__", "type": 2, "children": [{{"label": "Off", "type": 1, "is_initial": true}}, {{"label": "On", "type": 1}}]}}, "transitions": [{{"from": ["Off"], "to": ["On"], "event": "TURN_ON"}}, {{"from": ["On"], "to": ["Off"], "event": "TURN_OFF"}}]}}

Induced statechart:"""

        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.2)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=600,
                sampler=sampler,
            )
            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return "{}"

    def _parse_statechart(self, output: str) -> Optional[Dict]:
        """Parse statechart from LLM output."""
        try:
            # Find JSON object
            json_str = output.strip()
            if '{' in json_str:
                start = json_str.find('{')
                depth = 0
                end = len(json_str)
                for i, c in enumerate(json_str[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                json_str = json_str[start:end]

            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _count_states(self, sc: Dict) -> int:
        """Count states in statechart."""
        count = 0
        def recurse(state):
            nonlocal count
            if state.get("label") and state.get("label") != "__root__":
                count += 1
            for child in state.get("children", []):
                recurse(child)

        if "root_state" in sc:
            recurse(sc["root_state"])
        return count

    def _extract_events(self, sc: Dict) -> Set[str]:
        """Extract events from transitions."""
        events = set()
        for trans in sc.get("transitions", []):
            if "event" in trans:
                events.add(trans["event"])
        return events


def induce_statechart(
    traces: List[List[str]],
    model=None,
    tokenizer=None,
) -> InductionResult:
    """
    Convenience function to induce statechart from traces.
    """
    inducer = SCInducer(model, tokenizer)
    return inducer.induce(traces)


def compare_statecharts(induced: Dict, ground_truth: Dict) -> Tuple[float, float]:
    """
    Compare induced statechart to ground truth.

    Returns:
        (states_accuracy, transitions_accuracy)
    """
    # Extract states from both
    def get_states(sc):
        states = set()
        def recurse(state):
            label = state.get("label", "")
            if label and label != "__root__":
                states.add(label.lower())
            for child in state.get("children", []):
                recurse(child)
        if "root_state" in sc:
            recurse(sc["root_state"])
        return states

    induced_states = get_states(induced)
    truth_states = get_states(ground_truth)

    # State accuracy: Jaccard similarity
    if not truth_states:
        states_acc = 1.0 if not induced_states else 0.0
    else:
        intersection = induced_states & truth_states
        union = induced_states | truth_states
        states_acc = len(intersection) / len(union) if union else 0.0

    # Extract transitions
    def get_transitions(sc):
        trans = set()
        for t in sc.get("transitions", []):
            froms = tuple(sorted(s.lower() for s in t.get("from", [])))
            tos = tuple(sorted(s.lower() for s in t.get("to", [])))
            event = t.get("event", "").lower()
            trans.add((froms, tos, event))
        return trans

    induced_trans = get_transitions(induced)
    truth_trans = get_transitions(ground_truth)

    # Transition accuracy
    if not truth_trans:
        trans_acc = 1.0 if not induced_trans else 0.0
    else:
        intersection = induced_trans & truth_trans
        trans_acc = len(intersection) / len(truth_trans)

    return states_acc, trans_acc
