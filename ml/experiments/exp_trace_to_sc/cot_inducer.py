"""
Chain-of-Thought Statechart Inducer

Uses explicit reasoning steps to improve trace-to-SC induction:
1. Identify unique states from trace
2. Identify transitions between states
3. Determine initial state
4. Generate SC JSON

This addresses the problem of models copying example structure
instead of reasoning about the actual traces.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any

from .trace_analyzer import TraceAnalyzer, TraceAnalysis


def normalize_sc_schema(raw: Dict) -> Dict:
    """Normalize different JSON schemas to expected format."""
    if not isinstance(raw, dict):
        return raw

    # Already in expected format
    if "root_state" in raw:
        root = raw.get("root_state")
        if isinstance(root, dict) and "children" in root:
            if "label" not in root:
                root["label"] = "__root__"
            if "type" not in root:
                root["type"] = 2
            if "transitions" not in raw:
                raw["transitions"] = []
            return raw

    # Alternate format: {states: [...], transitions: [...]}
    if "states" in raw:
        children = []
        for s in raw["states"]:
            if not isinstance(s, dict):
                continue
            name = s.get("name") or s.get("label") or s.get("id", "S")
            is_initial = s.get("initial") or s.get("is_initial", False)
            children.append({
                "label": name,
                "type": 1,
                "is_initial": is_initial,
            })

        transitions = []
        for t in raw.get("transitions", []):
            if not isinstance(t, dict):
                continue
            src = t.get("source") or t.get("from")
            tgt = t.get("target") or t.get("to")
            event = t.get("event") or t.get("trigger", "")

            # Handle list or string
            if isinstance(src, str):
                src = [src]
            if isinstance(tgt, str):
                tgt = [tgt]
            if src is None:
                src = []
            if tgt is None:
                tgt = []

            transitions.append({
                "from": src,
                "to": tgt,
                "event": event,
            })

        return {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": children,
            },
            "transitions": transitions,
        }

    return raw


@dataclass
class CoTInductionResult:
    """Result of CoT-based induction."""
    induced_sc: Optional[Dict] = None
    is_valid_json: bool = False
    num_states: int = 0
    num_transitions: int = 0
    generation_time_ms: float = 0.0
    scratchpad: str = ""  # Intermediate reasoning

    # Accuracy metrics
    states_accuracy: float = 0.0
    transitions_accuracy: float = 0.0


class CoTInducer:
    """
    Chain-of-Thought Statechart Inducer.

    Forces the model to reason step-by-step before generating JSON.
    """

    def __init__(self, model=None, tokenizer=None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.trace_analyzer = TraceAnalyzer()

    def induce(
        self,
        traces: List[List[str]],
        use_analysis_hint: bool = True,
    ) -> CoTInductionResult:
        """
        Induce statechart using chain-of-thought.

        Args:
            traces: List of event sequences
            use_analysis_hint: Include trace analysis in prompt

        Returns:
            CoTInductionResult with induced statechart
        """
        result = CoTInductionResult()

        # Analyze traces first
        analysis = self.trace_analyzer.analyze(traces)

        # Build CoT prompt
        prompt = self._build_cot_prompt(traces, analysis, use_analysis_hint)

        # Generate
        start = time.time()
        output = self._generate(prompt)
        result.generation_time_ms = (time.time() - start) * 1000

        # Extract scratchpad and JSON
        result.scratchpad = self._extract_scratchpad(output)
        sc = self._parse_statechart(output)

        if sc:
            result.induced_sc = sc
            result.is_valid_json = True
            result.num_states = self._count_states(sc)
            result.num_transitions = len(sc.get("transitions", []))

        return result

    def _build_cot_prompt(
        self,
        traces: List[List[str]],
        analysis: TraceAnalysis,
        use_analysis_hint: bool,
    ) -> str:
        """Build chain-of-thought prompt."""

        # Format traces
        trace_strs = []
        for i, trace in enumerate(traces[:5]):
            trace_strs.append(f"  [{', '.join(trace)}]")
        traces_formatted = "\n".join(trace_strs)

        # Get analysis
        analysis_str = self.trace_analyzer.format_analysis(analysis)

        # Build states list from analysis
        states_list = sorted(analysis.unique_events)
        initial = analysis.most_likely_initial or states_list[0] if states_list else "S0"

        # Build transitions list
        trans_list = []
        for src, tgt in sorted(analysis.transitions):
            trans_list.append(f'{{"from": ["{src}"], "to": ["{tgt}"], "event": "{src}"}}')

        prompt = f"""Induce a statechart from these event traces.

TRACES:
{traces_formatted}

ANALYSIS:
{analysis_str}

Based on this analysis, generate the statechart JSON.
States are named after the events that enter them.
Initial state: {initial}

JSON:"""

        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)  # Lower temp for structured output
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=500,
                sampler=sampler,
            )

            if self.verbose:
                print(f"[CoT] Generated: {output[:200]}...")

            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return "{}"

    def _extract_scratchpad(self, output: str) -> str:
        """Extract reasoning steps before JSON."""
        if "```json" in output:
            return output.split("```json")[0]
        elif "{" in output:
            return output[:output.find("{")]
        return ""

    def _parse_statechart(self, output: str) -> Optional[Dict]:
        """Parse statechart from LLM output."""
        try:
            json_str = output.strip()

            # Find first { in output
            if '{' not in json_str:
                if self.verbose:
                    print("[CoT] No JSON found in output")
                return None

            start = json_str.find('{')

            # Balance braces to find complete JSON
            depth = 0
            end = len(json_str)
            in_string = False
            escape = False

            for i, c in enumerate(json_str[start:], start):
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            json_str = json_str[start:end]
            raw = json.loads(json_str)

            # Normalize to expected schema
            return normalize_sc_schema(raw)

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[CoT] JSON parse error: {e}")
            return None

    def _count_states(self, sc: Dict) -> int:
        """Count states in statechart."""
        count = 0

        def recurse(state):
            nonlocal count
            if not isinstance(state, dict):
                return
            if state.get("label") and state.get("label") != "__root__":
                count += 1
            for child in state.get("children", []):
                recurse(child)

        if "root_state" in sc:
            recurse(sc["root_state"])
        return count


class DirectInducer:
    """
    Direct prompting baseline (no CoT).

    Uses the original approach for comparison.
    """

    def __init__(self, model=None, tokenizer=None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

    def induce(self, traces: List[List[str]]) -> CoTInductionResult:
        """Induce statechart using direct prompting."""
        result = CoTInductionResult()

        # Build direct prompt (similar to original)
        prompt = self._build_direct_prompt(traces)

        # Generate
        start = time.time()
        output = self._generate(prompt)
        result.generation_time_ms = (time.time() - start) * 1000

        # Parse
        sc = self._parse_statechart(output)
        if sc:
            result.induced_sc = sc
            result.is_valid_json = True
            result.num_states = self._count_states(sc)
            result.num_transitions = len(sc.get("transitions", []))

        return result

    def _build_direct_prompt(self, traces: List[List[str]]) -> str:
        """Build direct prompt without CoT."""
        trace_strs = []
        for i, trace in enumerate(traces[:5]):
            trace_strs.append(f"  [{', '.join(trace)}]")
        traces_formatted = "\n".join(trace_strs)

        events = set(e for t in traces for e in t)
        events_str = ", ".join(sorted(events))

        prompt = f"""Induce a statechart from these event traces.

TRACES:
{traces_formatted}

Events observed: {events_str}

Generate the statechart as JSON with:
- root_state with children (each state has label, type=1, and is_initial for the start state)
- transitions with from, to, event

JSON:"""

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
        """Parse statechart from output."""
        try:
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
            raw = json.loads(json_str)
            return normalize_sc_schema(raw)
        except json.JSONDecodeError:
            return None

    def _count_states(self, sc: Dict) -> int:
        """Count states."""
        count = 0

        def recurse(state):
            nonlocal count
            if not isinstance(state, dict):
                return
            if state.get("label") and state.get("label") != "__root__":
                count += 1
            for child in state.get("children", []):
                recurse(child)

        if "root_state" in sc and isinstance(sc["root_state"], dict):
            recurse(sc["root_state"])
        return count


if __name__ == "__main__":
    print("CoT Inducer Test")
    print("=" * 60)

    # Test trace analysis formatting
    from .trace_analyzer import format_trace_analysis

    traces = [
        ["TURN_ON", "TURN_OFF", "TURN_ON"],
        ["TURN_ON", "TURN_OFF"],
    ]

    print("\nTrace analysis:")
    print(format_trace_analysis(traces))

    # Test without model
    print("\nCoT Induction (no model):")
    inducer = CoTInducer(verbose=True)
    result = inducer.induce(traces)
    print(f"  Valid: {result.is_valid_json}")
    print(f"  States: {result.num_states}")
