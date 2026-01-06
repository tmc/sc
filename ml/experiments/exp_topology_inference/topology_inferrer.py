"""
Topology Inferrer - Infer SC topology from execution traces using LLM.

Uses step-by-step enumeration approach:
1. Extract unique states from all traces
2. Extract transitions (consecutive state pairs)
3. Identify patterns (self-loops, branching, cycles)
4. Determine initial state (first state in traces)
"""

import json
import re
from typing import Set, List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from . import InferredTopology, ground_truth_topology


@dataclass
class InferenceResult:
    """Result of topology inference."""
    topology: InferredTopology
    raw_output: str
    parse_success: bool
    method: str


class TopologyInferrer:
    """Infers SC topology from execution traces using LLM."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the LLM model."""
        if self.model is None and MLX_AVAILABLE:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded.")

    def _build_enumeration_prompt(self, traces: List[List[str]]) -> str:
        """Build prompt with step-by-step enumeration approach."""
        # Format traces
        traces_str = "\n".join([f"  Trace {i+1}: {trace}" for i, trace in enumerate(traces)])

        prompt = f"""Analyze execution traces and infer the state machine topology.

APPROACH - Follow these steps EXACTLY:

Step 1 - Extract unique states:
  Look at ALL states that appear in ANY trace.
  List them as a set.

Step 2 - Extract transitions:
  For each trace, find consecutive state pairs.
  Trace [A, B, C] has transitions: A→B, B→C
  Collect ALL unique transitions across all traces.

Step 3 - Identify patterns:
  - Self-loop: State transitions to itself (X→X)
  - Branching: State has multiple outgoing transitions
  - Cycle: Can return to initial state

Step 4 - Determine initial state:
  First state of the first trace is the initial state.

EXAMPLE:
Input traces:
  Trace 1: [A, B, C, A]
  Trace 2: [A, C, B, A]
  Trace 3: [A, B, B, C]

SCRATCHPAD:
Step 1 - States: {{A, B, C}}
Step 2 - Transitions:
  Trace 1: A→B, B→C, C→A
  Trace 2: A→C, C→B, B→A
  Trace 3: A→B, B→B, B→C
  Union: {{A→B, A→C, B→A, B→B, B→C, C→A, C→B}}
Step 3 - Patterns:
  B→B = self-loop on B
  A has A→B, A→C = branching from A
  C→A exists = cycle back to initial
Step 4 - Initial: A (first state of first trace)

OUTPUT:
States: [A, B, C]
Transitions: [A→B, A→C, B→A, B→B, B→C, C→A, C→B]
Initial: A
Features: [self-loop on B, branching from A, cycle]

---

Now analyze these traces:
{traces_str}

SCRATCHPAD:
Step 1 - States:"""

        return prompt

    def _build_simple_prompt(self, traces: List[List[str]]) -> str:
        """Build simpler prompt for direct inference."""
        traces_str = "\n".join([f"  {trace}" for trace in traces])

        prompt = f"""Given these execution traces, extract the state machine topology.

Traces:
{traces_str}

Extract:
1. All unique states that appear
2. All transitions (consecutive state pairs)
3. The initial state (first state of first trace)
4. Any patterns (self-loops, branching, cycles)

Format your answer as:
States: [state1, state2, ...]
Transitions: [A→B, B→C, ...]
Initial: <state>
Features: [pattern1, pattern2, ...]"""

        return prompt

    def infer(
        self,
        traces: List[List[str]],
        method: str = "enumeration",
        max_tokens: int = 400,
    ) -> InferenceResult:
        """Infer topology from traces."""
        self.load_model()

        if not self.model or not self.tokenizer:
            return self._fallback_infer(traces, method)

        # Select prompt
        if method == "enumeration":
            prompt = self._build_enumeration_prompt(traces)
        else:
            prompt = self._build_simple_prompt(traces)

        # Format for chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Parse output
        topology, parse_success = self._parse_output(output, traces)

        return InferenceResult(
            topology=topology,
            raw_output=output,
            parse_success=parse_success,
            method=method,
        )

    def _parse_output(
        self,
        output: str,
        traces: List[List[str]],
    ) -> Tuple[InferredTopology, bool]:
        """Parse LLM output to extract topology."""
        # Get all states mentioned in traces for validation
        all_trace_states = set()
        for trace in traces:
            all_trace_states.update(trace)

        states = set()
        transitions = set()
        initial = None
        features = []

        # Parse states
        states_match = re.search(r'States:\s*\[(.*?)\]', output, re.IGNORECASE)
        if states_match:
            states_str = states_match.group(1)
            # Extract state names (words, possibly quoted)
            state_names = re.findall(r'["\']?(\w+)["\']?', states_str)
            states = set(state_names)

        # Parse transitions
        trans_match = re.search(r'Transitions:\s*\[(.*?)\]', output, re.IGNORECASE | re.DOTALL)
        if trans_match:
            trans_str = trans_match.group(1)
            # Match patterns like A→B, A->B, A to B
            trans_patterns = re.findall(r'(\w+)\s*(?:→|->|to)\s*(\w+)', trans_str)
            transitions = set(trans_patterns)

        # Also look for transitions in scratchpad format
        scratchpad_trans = re.findall(r'(\w+)→(\w+)', output)
        transitions.update(scratchpad_trans)

        # Parse initial state
        initial_match = re.search(r'Initial:\s*(\w+)', output, re.IGNORECASE)
        if initial_match:
            initial = initial_match.group(1)

        # Parse features
        features_match = re.search(r'Features:\s*\[(.*?)\]', output, re.IGNORECASE | re.DOTALL)
        if features_match:
            features_str = features_match.group(1)
            # Split by comma and clean
            raw_features = [f.strip().strip('"\'') for f in features_str.split(',')]
            features = [f for f in raw_features if f]

        # Also detect features from text
        if re.search(r'self[- ]?loop', output, re.IGNORECASE):
            # Find which state has self-loop
            for s in states:
                if (s, s) in transitions:
                    feat = f"self-loop on {s}"
                    if feat not in features:
                        features.append(feat)

        if re.search(r'cycle|cyclic|returns? to', output, re.IGNORECASE):
            if "cycle" not in features:
                features.append("cycle")

        if re.search(r'branch', output, re.IGNORECASE):
            # Find branching states
            for s in states:
                outgoing = [t for t in transitions if t[0] == s]
                if len(outgoing) > 1:
                    targets = sorted([t[1] for t in outgoing])
                    feat = f"branch from {s} to {','.join(targets)}"
                    if feat not in features and "branch" not in " ".join(features):
                        features.append(feat)

        # Validation: states should contain all transition endpoints
        for src, tgt in transitions:
            states.add(src)
            states.add(tgt)

        # If no initial found, use first state of first trace
        if not initial and traces and traces[0]:
            initial = traces[0][0]

        # Determine parse success
        parse_success = bool(states) and bool(transitions or len(all_trace_states) == 1)

        return InferredTopology(
            states=states,
            transitions=transitions,
            initial_state=initial,
            features=sorted(features),
        ), parse_success

    def _fallback_infer(
        self,
        traces: List[List[str]],
        method: str,
    ) -> InferenceResult:
        """Fallback to deterministic inference."""
        topology = ground_truth_topology(traces)
        return InferenceResult(
            topology=topology,
            raw_output="[fallback: deterministic inference]",
            parse_success=True,
            method=method,
        )


def demo():
    """Demonstrate topology inference."""
    print("=" * 60)
    print("Topology Inferrer Demo")
    print("=" * 60)

    inferrer = TopologyInferrer()

    test_traces = [
        # Linear
        [["A", "B", "C", "D"], ["A", "B", "C"]],
        # Cycle
        [["X", "Y", "Z", "X", "Y"], ["X", "Y", "Z", "X"]],
        # Self-loop
        [["S", "S", "S", "T"], ["S", "T"]],
    ]

    for i, traces in enumerate(test_traces):
        print(f"\n--- Test {i+1} ---")
        print(f"Traces: {traces}")

        result = inferrer.infer(traces)
        print(f"States: {result.topology.states}")
        print(f"Transitions: {result.topology.transitions}")
        print(f"Initial: {result.topology.initial_state}")
        print(f"Features: {result.topology.features}")
        print(f"Parse success: {result.parse_success}")


if __name__ == "__main__":
    demo()
