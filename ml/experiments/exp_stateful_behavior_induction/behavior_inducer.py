"""
Behavior Inducer - Extract rules from I/O sequences using LLM.

Uses pattern generalization approach:
1. Identify state representation type
2. Extract transition rules per event
3. Generalize to abstract rules
4. Build predictive model
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from . import IOStep, InducedRule, InducedMachine


@dataclass
class InductionResult:
    """Result of rule induction."""
    machine: InducedMachine
    raw_output: str
    parse_success: bool
    rules_text: str


class BehaviorInducer:
    """Induces state machine rules from I/O sequences."""

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

    def _format_trace(self, trace: List[IOStep]) -> str:
        """Format trace for prompt."""
        lines = []
        for i, step in enumerate(trace):
            arg_str = f" {step.input_arg}" if step.input_arg is not None else ""
            lines.append(
                f"  Step {i+1}: ({step.input_event}{arg_str}, {step.state_before}) → {step.state_after}"
            )
        return "\n".join(lines)

    def _build_induction_prompt(self, traces: List[List[IOStep]]) -> str:
        """Build prompt for rule induction."""
        # Format all traces
        traces_text = ""
        for i, trace in enumerate(traces[:4]):  # Limit to 4 traces
            traces_text += f"\nTrace {i+1}:\n{self._format_trace(trace)}\n"

        prompt = f"""Analyze these I/O sequences and induce the transition rules.

Each step shows: (EVENT arg, state_before) → state_after

{traces_text}

TASK: Identify the pattern for each event type.

SCRATCHPAD:

Step 1 - Identify state type:
  What kind of data structure is the state? (list, int, tuple, etc.)

Step 2 - Group by event:
  List all observations for each event type.

Step 3 - Find pattern for each event:
  What transformation does each event perform on the state?

Step 4 - Generalize rules:
  Write abstract rules using variables (x for argument, s for state).

OUTPUT FORMAT:
State type: <type>
Events: [event1, event2, ...]
Rules:
  EVENT1: <pattern description> | <formula: s' = f(s, x)>
  EVENT2: <pattern description> | <formula: s' = f(s)>
  ...

Example output:
State type: list
Events: [PUSH, POP]
Rules:
  PUSH: append argument to list | s' = s + [x]
  POP: remove last element | s' = s[:-1]

Now analyze the traces above:"""

        return prompt

    def induce(
        self,
        traces: List[List[IOStep]],
        max_tokens: int = 500,
    ) -> InductionResult:
        """Induce rules from traces."""
        self.load_model()

        if not self.model or not self.tokenizer:
            return self._fallback_induce(traces)

        prompt = self._build_induction_prompt(traces)

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
        machine, parse_success = self._parse_output(output, traces)

        return InductionResult(
            machine=machine,
            raw_output=output,
            parse_success=parse_success,
            rules_text=self._extract_rules_text(output),
        )

    def _parse_output(
        self,
        output: str,
        traces: List[List[IOStep]],
    ) -> Tuple[InducedMachine, bool]:
        """Parse LLM output to extract induced machine."""
        # Extract state type
        state_type = "unknown"
        type_match = re.search(r'State type:\s*(\w+)', output, re.IGNORECASE)
        if type_match:
            state_type = type_match.group(1).lower()

        # Infer state type from traces if not found
        if state_type == "unknown" and traces and traces[0]:
            sample_state = traces[0][0].state_before
            if isinstance(sample_state, list):
                state_type = "list"
            elif isinstance(sample_state, int):
                state_type = "int"
            elif isinstance(sample_state, tuple):
                state_type = "tuple"

        # Extract events
        events = []
        events_match = re.search(r'Events:\s*\[(.*?)\]', output, re.IGNORECASE)
        if events_match:
            events_str = events_match.group(1)
            events = [e.strip().strip('"\'') for e in events_str.split(',')]
            events = [e for e in events if e]

        # If no events found, extract from traces
        if not events and traces:
            seen_events = set()
            for trace in traces:
                for step in trace:
                    seen_events.add(step.input_event)
            events = sorted(seen_events)

        # Extract rules
        rules = []
        rules_section = re.search(r'Rules:(.*?)(?:\n\n|$)', output, re.IGNORECASE | re.DOTALL)
        if rules_section:
            rules_text = rules_section.group(1)
            # Parse each rule line
            rule_lines = re.findall(r'(\w+):\s*(.+?)(?:\||$)', rules_text)
            for event, description in rule_lines:
                event = event.strip().upper()
                description = description.strip()
                if event and description:
                    rules.append(InducedRule(
                        event=event,
                        condition="always",
                        action=description,
                        examples=[],
                    ))

        # Get initial state from traces
        initial_state = None
        if traces and traces[0]:
            initial_state = traces[0][0].state_before

        parse_success = bool(events) and bool(rules)

        return InducedMachine(
            state_type=state_type,
            events=events,
            rules=rules,
            initial_state=initial_state,
        ), parse_success

    def _extract_rules_text(self, output: str) -> str:
        """Extract just the rules section from output."""
        rules_section = re.search(r'Rules:(.*?)(?:\n\n|OUTPUT|$)', output, re.IGNORECASE | re.DOTALL)
        if rules_section:
            return "Rules:" + rules_section.group(1).strip()
        return output[-200:]  # Last 200 chars as fallback

    def _fallback_induce(self, traces: List[List[IOStep]]) -> InductionResult:
        """Fallback to deterministic rule extraction."""
        # Analyze traces
        if not traces or not traces[0]:
            return InductionResult(
                machine=InducedMachine("unknown", [], [], None),
                raw_output="[fallback: empty traces]",
                parse_success=False,
                rules_text="",
            )

        sample_state = traces[0][0].state_before
        if isinstance(sample_state, list):
            state_type = "list"
        elif isinstance(sample_state, int):
            state_type = "int"
        elif isinstance(sample_state, tuple):
            state_type = "tuple"
        else:
            state_type = str(type(sample_state).__name__)

        # Extract events
        seen_events = set()
        for trace in traces:
            for step in trace:
                seen_events.add(step.input_event)
        events = sorted(seen_events)

        # Create simple rules based on observed patterns
        rules = []
        for event in events:
            # Collect examples for this event
            examples = []
            for trace in traces:
                for step in trace:
                    if step.input_event == event:
                        examples.append(step)

            if examples:
                # Describe based on observed changes
                rules.append(InducedRule(
                    event=event,
                    condition="always",
                    action=f"transforms state based on {len(examples)} examples",
                    examples=examples[:3],
                ))

        initial_state = traces[0][0].state_before

        return InductionResult(
            machine=InducedMachine(
                state_type=state_type,
                events=events,
                rules=rules,
                initial_state=initial_state,
            ),
            raw_output="[fallback: deterministic extraction]",
            parse_success=True,
            rules_text="\n".join([f"{r.event}: {r.action}" for r in rules]),
        )


def demo():
    """Demonstrate behavior induction."""
    from . import StackMachine, generate_trace

    print("=" * 60)
    print("Behavior Inducer Demo")
    print("=" * 60)

    machine = StackMachine()
    traces = [generate_trace(machine, 5, seed=i) for i in range(3)]

    print("\nTraces:")
    for i, trace in enumerate(traces):
        print(f"\nTrace {i+1}:")
        for step in trace:
            arg_str = f" {step.input_arg}" if step.input_arg is not None else ""
            print(f"  ({step.input_event}{arg_str}, {step.state_before}) → {step.state_after}")

    inducer = BehaviorInducer()
    result = inducer.induce(traces)

    print(f"\n--- Induced Machine ---")
    print(f"State type: {result.machine.state_type}")
    print(f"Events: {result.machine.events}")
    print(f"Rules:")
    for rule in result.machine.rules:
        print(f"  {rule.event}: {rule.action}")
    print(f"Parse success: {result.parse_success}")


if __name__ == "__main__":
    demo()
