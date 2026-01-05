"""
Trace Continuer: Predict next N states given partial trace.

Uses LLM to predict continuation of state machine execution traces.
"""

import json
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_execution_prediction import (
    CounterMachine,
    HierarchicalMachine,
    BranchingMachine,
    ParallelCounterMachine,
    TraceExecutor,
    execute_to_completion,
)


@dataclass
class ContinuationResult:
    """Result of trace continuation prediction."""
    partial_trace: List[str]
    predicted_states: List[str]
    actual_states: List[str]
    n_requested: int
    n_correct: int
    events_used: List[str]
    raw_output: str

    @property
    def accuracy(self) -> float:
        if not self.actual_states:
            return 0.0
        return self.n_correct / len(self.actual_states)

    @property
    def first_correct(self) -> bool:
        if not self.predicted_states or not self.actual_states:
            return False
        return self.predicted_states[0] == self.actual_states[0]


def create_continuation_prompt(
    sc_json: Dict,
    partial_trace: List[str],
    n_steps: int,
) -> str:
    """Create prompt for trace continuation."""
    # Extract state labels and transitions for context
    states = []
    transitions_info = []

    def collect_states(state, depth=0):
        label = state.get("label", "")
        if label and label != "__root__":
            states.append(label)
        for child in state.get("children", []):
            collect_states(child, depth + 1)

    if "root_state" in sc_json:
        collect_states(sc_json["root_state"])

    for t in sc_json.get("transitions", []):
        from_s = t.get("from", [])
        to_s = t.get("to", [])
        event = t.get("event", "")
        guard = t.get("guard")
        guard_str = ""
        if guard:
            guard_str = f" [guard: {guard.get('expression', guard) if isinstance(guard, dict) else guard}]"
        transitions_info.append(f"{from_s} --{event}--> {to_s}{guard_str}")

    prompt = f"""Given a state machine and partial execution trace, predict the next {n_steps} state(s).

STATE MACHINE:
States: {', '.join(states)}
Transitions:
{chr(10).join('  ' + t for t in transitions_info)}

PARTIAL TRACE (states visited so far):
{' -> '.join(partial_trace)}

Predict the next {n_steps} state(s) that will be visited.
Consider the transition structure and which transitions are available from the current state.

Output format: List only the state names, one per line.
Example:
StateA
StateB
StateC

Next {n_steps} state(s):
"""
    return prompt


def parse_continuation_output(output: str, valid_states: List[str]) -> List[str]:
    """Parse LLM output to extract predicted states."""
    predicted = []

    # Try to extract state names from output
    lines = output.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes
        line = re.sub(r'^[\d]+[.\)]\s*', '', line)  # Remove numbering
        line = re.sub(r'^-\s*', '', line)  # Remove bullet
        line = re.sub(r'^State:\s*', '', line, flags=re.IGNORECASE)

        # Check if this is a valid state
        for state in valid_states:
            if state.lower() == line.lower() or state in line:
                predicted.append(state)
                break

    return predicted


class TraceContinuer:
    """Predicts trace continuations using LLM."""

    def __init__(self, model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model if not already loaded."""
        if self.model is None:
            from mlx_lm import load
            print(f"Loading model: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path)
            print("Model loaded.")

    def predict_next_states(
        self,
        sc_json: Dict,
        partial_trace: List[str],
        n_steps: int = 1,
        max_tokens: int = 100,
    ) -> ContinuationResult:
        """Predict next N states given partial trace."""
        self.load_model()
        from mlx_lm import generate

        # Create prompt
        prompt = create_continuation_prompt(sc_json, partial_trace, n_steps)

        # Generate
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Extract valid states
        valid_states = []
        def collect_states(state):
            label = state.get("label", "")
            if label and label != "__root__":
                valid_states.append(label)
            for child in state.get("children", []):
                collect_states(child)

        if "root_state" in sc_json:
            collect_states(sc_json["root_state"])

        # Parse output
        predicted = parse_continuation_output(output, valid_states)

        return ContinuationResult(
            partial_trace=partial_trace,
            predicted_states=predicted[:n_steps],
            actual_states=[],  # Filled in by benchmark
            n_requested=n_steps,
            n_correct=0,  # Filled in by benchmark
            events_used=[],
            raw_output=output,
        )


def generate_continuation_dataset(
    n_samples: int = 20,
    trace_length_range: Tuple[int, int] = (3, 8),
    continuation_lengths: List[int] = [1, 3, 5],
) -> List[Dict]:
    """Generate dataset of continuation tasks."""
    dataset = []

    # Template generators
    templates = [
        lambda: CounterMachine(target=random.randint(3, 7)),
        lambda: HierarchicalMachine(),
        lambda: BranchingMachine(threshold=random.randint(30, 70)),
    ]

    # Event sequences for each template type
    event_sequences = {
        "CounterMachine": lambda n: ["BEGIN"] + ["TICK"] * n,
        "HierarchicalMachine": lambda n: ["START"] + ["WORK"] * min(n, 4) + ["ADVANCE", "BEGIN", "VALIDATE", "FINISH"][:max(0, n-4)],
        "BranchingMachine": lambda n: ["EVALUATE", "CONTINUE", "FINISH"][:n],
    }

    for i in range(n_samples):
        # Pick random template
        template = random.choice(templates)()
        sc_json = template.to_json()
        name = template.name

        # Generate full trace
        event_gen = event_sequences.get(name, lambda n: ["EVENT"] * n)
        events = event_gen(10)

        # Execute to get actual trace
        result = execute_to_completion(sc_json, events, template.initial_context)

        if not result.steps:
            continue

        # Extract state trace
        full_trace = [list(result.initial_config)[0] if result.initial_config else "Start"]
        for step in result.steps:
            # Get primary state (non-composite)
            target = list(step.target_config)
            # Prefer leaf states
            for s in target:
                if not any(s in t for t in target if t != s):
                    full_trace.append(s)
                    break

        if len(full_trace) < 4:
            continue

        # Create continuation tasks for different lengths
        for n_continue in continuation_lengths:
            # Pick random split point
            min_prefix = 2
            max_prefix = len(full_trace) - n_continue
            if max_prefix <= min_prefix:
                continue

            split_idx = random.randint(min_prefix, max_prefix)
            partial = full_trace[:split_idx]
            continuation = full_trace[split_idx:split_idx + n_continue]

            if len(continuation) < n_continue:
                continue

            dataset.append({
                "sc_json": sc_json,
                "partial_trace": partial,
                "actual_continuation": continuation,
                "n_steps": n_continue,
                "template_name": name,
                "full_trace": full_trace,
            })

    return dataset


if __name__ == "__main__":
    print("Trace Continuer Test")
    print("=" * 60)

    # Generate test dataset
    dataset = generate_continuation_dataset(n_samples=5)
    print(f"Generated {len(dataset)} continuation tasks")

    for i, task in enumerate(dataset[:3]):
        print(f"\nTask {i+1} ({task['template_name']}):")
        print(f"  Partial: {' -> '.join(task['partial_trace'])}")
        print(f"  Actual next {task['n_steps']}: {task['actual_continuation']}")
