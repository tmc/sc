"""
SC Code Generator - Generate Python code from statechart definitions.

Uses Qwen-1.5B with few-shot examples for code generation.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
from mlx_lm import load, generate


# Few-shot examples for training the model
FEW_SHOT_EXAMPLES = [
    # Example 1: Simple toggle
    {
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
            ]
        },
        "code": '''from enum import Enum, auto

class State(Enum):
    Off = auto()
    On = auto()

class Event(Enum):
    TOGGLE = auto()

class Toggle:
    def __init__(self):
        self.state = State.Off

    def send(self, event: Event) -> bool:
        if event == Event.TOGGLE:
            if self.state == State.Off:
                self.state = State.On
                return True
            elif self.state == State.On:
                self.state = State.Off
                return True
        return False

    def get_state(self) -> State:
        return self.state
'''
    },
    # Example 2: Traffic light cycle
    {
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Green", "type": 1},
                    {"label": "Yellow", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"}
            ]
        },
        "code": '''from enum import Enum, auto

class State(Enum):
    Red = auto()
    Green = auto()
    Yellow = auto()

class Event(Enum):
    NEXT = auto()

class TrafficLight:
    def __init__(self):
        self.state = State.Red

    def send(self, event: Event) -> bool:
        if event == Event.NEXT:
            if self.state == State.Red:
                self.state = State.Green
                return True
            elif self.state == State.Green:
                self.state = State.Yellow
                return True
            elif self.state == State.Yellow:
                self.state = State.Red
                return True
        return False

    def get_state(self) -> State:
        return self.state
'''
    },
]


@dataclass
class GenerationResult:
    """Result of code generation."""
    sc_name: str
    generated_code: str
    gen_time_s: float
    raw_output: str


class SCCodeGenerator:
    """Generate Python code from statechart definitions using Qwen-1.5B."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the MLX model."""
        if self.model is None:
            print(f"Loading model: {self.model_id}")
            self.model, self.tokenizer = load(self.model_id)
            print("Model loaded.")

    def build_prompt(self, sc: Dict, class_name: str) -> str:
        """Build few-shot prompt for code generation."""
        # Extract states and events from SC
        states = []
        root = sc.get("root_state", {})
        self._collect_states(root, states)

        events = set()
        transitions_desc = []
        for t in sc.get("transitions", []):
            event = t.get("event", "")
            events.add(event)
            src = t.get("from", [""])[0] if isinstance(t.get("from"), list) else ""
            tgt = t.get("to", [""])[0] if isinstance(t.get("to"), list) else ""
            guard = t.get("guard", {}).get("expression", "")
            guard_str = f" [{guard}]" if guard else ""
            transitions_desc.append(f"{src} --{event}{guard_str}--> {tgt}")

        # Find initial state
        initial = states[0] if states else "Unknown"
        for child in root.get("children", []):
            if child.get("is_initial"):
                initial = child.get("label", states[0])
                break

        # Build few-shot examples
        examples = ""
        for ex in FEW_SHOT_EXAMPLES:
            sc_json = json.dumps(ex["sc"], indent=2)
            examples += f"""Example statechart:
```json
{sc_json}
```

Generated Python code:
```python
{ex["code"]}
```

---

"""

        # Build target prompt
        sc_json = json.dumps(sc, indent=2)
        prompt = f"""You are a code generator that converts statechart JSON to Python code.

{examples}Now generate Python code for this statechart:
```json
{sc_json}
```

Requirements:
- Class name: {class_name}
- Use Enum for State and Event
- Initial state: {initial}
- States: {', '.join(states)}
- Events: {', '.join(sorted(events))}
- Transitions:
  {chr(10).join('  ' + t for t in transitions_desc)}
- send(event) method that returns True if transition occurred
- get_state() method that returns current state

Generate Python code:
```python
"""
        return prompt

    def _collect_states(self, node: Dict, states: List[str]):
        """Recursively collect state labels."""
        label = node.get("label", "")
        if label and label != "__root__":
            states.append(label)
        for child in node.get("children", []):
            self._collect_states(child, states)

    def generate(
        self,
        sc: Dict,
        class_name: str = "StateMachine",
        max_tokens: int = 500,
    ) -> GenerationResult:
        """Generate Python code from statechart."""
        self.load_model()

        prompt = self.build_prompt(sc, class_name)

        t0 = time.time()

        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        gen_time = time.time() - t0

        # Extract code from output
        code = self._extract_code(output)

        return GenerationResult(
            sc_name=class_name,
            generated_code=code,
            gen_time_s=gen_time,
            raw_output=output,
        )

    def _extract_code(self, output: str) -> str:
        """Extract Python code from LLM output."""
        # Add the imports that we started with in the prompt
        code = output.strip()

        # Remove closing markdown if present
        if "```" in code:
            code = code.split("```")[0]

        # Ensure it starts with proper imports
        if not code.startswith("from enum"):
            code = "from enum import Enum, auto\n\n" + code

        return code.strip()


def generate_python_from_sc(
    sc: Dict,
    class_name: str = "StateMachine",
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
) -> GenerationResult:
    """Convenience function to generate Python from SC."""
    generator = SCCodeGenerator(model_id=model_id)
    return generator.generate(sc, class_name)


if __name__ == "__main__":
    # Test with a simple SC
    test_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Locked", "type": 1, "is_initial": True},
                {"label": "Unlocked", "type": 1}
            ]
        },
        "transitions": [
            {"from": ["Locked"], "to": ["Unlocked"], "event": "COIN"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "PUSH"}
        ]
    }

    result = generate_python_from_sc(test_sc, "Turnstile")
    print("Generated code:")
    print(result.generated_code)
    print(f"\nGeneration time: {result.gen_time_s:.2f}s")
