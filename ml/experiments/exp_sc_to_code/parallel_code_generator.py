"""
Parallel Code Generator - Generate Python code for AND-state statecharts.

Uses explicit multi-region tracking template to teach LLM how to handle
concurrent orthogonal regions.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load, generate


# Few-shot example for parallel state machines
PARALLEL_FEW_SHOT = {
    "sc": {
        "root_state": {
            "label": "__root__",
            "type": 3,  # AND-state (parallel)
            "children": [
                {
                    "label": "Light",
                    "type": 2,
                    "children": [
                        {"label": "LightOff", "type": 1, "is_initial": True},
                        {"label": "LightOn", "type": 1}
                    ]
                },
                {
                    "label": "Fan",
                    "type": 2,
                    "children": [
                        {"label": "FanOff", "type": 1, "is_initial": True},
                        {"label": "FanOn", "type": 1}
                    ]
                }
            ]
        },
        "transitions": [
            {"from": ["LightOff"], "to": ["LightOn"], "event": "LIGHT_TOGGLE"},
            {"from": ["LightOn"], "to": ["LightOff"], "event": "LIGHT_TOGGLE"},
            {"from": ["FanOff"], "to": ["FanOn"], "event": "FAN_TOGGLE"},
            {"from": ["FanOn"], "to": ["FanOff"], "event": "FAN_TOGGLE"}
        ]
    },
    "code": '''from enum import Enum, auto

class LightState(Enum):
    LightOff = auto()
    LightOn = auto()

class FanState(Enum):
    FanOff = auto()
    FanOn = auto()

class Event(Enum):
    LIGHT_TOGGLE = auto()
    FAN_TOGGLE = auto()

class LightFanController:
    """Parallel state machine with two orthogonal regions."""

    def __init__(self):
        # Track each region independently
        self.regions = {
            'Light': LightState.LightOff,
            'Fan': FanState.FanOff
        }
        # Transition table: (region, current_state, event) -> new_state
        self.transitions = {
            ('Light', LightState.LightOff, Event.LIGHT_TOGGLE): LightState.LightOn,
            ('Light', LightState.LightOn, Event.LIGHT_TOGGLE): LightState.LightOff,
            ('Fan', FanState.FanOff, Event.FAN_TOGGLE): FanState.FanOn,
            ('Fan', FanState.FanOn, Event.FAN_TOGGLE): FanState.FanOff,
        }

    def send(self, event: Event) -> bool:
        """Process event across all regions. Returns True if any transition occurred."""
        transitioned = False
        for region, state in list(self.regions.items()):
            key = (region, state, event)
            if key in self.transitions:
                self.regions[region] = self.transitions[key]
                transitioned = True
        return transitioned

    def get_state(self, region: str):
        """Get current state of a specific region."""
        return self.regions.get(region)

    def get_all_states(self) -> dict:
        """Get current state of all regions."""
        return dict(self.regions)
'''
}


@dataclass
class ParallelGenResult:
    """Result of parallel code generation."""
    sc_name: str
    generated_code: str
    gen_time_s: float
    raw_output: str
    regions_detected: List[str]


def detect_parallel_regions(sc: Dict) -> List[Tuple[str, List[str]]]:
    """Detect orthogonal regions in a statechart.

    Returns list of (region_name, [state_names]) tuples.
    """
    regions = []
    root = sc.get("root_state", {})

    # Check if root is AND-state (type=3)
    if root.get("type") == 3:
        for region in root.get("children", []):
            region_name = region.get("label", "")
            states = []
            initial = None
            for child in region.get("children", []):
                label = child.get("label", "")
                if label:
                    states.append(label)
                    if child.get("is_initial"):
                        initial = label
            if region_name and states:
                regions.append((region_name, states, initial or states[0]))

    return regions


class ParallelCodeGenerator:
    """Generate Python code for parallel (AND-state) statecharts."""

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

    def build_parallel_prompt(self, sc: Dict, class_name: str) -> str:
        """Build prompt specifically for parallel statecharts."""
        regions = detect_parallel_regions(sc)

        # Build region descriptions
        region_desc = []
        for region_name, states, initial in regions:
            region_desc.append(f"- Region '{region_name}': states {states}, initial={initial}")

        # Build transition descriptions
        trans_desc = []
        for t in sc.get("transitions", []):
            src = t.get("from", [""])[0] if t.get("from") else ""
            tgt = t.get("to", [""])[0] if t.get("to") else ""
            event = t.get("event", "")
            trans_desc.append(f"{src} --{event}--> {tgt}")

        # Few-shot example
        example_sc = json.dumps(PARALLEL_FEW_SHOT["sc"], indent=2)
        example_code = PARALLEL_FEW_SHOT["code"]

        # Target SC
        sc_json = json.dumps(sc, indent=2)

        prompt = f"""You are generating Python code for PARALLEL statecharts (AND-states).
These have multiple orthogonal regions that run concurrently.

IMPORTANT: Use self.regions dict to track each region's state independently.
Do NOT use a single self.state variable.

Example parallel statechart:
```json
{example_sc}
```

Generated Python code:
```python
{example_code}
```

---

Now generate Python code for this parallel statechart:
```json
{sc_json}
```

Class name: {class_name}

Regions:
{chr(10).join(region_desc)}

Transitions:
{chr(10).join('  ' + t for t in trans_desc)}

Requirements:
1. Create separate Enum for each region's states
2. Use self.regions dict to track each region
3. Use self.transitions dict with (region, state, event) keys
4. send() method processes event across ALL regions
5. get_state(region) returns state of specific region
6. get_all_states() returns dict of all region states

Generate Python code:
```python
"""
        return prompt

    def generate(
        self,
        sc: Dict,
        class_name: str = "ParallelStateMachine",
        max_tokens: int = 800,
    ) -> ParallelGenResult:
        """Generate Python code for parallel statechart."""
        self.load_model()

        regions = detect_parallel_regions(sc)
        region_names = [r[0] for r in regions]

        prompt = self.build_parallel_prompt(sc, class_name)

        t0 = time.time()
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        gen_time = time.time() - t0

        # Extract code
        code = self._extract_code(output)

        return ParallelGenResult(
            sc_name=class_name,
            generated_code=code,
            gen_time_s=gen_time,
            raw_output=output,
            regions_detected=region_names,
        )

    def _extract_code(self, output: str) -> str:
        """Extract Python code from LLM output."""
        code = output.strip()

        # Remove closing markdown if present
        if "```" in code:
            code = code.split("```")[0]

        # Ensure it starts with proper imports
        if not code.startswith("from enum"):
            code = "from enum import Enum, auto\n\n" + code

        return code.strip()


def generate_parallel_code(
    sc: Dict,
    class_name: str = "ParallelStateMachine",
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
) -> ParallelGenResult:
    """Convenience function to generate code for parallel SC."""
    generator = ParallelCodeGenerator(model_id=model_id)
    return generator.generate(sc, class_name)


if __name__ == "__main__":
    # Test with Player parallel SC
    test_sc = {
        "root_state": {
            "label": "__root__",
            "type": 3,  # AND-state
            "children": [
                {
                    "label": "Movement",
                    "type": 2,
                    "children": [
                        {"label": "Standing", "type": 1, "is_initial": True},
                        {"label": "Walking", "type": 1},
                        {"label": "Running", "type": 1}
                    ]
                },
                {
                    "label": "Combat",
                    "type": 2,
                    "children": [
                        {"label": "Idle", "type": 1, "is_initial": True},
                        {"label": "Attacking", "type": 1},
                        {"label": "Defending", "type": 1}
                    ]
                }
            ]
        },
        "transitions": [
            {"from": ["Standing"], "to": ["Walking"], "event": "WALK"},
            {"from": ["Walking"], "to": ["Running"], "event": "RUN"},
            {"from": ["Running"], "to": ["Walking"], "event": "SLOW"},
            {"from": ["Walking"], "to": ["Standing"], "event": "STOP"},
            {"from": ["Idle"], "to": ["Attacking"], "event": "ATTACK"},
            {"from": ["Attacking"], "to": ["Idle"], "event": "FINISH"},
            {"from": ["Idle"], "to": ["Defending"], "event": "DEFEND"},
            {"from": ["Defending"], "to": ["Idle"], "event": "FINISH"}
        ]
    }

    result = generate_parallel_code(test_sc, "Player")
    print("Generated code:")
    print(result.generated_code)
    print(f"\nRegions detected: {result.regions_detected}")
    print(f"Generation time: {result.gen_time_s:.2f}s")
