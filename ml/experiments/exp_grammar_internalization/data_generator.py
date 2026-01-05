#!/usr/bin/env python3
"""
Data Generator: Create (prompt, valid_sc) pairs for training.

Uses constrained sampling to generate valid statecharts,
then pairs them with diverse prompts.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

# Import constrained decoding
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class TrainingExample:
    """A single training example."""
    prompt: str
    completion: str  # Valid SC JSON
    complexity: str  # simple, medium, complex


# Template statecharts with varying complexity
TEMPLATE_STATECHARTS = {
    "simple": [
        # Toggle
        {
            "root_state": {
                "label": "__root__", "type": 2,
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
        # Door
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Closed", "type": 1, "is_initial": True},
                    {"label": "Open", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"}
            ]
        },
        # Active/Inactive
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Inactive", "type": 1, "is_initial": True},
                    {"label": "Active", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Inactive"], "to": ["Active"], "event": "ACTIVATE"},
                {"from": ["Active"], "to": ["Inactive"], "event": "DEACTIVATE"}
            ]
        },
    ],
    "medium": [
        # Traffic Light
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Yellow", "type": 1},
                    {"label": "Green", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
                {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
                {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"}
            ]
        },
        # Loading States
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Loading", "type": 1},
                    {"label": "Success", "type": 1},
                    {"label": "Error", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Loading"], "event": "FETCH"},
                {"from": ["Loading"], "to": ["Success"], "event": "RESOLVE"},
                {"from": ["Loading"], "to": ["Error"], "event": "REJECT"},
                {"from": ["Error"], "to": ["Idle"], "event": "RETRY"},
                {"from": ["Success"], "to": ["Idle"], "event": "RESET"}
            ]
        },
        # Player States
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Stopped", "type": 1, "is_initial": True},
                    {"label": "Playing", "type": 1},
                    {"label": "Paused", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Paused"], "to": ["Stopped"], "event": "STOP"}
            ]
        },
    ],
    "complex": [
        # Hierarchical: Composite state
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {
                        "label": "On", "type": 2,
                        "children": [
                            {"label": "Running", "type": 1, "is_initial": True},
                            {"label": "Paused", "type": 1}
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "START"},
                {"from": ["On"], "to": ["Off"], "event": "STOP"},
                {"from": ["Running"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Running"], "event": "RESUME"}
            ]
        },
        # Multi-level hierarchy
        {
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {
                        "label": "Active", "type": 2,
                        "children": [
                            {"label": "Processing", "type": 1, "is_initial": True},
                            {"label": "Waiting", "type": 1}
                        ]
                    },
                    {"label": "Complete", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Active"], "event": "START"},
                {"from": ["Processing"], "to": ["Waiting"], "event": "BLOCK"},
                {"from": ["Waiting"], "to": ["Processing"], "event": "UNBLOCK"},
                {"from": ["Active"], "to": ["Complete"], "event": "FINISH"},
                {"from": ["Complete"], "to": ["Idle"], "event": "RESET"}
            ]
        },
    ]
}

# Prompt templates
PROMPT_TEMPLATES = [
    "Generate a valid statechart JSON:",
    "Create a statechart for a {domain} system:",
    "Write the JSON for a {domain} state machine:",
    "Define a statechart that models {domain}:",
    "Generate JSON for a finite state machine:",
    "Create a valid SC definition:",
    "Write a statechart with states and transitions:",
    "Generate a state machine in JSON format:",
]

DOMAINS = [
    "toggle switch", "door lock", "traffic light", "media player",
    "network connection", "user authentication", "order workflow",
    "game state", "elevator", "vending machine", "washing machine",
    "microwave", "alarm system", "thermostat", "phone call"
]


def generate_prompt(sc: Dict, complexity: str) -> str:
    """Generate a prompt for the given statechart."""
    template = random.choice(PROMPT_TEMPLATES)
    domain = random.choice(DOMAINS)
    return template.format(domain=domain)


def augment_statechart(base_sc: Dict) -> Dict:
    """Create variations of a base statechart."""
    sc = json.loads(json.dumps(base_sc))  # Deep copy

    # Randomly rename states
    state_prefixes = ["State", "S", "", "Mode", "Phase"]
    prefix = random.choice(state_prefixes)

    def rename_states(state: Dict, mapping: Dict[str, str]):
        old_label = state.get("label", "")
        if old_label != "__root__":
            new_label = f"{prefix}{old_label}" if prefix else old_label
            # Random suffix sometimes
            if random.random() < 0.3:
                new_label = f"{new_label}{random.randint(1, 9)}"
            mapping[old_label] = new_label
            state["label"] = new_label

        for child in state.get("children", []):
            rename_states(child, mapping)

    mapping = {}
    rename_states(sc["root_state"], mapping)

    # Update transitions with new names
    for trans in sc.get("transitions", []):
        trans["from"] = [mapping.get(s, s) for s in trans.get("from", [])]
        trans["to"] = [mapping.get(s, s) for s in trans.get("to", [])]

    # Randomly rename events
    event_styles = ["UPPER", "lower", "camelCase"]
    style = random.choice(event_styles)

    for trans in sc.get("transitions", []):
        event = trans.get("event", "")
        if style == "lower":
            trans["event"] = event.lower()
        elif style == "camelCase":
            parts = event.split("_")
            trans["event"] = parts[0].lower() + "".join(p.title() for p in parts[1:])

    return sc


def generate_training_data(
    n_examples: int = 1000,
    output_path: str = None,
) -> List[TrainingExample]:
    """
    Generate training data from template statecharts.

    Args:
        n_examples: Number of examples to generate
        output_path: Optional path to save JSONL

    Returns:
        List of TrainingExample
    """
    examples = []

    # Distribution: 40% simple, 40% medium, 20% complex
    complexity_dist = {
        "simple": int(n_examples * 0.4),
        "medium": int(n_examples * 0.4),
        "complex": n_examples - int(n_examples * 0.4) - int(n_examples * 0.4),
    }

    for complexity, count in complexity_dist.items():
        templates = TEMPLATE_STATECHARTS[complexity]

        for i in range(count):
            # Pick random template
            base_sc = random.choice(templates)

            # Augment it
            sc = augment_statechart(base_sc)

            # Generate prompt
            prompt = generate_prompt(sc, complexity)

            # Format completion (compact JSON)
            completion = json.dumps(sc, separators=(',', ':'))

            examples.append(TrainingExample(
                prompt=prompt,
                completion=completion,
                complexity=complexity,
            ))

    random.shuffle(examples)

    # Save if requested
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps({
                    "prompt": ex.prompt,
                    "completion": ex.completion,
                    "complexity": ex.complexity,
                }) + "\n")

        print(f"Saved {len(examples)} examples to {path}")

    return examples


def generate_training_jsonl(n_examples: int = 1000) -> Path:
    """Generate training data and save to JSONL."""
    output_path = Path(__file__).parent / "training_data.jsonl"
    generate_training_data(n_examples, str(output_path))
    return output_path


def demo():
    """Demo the data generator."""
    print("=" * 60)
    print("Training Data Generator Demo")
    print("=" * 60)

    examples = generate_training_data(n_examples=10)

    for i, ex in enumerate(examples[:5], 1):
        print(f"\n--- Example {i} ({ex.complexity}) ---")
        print(f"Prompt: {ex.prompt}")
        print(f"Completion: {ex.completion[:100]}...")

    print(f"\nGenerated {len(examples)} examples")


if __name__ == "__main__":
    demo()
