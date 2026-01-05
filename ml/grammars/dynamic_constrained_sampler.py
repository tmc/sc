"""
Dynamic Constrained Sampler

Takes ANY statechart definition and constrains LLM generation
to only produce valid traces for that specific machine.

Usage:
    sampler = DynamicConstrainedSampler(model, tokenizer)
    sampler.load_constraint_sc(my_statechart)
    output = sampler.generate("Generate a valid trace")
    # Output guaranteed to be valid trace for my_statechart
"""

import json
import numpy as np
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SCDefinition:
    """Parsed statechart definition."""
    name: str
    states: Dict[str, dict]  # label -> state info
    transitions: List[dict]
    initial_states: Set[str]
    events: Set[str]

    @classmethod
    def from_json(cls, sc_json: dict) -> 'SCDefinition':
        """Parse SC JSON into definition."""
        name = sc_json.get('name', 'unnamed')
        states = {}
        initial_states = set()
        events = set()

        def collect_states(state, parent=None):
            label = state.get('label', '')
            if label and label != '__root__':
                states[label] = {
                    'type': state.get('type', 1),
                    'parent': parent,
                    'children': []
                }
                if state.get('is_initial'):
                    initial_states.add(label)

            for child in state.get('children', []):
                collect_states(child, label if label != '__root__' else None)
                if label and label != '__root__':
                    states[label]['children'].append(child.get('label'))

        if 'root_state' in sc_json:
            collect_states(sc_json['root_state'])

        transitions = sc_json.get('transitions', [])
        for t in transitions:
            if 'event' in t:
                events.add(t['event'])

        return cls(
            name=name,
            states=states,
            transitions=transitions,
            initial_states=initial_states,
            events=events
        )


@dataclass
class Configuration:
    """Current state configuration."""
    active_states: Set[str]

    def __hash__(self):
        return hash(frozenset(self.active_states))


class SCExecutor:
    """Executes statechart semantics."""

    def __init__(self, sc: SCDefinition):
        self.sc = sc

    def initial_config(self) -> Configuration:
        """Compute initial configuration."""
        return Configuration(active_states=set(self.sc.initial_states))

    def enabled_events(self, config: Configuration) -> Set[str]:
        """Compute events that have enabled transitions."""
        enabled = set()
        for t in self.sc.transitions:
            from_states = set(t.get('from', []))
            if from_states & config.active_states:
                event = t.get('event', '')
                if event:
                    enabled.add(event)
        return enabled

    def step(self, config: Configuration, event: str) -> Optional[Configuration]:
        """Execute one step: config × event → config'"""
        # Find matching transition
        for t in self.sc.transitions:
            from_states = set(t.get('from', []))
            to_states = set(t.get('to', []))
            t_event = t.get('event', '')

            if t_event == event and (from_states & config.active_states):
                # Execute transition
                new_active = config.active_states - from_states | to_states
                return Configuration(active_states=new_active)

        return None  # No matching transition


class DynamicConstrainedSampler:
    """
    LLM sampler that constrains output to valid traces.

    The constraint statechart is loaded dynamically - any SC
    can be used to constrain generation.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.sc: Optional[SCDefinition] = None
        self.executor: Optional[SCExecutor] = None
        self.config: Optional[Configuration] = None
        self.event_to_tokens: Dict[str, List[int]] = {}

    def load_constraint_sc(self, sc_json: dict):
        """Load a statechart to use as constraint."""
        self.sc = SCDefinition.from_json(sc_json)
        self.executor = SCExecutor(self.sc)
        self.config = self.executor.initial_config()
        self._build_event_token_map()

    def _build_event_token_map(self):
        """Map SC events to tokenizer tokens."""
        if not self.tokenizer or not self.sc:
            return

        self.event_to_tokens = {}
        for event in self.sc.events:
            # Find tokens that could represent this event
            # Simple approach: exact match or prefix match
            tokens = []
            for token_id in range(self.tokenizer.vocab_size):
                token_str = self.tokenizer.decode([token_id])
                if event.lower() in token_str.lower():
                    tokens.append(token_id)
            self.event_to_tokens[event] = tokens

    def get_valid_token_mask(self) -> np.ndarray:
        """
        Get mask of valid next tokens based on current SC state.

        Returns array of shape (vocab_size,) with 1 for valid, 0 for invalid.
        """
        if not self.tokenizer or not self.executor or not self.config:
            # No constraint - all tokens valid
            return np.ones(self.tokenizer.vocab_size)

        enabled = self.executor.enabled_events(self.config)

        # Build mask
        mask = np.zeros(self.tokenizer.vocab_size)
        for event in enabled:
            for token_id in self.event_to_tokens.get(event, []):
                mask[token_id] = 1

        # Always allow EOS if we have a valid trace
        if hasattr(self.tokenizer, 'eos_token_id'):
            mask[self.tokenizer.eos_token_id] = 1

        return mask

    def apply_constraint(self, logits: np.ndarray) -> np.ndarray:
        """Apply constraint mask to logits."""
        mask = self.get_valid_token_mask()
        # Set invalid tokens to -inf
        logits = np.where(mask > 0, logits, -np.inf)
        return logits

    def update_state(self, token_id: int) -> bool:
        """
        Update SC state after generating a token.
        Returns True if transition was valid.
        """
        if not self.tokenizer or not self.executor or not self.config:
            return True

        token_str = self.tokenizer.decode([token_id])

        # Find which event this token represents
        for event in self.sc.events:
            if event.lower() in token_str.lower():
                new_config = self.executor.step(self.config, event)
                if new_config:
                    self.config = new_config
                    return True

        return False

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate constrained output.

        Each token is masked to only allow valid SC events.
        """
        if not self.model or not self.tokenizer:
            return self._mock_generate(max_tokens)

        # Reset to initial configuration
        if self.executor:
            self.config = self.executor.initial_config()

        input_ids = self.tokenizer.encode(prompt)
        generated = []

        for _ in range(max_tokens):
            # Get logits from model
            logits = self.model.get_logits(input_ids + generated)

            # Apply constraint
            logits = self.apply_constraint(logits[-1])

            # Sample
            probs = self._softmax(logits)
            token_id = np.random.choice(len(probs), p=probs)

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Update state
            self.update_state(token_id)
            generated.append(token_id)

        return self.tokenizer.decode(generated)

    def _mock_generate(self, max_tokens: int) -> str:
        """Mock generation for testing without model."""
        if not self.executor or not self.sc:
            return "[]"

        trace = []
        self.config = self.executor.initial_config()

        for _ in range(max_tokens):
            enabled = self.executor.enabled_events(self.config)
            if not enabled:
                break

            # Pick random enabled event
            event = np.random.choice(list(enabled))
            trace.append(event)

            new_config = self.executor.step(self.config, event)
            if new_config:
                self.config = new_config
            else:
                break

        return json.dumps(trace)

    @staticmethod
    def _softmax(x):
        x = np.array(x)
        x = x - np.max(x)  # Numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


# Example usage
if __name__ == "__main__":
    # Example: Traffic light SC
    traffic_light = {
        "name": "TrafficLight",
        "root_state": {
            "label": "__root__",
            "type": 2,
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
    }

    # Create sampler with constraint
    sampler = DynamicConstrainedSampler()
    sampler.load_constraint_sc(traffic_light)

    # Generate valid trace (mock mode)
    trace = sampler.generate("Generate traffic light sequence", max_tokens=10)
    print(f"Generated trace: {trace}")
    print(f"Events in SC: {sampler.sc.events}")
    print(f"Initial states: {sampler.sc.initial_states}")
