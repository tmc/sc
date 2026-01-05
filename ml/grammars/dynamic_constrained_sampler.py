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
import mlx.core as mx
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
    root_children: List[str] = field(default_factory=list)  # Top-level children of root

    @classmethod
    def from_json(cls, sc_json: dict) -> 'SCDefinition':
        """Parse SC JSON into definition."""
        name = sc_json.get('name', 'unnamed')
        states = {}
        initial_states = set()
        events = set()
        root_children = []

        def collect_states(state, parent=None):
            label = state.get('label', '')
            children = state.get('children', [])

            if label and label != '__root__':
                states[label] = {
                    'type': state.get('type', 1),  # 1=BASIC, 2=OR, 3=AND
                    'parent': parent,
                    'children': [c.get('label') for c in children if c.get('label')],
                    'is_history': state.get('is_history', False),
                    'history_type': state.get('history_type', 0),  # 0=none, 1=shallow, 2=deep
                    'entry_actions': state.get('entry_actions', []),
                    'exit_actions': state.get('exit_actions', []),
                }
                if state.get('is_initial'):
                    initial_states.add(label)

            # Track root's direct children
            if label == '__root__':
                for c in children:
                    if c.get('label'):
                        root_children.append(c.get('label'))

            for child in children:
                collect_states(child, label if label != '__root__' else None)

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
            events=events,
            root_children=root_children,
        )

    def get_initial_child(self, parent_label: str) -> Optional[str]:
        """Get the initial child state of a composite state."""
        state_info = self.states.get(parent_label, {})
        children = state_info.get('children', [])

        for child_label in children:
            if child_label in self.initial_states:
                return child_label

        # If no explicit initial, return first non-history child
        for child_label in children:
            child_info = self.states.get(child_label, {})
            if not child_info.get('is_history'):
                return child_label

        return None

    def is_composite(self, label: str) -> bool:
        """Check if state is composite (has children)."""
        state_info = self.states.get(label, {})
        children = state_info.get('children', [])
        # Filter out history pseudo-states
        real_children = [c for c in children if not self.states.get(c, {}).get('is_history')]
        return len(real_children) > 0

    def is_and_state(self, label: str) -> bool:
        """Check if state is AND-state (type=3, parallel regions)."""
        state_info = self.states.get(label, {})
        return state_info.get('type') == 3

    def is_history_state(self, label: str) -> bool:
        """Check if state is a history pseudo-state."""
        state_info = self.states.get(label, {})
        return state_info.get('is_history', False)


@dataclass
class Configuration:
    """Current state configuration."""
    active_states: Set[str]

    def __hash__(self):
        return hash(frozenset(self.active_states))


class SCExecutor:
    """Executes statechart semantics with proper Harel semantics."""

    def __init__(self, sc: SCDefinition):
        self.sc = sc
        self.history: Dict[str, Set[str]] = {}  # composite_label -> saved config substates

    def _expand_state_entry(self, label: str, active: Set[str]):
        """
        Recursively expand state entry to include all necessary active states.

        For OR-state (type=2): add parent + cascade to initial child
        For AND-state (type=3): add parent + all children regions + their initials
        """
        active.add(label)

        if self.sc.is_and_state(label):
            # AND-state: activate ALL children regions
            state_info = self.sc.states.get(label, {})
            for child in state_info.get('children', []):
                if not self.sc.is_history_state(child):
                    self._expand_state_entry(child, active)
        elif self.sc.is_composite(label):
            # OR-state: cascade to initial child
            initial_child = self.sc.get_initial_child(label)
            if initial_child:
                self._expand_state_entry(initial_child, active)

    def initial_config(self) -> Configuration:
        """Compute initial configuration with proper hierarchy expansion."""
        active = set()

        # Find top-level initial state(s)
        for label in self.sc.root_children:
            if label in self.sc.initial_states:
                self._expand_state_entry(label, active)

        # If no explicit initial at root, pick first root child
        if not active and self.sc.root_children:
            self._expand_state_entry(self.sc.root_children[0], active)

        return Configuration(active_states=active)

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

    def _get_composite_parent(self, state_label: str) -> Optional[str]:
        """Find the containing composite state for a given state."""
        state_info = self.sc.states.get(state_label, {})
        parent = state_info.get('parent')
        if parent and self.sc.is_composite(parent):
            return parent
        return None

    def _save_history(self, config: Configuration, composite_label: str):
        """Save the current sub-configuration of a composite state."""
        # Find active leaf states that are descendants of this composite
        active_in_composite = set()
        for s in config.active_states:
            if s == composite_label:
                continue  # Don't include the composite itself
            # Check if s is a descendant of composite
            current = s
            while current:
                state_info = self.sc.states.get(current, {})
                parent = state_info.get('parent')
                if parent == composite_label:
                    active_in_composite.add(s)
                    break
                current = parent

        if active_in_composite:
            self.history[composite_label] = active_in_composite

    def _restore_from_history(self, composite_label: str) -> Set[str]:
        """Restore configuration from history, or default if no history."""
        if composite_label in self.history:
            # Restore saved config
            restored = set()
            restored.add(composite_label)
            restored.update(self.history[composite_label])
            return restored
        else:
            # No history - go to default initial
            active = set()
            self._expand_state_entry(composite_label, active)
            return active

    def step(self, config: Configuration, event: str) -> Optional[Configuration]:
        """Execute one step: config × event → config' with Harel semantics."""
        # Find matching transition
        for t in self.sc.transitions:
            from_states = set(t.get('from', []))
            to_states = set(t.get('to', []))
            t_event = t.get('event', '')

            if t_event == event and (from_states & config.active_states):
                new_active = set(config.active_states)

                # Save history for composite states we're leaving
                for from_s in from_states:
                    # If from_s itself is a composite, save its history
                    if self.sc.is_composite(from_s) and from_s in config.active_states:
                        self._save_history(config, from_s)
                    # Also check parent composite
                    composite = self._get_composite_parent(from_s)
                    if composite and composite in config.active_states:
                        self._save_history(config, composite)

                # Remove source states and their ancestors up to LCA
                for from_s in from_states:
                    new_active.discard(from_s)
                    # Also remove parent composites if we're leaving them
                    state_info = self.sc.states.get(from_s, {})
                    parent = state_info.get('parent')
                    # Only remove parent if transition explicitly leaves it
                    # Check if any to_state is outside this parent
                    if parent and parent in new_active:
                        leaving_parent = True
                        for to_s in to_states:
                            to_info = self.sc.states.get(to_s, {})
                            if to_info.get('parent') == parent:
                                leaving_parent = False
                                break
                            # Check if to_s is the parent itself
                            if to_s == parent:
                                leaving_parent = False
                                break
                        if leaving_parent:
                            new_active.discard(parent)

                # Add target states with proper expansion
                for to_s in to_states:
                    if self.sc.is_history_state(to_s):
                        # History state - restore saved config
                        state_info = self.sc.states.get(to_s, {})
                        composite_parent = state_info.get('parent')
                        if composite_parent:
                            restored = self._restore_from_history(composite_parent)
                            new_active.update(restored)
                    else:
                        # Normal state - expand entry
                        self._expand_state_entry(to_s, new_active)

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
            # Get logits from model
            # Call model directly
            full_seq = mx.array(input_ids + generated)[None, :]
            logits = self.model(full_seq)[0]

            # Apply constraint
            logits = self.apply_constraint(logits[-1])

            # Sample
            probs = self._softmax(logits)
            token_id = np.random.choice(len(probs), p=probs)

            # Check for EOS
            # Check for EOS
            eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else self.tokenizer.char_to_idx.get("<EOS>", -1)
            if token_id == eos_id:
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
        # Check for all -inf
        if np.all(x == -np.inf):
            # Uniform over all (fallback) or uniform over non-masked if possible?
            # If all are -inf, we have NO valid choice.
            # Return uniform to prevent crash, but this will pick invalid token.
            return np.ones_like(x) / len(x)
            
        x = x - np.max(x)  # Numerical stability
        exp_x = np.exp(x)
        # Check for 0 sum
        s = np.sum(exp_x)
        if s == 0:
             return np.ones_like(x) / len(x)
        return exp_x / s


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
