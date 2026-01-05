"""Extended tokenizer for statechart manipulation events.

Adds special tokens for:
- State entry/exit: <ENTER:label>, <EXIT:label>
- Events: <EVENT:name>
- Guards: <GUARD_PASS>, <GUARD_FAIL>
- Transitions: <TRANS:from->to>
- Configuration: <CONFIG>, </CONFIG>
- History: <SAVE_HIST>, <RESTORE_HIST>
"""

import json
import re
from typing import Optional


class StatechartTokenizer:
    """Tokenizer with statechart-specific tokens for ML training."""

    # Special token prefixes
    SPECIAL_PREFIXES = [
        "<PAD>", "<UNK>", "<SOS>", "<EOS>",
        "<CONFIG>", "</CONFIG>",
        "<STEP>", "</STEP>",
        "<GUARD_PASS>", "<GUARD_FAIL>",
        "<SAVE_HIST>", "<RESTORE_HIST>",
        "<NO_TRANSITION>",
    ]

    def __init__(self, state_labels: Optional[list] = None, events: Optional[list] = None):
        """Initialize tokenizer with statechart vocabulary.

        Args:
            state_labels: List of state label strings
            events: List of event name strings
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.state_labels = state_labels or []
        self.events = events or []
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from special tokens, states, and events."""
        tokens = list(self.SPECIAL_PREFIXES)

        # Add state-specific tokens
        for label in self.state_labels:
            tokens.append(f"<ENTER:{label}>")
            tokens.append(f"<EXIT:{label}>")
            tokens.append(f"<ACTIVE:{label}>")
            tokens.append(f"<INACTIVE:{label}>")

        # Add event tokens
        for event in self.events:
            tokens.append(f"<EVENT:{event}>")

        # Add transition tokens (from -> to pairs)
        for src in self.state_labels:
            for tgt in self.state_labels:
                if src != tgt:
                    tokens.append(f"<TRANS:{src}->{tgt}>")

        # Build mappings
        self.chars = tokens
        self.char_to_idx = {tok: i for i, tok in enumerate(tokens)}
        self.idx_to_char = {i: tok for i, tok in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def add_state(self, label: str):
        """Add a new state label to vocabulary."""
        if label not in self.state_labels:
            self.state_labels.append(label)
            self._build_vocab()

    def add_event(self, event: str):
        """Add a new event to vocabulary."""
        if event not in self.events:
            self.events.append(event)
            self._build_vocab()

    def encode_step(self, event: str, exited: list, entered: list,
                    guard_passed: bool = True) -> list:
        """Encode a single statechart step as token sequence.

        Args:
            event: Event name that triggered the step
            exited: List of state labels that were exited
            entered: List of state labels that were entered
            guard_passed: Whether the guard condition passed

        Returns:
            List of token indices
        """
        tokens = []
        tokens.append(self.char_to_idx["<STEP>"])

        # Event
        event_tok = f"<EVENT:{event}>"
        if event_tok in self.char_to_idx:
            tokens.append(self.char_to_idx[event_tok])

        # Guard result
        if guard_passed:
            tokens.append(self.char_to_idx["<GUARD_PASS>"])
        else:
            tokens.append(self.char_to_idx["<GUARD_FAIL>"])

        # Exited states
        for label in exited:
            tok = f"<EXIT:{label}>"
            if tok in self.char_to_idx:
                tokens.append(self.char_to_idx[tok])

        # Entered states
        for label in entered:
            tok = f"<ENTER:{label}>"
            if tok in self.char_to_idx:
                tokens.append(self.char_to_idx[tok])

        tokens.append(self.char_to_idx["</STEP>"])
        return tokens

    def encode_config(self, active_states: list) -> list:
        """Encode a configuration as token sequence.

        Args:
            active_states: List of currently active state labels

        Returns:
            List of token indices
        """
        tokens = []
        tokens.append(self.char_to_idx["<CONFIG>"])

        for label in self.state_labels:
            if label in active_states:
                tok = f"<ACTIVE:{label}>"
            else:
                tok = f"<INACTIVE:{label}>"
            if tok in self.char_to_idx:
                tokens.append(self.char_to_idx[tok])

        tokens.append(self.char_to_idx["</CONFIG>"])
        return tokens

    def decode(self, indices: list) -> list:
        """Decode token indices back to token strings.

        Args:
            indices: List of token indices

        Returns:
            List of token strings
        """
        return [self.idx_to_char.get(idx, "<UNK>") for idx in indices]

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        with open(path, "w") as f:
            json.dump({
                "state_labels": self.state_labels,
                "events": self.events,
            }, f, indent=2)

    def load(self, path: str):
        """Load tokenizer from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
            self.state_labels = data.get("state_labels", [])
            self.events = data.get("events", [])
            self._build_vocab()

    @classmethod
    def from_statechart(cls, statechart: dict) -> "StatechartTokenizer":
        """Create tokenizer from a statechart definition.

        Args:
            statechart: Statechart definition dict with root_state and events

        Returns:
            Configured StatechartTokenizer
        """
        def extract_labels(state, labels=None):
            if labels is None:
                labels = []
            labels.append(state["label"])
            for child in state.get("children", []):
                extract_labels(child, labels)
            return labels

        state_labels = extract_labels(statechart["root_state"])
        events = [e["label"] for e in statechart.get("events", [])]

        return cls(state_labels=state_labels, events=events)


def demo():
    """Demo the statechart tokenizer."""
    # Create tokenizer for simple On/Off statechart
    tok = StatechartTokenizer(
        state_labels=["__root__", "Off", "On"],
        events=["TOGGLE", "TURN_ON", "TURN_OFF"]
    )

    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Tokens: {tok.chars[:20]}...")

    # Encode a step
    step_tokens = tok.encode_step(
        event="TOGGLE",
        exited=["Off"],
        entered=["On"],
        guard_passed=True
    )
    print(f"\nStep tokens: {step_tokens}")
    print(f"Decoded: {tok.decode(step_tokens)}")

    # Encode a config
    config_tokens = tok.encode_config(["__root__", "On"])
    print(f"\nConfig tokens: {config_tokens}")
    print(f"Decoded: {tok.decode(config_tokens)}")


if __name__ == "__main__":
    demo()
