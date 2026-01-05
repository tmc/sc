#!/usr/bin/env python3
"""
Statechart-based Sampler: Use actual statechart execution for guided generation.

Instead of hand-coded Python state machine, we:
1. Load the JSON parser statechart definition
2. Execute it using statechart semantics
3. Query enabled transitions to determine valid tokens
4. Use token mapping to build logit mask
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path

# Token type to event mapping
# Single-character tokens
CHAR_TO_EVENT = {
    '{': 'LBRACE',
    '}': 'RBRACE',
    '[': 'LBRACKET',
    ']': 'RBRACKET',
    ':': 'COLON',
    ',': 'COMMA',
    '"': 'QUOTE',
}

# Multi-character keyword tokens
KEYWORD_TO_EVENT = {
    'true': 'BOOL',
    'false': 'BOOL',
    'null': 'NULL',
}

# For backwards compatibility
TOKEN_TO_EVENT = {**CHAR_TO_EVENT, **KEYWORD_TO_EVENT}


class JSONTokenizer:
    """
    Tokenizes JSON text into statechart events.

    Handles:
    - Structural tokens: { } [ ] : ,
    - Strings: "..." with escape handling
    - Numbers: integers and floats
    - Keywords: true, false, null
    """

    def __init__(self):
        self.in_string = False
        self.current_token = ""
        self.escape_next = False

    def reset(self):
        self.in_string = False
        self.current_token = ""
        self.escape_next = False

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of statechart events."""
        events = []
        self.reset()

        for char in text:
            event = self.process_char(char)
            if event:
                events.append(event)

        return events

    def process_char(self, char: str) -> Optional[str]:
        """
        Process a single character and return statechart event if any.

        Returns event name or None.
        """
        # Handle escape sequences in strings
        if self.escape_next:
            self.escape_next = False
            return 'STRING'  # Escaped char is string content

        if self.in_string:
            if char == '\\':
                self.escape_next = True
                return None
            elif char == '"':
                self.in_string = False
                self.current_token = ""
                return 'QUOTE'
            else:
                return 'STRING'

        # Not in string
        if char == '"':
            self.in_string = True
            return 'QUOTE'

        if char in CHAR_TO_EVENT:
            return CHAR_TO_EVENT[char]

        # Numbers
        if char.isdigit() or char == '-':
            if not self.current_token or not self.current_token[0].isdigit():
                self.current_token = char
                return 'NUMBER'
            self.current_token += char
            return None  # Continuation of number

        if char == '.' and self.current_token and self.current_token[0].isdigit():
            self.current_token += char
            return None  # Decimal point in number

        # Keywords (true, false, null)
        if char.isalpha():
            self.current_token += char
            if self.current_token in ('true', 'false'):
                self.current_token = ""
                return 'BOOL'
            elif self.current_token == 'null':
                self.current_token = ""
                return 'NULL'
            return None  # Building keyword

        # Whitespace or other - end any current token
        if self.current_token:
            self.current_token = ""

        return None


@dataclass
class ParserContext:
    """Extended state for the JSON parser statechart."""
    depth: int = 0
    stack: List[str] = field(default_factory=list)
    element_count: int = 0
    max_depth: int = 8
    max_elements: int = 10

    def push(self, item: str):
        self.stack.append(item)

    def pop(self) -> Optional[str]:
        return self.stack.pop() if self.stack else None

    def top(self) -> Optional[str]:
        return self.stack[-1] if self.stack else None

    def copy(self) -> "ParserContext":
        ctx = ParserContext(
            depth=self.depth,
            element_count=self.element_count,
            max_depth=self.max_depth,
            max_elements=self.max_elements,
        )
        ctx.stack = self.stack.copy()
        return ctx


@dataclass
class StatechartMachine:
    """
    Simple statechart machine executor.

    Executes a statechart definition with extended state (context).
    """
    statechart: Dict[str, Any]
    current_state: str = "Ready"
    context: ParserContext = field(default_factory=ParserContext)

    # Index transitions by source state for fast lookup
    _transitions_from: Dict[str, List[Dict]] = field(default_factory=dict)

    def __post_init__(self):
        self._index_transitions()

    def _index_transitions(self):
        """Build transition index."""
        self._transitions_from = {}
        for t in self.statechart.get("transitions", []):
            for src in t.get("from", []):
                if src not in self._transitions_from:
                    self._transitions_from[src] = []
                self._transitions_from[src].append(t)

    def reset(self):
        """Reset to initial state."""
        self.current_state = "Ready"
        self.context = ParserContext(
            max_depth=self.statechart.get("variables", {}).get("max_depth", 8),
            max_elements=self.statechart.get("variables", {}).get("max_elements", 10),
        )

    def get_enabled_events(self) -> Set[str]:
        """Get events that are enabled from current state."""
        enabled = set()

        for t in self._transitions_from.get(self.current_state, []):
            event = t.get("event")
            if not event:
                continue

            # Check guard
            guard = t.get("guard", {}).get("expression", "")
            if guard and not self._eval_guard(guard):
                continue

            enabled.add(event)

        return enabled

    def _resolve_value(self, val_str: str) -> int:
        """Resolve a value string to an integer (handles literals and variables)."""
        val_str = val_str.strip()
        ctx = self.context

        # Check for variable names
        if val_str == "max_depth":
            return ctx.max_depth
        elif val_str == "max_elements":
            return ctx.max_elements
        elif val_str == "depth":
            return ctx.depth
        elif val_str == "element_count":
            return ctx.element_count
        else:
            # Try parsing as integer literal
            return int(val_str)

    def _eval_guard(self, expr: str) -> bool:
        """Evaluate guard expression against context."""
        ctx = self.context

        # Simple expression evaluator for our guards
        expr = expr.strip()

        # Handle compound guards with &&
        if "&&" in expr:
            parts = expr.split("&&")
            return all(self._eval_guard(p.strip()) for p in parts)

        if "depth > " in expr:
            val = self._resolve_value(expr.split(">")[1].strip())
            return ctx.depth > val
        elif "depth < " in expr:
            val = self._resolve_value(expr.split("<")[1].strip())
            return ctx.depth < val
        elif "depth ==" in expr:
            val = self._resolve_value(expr.split("==")[1].strip())
            return ctx.depth == val
        elif "depth >=" in expr:
            val = self._resolve_value(expr.split(">=")[1].strip())
            return ctx.depth >= val
        elif "stack.top() == 'object'" in expr:
            return ctx.top() == "object"
        elif "stack.top() == 'array'" in expr:
            return ctx.top() == "array"
        elif "element_count < max_elements" in expr:
            return ctx.element_count < ctx.max_elements
        elif "element_count >= max_elements" in expr:
            return ctx.element_count >= ctx.max_elements

        # Default: guard passes
        return True

    def _exec_action(self, expr: str):
        """Execute action expression."""
        ctx = self.context
        expr = expr.strip()

        # Parse simple action expressions
        for part in expr.split(";"):
            part = part.strip()
            if not part:
                continue

            if "depth++" in part:
                ctx.depth += 1
            elif "depth--" in part:
                ctx.depth -= 1
            elif "depth = " in part:
                ctx.depth = int(part.split("=")[1].strip())
            elif "stack.push('object')" in part:
                ctx.push("object")
            elif "stack.push('array')" in part:
                ctx.push("array")
            elif "stack.pop()" in part:
                ctx.pop()
            elif "element_count = 0" in part:
                ctx.element_count = 0
            elif "element_count++" in part:
                ctx.element_count += 1

    def send_event(self, event: str) -> bool:
        """
        Send an event to the machine.

        Returns True if transition was taken, False otherwise.
        """
        for t in self._transitions_from.get(self.current_state, []):
            if t.get("event") != event:
                continue

            # Check guard
            guard = t.get("guard", {}).get("expression", "")
            if guard and not self._eval_guard(guard):
                continue

            # Execute actions
            for action in t.get("actions", []):
                self._exec_action(action.get("expression", ""))

            # Transition to target state
            targets = t.get("to", [])
            if targets:
                self.current_state = targets[0]

            return True

        return False

    def is_complete(self) -> bool:
        return self.current_state == "Complete"

    def should_force_close(self) -> bool:
        return self.current_state == "ForceClose"


class StatechartGuidedSampler:
    """
    Sampler that uses statechart execution to guide token selection.
    """

    def __init__(self, tokenizer: Any, statechart_path: Optional[str] = None):
        self.tokenizer = tokenizer

        # Load statechart
        if statechart_path is None:
            statechart_path = Path(__file__).parent / "json_parser_v2.statechart.json"

        with open(statechart_path) as f:
            sc_def = json.load(f)

        self.machine = StatechartMachine(statechart=sc_def)

        # Build token -> event mapping for vocabulary
        self._build_token_event_map()

    def _build_token_event_map(self):
        """Map vocabulary tokens to statechart events."""
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        self.event_to_token_ids: Dict[str, Set[int]] = {
            'LBRACE': set(),
            'RBRACE': set(),
            'LBRACKET': set(),
            'RBRACKET': set(),
            'COLON': set(),
            'COMMA': set(),
            'QUOTE': set(),
            'STRING': set(),
            'NUMBER': set(),
            'BOOL': set(),
            'NULL': set(),
        }

        for token_str, token_id in vocab.items():
            # Classify token
            if '{' in token_str:
                self.event_to_token_ids['LBRACE'].add(token_id)
            if '}' in token_str:
                self.event_to_token_ids['RBRACE'].add(token_id)
            if '[' in token_str:
                self.event_to_token_ids['LBRACKET'].add(token_id)
            if ']' in token_str:
                self.event_to_token_ids['RBRACKET'].add(token_id)
            if ':' in token_str:
                self.event_to_token_ids['COLON'].add(token_id)
            if ',' in token_str:
                self.event_to_token_ids['COMMA'].add(token_id)
            if '"' in token_str:
                self.event_to_token_ids['QUOTE'].add(token_id)
            if token_str.strip() in ('true', 'false'):
                self.event_to_token_ids['BOOL'].add(token_id)
            if token_str.strip() == 'null':
                self.event_to_token_ids['NULL'].add(token_id)

            # Numbers
            try:
                float(token_str.strip())
                self.event_to_token_ids['NUMBER'].add(token_id)
            except ValueError:
                pass

            # String content (alphanumeric)
            if token_str.strip().isalnum():
                self.event_to_token_ids['STRING'].add(token_id)

    def reset(self):
        """Reset machine to initial state."""
        self.machine.reset()

    def get_valid_token_ids(self) -> Set[int]:
        """Get token IDs that correspond to enabled events."""
        enabled = self.machine.get_enabled_events()

        valid_ids = set()
        for event in enabled:
            valid_ids.update(self.event_to_token_ids.get(event, set()))

        return valid_ids

    def process_token(self, token_str: str) -> bool:
        """Process a generated token and update machine state."""
        # Map token to event
        for char in token_str:
            if char in TOKEN_TO_EVENT:
                event = TOKEN_TO_EVENT[char]
                self.machine.send_event(event)
            elif char.isalnum() or char == '_':
                self.machine.send_event('STRING')
            elif char.isdigit() or char == '.':
                self.machine.send_event('NUMBER')

        return not self.machine.should_force_close()

    def get_closing_sequence(self) -> str:
        """Get tokens needed to close current structure."""
        ctx = self.machine.context
        closing = ""

        # Close all open structures based on stack
        for item in reversed(ctx.stack):
            if item == "object":
                closing += "}"
            elif item == "array":
                closing += "]"

        return closing

    def is_complete(self) -> bool:
        return self.machine.is_complete()

    def should_force_close(self) -> bool:
        return self.machine.should_force_close()


def demo():
    """Demo the statechart-based sampler."""
    print("=" * 60)
    print("STATECHART-BASED SAMPLER DEMO")
    print("=" * 60)

    # Load statechart
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    machine = StatechartMachine(statechart=sc_def)

    # Simulate parsing: {"label": "test"}
    test_events = ['LBRACE', 'QUOTE', 'STRING', 'QUOTE', 'COLON', 'QUOTE', 'STRING', 'QUOTE', 'RBRACE']

    print(f"\nSimulating JSON parsing...")
    print(f"Initial state: {machine.current_state}")

    for event in test_events:
        enabled = machine.get_enabled_events()
        print(f"\n  State: {machine.current_state}")
        print(f"  Enabled events: {enabled}")
        print(f"  Sending: {event}")

        if event in enabled:
            machine.send_event(event)
            print(f"  -> New state: {machine.current_state}")
        else:
            print(f"  -> Event not enabled!")
            break

    print(f"\nFinal state: {machine.current_state}")
    print(f"Complete: {machine.is_complete()}")
    print(f"Context: depth={machine.context.depth}, stack={machine.context.stack}")


if __name__ == "__main__":
    demo()
