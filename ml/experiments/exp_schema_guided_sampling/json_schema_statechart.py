#!/usr/bin/env python3
"""
JSON Schema Statechart: Track JSON parsing state for guided generation.

Defines a statechart that tracks the current position in JSON structure,
enabling constrained decoding that ensures valid output.

States track:
- Current nesting level (object/array depth)
- Expected next token type (key, value, delimiter)
- SC-specific requirements (root_state, label, children)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set, List, Dict, Optional, Any


class TokenType(Enum):
    """Token types in JSON."""
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    COLON = auto()       # :
    COMMA = auto()       # ,
    QUOTE = auto()       # "
    STRING = auto()      # string content
    NUMBER = auto()      # numeric value
    TRUE = auto()        # true
    FALSE = auto()       # false
    NULL = auto()        # null
    WHITESPACE = auto()  # spaces, newlines
    OTHER = auto()       # unknown


class JSONState(Enum):
    """States in JSON parsing statechart."""
    START = auto()

    # Object states
    IN_OBJECT = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    AFTER_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING_VALUE = auto()
    AFTER_VALUE = auto()

    # Array states
    IN_ARRAY = auto()
    EXPECT_ELEMENT = auto()
    AFTER_ELEMENT = auto()

    # Terminal
    COMPLETE = auto()
    ERROR = auto()


# SC-specific field requirements
SC_REQUIRED_FIELDS = {"root_state", "label"}
SC_OPTIONAL_FIELDS = {"type", "children", "is_initial", "transitions", "event", "from", "to"}


@dataclass
class ParserState:
    """Current state of the JSON parser statechart."""
    state: JSONState = JSONState.START
    depth: int = 0
    max_depth: int = 10

    # Stack tracks nested structures
    stack: List[str] = field(default_factory=list)  # "object" or "array"

    # Current context
    current_key: str = ""
    in_string: bool = False
    string_buffer: str = ""

    # SC-specific tracking
    seen_fields: Set[str] = field(default_factory=set)
    in_root_state: bool = False
    in_children: bool = False
    state_count: int = 0

    # Array element tracking (to detect repetition)
    array_element_count: int = 0
    max_array_elements: int = 8
    total_tokens: int = 0
    max_tokens: int = 1024

    def copy(self) -> "ParserState":
        """Create a copy of this state."""
        new = ParserState(
            state=self.state,
            depth=self.depth,
            max_depth=self.max_depth,
            current_key=self.current_key,
            in_string=self.in_string,
            string_buffer=self.string_buffer,
            in_root_state=self.in_root_state,
            in_children=self.in_children,
            state_count=self.state_count,
            array_element_count=self.array_element_count,
            max_array_elements=self.max_array_elements,
            total_tokens=self.total_tokens,
            max_tokens=self.max_tokens,
        )
        new.stack = self.stack.copy()
        new.seen_fields = self.seen_fields.copy()
        return new


class JSONSchemaStatechart:
    """
    Statechart for JSON/SC schema validation during generation.

    Tracks parsing state and determines valid next tokens.
    """

    def __init__(self, max_depth: int = 10, max_states: int = 20):
        self.max_depth = max_depth
        self.max_states = max_states
        self.state = ParserState(max_depth=max_depth)

    def reset(self):
        """Reset to initial state."""
        self.state = ParserState(max_depth=self.max_depth)

    def get_valid_token_types(self) -> Set[TokenType]:
        """Get set of valid next token types based on current state."""
        s = self.state

        if s.state == JSONState.ERROR:
            return set()

        if s.state == JSONState.COMPLETE:
            return set()  # Nothing more allowed

        if s.state == JSONState.START:
            return {TokenType.LBRACE, TokenType.WHITESPACE}

        if s.state == JSONState.IN_OBJECT or s.state == JSONState.EXPECT_KEY:
            valid = {TokenType.QUOTE, TokenType.WHITESPACE}
            # Allow closing if we have required fields or at nested level
            if s.depth > 1 or "label" in s.seen_fields:
                valid.add(TokenType.RBRACE)
            return valid

        if s.state == JSONState.IN_KEY:
            return {TokenType.STRING, TokenType.QUOTE}

        if s.state == JSONState.AFTER_KEY:
            return {TokenType.COLON, TokenType.WHITESPACE}

        if s.state == JSONState.EXPECT_COLON:
            return {TokenType.COLON, TokenType.WHITESPACE}

        if s.state == JSONState.EXPECT_VALUE:
            valid = {TokenType.QUOTE, TokenType.NUMBER, TokenType.TRUE,
                    TokenType.FALSE, TokenType.NULL, TokenType.LBRACE,
                    TokenType.LBRACKET, TokenType.WHITESPACE}
            return valid

        if s.state == JSONState.IN_STRING_VALUE:
            return {TokenType.STRING, TokenType.QUOTE}

        if s.state == JSONState.AFTER_VALUE:
            valid = {TokenType.WHITESPACE}
            if s.stack and s.stack[-1] == "object":
                valid.add(TokenType.COMMA)
                valid.add(TokenType.RBRACE)
            elif s.stack and s.stack[-1] == "array":
                valid.add(TokenType.COMMA)
                valid.add(TokenType.RBRACKET)
            return valid

        if s.state == JSONState.IN_ARRAY or s.state == JSONState.EXPECT_ELEMENT:
            valid = {TokenType.QUOTE, TokenType.NUMBER, TokenType.TRUE,
                    TokenType.FALSE, TokenType.NULL, TokenType.LBRACE,
                    TokenType.LBRACKET, TokenType.RBRACKET, TokenType.WHITESPACE}
            return valid

        if s.state == JSONState.AFTER_ELEMENT:
            return {TokenType.COMMA, TokenType.RBRACKET, TokenType.WHITESPACE}

        return {TokenType.OTHER}

    def should_force_close(self) -> bool:
        """Check if we should force closing braces."""
        s = self.state
        # Force close if at max depth
        if s.depth >= s.max_depth:
            return True
        # Force close if too many states
        if s.state_count >= self.max_states:
            return True
        # Force close if too many array elements (repetition detection)
        if s.array_element_count >= s.max_array_elements:
            return True
        # Force close if too many tokens processed
        if s.total_tokens >= s.max_tokens:
            return True
        return False

    def process_token(self, token: str) -> bool:
        """
        Process a token and update state.

        Returns True if transition was valid, False otherwise.
        """
        s = self.state
        s.total_tokens += 1  # Track total tokens
        token_type = self._classify_token(token)

        # Check if token type is valid
        valid_types = self.get_valid_token_types()
        if token_type not in valid_types and token_type != TokenType.WHITESPACE:
            # Allow whitespace almost anywhere
            if token_type != TokenType.OTHER:
                s.state = JSONState.ERROR
                return False

        # State transitions
        if s.state == JSONState.START:
            if token == "{":
                s.state = JSONState.EXPECT_KEY
                s.stack.append("object")
                s.depth = 1
                return True

        elif s.state in (JSONState.IN_OBJECT, JSONState.EXPECT_KEY):
            if token == '"':
                s.state = JSONState.IN_KEY
                s.string_buffer = ""
                s.in_string = True
                return True
            elif token == "}":
                return self._close_object()

        elif s.state == JSONState.IN_KEY:
            if token == '"':
                s.current_key = s.string_buffer
                s.seen_fields.add(s.current_key)
                s.in_string = False
                s.state = JSONState.EXPECT_COLON
                # Track SC-specific context
                if s.current_key == "root_state":
                    s.in_root_state = True
                elif s.current_key == "children":
                    s.in_children = True
                elif s.current_key == "label":
                    s.state_count += 1
                return True
            else:
                s.string_buffer += token
                return True

        elif s.state == JSONState.EXPECT_COLON:
            if token == ":":
                s.state = JSONState.EXPECT_VALUE
                return True

        elif s.state == JSONState.EXPECT_VALUE:
            if token == '"':
                s.state = JSONState.IN_STRING_VALUE
                s.string_buffer = ""
                s.in_string = True
                return True
            elif token == "{":
                s.state = JSONState.EXPECT_KEY
                s.stack.append("object")
                s.depth += 1
                return True
            elif token == "[":
                s.state = JSONState.EXPECT_ELEMENT
                s.stack.append("array")
                s.depth += 1
                return True
            elif token in ("true", "false", "null") or self._is_number(token):
                s.state = JSONState.AFTER_VALUE
                return True

        elif s.state == JSONState.IN_STRING_VALUE:
            if token == '"':
                s.in_string = False
                s.state = JSONState.AFTER_VALUE
                return True
            else:
                s.string_buffer += token
                return True

        elif s.state == JSONState.AFTER_VALUE:
            if token == ",":
                if s.stack and s.stack[-1] == "object":
                    s.state = JSONState.EXPECT_KEY
                elif s.stack and s.stack[-1] == "array":
                    s.state = JSONState.EXPECT_ELEMENT
                    s.array_element_count += 1  # Count array elements
                s.current_key = ""
                return True
            elif token == "}":
                return self._close_object()
            elif token == "]":
                return self._close_array()

        elif s.state in (JSONState.IN_ARRAY, JSONState.EXPECT_ELEMENT):
            if token == '"':
                s.state = JSONState.IN_STRING_VALUE
                s.string_buffer = ""
                s.in_string = True
                return True
            elif token == "{":
                s.state = JSONState.EXPECT_KEY
                s.stack.append("object")
                s.depth += 1
                return True
            elif token == "[":
                s.state = JSONState.EXPECT_ELEMENT
                s.stack.append("array")
                s.depth += 1
                return True
            elif token == "]":
                return self._close_array()
            elif token in ("true", "false", "null") or self._is_number(token):
                s.state = JSONState.AFTER_ELEMENT
                return True

        elif s.state == JSONState.AFTER_ELEMENT:
            if token == ",":
                s.state = JSONState.EXPECT_ELEMENT
                s.array_element_count += 1  # Count array elements
                return True
            elif token == "]":
                return self._close_array()

        # Whitespace is generally allowed
        if token.strip() == "":
            return True

        return True

    def _close_object(self) -> bool:
        """Close current object."""
        s = self.state
        if s.stack and s.stack[-1] == "object":
            s.stack.pop()
            s.depth -= 1
            if s.depth == 0:
                s.state = JSONState.COMPLETE
            elif s.stack and s.stack[-1] == "object":
                s.state = JSONState.AFTER_VALUE
            elif s.stack and s.stack[-1] == "array":
                s.state = JSONState.AFTER_ELEMENT
            return True
        return False

    def _close_array(self) -> bool:
        """Close current array."""
        s = self.state
        if s.stack and s.stack[-1] == "array":
            s.stack.pop()
            s.depth -= 1
            s.in_children = False
            if s.stack and s.stack[-1] == "object":
                s.state = JSONState.AFTER_VALUE
            elif s.stack and s.stack[-1] == "array":
                s.state = JSONState.AFTER_ELEMENT
            return True
        return False

    def _classify_token(self, token: str) -> TokenType:
        """Classify a token."""
        if token == "{":
            return TokenType.LBRACE
        elif token == "}":
            return TokenType.RBRACE
        elif token == "[":
            return TokenType.LBRACKET
        elif token == "]":
            return TokenType.RBRACKET
        elif token == ":":
            return TokenType.COLON
        elif token == ",":
            return TokenType.COMMA
        elif token == '"':
            return TokenType.QUOTE
        elif token in ("true", "false"):
            return TokenType.TRUE if token == "true" else TokenType.FALSE
        elif token == "null":
            return TokenType.NULL
        elif token.strip() == "":
            return TokenType.WHITESPACE
        elif self._is_number(token):
            return TokenType.NUMBER
        else:
            return TokenType.STRING

    def _is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def is_complete(self) -> bool:
        """Check if parsing is complete."""
        return self.state.state == JSONState.COMPLETE

    def is_error(self) -> bool:
        """Check if in error state."""
        return self.state.state == JSONState.ERROR

    def get_closing_sequence(self) -> str:
        """Get sequence of tokens needed to close current structure."""
        s = self.state
        closing = ""

        # If in string, close it
        if s.in_string:
            closing += '"'

        # Close all open structures
        for item in reversed(s.stack):
            if item == "object":
                closing += "}"
            elif item == "array":
                closing += "]"

        return closing


def demo():
    """Demo the JSON schema statechart."""
    print("=" * 60)
    print("JSON SCHEMA STATECHART DEMO")
    print("=" * 60)

    sc = JSONSchemaStatechart(max_depth=5, max_states=10)

    # Test input
    test_json = '{"root_state": {"label": "test", "children": [{"label": "A"}]}}'

    print(f"\nParsing: {test_json}")
    print("\nToken-by-token:")

    i = 0
    while i < len(test_json):
        char = test_json[i]

        # Handle multi-char tokens
        if char == '"':
            sc.process_token(char)
            i += 1
            # Read string content
            string_content = ""
            while i < len(test_json) and test_json[i] != '"':
                string_content += test_json[i]
                i += 1
            if string_content:
                sc.process_token(string_content)
            if i < len(test_json):
                sc.process_token('"')
                i += 1
        else:
            sc.process_token(char)
            i += 1

        if sc.is_error():
            print(f"  ERROR at position {i}")
            break

    print(f"\nFinal state: {sc.state.state.name}")
    print(f"Complete: {sc.is_complete()}")
    print(f"Depth: {sc.state.depth}")
    print(f"State count: {sc.state.state_count}")
    print(f"Seen fields: {sc.state.seen_fields}")

    # Test partial input
    print("\n" + "-" * 40)
    print("Testing partial input...")

    sc.reset()
    partial = '{"root_state": {"label": "test", "children": ['

    # Parse partial
    i = 0
    while i < len(partial):
        char = partial[i]
        if char == '"':
            sc.process_token(char)
            i += 1
            string_content = ""
            while i < len(partial) and partial[i] != '"':
                string_content += partial[i]
                i += 1
            if string_content:
                sc.process_token(string_content)
            if i < len(partial):
                sc.process_token('"')
                i += 1
        else:
            sc.process_token(char)
            i += 1

    print(f"After partial: state={sc.state.state.name}, depth={sc.state.depth}")
    print(f"Valid next tokens: {[t.name for t in sc.get_valid_token_types()]}")
    print(f"Closing sequence needed: {sc.get_closing_sequence()}")


if __name__ == "__main__":
    demo()
