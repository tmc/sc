"""
JSON Grammar as a Finite State Machine

Models the complete JSON grammar as a statechart/FSM for constrained decoding.
Each state represents a parse position, transitions are valid next tokens.

Grammar based on ECMA-404 JSON standard.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple


class JSONState(Enum):
    """States in the JSON grammar FSM."""
    # Initial state
    START = auto()

    # Value states
    VALUE = auto()  # Expecting any value

    # Object states
    OBJECT_START = auto()      # After {
    OBJECT_KEY = auto()        # Expecting key string
    OBJECT_COLON = auto()      # After key, expecting :
    OBJECT_VALUE = auto()      # After :, expecting value
    OBJECT_COMMA = auto()      # After value, expecting , or }

    # Array states
    ARRAY_START = auto()       # After [
    ARRAY_VALUE = auto()       # Expecting array element
    ARRAY_COMMA = auto()       # After value, expecting , or ]

    # String states
    STRING_START = auto()      # After opening "
    STRING_CONTENT = auto()    # Inside string
    STRING_ESCAPE = auto()     # After \
    STRING_UNICODE = auto()    # After \u, expecting 4 hex digits

    # Number states
    NUMBER_START = auto()      # Start of number
    NUMBER_SIGN = auto()       # After -
    NUMBER_ZERO = auto()       # After leading 0
    NUMBER_INT = auto()        # Integer digits
    NUMBER_DOT = auto()        # After .
    NUMBER_FRAC = auto()       # Fractional digits
    NUMBER_EXP = auto()        # After e/E
    NUMBER_EXP_SIGN = auto()   # After e+/-
    NUMBER_EXP_DIGITS = auto() # Exponent digits

    # Literal states
    LITERAL_T = auto()         # true: t
    LITERAL_TR = auto()        # true: tr
    LITERAL_TRU = auto()       # true: tru
    LITERAL_F = auto()         # false: f
    LITERAL_FA = auto()        # false: fa
    LITERAL_FAL = auto()       # false: fal
    LITERAL_FALS = auto()      # false: fals
    LITERAL_N = auto()         # null: n
    LITERAL_NU = auto()        # null: nu
    LITERAL_NUL = auto()       # null: nul

    # Terminal states
    END = auto()               # Valid JSON complete
    ERROR = auto()             # Invalid state


@dataclass
class JSONTransition:
    """A transition in the JSON grammar FSM."""
    from_state: JSONState
    to_state: JSONState
    tokens: Set[str]  # Valid tokens for this transition
    description: str = ""


@dataclass
class JSONGrammar:
    """
    Complete JSON grammar as a finite state machine.

    Can be used to:
    1. Validate JSON token sequences
    2. Get valid next tokens at any parse position
    3. Constrain LLM generation to valid JSON
    """

    transitions: Dict[JSONState, List[JSONTransition]] = field(default_factory=dict)

    # Token categories
    WHITESPACE: Set[str] = field(default_factory=lambda: {' ', '\t', '\n', '\r'})
    DIGITS: Set[str] = field(default_factory=lambda: set('0123456789'))
    HEX_DIGITS: Set[str] = field(default_factory=lambda: set('0123456789abcdefABCDEF'))

    def __post_init__(self):
        """Build the grammar transitions."""
        self._build_transitions()

    def _build_transitions(self):
        """Build all grammar transitions."""
        t = self.transitions

        # START state - JSON must start with value
        t[JSONState.START] = [
            JSONTransition(JSONState.START, JSONState.OBJECT_START, {'{'}, "Start object"),
            JSONTransition(JSONState.START, JSONState.ARRAY_START, {'['}, "Start array"),
            JSONTransition(JSONState.START, JSONState.STRING_START, {'"'}, "Start string"),
            JSONTransition(JSONState.START, JSONState.NUMBER_START, set('-0123456789'), "Start number"),
            JSONTransition(JSONState.START, JSONState.LITERAL_T, {'t'}, "Start true"),
            JSONTransition(JSONState.START, JSONState.LITERAL_F, {'f'}, "Start false"),
            JSONTransition(JSONState.START, JSONState.LITERAL_N, {'n'}, "Start null"),
            JSONTransition(JSONState.START, JSONState.START, self.WHITESPACE, "Skip whitespace"),
        ]

        # OBJECT states
        t[JSONState.OBJECT_START] = [
            JSONTransition(JSONState.OBJECT_START, JSONState.STRING_START, {'"'}, "Object key"),
            JSONTransition(JSONState.OBJECT_START, JSONState.END, {'}'}, "Empty object"),
            JSONTransition(JSONState.OBJECT_START, JSONState.OBJECT_START, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.OBJECT_COLON] = [
            JSONTransition(JSONState.OBJECT_COLON, JSONState.OBJECT_VALUE, {':'}, "Colon"),
            JSONTransition(JSONState.OBJECT_COLON, JSONState.OBJECT_COLON, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.OBJECT_VALUE] = [
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.OBJECT_START, {'{'}, "Nested object"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.ARRAY_START, {'['}, "Nested array"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.STRING_START, {'"'}, "String value"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.NUMBER_START, set('-0123456789'), "Number value"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.LITERAL_T, {'t'}, "true value"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.LITERAL_F, {'f'}, "false value"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.LITERAL_N, {'n'}, "null value"),
            JSONTransition(JSONState.OBJECT_VALUE, JSONState.OBJECT_VALUE, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.OBJECT_COMMA] = [
            JSONTransition(JSONState.OBJECT_COMMA, JSONState.OBJECT_KEY, {','}, "Next key"),
            JSONTransition(JSONState.OBJECT_COMMA, JSONState.END, {'}'}, "End object"),
            JSONTransition(JSONState.OBJECT_COMMA, JSONState.OBJECT_COMMA, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.OBJECT_KEY] = [
            JSONTransition(JSONState.OBJECT_KEY, JSONState.STRING_START, {'"'}, "Key string"),
            JSONTransition(JSONState.OBJECT_KEY, JSONState.OBJECT_KEY, self.WHITESPACE, "Skip whitespace"),
        ]

        # ARRAY states
        t[JSONState.ARRAY_START] = [
            JSONTransition(JSONState.ARRAY_START, JSONState.OBJECT_START, {'{'}, "Object element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.ARRAY_START, {'['}, "Nested array"),
            JSONTransition(JSONState.ARRAY_START, JSONState.STRING_START, {'"'}, "String element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.NUMBER_START, set('-0123456789'), "Number element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.LITERAL_T, {'t'}, "true element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.LITERAL_F, {'f'}, "false element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.LITERAL_N, {'n'}, "null element"),
            JSONTransition(JSONState.ARRAY_START, JSONState.END, {']'}, "Empty array"),
            JSONTransition(JSONState.ARRAY_START, JSONState.ARRAY_START, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.ARRAY_COMMA] = [
            JSONTransition(JSONState.ARRAY_COMMA, JSONState.ARRAY_VALUE, {','}, "Next element"),
            JSONTransition(JSONState.ARRAY_COMMA, JSONState.END, {']'}, "End array"),
            JSONTransition(JSONState.ARRAY_COMMA, JSONState.ARRAY_COMMA, self.WHITESPACE, "Skip whitespace"),
        ]

        t[JSONState.ARRAY_VALUE] = [
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.OBJECT_START, {'{'}, "Object element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.ARRAY_START, {'['}, "Nested array"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.STRING_START, {'"'}, "String element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.NUMBER_START, set('-0123456789'), "Number element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.LITERAL_T, {'t'}, "true element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.LITERAL_F, {'f'}, "false element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.LITERAL_N, {'n'}, "null element"),
            JSONTransition(JSONState.ARRAY_VALUE, JSONState.ARRAY_VALUE, self.WHITESPACE, "Skip whitespace"),
        ]

        # STRING states
        t[JSONState.STRING_START] = [
            JSONTransition(JSONState.STRING_START, JSONState.STRING_CONTENT, None, "String content"),  # Any char except " and \
            JSONTransition(JSONState.STRING_START, JSONState.STRING_ESCAPE, {'\\'}, "Escape"),
            JSONTransition(JSONState.STRING_START, JSONState.END, {'"'}, "Empty string"),
        ]

        t[JSONState.STRING_CONTENT] = [
            JSONTransition(JSONState.STRING_CONTENT, JSONState.STRING_CONTENT, None, "More content"),
            JSONTransition(JSONState.STRING_CONTENT, JSONState.STRING_ESCAPE, {'\\'}, "Escape"),
            JSONTransition(JSONState.STRING_CONTENT, JSONState.END, {'"'}, "End string"),
        ]

        t[JSONState.STRING_ESCAPE] = [
            JSONTransition(JSONState.STRING_ESCAPE, JSONState.STRING_CONTENT,
                          {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}, "Escape char"),
            JSONTransition(JSONState.STRING_ESCAPE, JSONState.STRING_UNICODE, {'u'}, "Unicode escape"),
        ]

        t[JSONState.STRING_UNICODE] = [
            JSONTransition(JSONState.STRING_UNICODE, JSONState.STRING_CONTENT, self.HEX_DIGITS, "Hex digit"),
        ]

        # NUMBER states
        t[JSONState.NUMBER_START] = [
            JSONTransition(JSONState.NUMBER_START, JSONState.NUMBER_SIGN, {'-'}, "Negative"),
            JSONTransition(JSONState.NUMBER_START, JSONState.NUMBER_ZERO, {'0'}, "Leading zero"),
            JSONTransition(JSONState.NUMBER_START, JSONState.NUMBER_INT, set('123456789'), "Integer"),
        ]

        t[JSONState.NUMBER_SIGN] = [
            JSONTransition(JSONState.NUMBER_SIGN, JSONState.NUMBER_ZERO, {'0'}, "Leading zero"),
            JSONTransition(JSONState.NUMBER_SIGN, JSONState.NUMBER_INT, set('123456789'), "Integer"),
        ]

        t[JSONState.NUMBER_ZERO] = [
            JSONTransition(JSONState.NUMBER_ZERO, JSONState.NUMBER_DOT, {'.'}, "Decimal"),
            JSONTransition(JSONState.NUMBER_ZERO, JSONState.NUMBER_EXP, {'e', 'E'}, "Exponent"),
            JSONTransition(JSONState.NUMBER_ZERO, JSONState.END, None, "End number"),  # Implicit end
        ]

        t[JSONState.NUMBER_INT] = [
            JSONTransition(JSONState.NUMBER_INT, JSONState.NUMBER_INT, self.DIGITS, "More digits"),
            JSONTransition(JSONState.NUMBER_INT, JSONState.NUMBER_DOT, {'.'}, "Decimal"),
            JSONTransition(JSONState.NUMBER_INT, JSONState.NUMBER_EXP, {'e', 'E'}, "Exponent"),
            JSONTransition(JSONState.NUMBER_INT, JSONState.END, None, "End number"),
        ]

        t[JSONState.NUMBER_DOT] = [
            JSONTransition(JSONState.NUMBER_DOT, JSONState.NUMBER_FRAC, self.DIGITS, "Fraction"),
        ]

        t[JSONState.NUMBER_FRAC] = [
            JSONTransition(JSONState.NUMBER_FRAC, JSONState.NUMBER_FRAC, self.DIGITS, "More fraction"),
            JSONTransition(JSONState.NUMBER_FRAC, JSONState.NUMBER_EXP, {'e', 'E'}, "Exponent"),
            JSONTransition(JSONState.NUMBER_FRAC, JSONState.END, None, "End number"),
        ]

        t[JSONState.NUMBER_EXP] = [
            JSONTransition(JSONState.NUMBER_EXP, JSONState.NUMBER_EXP_SIGN, {'+', '-'}, "Exp sign"),
            JSONTransition(JSONState.NUMBER_EXP, JSONState.NUMBER_EXP_DIGITS, self.DIGITS, "Exp digits"),
        ]

        t[JSONState.NUMBER_EXP_SIGN] = [
            JSONTransition(JSONState.NUMBER_EXP_SIGN, JSONState.NUMBER_EXP_DIGITS, self.DIGITS, "Exp digits"),
        ]

        t[JSONState.NUMBER_EXP_DIGITS] = [
            JSONTransition(JSONState.NUMBER_EXP_DIGITS, JSONState.NUMBER_EXP_DIGITS, self.DIGITS, "More exp"),
            JSONTransition(JSONState.NUMBER_EXP_DIGITS, JSONState.END, None, "End number"),
        ]

        # LITERAL states
        t[JSONState.LITERAL_T] = [
            JSONTransition(JSONState.LITERAL_T, JSONState.LITERAL_TR, {'r'}, "true: r"),
        ]
        t[JSONState.LITERAL_TR] = [
            JSONTransition(JSONState.LITERAL_TR, JSONState.LITERAL_TRU, {'u'}, "true: u"),
        ]
        t[JSONState.LITERAL_TRU] = [
            JSONTransition(JSONState.LITERAL_TRU, JSONState.END, {'e'}, "true: e"),
        ]

        t[JSONState.LITERAL_F] = [
            JSONTransition(JSONState.LITERAL_F, JSONState.LITERAL_FA, {'a'}, "false: a"),
        ]
        t[JSONState.LITERAL_FA] = [
            JSONTransition(JSONState.LITERAL_FA, JSONState.LITERAL_FAL, {'l'}, "false: l"),
        ]
        t[JSONState.LITERAL_FAL] = [
            JSONTransition(JSONState.LITERAL_FAL, JSONState.LITERAL_FALS, {'s'}, "false: s"),
        ]
        t[JSONState.LITERAL_FALS] = [
            JSONTransition(JSONState.LITERAL_FALS, JSONState.END, {'e'}, "false: e"),
        ]

        t[JSONState.LITERAL_N] = [
            JSONTransition(JSONState.LITERAL_N, JSONState.LITERAL_NU, {'u'}, "null: u"),
        ]
        t[JSONState.LITERAL_NU] = [
            JSONTransition(JSONState.LITERAL_NU, JSONState.LITERAL_NUL, {'l'}, "null: l"),
        ]
        t[JSONState.LITERAL_NUL] = [
            JSONTransition(JSONState.LITERAL_NUL, JSONState.END, {'l'}, "null: l"),
        ]

    def get_valid_tokens(self, state: JSONState) -> Set[str]:
        """Get all valid tokens from a state."""
        if state not in self.transitions:
            return set()

        valid = set()
        for trans in self.transitions[state]:
            if trans.tokens is not None:
                valid.update(trans.tokens)
        return valid

    def transition(self, state: JSONState, token: str) -> Optional[JSONState]:
        """Get next state given current state and token."""
        if state not in self.transitions:
            return JSONState.ERROR

        for trans in self.transitions[state]:
            if trans.tokens is None:  # Wildcard transition
                continue
            if token in trans.tokens:
                return trans.to_state

        # Check for wildcard (None tokens = any char)
        for trans in self.transitions[state]:
            if trans.tokens is None:
                return trans.to_state

        return JSONState.ERROR


class JSONParser:
    """
    Incremental JSON parser using the grammar FSM.

    Tracks parse state and context stack for nested structures.
    """

    def __init__(self):
        self.grammar = JSONGrammar()
        self.state = JSONState.START
        self.stack: List[Tuple[JSONState, str]] = []  # (return_state, context_type)
        self.unicode_count = 0  # Track hex digits in \uXXXX

    def reset(self):
        """Reset parser to initial state."""
        self.state = JSONState.START
        self.stack.clear()
        self.unicode_count = 0

    def feed(self, token: str) -> bool:
        """
        Feed a token to the parser.

        Returns True if token was valid, False otherwise.
        """
        # Handle unicode escape counting
        if self.state == JSONState.STRING_UNICODE:
            if token in self.grammar.HEX_DIGITS:
                self.unicode_count += 1
                if self.unicode_count >= 4:
                    self.state = JSONState.STRING_CONTENT
                    self.unicode_count = 0
                return True
            return False

        next_state = self.grammar.transition(self.state, token)

        if next_state == JSONState.ERROR:
            return False

        # Handle context push/pop
        if next_state == JSONState.OBJECT_START and self.state != JSONState.OBJECT_START:
            self.stack.append((self._get_return_state(), 'object'))
        elif next_state == JSONState.ARRAY_START and self.state != JSONState.ARRAY_START:
            self.stack.append((self._get_return_state(), 'array'))
        elif next_state == JSONState.END:
            if token == '}':
                if self.stack and self.stack[-1][1] == 'object':
                    return_state, _ = self.stack.pop()
                    next_state = self._get_post_value_state()
            elif token == ']':
                if self.stack and self.stack[-1][1] == 'array':
                    return_state, _ = self.stack.pop()
                    next_state = self._get_post_value_state()
            elif token == '"':
                # String ended - determine context
                next_state = self._get_post_string_state()

        self.state = next_state
        return True

    def _get_return_state(self) -> JSONState:
        """Get state to return to after nested value."""
        if not self.stack:
            return JSONState.END
        _, context = self.stack[-1]
        if context == 'object':
            return JSONState.OBJECT_COMMA
        return JSONState.ARRAY_COMMA

    def _get_post_value_state(self) -> JSONState:
        """Get state after completing a value."""
        if not self.stack:
            return JSONState.END
        _, context = self.stack[-1]
        if context == 'object':
            return JSONState.OBJECT_COMMA
        return JSONState.ARRAY_COMMA

    def _get_post_string_state(self) -> JSONState:
        """Get state after completing a string."""
        if not self.stack:
            return JSONState.END
        _, context = self.stack[-1]
        if context == 'object':
            # Check if this was a key or value
            if self.state in (JSONState.STRING_START, JSONState.STRING_CONTENT):
                # Depends on previous state - simplified: assume key if after OBJECT_START
                return JSONState.OBJECT_COLON
            return JSONState.OBJECT_COMMA
        return JSONState.ARRAY_COMMA

    def get_valid_next_tokens(self) -> Set[str]:
        """Get valid tokens at current parse position."""
        return self.grammar.get_valid_tokens(self.state)

    def is_complete(self) -> bool:
        """Check if parser is in valid end state."""
        return self.state == JSONState.END and len(self.stack) == 0


def get_valid_next_tokens(partial_json: str) -> Set[str]:
    """
    Get valid next tokens for a partial JSON string.

    Args:
        partial_json: JSON string parsed so far

    Returns:
        Set of valid next characters/tokens
    """
    parser = JSONParser()

    for char in partial_json:
        if not parser.feed(char):
            return set()  # Invalid JSON

    return parser.get_valid_next_tokens()


def validate_json_grammar(json_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON using the grammar FSM.

    Returns:
        (is_valid, error_message)
    """
    parser = JSONParser()

    for i, char in enumerate(json_str):
        if not parser.feed(char):
            return False, f"Invalid character '{char}' at position {i}"

    if not parser.is_complete():
        return False, "Incomplete JSON"

    return True, None
