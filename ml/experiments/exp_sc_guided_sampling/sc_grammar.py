"""
Statechart JSON Grammar as Finite State Machine

Defines valid token sequences for statechart JSON generation.

GRAMMAR STATES:
- START: Beginning, expect {
- ROOT_OBJ: Inside root object
- ROOT_STATE_KEY: Expect "root_state"
- ROOT_STATE_VALUE: Expect state object
- STATE_OBJ: Inside state object
- LABEL_KEY: Expect "label"
- LABEL_VALUE: Expect string
- TYPE_KEY: Expect "type"
- TYPE_VALUE: Expect number
- CHILDREN_KEY: Expect "children"
- CHILDREN_ARRAY: Inside children array
- TRANSITIONS_KEY: Expect "transitions"
- TRANSITIONS_ARRAY: Inside transitions array
- TRANSITION_OBJ: Inside transition object
- END: Complete

TOKEN CLASSES:
- OPEN_BRACE: {
- CLOSE_BRACE: }
- OPEN_BRACKET: [
- CLOSE_BRACKET: ]
- COLON: :
- COMMA: ,
- QUOTE: "
- STRING_CHAR: any string character
- DIGIT: 0-9
- TRUE/FALSE: true, false
- WHITESPACE: space, newline, tab
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum, auto


class GrammarState(Enum):
    """States in the SC JSON grammar FSM."""
    START = auto()
    ROOT_OBJ = auto()
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    IN_ARRAY = auto()
    IN_OBJECT = auto()
    AFTER_VALUE = auto()

    # SC-specific states
    ROOT_STATE_KEY = auto()
    ROOT_STATE_OBJ = auto()
    STATE_LABEL = auto()
    STATE_TYPE = auto()
    STATE_CHILDREN = auto()
    STATE_IS_INITIAL = auto()
    TRANSITIONS_KEY = auto()
    TRANSITIONS_ARRAY = auto()
    TRANSITION_OBJ = auto()
    TRANSITION_FROM = auto()
    TRANSITION_TO = auto()
    TRANSITION_EVENT = auto()

    END = auto()


class TokenClass(Enum):
    """Token classes for the grammar."""
    OPEN_BRACE = "{"
    CLOSE_BRACE = "}"
    OPEN_BRACKET = "["
    CLOSE_BRACKET = "]"
    COLON = ":"
    COMMA = ","
    QUOTE = '"'
    DIGIT = "digit"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    WHITESPACE = "ws"
    STRING_CHAR = "str"
    MINUS = "-"
    DOT = "."


@dataclass
class ParserState:
    """Current state of the grammar parser."""
    state: GrammarState = GrammarState.START
    stack: List[GrammarState] = field(default_factory=list)
    current_key: str = ""
    in_string: bool = False
    string_buffer: str = ""
    depth: int = 0
    array_depth: int = 0

    # SC-specific tracking
    seen_root_state: bool = False
    seen_transitions: bool = False
    current_state_labels: Set[str] = field(default_factory=set)
    in_state_def: bool = False
    in_transition_def: bool = False


class SCGrammar:
    """
    Statechart JSON grammar as finite state machine.

    Tracks valid next tokens based on current parser state.
    """

    # Required keys in statechart
    SC_REQUIRED_KEYS = {"root_state", "transitions"}
    STATE_REQUIRED_KEYS = {"label", "type"}
    STATE_OPTIONAL_KEYS = {"children", "is_initial"}
    TRANSITION_REQUIRED_KEYS = {"from", "to", "event"}

    # Valid state types
    VALID_STATE_TYPES = {1, 2, 3}  # BASIC, COMPOUND, PARALLEL

    def __init__(self):
        self.state = ParserState()

    def reset(self):
        """Reset parser to initial state."""
        self.state = ParserState()

    def get_valid_tokens(self) -> Set[str]:
        """
        Get set of valid next tokens based on current state.

        Returns tokens that can appear next according to the grammar.
        """
        s = self.state
        valid = set()

        # Whitespace is almost always valid
        valid.add(" ")
        valid.add("\n")
        valid.add("\t")

        if s.in_string:
            # Inside string: any char except unescaped quote
            valid.update(self._string_chars())
            valid.add('"')  # End string
            return valid

        if s.state == GrammarState.START:
            valid.add("{")

        elif s.state == GrammarState.ROOT_OBJ:
            if not s.seen_root_state:
                valid.add('"')  # Start key
            elif not s.seen_transitions:
                valid.add('"')
                valid.add(",")
            else:
                valid.add("}")  # Close root
                valid.add(",")

        elif s.state == GrammarState.EXPECT_KEY:
            valid.add('"')
            if s.depth > 0:
                valid.add("}")  # Empty object OK

        elif s.state == GrammarState.EXPECT_COLON:
            valid.add(":")

        elif s.state == GrammarState.EXPECT_VALUE:
            valid.add("{")  # Object
            valid.add("[")  # Array
            valid.add('"')  # String
            valid.update("0123456789-")  # Number
            valid.add("t")  # true
            valid.add("f")  # false
            valid.add("n")  # null

        elif s.state == GrammarState.IN_ARRAY:
            valid.add("]")  # Close array
            valid.add(",")  # Next element
            valid.add("{")  # Object element
            valid.add("[")  # Nested array
            valid.add('"')  # String element
            valid.update("0123456789-")
            valid.add("t")
            valid.add("f")

        elif s.state == GrammarState.IN_OBJECT:
            valid.add("}")  # Close object
            valid.add(",")  # Next key-value
            valid.add('"')  # Key

        elif s.state == GrammarState.IN_NUMBER:
            valid.update("0123456789")
            valid.add(".")
            valid.add("e")
            valid.add("E")
            valid.add(",")  # End number
            valid.add("}")
            valid.add("]")

        elif s.state == GrammarState.AFTER_VALUE:
            if s.array_depth > 0:
                valid.add(",")
                valid.add("]")
            if s.depth > 0:
                valid.add(",")
                valid.add("}")

        elif s.state == GrammarState.END:
            pass  # No more tokens

        return valid

    def get_valid_token_ids(self, tokenizer: Any) -> List[int]:
        """
        Get valid token IDs for the tokenizer.

        Maps valid characters/strings to token IDs.
        """
        valid_chars = self.get_valid_tokens()
        valid_ids = set()

        # Get vocab
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}

        for token, token_id in vocab.items():
            # Check if token starts with valid char
            if token and token[0] in valid_chars:
                valid_ids.add(token_id)
            # Check for exact matches of multi-char tokens
            if token in valid_chars:
                valid_ids.add(token_id)
            # Check SC-specific keywords
            if self._is_valid_keyword(token):
                valid_ids.add(token_id)

        return list(valid_ids)

    def _is_valid_keyword(self, token: str) -> bool:
        """Check if token is a valid SC keyword."""
        keywords = {
            "root_state", "label", "type", "children", "is_initial",
            "transitions", "from", "to", "event", "guard", "action",
            "true", "false", "null",
            "__root__", "BASIC", "COMPOUND", "PARALLEL",
        }
        # Also allow state name patterns
        if token.startswith("s") and len(token) <= 3:
            return True
        return token in keywords or token.strip('"') in keywords

    def _string_chars(self) -> Set[str]:
        """Get valid string characters."""
        chars = set()
        # ASCII printable except backslash and quote
        for i in range(32, 127):
            c = chr(i)
            if c not in ('"', '\\'):
                chars.add(c)
        # Escaped chars
        chars.add('\\')
        return chars

    def advance(self, token: str) -> bool:
        """
        Advance parser state with new token.

        Returns True if token is valid, False otherwise.
        """
        s = self.state

        # Handle whitespace
        if token.strip() == "" and not s.in_string:
            return True

        # Handle string state
        if s.in_string:
            if token == '"' and (not s.string_buffer or s.string_buffer[-1] != '\\'):
                s.in_string = False
                self._handle_string_complete(s.string_buffer)
                s.string_buffer = ""
                s.state = GrammarState.AFTER_VALUE
            else:
                s.string_buffer += token
            return True

        # State transitions
        if token == '{':
            s.stack.append(s.state)
            s.depth += 1
            if s.state == GrammarState.START:
                s.state = GrammarState.ROOT_OBJ
            else:
                s.state = GrammarState.IN_OBJECT
            return True

        elif token == '}':
            s.depth -= 1
            if s.stack:
                s.state = s.stack.pop()
            if s.depth == 0:
                s.state = GrammarState.END
            else:
                s.state = GrammarState.AFTER_VALUE
            return True

        elif token == '[':
            s.stack.append(s.state)
            s.array_depth += 1
            s.state = GrammarState.IN_ARRAY
            return True

        elif token == ']':
            s.array_depth -= 1
            if s.stack:
                s.state = s.stack.pop()
            s.state = GrammarState.AFTER_VALUE
            return True

        elif token == '"':
            s.in_string = True
            s.string_buffer = ""
            return True

        elif token == ':':
            s.state = GrammarState.EXPECT_VALUE
            return True

        elif token == ',':
            if s.array_depth > 0:
                s.state = GrammarState.IN_ARRAY
            else:
                s.state = GrammarState.EXPECT_KEY
            return True

        elif token in '0123456789-':
            s.state = GrammarState.IN_NUMBER
            return True

        elif token in '.eE' and s.state == GrammarState.IN_NUMBER:
            return True

        elif token.startswith('t') or token.startswith('f') or token.startswith('n'):
            # true/false/null
            s.state = GrammarState.AFTER_VALUE
            return True

        # Default: check if valid
        return token in self.get_valid_tokens()

    def _handle_string_complete(self, value: str):
        """Handle completed string based on context."""
        s = self.state

        # Track SC-specific fields
        if s.current_key == "label":
            s.current_state_labels.add(value)
        elif value == "root_state":
            s.seen_root_state = True
            s.current_key = "root_state"
        elif value == "transitions":
            s.seen_transitions = True
            s.current_key = "transitions"
        else:
            s.current_key = value

    def is_complete(self) -> bool:
        """Check if parsing is complete."""
        return self.state.state == GrammarState.END

    def is_valid(self) -> bool:
        """Check if current state represents valid SC."""
        s = self.state
        return s.seen_root_state and s.depth == 0


class SCGrammarValidator:
    """
    Validates statechart JSON against grammar.
    """

    def validate(self, json_str: str) -> Tuple[bool, List[str]]:
        """
        Validate JSON string against SC grammar.

        Returns:
            (is_valid, errors)
        """
        grammar = SCGrammar()
        errors = []

        for i, char in enumerate(json_str):
            valid_tokens = grammar.get_valid_tokens()

            if char not in valid_tokens and char.strip():
                errors.append(f"Invalid token '{char}' at position {i}")

            grammar.advance(char)

        if not grammar.is_complete():
            errors.append("JSON incomplete")

        if not grammar.is_valid():
            if not grammar.state.seen_root_state:
                errors.append("Missing root_state")

        return len(errors) == 0, errors


def get_valid_next_tokens(
    partial_json: str,
    tokenizer: Any = None,
) -> Set[str]:
    """
    Get valid next tokens for partial JSON.

    Args:
        partial_json: JSON string generated so far
        tokenizer: Optional tokenizer for token IDs

    Returns:
        Set of valid next characters/tokens
    """
    grammar = SCGrammar()

    # Advance through partial JSON
    for char in partial_json:
        grammar.advance(char)

    return grammar.get_valid_tokens()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate grammar validation."""
    print("=" * 60)
    print("SC JSON Grammar")
    print("=" * 60)

    # Valid statechart
    valid_sc = '''{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "s0", "type": 1, "is_initial": true}]}, "transitions": []}'''

    # Invalid statechart (missing root_state)
    invalid_sc = '''{"states": [{"label": "s0"}]}'''

    validator = SCGrammarValidator()

    print("\nValid SC:")
    is_valid, errors = validator.validate(valid_sc)
    print(f"  Valid: {is_valid}")

    print("\nInvalid SC:")
    is_valid, errors = validator.validate(invalid_sc)
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")

    # Test next token prediction
    print("\nNext Token Prediction:")
    partial = '{"root_state": {"label": "'
    valid_tokens = get_valid_next_tokens(partial)
    print(f"  After '{partial}'")
    print(f"  Valid: {sorted(list(valid_tokens))[:20]}...")

    return validator


if __name__ == "__main__":
    demo()
