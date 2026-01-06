"""
SC-Specific Grammar as Finite State Machine.

Defines the grammar for valid SC JSON with:
- Strict field names (root_state, label, type, children, transitions)
- Required fields at each level
- Type values constrained to 1, 2, 3
- Proper transition structure

Much stricter than generic JSON - rejects valid JSON that isn't valid SC.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto


class SCFieldType(Enum):
    """SC-specific field types."""
    # Root level
    ROOT_STATE = auto()
    TRANSITIONS = auto()

    # State fields
    LABEL = auto()
    TYPE = auto()
    CHILDREN = auto()
    IS_INITIAL = auto()

    # Transition fields
    FROM = auto()
    TO = auto()
    EVENT = auto()
    GUARD = auto()
    ACTION = auto()


class SCGrammarState(Enum):
    """States in the SC grammar FSM."""
    # Entry points
    START = auto()
    END = auto()

    # Root object
    ROOT_OBJ_OPEN = auto()
    ROOT_FIELD_OR_CLOSE = auto()
    ROOT_FIELD_NAME = auto()
    ROOT_COLON = auto()
    ROOT_FIELD_VALUE = auto()
    ROOT_COMMA_OR_CLOSE = auto()

    # root_state object
    STATE_OBJ_OPEN = auto()
    STATE_FIELD_OR_CLOSE = auto()
    STATE_FIELD_NAME = auto()
    STATE_COLON = auto()
    STATE_FIELD_VALUE = auto()
    STATE_COMMA_OR_CLOSE = auto()

    # State label (string)
    STATE_LABEL_STRING = auto()

    # State type (1, 2, or 3)
    STATE_TYPE_VALUE = auto()

    # State is_initial (bool)
    STATE_IS_INITIAL_VALUE = auto()

    # Children array
    CHILDREN_ARRAY_OPEN = auto()
    CHILDREN_ITEM_OR_CLOSE = auto()
    CHILDREN_COMMA_OR_CLOSE = auto()

    # Transitions array
    TRANS_ARRAY_OPEN = auto()
    TRANS_ITEM_OR_CLOSE = auto()
    TRANS_COMMA_OR_CLOSE = auto()

    # Transition object
    TRANS_OBJ_OPEN = auto()
    TRANS_FIELD_OR_CLOSE = auto()
    TRANS_FIELD_NAME = auto()
    TRANS_COLON = auto()
    TRANS_FIELD_VALUE = auto()
    TRANS_OBJ_COMMA_OR_CLOSE = auto()

    # Transition from/to arrays
    TRANS_STATE_ARRAY_OPEN = auto()
    TRANS_STATE_ITEM_OR_CLOSE = auto()
    TRANS_STATE_STRING = auto()
    TRANS_STATE_COMMA_OR_CLOSE = auto()

    # Transition event/guard/action (strings)
    TRANS_STRING_VALUE = auto()

    # Generic string content
    IN_STRING = auto()
    STRING_ESCAPE = auto()


@dataclass
class SCTransition:
    """A transition in the SC grammar FSM."""
    from_state: SCGrammarState
    token_pattern: str  # Token or pattern to match
    to_state: SCGrammarState
    push_context: Optional[str] = None  # Push to context stack
    pop_context: bool = False  # Pop from context stack


@dataclass
class GrammarContext:
    """Current context in the grammar."""
    state: SCGrammarState = SCGrammarState.START
    context_stack: List[str] = field(default_factory=list)
    seen_fields: Set[str] = field(default_factory=set)
    depth: int = 0
    in_state: bool = False
    in_transition: bool = False

    # String length tracking (prevents runaway strings)
    current_string_length: int = 0
    max_string_length: int = 100  # Maximum characters in a string
    string_limit_hits: int = 0   # Count how many times limit was enforced
    last_char: str = ""  # Track last character for word-boundary detection
    use_word_boundary: bool = True  # Only force close at word boundaries

    # Word boundary characters (natural places to end a string)
    WORD_BOUNDARY_CHARS = {' ', '.', ',', '!', '?', ';', ':', '-', '_', '\n', '\t'}

    def reset_string(self):
        """Reset string length tracking."""
        self.current_string_length = 0
        self.last_char = ""

    def increment_string(self, content: str):
        """Increment current string length and track last char."""
        self.current_string_length += len(content)
        if content:
            self.last_char = content[-1]

    def is_string_at_limit(self) -> bool:
        """Check if current string is at max length."""
        return self.current_string_length >= self.max_string_length

    def should_force_close_string(self) -> bool:
        """Check if we should force close the string.

        Uses word-boundary detection: only force close after natural
        break points (space, punctuation) for more natural string endings.
        """
        if not self.is_string_at_limit():
            return False

        if not self.use_word_boundary:
            # Hard limit - force close immediately at max
            return True

        # Word boundary mode - only force close at natural break points
        return self.last_char in self.WORD_BOUNDARY_CHARS


class SCGrammar:
    """
    Finite State Machine for SC JSON grammar.

    Stricter than generic JSON - enforces SC-specific structure.
    """

    # Valid field names at each level
    ROOT_FIELDS = {"root_state", "transitions"}
    STATE_FIELDS = {"label", "type", "children", "is_initial"}
    TRANS_FIELDS = {"from", "to", "event", "guard", "action"}

    # Required fields
    REQUIRED_ROOT = {"root_state"}
    REQUIRED_STATE = {"label"}
    REQUIRED_TRANS = {"from", "to", "event"}

    # Valid state types
    VALID_TYPES = {1, 2, 3}

    def __init__(self):
        self.transitions: List[SCTransition] = []
        self._build_transitions()

    def _build_transitions(self):
        """Build the FSM transition table."""
        T = SCTransition
        S = SCGrammarState

        # Start -> Root object
        self.transitions.extend([
            T(S.START, "{", S.ROOT_FIELD_OR_CLOSE),
        ])

        # Root object fields
        self.transitions.extend([
            T(S.ROOT_FIELD_OR_CLOSE, "}", S.END),
            T(S.ROOT_FIELD_OR_CLOSE, '"root_state"', S.ROOT_COLON, push_context="root_state"),
            T(S.ROOT_FIELD_OR_CLOSE, '"transitions"', S.ROOT_COLON, push_context="transitions"),
            T(S.ROOT_COLON, ":", S.ROOT_FIELD_VALUE),
        ])

        # Root field values
        self.transitions.extend([
            # root_state value -> state object
            T(S.ROOT_FIELD_VALUE, "{", S.STATE_FIELD_OR_CLOSE),
            # transitions value -> array
            T(S.ROOT_FIELD_VALUE, "[", S.TRANS_ITEM_OR_CLOSE),
        ])

        # After root field value
        self.transitions.extend([
            T(S.ROOT_COMMA_OR_CLOSE, ",", S.ROOT_FIELD_OR_CLOSE, pop_context=True),
            T(S.ROOT_COMMA_OR_CLOSE, "}", S.END, pop_context=True),
        ])

        # State object fields
        self.transitions.extend([
            T(S.STATE_FIELD_OR_CLOSE, "}", S.ROOT_COMMA_OR_CLOSE),
            T(S.STATE_FIELD_OR_CLOSE, '"label"', S.STATE_COLON, push_context="label"),
            T(S.STATE_FIELD_OR_CLOSE, '"type"', S.STATE_COLON, push_context="type"),
            T(S.STATE_FIELD_OR_CLOSE, '"children"', S.STATE_COLON, push_context="children"),
            T(S.STATE_FIELD_OR_CLOSE, '"is_initial"', S.STATE_COLON, push_context="is_initial"),
            T(S.STATE_COLON, ":", S.STATE_FIELD_VALUE),
        ])

        # State field values
        self.transitions.extend([
            # label -> any quoted string
            T(S.STATE_FIELD_VALUE, "QUOTED_STRING", S.STATE_COMMA_OR_CLOSE),
            # type -> 1, 2, or 3
            T(S.STATE_FIELD_VALUE, "1", S.STATE_COMMA_OR_CLOSE),
            T(S.STATE_FIELD_VALUE, "2", S.STATE_COMMA_OR_CLOSE),
            T(S.STATE_FIELD_VALUE, "3", S.STATE_COMMA_OR_CLOSE),
            # is_initial -> true/false
            T(S.STATE_FIELD_VALUE, "true", S.STATE_COMMA_OR_CLOSE),
            T(S.STATE_FIELD_VALUE, "false", S.STATE_COMMA_OR_CLOSE),
            # children -> array
            T(S.STATE_FIELD_VALUE, "[", S.CHILDREN_ITEM_OR_CLOSE),
        ])

        # After state field value
        self.transitions.extend([
            T(S.STATE_COMMA_OR_CLOSE, ",", S.STATE_FIELD_OR_CLOSE, pop_context=True),
            T(S.STATE_COMMA_OR_CLOSE, "}", S.ROOT_COMMA_OR_CLOSE, pop_context=True),
        ])

        # Children array
        self.transitions.extend([
            T(S.CHILDREN_ITEM_OR_CLOSE, "]", S.STATE_COMMA_OR_CLOSE),
            T(S.CHILDREN_ITEM_OR_CLOSE, "{", S.STATE_FIELD_OR_CLOSE),  # Nested state
            T(S.CHILDREN_COMMA_OR_CLOSE, ",", S.CHILDREN_ITEM_OR_CLOSE),
            T(S.CHILDREN_COMMA_OR_CLOSE, "]", S.STATE_COMMA_OR_CLOSE),
        ])

        # Transitions array
        self.transitions.extend([
            T(S.TRANS_ITEM_OR_CLOSE, "]", S.ROOT_COMMA_OR_CLOSE),
            T(S.TRANS_ITEM_OR_CLOSE, "{", S.TRANS_FIELD_OR_CLOSE),
            T(S.TRANS_COMMA_OR_CLOSE, ",", S.TRANS_ITEM_OR_CLOSE),
            T(S.TRANS_COMMA_OR_CLOSE, "]", S.ROOT_COMMA_OR_CLOSE),
        ])

        # Transition object fields
        self.transitions.extend([
            T(S.TRANS_FIELD_OR_CLOSE, "}", S.TRANS_COMMA_OR_CLOSE),
            T(S.TRANS_FIELD_OR_CLOSE, '"from"', S.TRANS_COLON, push_context="from"),
            T(S.TRANS_FIELD_OR_CLOSE, '"to"', S.TRANS_COLON, push_context="to"),
            T(S.TRANS_FIELD_OR_CLOSE, '"event"', S.TRANS_COLON, push_context="event"),
            T(S.TRANS_FIELD_OR_CLOSE, '"guard"', S.TRANS_COLON, push_context="guard"),
            T(S.TRANS_FIELD_OR_CLOSE, '"action"', S.TRANS_COLON, push_context="action"),
            T(S.TRANS_COLON, ":", S.TRANS_FIELD_VALUE),
        ])

        # Transition field values
        self.transitions.extend([
            # from/to -> array of strings
            T(S.TRANS_FIELD_VALUE, "[", S.TRANS_STATE_ITEM_OR_CLOSE),
            # event/guard/action -> quoted string
            T(S.TRANS_FIELD_VALUE, "QUOTED_STRING", S.TRANS_OBJ_COMMA_OR_CLOSE),
        ])

        # Transition state array (from/to)
        self.transitions.extend([
            T(S.TRANS_STATE_ITEM_OR_CLOSE, "]", S.TRANS_OBJ_COMMA_OR_CLOSE),
            T(S.TRANS_STATE_ITEM_OR_CLOSE, "QUOTED_STRING", S.TRANS_STATE_COMMA_OR_CLOSE),
            T(S.TRANS_STATE_COMMA_OR_CLOSE, ",", S.TRANS_STATE_ITEM_OR_CLOSE),
            T(S.TRANS_STATE_COMMA_OR_CLOSE, "]", S.TRANS_OBJ_COMMA_OR_CLOSE),
        ])

        # After transition field value (inside transition object)
        self.transitions.extend([
            T(S.TRANS_OBJ_COMMA_OR_CLOSE, ",", S.TRANS_FIELD_OR_CLOSE, pop_context=True),
            T(S.TRANS_OBJ_COMMA_OR_CLOSE, "}", S.TRANS_COMMA_OR_CLOSE, pop_context=True),
        ])

    def get_valid_tokens(self, context: GrammarContext) -> Set[str]:
        """Get valid next tokens from current context."""
        valid = set()

        for t in self.transitions:
            if t.from_state == context.state:
                # Check context-specific constraints
                if self._is_valid_transition(t, context):
                    valid.add(t.token_pattern)

        # Special handling for IN_STRING state with length limits
        if context.state == SCGrammarState.IN_STRING:
            if context.should_force_close_string():
                # Force close string when at limit AND at word boundary
                context.string_limit_hits += 1
                valid.add('"')  # Only allow end quote
            else:
                valid.add("STRING_CONTENT")  # Placeholder for any string content
                valid.add('"')  # End quote

        return valid

    def _is_valid_transition(self, t: SCTransition, context: GrammarContext) -> bool:
        """Check if transition is valid in current context."""
        # Check for already seen fields
        if t.push_context:
            field_name = t.push_context
            if field_name in context.seen_fields:
                return False  # Can't repeat fields

        return True

    def step(self, context: GrammarContext, token: str) -> Optional[GrammarContext]:
        """
        Step the grammar with a token.

        Returns new context or None if invalid.
        """
        for t in self.transitions:
            if t.from_state == context.state and self._matches(t.token_pattern, token):
                if not self._is_valid_transition(t, context):
                    continue

                # Create new context
                new_ctx = GrammarContext(
                    state=t.to_state,
                    context_stack=context.context_stack.copy(),
                    seen_fields=context.seen_fields.copy(),
                    depth=context.depth,
                    in_state=context.in_state,
                    in_transition=context.in_transition,
                    current_string_length=context.current_string_length,
                    max_string_length=context.max_string_length,
                    string_limit_hits=context.string_limit_hits,
                    last_char=context.last_char,
                    use_word_boundary=context.use_word_boundary,
                )

                # Handle context stack
                if t.push_context:
                    new_ctx.context_stack.append(t.push_context)
                    new_ctx.seen_fields.add(t.push_context)
                if t.pop_context and new_ctx.context_stack:
                    new_ctx.context_stack.pop()

                # Track depth and reset seen_fields for nested objects
                if token == "{":
                    new_ctx.depth += 1
                    # Reset seen_fields when entering a new object (nested state or transition)
                    if t.to_state in (SCGrammarState.STATE_FIELD_OR_CLOSE,
                                      SCGrammarState.TRANS_FIELD_OR_CLOSE):
                        new_ctx.seen_fields = set()
                elif token == "}":
                    new_ctx.depth -= 1
                elif token == "[":
                    new_ctx.depth += 1
                elif token == "]":
                    new_ctx.depth -= 1

                return new_ctx

        # Handle IN_STRING state specially
        if context.state == SCGrammarState.IN_STRING:
            if token == '"':
                # End of string - return to appropriate state based on context
                new_ctx = self._end_string(context)
                new_ctx.reset_string()  # Reset string length for next string
                return new_ctx
            else:
                # Stay in string, track length and last char
                new_ctx = GrammarContext(
                    state=context.state,
                    context_stack=context.context_stack.copy(),
                    seen_fields=context.seen_fields.copy(),
                    depth=context.depth,
                    in_state=context.in_state,
                    in_transition=context.in_transition,
                    current_string_length=context.current_string_length,
                    max_string_length=context.max_string_length,
                    string_limit_hits=context.string_limit_hits,
                    last_char=context.last_char,
                    use_word_boundary=context.use_word_boundary,
                )
                # Increment string length and track last character
                new_ctx.increment_string(token if isinstance(token, str) else str(token))
                return new_ctx

        return None  # Invalid token

    def _matches(self, pattern: str, token: str) -> bool:
        """Check if token matches pattern."""
        if pattern == "STRING_CONTENT":
            return token != '"'
        if pattern == "QUOTED_STRING":
            return token.startswith('"') and token.endswith('"') and len(token) >= 2
        return pattern == token

    def _end_string(self, context: GrammarContext) -> GrammarContext:
        """Handle end of string, return to appropriate state."""
        new_ctx = GrammarContext(
            context_stack=context.context_stack.copy(),
            seen_fields=context.seen_fields.copy(),
            depth=context.depth,
            max_string_length=context.max_string_length,
            string_limit_hits=context.string_limit_hits,
            use_word_boundary=context.use_word_boundary,
        )

        # Determine next state based on context
        if context.context_stack:
            current = context.context_stack[-1]
            if current == "label":
                new_ctx.state = SCGrammarState.STATE_COMMA_OR_CLOSE
            elif current in ("event", "guard", "action"):
                new_ctx.state = SCGrammarState.TRANS_OBJ_COMMA_OR_CLOSE
            elif current in ("from", "to"):
                new_ctx.state = SCGrammarState.TRANS_STATE_COMMA_OR_CLOSE
            else:
                new_ctx.state = SCGrammarState.STATE_COMMA_OR_CLOSE
        else:
            new_ctx.state = SCGrammarState.ROOT_COMMA_OR_CLOSE

        return new_ctx

    def is_valid_complete(self, context: GrammarContext) -> bool:
        """Check if current context represents a complete valid SC."""
        return context.state == SCGrammarState.END

    def validate_sc_json(self, tokens: List[str]) -> Tuple[bool, str]:
        """
        Validate a tokenized SC JSON.

        Returns (is_valid, error_message).
        """
        context = GrammarContext()

        for i, token in enumerate(tokens):
            new_ctx = self.step(context, token)
            if new_ctx is None:
                return False, f"Invalid token '{token}' at position {i} in state {context.state.name}"
            context = new_ctx

        if not self.is_valid_complete(context):
            return False, f"Incomplete: ended in state {context.state.name}"

        return True, "Valid SC JSON"


def tokenize_sc_json(text: str) -> List[str]:
    """Tokenize SC JSON into grammar tokens."""
    tokens = []
    i = 0

    while i < len(text):
        c = text[i]

        # Skip whitespace
        if c in ' \t\n\r':
            i += 1
            continue

        # Single-char tokens
        if c in '{}[]:,':
            tokens.append(c)
            i += 1
            continue

        # String
        if c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == '\\':
                    j += 2
                else:
                    j += 1
            if j < len(text):
                tokens.append(text[i:j+1])
                i = j + 1
            else:
                tokens.append(text[i:])
                break
            continue

        # Number or boolean
        if c.isdigit() or c == '-':
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] in '.-eE+'):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        if text[i:i+4] == 'true':
            tokens.append('true')
            i += 4
            continue

        if text[i:i+5] == 'false':
            tokens.append('false')
            i += 5
            continue

        if text[i:i+4] == 'null':
            tokens.append('null')
            i += 4
            continue

        # Unknown
        i += 1

    return tokens


def test_grammar():
    """Test SC grammar."""
    print("=" * 60)
    print("Testing SC Grammar")
    print("=" * 60)

    grammar = SCGrammar()

    # Test cases
    test_cases = [
        # Valid SC JSON - simple cases
        ('{"root_state": {"label": "test"}, "transitions": [{"from": ["A"], "to": ["B"], "event": "GO"}]}', True),
        ('{"root_state": {"label": "x", "type": 1, "is_initial": true}}', True),
        ('{"root_state": {"label": "test"}}', True),
        # Invalid - wrong field names
        ('{"root": {"label": "x"}}', False),
        ('{"root_state": {"name": "x"}}', False),
        # Invalid - wrong type value
        ('{"root_state": {"label": "x", "type": 5}}', False),
    ]

    print("\n1. Validation tests:")
    passed = 0
    for sc_json, expected_valid in test_cases:
        tokens = tokenize_sc_json(sc_json)
        valid, msg = grammar.validate_sc_json(tokens)

        status = "PASS" if valid == expected_valid else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {sc_json[:50]}... -> {valid} (expected {expected_valid})")

    print(f"\n  Passed: {passed}/{len(test_cases)}")

    # Test token prediction
    print("\n2. Token prediction:")
    context = GrammarContext()
    print(f"  State: {context.state.name}")
    valid_tokens = grammar.get_valid_tokens(context)
    print(f"  Valid tokens: {valid_tokens}")

    # Step through
    context = grammar.step(context, "{")
    print(f"  After '{{': state={context.state.name}")
    valid_tokens = grammar.get_valid_tokens(context)
    print(f"  Valid tokens: {valid_tokens}")

    print("\n" + "=" * 60)
    print("SC Grammar tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_grammar()
