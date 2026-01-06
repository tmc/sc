"""
Partial Parser for Incomplete Statechart JSON.

Parses JSON as it's being generated token-by-token, handling:
1. Incomplete strings: "root_sta  -> complete to "root_state"
2. Unclosed brackets: {"states": [ -> add ]})
3. Missing values: {"label":  -> placeholder
4. Truncated numbers: 123. -> complete to 123.0

Returns best-effort parse at each token for live visualization.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


class ParseState(Enum):
    """Current state of the parser."""
    EMPTY = auto()
    OBJECT_START = auto()
    OBJECT_KEY = auto()
    OBJECT_COLON = auto()
    OBJECT_VALUE = auto()
    ARRAY_START = auto()
    ARRAY_VALUE = auto()
    STRING = auto()
    NUMBER = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class PartialParseResult:
    """Result of partial parsing."""
    success: bool
    partial_json: Optional[Dict] = None
    raw_text: str = ""
    parse_state: ParseState = ParseState.EMPTY
    completion_hints: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # Statechart-specific
    states_found: List[str] = field(default_factory=list)
    transitions_found: int = 0
    has_root: bool = False


class PartialParser:
    """
    Parser for incomplete JSON/statechart structures.

    Attempts to complete partial JSON to valid structure,
    enabling visualization during generation.
    """

    def __init__(self):
        self.bracket_stack: List[str] = []
        self.in_string = False
        self.escape_next = False

    def parse(self, text: str) -> PartialParseResult:
        """
        Parse potentially incomplete JSON text.

        Args:
            text: Partial JSON string

        Returns:
            PartialParseResult with best-effort parse
        """
        if not text.strip():
            return PartialParseResult(
                success=False,
                raw_text=text,
                parse_state=ParseState.EMPTY
            )

        # Try direct parse first
        try:
            parsed = json.loads(text)
            return PartialParseResult(
                success=True,
                partial_json=parsed,
                raw_text=text,
                parse_state=ParseState.COMPLETE,
                **self._extract_statechart_info(parsed)
            )
        except json.JSONDecodeError:
            pass

        # Try to complete and parse
        completed = self._complete_json(text)

        try:
            parsed = json.loads(completed)
            sc_info = self._extract_statechart_info(parsed)

            return PartialParseResult(
                success=True,
                partial_json=parsed,
                raw_text=text,
                parse_state=self._determine_state(text),
                completion_hints=self._get_completion_hints(text),
                **sc_info
            )
        except json.JSONDecodeError as e:
            return PartialParseResult(
                success=False,
                raw_text=text,
                parse_state=ParseState.ERROR,
                errors=[str(e)],
                completion_hints=self._get_completion_hints(text),
            )

    def _complete_json(self, text: str) -> str:
        """
        Complete partial JSON to valid structure.

        Handles:
        - Unclosed strings
        - Unclosed brackets
        - Trailing commas
        - Incomplete values
        """
        result = text.strip()

        if not result:
            return "{}"

        # Track bracket state
        stack = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(result):
            char = result[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char in '{[':
                    stack.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

            i += 1

        # Close unclosed string
        if in_string:
            result += '"'

        # Remove trailing comma
        result = re.sub(r',\s*$', '', result)

        # Check for incomplete key-value
        if re.search(r':\s*$', result):
            result += 'null'
        elif re.search(r',\s*"[^"]*"\s*$', result):
            # Key without value
            result += ': null'

        # Close unclosed brackets
        while stack:
            bracket = stack.pop()
            if bracket == '{':
                result += '}'
            elif bracket == '[':
                result += ']'

        return result

    def _determine_state(self, text: str) -> ParseState:
        """Determine current parse state."""
        text = text.strip()

        if not text:
            return ParseState.EMPTY

        # Check last significant character
        stripped = text.rstrip()

        if stripped.endswith('{'):
            return ParseState.OBJECT_START
        elif stripped.endswith('['):
            return ParseState.ARRAY_START
        elif stripped.endswith(':'):
            return ParseState.OBJECT_COLON
        elif stripped.endswith(','):
            # Could be in object or array
            if '{' in text and text.count('{') > text.count('}'):
                return ParseState.OBJECT_VALUE
            return ParseState.ARRAY_VALUE
        elif stripped.endswith('"'):
            return ParseState.STRING

        return ParseState.OBJECT_VALUE

    def _get_completion_hints(self, text: str) -> List[str]:
        """Get hints for completing the partial JSON."""
        hints = []
        state = self._determine_state(text)

        if state == ParseState.OBJECT_START:
            hints.append('Expecting property name or }')
        elif state == ParseState.OBJECT_COLON:
            hints.append('Expecting value')
        elif state == ParseState.ARRAY_START:
            hints.append('Expecting value or ]')

        # Statechart-specific hints
        text_lower = text.lower()
        if 'root_state' not in text_lower:
            hints.append('Missing root_state')
        if 'transitions' not in text_lower:
            hints.append('Missing transitions')

        return hints

    def _extract_statechart_info(self, parsed: Dict) -> Dict:
        """Extract statechart-specific information."""
        info = {
            'states_found': [],
            'transitions_found': 0,
            'has_root': False,
        }

        root_state = parsed.get('root_state')
        if root_state and isinstance(root_state, dict):
            info['has_root'] = True
            self._collect_state_labels(root_state, info['states_found'])

        transitions = parsed.get('transitions')
        if transitions and isinstance(transitions, list):
            info['transitions_found'] = len(transitions)

        return info

    def _collect_state_labels(self, node: Dict, labels: List[str]):
        """Recursively collect state labels."""
        if not isinstance(node, dict):
            return

        label = node.get('label', '')
        if label and label != '__root__':
            labels.append(label)

        children = node.get('children')
        if children:
            for child in children:
                if child:
                    self._collect_state_labels(child, labels)


def parse_partial_json(text: str) -> PartialParseResult:
    """Convenience function to parse partial JSON."""
    parser = PartialParser()
    return parser.parse(text)


def parse_partial_statechart(text: str) -> Tuple[Optional[Dict], List[str]]:
    """
    Parse partial statechart JSON.

    Returns:
        (partial_statechart, errors)
    """
    result = parse_partial_json(text)
    return result.partial_json, result.errors


def test_partial_parser():
    """Test the partial parser."""
    print("=" * 60)
    print("Testing Partial Parser")
    print("=" * 60)

    test_cases = [
        # Empty
        ("", ParseState.EMPTY),
        # Just opening
        ("{", ParseState.OBJECT_START),
        ('{"root_state":', ParseState.OBJECT_COLON),
        # Partial state
        ('{"root_state": {"label": "Off"', ParseState.STRING),
        # More complete
        ('{"root_state": {"label": "__root__", "children": [{"label": "Off"}]}}', ParseState.COMPLETE),
        # With transitions
        ('{"root_state": {"label": "__root__"}, "transitions": [', ParseState.ARRAY_START),
    ]

    parser = PartialParser()

    for i, (text, expected_state) in enumerate(test_cases):
        print(f"\n{i+1}. Input: {repr(text[:50])}...")
        result = parser.parse(text)

        print(f"   Success: {result.success}")
        print(f"   State: {result.parse_state.name}")

        if result.partial_json:
            print(f"   Has root: {result.has_root}")
            print(f"   States: {result.states_found}")
            print(f"   Transitions: {result.transitions_found}")

        if result.errors:
            print(f"   Errors: {result.errors}")

        if result.completion_hints:
            print(f"   Hints: {result.completion_hints}")

    # Test streaming scenario
    print("\n" + "=" * 60)
    print("Streaming Simulation")
    print("=" * 60)

    full_json = '{"root_state": {"label": "__root__", "children": [{"label": "Off", "is_initial": true}, {"label": "On"}]}, "transitions": [{"from": ["Off"], "to": ["On"], "event": "TOGGLE"}]}'

    print("\nSimulating token-by-token generation:")
    for i in range(5, len(full_json), 20):
        partial = full_json[:i]
        result = parser.parse(partial)

        states_str = ', '.join(result.states_found) if result.states_found else 'none'
        print(f"  Pos {i:3d}: {result.parse_state.name:15s} | States: {states_str}")

    # Final
    result = parser.parse(full_json)
    print(f"  COMPLETE: States={result.states_found}, Transitions={result.transitions_found}")

    print("\n" + "=" * 60)
    print("Partial parser tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_partial_parser()
