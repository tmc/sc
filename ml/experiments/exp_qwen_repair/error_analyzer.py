#!/usr/bin/env python3
"""
Error Analyzer: Parse and categorize statechart validation errors.

Parses `sc validate` output into structured error types for targeted repair.
"""

import re
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto


class ErrorType(Enum):
    """Categories of statechart validation errors."""
    # State errors
    DUPLICATE_STATE = auto()          # duplicate state label: X
    MISSING_STATE = auto()            # state X not found
    ORPHAN_STATE = auto()             # state X has no incoming transitions
    UNREACHABLE_STATE = auto()        # state X not reachable from initial

    # Transition errors
    INVALID_SOURCE = auto()           # transition source state X not found
    INVALID_TARGET = auto()           # transition target state X not found
    MISSING_EVENT = auto()            # transition missing event
    DUPLICATE_TRANSITION = auto()     # duplicate transition

    # Structure errors
    NO_INITIAL_STATE = auto()         # no initial state in compound state X
    MULTIPLE_INITIAL = auto()         # multiple initial states in X
    EMPTY_COMPOUND = auto()           # compound state X must have children
    INVALID_HIERARCHY = auto()        # parent/child structure invalid

    # Type errors
    BASIC_WITH_CHILDREN = auto()      # basic state X cannot have children
    PARALLEL_NEEDS_CHILDREN = auto()  # parallel state X needs 2+ children

    # Other
    UNKNOWN = auto()


@dataclass
class ValidationError:
    """Structured validation error."""
    error_type: ErrorType
    message: str
    state_label: Optional[str] = None
    transition_info: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.error_type.name}: {self.message}"


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    raw_output: str = ""

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def errors_by_type(self) -> Dict[ErrorType, List[ValidationError]]:
        """Group errors by type."""
        result = {}
        for err in self.errors:
            if err.error_type not in result:
                result[err.error_type] = []
            result[err.error_type].append(err)
        return result


class ErrorAnalyzer:
    """
    Parses validation output into structured errors.

    Uses pattern matching to extract error type and relevant details.
    """

    # Error patterns: (regex, error_type, extractor_fn)
    # Note: More specific patterns should come first
    ERROR_PATTERNS = [
        # Transition errors (specific, check first)
        (r"transition source state (\w+) not found", ErrorType.INVALID_SOURCE, "state"),
        (r"transition target state (\w+) not found", ErrorType.INVALID_TARGET, "state"),

        # State errors
        (r"duplicate state label: (\w+)", ErrorType.DUPLICATE_STATE, "state"),
        (r"state (\w+) not found", ErrorType.MISSING_STATE, "state"),

        # Structure errors
        (r"no initial state in (?:compound state )?(\w+)", ErrorType.NO_INITIAL_STATE, "state"),
        (r"(?:normal|compound) state (\w+) must have exactly one initial", ErrorType.NO_INITIAL_STATE, "state"),
        (r"multiple initial states in (\w+)", ErrorType.MULTIPLE_INITIAL, "state"),
        (r"compound state (\w+) must have children", ErrorType.EMPTY_COMPOUND, "state"),

        # Type errors
        (r"basic state (\w+) cannot have children", ErrorType.BASIC_WITH_CHILDREN, "state"),
        (r"parallel state (\w+) (?:needs|requires) (?:at least )?2", ErrorType.PARALLEL_NEEDS_CHILDREN, "state"),

        # Transition errors
        (r"transition (?:from|between) \[?(\w+)\]? (?:to|and) \[?(\w+)\]? missing event",
         ErrorType.MISSING_EVENT, "transition"),
        (r"duplicate transition", ErrorType.DUPLICATE_TRANSITION, None),
    ]

    def __init__(self, sc_path: str = "./sc"):
        self.sc_path = sc_path

    def validate_chart(self, chart: Dict[str, Any]) -> ValidationResult:
        """
        Validate a statechart and return structured errors.

        Args:
            chart: Statechart as dict (JSON-serializable)

        Returns:
            ValidationResult with parsed errors
        """
        # Run sc validate
        try:
            result = subprocess.run(
                [self.sc_path, "validate"],
                input=json.dumps(chart),
                capture_output=True,
                text=True,
                timeout=10,
            )
            raw_output = result.stdout + result.stderr
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    error_type=ErrorType.UNKNOWN,
                    message=f"Validation failed: {e}",
                )],
                raw_output=str(e),
            )

        # Check if valid
        if "VALID" in raw_output and "INVALID" not in raw_output:
            return ValidationResult(is_valid=True, raw_output=raw_output)

        # Parse errors
        errors = self._parse_errors(raw_output)

        return ValidationResult(
            is_valid=False,
            errors=errors,
            raw_output=raw_output,
        )

    def validate_json(self, json_str: str) -> ValidationResult:
        """Validate from JSON string."""
        try:
            chart = json.loads(json_str)
            return self.validate_chart(chart)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    error_type=ErrorType.UNKNOWN,
                    message=f"Invalid JSON: {e}",
                )],
                raw_output=str(e),
            )

    def _parse_errors(self, output: str) -> List[ValidationError]:
        """Parse error messages into structured errors."""
        errors = []

        # Split into lines and process each
        for line in output.split('\n'):
            line = line.strip()
            if not line or line == "VALID":
                continue

            # Remove "INVALID: " prefix if present
            if line.startswith("INVALID:"):
                line = line[8:].strip()

            # Try to match patterns
            matched = False
            for pattern, error_type, extractor in self.ERROR_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    error = ValidationError(
                        error_type=error_type,
                        message=line,
                    )

                    # Extract state or transition info
                    if extractor == "state" and match.groups():
                        error.state_label = match.group(1)
                    elif extractor == "transition" and len(match.groups()) >= 2:
                        error.transition_info = {
                            "from": match.group(1),
                            "to": match.group(2),
                        }

                    errors.append(error)
                    matched = True
                    break

            # Unknown error
            if not matched and line:
                errors.append(ValidationError(
                    error_type=ErrorType.UNKNOWN,
                    message=line,
                ))

        return errors

    def get_repair_hints(self, errors: List[ValidationError], chart: Dict[str, Any]) -> List[str]:
        """
        Generate repair hints for each error.

        Returns natural language hints for the LLM.
        """
        hints = []

        # Collect all states in chart
        all_states = self._collect_states(chart)

        for error in errors:
            hint = self._generate_hint(error, all_states, chart)
            if hint:
                hints.append(hint)

        return hints

    def _collect_states(self, chart: Dict[str, Any]) -> Set[str]:
        """Collect all state labels from chart."""
        states = set()

        def collect(state: Dict[str, Any]):
            if "label" in state:
                states.add(state["label"])
            for child in state.get("children", []):
                collect(child)

        if "root_state" in chart:
            collect(chart["root_state"])

        return states

    def _generate_hint(
        self,
        error: ValidationError,
        all_states: Set[str],
        chart: Dict[str, Any],
    ) -> str:
        """Generate repair hint for an error."""
        if error.error_type == ErrorType.DUPLICATE_STATE:
            return f"Rename or remove duplicate state '{error.state_label}'"

        elif error.error_type == ErrorType.INVALID_SOURCE:
            similar = self._find_similar(error.state_label, all_states)
            if similar:
                return f"Change transition source '{error.state_label}' to '{similar}'"
            return f"Add state '{error.state_label}' or fix transition source"

        elif error.error_type == ErrorType.INVALID_TARGET:
            similar = self._find_similar(error.state_label, all_states)
            if similar:
                return f"Change transition target '{error.state_label}' to '{similar}'"
            return f"Add state '{error.state_label}' or fix transition target"

        elif error.error_type == ErrorType.NO_INITIAL_STATE:
            return f"Add 'is_initial': true to one child of '{error.state_label}'"

        elif error.error_type == ErrorType.MULTIPLE_INITIAL:
            return f"Remove 'is_initial': true from all but one child of '{error.state_label}'"

        elif error.error_type == ErrorType.EMPTY_COMPOUND:
            return f"Add children to compound state '{error.state_label}' or change to basic type"

        elif error.error_type == ErrorType.BASIC_WITH_CHILDREN:
            return f"Remove children from '{error.state_label}' or change type to NORMAL"

        elif error.error_type == ErrorType.PARALLEL_NEEDS_CHILDREN:
            return f"Add at least 2 children to parallel state '{error.state_label}'"

        elif error.error_type == ErrorType.MISSING_EVENT:
            return f"Add 'event' field to transition"

        return f"Fix: {error.message}"

    def _find_similar(self, target: str, candidates: Set[str]) -> Optional[str]:
        """Find similar state name (typo correction)."""
        if not target or not candidates:
            return None

        # Exact match (case insensitive)
        for c in candidates:
            if c.lower() == target.lower():
                return c

        # Prefix match
        for c in candidates:
            if c.lower().startswith(target.lower()[:3]):
                return c

        # Levenshtein-like: single character difference
        for c in candidates:
            if abs(len(c) - len(target)) <= 1:
                diff = sum(1 for a, b in zip(c.lower(), target.lower()) if a != b)
                if diff <= 1:
                    return c

        return None


def analyze_errors(chart: Dict[str, Any], sc_path: str = "./sc") -> ValidationResult:
    """Convenience function to analyze errors."""
    analyzer = ErrorAnalyzer(sc_path)
    return analyzer.validate_chart(chart)


def demo():
    """Demo error analysis."""
    print("=" * 60)
    print("ERROR ANALYZER DEMO")
    print("=" * 60)

    analyzer = ErrorAnalyzer()

    # Test cases
    test_cases = [
        # Duplicate state
        {
            "name": "Duplicate state",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "children": [
                        {"label": "A"},
                        {"label": "A"},
                    ]
                }
            }
        },
        # Invalid transition source
        {
            "name": "Invalid source",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "children": [
                        {"label": "A", "is_initial": True},
                        {"label": "B"},
                    ]
                },
                "transitions": [
                    {"from": ["X"], "to": ["B"], "event": "GO"}
                ]
            }
        },
        # Missing initial
        {
            "name": "No initial state",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "A"},
                        {"label": "B"},
                    ]
                }
            }
        },
        # Valid chart
        {
            "name": "Valid chart",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "A", "is_initial": True},
                        {"label": "B"},
                    ]
                },
                "transitions": [
                    {"from": ["A"], "to": ["B"], "event": "GO"}
                ]
            }
        },
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        result = analyzer.validate_chart(tc["chart"])

        print(f"Valid: {result.is_valid}")
        if result.errors:
            for err in result.errors:
                print(f"  Error: {err}")

            hints = analyzer.get_repair_hints(result.errors, tc["chart"])
            for hint in hints:
                print(f"  Hint: {hint}")

    print("\n" + "=" * 60)
    print("ERROR ANALYZER DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
