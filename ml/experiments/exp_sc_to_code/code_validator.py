"""
Code Validator - Validate generated Python code.

Tests:
1. Syntax validity (ast.parse)
2. Runnable (exec without error)
3. Behavioral correctness (transitions match SC)
"""

import ast
import sys
from io import StringIO
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class ValidationResult:
    """Result of code validation."""
    syntax_valid: bool
    runnable: bool
    behavior_correct: bool
    errors: List[str] = field(default_factory=list)
    transition_tests: List[Dict] = field(default_factory=list)


class CodeValidator:
    """Validate generated Python code against statechart definition."""

    def validate(self, code: str, sc: Dict, class_name: str) -> ValidationResult:
        """
        Validate generated code.

        Args:
            code: Generated Python code
            sc: Original statechart definition
            class_name: Expected class name

        Returns:
            ValidationResult with validation status
        """
        errors = []
        transition_tests = []

        # Step 1: Syntax validation
        syntax_valid, syntax_error = self._check_syntax(code)
        if not syntax_valid:
            errors.append(f"Syntax error: {syntax_error}")
            return ValidationResult(
                syntax_valid=False,
                runnable=False,
                behavior_correct=False,
                errors=errors,
            )

        # Step 2: Runnable check (exec)
        runnable, namespace, exec_error = self._check_runnable(code)
        if not runnable:
            errors.append(f"Execution error: {exec_error}")
            return ValidationResult(
                syntax_valid=True,
                runnable=False,
                behavior_correct=False,
                errors=errors,
            )

        # Step 3: Behavioral correctness
        behavior_correct, behavior_errors, tests = self._check_behavior(
            namespace, sc, class_name
        )
        errors.extend(behavior_errors)

        return ValidationResult(
            syntax_valid=True,
            runnable=True,
            behavior_correct=behavior_correct,
            errors=errors,
            transition_tests=tests,
        )

    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _check_runnable(self, code: str) -> Tuple[bool, Dict, str]:
        """Check if code can be executed."""
        namespace = {}

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            exec(compile(code, '<generated>', 'exec'), namespace)
            return True, namespace, ""
        except Exception as e:
            return False, {}, str(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _check_behavior(
        self, namespace: Dict, sc: Dict, class_name: str
    ) -> Tuple[bool, List[str], List[Dict]]:
        """Check if generated code behavior matches statechart."""
        errors = []
        tests = []

        # Find the state machine class
        SM = None
        from enum import Enum as EnumType

        # First try the expected class name
        SM = namespace.get(class_name)

        # If not found or is an enum, search for state machine class
        if SM is None or (isinstance(SM, type) and issubclass(SM, EnumType)):
            SM = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name not in ('Enum', 'auto', 'State', 'Event'):
                    # Skip enums - we want the actual state machine class
                    if issubclass(obj, EnumType):
                        continue
                    # Found a class - might be our state machine
                    if hasattr(obj, 'send') or hasattr(obj, '__init__'):
                        SM = obj
                        break

        if SM is None:
            errors.append(f"No state machine class found")
            return False, errors, tests

        # Extract root for state/event lookup
        root = sc.get("root_state", {})

        # Get State and Event enums - be flexible about naming
        State = namespace.get('State')
        Event = namespace.get('Event')

        # If State not found, look for state-like enums
        if State is None:
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, EnumType):
                    if name not in ('Enum', 'Event') and 'State' in name:
                        State = obj
                        break
                    # Or check if it contains expected state names
                    if name not in ('Enum', 'Event'):
                        state_names = {child.get("label") for child in root.get("children", [])}
                        enum_names = {m.name for m in obj}
                        if state_names & enum_names:  # Has overlap
                            State = obj
                            break

        # If Event not found, look for event-like enums
        if Event is None:
            expected_events = {t.get("event") for t in sc.get("transitions", [])}
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, EnumType):
                    if name == 'Event' or 'Event' in name:
                        Event = obj
                        break
                    # Check if enum contains expected events
                    if name not in ('Enum', str(State)) if State else True:
                        enum_names = {m.name for m in obj}
                        if expected_events & enum_names:
                            Event = obj
                            break

        if State is None:
            errors.append("No State enum found")
            return False, errors, tests

        if Event is None:
            errors.append("No Event enum found")
            return False, errors, tests

        # Extract expected initial state and transitions
        states = []
        initial_state = None
        self._collect_states(root, states)

        for child in root.get("children", []):
            if child.get("is_initial"):
                initial_state = child.get("label")
                break

        if not initial_state and states:
            initial_state = states[0]

        # Build transition map
        transition_map = {}  # (from_state, event) -> to_state
        for t in sc.get("transitions", []):
            src = t.get("from", [""])[0] if isinstance(t.get("from"), list) else ""
            tgt = t.get("to", [""])[0] if isinstance(t.get("to"), list) else ""
            event = t.get("event", "")
            transition_map[(src, event)] = tgt

        # Test 1: Initial state
        try:
            sm = SM()

            # Check if initial state matches
            current = self._get_current_state(sm, State)
            expected_initial = self._find_state_enum(State, initial_state)

            if current is None:
                errors.append("Cannot determine current state")
                return False, errors, tests

            if expected_initial is None:
                errors.append(f"Initial state '{initial_state}' not found in State enum")
                return False, errors, tests

            initial_correct = (current == expected_initial)
            tests.append({
                "test": "initial_state",
                "expected": initial_state,
                "actual": str(current),
                "passed": initial_correct,
            })

            if not initial_correct:
                errors.append(f"Initial state mismatch: expected {initial_state}, got {current}")

        except Exception as e:
            errors.append(f"Failed to create instance: {e}")
            return False, errors, tests

        # Test 2: Transitions
        transitions_correct = 0
        transitions_total = 0

        for (src, event), tgt in transition_map.items():
            transitions_total += 1

            # Reset state machine
            try:
                sm = SM()

                # Get to source state first
                if src != initial_state:
                    # Try to reach source state
                    reached = self._reach_state(sm, State, Event, src, transition_map, initial_state)
                    if not reached:
                        tests.append({
                            "test": f"{src}--{event}-->{tgt}",
                            "expected": tgt,
                            "actual": "could not reach source",
                            "passed": False,
                        })
                        continue

                # Send event
                event_enum = self._find_event_enum(Event, event)
                if event_enum is None:
                    tests.append({
                        "test": f"{src}--{event}-->{tgt}",
                        "expected": tgt,
                        "actual": "event not found",
                        "passed": False,
                    })
                    continue

                # Call send method
                try:
                    if hasattr(sm, 'send'):
                        sm.send(event_enum)
                    elif hasattr(sm, 'transition'):
                        sm.transition(event_enum)
                except Exception as e:
                    tests.append({
                        "test": f"{src}--{event}-->{tgt}",
                        "expected": tgt,
                        "actual": f"error: {e}",
                        "passed": False,
                    })
                    continue

                # Check new state
                new_state = self._get_current_state(sm, State)
                expected_state = self._find_state_enum(State, tgt)

                if new_state == expected_state:
                    transitions_correct += 1
                    tests.append({
                        "test": f"{src}--{event}-->{tgt}",
                        "expected": tgt,
                        "actual": str(new_state),
                        "passed": True,
                    })
                else:
                    tests.append({
                        "test": f"{src}--{event}-->{tgt}",
                        "expected": tgt,
                        "actual": str(new_state),
                        "passed": False,
                    })

            except Exception as e:
                tests.append({
                    "test": f"{src}--{event}-->{tgt}",
                    "expected": tgt,
                    "actual": f"error: {e}",
                    "passed": False,
                })

        # Determine if behavior is correct (at least initial state + some transitions)
        behavior_correct = (
            initial_correct and
            transitions_correct > 0 and
            transitions_correct >= transitions_total * 0.5  # At least 50% of transitions work
        )

        return behavior_correct, errors, tests

    def _collect_states(self, node: Dict, states: List[str]):
        """Recursively collect state labels."""
        label = node.get("label", "")
        if label and label != "__root__":
            states.append(label)
        for child in node.get("children", []):
            self._collect_states(child, states)

    def _get_current_state(self, sm: Any, State) -> Optional[Any]:
        """Get current state from state machine instance."""
        if hasattr(sm, 'state'):
            return sm.state
        elif hasattr(sm, 'current_state'):
            return sm.current_state
        elif hasattr(sm, 'get_state'):
            return sm.get_state()
        return None

    def _find_state_enum(self, State, state_name: str) -> Optional[Any]:
        """Find state enum member by name."""
        if hasattr(State, state_name):
            return getattr(State, state_name)
        # Try case variations
        for member in State:
            if member.name.lower() == state_name.lower():
                return member
        return None

    def _find_event_enum(self, Event, event_name: str) -> Optional[Any]:
        """Find event enum member by name."""
        if hasattr(Event, event_name):
            return getattr(Event, event_name)
        # Try case variations
        for member in Event:
            if member.name.upper() == event_name.upper():
                return member
        return None

    def _reach_state(
        self,
        sm: Any,
        State,
        Event,
        target_state: str,
        transitions: Dict,
        initial: str,
        max_steps: int = 10
    ) -> bool:
        """Try to reach a target state from initial state."""
        # Simple BFS to find path to target state
        current = initial
        visited = {current}
        queue = [(current, [])]

        while queue:
            state, path = queue.pop(0)
            if state == target_state:
                # Execute the path
                for event in path:
                    event_enum = self._find_event_enum(Event, event)
                    if event_enum and hasattr(sm, 'send'):
                        sm.send(event_enum)
                return True

            for (src, evt), tgt in transitions.items():
                if src == state and tgt not in visited:
                    visited.add(tgt)
                    queue.append((tgt, path + [evt]))

            if len(visited) > max_steps:
                break

        return False


def validate_generated_code(code: str, sc: Dict, class_name: str) -> ValidationResult:
    """Convenience function to validate code."""
    validator = CodeValidator()
    return validator.validate(code, sc, class_name)


if __name__ == "__main__":
    # Test the validator
    test_code = '''from enum import Enum, auto

class State(Enum):
    Off = auto()
    On = auto()

class Event(Enum):
    TOGGLE = auto()

class Toggle:
    def __init__(self):
        self.state = State.Off

    def send(self, event: Event) -> bool:
        if event == Event.TOGGLE:
            if self.state == State.Off:
                self.state = State.On
                return True
            elif self.state == State.On:
                self.state = State.Off
                return True
        return False
'''

    test_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1}
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
        ]
    }

    result = validate_generated_code(test_code, test_sc, "Toggle")
    print(f"Syntax valid: {result.syntax_valid}")
    print(f"Runnable: {result.runnable}")
    print(f"Behavior correct: {result.behavior_correct}")
    print(f"Errors: {result.errors}")
    for test in result.transition_tests:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  {status}: {test['test']} (expected={test['expected']}, actual={test['actual']})")
