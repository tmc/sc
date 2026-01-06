"""
Test Generator for Generated State Machine Code.

Generates test cases to verify the correctness of generated code:
1. State transition tests
2. Invalid transition tests
3. Callback tests
4. Edge case tests
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TestCase:
    """A single test case."""
    name: str
    description: str
    events: List[str]         # Sequence of events to send
    expected_states: List[str]  # Expected state after each event
    expected_success: List[bool] = field(default_factory=list)


@dataclass
class TestSuite:
    """Collection of test cases."""
    name: str
    test_cases: List[TestCase]
    setup_code: str = ""
    teardown_code: str = ""


class TestGenerator:
    """Generate tests for state machine code."""

    def __init__(self, statechart_json: Dict):
        self.statechart = statechart_json
        self.states = self._extract_states()
        self.events = self._extract_events()
        self.transitions = self._extract_transitions()
        self.initial_state = self._find_initial()

    def _extract_states(self) -> List[str]:
        """Extract all state labels."""
        states = []
        root = self.statechart.get('root_state', {})
        self._collect_states(root, states)
        return states

    def _collect_states(self, node: Dict, states: List[str]):
        """Recursively collect states."""
        label = node.get('label', '')
        if label and label != '__root__':
            states.append(label)
        for child in node.get('children', []):
            self._collect_states(child, states)

    def _extract_events(self) -> List[str]:
        """Extract all event names."""
        events = set()
        for t in self.statechart.get('transitions', []):
            event = t.get('event', '')
            if event:
                events.add(event)
        return sorted(events)

    def _extract_transitions(self) -> Dict[Tuple[str, str], str]:
        """Extract transitions as (source, event) -> target."""
        transitions = {}
        for t in self.statechart.get('transitions', []):
            src = t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', '')
            tgt = t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', '')
            event = t.get('event', '')
            if src and event and tgt:
                transitions[(src, event)] = tgt
        return transitions

    def _find_initial(self) -> str:
        """Find initial state."""
        root = self.statechart.get('root_state', {})
        initial = self._find_initial_recursive(root)
        return initial or (self.states[0] if self.states else 'Unknown')

    def _find_initial_recursive(self, node: Dict) -> Optional[str]:
        """Find initial state recursively."""
        if node.get('is_initial'):
            return node.get('label', '')
        for child in node.get('children', []):
            result = self._find_initial_recursive(child)
            if result:
                return result
        children = node.get('children', [])
        if children:
            return children[0].get('label', '')
        return None

    def generate_test_suite(self) -> TestSuite:
        """Generate a complete test suite."""
        tests = []

        # Test 1: Initial state
        tests.append(TestCase(
            name="test_initial_state",
            description="Verify initial state is correct",
            events=[],
            expected_states=[self.initial_state],
            expected_success=[True],
        ))

        # Test 2: Valid transitions
        for (src, event), tgt in self.transitions.items():
            if src == self.initial_state:
                tests.append(TestCase(
                    name=f"test_transition_{src}_to_{tgt}",
                    description=f"Transition from {src} to {tgt} on {event}",
                    events=[event],
                    expected_states=[tgt],
                    expected_success=[True],
                ))

        # Test 3: Invalid transitions
        for state in self.states:
            for event in self.events:
                if (state, event) not in self.transitions:
                    # This is an invalid transition from initial state
                    if state == self.initial_state:
                        tests.append(TestCase(
                            name=f"test_invalid_{state}_{event}",
                            description=f"Invalid transition from {state} on {event}",
                            events=[event],
                            expected_states=[state],  # Should stay in same state
                            expected_success=[False],
                        ))
                        break  # Just one invalid test

        # Test 4: Sequence of transitions
        sequence = self._find_valid_sequence()
        if sequence:
            events, states = sequence
            tests.append(TestCase(
                name="test_transition_sequence",
                description="Multiple transitions in sequence",
                events=events,
                expected_states=states,
                expected_success=[True] * len(events),
            ))

        return TestSuite(
            name="StateMachineTests",
            test_cases=tests,
        )

    def _find_valid_sequence(self) -> Optional[Tuple[List[str], List[str]]]:
        """Find a valid sequence of transitions."""
        events = []
        states = []
        current = self.initial_state
        visited = {current}

        while True:
            # Find an outgoing transition
            found = False
            for (src, event), tgt in self.transitions.items():
                if src == current and tgt not in visited:
                    events.append(event)
                    states.append(tgt)
                    visited.add(tgt)
                    current = tgt
                    found = True
                    break

            if not found:
                break

        return (events, states) if events else None


def generate_go_tests(statechart_json: Dict, package_name: str = "statemachine") -> str:
    """Generate Go test code."""
    gen = TestGenerator(statechart_json)
    suite = gen.generate_test_suite()

    lines = [
        f'package {package_name}',
        '',
        'import "testing"',
        '',
    ]

    for tc in suite.test_cases:
        lines.append(f'func {tc.name.replace("test_", "Test")}(t *testing.T) {{')
        lines.append(f'\t// {tc.description}')
        lines.append('\tsm := NewStateMachine()')
        lines.append('')

        if not tc.events:
            # Just check initial state
            lines.append(f'\tif sm.CurrentState() != State{gen.initial_state} {{')
            lines.append(f'\t\tt.Errorf("expected State{gen.initial_state}, got %v", sm.CurrentState())')
            lines.append('\t}')
        else:
            for i, (event, expected, success) in enumerate(
                zip(tc.events, tc.expected_states, tc.expected_success)
            ):
                lines.append(f'\terr := sm.Send(Event{event})')
                if success:
                    lines.append('\tif err != nil {')
                    lines.append(f'\t\tt.Errorf("unexpected error: %v", err)')
                    lines.append('\t}')
                else:
                    lines.append('\tif err == nil {')
                    lines.append('\t\tt.Error("expected error for invalid transition")')
                    lines.append('\t}')

                lines.append(f'\tif sm.CurrentState() != State{expected} {{')
                lines.append(f'\t\tt.Errorf("expected State{expected}, got %v", sm.CurrentState())')
                lines.append('\t}')
                lines.append('')

        lines.append('}')
        lines.append('')

    return '\n'.join(lines)


def generate_python_tests(statechart_json: Dict, class_name: str = "StateMachine") -> str:
    """Generate Python test code."""
    gen = TestGenerator(statechart_json)
    suite = gen.generate_test_suite()

    lines = [
        'import unittest',
        f'from statemachine import {class_name}, State, Event',
        '',
        '',
        f'class Test{class_name}(unittest.TestCase):',
        '    """Test suite for the state machine."""',
        '',
    ]

    for tc in suite.test_cases:
        lines.append(f'    def {tc.name}(self):')
        lines.append(f'        """{tc.description}"""')
        lines.append(f'        sm = {class_name}()')
        lines.append('')

        if not tc.events:
            # Just check initial state
            lines.append(f'        self.assertEqual(sm.state, State.{gen.initial_state})')
        else:
            for i, (event, expected, success) in enumerate(
                zip(tc.events, tc.expected_states, tc.expected_success)
            ):
                lines.append(f'        result = sm.send(Event.{event})')
                if success:
                    lines.append('        self.assertTrue(result)')
                else:
                    lines.append('        self.assertFalse(result)')

                lines.append(f'        self.assertEqual(sm.state, State.{expected})')
                lines.append('')

        lines.append('')

    # Main block
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    unittest.main()')
    lines.append('')

    return '\n'.join(lines)


def test_test_generator():
    """Test the test generator."""
    print("=" * 60)
    print("Testing Test Generator")
    print("=" * 60)

    # Sample statechart
    statechart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "Running", "type": 1},
                {"label": "Paused", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Running"], "event": "START"},
            {"from": ["Running"], "to": ["Paused"], "event": "PAUSE"},
            {"from": ["Paused"], "to": ["Running"], "event": "RESUME"},
            {"from": ["Running"], "to": ["Idle"], "event": "STOP"},
            {"from": ["Paused"], "to": ["Idle"], "event": "STOP"},
        ]
    }

    gen = TestGenerator(statechart)

    print("\n1. Extracted info:")
    print(f"  States: {gen.states}")
    print(f"  Events: {gen.events}")
    print(f"  Initial: {gen.initial_state}")
    print(f"  Transitions: {gen.transitions}")

    print("\n2. Generated test suite:")
    suite = gen.generate_test_suite()
    print(f"  Suite name: {suite.name}")
    print(f"  Test cases: {len(suite.test_cases)}")
    for tc in suite.test_cases:
        print(f"    - {tc.name}: {tc.description}")

    print("\n3. Go test code:")
    go_tests = generate_go_tests(statechart)
    print(go_tests[:500] + "...")

    print("\n4. Python test code:")
    py_tests = generate_python_tests(statechart)
    print(py_tests[:500] + "...")

    print("\n" + "=" * 60)
    print("Test generator tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_test_generator()
