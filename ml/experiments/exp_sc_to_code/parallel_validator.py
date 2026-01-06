"""
Parallel Validator - Validate generated code for parallel statecharts.

Tests:
1. Syntax validity (ast.parse)
2. Runnable (exec without error)
3. Has regions tracking (self.regions dict)
4. Multi-region behavior (independent transitions)
"""

import ast
import sys
from io import StringIO
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum as EnumType

from .parallel_code_generator import detect_parallel_regions


@dataclass
class ParallelValidationResult:
    """Result of parallel code validation."""
    syntax_valid: bool
    runnable: bool
    has_regions_dict: bool
    regions_correct: int
    regions_total: int
    transitions_correct: int
    transitions_total: int
    errors: List[str] = field(default_factory=list)
    test_details: List[Dict] = field(default_factory=list)

    @property
    def regions_accuracy(self) -> float:
        return self.regions_correct / self.regions_total if self.regions_total > 0 else 0.0

    @property
    def transitions_accuracy(self) -> float:
        return self.transitions_correct / self.transitions_total if self.transitions_total > 0 else 0.0

    @property
    def overall_correct(self) -> bool:
        """Overall correctness: has regions, >=50% regions work, >=50% transitions work."""
        return (
            self.has_regions_dict and
            self.regions_accuracy >= 0.5 and
            self.transitions_accuracy >= 0.5
        )


class ParallelValidator:
    """Validate generated Python code for parallel statecharts."""

    def validate(self, code: str, sc: Dict, class_name: str) -> ParallelValidationResult:
        """
        Validate generated parallel code.

        Args:
            code: Generated Python code
            sc: Original statechart definition
            class_name: Expected class name

        Returns:
            ParallelValidationResult with validation status
        """
        errors = []
        test_details = []

        # Step 1: Syntax validation
        syntax_valid, syntax_error = self._check_syntax(code)
        if not syntax_valid:
            errors.append(f"Syntax error: {syntax_error}")
            return ParallelValidationResult(
                syntax_valid=False,
                runnable=False,
                has_regions_dict=False,
                regions_correct=0,
                regions_total=0,
                transitions_correct=0,
                transitions_total=0,
                errors=errors,
            )

        # Step 2: Runnable check
        runnable, namespace, exec_error = self._check_runnable(code)
        if not runnable:
            errors.append(f"Execution error: {exec_error}")
            return ParallelValidationResult(
                syntax_valid=True,
                runnable=False,
                has_regions_dict=False,
                regions_correct=0,
                regions_total=0,
                transitions_correct=0,
                transitions_total=0,
                errors=errors,
            )

        # Step 3: Check for regions dict
        has_regions, sm, regions_error = self._check_regions_dict(namespace, class_name)
        if not has_regions:
            if regions_error:
                errors.append(regions_error)
            return ParallelValidationResult(
                syntax_valid=True,
                runnable=True,
                has_regions_dict=False,
                regions_correct=0,
                regions_total=0,
                transitions_correct=0,
                transitions_total=0,
                errors=errors,
            )

        # Step 4: Check region initial states
        regions = detect_parallel_regions(sc)
        regions_correct, regions_total, region_tests = self._check_regions_initial(
            sm, regions, namespace
        )
        test_details.extend(region_tests)

        # Step 5: Check transitions
        trans_correct, trans_total, trans_tests = self._check_transitions(
            code, namespace, sc, class_name, regions
        )
        test_details.extend(trans_tests)

        return ParallelValidationResult(
            syntax_valid=True,
            runnable=True,
            has_regions_dict=True,
            regions_correct=regions_correct,
            regions_total=regions_total,
            transitions_correct=trans_correct,
            transitions_total=trans_total,
            errors=errors,
            test_details=test_details,
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

    def _check_regions_dict(
        self, namespace: Dict, class_name: str
    ) -> Tuple[bool, Any, Optional[str]]:
        """Check if class has self.regions dict."""
        # Find the class
        SM = namespace.get(class_name)

        # If not found or is enum, search
        if SM is None or (isinstance(SM, type) and issubclass(SM, EnumType)):
            SM = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name not in ('Enum', 'auto'):
                    if issubclass(obj, EnumType):
                        continue
                    if hasattr(obj, 'send') or hasattr(obj, '__init__'):
                        SM = obj
                        break

        if SM is None:
            return False, None, "No state machine class found"

        # Create instance and check for regions
        try:
            sm = SM()
            if hasattr(sm, 'regions') and isinstance(sm.regions, dict):
                return True, sm, None
            else:
                return False, sm, "No self.regions dict found (uses flat state instead)"
        except Exception as e:
            return False, None, f"Failed to create instance: {e}"

    def _check_regions_initial(
        self, sm: Any, regions: List[Tuple], namespace: Dict
    ) -> Tuple[int, int, List[Dict]]:
        """Check if regions have correct initial states."""
        correct = 0
        total = len(regions)
        tests = []

        for region_name, states, initial_state in regions:
            test = {
                "test": f"region_initial_{region_name}",
                "expected": initial_state,
            }

            # Try to get region state
            if hasattr(sm, 'get_state'):
                try:
                    state = sm.get_state(region_name)
                    actual = state.name if hasattr(state, 'name') else str(state)
                    test["actual"] = actual
                    if actual == initial_state:
                        correct += 1
                        test["passed"] = True
                    else:
                        test["passed"] = False
                except Exception as e:
                    test["actual"] = f"error: {e}"
                    test["passed"] = False
            elif hasattr(sm, 'regions'):
                state = sm.regions.get(region_name)
                if state:
                    actual = state.name if hasattr(state, 'name') else str(state)
                    test["actual"] = actual
                    if actual == initial_state:
                        correct += 1
                        test["passed"] = True
                    else:
                        test["passed"] = False
                else:
                    test["actual"] = "region not found"
                    test["passed"] = False
            else:
                test["actual"] = "no regions access"
                test["passed"] = False

            tests.append(test)

        return correct, total, tests

    def _check_transitions(
        self, code: str, namespace: Dict, sc: Dict, class_name: str, regions: List[Tuple]
    ) -> Tuple[int, int, List[Dict]]:
        """Check if transitions work correctly."""
        correct = 0
        total = 0
        tests = []

        # Build map of state -> region
        state_to_region = {}
        for region_name, states, _ in regions:
            for state in states:
                state_to_region[state] = region_name

        # Get transitions
        transitions = sc.get("transitions", [])
        total = len(transitions)

        # Find SM class and Event enum
        SM = self._find_class(namespace, class_name)
        if SM is None:
            return 0, total, [{"test": "transitions", "passed": False, "actual": "no SM class"}]

        for t in transitions:
            src = t.get("from", [""])[0] if t.get("from") else ""
            tgt = t.get("to", [""])[0] if t.get("to") else ""
            event = t.get("event", "")

            test = {
                "test": f"{src}--{event}-->{tgt}",
                "expected": tgt,
            }

            # Determine which region this transition affects
            region = state_to_region.get(src, "")
            if not region:
                test["actual"] = "unknown region"
                test["passed"] = False
                tests.append(test)
                continue

            try:
                # Create fresh instance
                sm = SM()

                # Navigate to source state if not initial
                # For now, just set region directly if possible
                self._set_region_state(sm, region, src, namespace)

                # Send event
                event_enum = self._find_event_enum(namespace, event)
                if event_enum is None:
                    test["actual"] = f"event {event} not found"
                    test["passed"] = False
                    tests.append(test)
                    continue

                # Send the event
                if hasattr(sm, 'send'):
                    sm.send(event_enum)

                # Check resulting state
                new_state = self._get_region_state(sm, region)
                test["actual"] = new_state

                if new_state == tgt:
                    correct += 1
                    test["passed"] = True
                else:
                    test["passed"] = False

            except Exception as e:
                test["actual"] = f"error: {e}"
                test["passed"] = False

            tests.append(test)

        return correct, total, tests

    def _find_class(self, namespace: Dict, class_name: str) -> Optional[type]:
        """Find the state machine class."""
        SM = namespace.get(class_name)
        if SM is not None and isinstance(SM, type) and not issubclass(SM, EnumType):
            return SM

        for name, obj in namespace.items():
            if isinstance(obj, type) and name not in ('Enum', 'auto'):
                if issubclass(obj, EnumType):
                    continue
                if hasattr(obj, 'send') or hasattr(obj, 'regions'):
                    return obj
        return None

    def _find_event_enum(self, namespace: Dict, event_name: str) -> Optional[Any]:
        """Find event enum member."""
        Event = namespace.get('Event')
        if Event and hasattr(Event, event_name):
            return getattr(Event, event_name)

        # Search all enums
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, EnumType):
                if hasattr(obj, event_name):
                    return getattr(obj, event_name)
        return None

    def _get_region_state(self, sm: Any, region: str) -> str:
        """Get current state of a region."""
        if hasattr(sm, 'get_state'):
            try:
                state = sm.get_state(region)
                return state.name if hasattr(state, 'name') else str(state)
            except:
                pass

        if hasattr(sm, 'regions'):
            state = sm.regions.get(region)
            if state:
                return state.name if hasattr(state, 'name') else str(state)

        return "unknown"

    def _set_region_state(self, sm: Any, region: str, state_name: str, namespace: Dict):
        """Try to set region to specific state (for testing transitions from non-initial states)."""
        if not hasattr(sm, 'regions'):
            return

        # Find the enum for this region's states
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, EnumType):
                if hasattr(obj, state_name):
                    sm.regions[region] = getattr(obj, state_name)
                    return


def validate_parallel_code(code: str, sc: Dict, class_name: str) -> ParallelValidationResult:
    """Convenience function to validate parallel code."""
    validator = ParallelValidator()
    return validator.validate(code, sc, class_name)


if __name__ == "__main__":
    # Test with the parallel few-shot example code
    test_code = '''from enum import Enum, auto

class MovementState(Enum):
    Standing = auto()
    Walking = auto()
    Running = auto()

class CombatState(Enum):
    Idle = auto()
    Attacking = auto()
    Defending = auto()

class Event(Enum):
    WALK = auto()
    RUN = auto()
    SLOW = auto()
    STOP = auto()
    ATTACK = auto()
    FINISH = auto()
    DEFEND = auto()

class Player:
    def __init__(self):
        self.regions = {
            'Movement': MovementState.Standing,
            'Combat': CombatState.Idle
        }
        self.transitions = {
            ('Movement', MovementState.Standing, Event.WALK): MovementState.Walking,
            ('Movement', MovementState.Walking, Event.RUN): MovementState.Running,
            ('Movement', MovementState.Running, Event.SLOW): MovementState.Walking,
            ('Movement', MovementState.Walking, Event.STOP): MovementState.Standing,
            ('Combat', CombatState.Idle, Event.ATTACK): CombatState.Attacking,
            ('Combat', CombatState.Attacking, Event.FINISH): CombatState.Idle,
            ('Combat', CombatState.Idle, Event.DEFEND): CombatState.Defending,
            ('Combat', CombatState.Defending, Event.FINISH): CombatState.Idle,
        }

    def send(self, event):
        transitioned = False
        for region, state in list(self.regions.items()):
            key = (region, state, event)
            if key in self.transitions:
                self.regions[region] = self.transitions[key]
                transitioned = True
        return transitioned

    def get_state(self, region):
        return self.regions.get(region)
'''

    test_sc = {
        "root_state": {
            "label": "__root__",
            "type": 3,
            "children": [
                {
                    "label": "Movement",
                    "type": 2,
                    "children": [
                        {"label": "Standing", "type": 1, "is_initial": True},
                        {"label": "Walking", "type": 1},
                        {"label": "Running", "type": 1}
                    ]
                },
                {
                    "label": "Combat",
                    "type": 2,
                    "children": [
                        {"label": "Idle", "type": 1, "is_initial": True},
                        {"label": "Attacking", "type": 1},
                        {"label": "Defending", "type": 1}
                    ]
                }
            ]
        },
        "transitions": [
            {"from": ["Standing"], "to": ["Walking"], "event": "WALK"},
            {"from": ["Walking"], "to": ["Running"], "event": "RUN"},
            {"from": ["Running"], "to": ["Walking"], "event": "SLOW"},
            {"from": ["Walking"], "to": ["Standing"], "event": "STOP"},
            {"from": ["Idle"], "to": ["Attacking"], "event": "ATTACK"},
            {"from": ["Attacking"], "to": ["Idle"], "event": "FINISH"},
            {"from": ["Idle"], "to": ["Defending"], "event": "DEFEND"},
            {"from": ["Defending"], "to": ["Idle"], "event": "FINISH"}
        ]
    }

    result = validate_parallel_code(test_code, test_sc, "Player")
    print(f"Syntax valid: {result.syntax_valid}")
    print(f"Runnable: {result.runnable}")
    print(f"Has regions dict: {result.has_regions_dict}")
    print(f"Regions: {result.regions_correct}/{result.regions_total}")
    print(f"Transitions: {result.transitions_correct}/{result.transitions_total}")
    print(f"Overall correct: {result.overall_correct}")
    print("\nTest details:")
    for test in result.test_details:
        status = "PASS" if test.get("passed") else "FAIL"
        print(f"  {status}: {test.get('test')} (expected={test.get('expected')}, actual={test.get('actual')})")
