"""
Test cases for SC debugging.

Each test case has:
- A statechart (with a bug)
- An execution trace that fails
- Ground truth: what's wrong and how to fix it
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class BugType(Enum):
    """Types of bugs in statecharts."""
    MISSING_TRANSITION = "missing_transition"
    WRONG_TARGET = "wrong_target"
    WRONG_EVENT = "wrong_event"
    MISSING_GUARD = "missing_guard"
    WRONG_GUARD = "wrong_guard"
    UNREACHABLE_STATE = "unreachable_state"


@dataclass
class DebugTestCase:
    """A test case for SC debugging."""
    name: str
    description: str
    statechart: Dict[str, Any]  # The buggy SC
    trace: List[Dict[str, Any]]  # Execution trace that fails
    expected_behavior: str  # What should happen
    actual_behavior: str  # What actually happens
    bug_type: BugType
    bug_location: str  # Which transition/state is buggy
    fix_description: str  # How to fix it
    correct_statechart: Optional[Dict[str, Any]] = None  # Fixed version


def generate_test_cases() -> List[DebugTestCase]:
    """Generate test cases for debugging benchmark."""
    cases = []

    # Case 1: Missing transition (toggle switch missing OFF->ON)
    cases.append(DebugTestCase(
        name="toggle_missing_on",
        description="Toggle switch missing transition from Off to On",
        statechart={
            "root_state": {
                "label": "Toggle",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
                # Missing: Off -> On on TOGGLE
            ]
        },
        trace=[
            {"state": "Off", "event": "TOGGLE", "result": "stuck_in_Off"}
        ],
        expected_behavior="When in Off state and TOGGLE event occurs, should transition to On",
        actual_behavior="System stays in Off state, TOGGLE event has no effect",
        bug_type=BugType.MISSING_TRANSITION,
        bug_location="transitions",
        fix_description="Add transition from Off to On on TOGGLE event",
        correct_statechart={
            "root_state": {
                "label": "Toggle",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"}
            ]
        }
    ))

    # Case 2: Wrong target state
    cases.append(DebugTestCase(
        name="traffic_light_wrong_target",
        description="Traffic light goes to wrong state",
        statechart={
            "root_state": {
                "label": "TrafficLight",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Yellow", "type": 1},
                    {"label": "Green", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                {"from": ["Green"], "to": ["Red"], "event": "NEXT"},  # BUG: should go to Yellow
                {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"}
            ]
        },
        trace=[
            {"state": "Red", "event": "NEXT", "result": "Green"},
            {"state": "Green", "event": "NEXT", "result": "Red"}  # Should be Yellow!
        ],
        expected_behavior="Green -> NEXT -> Yellow (then Yellow -> NEXT -> Red)",
        actual_behavior="Green -> NEXT -> Red (skips Yellow)",
        bug_type=BugType.WRONG_TARGET,
        bug_location="transition from Green",
        fix_description="Change transition from Green to target Yellow instead of Red"
    ))

    # Case 3: Wrong event name
    cases.append(DebugTestCase(
        name="door_wrong_event",
        description="Door uses wrong event name",
        statechart={
            "root_state": {
                "label": "Door",
                "type": 2,
                "children": [
                    {"label": "Closed", "type": 1, "is_initial": True},
                    {"label": "Open", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "UNLOCK"},  # BUG: should be OPEN
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"}
            ]
        },
        trace=[
            {"state": "Closed", "event": "OPEN", "result": "stuck_in_Closed"}
        ],
        expected_behavior="OPEN event should open the door",
        actual_behavior="OPEN event does nothing, door expects UNLOCK",
        bug_type=BugType.WRONG_EVENT,
        bug_location="transition from Closed",
        fix_description="Change event from UNLOCK to OPEN"
    ))

    # Case 4: Missing guard condition
    cases.append(DebugTestCase(
        name="vending_missing_guard",
        description="Vending machine dispenses without payment",
        statechart={
            "root_state": {
                "label": "VendingMachine",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Paid", "type": 1},
                    {"label": "Dispensing", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Paid"], "event": "INSERT_COIN"},
                {"from": ["Idle"], "to": ["Dispensing"], "event": "SELECT"},  # BUG: missing guard
                {"from": ["Paid"], "to": ["Dispensing"], "event": "SELECT"},
                {"from": ["Dispensing"], "to": ["Idle"], "event": "DONE"}
            ]
        },
        trace=[
            {"state": "Idle", "event": "SELECT", "result": "Dispensing"}  # Should not happen!
        ],
        expected_behavior="SELECT from Idle should do nothing (no payment)",
        actual_behavior="SELECT from Idle goes to Dispensing without payment",
        bug_type=BugType.MISSING_GUARD,
        bug_location="transition from Idle to Dispensing",
        fix_description="Remove transition from Idle to Dispensing, or add guard [hasPaid]"
    ))

    # Case 5: Unreachable state
    cases.append(DebugTestCase(
        name="game_unreachable_state",
        description="Game has unreachable victory state",
        statechart={
            "root_state": {
                "label": "Game",
                "type": 2,
                "children": [
                    {"label": "Playing", "type": 1, "is_initial": True},
                    {"label": "Paused", "type": 1},
                    {"label": "Victory", "type": 1},  # Unreachable!
                    {"label": "GameOver", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "RESUME"},
                {"from": ["Playing"], "to": ["GameOver"], "event": "LOSE"}
                # Missing: Playing -> Victory on WIN
            ]
        },
        trace=[
            {"state": "Playing", "event": "WIN", "result": "stuck_in_Playing"}
        ],
        expected_behavior="WIN event should transition to Victory state",
        actual_behavior="WIN event has no effect, Victory state is unreachable",
        bug_type=BugType.UNREACHABLE_STATE,
        bug_location="Victory state",
        fix_description="Add transition from Playing to Victory on WIN event"
    ))

    # Case 6: Wrong guard logic
    cases.append(DebugTestCase(
        name="elevator_wrong_guard",
        description="Elevator moves when doors are open",
        statechart={
            "root_state": {
                "label": "Elevator",
                "type": 2,
                "children": [
                    {"label": "Stopped", "type": 1, "is_initial": True},
                    {"label": "Moving", "type": 1},
                    {"label": "DoorsOpen", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["DoorsOpen"], "event": "OPEN_DOORS"},
                {"from": ["DoorsOpen"], "to": ["Stopped"], "event": "CLOSE_DOORS"},
                {"from": ["Stopped"], "to": ["Moving"], "event": "GO", "guard": "doorsOpen"},  # BUG: wrong guard
                {"from": ["Moving"], "to": ["Stopped"], "event": "ARRIVED"}
            ]
        },
        trace=[
            {"state": "Stopped", "event": "OPEN_DOORS", "result": "DoorsOpen"},
            {"state": "DoorsOpen", "event": "CLOSE_DOORS", "result": "Stopped"},
            {"state": "Stopped", "event": "GO", "context": {"doorsOpen": True}, "result": "Moving"}
        ],
        expected_behavior="Elevator should only move when doors are closed",
        actual_behavior="Elevator moves when guard doorsOpen is true (wrong polarity)",
        bug_type=BugType.WRONG_GUARD,
        bug_location="transition from Stopped to Moving",
        fix_description="Change guard from 'doorsOpen' to '!doorsOpen' or 'doorsClosed'"
    ))

    # Case 7: Hierarchical state missing internal transition
    cases.append(DebugTestCase(
        name="phone_missing_internal",
        description="Phone call doesn't handle mute in active state",
        statechart={
            "root_state": {
                "label": "PhoneCall",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {
                        "label": "Active",
                        "type": 2,
                        "children": [
                            {"label": "Talking", "type": 1, "is_initial": True},
                            {"label": "Muted", "type": 1}
                        ]
                    },
                    {"label": "Ended", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Active"], "event": "ANSWER"},
                {"from": ["Active"], "to": ["Ended"], "event": "HANGUP"},
                {"from": ["Talking"], "to": ["Muted"], "event": "MUTE"}
                # Missing: Muted -> Talking on UNMUTE
            ]
        },
        trace=[
            {"state": "Idle", "event": "ANSWER", "result": "Active.Talking"},
            {"state": "Active.Talking", "event": "MUTE", "result": "Active.Muted"},
            {"state": "Active.Muted", "event": "UNMUTE", "result": "stuck_in_Muted"}
        ],
        expected_behavior="UNMUTE should transition from Muted back to Talking",
        actual_behavior="UNMUTE has no effect, caller stays muted",
        bug_type=BugType.MISSING_TRANSITION,
        bug_location="transitions within Active state",
        fix_description="Add transition from Muted to Talking on UNMUTE event"
    ))

    # Case 8: Conflicting transitions (non-deterministic)
    cases.append(DebugTestCase(
        name="login_conflicting",
        description="Login has conflicting transitions on same event",
        statechart={
            "root_state": {
                "label": "Login",
                "type": 2,
                "children": [
                    {"label": "Form", "type": 1, "is_initial": True},
                    {"label": "Success", "type": 1},
                    {"label": "Error", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Form"], "to": ["Success"], "event": "SUBMIT"},
                {"from": ["Form"], "to": ["Error"], "event": "SUBMIT"},  # Conflict!
                {"from": ["Error"], "to": ["Form"], "event": "RETRY"}
            ]
        },
        trace=[
            {"state": "Form", "event": "SUBMIT", "result": "non_deterministic"}
        ],
        expected_behavior="SUBMIT should go to Success if valid, Error if invalid",
        actual_behavior="Two transitions match SUBMIT from Form, behavior is undefined",
        bug_type=BugType.MISSING_GUARD,
        bug_location="transitions from Form on SUBMIT",
        fix_description="Add guards to distinguish: [valid]->Success, [!valid]->Error"
    ))

    return cases


def test_cases():
    """Test that all cases are valid."""
    cases = generate_test_cases()
    print(f"Generated {len(cases)} test cases:")
    for case in cases:
        print(f"  - {case.name}: {case.bug_type.value}")
    return cases


if __name__ == "__main__":
    test_cases()
