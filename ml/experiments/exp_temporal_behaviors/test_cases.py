"""
Test Cases for Temporal Behavior Experiment.

8 test cases covering temporal statechart patterns:
TC1: Simple after() timeout
TC2: Timer reset on activity
TC3: Exponential backoff
TC4: Deadline/timeout race
TC5: Debounce pattern
TC6: Nested timeouts
TC7: Timeout in parallel regions
TC8: Time-based guards
"""

from typing import Dict, List, Any


# TC1: Simple AFTER timeout
TC1_SIMPLE_TIMEOUT = {
    "name": "simple_timeout",
    "description": "Transition after 30s of inactivity",
    "statechart": {
        "states": ["Idle", "Screensaver"],
        "initial": "Idle",
        "transitions": [
            {"from": "Idle", "to": "Screensaver", "after_ms": 30000},
            {"from": "Screensaver", "to": "Idle", "event": "activity"}
        ]
    },
    "scenarios": [
        {
            "name": "timeout_fires",
            "duration_ms": 35000,
            "events": [],
            "expected_final": "Screensaver",
            "expected_sequence": ["Idle", "Screensaver"]
        },
        {
            "name": "activity_prevents",
            "duration_ms": 35000,
            "events": [(25000, "activity")],  # Activity before timeout
            "expected_final": "Idle",
            "expected_sequence": ["Idle"]  # Never transitions
        }
    ]
}


# TC2: Timer reset on activity
TC2_ACTIVITY_RESET = {
    "name": "activity_reset",
    "description": "Activity resets the timeout timer",
    "statechart": {
        "states": ["Active", "Idle"],
        "initial": "Active",
        "transitions": [
            {"from": "Active", "to": "Idle", "after_ms": 5000},
            {"from": "Active", "to": "Active", "event": "input", "resets_timer": True},
            {"from": "Idle", "to": "Active", "event": "input"}
        ]
    },
    "scenarios": [
        {
            "name": "no_activity_timeout",
            "duration_ms": 6000,
            "events": [],
            "expected_final": "Idle",
            "expected_sequence": ["Active", "Idle"]
        },
        {
            "name": "activity_resets",
            "duration_ms": 12000,
            "events": [(3000, "input"), (6000, "input"), (9000, "input")],
            "expected_final": "Active",
            "expected_sequence": ["Active", "Active", "Active", "Active"]
        },
        {
            "name": "timeout_after_inactivity",
            "duration_ms": 10000,
            "events": [(2000, "input")],  # Activity at 2s, then nothing
            "expected_final": "Idle",
            "expected_sequence": ["Active", "Active", "Idle"]
        }
    ]
}


# TC3: Retry with exponential backoff
TC3_EXPONENTIAL_BACKOFF = {
    "name": "exponential_backoff",
    "description": "Retry with increasing wait times",
    "statechart": {
        "states": ["Connecting", "Connected", "Failed", "Waiting"],
        "initial": "Connecting",
        "transitions": [
            {"from": "Connecting", "to": "Connected", "event": "success"},
            {"from": "Connecting", "to": "Waiting", "event": "fail", "action": "attempts++"},
            {"from": "Waiting", "to": "Connecting", "after_ms": 1000, "guard": "attempts == 1"},
            {"from": "Waiting", "to": "Connecting", "after_ms": 2000, "guard": "attempts == 2"},
            {"from": "Waiting", "to": "Connecting", "after_ms": 4000, "guard": "attempts == 3"},
            {"from": "Waiting", "to": "Failed", "guard": "attempts > 3"}
        ],
        "variables": {"attempts": 0}
    },
    "scenarios": [
        {
            "name": "success_first_try",
            "duration_ms": 1000,
            "events": [(500, "success")],
            "expected_final": "Connected",
            "expected_sequence": ["Connecting", "Connected"]
        },
        {
            "name": "success_after_retry",
            "duration_ms": 5000,
            "events": [(100, "fail"), (1500, "success")],
            "expected_final": "Connected",
            "expected_sequence": ["Connecting", "Waiting", "Connecting", "Connected"]
        },
        {
            "name": "all_retries_fail",
            "duration_ms": 15000,
            "events": [(100, "fail"), (1500, "fail"), (4000, "fail"), (8500, "fail")],
            "expected_final": "Failed",
            "expected_variables": {"attempts": 4}
        }
    ]
}


# TC4: Deadline/timeout race
TC4_DEADLINE_RACE = {
    "name": "deadline_race",
    "description": "Complete before timeout or fail",
    "statechart": {
        "states": ["Processing", "Done", "TimedOut"],
        "initial": "Processing",
        "transitions": [
            {"from": "Processing", "to": "Done", "event": "complete"},
            {"from": "Processing", "to": "TimedOut", "after_ms": 10000}
        ]
    },
    "scenarios": [
        {
            "name": "complete_before_deadline",
            "duration_ms": 15000,
            "events": [(5000, "complete")],
            "expected_final": "Done",
            "expected_sequence": ["Processing", "Done"]
        },
        {
            "name": "timeout_wins",
            "duration_ms": 15000,
            "events": [],
            "expected_final": "TimedOut",
            "expected_sequence": ["Processing", "TimedOut"]
        },
        {
            "name": "complete_at_deadline",
            "duration_ms": 15000,
            "events": [(10000, "complete")],  # Right at deadline
            "expected_final": "Done",  # Event should win tie
            "expected_sequence": ["Processing", "Done"]
        }
    ]
}


# TC5: Debounce pattern
TC5_DEBOUNCE = {
    "name": "debounce",
    "description": "Wait for input to stabilize before processing",
    "statechart": {
        "states": ["Waiting", "Debouncing", "Processing"],
        "initial": "Waiting",
        "transitions": [
            {"from": "Waiting", "to": "Debouncing", "event": "input"},
            {"from": "Debouncing", "to": "Debouncing", "event": "input", "resets_timer": True},
            {"from": "Debouncing", "to": "Processing", "after_ms": 300},
            {"from": "Processing", "to": "Waiting", "event": "done"}
        ]
    },
    "scenarios": [
        {
            "name": "single_input",
            "duration_ms": 1000,
            "events": [(100, "input")],
            "expected_final": "Processing",
            "expected_sequence": ["Waiting", "Debouncing", "Processing"]
        },
        {
            "name": "rapid_inputs_debounced",
            "duration_ms": 2000,
            "events": [
                (100, "input"),
                (200, "input"),
                (250, "input"),
                (300, "input"),
                (350, "input")
            ],
            "expected_final": "Processing",
            # Should only process once after last input stabilizes
            "debounce_count": 1
        },
        {
            "name": "two_debounced_groups",
            "duration_ms": 2000,
            "events": [
                (100, "input"),
                (200, "input"),  # First group
                (1000, "input"),
                (1100, "input")  # Second group
            ],
            "expected_final": "Processing",
            "debounce_count": 2
        }
    ]
}


# TC6: Nested timeouts (hierarchical)
TC6_NESTED_TIMEOUTS = {
    "name": "nested_timeouts",
    "description": "Parent and child states both have timeouts",
    "statechart": {
        "states": ["Active", "Working", "Paused", "Idle", "Suspended"],
        "initial": "Working",
        "hierarchy": {
            "Active": ["Working", "Paused"]
        },
        "transitions": [
            # Within Active
            {"from": "Working", "to": "Paused", "event": "pause"},
            {"from": "Paused", "to": "Working", "event": "resume"},
            # Child timeout: Working -> Paused after 10s
            {"from": "Working", "to": "Paused", "after_ms": 10000},
            # Parent timeout: Active -> Idle after 60s
            {"from": "Active", "to": "Idle", "after_ms": 60000},
            # System timeout: any -> Suspended after 300s
            {"from": "Idle", "to": "Suspended", "after_ms": 240000}
        ]
    },
    "scenarios": [
        {
            "name": "child_timeout_first",
            "duration_ms": 15000,
            "events": [],
            "expected_final": "Paused",
            "expected_sequence": ["Working", "Paused"]
        },
        {
            "name": "parent_timeout_from_paused",
            "duration_ms": 65000,
            "events": [(5000, "pause")],  # Go to Paused early
            "expected_final": "Idle",
            # Should timeout at 65s from Active
        }
    ]
}


# TC7: Timeout in parallel regions
TC7_PARALLEL_TIMEOUTS = {
    "name": "parallel_timeouts",
    "description": "Independent timeouts in parallel regions",
    "statechart": {
        "states": ["Running", "Stopped"],
        "regions": {
            "Running": {
                "Display": {
                    "states": ["On", "Dimmed", "Off"],
                    "initial": "On",
                    "transitions": [
                        {"from": "On", "to": "Dimmed", "after_ms": 30000},
                        {"from": "Dimmed", "to": "Off", "after_ms": 30000},
                        {"from": "Off", "to": "On", "event": "touch"},
                        {"from": "Dimmed", "to": "On", "event": "touch"}
                    ]
                },
                "Audio": {
                    "states": ["Playing", "Quiet"],
                    "initial": "Playing",
                    "transitions": [
                        {"from": "Playing", "to": "Quiet", "after_ms": 60000},
                        {"from": "Quiet", "to": "Playing", "event": "play"}
                    ]
                }
            }
        },
        "initial": "Running"
    },
    "scenarios": [
        {
            "name": "display_dims_first",
            "duration_ms": 35000,
            "events": [],
            "expected_display": "Dimmed",
            "expected_audio": "Playing"  # Audio timeout is longer
        },
        {
            "name": "both_timeout",
            "duration_ms": 65000,
            "events": [],
            "expected_display": "Off",
            "expected_audio": "Quiet"
        },
        {
            "name": "touch_resets_display_only",
            "duration_ms": 40000,
            "events": [(35000, "touch")],
            "expected_display": "On",  # Reset by touch
            "expected_audio": "Playing"  # Not affected
        }
    ]
}


# TC8: Time-based guards
TC8_TIME_GUARDS = {
    "name": "time_guards",
    "description": "Guards that depend on elapsed time",
    "statechart": {
        "states": ["Idle", "Running", "Warmup", "Normal", "Overheated", "Cooldown"],
        "initial": "Idle",
        "transitions": [
            {"from": "Idle", "to": "Warmup", "event": "start"},
            # Can only transition to Normal after warmup period
            {"from": "Warmup", "to": "Normal", "event": "ready", "guard": "elapsed > 5000"},
            {"from": "Warmup", "to": "Warmup", "event": "ready", "guard": "elapsed <= 5000"},
            # Running too long causes overheat
            {"from": "Normal", "to": "Overheated", "guard": "elapsed > 30000"},
            {"from": "Overheated", "to": "Cooldown", "event": "stop"},
            {"from": "Cooldown", "to": "Idle", "after_ms": 10000},
            {"from": "Normal", "to": "Idle", "event": "stop"}
        ]
    },
    "scenarios": [
        {
            "name": "warmup_respected",
            "duration_ms": 10000,
            "events": [(1000, "start"), (2000, "ready")],
            "expected_final": "Warmup",  # Too early for Normal
            "reason": "Guard prevents early transition"
        },
        {
            "name": "warmup_complete",
            "duration_ms": 10000,
            "events": [(1000, "start"), (7000, "ready")],
            "expected_final": "Normal",  # 6s elapsed in Warmup
            "reason": "Guard allows transition after warmup"
        },
        {
            "name": "overheat_prevention",
            "duration_ms": 40000,
            "events": [(1000, "start"), (7000, "ready")],
            "expected_final": "Overheated",
            "reason": "Time guard triggers overheat after 30s in Normal"
        }
    ]
}


# All test cases
TEST_CASES = [
    TC1_SIMPLE_TIMEOUT,
    TC2_ACTIVITY_RESET,
    TC3_EXPONENTIAL_BACKOFF,
    TC4_DEADLINE_RACE,
    TC5_DEBOUNCE,
    TC6_NESTED_TIMEOUTS,
    TC7_PARALLEL_TIMEOUTS,
    TC8_TIME_GUARDS,
]


def get_test_case(name: str) -> Dict[str, Any]:
    """Get a test case by name."""
    for tc in TEST_CASES:
        if tc["name"] == name:
            return tc
    return {}


def get_all_scenarios() -> List[Dict[str, Any]]:
    """Get all scenarios from all test cases."""
    scenarios = []
    for tc in TEST_CASES:
        for scenario in tc.get("scenarios", []):
            scenarios.append({
                "test_case": tc["name"],
                "statechart": tc["statechart"],
                **scenario
            })
    return scenarios


if __name__ == "__main__":
    print("Temporal Behavior Test Cases")
    print("=" * 60)

    for tc in TEST_CASES:
        n_scenarios = len(tc.get("scenarios", []))
        print(f"\n{tc['name']}:")
        print(f"  Description: {tc['description']}")
        print(f"  Scenarios: {n_scenarios}")

    print(f"\nTotal test cases: {len(TEST_CASES)}")
    print(f"Total scenarios: {len(get_all_scenarios())}")
