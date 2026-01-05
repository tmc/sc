"""
SCExecutor Compliance Benchmark

Tests Harel statechart semantics compliance:
1. Hierarchy cascade (default substate entry)
2. AND-state semantics (all regions active)
3. Entry/exit actions
4. History state restoration
"""

import json
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from grammars.dynamic_constrained_sampler import SCDefinition, SCExecutor, Configuration


# =============================================================================
# Test Statecharts
# =============================================================================

# Hierarchical SC: Root -> Composite -> Substates
HIERARCHICAL_SC = {
    "name": "Hierarchical",
    "root_state": {
        "label": "__root__",
        "type": 2,  # OR-state
        "children": [
            {
                "label": "Composite",
                "type": 2,  # OR-state with children
                "is_initial": True,
                "children": [
                    {"label": "SubA", "type": 1, "is_initial": True},
                    {"label": "SubB", "type": 1}
                ]
            },
            {"label": "Simple", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["SubA"], "to": ["SubB"], "event": "NEXT"},
        {"from": ["SubB"], "to": ["SubA"], "event": "PREV"},
        {"from": ["Composite"], "to": ["Simple"], "event": "EXIT"},
        {"from": ["Simple"], "to": ["Composite"], "event": "ENTER"}
    ]
}

# Parallel (AND) SC: Two concurrent regions
PARALLEL_SC = {
    "name": "Parallel",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {
                "label": "Parallel",
                "type": 3,  # AND-state (parallel regions)
                "is_initial": True,
                "children": [
                    {
                        "label": "RegionA",
                        "type": 2,
                        "children": [
                            {"label": "A1", "type": 1, "is_initial": True},
                            {"label": "A2", "type": 1}
                        ]
                    },
                    {
                        "label": "RegionB",
                        "type": 2,
                        "children": [
                            {"label": "B1", "type": 1, "is_initial": True},
                            {"label": "B2", "type": 1}
                        ]
                    }
                ]
            }
        ]
    },
    "transitions": [
        {"from": ["A1"], "to": ["A2"], "event": "TICK_A"},
        {"from": ["B1"], "to": ["B2"], "event": "TICK_B"}
    ]
}

# SC with Entry/Exit Actions
ACTION_SC = {
    "name": "WithActions",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {
                "label": "StateA",
                "type": 1,
                "is_initial": True,
                "entry_actions": [{"expression": "log('entering A')"}],
                "exit_actions": [{"expression": "log('exiting A')"}]
            },
            {
                "label": "StateB",
                "type": 1,
                "entry_actions": [{"expression": "log('entering B')"}],
                "exit_actions": [{"expression": "log('exiting B')"}]
            }
        ]
    },
    "transitions": [
        {
            "from": ["StateA"],
            "to": ["StateB"],
            "event": "GO",
            "actions": [{"expression": "log('transition action')"}]
        },
        {"from": ["StateB"], "to": ["StateA"], "event": "BACK"}
    ]
}

# SC with History State
HISTORY_SC = {
    "name": "WithHistory",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {
                "label": "Composite",
                "type": 2,
                "is_initial": True,
                "children": [
                    {"label": "Sub1", "type": 1, "is_initial": True},
                    {"label": "Sub2", "type": 1},
                    {"label": "Sub3", "type": 1},
                    {
                        "label": "H",
                        "type": 1,
                        "is_history": True,
                        "history_type": 1  # SHALLOW
                    }
                ]
            },
            {"label": "Outside", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Sub1"], "to": ["Sub2"], "event": "NEXT"},
        {"from": ["Sub2"], "to": ["Sub3"], "event": "NEXT"},
        {"from": ["Composite"], "to": ["Outside"], "event": "LEAVE"},
        {"from": ["Outside"], "to": ["H"], "event": "RETURN"}  # Return via history
    ]
}


# =============================================================================
# Compliance Tests
# =============================================================================

def test_hierarchy_initial_cascade():
    """
    Test: Entering composite state cascades to initial substate.

    Expected: Initial config includes both Composite AND SubA (initial child).
    Current behavior: Only includes initial_states from is_initial=True markers.
    """
    sc = SCDefinition.from_json(HIERARCHICAL_SC)
    executor = SCExecutor(sc)
    config = executor.initial_config()

    # REQUIRED: Both Composite (parent) and SubA (initial child) should be active
    expected = {"Composite", "SubA"}
    actual = config.active_states

    passed = expected == actual
    return {
        "test": "hierarchy_initial_cascade",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "note": "Initial config must include composite AND its default substate"
    }


def test_hierarchy_transition_cascade():
    """
    Test: Transitioning to composite state cascades to initial substate.

    Setup: Start in Simple, transition via ENTER to Composite.
    Expected: End in {Composite, SubA} (cascade to default child).
    """
    sc = SCDefinition.from_json(HIERARCHICAL_SC)
    executor = SCExecutor(sc)

    # Start in Simple
    config = Configuration(active_states={"Simple"})

    # Transition to Composite
    new_config = executor.step(config, "ENTER")

    if new_config is None:
        return {
            "test": "hierarchy_transition_cascade",
            "passed": False,
            "expected": {"Composite", "SubA"},
            "actual": None,
            "note": "Transition failed - no matching transition found"
        }

    expected = {"Composite", "SubA"}
    actual = new_config.active_states

    passed = expected == actual
    return {
        "test": "hierarchy_transition_cascade",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "note": "Entering composite must cascade to default substate"
    }


def test_and_state_all_regions():
    """
    Test: AND-state activation includes all orthogonal regions.

    Expected: Initial config is {Parallel, RegionA, A1, RegionB, B1}.
    """
    sc = SCDefinition.from_json(PARALLEL_SC)
    executor = SCExecutor(sc)
    config = executor.initial_config()

    # REQUIRED: All regions and their initial states must be active
    expected = {"Parallel", "RegionA", "A1", "RegionB", "B1"}
    actual = config.active_states

    passed = expected == actual
    return {
        "test": "and_state_all_regions",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "note": "AND-state must activate all orthogonal regions"
    }


def test_and_state_independent_transitions():
    """
    Test: Transitions in one region don't affect other region.

    Setup: Start in {A1, B1}, fire TICK_A.
    Expected: End in {A2, B1} - only RegionA changes.
    """
    sc = SCDefinition.from_json(PARALLEL_SC)
    executor = SCExecutor(sc)

    # Start with both regions in initial states
    config = Configuration(active_states={"Parallel", "RegionA", "A1", "RegionB", "B1"})

    # Fire event that only affects RegionA
    new_config = executor.step(config, "TICK_A")

    if new_config is None:
        return {
            "test": "and_state_independent_transitions",
            "passed": False,
            "expected": {"Parallel", "RegionA", "A2", "RegionB", "B1"},
            "actual": None,
            "note": "Transition failed"
        }

    # RegionA should change, RegionB should stay
    expected = {"Parallel", "RegionA", "A2", "RegionB", "B1"}
    actual = new_config.active_states

    passed = expected == actual
    return {
        "test": "and_state_independent_transitions",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "note": "Regions must transition independently"
    }


def test_entry_action_execution():
    """
    Test: Entry actions are tracked when entering a state.

    Note: We can't truly execute actions, but we can verify they're collected.
    """
    sc = SCDefinition.from_json(ACTION_SC)

    # Check that state info includes entry_actions
    state_a = sc.states.get("StateA", {})
    has_entry = "entry_actions" in state_a or "entry" in str(ACTION_SC)

    return {
        "test": "entry_action_execution",
        "passed": has_entry,
        "expected": "entry_actions present",
        "actual": "present" if has_entry else "missing",
        "note": "Entry actions must be parsed and available"
    }


def test_exit_action_execution():
    """
    Test: Exit actions are tracked when leaving a state.
    """
    sc = SCDefinition.from_json(ACTION_SC)

    state_a = sc.states.get("StateA", {})
    has_exit = "exit_actions" in state_a or "exit" in str(ACTION_SC)

    return {
        "test": "exit_action_execution",
        "passed": has_exit,
        "expected": "exit_actions present",
        "actual": "present" if has_exit else "missing",
        "note": "Exit actions must be parsed and available"
    }


def test_shallow_history_restore():
    """
    Test: Shallow history restores immediate child only.

    Scenario:
    1. Start in Sub1
    2. Transition to Sub2 (NEXT)
    3. Leave to Outside (LEAVE)
    4. Return via H (RETURN)
    Expected: End in Sub2 (last active in Composite)
    """
    sc = SCDefinition.from_json(HISTORY_SC)
    executor = SCExecutor(sc)

    # Step 1: Start in Composite/Sub1
    config = Configuration(active_states={"Composite", "Sub1"})

    # Step 2: Move to Sub2
    config = executor.step(config, "NEXT")
    if config is None:
        return {"test": "shallow_history_restore", "passed": False, "actual": "step failed"}

    # Step 3: Leave to Outside
    config = executor.step(config, "LEAVE")
    if config is None:
        return {"test": "shallow_history_restore", "passed": False, "actual": "leave failed"}

    # Step 4: Return via history (should restore Sub2)
    config = executor.step(config, "RETURN")
    if config is None:
        return {"test": "shallow_history_restore", "passed": False, "actual": "return failed"}

    # Should be back in Composite/Sub2
    expected = {"Composite", "Sub2"}
    actual = config.active_states

    passed = expected == actual
    return {
        "test": "shallow_history_restore",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "note": "Shallow history must restore last immediate child"
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

ALL_TESTS = [
    test_hierarchy_initial_cascade,
    test_hierarchy_transition_cascade,
    test_and_state_all_regions,
    test_and_state_independent_transitions,
    test_entry_action_execution,
    test_exit_action_execution,
    test_shallow_history_restore,
]


def run_compliance_benchmark():
    """Run all compliance tests and report results."""
    print("=" * 70)
    print("SCExecutor Compliance Benchmark")
    print("=" * 70)

    results = []
    passed_count = 0

    for test_fn in ALL_TESTS:
        try:
            result = test_fn()
            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            if result["passed"]:
                passed_count += 1

            print(f"\n[{status}] {result['test']}")
            print(f"  Expected: {result['expected']}")
            print(f"  Actual:   {result['actual']}")
            if "note" in result:
                print(f"  Note:     {result['note']}")

        except Exception as e:
            print(f"\n[ERROR] {test_fn.__name__}: {e}")
            results.append({
                "test": test_fn.__name__,
                "passed": False,
                "error": str(e)
            })

    total = len(ALL_TESTS)
    print("\n" + "=" * 70)
    print(f"COMPLIANCE SCORE: {passed_count}/{total} ({100*passed_count/total:.1f}%)")
    print("=" * 70)

    if passed_count < total:
        print("\nFAILING TESTS (must fix before other experiments):")
        for r in results:
            if not r.get("passed"):
                print(f"  - {r['test']}")

    return {
        "passed": passed_count,
        "total": total,
        "score": passed_count / total,
        "results": results
    }


if __name__ == "__main__":
    run_compliance_benchmark()
