"""
exp_parallel_regions: Test hierarchical generation with parallel (AND) regions.

Parallel states (type=3) have multiple orthogonal regions that:
1. Are all active simultaneously when parent is active
2. Advance independently based on their own transitions
3. Can handle events targeted at specific regions

This extends the hierarchical constrained generation to AND-decomposition.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/grammars')

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
import json

from dynamic_constrained_sampler import SCDefinition, Configuration


# Parallel state SC: A machine with orthogonal regions
# Example: A car has both Engine state and AC state running in parallel
CAR_PARALLEL_SC = {
    "name": "Car",
    "root_state": {
        "label": "__root__",
        "type": 2,  # Normal (OR) at root
        "children": [
            {"label": "Off", "type": 1, "is_initial": True},
            {
                "label": "Running",
                "type": 3,  # PARALLEL - AND-decomposition
                "children": [
                    {
                        "label": "EngineRegion",
                        "type": 2,  # OR region
                        "children": [
                            {"label": "Idle", "type": 1, "is_initial": True},
                            {"label": "Accelerating", "type": 1},
                            {"label": "Cruising", "type": 1}
                        ]
                    },
                    {
                        "label": "ACRegion",
                        "type": 2,  # OR region
                        "children": [
                            {"label": "ACOff", "type": 1, "is_initial": True},
                            {"label": "ACOn", "type": 1},
                            {"label": "ACMax", "type": 1}
                        ]
                    }
                ]
            }
        ]
    },
    "transitions": [
        # Top-level transitions
        {"from": ["Off"], "to": ["Running"], "event": "START"},
        {"from": ["Running"], "to": ["Off"], "event": "STOP"},

        # Engine region transitions
        {"from": ["Idle"], "to": ["Accelerating"], "event": "ACCELERATE"},
        {"from": ["Accelerating"], "to": ["Cruising"], "event": "CRUISE"},
        {"from": ["Cruising"], "to": ["Idle"], "event": "BRAKE"},
        {"from": ["Accelerating"], "to": ["Idle"], "event": "BRAKE"},

        # AC region transitions (independent)
        {"from": ["ACOff"], "to": ["ACOn"], "event": "AC_ON"},
        {"from": ["ACOn"], "to": ["ACMax"], "event": "AC_MAX"},
        {"from": ["ACMax"], "to": ["ACOn"], "event": "AC_NORMAL"},
        {"from": ["ACOn"], "to": ["ACOff"], "event": "AC_OFF"},
        {"from": ["ACMax"], "to": ["ACOff"], "event": "AC_OFF"},
    ]
}


# Simpler parallel SC for basic testing
SIMPLE_PARALLEL_SC = {
    "name": "SimpleParallel",
    "root_state": {
        "label": "__root__",
        "type": 3,  # Root is parallel
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
    },
    "transitions": [
        {"from": ["A1"], "to": ["A2"], "event": "NEXT_A"},
        {"from": ["A2"], "to": ["A1"], "event": "RESET_A"},
        {"from": ["B1"], "to": ["B2"], "event": "NEXT_B"},
        {"from": ["B2"], "to": ["B1"], "event": "RESET_B"},
    ]
}


class ParallelRegionExecutor:
    """
    Extended executor that properly handles parallel regions.

    Key semantics:
    1. When entering a parallel state, enter initial states of ALL regions
    2. Each region advances independently
    3. Events only affect their targeted region
    """

    def __init__(self, sc_json: dict):
        self.sc_json = sc_json
        self.states = {}  # label -> state info
        self.transitions = sc_json.get("transitions", [])
        self.parallel_regions = {}  # parallel_state -> [region_labels]
        self.root_type = sc_json.get("root_state", {}).get("type", 2)  # Track root type
        self.root_children = [c.get("label") for c in sc_json.get("root_state", {}).get("children", [])]
        self._parse_states(sc_json.get("root_state", {}), None)

    def _parse_states(self, state: dict, parent: Optional[str]):
        """Parse state hierarchy, tracking parallel regions."""
        label = state.get("label", "")
        state_type = state.get("type", 1)
        children = state.get("children", [])

        if label and not label.startswith("__"):
            self.states[label] = {
                "type": state_type,
                "parent": parent,
                "children": [c.get("label") for c in children if c.get("label")],
                "is_initial": state.get("is_initial", False),
            }

            # Track parallel regions
            if state_type == 3:  # PARALLEL
                self.parallel_regions[label] = [
                    c.get("label") for c in children if c.get("label")
                ]

        for child in children:
            child_label = child.get("label", "")
            self._parse_states(child, label if label and not label.startswith("__") else parent)

    def initial_config(self) -> Set[str]:
        """Get initial configuration, entering all parallel regions."""
        active = set()

        def enter_initial(state_label: Optional[str]):
            """Recursively enter initial states."""
            if state_label is None:
                # Root level handling
                if self.root_type == 3:  # PARALLEL root - enter all regions
                    for child_label in self.root_children:
                        if child_label:
                            # Don't add region, just enter its initial state
                            enter_initial_in_region(child_label)
                else:  # OR root - find initial states
                    for label, info in self.states.items():
                        if info["parent"] is None and info["is_initial"]:
                            active.add(label)
                            enter_children(label)
            else:
                info = self.states.get(state_label, {})
                if info.get("type") == 3:  # PARALLEL - enter all regions
                    for region in info.get("children", []):
                        # Don't add region, just enter its initial state
                        enter_initial_in_region(region)
                elif info.get("type") == 2:  # OR - enter initial child
                    enter_initial_in_region(state_label)

        def enter_initial_in_region(region_label: str):
            """Enter initial state within a region."""
            region_info = self.states.get(region_label, {})
            for child in region_info.get("children", []):
                child_info = self.states.get(child, {})
                if child_info.get("is_initial"):
                    active.add(child)
                    enter_children(child)
                    break

        def enter_children(label: str):
            """Enter children of a state."""
            info = self.states.get(label, {})
            if info.get("type") == 3:  # PARALLEL
                for child in info.get("children", []):
                    # Don't add region, just enter its initial state
                    enter_initial_in_region(child)
            elif info.get("type") == 2:  # OR
                enter_initial_in_region(label)

        enter_initial(None)
        return active

    def enabled_events(self, active: Set[str]) -> Set[str]:
        """Get events enabled from current configuration."""
        enabled = set()
        for t in self.transitions:
            from_states = set(t.get("from", []))
            if from_states & active:
                event = t.get("event", "")
                if event:
                    enabled.add(event)
        return enabled

    def step(self, active: Set[str], event: str) -> Optional[Set[str]]:
        """Execute event, returning new configuration."""
        for t in self.transitions:
            from_states = set(t.get("from", []))
            to_states = set(t.get("to", []))
            t_event = t.get("event", "")

            if t_event == event and (from_states & active):
                # Execute: remove source, add target
                new_active = active - from_states

                # For each target state, enter it properly (handle composite states)
                for target in to_states:
                    new_active = new_active | self._enter_state(target)

                return new_active

        return None

    def _enter_state(self, state_label: str) -> Set[str]:
        """Enter a state, recursively entering initial substates if composite."""
        result = set()
        info = self.states.get(state_label, {})
        state_type = info.get("type", 1)

        if state_type == 1:  # BASIC - just add this state
            result.add(state_label)
        elif state_type == 3:  # PARALLEL - enter all regions
            for region in info.get("children", []):
                result = result | self._enter_region(region)
        elif state_type == 2:  # OR - enter initial child
            result = result | self._enter_region(state_label)

        return result

    def _enter_region(self, region_label: str) -> Set[str]:
        """Enter the initial state within a region."""
        result = set()
        region_info = self.states.get(region_label, {})
        for child in region_info.get("children", []):
            child_info = self.states.get(child, {})
            if child_info.get("is_initial"):
                result = result | self._enter_state(child)
                break
        return result


def test_parallel_initial_config():
    """Test that parallel regions initialize correctly."""
    executor = ParallelRegionExecutor(SIMPLE_PARALLEL_SC)
    active = executor.initial_config()

    # Should have both A1 and B1 active
    assert "A1" in active, f"A1 should be active, got {active}"
    assert "B1" in active, f"B1 should be active, got {active}"

    return True


def test_independent_region_advancement():
    """Test that regions advance independently."""
    executor = ParallelRegionExecutor(SIMPLE_PARALLEL_SC)
    active = executor.initial_config()

    # Initial: {A1, B1}
    assert active == {"A1", "B1"}, f"Initial should be {{A1, B1}}, got {active}"

    # Advance region A only
    active = executor.step(active, "NEXT_A")
    assert active == {"A2", "B1"}, f"After NEXT_A should be {{A2, B1}}, got {active}"

    # Advance region B only
    active = executor.step(active, "NEXT_B")
    assert active == {"A2", "B2"}, f"After NEXT_B should be {{A2, B2}}, got {active}"

    # Reset A
    active = executor.step(active, "RESET_A")
    assert active == {"A1", "B2"}, f"After RESET_A should be {{A1, B2}}, got {active}"

    return True


def test_car_parallel_sc():
    """Test the car SC with engine and AC regions."""
    executor = ParallelRegionExecutor(CAR_PARALLEL_SC)
    active = executor.initial_config()

    # Initial: Off
    assert "Off" in active, f"Should start Off, got {active}"

    # Start car - should enter both regions
    active = executor.step(active, "START")
    assert "Idle" in active, f"Engine should be Idle, got {active}"
    assert "ACOff" in active, f"AC should be Off, got {active}"

    # Accelerate (engine only)
    active = executor.step(active, "ACCELERATE")
    assert "Accelerating" in active, f"Engine should be Accelerating, got {active}"
    assert "ACOff" in active, f"AC should still be Off, got {active}"

    # Turn on AC (AC only, independent)
    active = executor.step(active, "AC_ON")
    assert "Accelerating" in active, f"Engine should still be Accelerating, got {active}"
    assert "ACOn" in active, f"AC should be On, got {active}"

    # Cruise (engine only)
    active = executor.step(active, "CRUISE")
    assert "Cruising" in active, f"Engine should be Cruising, got {active}"
    assert "ACOn" in active, f"AC should still be On, got {active}"

    return True


def test_enabled_events_parallel():
    """Test that enabled events come from all active regions."""
    executor = ParallelRegionExecutor(SIMPLE_PARALLEL_SC)
    active = executor.initial_config()  # {A1, B1}

    enabled = executor.enabled_events(active)

    # Should have events from both regions
    assert "NEXT_A" in enabled, f"NEXT_A should be enabled, got {enabled}"
    assert "NEXT_B" in enabled, f"NEXT_B should be enabled, got {enabled}"

    # After advancing A
    active = executor.step(active, "NEXT_A")  # {A2, B1}
    enabled = executor.enabled_events(active)

    assert "RESET_A" in enabled, f"RESET_A should be enabled, got {enabled}"
    assert "NEXT_B" in enabled, f"NEXT_B should still be enabled, got {enabled}"

    return True


def test_parallel_with_dynamic_sampler():
    """Test parallel regions with DynamicConstrainedSampler."""
    # DynamicConstrainedSampler uses SCExecutor for trace generation
    # We test if it can handle parallel-like scenarios by running
    # the executor on the car parallel SC

    executor = ParallelRegionExecutor(CAR_PARALLEL_SC)

    # Test that we can generate valid traces through both regions
    active = executor.initial_config()
    assert "Off" in active

    # Start the car
    active = executor.step(active, "START")
    assert "Idle" in active and "ACOff" in active

    # Generate a mixed trace through both regions
    trace = []
    for event in ["ACCELERATE", "AC_ON", "CRUISE", "AC_MAX", "BRAKE", "AC_OFF"]:
        new_active = executor.step(active, event)
        if new_active:
            trace.append(event)
            active = new_active

    # Should have executed all events
    assert len(trace) == 6, f"Expected 6 events, got {len(trace)}"

    # Final state should be Idle + ACOff
    assert "Idle" in active, f"Expected Idle, got {active}"
    assert "ACOff" in active, f"Expected ACOff, got {active}"

    return True


def benchmark():
    """Run all parallel region tests."""
    results = {
        "parallel_initial_config": test_parallel_initial_config(),
        "independent_region_advancement": test_independent_region_advancement(),
        "car_parallel_sc": test_car_parallel_sc(),
        "enabled_events_parallel": test_enabled_events_parallel(),
        "parallel_with_dynamic_sampler": test_parallel_with_dynamic_sampler(),
    }

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Parallel Regions Test Results:")
    print(f"  Passed: {passed}/{total}")
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    return results


if __name__ == "__main__":
    benchmark()
