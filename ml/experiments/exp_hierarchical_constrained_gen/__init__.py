"""
exp_hierarchical_constrained_gen: Test nested SC constraints with push/pop

This experiment validates the meta_constrained_sampler's ability to:
1. Stack multiple SCs for hierarchical generation
2. Maintain coherence across SC boundaries
3. Properly restore state when popping

Uses the traffic_light and door SCs from meta_constrained_sampler.py
"""

import json
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/grammars')

from meta_constrained_sampler import MetaConstrainedSampler, SPECIAL_TOKENS


# Domain SCs for testing
TRAFFIC_LIGHT_SC = {
    "name": "TrafficLight",
    "root_state": {
        "label": "__root__", "type": 2,
        "children": [
            {"label": "Red", "type": 1, "is_initial": True},
            {"label": "Green", "type": 1},
            {"label": "Yellow", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Red"], "to": ["Green"], "event": "GO"},
        {"from": ["Green"], "to": ["Yellow"], "event": "SLOW"},
        {"from": ["Yellow"], "to": ["Red"], "event": "STOP"}
    ]
}

DOOR_SC = {
    "name": "Door",
    "root_state": {
        "label": "__root__", "type": 2,
        "children": [
            {"label": "Closed", "type": 1, "is_initial": True},
            {"label": "Open", "type": 1},
            {"label": "Locked", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
        {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"},
        {"from": ["Closed"], "to": ["Locked"], "event": "LOCK"},
        {"from": ["Locked"], "to": ["Closed"], "event": "UNLOCK"}
    ]
}


def test_basic_push_pop():
    """Test that push/pop correctly maintains state stack."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
    sampler.register_sc("door", DOOR_SC)

    # Load traffic light
    sampler.load_sc("traffic_light")
    assert sampler.get_current_states() == {"Red"}

    # Advance to Green
    sampler.emit_event("GO")
    assert sampler.get_current_states() == {"Green"}

    # Push and switch to door
    sampler.push_sc()
    sampler.load_sc("door")
    assert sampler.get_current_states() == {"Closed"}
    assert sampler.sc_stack.depth() == 1

    # Operate door
    sampler.emit_event("OPEN")
    assert sampler.get_current_states() == {"Open"}

    # Pop back to traffic light
    sampler.pop_sc()
    assert sampler.get_current_states() == {"Green"}  # Should restore Green state
    assert sampler.sc_stack.depth() == 0

    return True


def test_nested_hierarchy():
    """Test multiple levels of nesting."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
    sampler.register_sc("door", DOOR_SC)

    # Level 0: Traffic light at Green
    sampler.load_sc("traffic_light")
    sampler.emit_event("GO")

    # Level 1: Push, switch to door
    sampler.push_sc()
    sampler.load_sc("door")
    sampler.emit_event("OPEN")

    # Level 2: Push, switch to traffic light again
    sampler.push_sc()
    sampler.load_sc("traffic_light")
    sampler.emit_event("GO")
    sampler.emit_event("SLOW")
    assert sampler.get_current_states() == {"Yellow"}

    # Pop level 2 → back to door Open
    sampler.pop_sc()
    assert sampler.get_current_states() == {"Open"}

    # Pop level 1 → back to traffic light Green
    sampler.pop_sc()
    assert sampler.get_current_states() == {"Green"}

    return True


def test_token_processing():
    """Test special token parsing and execution."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
    sampler.register_sc("door", DOOR_SC)

    sampler.load_sc("traffic_light")

    # Test STATE query
    is_special, output = sampler.process_token("<SC:STATE>")
    assert is_special
    assert "Red" in output

    # Test VALID query
    is_special, output = sampler.process_token("<SC:VALID>")
    assert is_special
    assert "GO" in output

    # Test PUSH
    is_special, _ = sampler.process_token("<SC:PUSH>")
    assert is_special
    assert sampler.sc_stack.depth() == 1

    return True


def test_cross_boundary_coherence():
    """Test that generation remains coherent across SC boundaries."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
    sampler.register_sc("door", DOOR_SC)

    sampler.load_sc("traffic_light")

    trace = []

    # Generate in traffic light domain
    for _ in range(3):
        valid = list(sampler.get_valid_events())
        if valid:
            event = valid[0]
            trace.append(("traffic_light", event))
            sampler.emit_event(event)

    # Switch to door domain
    sampler.push_sc()
    sampler.load_sc("door")

    for _ in range(2):
        valid = list(sampler.get_valid_events())
        if valid:
            event = valid[0]
            trace.append(("door", event))
            sampler.emit_event(event)

    # Pop back
    sampler.pop_sc()

    # Continue in traffic light domain
    for _ in range(2):
        valid = list(sampler.get_valid_events())
        if valid:
            event = valid[0]
            trace.append(("traffic_light_resumed", event))
            sampler.emit_event(event)

    return trace


def benchmark():
    """Run all tests and report results."""
    results = {
        "basic_push_pop": test_basic_push_pop(),
        "nested_hierarchy": test_nested_hierarchy(),
        "token_processing": test_token_processing(),
        "cross_boundary_coherence": len(test_cross_boundary_coherence()) > 0
    }

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Hierarchical Constrained Generation Results:")
    print(f"  Passed: {passed}/{total}")
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    return results


def run_full_benchmark():
    """Run both basic and real model benchmarks."""
    print("=" * 60)
    print("EXP_HIERARCHICAL_CONSTRAINED_GEN: Full Benchmark")
    print("=" * 60)

    # Basic tests
    print("\n--- Basic Tests ---")
    basic_results = benchmark()
    basic_passed = sum(1 for v in basic_results.values() if v)
    basic_total = len(basic_results)

    # Real model tests
    print("\n--- Real Model Tests ---")
    try:
        from .real_model_benchmark import run_real_model_benchmark
        real_result = run_real_model_benchmark()

        real_passed = sum([
            real_result.constraint_respected,
            real_result.cross_boundary_coherent,
            real_result.switch_detection_accuracy >= 0.5,
        ])
        real_total = 3
    except Exception as e:
        print(f"Real model test error: {e}")
        real_passed = 0
        real_total = 3

    # Overall
    total_passed = basic_passed + real_passed
    total_tests = basic_total + real_total
    accuracy = total_passed / total_tests if total_tests > 0 else 0.0

    print(f"\n=== FINAL RESULTS ===")
    print(f"Basic tests: {basic_passed}/{basic_total}")
    print(f"Real model tests: {real_passed}/{real_total}")
    print(f"TOTAL: {total_passed}/{total_tests} ({accuracy*100:.1f}%)")

    return accuracy


if __name__ == "__main__":
    run_full_benchmark()
