"""
exp_history_states: Test shallow (H) and deep (H*) history state restoration.

History states remember the last active substate when exiting a composite state:
- Shallow history (H): Remembers only the immediate child
- Deep history (H*): Remembers the full nested configuration

This experiment validates:
1. Shallow history correctly restores immediate child
2. Deep history correctly restores full nested configuration
3. Default behavior when no history exists (fallback to initial state)
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_history_states')

from typing import Set, Dict, List, Any
from history_executor import HistoryExecutor, HistoryType


# =============================================================================
# Test Statecharts
# =============================================================================

# Simple SC with shallow history - models a text editor
TEXT_EDITOR_SC = {
    "name": "TextEditor",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Inactive", "type": 1, "is_initial": True},
            {
                "label": "Active",
                "type": 2,  # Normal (OR) composite
                "children": [
                    {"label": "Editing", "type": 1, "is_initial": True},
                    {"label": "Searching", "type": 1},
                    {"label": "Formatting", "type": 1},
                    {
                        "label": "H",
                        "type": 1,
                        "is_history": True,
                        "history_type": "shallow",
                    },
                ]
            },
            {
                "label": "Settings",
                "type": 2,
                "children": [
                    {"label": "General", "type": 1, "is_initial": True},
                    {"label": "Display", "type": 1},
                    {"label": "Advanced", "type": 1},
                ]
            },
        ]
    },
    "transitions": [
        # Top-level transitions
        {"from": ["Inactive"], "to": ["Active"], "event": "OPEN"},
        {"from": ["Active"], "to": ["Inactive"], "event": "CLOSE"},
        {"from": ["Active"], "to": ["Settings"], "event": "SETTINGS"},
        {"from": ["Settings"], "to": ["H"], "event": "BACK"},  # Return via history

        # Active state transitions
        {"from": ["Editing"], "to": ["Searching"], "event": "SEARCH"},
        {"from": ["Searching"], "to": ["Editing"], "event": "CANCEL"},
        {"from": ["Editing"], "to": ["Formatting"], "event": "FORMAT"},
        {"from": ["Formatting"], "to": ["Editing"], "event": "DONE"},

        # Settings transitions
        {"from": ["General"], "to": ["Display"], "event": "DISPLAY"},
        {"from": ["General"], "to": ["Advanced"], "event": "ADV"},
        {"from": ["Display"], "to": ["General"], "event": "GEN"},
        {"from": ["Advanced"], "to": ["General"], "event": "GEN"},
    ]
}


# Nested SC with deep history - models a media player with nested states
MEDIA_PLAYER_SC = {
    "name": "MediaPlayer",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Off", "type": 1, "is_initial": True},
            {
                "label": "On",
                "type": 2,
                "children": [
                    {
                        "label": "Playing",
                        "type": 2,
                        "is_initial": True,
                        "children": [
                            {"label": "Normal", "type": 1, "is_initial": True},
                            {"label": "FastForward", "type": 1},
                            {"label": "Rewind", "type": 1},
                        ]
                    },
                    {"label": "Paused", "type": 1},
                    {
                        "label": "H_star",
                        "type": 1,
                        "is_history": True,
                        "history_type": "deep",
                    },
                ]
            },
            {
                "label": "Menu",
                "type": 2,
                "children": [
                    {"label": "MainMenu", "type": 1, "is_initial": True},
                    {"label": "Playlist", "type": 1},
                ]
            },
        ]
    },
    "transitions": [
        # Power transitions
        {"from": ["Off"], "to": ["On"], "event": "POWER"},
        {"from": ["On"], "to": ["Off"], "event": "POWER"},
        {"from": ["On"], "to": ["Menu"], "event": "MENU"},
        {"from": ["Menu"], "to": ["H_star"], "event": "BACK"},  # Deep history

        # Playing state transitions
        {"from": ["Normal"], "to": ["FastForward"], "event": "FF"},
        {"from": ["FastForward"], "to": ["Normal"], "event": "PLAY"},
        {"from": ["Normal"], "to": ["Rewind"], "event": "RW"},
        {"from": ["Rewind"], "to": ["Normal"], "event": "PLAY"},

        # Play/Pause
        {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
        {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},

        # Menu transitions
        {"from": ["MainMenu"], "to": ["Playlist"], "event": "LIST"},
        {"from": ["Playlist"], "to": ["MainMenu"], "event": "MAIN"},
    ]
}


# Mixed SC with both shallow and deep history
MIXED_HISTORY_SC = {
    "name": "MixedHistory",
    "root_state": {
        "label": "__root__",
        "type": 2,
        "children": [
            {"label": "Idle", "type": 1, "is_initial": True},
            {
                "label": "Mode1",
                "type": 2,
                "children": [
                    {"label": "M1_A", "type": 1, "is_initial": True},
                    {"label": "M1_B", "type": 1},
                    {"label": "M1_C", "type": 1},
                    {
                        "label": "H1",
                        "type": 1,
                        "is_history": True,
                        "history_type": "shallow",
                    },
                ]
            },
            {
                "label": "Mode2",
                "type": 2,
                "children": [
                    {
                        "label": "M2_Outer",
                        "type": 2,
                        "is_initial": True,
                        "children": [
                            {"label": "M2_Inner1", "type": 1, "is_initial": True},
                            {"label": "M2_Inner2", "type": 1},
                        ]
                    },
                    {"label": "M2_Alt", "type": 1},
                    {
                        "label": "H2_star",
                        "type": 1,
                        "is_history": True,
                        "history_type": "deep",
                    },
                ]
            },
            {
                "label": "Pause",
                "type": 2,
                "children": [
                    {"label": "PauseMain", "type": 1, "is_initial": True},
                ]
            },
        ]
    },
    "transitions": [
        # Entry
        {"from": ["Idle"], "to": ["Mode1"], "event": "START1"},
        {"from": ["Idle"], "to": ["Mode2"], "event": "START2"},

        # Mode1 transitions
        {"from": ["M1_A"], "to": ["M1_B"], "event": "NEXT"},
        {"from": ["M1_B"], "to": ["M1_C"], "event": "NEXT"},
        {"from": ["M1_C"], "to": ["M1_A"], "event": "NEXT"},
        {"from": ["Mode1"], "to": ["Pause"], "event": "PAUSE"},
        {"from": ["Pause"], "to": ["H1"], "event": "RESUME1"},  # Shallow

        # Mode2 transitions
        {"from": ["M2_Inner1"], "to": ["M2_Inner2"], "event": "TOGGLE"},
        {"from": ["M2_Inner2"], "to": ["M2_Inner1"], "event": "TOGGLE"},
        {"from": ["M2_Outer"], "to": ["M2_Alt"], "event": "ALT"},
        {"from": ["M2_Alt"], "to": ["M2_Outer"], "event": "ALT"},
        {"from": ["Mode2"], "to": ["Pause"], "event": "PAUSE"},
        {"from": ["Pause"], "to": ["H2_star"], "event": "RESUME2"},  # Deep
    ]
}


# =============================================================================
# Test Functions
# =============================================================================

def test_shallow_history_basic():
    """Test basic shallow history - remembers immediate child."""
    executor = HistoryExecutor(TEXT_EDITOR_SC)
    active = executor.initial_config()

    # Should start in Inactive
    assert "Inactive" in active, f"Should start Inactive, got {active}"

    # Open -> Active (enters Editing by default)
    active = executor.step(active, "OPEN")
    assert "Editing" in active, f"Should be Editing, got {active}"

    # Search -> Searching
    active = executor.step(active, "SEARCH")
    assert "Searching" in active, f"Should be Searching, got {active}"

    # Settings -> exits Active (saves history), enters Settings
    active = executor.step(active, "SETTINGS")
    assert "General" in active, f"Should be in General, got {active}"
    assert "Searching" not in active, f"Should have exited Searching, got {active}"

    # Verify history was saved
    saved = executor.get_shallow_history("Active")
    assert saved == "Searching", f"Shallow history should be Searching, got {saved}"

    # Back -> returns via H, should restore to Searching
    active = executor.step(active, "BACK")
    assert "Searching" in active, f"Should restore to Searching via history, got {active}"

    return True


def test_shallow_history_no_history():
    """Test shallow history when no history exists (default behavior)."""
    executor = HistoryExecutor(TEXT_EDITOR_SC)
    active = executor.initial_config()

    # Open -> Active (enters Editing)
    active = executor.step(active, "OPEN")
    assert "Editing" in active

    # Go directly to Settings without navigating within Active
    active = executor.step(active, "SETTINGS")
    assert "General" in active

    # History should have Editing (the initial/default)
    saved = executor.get_shallow_history("Active")
    assert saved == "Editing", f"Should have saved Editing, got {saved}"

    # Back should restore to Editing (the saved state)
    active = executor.step(active, "BACK")
    assert "Editing" in active, f"Should restore to Editing, got {active}"

    return True


def test_deep_history_nested():
    """Test deep history - remembers full nested configuration."""
    executor = HistoryExecutor(MEDIA_PLAYER_SC)
    active = executor.initial_config()

    # Power on -> On/Playing/Normal
    active = executor.step(active, "POWER")
    assert "Normal" in active, f"Should be Normal, got {active}"

    # FastForward -> Playing/FastForward
    active = executor.step(active, "FF")
    assert "FastForward" in active, f"Should be FastForward, got {active}"

    # Menu -> exits On (saves deep history), enters Menu
    active = executor.step(active, "MENU")
    assert "MainMenu" in active, f"Should be in MainMenu, got {active}"

    # Verify deep history was saved
    saved = executor.get_deep_history("On")
    assert saved is not None, "Deep history should be saved"
    assert "FastForward" in saved, f"Deep history should contain FastForward, got {saved}"

    # Back -> returns via H_star, should restore to Playing/FastForward
    active = executor.step(active, "BACK")
    assert "FastForward" in active, f"Should restore to FastForward, got {active}"

    return True


def test_deep_history_vs_shallow():
    """Test that deep history preserves more than shallow would."""
    executor = HistoryExecutor(MEDIA_PLAYER_SC)
    active = executor.initial_config()

    # Power on, then navigate to nested state
    active = executor.step(active, "POWER")
    active = executor.step(active, "RW")  # Playing/Rewind

    assert "Rewind" in active, f"Should be in Rewind, got {active}"

    # Menu -> save history
    active = executor.step(active, "MENU")

    # Check deep history preserves the nested state
    deep = executor.get_deep_history("On")
    assert "Rewind" in deep, f"Deep history should have Rewind, got {deep}"

    # Back -> should restore to Rewind specifically
    active = executor.step(active, "BACK")
    assert "Rewind" in active, f"Deep history should restore Rewind, got {active}"

    return True


def test_mixed_shallow_and_deep():
    """Test SC with both shallow and deep history."""
    executor = HistoryExecutor(MIXED_HISTORY_SC)
    active = executor.initial_config()

    # Test Mode1 with shallow history (H1)
    active = executor.step(active, "START1")
    assert "M1_A" in active

    active = executor.step(active, "NEXT")
    active = executor.step(active, "NEXT")
    assert "M1_C" in active, f"Should be M1_C, got {active}"

    # Pause and resume via shallow history
    active = executor.step(active, "PAUSE")
    assert "PauseMain" in active

    active = executor.step(active, "RESUME1")
    # Shallow history restores M1_C
    assert "M1_C" in active, f"Shallow should restore M1_C, got {active}"

    # Reset for Mode2 test
    executor.reset_history()
    active = executor.initial_config()

    # Test Mode2 with deep history (H2_star)
    active = executor.step(active, "START2")
    assert "M2_Inner1" in active, f"Should start in M2_Inner1, got {active}"

    active = executor.step(active, "TOGGLE")
    assert "M2_Inner2" in active, f"Should be M2_Inner2, got {active}"

    # Pause and resume via deep history
    active = executor.step(active, "PAUSE")
    assert "PauseMain" in active

    active = executor.step(active, "RESUME2")
    # Deep history restores full nested config including M2_Inner2
    assert "M2_Inner2" in active, f"Deep should restore M2_Inner2, got {active}"

    return True


def test_history_default_no_prior_visit():
    """Test that history falls back to default when composite was never visited."""
    # Create executor fresh
    executor = HistoryExecutor(TEXT_EDITOR_SC)

    # Manually clear any history
    executor.reset_history()

    # Verify no history exists
    assert executor.get_shallow_history("Active") is None

    # The history pseudostate's default should be the initial state
    h_state = executor.history_states.get("H")
    assert h_state is not None, "H should be registered"
    assert h_state.default_state == "Editing", f"Default should be Editing, got {h_state.default_state}"

    return True


def test_multiple_history_saves():
    """Test that history is updated on each exit."""
    executor = HistoryExecutor(TEXT_EDITOR_SC)
    active = executor.initial_config()

    # First visit: Open, go to Searching
    active = executor.step(active, "OPEN")
    active = executor.step(active, "SEARCH")
    active = executor.step(active, "SETTINGS")

    saved1 = executor.get_shallow_history("Active")
    assert saved1 == "Searching"

    # Return via history
    active = executor.step(active, "BACK")
    assert "Searching" in active

    # Navigate to Formatting
    active = executor.step(active, "CANCEL")  # Back to Editing
    active = executor.step(active, "FORMAT")  # To Formatting
    assert "Formatting" in active

    # Exit again
    active = executor.step(active, "SETTINGS")

    # History should now be Formatting
    saved2 = executor.get_shallow_history("Active")
    assert saved2 == "Formatting", f"History should update to Formatting, got {saved2}"

    return True


def test_deep_history_parallel_regions():
    """Test deep history with parallel regions."""
    # Create SC with parallel regions and deep history
    # Note: History pseudostate must be inside the composite it remembers
    parallel_history_sc = {
        "name": "ParallelHistory",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {
                    "label": "ActiveWrapper",
                    "type": 2,  # Normal wrapper for history
                    "children": [
                        {
                            "label": "Active",
                            "type": 3,  # PARALLEL
                            "is_initial": True,
                            "children": [
                                {
                                    "label": "RegionA",
                                    "type": 2,
                                    "children": [
                                        {"label": "A1", "type": 1, "is_initial": True},
                                        {"label": "A2", "type": 1},
                                    ]
                                },
                                {
                                    "label": "RegionB",
                                    "type": 2,
                                    "children": [
                                        {"label": "B1", "type": 1, "is_initial": True},
                                        {"label": "B2", "type": 1},
                                    ]
                                },
                            ]
                        },
                        {
                            "label": "H_deep",
                            "type": 1,
                            "is_history": True,
                            "history_type": "deep",
                        },
                    ]
                },
                {
                    "label": "Paused",
                    "type": 2,
                    "children": [
                        {"label": "PauseState", "type": 1, "is_initial": True},
                    ]
                },
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["ActiveWrapper"], "event": "START"},
            {"from": ["ActiveWrapper"], "to": ["Paused"], "event": "PAUSE"},
            {"from": ["Paused"], "to": ["H_deep"], "event": "RESUME"},
            {"from": ["A1"], "to": ["A2"], "event": "NEXT_A"},
            {"from": ["B1"], "to": ["B2"], "event": "NEXT_B"},
        ]
    }

    executor = HistoryExecutor(parallel_history_sc)
    active = executor.initial_config()

    # Start -> ActiveWrapper -> Active (parallel: A1 + B1)
    active = executor.step(active, "START")
    assert "A1" in active and "B1" in active, f"Should have A1, B1, got {active}"

    # Advance both regions
    active = executor.step(active, "NEXT_A")
    active = executor.step(active, "NEXT_B")
    assert "A2" in active and "B2" in active, f"Should have A2, B2, got {active}"

    # Pause (save deep history for ActiveWrapper)
    active = executor.step(active, "PAUSE")
    assert "PauseState" in active

    # Deep history should have both A2 and B2 (saved for ActiveWrapper)
    deep = executor.get_deep_history("ActiveWrapper")
    assert deep is not None, f"Deep history for ActiveWrapper should exist"
    assert "A2" in deep and "B2" in deep, f"Deep should have A2, B2, got {deep}"

    # Resume via deep history
    active = executor.step(active, "RESUME")
    assert "A2" in active and "B2" in active, f"Should restore A2, B2, got {active}"

    return True


# =============================================================================
# Benchmark
# =============================================================================

def benchmark():
    """Run all history state tests."""
    results = {
        "shallow_history_basic": test_shallow_history_basic(),
        "shallow_history_no_history": test_shallow_history_no_history(),
        "deep_history_nested": test_deep_history_nested(),
        "deep_history_vs_shallow": test_deep_history_vs_shallow(),
        "mixed_shallow_and_deep": test_mixed_shallow_and_deep(),
        "history_default_no_prior_visit": test_history_default_no_prior_visit(),
        "multiple_history_saves": test_multiple_history_saves(),
        "deep_history_parallel_regions": test_deep_history_parallel_regions(),
    }

    # Categorize results
    shallow_tests = [
        "shallow_history_basic",
        "shallow_history_no_history",
        "history_default_no_prior_visit",
        "multiple_history_saves",
    ]
    deep_tests = [
        "deep_history_nested",
        "deep_history_vs_shallow",
        "deep_history_parallel_regions",
    ]
    mixed_tests = ["mixed_shallow_and_deep"]

    shallow_passed = sum(1 for t in shallow_tests if results.get(t))
    deep_passed = sum(1 for t in deep_tests if results.get(t))
    default_passed = sum(1 for t in ["history_default_no_prior_visit"] if results.get(t))

    total_passed = sum(1 for v in results.values() if v)
    total = len(results)

    shallow_pct = (shallow_passed / len(shallow_tests)) * 100
    deep_pct = (deep_passed / len(deep_tests)) * 100

    print(f"History States Test Results:")
    print(f"  Total: {total_passed}/{total}")
    print(f"  Shallow history: {shallow_passed}/{len(shallow_tests)} ({shallow_pct:.0f}%)")
    print(f"  Deep history: {deep_passed}/{len(deep_tests)} ({deep_pct:.0f}%)")
    print(f"  Default behavior: {default_passed}/1")
    print()
    for name, result in results.items():
        status = "pass" if result else "FAIL"
        print(f"  [{status}] {name}")

    return {
        "results": results,
        "shallow_accuracy": shallow_pct,
        "deep_accuracy": deep_pct,
        "total_passed": total_passed,
        "total": total,
    }


if __name__ == "__main__":
    benchmark()
