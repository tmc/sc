"""
Benchmark for history states experiment.

Runs comprehensive tests on shallow and deep history state semantics.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_history_states')

from history_executor import HistoryExecutor, HistoryType


def run_benchmark():
    """Run the full history states benchmark."""
    from . import benchmark as run_tests
    return run_tests()


def demo():
    """Interactive demo of history state behavior."""
    from . import TEXT_EDITOR_SC, MEDIA_PLAYER_SC

    print("=" * 60)
    print("HISTORY STATES DEMO")
    print("=" * 60)

    # Demo 1: Shallow history
    print("\n--- Demo 1: Shallow History (Text Editor) ---")
    executor = HistoryExecutor(TEXT_EDITOR_SC)
    active = executor.initial_config()
    print(f"Initial: {active}")

    events = ["OPEN", "SEARCH", "SETTINGS", "BACK"]
    for event in events:
        active = executor.step(active, event)
        h = executor.get_shallow_history("Active")
        print(f"After {event}: {active} (history: {h})")

    # Demo 2: Deep history
    print("\n--- Demo 2: Deep History (Media Player) ---")
    executor = HistoryExecutor(MEDIA_PLAYER_SC)
    active = executor.initial_config()
    print(f"Initial: {active}")

    events = ["POWER", "FF", "MENU", "BACK"]
    for event in events:
        active = executor.step(active, event)
        h = executor.get_deep_history("On")
        print(f"After {event}: {active} (deep history: {h})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
    print()
    from . import benchmark
    benchmark()
