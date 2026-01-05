"""
History Scenarios: Test Cases for Deep vs Shallow History Learning

This module creates scenarios that REQUIRE specific history types:
- Scenarios where DEEP history is optimal (nested context matters)
- Scenarios where SHALLOW history is optimal (only direct child matters)
- Scenarios where NO history is needed

The evolution must discover which history type works for each scenario.
NO HARDCODING - the patterns are learned through fitness evaluation.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict
from abc import ABC, abstractmethod
from enum import IntEnum


# =============================================================================
# Scenario Base Class
# =============================================================================

@dataclass
class HistoryScenario(ABC):
    """
    Base class for history test scenarios.

    Each scenario defines:
    - A statechart structure (states, hierarchy, transitions)
    - A set of (event_sequence, expected_final_state) test cases
    - An optimal history configuration for validation
    """
    name: str
    description: str

    @abstractmethod
    def create_structure(self) -> Dict:
        """
        Return structure definition:
        {
            'n_states': int,
            'parent': List[int],
            'state_type': List[int],
            'transitions': List[(src, tgt, event)],
            'initial_state': int
        }
        """
        pass

    @abstractmethod
    def generate_cases(self, n_cases: int = 100) -> List[Tuple[List[int], int]]:
        """
        Generate test cases: (events, expected_state)

        Cases should be designed so that:
        - Correct history type leads to expected_state
        - Wrong history type leads to different state
        """
        pass

    @abstractmethod
    def optimal_history_type(self) -> List[int]:
        """
        Return the optimal history_type configuration.
        Used for validation, NOT for evolution (which learns it).
        """
        pass


# =============================================================================
# Nested Navigation Scenario
# =============================================================================

class NestedNavigationScenario(HistoryScenario):
    """
    Scenario: Multi-level menu navigation with interruption.

    Structure:
        Root
        └── Menu (composite)
            ├── Settings (composite)
            │   ├── General
            │   ├── Display
            │   └── Audio
            └── Popup (interrupt)

    Test cases:
    1. Navigate to Settings > Audio
    2. Popup interrupts
    3. Return via history

    DEEP HISTORY: Returns to Settings.Audio (correct)
    SHALLOW HISTORY: Returns to Settings (then enters Settings.General)
    NO HISTORY: Enters Settings.General

    This scenario REQUIRES DEEP history for correct behavior.
    """

    def __init__(self):
        super().__init__(
            name="NestedNavigation",
            description="Multi-level menu with interruption requiring deep history"
        )

    def create_structure(self) -> Dict:
        # States:
        # 0: Root
        # 1: Menu (composite)
        # 2: Settings (composite)
        # 3: Popup (leaf - interrupt target)
        # 4: General (leaf)
        # 5: Display (leaf)
        # 6: Audio (leaf)
        return {
            'n_states': 7,
            'parent': [-1, 0, 1, 1, 2, 2, 2],
            'state_type': [1, 1, 1, 0, 0, 0, 0],  # OR, OR, OR, BASIC...
            'transitions': [
                # Within Settings
                (4, 5, 0),  # General -> Display
                (5, 6, 0),  # Display -> Audio
                (6, 4, 1),  # Audio -> General
                (4, 6, 2),  # General -> Audio (shortcut)

                # Popup interrupt from any Settings state
                (4, 3, 3),  # General -> Popup
                (5, 3, 3),  # Display -> Popup
                (6, 3, 3),  # Audio -> Popup

                # Return from Popup -> Settings COMPOSITE (uses history!)
                # This goes to Settings(2), which then resolves via history
                (3, 2, 4),  # Popup -> Settings (history resolves to child)
            ],
            'initial_state': 4  # Start at General
        }

    def generate_cases(self, n_cases: int = 100) -> List[Tuple[List[int], int]]:
        cases = []

        for _ in range(n_cases):
            # Pattern: Navigate deep, get interrupted, return
            # Start at General (4)

            # 1. Navigate to specific sub-state
            target_depths = [
                ([0, 0], 6),      # General -> Display -> Audio
                ([2], 6),         # General -> Audio (shortcut)
                ([0], 5),         # General -> Display
            ]
            nav_events, mid_state = random.choice(target_depths)

            # 2. Interrupt (event 3)
            interrupt = [3]

            # 3. Return (event 4)
            return_event = [4]

            # Full sequence
            events = nav_events + interrupt + return_event

            # Expected: with DEEP history, should return to mid_state
            # Without deep history, returns to General (4)
            expected = mid_state

            cases.append((events, expected))

        return cases

    def optimal_history_type(self) -> List[int]:
        """
        Optimal: Menu uses DEEP history, Settings uses DEEP history.
        """
        from .history_evolver import HistoryType
        return [
            HistoryType.NONE,   # Root
            HistoryType.DEEP,   # Menu - needs DEEP to remember Settings.Audio
            HistoryType.DEEP,   # Settings - needs DEEP to remember Audio
            HistoryType.NONE,   # Popup (leaf)
            HistoryType.NONE,   # General (leaf)
            HistoryType.NONE,   # Display (leaf)
            HistoryType.NONE,   # Audio (leaf)
        ]


# =============================================================================
# Text Editor Scenario
# =============================================================================

class TextEditorScenario(HistoryScenario):
    """
    Scenario: Text editor with mode memory.

    Structure:
        Root
        ├── Editing (composite)
        │   ├── Normal
        │   ├── Insert
        │   └── Visual
        └── Settings

    Test cases:
    1. Switch modes within Editing
    2. Go to Settings
    3. Return

    SHALLOW HISTORY: Returns to last mode (e.g., Insert)
    NO HISTORY: Returns to Normal

    This scenario requires SHALLOW history (mode matters, not substates).
    """

    def __init__(self):
        super().__init__(
            name="TextEditor",
            description="Editor modes requiring shallow history"
        )

    def create_structure(self) -> Dict:
        # States:
        # 0: Root
        # 1: Editing (composite)
        # 2: Settings (leaf)
        # 3: Normal (leaf)
        # 4: Insert (leaf)
        # 5: Visual (leaf)
        return {
            'n_states': 6,
            'parent': [-1, 0, 0, 1, 1, 1],
            'state_type': [1, 1, 0, 0, 0, 0],  # OR, OR, BASIC...
            'transitions': [
                # Mode switching
                (3, 4, 0),  # Normal -> Insert (press 'i')
                (4, 3, 1),  # Insert -> Normal (press ESC)
                (3, 5, 2),  # Normal -> Visual (press 'v')
                (5, 3, 1),  # Visual -> Normal (press ESC)

                # Settings access
                (3, 2, 3),  # Normal -> Settings
                (4, 2, 3),  # Insert -> Settings
                (5, 2, 3),  # Visual -> Settings

                # Return from Settings -> Editing COMPOSITE (uses history!)
                (2, 1, 4),  # Settings -> Editing (history resolves to mode)
            ],
            'initial_state': 3  # Start at Normal
        }

    def generate_cases(self, n_cases: int = 100) -> List[Tuple[List[int], int]]:
        cases = []

        for _ in range(n_cases):
            # Pattern: Enter mode, go to settings, return
            # Start at Normal (3)

            mode_sequences = [
                ([0], 4),      # Normal -> Insert
                ([2], 5),      # Normal -> Visual
                ([0, 1, 0], 4),  # Normal -> Insert -> Normal -> Insert
            ]
            mode_events, mode = random.choice(mode_sequences)

            # Go to settings
            settings = [3]

            # Return
            return_event = [4]

            events = mode_events + settings + return_event

            # Expected: with SHALLOW history, returns to last mode
            expected = mode

            cases.append((events, expected))

        return cases

    def optimal_history_type(self) -> List[int]:
        """
        Optimal: Editing uses SHALLOW history.
        """
        from .history_evolver import HistoryType
        return [
            HistoryType.NONE,     # Root
            HistoryType.SHALLOW,  # Editing - only need to remember mode
            HistoryType.NONE,     # Settings (leaf)
            HistoryType.NONE,     # Normal (leaf)
            HistoryType.NONE,     # Insert (leaf)
            HistoryType.NONE,     # Visual (leaf)
        ]


# =============================================================================
# Game Pause Scenario
# =============================================================================

class GamePauseScenario(HistoryScenario):
    """
    Scenario: Game with pause menu, no history needed.

    Structure:
        Root
        ├── Playing (composite)
        │   ├── Exploring
        │   └── Combat
        └── Paused

    Test cases:
    1. Be in some game state
    2. Pause
    3. Resume - ALWAYS returns to Exploring (reset behavior)

    NO HISTORY: Always starts fresh (intended design)
    SHALLOW/DEEP: Would incorrectly remember state

    This scenario requires NO history - fresh start is the design.
    """

    def __init__(self):
        super().__init__(
            name="GamePause",
            description="Game pause requiring NO history (fresh start)"
        )

    def create_structure(self) -> Dict:
        # States:
        # 0: Root
        # 1: Playing (composite)
        # 2: Paused (leaf)
        # 3: Exploring (leaf) - initial
        # 4: Combat (leaf)
        return {
            'n_states': 5,
            'parent': [-1, 0, 0, 1, 1],
            'state_type': [1, 1, 0, 0, 0],
            'transitions': [
                # Game flow
                (3, 4, 0),  # Exploring -> Combat (encounter enemy)
                (4, 3, 1),  # Combat -> Exploring (flee/win)

                # Pause from any state
                (3, 2, 2),  # Exploring -> Paused
                (4, 2, 2),  # Combat -> Paused

                # Resume - always to Exploring (checkpoint behavior)
                (2, 3, 3),  # Paused -> Playing (always starts at Exploring)
            ],
            'initial_state': 3  # Start at Exploring
        }

    def generate_cases(self, n_cases: int = 100) -> List[Tuple[List[int], int]]:
        cases = []

        for _ in range(n_cases):
            # Pattern: Play, pause, resume -> always Exploring
            pre_pause = random.choice([
                [],           # Pause immediately
                [0],          # Go to Combat, then pause
                [0, 1],       # Combat and back
                [0, 1, 0],    # Combat, back, combat
            ])

            # Pause (event 2)
            pause = [2]

            # Resume (event 3)
            resume = [3]

            events = pre_pause + pause + resume

            # Expected: ALWAYS Exploring (3) - this is the design intent
            expected = 3

            cases.append((events, expected))

        return cases

    def optimal_history_type(self) -> List[int]:
        """
        Optimal: NO history anywhere.
        """
        from .history_evolver import HistoryType
        return [
            HistoryType.NONE,  # Root
            HistoryType.NONE,  # Playing - no history (checkpoint design)
            HistoryType.NONE,  # Paused (leaf)
            HistoryType.NONE,  # Exploring (leaf)
            HistoryType.NONE,  # Combat (leaf)
        ]


# =============================================================================
# Scenario Registry
# =============================================================================

SCENARIOS = {
    'nested_navigation': NestedNavigationScenario,
    'text_editor': TextEditorScenario,
    'game_pause': GamePauseScenario,
}


def get_scenario(name: str) -> HistoryScenario:
    """Get scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()


# =============================================================================
# Unified Dataset Generator
# =============================================================================

def create_scenario_dataset(
    scenario: HistoryScenario,
    n_cases: int = 100
) -> Tuple[Dict, List[Tuple[List[int], int]]]:
    """
    Create complete dataset from a scenario.

    Returns (structure, test_cases).
    """
    structure = scenario.create_structure()
    cases = scenario.generate_cases(n_cases)
    return structure, cases


def create_mixed_scenario_generator(
    scenario_weights: Dict[str, float] = None,
    n_cases_per_scenario: int = 50
) -> Callable:
    """
    Create generator that mixes multiple scenarios.

    Returns a callable that generates mixed test cases.
    """
    weights = scenario_weights or {
        'nested_navigation': 1.0,
        'text_editor': 1.0,
        'game_pause': 1.0,
    }

    def generator():
        all_cases = []

        for scenario_name, weight in weights.items():
            if weight <= 0:
                continue

            scenario = get_scenario(scenario_name)
            n_cases = int(n_cases_per_scenario * weight)
            _, cases = create_scenario_dataset(scenario, n_cases)
            all_cases.extend(cases)

        random.shuffle(all_cases)
        return all_cases

    return generator


# =============================================================================
# Testing
# =============================================================================

def test_scenarios():
    """Test all scenarios."""
    print("=" * 60)
    print("HISTORY SCENARIOS TEST")
    print("=" * 60)

    for name, scenario_class in SCENARIOS.items():
        print(f"\n--- {name} ---")
        scenario = scenario_class()
        print(f"Description: {scenario.description}")

        structure = scenario.create_structure()
        print(f"States: {structure['n_states']}")
        print(f"Transitions: {len(structure['transitions'])}")

        cases = scenario.generate_cases(10)
        print(f"Sample cases:")
        for events, expected in cases[:3]:
            print(f"  {events} -> {expected}")

        optimal = scenario.optimal_history_type()
        print(f"Optimal history: {optimal}")

    print("\n" + "=" * 60)
    print("All scenarios validated!")


if __name__ == "__main__":
    test_scenarios()
