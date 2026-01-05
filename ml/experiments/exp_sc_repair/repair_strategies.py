"""
Repair Strategies: Specialized repair logic per error type.

Each strategy handles a specific error category with
targeted repair logic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from abc import ABC, abstractmethod
import copy


@dataclass
class RepairAction:
    """A single repair action."""
    action_type: str
    target: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class RepairStrategy(ABC):
    """Base class for repair strategies."""

    @abstractmethod
    def can_repair(self, error_category: str) -> bool:
        """Check if this strategy handles the error category."""
        pass

    @abstractmethod
    def generate_repairs(
        self,
        statechart: Dict[str, Any],
        error: Any,
    ) -> List[RepairAction]:
        """Generate possible repairs for the error."""
        pass

    @abstractmethod
    def apply_repair(
        self,
        statechart: Dict[str, Any],
        action: RepairAction,
    ) -> Dict[str, Any]:
        """Apply a repair action to the statechart."""
        pass


class DuplicateLabelStrategy(RepairStrategy):
    """Strategy for handling duplicate state labels."""

    def can_repair(self, error_category: str) -> bool:
        return error_category == "duplicate_element"

    def generate_repairs(
        self,
        statechart: Dict[str, Any],
        error: Any,
    ) -> List[RepairAction]:
        """Generate repairs for duplicate labels."""
        repairs = []
        state_name = getattr(error, 'element_name', '')

        if not state_name:
            return repairs

        # Option 1: Rename duplicates with suffix
        repairs.append(RepairAction(
            action_type="rename_with_suffix",
            target=state_name,
            details={"suffix_pattern": "_{n}"},
            confidence=0.9,
        ))

        # Option 2: Merge duplicates (keep first, update refs)
        repairs.append(RepairAction(
            action_type="merge_duplicates",
            target=state_name,
            details={"keep": "first"},
            confidence=0.7,
        ))

        return repairs

    def apply_repair(
        self,
        statechart: Dict[str, Any],
        action: RepairAction,
    ) -> Dict[str, Any]:
        """Apply duplicate label repair."""
        sc = copy.deepcopy(statechart)
        state_name = action.target

        if action.action_type == "rename_with_suffix":
            counter = [0]
            first_found = [False]

            def rename(state):
                if state.get("label") == state_name:
                    if first_found[0]:
                        counter[0] += 1
                        state["label"] = f"{state_name}_{counter[0]}"
                    else:
                        first_found[0] = True
                for child in state.get("children", []):
                    rename(child)

            rename(sc.get("root_state", {}))

        elif action.action_type == "merge_duplicates":
            # Keep first occurrence, remove others
            found = [False]

            def remove_dups(state):
                new_children = []
                for child in state.get("children", []):
                    if child.get("label") == state_name:
                        if not found[0]:
                            found[0] = True
                            new_children.append(child)
                        # Skip duplicates
                    else:
                        new_children.append(child)
                    remove_dups(child)
                state["children"] = new_children

            remove_dups(sc.get("root_state", {}))

        return sc


class UnreachableStateStrategy(RepairStrategy):
    """Strategy for handling unreachable states."""

    def can_repair(self, error_category: str) -> bool:
        return error_category == "unreachable_state"

    def generate_repairs(
        self,
        statechart: Dict[str, Any],
        error: Any,
    ) -> List[RepairAction]:
        """Generate repairs for unreachable states."""
        repairs = []
        state_name = getattr(error, 'unreachable_state', '')

        if not state_name:
            return repairs

        # Get initial states for connection
        initial = self._get_initial_states(statechart)

        # Option 1: Add transition from initial state
        if initial:
            repairs.append(RepairAction(
                action_type="add_transition",
                target=state_name,
                details={
                    "from": list(initial)[0],
                    "event": f"GO_TO_{state_name.upper()}",
                },
                confidence=0.8,
            ))

        # Option 2: Remove the unreachable state
        repairs.append(RepairAction(
            action_type="remove_state",
            target=state_name,
            details={},
            confidence=0.6,
        ))

        # Option 3: Make it initial (if no initial exists)
        if not initial:
            repairs.append(RepairAction(
                action_type="make_initial",
                target=state_name,
                details={},
                confidence=0.7,
            ))

        return repairs

    def apply_repair(
        self,
        statechart: Dict[str, Any],
        action: RepairAction,
    ) -> Dict[str, Any]:
        """Apply unreachable state repair."""
        sc = copy.deepcopy(statechart)
        state_name = action.target

        if action.action_type == "add_transition":
            sc.setdefault("transitions", []).append({
                "from": [action.details["from"]],
                "to": [state_name],
                "event": action.details.get("event", ""),
            })

        elif action.action_type == "remove_state":
            def remove(state):
                state["children"] = [
                    c for c in state.get("children", [])
                    if c.get("label") != state_name
                ]
                for c in state.get("children", []):
                    remove(c)

            remove(sc.get("root_state", {}))

            # Remove transitions involving this state
            sc["transitions"] = [
                t for t in sc.get("transitions", [])
                if state_name not in t.get("from", []) and
                   state_name not in t.get("to", [])
            ]

        elif action.action_type == "make_initial":
            def set_initial(state):
                if state.get("label") == state_name:
                    state["is_initial"] = True
                    return True
                for c in state.get("children", []):
                    if set_initial(c):
                        return True
                return False

            set_initial(sc.get("root_state", {}))

        return sc

    def _get_initial_states(self, sc: Dict) -> Set[str]:
        initial = set()

        def collect(state):
            if state.get("is_initial"):
                label = state.get("label", "")
                if label and not label.startswith("__"):
                    initial.add(label)
            for c in state.get("children", []):
                collect(c)

        collect(sc.get("root_state", {}))
        return initial


class MissingInitialStrategy(RepairStrategy):
    """Strategy for handling missing initial states."""

    def can_repair(self, error_category: str) -> bool:
        return error_category == "hierarchy_violation"

    def generate_repairs(
        self,
        statechart: Dict[str, Any],
        error: Any,
    ) -> List[RepairAction]:
        """Generate repairs for missing initial state."""
        repairs = []

        # Check if this is specifically about initial states
        if hasattr(error, 'subtype'):
            subtype_str = str(getattr(error.subtype, 'value', ''))
            if 'initial' not in subtype_str:
                return repairs

        state_label = getattr(error, 'state_label', '')

        # Option 1: Set first child as initial
        repairs.append(RepairAction(
            action_type="set_first_initial",
            target=state_label,
            details={},
            confidence=0.9,
        ))

        # Option 2: Set specific child as initial (would need more info)
        repairs.append(RepairAction(
            action_type="clear_multiple_initial",
            target=state_label,
            details={"keep": "first"},
            confidence=0.8,
        ))

        return repairs

    def apply_repair(
        self,
        statechart: Dict[str, Any],
        action: RepairAction,
    ) -> Dict[str, Any]:
        """Apply missing initial repair."""
        sc = copy.deepcopy(statechart)
        parent = action.target

        def find_and_fix(state):
            label = state.get("label", "")
            if label == parent or (not parent and label.startswith("__")):
                children = state.get("children", [])

                if action.action_type == "set_first_initial":
                    # Clear all, set first
                    for c in children:
                        c["is_initial"] = False
                    if children:
                        children[0]["is_initial"] = True

                elif action.action_type == "clear_multiple_initial":
                    # Keep only first initial
                    found_initial = False
                    for c in children:
                        if c.get("is_initial"):
                            if found_initial:
                                c["is_initial"] = False
                            else:
                                found_initial = True

                return True

            for c in state.get("children", []):
                if find_and_fix(c):
                    return True
            return False

        find_and_fix(sc.get("root_state", {}))
        return sc


class InvalidHierarchyStrategy(RepairStrategy):
    """Strategy for handling invalid hierarchy structures."""

    def can_repair(self, error_category: str) -> bool:
        return error_category == "hierarchy_violation"

    def generate_repairs(
        self,
        statechart: Dict[str, Any],
        error: Any,
    ) -> List[RepairAction]:
        """Generate repairs for hierarchy violations."""
        repairs = []
        state_label = getattr(error, 'state_label', '')
        violation_type = getattr(error, 'violation_type', '')

        if 'parallel' in violation_type.lower():
            repairs.append(RepairAction(
                action_type="convert_to_normal",
                target=state_label,
                details={"new_type": 2},  # NORMAL
                confidence=0.7,
            ))

        if 'nesting' in violation_type.lower():
            repairs.append(RepairAction(
                action_type="flatten_children",
                target=state_label,
                details={},
                confidence=0.6,
            ))

        if 'empty' in violation_type.lower():
            repairs.append(RepairAction(
                action_type="add_placeholder_child",
                target=state_label,
                details={"child_name": f"{state_label}_default"},
                confidence=0.8,
            ))

        return repairs

    def apply_repair(
        self,
        statechart: Dict[str, Any],
        action: RepairAction,
    ) -> Dict[str, Any]:
        """Apply hierarchy repair."""
        sc = copy.deepcopy(statechart)
        state_label = action.target

        def find_and_fix(state):
            if state.get("label") == state_label:
                if action.action_type == "convert_to_normal":
                    state["type"] = action.details.get("new_type", 2)

                elif action.action_type == "flatten_children":
                    # Convert composite to basic if problematic
                    state["type"] = 1
                    state["children"] = []

                elif action.action_type == "add_placeholder_child":
                    child_name = action.details.get("child_name", "default")
                    state.setdefault("children", []).append({
                        "label": child_name,
                        "type": 1,
                        "is_initial": True,
                    })

                return True

            for c in state.get("children", []):
                if find_and_fix(c):
                    return True
            return False

        find_and_fix(sc.get("root_state", {}))
        return sc


def get_all_strategies() -> List[RepairStrategy]:
    """Get all available repair strategies."""
    return [
        DuplicateLabelStrategy(),
        UnreachableStateStrategy(),
        MissingInitialStrategy(),
        InvalidHierarchyStrategy(),
    ]


def demo():
    """Demonstrate repair strategies."""
    print("=" * 60)
    print("REPAIR STRATEGIES: Per-Error Type Repairs")
    print("=" * 60)

    strategies = get_all_strategies()
    print(f"\nAvailable strategies: {len(strategies)}")

    for strategy in strategies:
        print(f"  - {strategy.__class__.__name__}")

    # Test duplicate label strategy
    print("\n--- Duplicate Label Example ---")
    dup_strategy = DuplicateLabelStrategy()

    test_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1},
                {"label": "A", "type": 1},  # Duplicate
                {"label": "B", "type": 1},
            ]
        }
    }

    class MockError:
        element_name = "A"

    repairs = dup_strategy.generate_repairs(test_sc, MockError())
    print(f"Generated repairs: {len(repairs)}")

    if repairs:
        fixed = dup_strategy.apply_repair(test_sc, repairs[0])
        labels = [c.get("label") for c in fixed["root_state"]["children"]]
        print(f"After repair: {labels}")

    return strategies


if __name__ == "__main__":
    demo()
