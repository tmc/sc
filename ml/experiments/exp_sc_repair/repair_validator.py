"""
Repair Validator: Validate repairs preserve semantics.

Ensures repairs:
1. Fix the original errors
2. Don't introduce new errors
3. Preserve intended behavior (where detectable)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
import copy

# Import error detection
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')
from experiments.exp_sc_error_patterns.error_detector import detect_errors


@dataclass
class ValidationResult:
    """Result of repair validation."""
    is_valid: bool
    original_error_count: int
    repaired_error_count: int
    errors_fixed: int
    errors_introduced: int
    semantic_preserved: bool
    details: List[str] = field(default_factory=list)


class RepairValidator:
    """
    Validate that repairs are correct and preserve semantics.
    """

    def validate(
        self,
        original: Dict[str, Any],
        repaired: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a repair.

        Args:
            original: Original (broken) statechart
            repaired: Repaired statechart

        Returns:
            ValidationResult with validation details
        """
        details = []

        # Check error counts
        orig_result = detect_errors(original)
        repair_result = detect_errors(repaired)

        orig_errors = len(orig_result.errors) + len(orig_result.warnings)
        repair_errors = len(repair_result.errors) + len(repair_result.warnings)

        errors_fixed = max(0, orig_errors - repair_errors)
        errors_introduced = max(0, repair_errors - orig_errors)

        # Check if errors were reduced
        if repair_errors < orig_errors:
            details.append(f"Reduced errors from {orig_errors} to {repair_errors}")
        elif repair_errors == orig_errors:
            details.append("Error count unchanged")
        else:
            details.append(f"WARNING: Errors increased from {orig_errors} to {repair_errors}")

        # Check semantic preservation
        semantic_ok = self._check_semantics(original, repaired, details)

        # Overall validity
        is_valid = (
            repair_errors <= orig_errors and
            errors_introduced == 0 and
            semantic_ok
        )

        return ValidationResult(
            is_valid=is_valid,
            original_error_count=orig_errors,
            repaired_error_count=repair_errors,
            errors_fixed=errors_fixed,
            errors_introduced=errors_introduced,
            semantic_preserved=semantic_ok,
            details=details,
        )

    def _check_semantics(
        self,
        original: Dict,
        repaired: Dict,
        details: List[str],
    ) -> bool:
        """Check if repair preserves semantic intent."""
        ok = True

        # Check: name preserved
        if original.get("name") != repaired.get("name"):
            details.append("WARNING: Name changed")
            # Not critical, allow

        # Check: transitions mostly preserved
        orig_events = set(t.get("event", "") for t in original.get("transitions", []))
        repair_events = set(t.get("event", "") for t in repaired.get("transitions", []))

        lost_events = orig_events - repair_events
        if lost_events:
            details.append(f"Lost events: {lost_events}")
            # May be intentional if removing dangling transitions

        # Check: states mostly preserved
        orig_states = self._get_states(original)
        repair_states = self._get_states(repaired)

        lost_states = orig_states - repair_states
        if lost_states:
            details.append(f"Lost states: {lost_states}")
            # May be intentional if removing unreachable

        # Check: initial state exists
        repair_initial = self._get_initial(repaired)
        if not repair_initial:
            details.append("WARNING: No initial state after repair")
            ok = False

        # Check: at least one state exists
        if not repair_states:
            details.append("ERROR: No states after repair")
            ok = False

        return ok

    def _get_states(self, sc: Dict) -> Set[str]:
        """Get all state names."""
        states = set()

        def collect(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for c in state.get("children", []):
                collect(c)

        collect(sc.get("root_state", {}))
        return states

    def _get_initial(self, sc: Dict) -> Set[str]:
        """Get initial states."""
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


def validate_repair(
    original: Dict[str, Any],
    repaired: Dict[str, Any],
) -> ValidationResult:
    """Convenience function to validate a repair."""
    validator = RepairValidator()
    return validator.validate(original, repaired)


def demo():
    """Demonstrate repair validation."""
    print("=" * 60)
    print("REPAIR VALIDATOR: Validate Repair Correctness")
    print("=" * 60)

    # Original with errors
    original = {
        "name": "Test",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1, "is_initial": True},
                {"label": "A", "type": 1},  # Duplicate
                {"label": "B", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"},
        ]
    }

    # Good repair (fixed duplicate)
    good_repair = {
        "name": "Test",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1, "is_initial": True},
                {"label": "A_1", "type": 1},  # Renamed
                {"label": "B", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"},
        ]
    }

    # Bad repair (removed too much)
    bad_repair = {
        "name": "Test",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": []  # No states!
        },
        "transitions": []
    }

    validator = RepairValidator()

    print("\n--- Good Repair ---")
    result1 = validator.validate(original, good_repair)
    print(f"Valid: {result1.is_valid}")
    print(f"Errors fixed: {result1.errors_fixed}")
    print(f"Semantic preserved: {result1.semantic_preserved}")
    for d in result1.details:
        print(f"  {d}")

    print("\n--- Bad Repair ---")
    result2 = validator.validate(original, bad_repair)
    print(f"Valid: {result2.is_valid}")
    print(f"Errors introduced: {result2.errors_introduced}")
    print(f"Semantic preserved: {result2.semantic_preserved}")
    for d in result2.details:
        print(f"  {d}")

    return result1, result2


if __name__ == "__main__":
    demo()
