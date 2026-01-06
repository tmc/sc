#!/usr/bin/env python3
"""
Repair Strategies: Deterministic fixes for common statechart errors.

Provides rule-based repairs that can be applied before/instead of LLM repair.
These are fast, deterministic, and handle common cases reliably.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto

from .error_analyzer import (
    ErrorType, ValidationError, ValidationResult, ErrorAnalyzer
)


class RepairResult(Enum):
    """Outcome of a repair attempt."""
    FIXED = auto()      # Error was fixed
    PARTIAL = auto()    # Error partially addressed
    FAILED = auto()     # Could not fix
    NOT_APPLICABLE = auto()  # Strategy doesn't apply


@dataclass
class RepairAction:
    """A single repair action."""
    description: str
    result: RepairResult
    changes: List[str] = field(default_factory=list)


@dataclass
class RepairPlan:
    """Complete repair plan for a chart."""
    original_chart: Dict[str, Any]
    repaired_chart: Dict[str, Any]
    actions: List[RepairAction] = field(default_factory=list)
    errors_fixed: int = 0
    errors_remaining: int = 0

    @property
    def success_rate(self) -> float:
        total = self.errors_fixed + self.errors_remaining
        return self.errors_fixed / total if total > 0 else 1.0


class RepairStrategies:
    """
    Collection of deterministic repair strategies.

    Each strategy handles one error type with rule-based fixes.
    """

    def __init__(self):
        # Register strategies by error type
        self.strategies: Dict[ErrorType, Callable] = {
            ErrorType.DUPLICATE_STATE: self._fix_duplicate_state,
            ErrorType.INVALID_SOURCE: self._fix_invalid_source,
            ErrorType.INVALID_TARGET: self._fix_invalid_target,
            ErrorType.NO_INITIAL_STATE: self._fix_no_initial,
            ErrorType.MULTIPLE_INITIAL: self._fix_multiple_initial,
            ErrorType.EMPTY_COMPOUND: self._fix_empty_compound,
            ErrorType.BASIC_WITH_CHILDREN: self._fix_basic_with_children,
            ErrorType.PARALLEL_NEEDS_CHILDREN: self._fix_parallel_needs_children,
            ErrorType.MISSING_EVENT: self._fix_missing_event,
        }

    def apply_all(
        self,
        chart: Dict[str, Any],
        errors: List[ValidationError],
    ) -> RepairPlan:
        """
        Apply all applicable strategies to fix errors.

        Args:
            chart: Original chart (will not be modified)
            errors: List of validation errors

        Returns:
            RepairPlan with fixed chart and action log
        """
        repaired = copy.deepcopy(chart)
        plan = RepairPlan(
            original_chart=chart,
            repaired_chart=repaired,
        )

        for error in errors:
            strategy = self.strategies.get(error.error_type)
            if strategy:
                action = strategy(repaired, error)
                plan.actions.append(action)
                if action.result == RepairResult.FIXED:
                    plan.errors_fixed += 1
                else:
                    plan.errors_remaining += 1
            else:
                plan.actions.append(RepairAction(
                    description=f"No strategy for {error.error_type.name}",
                    result=RepairResult.NOT_APPLICABLE,
                ))
                plan.errors_remaining += 1

        plan.repaired_chart = repaired
        return plan

    def _collect_states(self, chart: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Collect all states with their paths."""
        states = {}

        def collect(state: Dict[str, Any], path: List[str]):
            label = state.get("label", "")
            if label:
                states[label] = {"state": state, "path": path.copy()}
            for i, child in enumerate(state.get("children", [])):
                collect(child, path + [label, f"children[{i}]"])

        if "root_state" in chart:
            collect(chart["root_state"], ["root_state"])

        return states

    def _find_state(self, chart: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        """Find a state by label."""
        states = self._collect_states(chart)
        if label in states:
            return states[label]["state"]
        return None

    # =========================================================================
    # REPAIR STRATEGIES
    # =========================================================================

    def _fix_duplicate_state(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix duplicate state by renaming second occurrence."""
        if not error.state_label:
            return RepairAction(
                description="Cannot fix: no state label",
                result=RepairResult.FAILED,
            )

        label = error.state_label
        counter = 2

        def rename_duplicates(state: Dict[str, Any], found_first: bool) -> bool:
            nonlocal counter
            if state.get("label") == label:
                if found_first:
                    # Rename this duplicate
                    new_label = f"{label}_{counter}"
                    while new_label in self._collect_states(chart):
                        counter += 1
                        new_label = f"{label}_{counter}"
                    state["label"] = new_label
                    return True
                return True  # Mark first found
            for child in state.get("children", []):
                found_first = rename_duplicates(child, found_first)
            return found_first

        if "root_state" in chart:
            rename_duplicates(chart["root_state"], False)

        return RepairAction(
            description=f"Renamed duplicate state '{label}'",
            result=RepairResult.FIXED,
            changes=[f"Renamed duplicate to '{label}_{counter}'"],
        )

    def _fix_invalid_source(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix invalid transition source by finding similar state."""
        if not error.state_label:
            return RepairAction(
                description="Cannot fix: no state label",
                result=RepairResult.FAILED,
            )

        invalid_label = error.state_label
        all_states = set(self._collect_states(chart).keys())

        # Find similar state
        replacement = self._find_similar(invalid_label, all_states)

        if not replacement:
            # Can't fix - need to add state or remove transition
            return RepairAction(
                description=f"Cannot fix: no similar state for '{invalid_label}'",
                result=RepairResult.FAILED,
            )

        # Fix transitions
        fixed = False
        for trans in chart.get("transitions", []):
            if invalid_label in trans.get("from", []):
                trans["from"] = [replacement if s == invalid_label else s for s in trans["from"]]
                fixed = True

        if fixed:
            return RepairAction(
                description=f"Fixed source '{invalid_label}' -> '{replacement}'",
                result=RepairResult.FIXED,
                changes=[f"Changed transition source from '{invalid_label}' to '{replacement}'"],
            )

        return RepairAction(
            description="No transitions to fix",
            result=RepairResult.NOT_APPLICABLE,
        )

    def _fix_invalid_target(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix invalid transition target by finding similar state."""
        if not error.state_label:
            return RepairAction(
                description="Cannot fix: no state label",
                result=RepairResult.FAILED,
            )

        invalid_label = error.state_label
        all_states = set(self._collect_states(chart).keys())

        replacement = self._find_similar(invalid_label, all_states)

        if not replacement:
            return RepairAction(
                description=f"Cannot fix: no similar state for '{invalid_label}'",
                result=RepairResult.FAILED,
            )

        fixed = False
        for trans in chart.get("transitions", []):
            if invalid_label in trans.get("to", []):
                trans["to"] = [replacement if s == invalid_label else s for s in trans["to"]]
                fixed = True

        if fixed:
            return RepairAction(
                description=f"Fixed target '{invalid_label}' -> '{replacement}'",
                result=RepairResult.FIXED,
                changes=[f"Changed transition target from '{invalid_label}' to '{replacement}'"],
            )

        return RepairAction(
            description="No transitions to fix",
            result=RepairResult.NOT_APPLICABLE,
        )

    def _fix_no_initial(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix missing initial state by marking first child."""
        state_label = error.state_label or "__root__"
        state = self._find_state(chart, state_label)

        if not state:
            return RepairAction(
                description=f"Cannot find state '{state_label}'",
                result=RepairResult.FAILED,
            )

        children = state.get("children", [])
        if not children:
            return RepairAction(
                description="No children to mark as initial",
                result=RepairResult.FAILED,
            )

        # Mark first child as initial
        children[0]["is_initial"] = True

        return RepairAction(
            description=f"Marked first child as initial in '{state_label}'",
            result=RepairResult.FIXED,
            changes=[f"Set is_initial=true on '{children[0].get('label', 'first child')}'"],
        )

    def _fix_multiple_initial(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix multiple initial states by keeping only first."""
        state_label = error.state_label or "__root__"
        state = self._find_state(chart, state_label)

        if not state:
            return RepairAction(
                description=f"Cannot find state '{state_label}'",
                result=RepairResult.FAILED,
            )

        children = state.get("children", [])
        found_initial = False
        cleared = []

        for child in children:
            if child.get("is_initial"):
                if found_initial:
                    child["is_initial"] = False
                    cleared.append(child.get("label", "?"))
                else:
                    found_initial = True

        if cleared:
            return RepairAction(
                description=f"Cleared extra initial flags in '{state_label}'",
                result=RepairResult.FIXED,
                changes=[f"Cleared is_initial from: {cleared}"],
            )

        return RepairAction(
            description="No multiple initials found",
            result=RepairResult.NOT_APPLICABLE,
        )

    def _fix_empty_compound(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix empty compound state by changing to basic type."""
        state_label = error.state_label
        state = self._find_state(chart, state_label)

        if not state:
            return RepairAction(
                description=f"Cannot find state '{state_label}'",
                result=RepairResult.FAILED,
            )

        # Change to basic type (type=1 or remove type)
        state["type"] = 1  # BASIC
        state.pop("children", None)

        return RepairAction(
            description=f"Changed '{state_label}' to basic type",
            result=RepairResult.FIXED,
            changes=[f"Set type=1 (BASIC) on '{state_label}'"],
        )

    def _fix_basic_with_children(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix basic state with children by changing to normal type."""
        state_label = error.state_label
        state = self._find_state(chart, state_label)

        if not state:
            return RepairAction(
                description=f"Cannot find state '{state_label}'",
                result=RepairResult.FAILED,
            )

        # Change to normal/compound type
        state["type"] = 2  # NORMAL

        return RepairAction(
            description=f"Changed '{state_label}' to normal type",
            result=RepairResult.FIXED,
            changes=[f"Set type=2 (NORMAL) on '{state_label}'"],
        )

    def _fix_parallel_needs_children(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix parallel state needing children by adding placeholder or changing type."""
        state_label = error.state_label
        state = self._find_state(chart, state_label)

        if not state:
            return RepairAction(
                description=f"Cannot find state '{state_label}'",
                result=RepairResult.FAILED,
            )

        children = state.get("children", [])
        if len(children) < 2:
            # Add placeholder children or change to normal type
            if len(children) == 0:
                # No children - change to normal type
                state["type"] = 2
                state["children"] = [
                    {"label": f"{state_label}_Child1", "is_initial": True},
                    {"label": f"{state_label}_Child2"},
                ]
                return RepairAction(
                    description=f"Added placeholder children to '{state_label}'",
                    result=RepairResult.FIXED,
                    changes=["Added 2 placeholder children for parallel state"],
                )
            else:
                # One child - add another
                state["children"].append({
                    "label": f"{state_label}_Child2",
                })
                return RepairAction(
                    description=f"Added second child to '{state_label}'",
                    result=RepairResult.FIXED,
                    changes=["Added placeholder child for parallel state"],
                )

        return RepairAction(
            description="Parallel state already has enough children",
            result=RepairResult.NOT_APPLICABLE,
        )

    def _fix_missing_event(
        self,
        chart: Dict[str, Any],
        error: ValidationError,
    ) -> RepairAction:
        """Fix missing event by adding placeholder event name."""
        transitions = chart.get("transitions", [])

        fixed = 0
        for trans in transitions:
            if not trans.get("event"):
                # Generate event name from source/target
                from_states = trans.get("from", ["?"])
                to_states = trans.get("to", ["?"])
                event_name = f"{from_states[0]}_TO_{to_states[0]}"
                trans["event"] = event_name
                fixed += 1

        if fixed:
            return RepairAction(
                description=f"Added events to {fixed} transition(s)",
                result=RepairResult.FIXED,
                changes=[f"Generated event names for {fixed} transitions"],
            )

        return RepairAction(
            description="No transitions missing events",
            result=RepairResult.NOT_APPLICABLE,
        )

    def _find_similar(self, target: str, candidates: Set[str]) -> Optional[str]:
        """Find similar state name (typo correction)."""
        if not target or not candidates:
            return None

        # Filter out root
        candidates = {c for c in candidates if c != "__root__"}

        # Exact match (case insensitive)
        for c in candidates:
            if c.lower() == target.lower():
                return c

        # Prefix match (at least 3 chars)
        if len(target) >= 3:
            for c in candidates:
                if c.lower().startswith(target.lower()[:3]):
                    return c
                if target.lower().startswith(c.lower()[:3]):
                    return c

        # Levenshtein-like: single character difference
        for c in candidates:
            if abs(len(c) - len(target)) <= 2:
                diff = sum(1 for a, b in zip(c.lower(), target.lower()) if a != b)
                if diff <= 2:
                    return c

        # Return first available if nothing matches
        if candidates:
            return sorted(candidates)[0]

        return None


def apply_strategies(
    chart: Dict[str, Any],
    errors: List[ValidationError],
) -> RepairPlan:
    """Convenience function to apply all strategies."""
    strategies = RepairStrategies()
    return strategies.apply_all(chart, errors)


def demo():
    """Demo repair strategies."""
    print("=" * 60)
    print("REPAIR STRATEGIES DEMO")
    print("=" * 60)

    analyzer = ErrorAnalyzer()
    strategies = RepairStrategies()

    # Test cases with known errors
    test_cases = [
        {
            "name": "Invalid source (typo)",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Idle", "is_initial": True},
                        {"label": "Running"},
                    ]
                },
                "transitions": [
                    {"from": ["Idel"], "to": ["Running"], "event": "START"}  # Typo
                ]
            }
        },
        {
            "name": "No initial state",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "A"},
                        {"label": "B"},
                    ]
                }
            }
        },
        {
            "name": "Multiple errors",
            "chart": {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Start"},
                        {"label": "Start"},  # Duplicate
                    ]
                },
                "transitions": [
                    {"from": ["Strat"], "to": ["End"], "event": "GO"}  # Both invalid
                ]
            }
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print("=" * 60)

        # Analyze errors
        result = analyzer.validate_chart(tc["chart"])
        print(f"Original: {len(result.errors)} error(s)")
        for err in result.errors:
            print(f"  - {err}")

        # Apply strategies
        plan = strategies.apply_all(tc["chart"], result.errors)
        print(f"\nRepair actions: {len(plan.actions)}")
        for action in plan.actions:
            print(f"  [{action.result.name}] {action.description}")

        # Validate repaired chart
        repaired_result = analyzer.validate_chart(plan.repaired_chart)
        print(f"\nRepaired: {len(repaired_result.errors)} error(s)")
        if repaired_result.errors:
            for err in repaired_result.errors:
                print(f"  - {err}")
        else:
            print("  VALID!")

        print(f"\nSuccess rate: {plan.success_rate:.1%}")

    print("\n" + "=" * 60)
    print("REPAIR STRATEGIES DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
