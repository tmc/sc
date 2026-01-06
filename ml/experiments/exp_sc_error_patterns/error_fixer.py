"""
Error Fixer: Automatic Fix Suggestions and Application

Generates and optionally applies fixes for detected errors:
- State name corrections (typos)
- Missing state additions
- Guard syntax fixes
- Hierarchy corrections
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
import copy
import re

from .error_taxonomy import (
    ErrorCategory,
    StatechartError,
    MissingStateError,
    DanglingTransitionError,
    InvalidGuardError,
    HierarchyViolationError,
)


class FixType(Enum):
    """Type of fix action."""
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    RENAME_STATE = "rename_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    UPDATE_TRANSITION = "update_transition"
    FIX_GUARD = "fix_guard"
    SET_INITIAL = "set_initial"
    CHANGE_TYPE = "change_type"


class FixConfidence(Enum):
    """Confidence level in fix."""
    HIGH = "high"        # Very likely correct
    MEDIUM = "medium"    # Probably correct
    LOW = "low"          # Uncertain


@dataclass
class FixSuggestion:
    """A suggested fix for an error."""
    error: StatechartError
    fix_type: FixType
    description: str
    confidence: FixConfidence
    changes: List[Dict[str, Any]]  # List of changes to apply
    side_effects: List[str] = field(default_factory=list)

    def explain(self) -> str:
        """Human-readable explanation of fix."""
        return f"[{self.confidence.value}] {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fix_type": self.fix_type.value,
            "description": self.description,
            "confidence": self.confidence.value,
            "changes": self.changes,
        }


@dataclass
class FixResult:
    """Result of applying fixes."""
    original: Dict[str, Any]
    fixed: Dict[str, Any]
    applied_fixes: List[FixSuggestion]
    skipped_fixes: List[FixSuggestion]
    remaining_errors: int


class ErrorFixer:
    """
    Generate and apply fixes for statechart errors.
    """

    def __init__(self):
        self.fix_generators: Dict[ErrorCategory, Callable] = {
            ErrorCategory.MISSING_STATE: self._fix_missing_state,
            ErrorCategory.DANGLING_TRANSITION: self._fix_dangling_transition,
            ErrorCategory.INVALID_GUARD: self._fix_invalid_guard,
            ErrorCategory.HIERARCHY_VIOLATION: self._fix_hierarchy_violation,
            ErrorCategory.UNREACHABLE_STATE: self._fix_unreachable_state,
            ErrorCategory.NON_DETERMINISM: self._fix_non_determinism,
            ErrorCategory.DUPLICATE_ELEMENT: self._fix_duplicate,
        }

    def suggest_fixes(
        self,
        errors: List[StatechartError],
        statechart: Dict[str, Any],
    ) -> List[FixSuggestion]:
        """Generate fix suggestions for errors."""
        suggestions = []

        for error in errors:
            generator = self.fix_generators.get(error.category)
            if generator:
                fixes = generator(error, statechart)
                suggestions.extend(fixes)

        return suggestions

    def apply_fixes(
        self,
        statechart: Dict[str, Any],
        fixes: List[FixSuggestion],
        min_confidence: FixConfidence = FixConfidence.MEDIUM,
    ) -> FixResult:
        """Apply fixes to statechart."""
        fixed = copy.deepcopy(statechart)
        applied = []
        skipped = []

        confidence_order = {
            FixConfidence.HIGH: 3,
            FixConfidence.MEDIUM: 2,
            FixConfidence.LOW: 1,
        }
        min_level = confidence_order[min_confidence]

        for fix in fixes:
            if confidence_order[fix.confidence] >= min_level:
                try:
                    fixed = self._apply_fix(fixed, fix)
                    applied.append(fix)
                except Exception as e:
                    fix.side_effects.append(f"Apply failed: {e}")
                    skipped.append(fix)
            else:
                skipped.append(fix)

        return FixResult(
            original=statechart,
            fixed=fixed,
            applied_fixes=applied,
            skipped_fixes=skipped,
            remaining_errors=0,  # Would need re-detection
        )

    def _apply_fix(
        self,
        sc: Dict[str, Any],
        fix: FixSuggestion,
    ) -> Dict[str, Any]:
        """Apply a single fix to statechart."""
        for change in fix.changes:
            action = change.get("action")

            if action == "add_state":
                sc = self._add_state(sc, change)
            elif action == "rename_state":
                sc = self._rename_state(sc, change)
            elif action == "remove_transition":
                sc = self._remove_transition(sc, change)
            elif action == "update_transition":
                sc = self._update_transition(sc, change)
            elif action == "update_guard":
                sc = self._update_guard(sc, change)
            elif action == "set_initial":
                sc = self._set_initial(sc, change)

        return sc

    def _add_state(self, sc: Dict, change: Dict) -> Dict:
        """Add a new state."""
        state_name = change.get("state_name")
        parent = change.get("parent")
        state_type = change.get("state_type", 1)

        new_state = {
            "label": state_name,
            "type": state_type,
            "is_initial": change.get("is_initial", False),
        }

        def add_to_parent(state):
            label = state.get("label", "")
            if label == parent or (not parent and label.startswith("__")):
                if "children" not in state:
                    state["children"] = []
                state["children"].append(new_state)
                return True
            for child in state.get("children", []):
                if add_to_parent(child):
                    return True
            return False

        add_to_parent(sc.get("root_state", {}))
        return sc

    def _rename_state(self, sc: Dict, change: Dict) -> Dict:
        """Rename a state and update references."""
        old_name = change.get("old_name")
        new_name = change.get("new_name")

        def rename_in_tree(state):
            if state.get("label") == old_name:
                state["label"] = new_name
            for child in state.get("children", []):
                rename_in_tree(child)

        rename_in_tree(sc.get("root_state", {}))

        # Update transitions
        for trans in sc.get("transitions", []):
            trans["from"] = [new_name if s == old_name else s for s in trans.get("from", [])]
            trans["to"] = [new_name if s == old_name else s for s in trans.get("to", [])]

        return sc

    def _remove_transition(self, sc: Dict, change: Dict) -> Dict:
        """Remove a transition."""
        event = change.get("event")
        source = change.get("from")
        target = change.get("to")

        sc["transitions"] = [
            t for t in sc.get("transitions", [])
            if not (
                t.get("event") == event and
                (source is None or source in t.get("from", [])) and
                (target is None or target in t.get("to", []))
            )
        ]
        return sc

    def _update_transition(self, sc: Dict, change: Dict) -> Dict:
        """Update transition source or target."""
        event = change.get("event")
        updates = change.get("updates", {})

        for trans in sc.get("transitions", []):
            if trans.get("event") == event:
                if "from" in updates:
                    trans["from"] = updates["from"]
                if "to" in updates:
                    trans["to"] = updates["to"]
                break

        return sc

    def _update_guard(self, sc: Dict, change: Dict) -> Dict:
        """Update guard expression."""
        event = change.get("event")
        new_guard = change.get("new_guard")

        for trans in sc.get("transitions", []):
            if trans.get("event") == event:
                trans["guard"] = new_guard
                break

        return sc

    def _set_initial(self, sc: Dict, change: Dict) -> Dict:
        """Set initial state in composite."""
        parent = change.get("parent")
        initial_state = change.get("initial_state")

        def set_in_tree(state):
            label = state.get("label", "")
            if label == parent or (not parent and label.startswith("__")):
                for child in state.get("children", []):
                    child["is_initial"] = (child.get("label") == initial_state)
                return True
            for child in state.get("children", []):
                if set_in_tree(child):
                    return True
            return False

        set_in_tree(sc.get("root_state", {}))
        return sc

    # Fix generators for each error category

    def _fix_missing_state(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for missing state errors."""
        fixes = []

        if isinstance(error, MissingStateError):
            missing = error.missing_state
            defined = self._get_defined_states(sc)

            # Option 1: Add the missing state
            fixes.append(FixSuggestion(
                error=error,
                fix_type=FixType.ADD_STATE,
                description=f"Add missing state '{missing}'",
                confidence=FixConfidence.MEDIUM,
                changes=[{
                    "action": "add_state",
                    "state_name": missing,
                    "parent": None,  # Add to root
                    "state_type": 1,
                }],
                side_effects=["New state may need transitions"],
            ))

            # Option 2: Check for similar state (typo fix)
            similar = self._find_similar_state(missing, defined)
            if similar:
                fixes.insert(0, FixSuggestion(
                    error=error,
                    fix_type=FixType.UPDATE_TRANSITION,
                    description=f"Replace '{missing}' with '{similar}' (likely typo)",
                    confidence=FixConfidence.HIGH,
                    changes=[{
                        "action": "rename_reference",
                        "old_name": missing,
                        "new_name": similar,
                    }],
                ))

        return fixes

    def _fix_dangling_transition(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for dangling transitions."""
        fixes = []

        if isinstance(error, DanglingTransitionError):
            # Option 1: Remove the broken transition
            fixes.append(FixSuggestion(
                error=error,
                fix_type=FixType.REMOVE_TRANSITION,
                description=f"Remove transition on '{error.transition_event}'",
                confidence=FixConfidence.MEDIUM,
                changes=[{
                    "action": "remove_transition",
                    "event": error.transition_event,
                }],
            ))

            # Option 2: Fix with valid state
            if error.valid_states:
                first_valid = error.valid_states[0]
                fixes.insert(0, FixSuggestion(
                    error=error,
                    fix_type=FixType.UPDATE_TRANSITION,
                    description=f"Update {error.invalid_end} to '{first_valid}'",
                    confidence=FixConfidence.LOW,
                    changes=[{
                        "action": "update_transition",
                        "event": error.transition_event,
                        "updates": {error.invalid_end: [first_valid]},
                    }],
                ))

        return fixes

    def _fix_invalid_guard(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for invalid guards."""
        fixes = []

        if isinstance(error, InvalidGuardError):
            guard = error.guard_expression
            fixed_guard = self._try_fix_guard(guard)

            if fixed_guard and fixed_guard != guard:
                fixes.append(FixSuggestion(
                    error=error,
                    fix_type=FixType.FIX_GUARD,
                    description=f"Fix guard: '{guard}' -> '{fixed_guard}'",
                    confidence=FixConfidence.HIGH,
                    changes=[{
                        "action": "update_guard",
                        "event": error.transition_event,
                        "new_guard": fixed_guard,
                    }],
                ))

            # Option: Remove guard
            fixes.append(FixSuggestion(
                error=error,
                fix_type=FixType.FIX_GUARD,
                description="Remove invalid guard",
                confidence=FixConfidence.LOW,
                changes=[{
                    "action": "update_guard",
                    "event": error.transition_event,
                    "new_guard": "",
                }],
            ))

        return fixes

    def _fix_hierarchy_violation(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for hierarchy violations."""
        fixes = []

        if isinstance(error, HierarchyViolationError):
            if "initial" in error.violation_type.lower():
                # Set first child as initial
                fixes.append(FixSuggestion(
                    error=error,
                    fix_type=FixType.SET_INITIAL,
                    description=f"Set first child as initial in '{error.state_label}'",
                    confidence=FixConfidence.MEDIUM,
                    changes=[{
                        "action": "set_initial",
                        "parent": error.state_label,
                        "initial_state": None,  # Will pick first
                    }],
                ))

        return fixes

    def _fix_unreachable_state(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for unreachable states."""
        return [FixSuggestion(
            error=error,
            fix_type=FixType.REMOVE_STATE,
            description=f"Remove unreachable state",
            confidence=FixConfidence.LOW,
            changes=[{
                "action": "remove_state",
                "state_name": getattr(error, "unreachable_state", ""),
            }],
        )]

    def _fix_non_determinism(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for non-determinism."""
        return [FixSuggestion(
            error=error,
            fix_type=FixType.FIX_GUARD,
            description="Add mutually exclusive guards",
            confidence=FixConfidence.LOW,
            changes=[],  # Would need manual intervention
            side_effects=["Requires manual guard creation"],
        )]

    def _fix_duplicate(
        self,
        error: StatechartError,
        sc: Dict,
    ) -> List[FixSuggestion]:
        """Generate fixes for duplicates."""
        return [FixSuggestion(
            error=error,
            fix_type=FixType.REMOVE_STATE,
            description="Remove duplicate definition",
            confidence=FixConfidence.MEDIUM,
            changes=[{
                "action": "remove_duplicate",
                "element": getattr(error, "element_name", ""),
            }],
        )]

    # Helper methods

    def _get_defined_states(self, sc: Dict) -> Set[str]:
        states = set()
        def collect(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for child in state.get("children", []):
                collect(child)
        collect(sc.get("root_state", {}))
        return states

    def _find_similar_state(self, target: str, candidates: Set[str]) -> Optional[str]:
        target_lower = target.lower()
        for c in candidates:
            if target_lower in c.lower() or c.lower() in target_lower:
                return c
            # Simple edit distance check
            if abs(len(target) - len(c)) <= 2:
                diffs = sum(1 for a, b in zip(target.lower(), c.lower()) if a != b)
                if diffs <= 2:
                    return c
        return None

    def _try_fix_guard(self, guard: str) -> Optional[str]:
        """Try to automatically fix common guard issues."""
        fixed = guard

        # Fix single = to ==
        fixed = re.sub(r'([^<>=!])=([^=])', r'\1==\2', fixed)

        # Fix >> to >
        fixed = fixed.replace('>>', '>')

        # Fix << to <
        fixed = fixed.replace('<<', '<')

        # Fix single & to &&
        fixed = re.sub(r'([^&])&([^&])', r'\1&&\2', fixed)

        # Fix single | to ||
        fixed = re.sub(r'([^|])\|([^|])', r'\1||\2', fixed)

        return fixed if fixed != guard else None


def suggest_fixes(
    errors: List[StatechartError],
    statechart: Dict[str, Any],
) -> List[FixSuggestion]:
    """Convenience function to suggest fixes."""
    fixer = ErrorFixer()
    return fixer.suggest_fixes(errors, statechart)


def demo():
    """Demonstrate error fixing."""
    print("=" * 60)
    print("ERROR FIXER: Automatic Fix Suggestions")
    print("=" * 60)

    from .error_detector import detect_errors

    broken_sc = {
        "name": "Fixable Statechart",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Start", "type": 1, "is_initial": True},
                {"label": "Processing", "type": 1},
                {"label": "Done", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Start"], "to": ["Procesing"], "event": "BEGIN"},  # Typo
            {"from": ["Processing"], "to": ["Done"], "event": "FINISH",
             "guard": "count = 5"},  # Bad guard
        ]
    }

    # Detect errors
    result = detect_errors(broken_sc)
    print(f"\nDetected {len(result.errors)} errors:\n")

    # Generate fixes
    fixer = ErrorFixer()
    all_errors = result.errors + result.warnings
    suggestions = fixer.suggest_fixes(all_errors, broken_sc)

    print(f"Generated {len(suggestions)} fix suggestions:\n")
    for fix in suggestions:
        print(f"  {fix.explain()}")
        print(f"    Changes: {fix.changes}")
        if fix.side_effects:
            print(f"    Side effects: {fix.side_effects}")
        print()

    # Apply high-confidence fixes
    print("Applying HIGH confidence fixes...")
    fix_result = fixer.apply_fixes(broken_sc, suggestions, FixConfidence.HIGH)

    print(f"Applied: {len(fix_result.applied_fixes)}")
    print(f"Skipped: {len(fix_result.skipped_fixes)}")

    return suggestions, fix_result


if __name__ == "__main__":
    demo()
