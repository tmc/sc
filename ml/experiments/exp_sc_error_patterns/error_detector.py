"""
Error Detector: Detection Engine for Statechart Errors

Validates statecharts and detects various error types:
- Structural errors (missing states, dangling transitions)
- Semantic errors (invalid guards, hierarchy violations)
- Behavioral errors (unreachable states, deadlocks)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum
import re

from .error_taxonomy import (
    ErrorCategory,
    ErrorSeverity,
    ErrorLocation,
    StatechartError,
    MissingStateError,
    DanglingTransitionError,
    InvalidGuardError,
    HierarchyViolationError,
    UnreachableStateError,
    NonDeterminismError,
    DuplicateElementError,
)


@dataclass
class DetectionResult:
    """Result of error detection on a statechart."""
    statechart_name: str
    is_valid: bool
    errors: List[StatechartError]
    warnings: List[StatechartError]
    info: List[StatechartError]

    # Statistics
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Coverage
    states_checked: int = 0
    transitions_checked: int = 0
    guards_checked: int = 0

    def __post_init__(self):
        self.error_count = len([e for e in self.errors if e.severity == ErrorSeverity.ERROR or e.severity == ErrorSeverity.CRITICAL])
        self.warning_count = len([e for e in self.errors if e.severity == ErrorSeverity.WARNING]) + len(self.warnings)
        self.info_count = len([e for e in self.errors if e.severity == ErrorSeverity.INFO]) + len(self.info)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"{self.statechart_name}: {status} "
            f"({self.error_count} errors, {self.warning_count} warnings, {self.info_count} info)"
        )

    def all_issues(self) -> List[StatechartError]:
        return self.errors + self.warnings + self.info


class Validator:
    """Base class for validators."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        raise NotImplementedError


class MissingStateValidator(Validator):
    """Detect missing states referenced in transitions."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        defined_states = context.get("defined_states", set())

        for trans in sc.get("transitions", []):
            # Check sources
            for src in trans.get("from", []):
                if src not in defined_states:
                    errors.append(MissingStateError(
                        category=ErrorCategory.MISSING_STATE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Source state '{src}' not found",
                        location=ErrorLocation("transition", None, trans.get("event")),
                        missing_state=src,
                        referenced_by=f"transition on '{trans.get('event', 'unknown')}'",
                        reference_type="transition_source",
                    ))

            # Check targets
            for tgt in trans.get("to", []):
                if tgt not in defined_states:
                    errors.append(MissingStateError(
                        category=ErrorCategory.MISSING_STATE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Target state '{tgt}' not found",
                        location=ErrorLocation("transition", None, trans.get("event")),
                        missing_state=tgt,
                        referenced_by=f"transition on '{trans.get('event', 'unknown')}'",
                        reference_type="transition_target",
                    ))

        return errors


class DanglingTransitionValidator(Validator):
    """Detect transitions with no valid source or target."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        defined_states = context.get("defined_states", set())

        for trans in sc.get("transitions", []):
            sources = trans.get("from", [])
            targets = trans.get("to", [])

            # Check for empty sources
            if not sources:
                errors.append(DanglingTransitionError(
                    category=ErrorCategory.DANGLING_TRANSITION,
                    severity=ErrorSeverity.ERROR,
                    message="Transition has no source states",
                    location=ErrorLocation("transition", None, trans.get("event")),
                    transition_event=trans.get("event", ""),
                    invalid_end="source",
                    invalid_state="(none)",
                    valid_states=list(defined_states),
                ))

            # Check for empty targets
            if not targets:
                errors.append(DanglingTransitionError(
                    category=ErrorCategory.DANGLING_TRANSITION,
                    severity=ErrorSeverity.ERROR,
                    message="Transition has no target states",
                    location=ErrorLocation("transition", None, trans.get("event")),
                    transition_event=trans.get("event", ""),
                    invalid_end="target",
                    invalid_state="(none)",
                    valid_states=list(defined_states),
                ))

        return errors


class InvalidGuardValidator(Validator):
    """Detect invalid guard expressions."""

    # Simple patterns for guard validation
    VALID_OPERATORS = {"==", "!=", "<", ">", "<=", ">=", "&&", "||", "!"}
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        defined_vars = context.get("defined_variables", set())

        for trans in sc.get("transitions", []):
            guard = trans.get("guard", "")
            if not guard:
                continue

            # Check for syntax issues
            syntax_errors = self._check_syntax(guard)
            if syntax_errors:
                errors.append(InvalidGuardError(
                    category=ErrorCategory.INVALID_GUARD,
                    severity=ErrorSeverity.ERROR,
                    message=f"Guard syntax error: {syntax_errors}",
                    location=ErrorLocation("guard", None, guard),
                    guard_expression=guard,
                    transition_event=trans.get("event", ""),
                    error_detail=syntax_errors,
                ))
                continue

            # Check for undefined variables
            undefined = self._find_undefined_vars(guard, defined_vars)
            if undefined and defined_vars:  # Only check if we have a var list
                errors.append(InvalidGuardError(
                    category=ErrorCategory.INVALID_GUARD,
                    severity=ErrorSeverity.WARNING,
                    message=f"Guard uses undefined variable(s): {', '.join(undefined)}",
                    location=ErrorLocation("guard", None, guard),
                    guard_expression=guard,
                    transition_event=trans.get("event", ""),
                    error_detail=f"Undefined: {', '.join(undefined)}",
                ))

        return errors

    def _check_syntax(self, guard: str) -> Optional[str]:
        """Check for basic syntax errors in guard."""
        # Unbalanced parentheses
        if guard.count('(') != guard.count(')'):
            return "Unbalanced parentheses"

        # Empty comparison
        if re.search(r'[<>=!]=?\s*[<>=!]', guard):
            return "Invalid operator sequence"

        # Missing operand
        if re.search(r'^[<>=!&|]|[<>=!&|]$', guard.strip()):
            return "Missing operand"

        return None

    def _find_undefined_vars(self, guard: str, defined: Set[str]) -> List[str]:
        """Find variables used in guard but not defined."""
        # Extract identifiers
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', guard)

        # Filter out keywords and operators
        keywords = {"true", "false", "and", "or", "not", "in", "is"}
        undefined = []

        for token in tokens:
            if token.lower() not in keywords and token not in defined:
                undefined.append(token)

        return undefined


class HierarchyValidator(Validator):
    """Detect hierarchy violations in state structure."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        root = sc.get("root_state", {})

        # Check hierarchy
        errors.extend(self._validate_state(root, None, context))

        return errors

    def _validate_state(
        self,
        state: Dict,
        parent: Optional[str],
        context: Dict,
    ) -> List[StatechartError]:
        errors = []
        label = state.get("label", "")
        state_type = state.get("type", 1)  # 1=BASIC, 2=NORMAL, 3=PARALLEL
        children = state.get("children", [])

        # Handle root markers - still check initial state for root's children
        if label.startswith("__"):
            # Check for exactly one initial state at root level
            if children:
                initial_count = sum(1 for c in children if c.get("is_initial"))
                if initial_count == 0:
                    errors.append(HierarchyViolationError(
                        category=ErrorCategory.HIERARCHY_VIOLATION,
                        severity=ErrorSeverity.WARNING,
                        message="No initial state at root level",
                        location=ErrorLocation("state", None, label),
                        violation_type="missing_initial",
                        state_label=label,
                        parent_label="",
                        detail="Root level should have one initial child state",
                    ))
                elif initial_count > 1:
                    errors.append(HierarchyViolationError(
                        category=ErrorCategory.HIERARCHY_VIOLATION,
                        severity=ErrorSeverity.ERROR,
                        message="Multiple initial states at root level",
                        location=ErrorLocation("state", None, label),
                        violation_type="multiple_initial",
                        state_label=label,
                        parent_label="",
                        detail=f"Found {initial_count} initial states, expected 1",
                    ))
            for child in children:
                errors.extend(self._validate_state(child, None, context))
            return errors

        # Check: basic states should not have children
        if state_type == 1 and children:
            errors.append(HierarchyViolationError(
                category=ErrorCategory.HIERARCHY_VIOLATION,
                severity=ErrorSeverity.ERROR,
                message=f"Basic state '{label}' has children",
                location=ErrorLocation("state", None, label),
                violation_type="invalid_nesting",
                state_label=label,
                parent_label=parent or "",
                detail="Basic states (type=1) cannot have child states",
            ))

        # Check: composite states should have children
        if state_type in (2, 3) and not children:
            errors.append(HierarchyViolationError(
                category=ErrorCategory.HIERARCHY_VIOLATION,
                severity=ErrorSeverity.WARNING,
                message=f"Composite state '{label}' has no children",
                location=ErrorLocation("state", None, label),
                violation_type="empty_composite",
                state_label=label,
                parent_label=parent or "",
                detail="Composite states should have at least one child",
            ))

        # Check: exactly one initial state in composite
        if state_type == 2 and children:
            initial_count = sum(1 for c in children if c.get("is_initial"))
            if initial_count == 0:
                errors.append(HierarchyViolationError(
                    category=ErrorCategory.HIERARCHY_VIOLATION,
                    severity=ErrorSeverity.WARNING,
                    message=f"No initial state in composite '{label}'",
                    location=ErrorLocation("state", None, label),
                    violation_type="missing_initial",
                    state_label=label,
                    parent_label=parent or "",
                    detail="Composite states should have one initial child state",
                ))
            elif initial_count > 1:
                errors.append(HierarchyViolationError(
                    category=ErrorCategory.HIERARCHY_VIOLATION,
                    severity=ErrorSeverity.ERROR,
                    message=f"Multiple initial states in composite '{label}'",
                    location=ErrorLocation("state", None, label),
                    violation_type="multiple_initial",
                    state_label=label,
                    parent_label=parent or "",
                    detail=f"Found {initial_count} initial states, expected 1",
                ))

        # Check: parallel states need multiple children
        if state_type == 3 and len(children) < 2:
            errors.append(HierarchyViolationError(
                category=ErrorCategory.HIERARCHY_VIOLATION,
                severity=ErrorSeverity.WARNING,
                message=f"Parallel state '{label}' has fewer than 2 regions",
                location=ErrorLocation("state", None, label),
                violation_type="parallel_insufficient",
                state_label=label,
                parent_label=parent or "",
                detail="Parallel states should have at least 2 concurrent regions",
            ))

        # Recurse into children
        for child in children:
            errors.extend(self._validate_state(child, label, context))

        return errors


class ReachabilityValidator(Validator):
    """Detect unreachable states."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        defined_states = context.get("defined_states", set())
        initial_states = context.get("initial_states", set())

        if not initial_states:
            return errors

        # Build reachability graph
        reachable = self._compute_reachable(sc, initial_states)

        # Find unreachable states
        for state in defined_states:
            if state not in reachable:
                errors.append(UnreachableStateError(
                    category=ErrorCategory.UNREACHABLE_STATE,
                    severity=ErrorSeverity.WARNING,
                    message=f"State '{state}' is unreachable",
                    location=ErrorLocation("state", None, state),
                    unreachable_state=state,
                    reason="No path from initial state(s)",
                ))

        return errors

    def _compute_reachable(
        self,
        sc: Dict[str, Any],
        initial: Set[str],
    ) -> Set[str]:
        """Compute set of reachable states from initial states."""
        reachable = set(initial)
        worklist = list(initial)

        # Build adjacency
        adj: Dict[str, Set[str]] = {}
        for trans in sc.get("transitions", []):
            for src in trans.get("from", []):
                if src not in adj:
                    adj[src] = set()
                for tgt in trans.get("to", []):
                    adj[src].add(tgt)

        # BFS
        while worklist:
            current = worklist.pop(0)
            for neighbor in adj.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    worklist.append(neighbor)

        return reachable


class NonDeterminismValidator(Validator):
    """Detect non-deterministic transitions."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []

        # Group transitions by (source, event)
        trans_groups: Dict[Tuple[str, str], List[Dict]] = {}

        for trans in sc.get("transitions", []):
            event = trans.get("event", "")
            for src in trans.get("from", []):
                key = (src, event)
                if key not in trans_groups:
                    trans_groups[key] = []
                trans_groups[key].append(trans)

        # Check for conflicts
        for (src, event), transitions in trans_groups.items():
            if len(transitions) > 1:
                # Check if guards are mutually exclusive
                guards = [t.get("guard", "") for t in transitions]
                if not self._guards_exclusive(guards):
                    targets = []
                    for t in transitions:
                        targets.extend(t.get("to", []))

                    errors.append(NonDeterminismError(
                        category=ErrorCategory.NON_DETERMINISM,
                        severity=ErrorSeverity.WARNING,
                        message=f"Non-determinism in '{src}' on '{event}'",
                        location=ErrorLocation("state", None, src),
                        state=src,
                        event=event,
                        conflicting_targets=list(set(targets)),
                    ))

        return errors

    def _guards_exclusive(self, guards: List[str]) -> bool:
        """Check if guards are mutually exclusive (heuristic)."""
        # Empty guards = definitely not exclusive
        if all(not g for g in guards):
            return False

        # If all have guards, assume programmer made them exclusive
        if all(g for g in guards):
            return True

        return False


class DuplicateValidator(Validator):
    """Detect duplicate state definitions."""

    def validate(self, sc: Dict[str, Any], context: Dict[str, Any]) -> List[StatechartError]:
        errors = []
        state_counts = context.get("state_counts", {})

        for state, count in state_counts.items():
            if count > 1:
                errors.append(DuplicateElementError(
                    category=ErrorCategory.DUPLICATE_ELEMENT,
                    severity=ErrorSeverity.ERROR,
                    message=f"Duplicate state '{state}'",
                    location=ErrorLocation("state", None, state),
                    element_type="state",
                    element_name=state,
                    occurrences=count,
                ))

        return errors


class ErrorDetector:
    """
    Main error detection engine.

    Runs multiple validators and aggregates results.
    """

    def __init__(self):
        self.validators: List[Validator] = [
            MissingStateValidator(),
            DanglingTransitionValidator(),
            InvalidGuardValidator(),
            HierarchyValidator(),
            ReachabilityValidator(),
            NonDeterminismValidator(),
            DuplicateValidator(),
        ]

    def detect(self, statechart: Dict[str, Any]) -> DetectionResult:
        """Run all validators on statechart."""
        # Build context
        context = self._build_context(statechart)

        # Run validators
        all_errors: List[StatechartError] = []
        for validator in self.validators:
            errors = validator.validate(statechart, context)
            all_errors.extend(errors)

        # Separate by severity
        errors = [e for e in all_errors if e.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.ERROR)]
        warnings = [e for e in all_errors if e.severity == ErrorSeverity.WARNING]
        info = [e for e in all_errors if e.severity == ErrorSeverity.INFO]

        is_valid = len(errors) == 0

        return DetectionResult(
            statechart_name=statechart.get("name", "Unknown"),
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            states_checked=len(context.get("defined_states", [])),
            transitions_checked=len(statechart.get("transitions", [])),
            guards_checked=sum(1 for t in statechart.get("transitions", []) if t.get("guard")),
        )

    def _build_context(self, sc: Dict[str, Any]) -> Dict[str, Any]:
        """Build validation context from statechart."""
        context = {
            "defined_states": set(),
            "initial_states": set(),
            "final_states": set(),
            "state_counts": {},
            "defined_variables": set(),
        }

        # Extract states
        root = sc.get("root_state", {})
        self._extract_states(root, context)

        return context

    def _extract_states(self, state: Dict, context: Dict):
        """Recursively extract state information."""
        label = state.get("label", "")

        if label and not label.startswith("__"):
            context["defined_states"].add(label)
            context["state_counts"][label] = context["state_counts"].get(label, 0) + 1

            if state.get("is_initial"):
                context["initial_states"].add(label)
            if state.get("is_final"):
                context["final_states"].add(label)

        for child in state.get("children", []):
            self._extract_states(child, context)


def detect_errors(statechart: Dict[str, Any]) -> DetectionResult:
    """Convenience function to detect errors in statechart."""
    detector = ErrorDetector()
    return detector.detect(statechart)


def demo():
    """Demonstrate error detection."""
    print("=" * 60)
    print("ERROR DETECTOR: Statechart Validation")
    print("=" * 60)

    # Example with multiple errors
    broken_sc = {
        "name": "Broken Statechart",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Start", "type": 1, "is_initial": True},
                {"label": "Start", "type": 1},  # Duplicate!
                {"label": "Middle", "type": 1},
                {"label": "End", "type": 1, "is_final": True},
                {"label": "Orphan", "type": 1},  # Unreachable
            ]
        },
        "transitions": [
            {"from": ["Start"], "to": ["Middle"], "event": "GO"},
            {"from": ["Middle"], "to": ["NonExistent"], "event": "NEXT"},  # Missing target
            {"from": ["Midle"], "to": ["End"], "event": "FINISH"},  # Typo in source
            {"from": ["Middle"], "to": ["End"], "event": "FINISH"},  # Non-determinism
            {"from": [], "to": ["Start"], "event": "RESET"},  # No source
            {"from": ["Start"], "to": ["Middle"], "event": "GO",
             "guard": "count >> 5"},  # Invalid guard
        ]
    }

    print("\n--- Checking Broken Statechart ---")
    result = detect_errors(broken_sc)
    print(result.summary())

    print("\nErrors:")
    for err in result.errors:
        print(f"  [{err.category.value}] {err.message}")
        print(f"    Fix: {err.suggest_fix()}")

    print("\nWarnings:")
    for warn in result.warnings:
        print(f"  [{warn.category.value}] {warn.message}")

    # Valid statechart
    valid_sc = {
        "name": "Valid Statechart",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "Active", "type": 1},
                {"label": "Done", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Active"], "event": "START"},
            {"from": ["Active"], "to": ["Done"], "event": "FINISH"},
            {"from": ["Active"], "to": ["Idle"], "event": "CANCEL"},
        ]
    }

    print("\n--- Checking Valid Statechart ---")
    result2 = detect_errors(valid_sc)
    print(result2.summary())

    return result, result2


if __name__ == "__main__":
    demo()
