"""
Error Taxonomy: Classification of Statechart Errors

Hierarchical error classification:
- Category (what kind of error)
- Severity (how bad is it)
- Location (where in the statechart)
- Context (surrounding elements)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class ErrorCategory(Enum):
    """Top-level error categories."""
    MISSING_STATE = "missing_state"
    DANGLING_TRANSITION = "dangling_transition"
    INVALID_GUARD = "invalid_guard"
    HIERARCHY_VIOLATION = "hierarchy_violation"
    INVALID_EVENT = "invalid_event"
    DUPLICATE_ELEMENT = "duplicate_element"
    UNREACHABLE_STATE = "unreachable_state"
    DEADLOCK = "deadlock"
    NON_DETERMINISM = "non_determinism"
    MALFORMED_STRUCTURE = "malformed_structure"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"    # Statechart cannot execute
    ERROR = "error"          # Will cause runtime failures
    WARNING = "warning"      # May cause unexpected behavior
    INFO = "info"            # Style/best practice issues


class ErrorSubtype(Enum):
    """Detailed error subtypes."""
    # Missing state subtypes
    MISSING_INITIAL = "missing_initial"
    MISSING_TARGET = "missing_target"
    MISSING_SOURCE = "missing_source"
    MISSING_PARENT = "missing_parent"

    # Dangling transition subtypes
    ORPHAN_TRANSITION = "orphan_transition"
    BROKEN_SOURCE = "broken_source"
    BROKEN_TARGET = "broken_target"

    # Invalid guard subtypes
    SYNTAX_ERROR = "syntax_error"
    UNDEFINED_VARIABLE = "undefined_variable"
    TYPE_MISMATCH = "type_mismatch"
    ALWAYS_FALSE = "always_false"
    ALWAYS_TRUE = "always_true"

    # Hierarchy subtypes
    INVALID_NESTING = "invalid_nesting"
    PARALLEL_CONFLICT = "parallel_conflict"
    CROSS_BOUNDARY = "cross_boundary"
    MISSING_DEFAULT = "missing_default"
    MULTIPLE_INITIAL = "multiple_initial"


@dataclass
class ErrorLocation:
    """Location of error in statechart."""
    element_type: str          # "state", "transition", "guard", etc.
    element_id: Optional[str]  # ID if available
    element_label: Optional[str]
    path: List[str] = field(default_factory=list)  # Path in hierarchy

    def __str__(self):
        if self.element_label:
            return f"{self.element_type}:{self.element_label}"
        elif self.element_id:
            return f"{self.element_type}:{self.element_id}"
        return self.element_type


@dataclass
class StatechartError(ABC):
    """Base class for all statechart errors."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    location: ErrorLocation
    subtype: Optional[ErrorSubtype] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def explain(self) -> str:
        """Generate human-readable explanation."""
        pass

    @abstractmethod
    def suggest_fix(self) -> str:
        """Suggest how to fix the error."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": str(self.location),
            "subtype": self.subtype.value if self.subtype else None,
        }


@dataclass
class MissingStateError(StatechartError):
    """Error when a referenced state doesn't exist."""
    missing_state: str = ""
    referenced_by: str = ""
    reference_type: str = ""  # "transition_source", "transition_target", "parent"

    def __post_init__(self):
        self.category = ErrorCategory.MISSING_STATE
        if not self.subtype:
            if self.reference_type == "transition_target":
                self.subtype = ErrorSubtype.MISSING_TARGET
            elif self.reference_type == "transition_source":
                self.subtype = ErrorSubtype.MISSING_SOURCE
            elif self.reference_type == "parent":
                self.subtype = ErrorSubtype.MISSING_PARENT
            else:
                self.subtype = ErrorSubtype.MISSING_TARGET

    def explain(self) -> str:
        return (
            f"State '{self.missing_state}' is referenced but not defined. "
            f"It is used as {self.reference_type} by {self.referenced_by}."
        )

    def suggest_fix(self) -> str:
        return (
            f"Either add a state named '{self.missing_state}' to the statechart, "
            f"or update the reference in {self.referenced_by} to use an existing state."
        )


@dataclass
class DanglingTransitionError(StatechartError):
    """Error when a transition has invalid source or target."""
    transition_event: str = ""
    invalid_end: str = ""  # "source" or "target"
    invalid_state: str = ""
    valid_states: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.category = ErrorCategory.DANGLING_TRANSITION
        if self.invalid_end == "source":
            self.subtype = ErrorSubtype.BROKEN_SOURCE
        else:
            self.subtype = ErrorSubtype.BROKEN_TARGET

    def explain(self) -> str:
        return (
            f"Transition on '{self.transition_event}' has invalid {self.invalid_end} "
            f"'{self.invalid_state}'. This state does not exist in the statechart."
        )

    def suggest_fix(self) -> str:
        if self.valid_states:
            similar = self._find_similar(self.invalid_state, self.valid_states)
            if similar:
                return f"Did you mean '{similar}'? Available states: {', '.join(self.valid_states[:5])}"
        return f"Update the {self.invalid_end} to one of: {', '.join(self.valid_states[:5])}"

    def _find_similar(self, target: str, candidates: List[str]) -> Optional[str]:
        """Find most similar state name."""
        target_lower = target.lower()
        for c in candidates:
            if target_lower in c.lower() or c.lower() in target_lower:
                return c
        return None


@dataclass
class InvalidGuardError(StatechartError):
    """Error when a guard expression is invalid."""
    guard_expression: str = ""
    transition_event: str = ""
    error_detail: str = ""

    def __post_init__(self):
        self.category = ErrorCategory.INVALID_GUARD
        if "undefined" in self.error_detail.lower():
            self.subtype = ErrorSubtype.UNDEFINED_VARIABLE
        elif "syntax" in self.error_detail.lower():
            self.subtype = ErrorSubtype.SYNTAX_ERROR
        elif "type" in self.error_detail.lower():
            self.subtype = ErrorSubtype.TYPE_MISMATCH

    def explain(self) -> str:
        return (
            f"Guard expression '{self.guard_expression}' on transition '{self.transition_event}' "
            f"is invalid: {self.error_detail}"
        )

    def suggest_fix(self) -> str:
        if self.subtype == ErrorSubtype.UNDEFINED_VARIABLE:
            return "Define the variable in the statechart context or use an existing variable."
        elif self.subtype == ErrorSubtype.SYNTAX_ERROR:
            return "Check the guard syntax. Guards should be boolean expressions."
        elif self.subtype == ErrorSubtype.TYPE_MISMATCH:
            return "Ensure the guard expression evaluates to a boolean value."
        return "Review and correct the guard expression."


@dataclass
class HierarchyViolationError(StatechartError):
    """Error when state hierarchy rules are violated."""
    violation_type: str = ""
    state_label: str = ""
    parent_label: str = ""
    detail: str = ""

    def __post_init__(self):
        self.category = ErrorCategory.HIERARCHY_VIOLATION
        if "parallel" in self.violation_type.lower():
            self.subtype = ErrorSubtype.PARALLEL_CONFLICT
        elif "nesting" in self.violation_type.lower():
            self.subtype = ErrorSubtype.INVALID_NESTING
        elif "boundary" in self.violation_type.lower():
            self.subtype = ErrorSubtype.CROSS_BOUNDARY
        elif "initial" in self.violation_type.lower():
            self.subtype = ErrorSubtype.MISSING_DEFAULT
        elif "multiple" in self.violation_type.lower():
            self.subtype = ErrorSubtype.MULTIPLE_INITIAL

    def explain(self) -> str:
        base = f"Hierarchy violation in state '{self.state_label}'"
        if self.parent_label:
            base += f" under parent '{self.parent_label}'"
        return f"{base}: {self.detail}"

    def suggest_fix(self) -> str:
        if self.subtype == ErrorSubtype.PARALLEL_CONFLICT:
            return "Ensure parallel regions have non-overlapping state sets."
        elif self.subtype == ErrorSubtype.INVALID_NESTING:
            return "Check that composite states have valid children."
        elif self.subtype == ErrorSubtype.CROSS_BOUNDARY:
            return "Transitions should not cross parallel region boundaries without proper semantics."
        elif self.subtype == ErrorSubtype.MISSING_DEFAULT:
            return "Add an initial state to the composite state."
        elif self.subtype == ErrorSubtype.MULTIPLE_INITIAL:
            return "Only one state should be marked as initial within a region."
        return "Review the state hierarchy structure."


@dataclass
class UnreachableStateError(StatechartError):
    """Error when a state cannot be reached from initial state."""
    unreachable_state: str = ""
    reason: str = ""

    def __post_init__(self):
        self.category = ErrorCategory.UNREACHABLE_STATE
        self.severity = ErrorSeverity.WARNING

    def explain(self) -> str:
        return f"State '{self.unreachable_state}' is unreachable: {self.reason}"

    def suggest_fix(self) -> str:
        return f"Add a transition leading to '{self.unreachable_state}' or remove it if unused."


@dataclass
class NonDeterminismError(StatechartError):
    """Error when multiple transitions are enabled simultaneously."""
    state: str = ""
    event: str = ""
    conflicting_targets: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.category = ErrorCategory.NON_DETERMINISM
        self.severity = ErrorSeverity.WARNING

    def explain(self) -> str:
        targets = ", ".join(self.conflicting_targets)
        return (
            f"Non-determinism in state '{self.state}' on event '{self.event}': "
            f"multiple transitions to {targets} may be enabled simultaneously."
        )

    def suggest_fix(self) -> str:
        return "Add mutually exclusive guards to ensure only one transition fires."


@dataclass
class DuplicateElementError(StatechartError):
    """Error when an element is defined multiple times."""
    element_type: str = ""
    element_name: str = ""
    occurrences: int = 0

    def __post_init__(self):
        self.category = ErrorCategory.DUPLICATE_ELEMENT
        self.severity = ErrorSeverity.ERROR

    def explain(self) -> str:
        return f"Duplicate {self.element_type} '{self.element_name}' found {self.occurrences} times."

    def suggest_fix(self) -> str:
        return f"Remove duplicate definitions of {self.element_type} '{self.element_name}'."


def classify_error(error_data: Dict[str, Any]) -> StatechartError:
    """Factory function to create appropriate error type."""
    category = error_data.get("category", "")

    if category == "missing_state" or "missing" in str(error_data.get("message", "")).lower():
        return MissingStateError(
            category=ErrorCategory.MISSING_STATE,
            severity=ErrorSeverity(error_data.get("severity", "error")),
            message=error_data.get("message", ""),
            location=ErrorLocation(
                element_type="state",
                element_label=error_data.get("state"),
            ),
            missing_state=error_data.get("missing_state", ""),
            referenced_by=error_data.get("referenced_by", ""),
            reference_type=error_data.get("reference_type", ""),
        )

    elif category == "dangling_transition" or "dangling" in str(error_data.get("message", "")).lower():
        return DanglingTransitionError(
            category=ErrorCategory.DANGLING_TRANSITION,
            severity=ErrorSeverity(error_data.get("severity", "error")),
            message=error_data.get("message", ""),
            location=ErrorLocation(
                element_type="transition",
                element_label=error_data.get("event"),
            ),
            transition_event=error_data.get("event", ""),
            invalid_end=error_data.get("invalid_end", "target"),
            invalid_state=error_data.get("invalid_state", ""),
        )

    elif category == "invalid_guard" or "guard" in str(error_data.get("message", "")).lower():
        return InvalidGuardError(
            category=ErrorCategory.INVALID_GUARD,
            severity=ErrorSeverity(error_data.get("severity", "error")),
            message=error_data.get("message", ""),
            location=ErrorLocation(
                element_type="guard",
                element_label=error_data.get("guard"),
            ),
            guard_expression=error_data.get("guard", ""),
            transition_event=error_data.get("event", ""),
            error_detail=error_data.get("detail", ""),
        )

    elif category == "hierarchy_violation" or "hierarchy" in str(error_data.get("message", "")).lower():
        return HierarchyViolationError(
            category=ErrorCategory.HIERARCHY_VIOLATION,
            severity=ErrorSeverity(error_data.get("severity", "error")),
            message=error_data.get("message", ""),
            location=ErrorLocation(
                element_type="state",
                element_label=error_data.get("state"),
            ),
            violation_type=error_data.get("violation_type", ""),
            state_label=error_data.get("state", ""),
            parent_label=error_data.get("parent", ""),
            detail=error_data.get("detail", ""),
        )

    # Default to generic error
    return MissingStateError(
        category=ErrorCategory.MISSING_STATE,
        severity=ErrorSeverity.ERROR,
        message=error_data.get("message", "Unknown error"),
        location=ErrorLocation(element_type="unknown", element_label=None),
    )


def demo():
    """Demonstrate error taxonomy."""
    print("=" * 60)
    print("ERROR TAXONOMY: Statechart Error Classification")
    print("=" * 60)

    # Example errors
    errors = [
        MissingStateError(
            category=ErrorCategory.MISSING_STATE,
            severity=ErrorSeverity.ERROR,
            message="State 'Processing' not found",
            location=ErrorLocation("transition", None, "SUBMIT"),
            missing_state="Processing",
            referenced_by="transition on SUBMIT",
            reference_type="transition_target",
        ),
        DanglingTransitionError(
            category=ErrorCategory.DANGLING_TRANSITION,
            severity=ErrorSeverity.ERROR,
            message="Transition target invalid",
            location=ErrorLocation("transition", None, "COMPLETE"),
            transition_event="COMPLETE",
            invalid_end="target",
            invalid_state="Donee",  # Typo
            valid_states=["Done", "Pending", "Active"],
        ),
        InvalidGuardError(
            category=ErrorCategory.INVALID_GUARD,
            severity=ErrorSeverity.ERROR,
            message="Guard syntax error",
            location=ErrorLocation("guard", None, "count > 3"),
            guard_expression="count >> 3",
            transition_event="INCREMENT",
            error_detail="Syntax error: unexpected '>>'",
        ),
        HierarchyViolationError(
            category=ErrorCategory.HIERARCHY_VIOLATION,
            severity=ErrorSeverity.ERROR,
            message="Invalid parallel region",
            location=ErrorLocation("state", None, "Parallel1"),
            violation_type="parallel_conflict",
            state_label="Parallel1",
            parent_label="Main",
            detail="Parallel regions share state 'Shared'",
        ),
    ]

    for err in errors:
        print(f"\n--- {err.category.value.upper()} ---")
        print(f"Severity: {err.severity.value}")
        print(f"Location: {err.location}")
        print(f"Explanation: {err.explain()}")
        print(f"Suggested fix: {err.suggest_fix()}")

    return errors


if __name__ == "__main__":
    demo()
