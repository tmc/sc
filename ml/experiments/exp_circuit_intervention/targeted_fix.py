"""
Targeted Fix for Statechart Errors

Applies targeted circuit interventions to fix specific error types.

ERROR-TO-CIRCUIT MAPPING:
- Invalid transition refs -> TRANSITION_VALIDITY (L8-14)
- Missing hierarchy -> HIERARCHY (L0-6)
- Malformed structure -> STRUCTURAL (L0-6)
- Duplicate state names -> STATE_NAME_MEMORY (L12-18)

STRATEGY:
1. Detect error type in generated output
2. Select appropriate circuit to amplify
3. Re-generate with intervention
4. Validate improvement
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum

from .circuit_amplifier import (
    CircuitAmplifier,
    CircuitType,
    CircuitSpec,
    AmplificationType,
    AmplificationConfig,
    transition_fix_config,
    hierarchy_boost_config,
    structural_fix_config,
)

# Import validity measurer from sibling experiment
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
try:
    from exp_mlux_sc_circuits.validity_measurer import ValidityMeasurer, ValidityMetrics
except ImportError:
    ValidityMeasurer = None
    ValidityMetrics = None


class ErrorType(Enum):
    """Types of statechart generation errors."""
    INVALID_TRANSITION_REF = "invalid_transition_ref"
    MISSING_INITIAL_STATE = "missing_initial_state"
    HIERARCHY_ERROR = "hierarchy_error"
    STRUCTURAL_ERROR = "structural_error"
    DUPLICATE_STATE = "duplicate_state"
    MISSING_EVENT = "missing_event"
    INVALID_JSON = "invalid_json"
    UNKNOWN = "unknown"


@dataclass
class ErrorDiagnosis:
    """Diagnosis of errors in a statechart."""
    error_types: List[ErrorType]
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_circuits: List[CircuitType] = field(default_factory=list)
    severity: float = 0.0  # 0-1, higher = worse

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_types": [e.value for e in self.error_types],
            "details": self.details,
            "recommended_circuits": [c.value for c in self.recommended_circuits],
            "severity": self.severity,
        }


@dataclass
class FixResult:
    """Result of applying a targeted fix."""
    original_chart: Dict[str, Any]
    fixed_chart: Optional[Dict[str, Any]]
    diagnosis: ErrorDiagnosis
    interventions_applied: List[CircuitType]
    attempts: int
    success: bool
    validity_before: float
    validity_after: float

    @property
    def improvement(self) -> float:
        return self.validity_after - self.validity_before


class ErrorDiagnoser:
    """
    Diagnoses errors in statechart generation.

    Maps errors to circuits that can fix them.
    """

    # Error-to-circuit mapping
    ERROR_CIRCUIT_MAP = {
        ErrorType.INVALID_TRANSITION_REF: [CircuitType.TRANSITION_VALIDITY],
        ErrorType.MISSING_INITIAL_STATE: [CircuitType.HIERARCHY, CircuitType.STRUCTURAL],
        ErrorType.HIERARCHY_ERROR: [CircuitType.HIERARCHY],
        ErrorType.STRUCTURAL_ERROR: [CircuitType.STRUCTURAL],
        ErrorType.DUPLICATE_STATE: [CircuitType.TRANSITION_VALIDITY],  # Name memory
        ErrorType.MISSING_EVENT: [CircuitType.TRANSITION_VALIDITY],
        ErrorType.INVALID_JSON: [CircuitType.STRUCTURAL],
    }

    def diagnose(self, chart_or_json: Any) -> ErrorDiagnosis:
        """
        Diagnose errors in a statechart.

        Args:
            chart_or_json: Dict or JSON string

        Returns:
            ErrorDiagnosis with detected issues
        """
        # Parse if string
        if isinstance(chart_or_json, str):
            try:
                chart = json.loads(chart_or_json)
            except json.JSONDecodeError:
                return ErrorDiagnosis(
                    error_types=[ErrorType.INVALID_JSON],
                    details={"parse_error": "Invalid JSON"},
                    recommended_circuits=[CircuitType.STRUCTURAL],
                    severity=1.0,
                )
        else:
            chart = chart_or_json

        errors = []
        details = {}
        severity = 0.0

        # Check structural issues
        if "root_state" not in chart:
            errors.append(ErrorType.STRUCTURAL_ERROR)
            details["missing_root"] = True
            severity += 0.3

        root = chart.get("root_state", {})

        # Collect all states
        all_states = set()
        def collect_states(state):
            if isinstance(state, dict):
                all_states.add(state.get("label", ""))
                for child in state.get("children", []):
                    collect_states(child)
        collect_states(root)

        # Check for duplicate states
        seen = set()
        duplicates = set()
        def find_duplicates(state):
            if isinstance(state, dict):
                label = state.get("label", "")
                if label in seen:
                    duplicates.add(label)
                seen.add(label)
                for child in state.get("children", []):
                    find_duplicates(child)
        find_duplicates(root)

        if duplicates:
            errors.append(ErrorType.DUPLICATE_STATE)
            details["duplicates"] = list(duplicates)
            severity += 0.2

        # Check transitions
        transitions = chart.get("transitions", [])
        invalid_refs = []
        missing_events = []

        for i, t in enumerate(transitions):
            if not isinstance(t, dict):
                continue

            # Check state references
            for s in t.get("from", []) + t.get("to", []):
                if s not in all_states:
                    invalid_refs.append((i, s))

            # Check event
            if not t.get("event"):
                missing_events.append(i)

        if invalid_refs:
            errors.append(ErrorType.INVALID_TRANSITION_REF)
            details["invalid_refs"] = invalid_refs
            severity += 0.3

        if missing_events:
            errors.append(ErrorType.MISSING_EVENT)
            details["missing_events"] = missing_events
            severity += 0.1

        # Check hierarchy
        def check_hierarchy(state, depth=0):
            issues = []
            if not isinstance(state, dict):
                return issues

            state_type = state.get("type", 1)
            children = state.get("children", [])

            # Compound without children
            if state_type in (2, 3) and not children:
                issues.append(("empty_compound", state.get("label", "")))

            # Check for initial state in compound
            if state_type == 2 and children:
                has_initial = any(
                    c.get("is_initial", False) for c in children
                    if isinstance(c, dict)
                )
                if not has_initial:
                    issues.append(("missing_initial", state.get("label", "")))

            for child in children:
                issues.extend(check_hierarchy(child, depth + 1))

            return issues

        hier_issues = check_hierarchy(root)
        if hier_issues:
            errors.append(ErrorType.HIERARCHY_ERROR)
            details["hierarchy_issues"] = hier_issues
            severity += 0.2

        # Determine recommended circuits
        recommended = set()
        for error in errors:
            circuits = self.ERROR_CIRCUIT_MAP.get(error, [])
            recommended.update(circuits)

        if not errors:
            errors = [ErrorType.UNKNOWN]

        return ErrorDiagnosis(
            error_types=errors,
            details=details,
            recommended_circuits=list(recommended),
            severity=min(1.0, severity),
        )


class TargetedFixer:
    """
    Applies targeted circuit interventions to fix errors.
    """

    def __init__(
        self,
        amplifier: Optional[CircuitAmplifier] = None,
        measurer: Optional[Any] = None,
        max_attempts: int = 3,
        verbose: bool = True,
    ):
        self.amplifier = amplifier or CircuitAmplifier(verbose=False)
        self.measurer = measurer
        self.diagnoser = ErrorDiagnoser()
        self.max_attempts = max_attempts
        self.verbose = verbose

        # Try to import measurer
        if self.measurer is None and ValidityMeasurer is not None:
            self.measurer = ValidityMeasurer()

    def get_config_for_circuit(
        self,
        circuit_type: CircuitType,
        strength: float = 1.5,
    ) -> AmplificationConfig:
        """Get amplification config for a circuit type."""
        if circuit_type == CircuitType.TRANSITION_VALIDITY:
            return transition_fix_config(strength)
        elif circuit_type == CircuitType.HIERARCHY:
            return hierarchy_boost_config(strength)
        elif circuit_type == CircuitType.STRUCTURAL:
            return structural_fix_config(strength)
        else:
            # Default
            return AmplificationConfig(
                circuit=CircuitSpec(
                    circuit_type=circuit_type,
                    layers=list(range(8, 15)),
                    components=["attention", "mlp"],
                ),
                amp_type=AmplificationType.SCALE,
                strength=strength,
            )

    def fix(
        self,
        prompt: str,
        original_output: str,
    ) -> FixResult:
        """
        Attempt to fix a problematic generation.

        Args:
            prompt: Original generation prompt
            original_output: Problematic output

        Returns:
            FixResult with fixed chart (if successful)
        """
        # Diagnose errors
        diagnosis = self.diagnoser.diagnose(original_output)

        if self.verbose:
            print(f"Diagnosis: {[e.value for e in diagnosis.error_types]}")
            print(f"Recommended circuits: {[c.value for c in diagnosis.recommended_circuits]}")

        # Parse original for comparison
        try:
            original_chart = json.loads(original_output)
        except json.JSONDecodeError:
            original_chart = {}

        # Compute original validity
        validity_before = 0.0
        if self.measurer and original_chart:
            metrics = self.measurer.measure(original_chart)
            validity_before = metrics.total_score

        # Try interventions
        interventions_applied = []
        fixed_chart = None
        best_validity = validity_before

        for attempt in range(self.max_attempts):
            for circuit_type in diagnosis.recommended_circuits:
                # Increase strength with each attempt
                strength = 1.2 + 0.2 * attempt

                config = self.get_config_for_circuit(circuit_type, strength)
                output = self.amplifier.amplify(prompt, config)

                try:
                    chart = json.loads(output)
                except json.JSONDecodeError:
                    continue

                # Check validity
                if self.measurer:
                    metrics = self.measurer.measure(chart)
                    validity = metrics.total_score
                else:
                    # Basic check
                    validity = self._basic_validity(chart)

                if validity > best_validity:
                    best_validity = validity
                    fixed_chart = chart
                    interventions_applied.append(circuit_type)

                    if self.verbose:
                        print(f"  Improved with {circuit_type.value}: {validity:.2%}")

                # Stop if we reach high validity
                if validity >= 0.95:
                    break

            if best_validity >= 0.95:
                break

        success = best_validity > validity_before + 0.1 or best_validity >= 0.9

        return FixResult(
            original_chart=original_chart,
            fixed_chart=fixed_chart or original_chart,
            diagnosis=diagnosis,
            interventions_applied=interventions_applied,
            attempts=attempt + 1,
            success=success,
            validity_before=validity_before,
            validity_after=best_validity,
        )

    def _basic_validity(self, chart: Dict[str, Any]) -> float:
        """Basic validity check without full measurer."""
        score = 0.0

        if "root_state" in chart:
            score += 0.3
            root = chart["root_state"]

            if "label" in root:
                score += 0.1

            children = root.get("children", [])
            if children:
                score += 0.2

            # Check for initial
            if any(c.get("is_initial", False) for c in children if isinstance(c, dict)):
                score += 0.1

        # Check transitions
        transitions = chart.get("transitions", [])
        if transitions:
            score += 0.2

            # Check references
            all_states = set()
            def collect(state):
                if isinstance(state, dict):
                    all_states.add(state.get("label", ""))
                    for c in state.get("children", []):
                        collect(c)
            collect(chart.get("root_state", {}))

            valid_trans = 0
            for t in transitions:
                refs = t.get("from", []) + t.get("to", [])
                if all(r in all_states for r in refs):
                    valid_trans += 1

            if transitions:
                score += 0.1 * (valid_trans / len(transitions))

        return min(1.0, score)


# =============================================================================
# Convenience functions
# =============================================================================

def fix_transitions(prompt: str, output: str) -> FixResult:
    """Fix transition validity issues."""
    fixer = TargetedFixer(verbose=False)
    # Override diagnosis to focus on transitions
    result = fixer.fix(prompt, output)
    return result


def fix_hierarchy(prompt: str, output: str) -> FixResult:
    """Fix hierarchy issues."""
    fixer = TargetedFixer(verbose=False)
    result = fixer.fix(prompt, output)
    return result


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate targeted fixing."""
    print("=" * 60)
    print("Targeted Fix for Statechart Errors")
    print("=" * 60)

    # Example with invalid transition references
    broken_chart = json.dumps({
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "s0", "type": 1, "is_initial": True},
                {"label": "s1", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["s0"], "to": ["TYPO_s1"], "event": "E1"},  # Bad ref
            {"from": ["nonexistent"], "to": ["s0"], "event": "E2"},  # Bad ref
        ]
    })

    diagnoser = ErrorDiagnoser()
    diagnosis = diagnoser.diagnose(broken_chart)

    print("\n1. Error Diagnosis:")
    print(f"   Errors: {[e.value for e in diagnosis.error_types]}")
    print(f"   Details: {diagnosis.details}")
    print(f"   Recommended: {[c.value for c in diagnosis.recommended_circuits]}")
    print(f"   Severity: {diagnosis.severity:.2%}")

    # Apply fix
    print("\n2. Applying Targeted Fix:")
    fixer = TargetedFixer(verbose=True)
    result = fixer.fix(
        prompt="Generate a state machine:",
        original_output=broken_chart,
    )

    print(f"\n3. Fix Result:")
    print(f"   Success: {result.success}")
    print(f"   Attempts: {result.attempts}")
    print(f"   Validity: {result.validity_before:.2%} -> {result.validity_after:.2%}")
    print(f"   Improvement: {result.improvement:+.2%}")
    print(f"   Interventions: {[c.value for c in result.interventions_applied]}")

    if result.fixed_chart:
        print(f"\n4. Fixed Chart:")
        print(f"   States: {len(result.fixed_chart.get('root_state', {}).get('children', []))}")
        print(f"   Transitions: {len(result.fixed_chart.get('transitions', []))}")

    return result


if __name__ == "__main__":
    demo()
