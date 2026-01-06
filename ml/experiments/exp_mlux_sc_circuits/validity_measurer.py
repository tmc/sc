"""
Validity Measurer for Statechart Outputs

Measures different aspects of statechart validity:
1. State name consistency - same state names used throughout
2. Transition validity - transitions reference existing states
3. Hierarchy validity - proper parent-child relationships
4. Structural validity - required fields present
5. Semantic validity - initial states, reachability

These metrics help identify which model components are responsible
for each aspect of valid statechart generation.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict


@dataclass
class ValidityMetrics:
    """Detailed validity metrics for a statechart."""
    # Overall
    is_valid: bool = False
    total_score: float = 0.0

    # Component scores (0-1)
    state_name_consistency: float = 0.0
    transition_validity: float = 0.0
    hierarchy_validity: float = 0.0
    structural_validity: float = 0.0
    semantic_validity: float = 0.0

    # Detailed issues
    issues: List[str] = field(default_factory=list)

    # Counts
    n_states: int = 0
    n_transitions: int = 0
    n_invalid_refs: int = 0
    n_missing_fields: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_score": self.total_score,
            "state_name_consistency": self.state_name_consistency,
            "transition_validity": self.transition_validity,
            "hierarchy_validity": self.hierarchy_validity,
            "structural_validity": self.structural_validity,
            "semantic_validity": self.semantic_validity,
            "n_states": self.n_states,
            "n_transitions": self.n_transitions,
            "n_issues": len(self.issues),
        }


class ValidityMeasurer:
    """
    Measures statechart validity with fine-grained metrics.

    Each metric captures a different aspect of validity, enabling
    identification of which model components affect which validity aspect.
    """

    # Required fields for each element type
    REQUIRED_STATE_FIELDS = {"label"}
    REQUIRED_TRANSITION_FIELDS = {"event"}
    REQUIRED_ROOT_FIELDS = {"root_state"}

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, any issue makes is_valid=False
        """
        self.strict = strict

    def measure(self, chart: Dict[str, Any]) -> ValidityMetrics:
        """
        Measure validity of a statechart.

        Returns comprehensive metrics breaking down validity by aspect.
        """
        metrics = ValidityMetrics()

        # Check structural validity first
        struct_score, struct_issues = self._check_structural(chart)
        metrics.structural_validity = struct_score
        metrics.issues.extend(struct_issues)

        if struct_score < 0.5:
            # Can't analyze further without basic structure
            metrics.total_score = struct_score * 0.3
            metrics.is_valid = False
            return metrics

        # Extract states and transitions
        all_states = self._collect_states(chart)
        all_transitions = chart.get("transitions", [])
        metrics.n_states = len(all_states)
        metrics.n_transitions = len(all_transitions)

        # State name consistency
        name_score, name_issues = self._check_state_names(all_states)
        metrics.state_name_consistency = name_score
        metrics.issues.extend(name_issues)

        # Transition validity
        trans_score, trans_issues, n_invalid = self._check_transitions(
            all_transitions, set(all_states.keys())
        )
        metrics.transition_validity = trans_score
        metrics.issues.extend(trans_issues)
        metrics.n_invalid_refs = n_invalid

        # Hierarchy validity
        hier_score, hier_issues = self._check_hierarchy(chart)
        metrics.hierarchy_validity = hier_score
        metrics.issues.extend(hier_issues)

        # Semantic validity
        sem_score, sem_issues = self._check_semantics(chart, all_states)
        metrics.semantic_validity = sem_score
        metrics.issues.extend(sem_issues)

        # Compute total score (weighted average)
        metrics.total_score = (
            0.15 * metrics.structural_validity +
            0.25 * metrics.state_name_consistency +
            0.25 * metrics.transition_validity +
            0.15 * metrics.hierarchy_validity +
            0.20 * metrics.semantic_validity
        )

        # Determine overall validity
        if self.strict:
            metrics.is_valid = len(metrics.issues) == 0
        else:
            metrics.is_valid = metrics.total_score >= 0.8

        return metrics

    def _check_structural(self, chart: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check basic structural requirements."""
        issues = []
        score = 1.0

        # Check root structure
        if not isinstance(chart, dict):
            return 0.0, ["Chart is not a dictionary"]

        if "root_state" not in chart:
            issues.append("Missing root_state")
            score -= 0.5

        root = chart.get("root_state", {})
        if not isinstance(root, dict):
            issues.append("root_state is not a dictionary")
            score -= 0.3

        if "label" not in root:
            issues.append("root_state missing label")
            score -= 0.2

        # Check transitions is a list
        trans = chart.get("transitions", [])
        if not isinstance(trans, list):
            issues.append("transitions is not a list")
            score -= 0.2

        return max(0.0, score), issues

    def _collect_states(
        self, chart: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect all states with their info."""
        states = {}

        def traverse(state: Dict[str, Any], parent: Optional[str] = None, depth: int = 0):
            label = state.get("label", f"unnamed_{len(states)}")
            states[label] = {
                "state": state,
                "parent": parent,
                "depth": depth,
                "children": [c.get("label", "") for c in state.get("children", [])],
            }
            for child in state.get("children", []):
                if isinstance(child, dict):
                    traverse(child, label, depth + 1)

        root = chart.get("root_state", {})
        if isinstance(root, dict):
            traverse(root)

        return states

    def _check_state_names(
        self, states: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """Check state name consistency."""
        issues = []

        if not states:
            return 0.0, ["No states found"]

        # Check for duplicate names (should be unique)
        seen_names: Dict[str, int] = defaultdict(int)
        for label in states:
            seen_names[label] += 1

        duplicates = [name for name, count in seen_names.items() if count > 1]
        if duplicates:
            issues.append(f"Duplicate state names: {duplicates}")

        # Check for invalid names (empty, special chars, etc)
        invalid_names = []
        for label in states:
            if not label or not isinstance(label, str):
                invalid_names.append(str(label))
            elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label) and label != "__root__":
                # Allow __root__ as special case
                if not label.startswith("__"):
                    invalid_names.append(label)

        if invalid_names:
            issues.append(f"Invalid state names: {invalid_names[:5]}")

        # Compute score
        n_total = len(states)
        n_problems = len(duplicates) + len(invalid_names)
        score = max(0.0, 1.0 - (n_problems / n_total)) if n_total > 0 else 0.0

        return score, issues

    def _check_transitions(
        self,
        transitions: List[Dict[str, Any]],
        valid_states: Set[str],
    ) -> Tuple[float, List[str], int]:
        """Check transition validity."""
        issues = []
        n_invalid_refs = 0

        if not transitions:
            # No transitions is valid (for simple charts)
            return 1.0, [], 0

        n_valid = 0
        for i, t in enumerate(transitions):
            if not isinstance(t, dict):
                issues.append(f"Transition {i} is not a dictionary")
                continue

            trans_valid = True

            # Check source states
            from_states = t.get("from", [])
            if not from_states:
                issues.append(f"Transition {i} has no source states")
                trans_valid = False
            else:
                for s in from_states:
                    if s not in valid_states:
                        n_invalid_refs += 1
                        trans_valid = False

            # Check target states
            to_states = t.get("to", [])
            if not to_states:
                issues.append(f"Transition {i} has no target states")
                trans_valid = False
            else:
                for s in to_states:
                    if s not in valid_states:
                        n_invalid_refs += 1
                        trans_valid = False

            # Check event (required)
            if "event" not in t or not t["event"]:
                issues.append(f"Transition {i} missing event")
                trans_valid = False

            if trans_valid:
                n_valid += 1

        if n_invalid_refs > 0:
            issues.append(f"Found {n_invalid_refs} invalid state references in transitions")

        score = n_valid / len(transitions) if transitions else 1.0
        return score, issues, n_invalid_refs

    def _check_hierarchy(self, chart: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check hierarchy validity."""
        issues = []

        root = chart.get("root_state", {})
        if not isinstance(root, dict):
            return 0.0, ["No valid root state"]

        def check_state(state: Dict[str, Any], path: str = "") -> int:
            """Check a state and its children. Returns number of issues."""
            n_issues = 0

            state_type = state.get("type", 1)
            children = state.get("children", [])

            # Compound states (type 2 or 3) should have children
            if state_type in (2, 3) and not children:
                issues.append(f"Compound state {state.get('label', path)} has no children")
                n_issues += 1

            # Basic states (type 1) should not have children
            if state_type == 1 and children:
                issues.append(f"Basic state {state.get('label', path)} has children")
                n_issues += 1

            # Check children
            for i, child in enumerate(children):
                if isinstance(child, dict):
                    n_issues += check_state(child, f"{path}.children[{i}]")
                else:
                    issues.append(f"Invalid child at {path}.children[{i}]")
                    n_issues += 1

            return n_issues

        n_issues = check_state(root, "root_state")
        n_states = len(self._collect_states(chart))

        score = max(0.0, 1.0 - (n_issues / n_states)) if n_states > 0 else 0.0
        return score, issues

    def _check_semantics(
        self,
        chart: Dict[str, Any],
        states: Dict[str, Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        """Check semantic validity (initial states, reachability, etc)."""
        issues = []
        n_checks = 0
        n_passed = 0

        # Check for initial state in compound states
        def check_initial(state: Dict[str, Any], path: str):
            nonlocal n_checks, n_passed

            children = state.get("children", [])
            state_type = state.get("type", 1)

            if state_type == 2 and children:  # OR state
                n_checks += 1
                has_initial = any(c.get("is_initial", False) for c in children if isinstance(c, dict))
                if has_initial:
                    n_passed += 1
                else:
                    issues.append(f"Compound state {state.get('label', path)} has no initial child")

            for i, child in enumerate(children):
                if isinstance(child, dict):
                    check_initial(child, f"{path}.children[{i}]")

        root = chart.get("root_state", {})
        if isinstance(root, dict):
            check_initial(root, "root_state")

        # Check for at least one transition (if there are multiple states)
        n_states = len(states)
        n_transitions = len(chart.get("transitions", []))

        if n_states > 1:
            n_checks += 1
            if n_transitions > 0:
                n_passed += 1
            else:
                issues.append("Multiple states but no transitions")

        # Compute score
        score = n_passed / n_checks if n_checks > 0 else 1.0
        return score, issues

    def measure_from_json(self, json_str: str) -> ValidityMetrics:
        """Measure validity from JSON string."""
        try:
            chart = json.loads(json_str)
            return self.measure(chart)
        except json.JSONDecodeError as e:
            metrics = ValidityMetrics()
            metrics.issues.append(f"Invalid JSON: {e}")
            metrics.structural_validity = 0.0
            metrics.total_score = 0.0
            return metrics

    def measure_batch(
        self, charts: List[Dict[str, Any]]
    ) -> Tuple[List[ValidityMetrics], Dict[str, float]]:
        """
        Measure validity for a batch of charts.

        Returns:
            (individual_metrics, aggregate_stats)
        """
        metrics_list = [self.measure(chart) for chart in charts]

        # Aggregate statistics
        n = len(metrics_list)
        if n == 0:
            return [], {}

        aggregate = {
            "valid_rate": sum(1 for m in metrics_list if m.is_valid) / n,
            "mean_total_score": sum(m.total_score for m in metrics_list) / n,
            "mean_state_name": sum(m.state_name_consistency for m in metrics_list) / n,
            "mean_transition": sum(m.transition_validity for m in metrics_list) / n,
            "mean_hierarchy": sum(m.hierarchy_validity for m in metrics_list) / n,
            "mean_structural": sum(m.structural_validity for m in metrics_list) / n,
            "mean_semantic": sum(m.semantic_validity for m in metrics_list) / n,
        }

        return metrics_list, aggregate


# =============================================================================
# Specific validity tests for circuit analysis
# =============================================================================

class ComponentValidityTests:
    """
    Specific tests for identifying which components affect which validity aspect.

    Each test returns a score (0-1) for a specific validity dimension.
    """

    @staticmethod
    def test_state_name_memory(chart: Dict[str, Any]) -> float:
        """
        Test if state names are remembered consistently.

        Checks if states referenced in transitions match defined states.
        This tests the model's ability to maintain state name memory.
        """
        states = set()

        def collect(state):
            if isinstance(state, dict):
                states.add(state.get("label", ""))
                for child in state.get("children", []):
                    collect(child)

        collect(chart.get("root_state", {}))

        if not states:
            return 0.0

        referenced = set()
        for t in chart.get("transitions", []):
            referenced.update(t.get("from", []))
            referenced.update(t.get("to", []))

        if not referenced:
            return 1.0  # No transitions = no consistency issues

        valid_refs = len(referenced & states)
        total_refs = len(referenced)

        return valid_refs / total_refs if total_refs > 0 else 1.0

    @staticmethod
    def test_hierarchy_depth(chart: Dict[str, Any]) -> float:
        """
        Test if hierarchy is maintained correctly.

        Checks parent-child relationships and type consistency.
        """
        issues = 0
        n_states = 0

        def check(state, parent_type=None):
            nonlocal issues, n_states
            if not isinstance(state, dict):
                return

            n_states += 1
            state_type = state.get("type", 1)
            children = state.get("children", [])

            # Compound state should have children
            if state_type in (2, 3) and not children:
                issues += 1

            # Basic state should not have children
            if state_type == 1 and children:
                issues += 1

            for child in children:
                check(child, state_type)

        check(chart.get("root_state", {}))

        return max(0.0, 1.0 - issues / n_states) if n_states > 0 else 0.0

    @staticmethod
    def test_transition_syntax(chart: Dict[str, Any]) -> float:
        """
        Test if transitions have correct syntax.

        Checks for required fields and proper formatting.
        """
        transitions = chart.get("transitions", [])
        if not transitions:
            return 1.0

        valid = 0
        for t in transitions:
            if not isinstance(t, dict):
                continue

            has_from = "from" in t and isinstance(t["from"], list) and t["from"]
            has_to = "to" in t and isinstance(t["to"], list) and t["to"]
            has_event = "event" in t and t["event"]

            if has_from and has_to and has_event:
                valid += 1

        return valid / len(transitions)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate validity measurement."""
    print("=" * 60)
    print("Statechart Validity Measurer")
    print("=" * 60)

    measurer = ValidityMeasurer()

    # Valid chart
    valid_chart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "idle", "type": 1, "is_initial": True},
                {"label": "running", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["idle"], "to": ["running"], "event": "START"},
            {"from": ["running"], "to": ["idle"], "event": "STOP"},
        ]
    }

    print("\n1. Valid chart:")
    metrics = measurer.measure(valid_chart)
    print(f"   Total score: {metrics.total_score:.2%}")
    print(f"   Is valid: {metrics.is_valid}")
    print(f"   Issues: {len(metrics.issues)}")

    # Invalid chart (bad transition references)
    invalid_chart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "idle", "type": 1, "is_initial": True},
                {"label": "running", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["idle"], "to": ["TYPO_running"], "event": "START"},  # Typo
            {"from": ["nonexistent"], "to": ["idle"], "event": "STOP"},  # Bad ref
        ]
    }

    print("\n2. Invalid chart (bad refs):")
    metrics = measurer.measure(invalid_chart)
    print(f"   Total score: {metrics.total_score:.2%}")
    print(f"   Is valid: {metrics.is_valid}")
    print(f"   Transition validity: {metrics.transition_validity:.2%}")
    print(f"   Invalid refs: {metrics.n_invalid_refs}")

    # Missing structure
    incomplete_chart = {
        "transitions": [
            {"from": ["a"], "to": ["b"], "event": "X"}
        ]
    }

    print("\n3. Missing structure:")
    metrics = measurer.measure(incomplete_chart)
    print(f"   Total score: {metrics.total_score:.2%}")
    print(f"   Structural validity: {metrics.structural_validity:.2%}")

    return measurer


if __name__ == "__main__":
    demo()
