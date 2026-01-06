#!/usr/bin/env python3
"""
Change Detector: Programmatically detect changes between statecharts.

Parses statecharts and identifies:
- Added/removed/modified states
- Added/removed/modified transitions
- Breaking vs compatible changes
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto


class ChangeType(Enum):
    """Type of change."""
    ADDED = auto()
    REMOVED = auto()
    MODIFIED = auto()
    RENAMED = auto()


class Compatibility(Enum):
    """Change compatibility level."""
    COMPATIBLE = auto()      # Safe to upgrade
    BREAKING = auto()        # May break existing machines
    MIGRATION_REQUIRED = auto()  # Needs migration plan


@dataclass
class StateChange:
    """Change to a state."""
    change_type: ChangeType
    state_label: str
    old_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    details: str = ""

    def __str__(self):
        if self.change_type == ChangeType.ADDED:
            return f"+ State '{self.state_label}' added"
        elif self.change_type == ChangeType.REMOVED:
            return f"- State '{self.state_label}' removed"
        elif self.change_type == ChangeType.MODIFIED:
            return f"~ State '{self.state_label}' modified: {self.details}"
        elif self.change_type == ChangeType.RENAMED:
            return f"→ State renamed: {self.old_state} -> {self.new_state}"
        return f"? State '{self.state_label}' changed"


@dataclass
class TransitionChange:
    """Change to a transition."""
    change_type: ChangeType
    from_states: List[str]
    to_states: List[str]
    event: str
    old_transition: Optional[Dict[str, Any]] = None
    new_transition: Optional[Dict[str, Any]] = None
    details: str = ""

    def __str__(self):
        trans_str = f"{self.from_states} -> {self.to_states} [{self.event}]"
        if self.change_type == ChangeType.ADDED:
            return f"+ Transition {trans_str} added"
        elif self.change_type == ChangeType.REMOVED:
            return f"- Transition {trans_str} removed"
        elif self.change_type == ChangeType.MODIFIED:
            return f"~ Transition {trans_str} modified: {self.details}"
        return f"? Transition {trans_str} changed"


@dataclass
class DiffResult:
    """Complete diff between two statecharts."""
    before_chart: Dict[str, Any]
    after_chart: Dict[str, Any]
    state_changes: List[StateChange] = field(default_factory=list)
    transition_changes: List[TransitionChange] = field(default_factory=list)
    compatibility: Compatibility = Compatibility.COMPATIBLE
    breaking_reasons: List[str] = field(default_factory=list)

    @property
    def states_added(self) -> List[StateChange]:
        return [c for c in self.state_changes if c.change_type == ChangeType.ADDED]

    @property
    def states_removed(self) -> List[StateChange]:
        return [c for c in self.state_changes if c.change_type == ChangeType.REMOVED]

    @property
    def states_modified(self) -> List[StateChange]:
        return [c for c in self.state_changes if c.change_type == ChangeType.MODIFIED]

    @property
    def transitions_added(self) -> List[TransitionChange]:
        return [c for c in self.transition_changes if c.change_type == ChangeType.ADDED]

    @property
    def transitions_removed(self) -> List[TransitionChange]:
        return [c for c in self.transition_changes if c.change_type == ChangeType.REMOVED]

    @property
    def is_empty(self) -> bool:
        return len(self.state_changes) == 0 and len(self.transition_changes) == 0

    def summary(self) -> str:
        """Generate summary string."""
        parts = []
        if self.states_added:
            parts.append(f"+{len(self.states_added)} states")
        if self.states_removed:
            parts.append(f"-{len(self.states_removed)} states")
        if self.states_modified:
            parts.append(f"~{len(self.states_modified)} states")
        if self.transitions_added:
            parts.append(f"+{len(self.transitions_added)} transitions")
        if self.transitions_removed:
            parts.append(f"-{len(self.transitions_removed)} transitions")
        return ", ".join(parts) if parts else "No changes"


class ChangeDetector:
    """
    Detects changes between two statecharts.

    Uses both programmatic comparison and sc diff CLI.
    """

    def __init__(self, sc_path: str = "./sc"):
        self.sc_path = sc_path

    def diff(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> DiffResult:
        """
        Compare two statecharts and identify changes.

        Args:
            before: Original statechart
            after: Modified statechart

        Returns:
            DiffResult with all changes
        """
        result = DiffResult(before_chart=before, after_chart=after)

        # Collect states from both charts
        before_states = self._collect_states(before)
        after_states = self._collect_states(after)

        # Detect state changes
        self._detect_state_changes(before_states, after_states, result)

        # Collect transitions
        before_trans = self._normalize_transitions(before.get("transitions", []))
        after_trans = self._normalize_transitions(after.get("transitions", []))

        # Detect transition changes
        self._detect_transition_changes(before_trans, after_trans, result)

        # Determine compatibility
        self._assess_compatibility(result)

        return result

    def _collect_states(self, chart: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Collect all states with their full info."""
        states = {}

        def collect(state: Dict[str, Any], parent: str = ""):
            label = state.get("label", "")
            if label and label != "__root__":
                states[label] = {
                    "label": label,
                    "type": state.get("type", 1),
                    "is_initial": state.get("is_initial", False),
                    "parent": parent,
                    "has_children": len(state.get("children", [])) > 0,
                }
            for child in state.get("children", []):
                collect(child, label)

        if "root_state" in chart:
            collect(chart["root_state"])

        return states

    def _normalize_transitions(
        self,
        transitions: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize transitions for comparison."""
        result = {}
        for t in transitions:
            from_states = tuple(sorted(t.get("from", [])))
            to_states = tuple(sorted(t.get("to", [])))
            event = t.get("event", "")
            key = (from_states, to_states, event)
            result[key] = t
        return result

    def _detect_state_changes(
        self,
        before: Dict[str, Dict[str, Any]],
        after: Dict[str, Dict[str, Any]],
        result: DiffResult,
    ):
        """Detect added, removed, and modified states."""
        before_labels = set(before.keys())
        after_labels = set(after.keys())

        # Added states
        for label in after_labels - before_labels:
            result.state_changes.append(StateChange(
                change_type=ChangeType.ADDED,
                state_label=label,
                new_state=after[label],
            ))

        # Removed states
        for label in before_labels - after_labels:
            result.state_changes.append(StateChange(
                change_type=ChangeType.REMOVED,
                state_label=label,
                old_state=before[label],
            ))

        # Modified states
        for label in before_labels & after_labels:
            old = before[label]
            new = after[label]

            changes = []
            if old.get("type") != new.get("type"):
                changes.append(f"type: {old.get('type')} -> {new.get('type')}")
            if old.get("is_initial") != new.get("is_initial"):
                changes.append(f"is_initial: {old.get('is_initial')} -> {new.get('is_initial')}")
            if old.get("parent") != new.get("parent"):
                changes.append(f"parent: {old.get('parent')} -> {new.get('parent')}")

            if changes:
                result.state_changes.append(StateChange(
                    change_type=ChangeType.MODIFIED,
                    state_label=label,
                    old_state=old,
                    new_state=new,
                    details="; ".join(changes),
                ))

    def _detect_transition_changes(
        self,
        before: Dict[tuple, Dict[str, Any]],
        after: Dict[tuple, Dict[str, Any]],
        result: DiffResult,
    ):
        """Detect added, removed transitions."""
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        # Added transitions
        for key in after_keys - before_keys:
            from_states, to_states, event = key
            result.transition_changes.append(TransitionChange(
                change_type=ChangeType.ADDED,
                from_states=list(from_states),
                to_states=list(to_states),
                event=event,
                new_transition=after[key],
            ))

        # Removed transitions
        for key in before_keys - after_keys:
            from_states, to_states, event = key
            result.transition_changes.append(TransitionChange(
                change_type=ChangeType.REMOVED,
                from_states=list(from_states),
                to_states=list(to_states),
                event=event,
                old_transition=before[key],
            ))

    def _assess_compatibility(self, result: DiffResult):
        """Determine if changes are breaking."""
        # Removals are breaking
        if result.states_removed:
            result.compatibility = Compatibility.BREAKING
            for change in result.states_removed:
                result.breaking_reasons.append(f"State removed: {change.state_label}")

        if result.transitions_removed:
            result.compatibility = Compatibility.BREAKING
            for change in result.transitions_removed:
                result.breaking_reasons.append(
                    f"Transition removed: {change.from_states}->{change.to_states}[{change.event}]"
                )

        # Initial state changes are breaking
        for change in result.states_modified:
            if "is_initial" in change.details:
                result.compatibility = Compatibility.BREAKING
                result.breaking_reasons.append(f"Initial state changed: {change.state_label}")

    def diff_via_cli(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> str:
        """Get diff via sc CLI for comparison."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump(before, f1)
            f1_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump(after, f2)
            f2_path = f2.name

        try:
            result = subprocess.run(
                [self.sc_path, "diff", f1_path, f2_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout + result.stderr
        finally:
            os.unlink(f1_path)
            os.unlink(f2_path)


def detect_changes(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> DiffResult:
    """Convenience function to detect changes."""
    detector = ChangeDetector()
    return detector.diff(before, after)


def demo():
    """Demo change detection."""
    print("=" * 60)
    print("CHANGE DETECTOR DEMO")
    print("=" * 60)

    detector = ChangeDetector()

    # Test case 1: Added state
    before1 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "is_initial": True},
            ]
        }
    }
    after1 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "is_initial": True},
                {"label": "B"},
            ]
        }
    }

    print("\n--- Test 1: Added state ---")
    result1 = detector.diff(before1, after1)
    print(f"Summary: {result1.summary()}")
    print(f"Compatibility: {result1.compatibility.name}")
    for change in result1.state_changes:
        print(f"  {change}")

    # Test case 2: Removed state and transition
    before2 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "is_initial": True},
                {"label": "B"},
                {"label": "C"},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"},
            {"from": ["B"], "to": ["C"], "event": "NEXT"},
        ]
    }
    after2 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "is_initial": True},
                {"label": "C"},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["C"], "event": "SKIP"},
        ]
    }

    print("\n--- Test 2: Removed state and transitions ---")
    result2 = detector.diff(before2, after2)
    print(f"Summary: {result2.summary()}")
    print(f"Compatibility: {result2.compatibility.name}")
    for change in result2.state_changes:
        print(f"  {change}")
    for change in result2.transition_changes:
        print(f"  {change}")
    if result2.breaking_reasons:
        print("Breaking reasons:")
        for reason in result2.breaking_reasons:
            print(f"  ! {reason}")

    # Test case 3: No changes
    print("\n--- Test 3: No changes ---")
    result3 = detector.diff(before1, before1)
    print(f"Summary: {result3.summary()}")
    print(f"Is empty: {result3.is_empty}")

    print("\n" + "=" * 60)
    print("CHANGE DETECTOR DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
