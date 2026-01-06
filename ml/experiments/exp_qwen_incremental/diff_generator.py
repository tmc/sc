"""
Diff Generator for Statechart Incremental Updates

Generates and applies minimal diffs to statecharts.

DIFF OPERATIONS:
- ADD_STATE: Add a new state to the chart
- REMOVE_STATE: Remove an existing state
- MODIFY_STATE: Change state properties (type, initial, final, actions)
- ADD_TRANSITION: Add a new transition
- REMOVE_TRANSITION: Remove an existing transition
- MODIFY_TRANSITION: Change transition properties (event, guard, action)
- RENAME_STATE: Rename a state (updates all references)

OUTPUT FORMAT:
Compact JSON diff that can be applied incrementally.
"""

import json
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Union
from enum import Enum


class DiffOp(Enum):
    """Diff operation types."""
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    MODIFY_STATE = "modify_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    MODIFY_TRANSITION = "modify_transition"
    RENAME_STATE = "rename_state"
    SET_INITIAL = "set_initial"
    SET_PROPERTY = "set_property"


@dataclass
class DiffOperation:
    """A single diff operation."""
    op: DiffOp
    path: str  # JSON path to the element
    value: Optional[Any] = None  # New value for add/modify
    old_value: Optional[Any] = None  # Previous value (for undo)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = {"op": self.op.value, "path": self.path}
        if self.value is not None:
            d["value"] = self.value
        if self.old_value is not None:
            d["old"] = self.old_value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiffOperation":
        """Create from dictionary."""
        return cls(
            op=DiffOp(d["op"]),
            path=d["path"],
            value=d.get("value"),
            old_value=d.get("old"),
        )


@dataclass
class StatechartDiff:
    """A complete diff between two statecharts."""
    operations: List[DiffOperation] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatechartDiff":
        """Create from dictionary."""
        return cls(
            operations=[DiffOperation.from_dict(op) for op in d.get("operations", [])],
            description=d.get("description", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "StatechartDiff":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))

    def __len__(self) -> int:
        return len(self.operations)

    def is_empty(self) -> bool:
        return len(self.operations) == 0


class DiffGenerator:
    """
    Generates diffs between statecharts.

    Computes minimal set of operations to transform one chart into another.
    """

    def __init__(self):
        self._path_cache: Dict[str, str] = {}

    def compute_diff(
        self,
        original: Dict[str, Any],
        modified: Dict[str, Any],
        description: str = "",
    ) -> StatechartDiff:
        """
        Compute diff between original and modified statechart.

        Returns minimal set of operations.
        """
        diff = StatechartDiff(description=description)

        # Extract states from both charts
        orig_states = self._extract_states(original)
        mod_states = self._extract_states(modified)

        # Extract transitions
        orig_trans = self._extract_transitions(original)
        mod_trans = self._extract_transitions(modified)

        # Find state changes
        diff.operations.extend(self._diff_states(orig_states, mod_states))

        # Find transition changes
        diff.operations.extend(self._diff_transitions(orig_trans, mod_trans))

        return diff

    def _extract_states(self, chart: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract all states with their paths."""
        states = {}

        def traverse(state: Dict[str, Any], path: str):
            label = state.get("label", "")
            states[label] = {
                "path": path,
                "type": state.get("type", 1),
                "is_initial": state.get("is_initial", False),
                "is_final": state.get("is_final", False),
                "entry_action": state.get("entry_action"),
                "exit_action": state.get("exit_action"),
                "children": [c.get("label", "") for c in state.get("children", [])],
            }
            for i, child in enumerate(state.get("children", [])):
                traverse(child, f"{path}.children[{i}]")

        if "root_state" in chart:
            traverse(chart["root_state"], "root_state")

        return states

    def _extract_transitions(
        self, chart: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract transitions with keys for comparison."""
        transitions = []
        for i, t in enumerate(chart.get("transitions", [])):
            key = self._transition_key(t)
            transitions.append((key, {
                "index": i,
                "from": t.get("from", []),
                "to": t.get("to", []),
                "event": t.get("event", ""),
                "guard": t.get("guard"),
                "action": t.get("action"),
            }))
        return transitions

    def _transition_key(self, t: Dict[str, Any]) -> str:
        """Create a unique key for a transition."""
        from_states = ",".join(sorted(t.get("from", [])))
        to_states = ",".join(sorted(t.get("to", [])))
        event = t.get("event", "")
        return f"{from_states}->{to_states}@{event}"

    def _diff_states(
        self,
        orig: Dict[str, Dict[str, Any]],
        mod: Dict[str, Dict[str, Any]],
    ) -> List[DiffOperation]:
        """Compute state differences."""
        ops = []
        orig_labels = set(orig.keys())
        mod_labels = set(mod.keys())

        # Removed states
        for label in orig_labels - mod_labels:
            ops.append(DiffOperation(
                op=DiffOp.REMOVE_STATE,
                path=orig[label]["path"],
                old_value=orig[label],
            ))

        # Added states
        for label in mod_labels - orig_labels:
            ops.append(DiffOperation(
                op=DiffOp.ADD_STATE,
                path=mod[label]["path"],
                value=mod[label],
            ))

        # Modified states
        for label in orig_labels & mod_labels:
            orig_state = orig[label]
            mod_state = mod[label]

            # Check each property
            for prop in ["type", "is_initial", "is_final", "entry_action", "exit_action"]:
                if orig_state.get(prop) != mod_state.get(prop):
                    ops.append(DiffOperation(
                        op=DiffOp.MODIFY_STATE,
                        path=f"{mod_state['path']}.{prop}",
                        value=mod_state.get(prop),
                        old_value=orig_state.get(prop),
                    ))

        return ops

    def _diff_transitions(
        self,
        orig: List[Tuple[str, Dict[str, Any]]],
        mod: List[Tuple[str, Dict[str, Any]]],
    ) -> List[DiffOperation]:
        """Compute transition differences."""
        ops = []
        orig_map = {k: v for k, v in orig}
        mod_map = {k: v for k, v in mod}

        orig_keys = set(orig_map.keys())
        mod_keys = set(mod_map.keys())

        # Removed transitions
        for key in orig_keys - mod_keys:
            ops.append(DiffOperation(
                op=DiffOp.REMOVE_TRANSITION,
                path=f"transitions[{orig_map[key]['index']}]",
                old_value=orig_map[key],
            ))

        # Added transitions
        for key in mod_keys - orig_keys:
            ops.append(DiffOperation(
                op=DiffOp.ADD_TRANSITION,
                path="transitions",
                value=mod_map[key],
            ))

        # Modified transitions (same key but different properties)
        for key in orig_keys & mod_keys:
            orig_t = orig_map[key]
            mod_t = mod_map[key]

            for prop in ["guard", "action"]:
                if orig_t.get(prop) != mod_t.get(prop):
                    ops.append(DiffOperation(
                        op=DiffOp.MODIFY_TRANSITION,
                        path=f"transitions[{mod_t['index']}].{prop}",
                        value=mod_t.get(prop),
                        old_value=orig_t.get(prop),
                    ))

        return ops


class DiffApplier:
    """
    Applies diffs to statecharts.

    Supports both forward and reverse application.
    """

    def apply(
        self,
        chart: Dict[str, Any],
        diff: StatechartDiff,
        reverse: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply diff to chart.

        Args:
            chart: Original statechart
            diff: Diff to apply
            reverse: If True, apply in reverse (undo)

        Returns:
            Modified statechart
        """
        result = copy.deepcopy(chart)
        ops = list(reversed(diff.operations)) if reverse else diff.operations

        for op in ops:
            if reverse:
                result = self._apply_reverse(result, op)
            else:
                result = self._apply_forward(result, op)

        return result

    def _apply_forward(
        self,
        chart: Dict[str, Any],
        op: DiffOperation,
    ) -> Dict[str, Any]:
        """Apply operation forward."""
        if op.op == DiffOp.ADD_STATE:
            return self._add_state(chart, op.path, op.value)
        elif op.op == DiffOp.REMOVE_STATE:
            return self._remove_state(chart, op.path)
        elif op.op == DiffOp.MODIFY_STATE:
            return self._set_value(chart, op.path, op.value)
        elif op.op == DiffOp.ADD_TRANSITION:
            return self._add_transition(chart, op.value)
        elif op.op == DiffOp.REMOVE_TRANSITION:
            return self._remove_transition(chart, op.path)
        elif op.op == DiffOp.MODIFY_TRANSITION:
            return self._set_value(chart, op.path, op.value)
        elif op.op == DiffOp.RENAME_STATE:
            return self._rename_state(chart, op.old_value, op.value)
        elif op.op == DiffOp.SET_INITIAL:
            return self._set_value(chart, op.path, op.value)
        elif op.op == DiffOp.SET_PROPERTY:
            return self._set_value(chart, op.path, op.value)
        return chart

    def _apply_reverse(
        self,
        chart: Dict[str, Any],
        op: DiffOperation,
    ) -> Dict[str, Any]:
        """Apply operation in reverse (undo)."""
        if op.op == DiffOp.ADD_STATE:
            return self._remove_state(chart, op.path)
        elif op.op == DiffOp.REMOVE_STATE:
            return self._add_state(chart, op.path, op.old_value)
        elif op.op in (DiffOp.MODIFY_STATE, DiffOp.MODIFY_TRANSITION,
                       DiffOp.SET_INITIAL, DiffOp.SET_PROPERTY):
            return self._set_value(chart, op.path, op.old_value)
        elif op.op == DiffOp.ADD_TRANSITION:
            # Find and remove the transition
            return self._remove_transition_by_value(chart, op.value)
        elif op.op == DiffOp.REMOVE_TRANSITION:
            return self._add_transition(chart, op.old_value)
        elif op.op == DiffOp.RENAME_STATE:
            return self._rename_state(chart, op.value, op.old_value)
        return chart

    def _parse_path(self, path: str) -> List[Union[str, int]]:
        """Parse JSON path into components."""
        import re
        parts = []
        for part in re.split(r'\.|\[|\]', path):
            if part:
                try:
                    parts.append(int(part))
                except ValueError:
                    parts.append(part)
        return parts

    def _get_by_path(self, obj: Any, path: str) -> Any:
        """Get value at path."""
        parts = self._parse_path(path)
        for part in parts:
            if isinstance(part, int):
                obj = obj[part]
            else:
                obj = obj[part]
        return obj

    def _set_value(self, chart: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
        """Set value at path."""
        parts = self._parse_path(path)
        obj = chart
        for part in parts[:-1]:
            if isinstance(part, int):
                obj = obj[part]
            else:
                obj = obj[part]
        last = parts[-1]
        if isinstance(last, int):
            obj[last] = value
        else:
            obj[last] = value
        return chart

    def _add_state(self, chart: Dict[str, Any], path: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Add a state at the given path."""
        # Parse parent path and add to children
        parts = self._parse_path(path)
        # Navigate to parent
        if len(parts) >= 2 and parts[-2] == "children":
            parent_path = ".".join(str(p) for p in parts[:-2])
            parent = self._get_by_path(chart, parent_path) if parent_path else chart.get("root_state", {})
            if "children" not in parent:
                parent["children"] = []
            parent["children"].append({
                "label": state.get("label", f"new_state_{len(parent['children'])}"),
                "type": state.get("type", 1),
                "is_initial": state.get("is_initial", False),
            })
        return chart

    def _remove_state(self, chart: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Remove state at path."""
        parts = self._parse_path(path)
        if len(parts) >= 2:
            parent_parts = parts[:-2]
            if parent_parts:
                parent_path = ".".join(str(p) for p in parent_parts)
                parent = self._get_by_path(chart, parent_path)
            else:
                parent = chart.get("root_state", {})
            if "children" in parent and isinstance(parts[-1], int):
                del parent["children"][parts[-1]]
        return chart

    def _add_transition(self, chart: Dict[str, Any], trans: Dict[str, Any]) -> Dict[str, Any]:
        """Add a transition."""
        if "transitions" not in chart:
            chart["transitions"] = []
        chart["transitions"].append({
            "from": trans.get("from", []),
            "to": trans.get("to", []),
            "event": trans.get("event", ""),
            "guard": trans.get("guard"),
            "action": trans.get("action"),
        })
        return chart

    def _remove_transition(self, chart: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Remove transition at index."""
        parts = self._parse_path(path)
        if len(parts) >= 2 and parts[0] == "transitions":
            idx = parts[1]
            if "transitions" in chart and idx < len(chart["transitions"]):
                del chart["transitions"][idx]
        return chart

    def _remove_transition_by_value(self, chart: Dict[str, Any], trans: Dict[str, Any]) -> Dict[str, Any]:
        """Remove transition by matching values."""
        if "transitions" in chart:
            chart["transitions"] = [
                t for t in chart["transitions"]
                if not (t.get("from") == trans.get("from") and
                       t.get("to") == trans.get("to") and
                       t.get("event") == trans.get("event"))
            ]
        return chart

    def _rename_state(self, chart: Dict[str, Any], old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a state and update all references."""
        def rename_in_state(state: Dict[str, Any]):
            if state.get("label") == old_name:
                state["label"] = new_name
            for child in state.get("children", []):
                rename_in_state(child)

        if "root_state" in chart:
            rename_in_state(chart["root_state"])

        # Update transitions
        for t in chart.get("transitions", []):
            t["from"] = [new_name if s == old_name else s for s in t.get("from", [])]
            t["to"] = [new_name if s == old_name else s for s in t.get("to", [])]

        return chart


def diff_size(diff: StatechartDiff) -> int:
    """Compute the size of a diff in characters."""
    return len(diff.to_json(indent=None))


def chart_size(chart: Dict[str, Any]) -> int:
    """Compute the size of a chart in characters."""
    return len(json.dumps(chart, separators=(",", ":")))


def compression_ratio(original: Dict[str, Any], diff: StatechartDiff) -> float:
    """Compute compression ratio (diff size / chart size)."""
    orig_size = chart_size(original)
    diff_sz = diff_size(diff)
    return diff_sz / orig_size if orig_size > 0 else 1.0


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate diff generation and application."""
    print("=" * 60)
    print("Statechart Diff Generator")
    print("=" * 60)

    # Original chart
    original = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "idle", "type": 1, "is_initial": True},
                {"label": "running", "type": 1},
                {"label": "stopped", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["idle"], "to": ["running"], "event": "START"},
            {"from": ["running"], "to": ["stopped"], "event": "STOP"},
        ]
    }

    # Modified chart (added state, modified transition)
    modified = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "idle", "type": 1, "is_initial": True},
                {"label": "running", "type": 1},
                {"label": "paused", "type": 1},  # New state
                {"label": "stopped", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["idle"], "to": ["running"], "event": "START"},
            {"from": ["running"], "to": ["paused"], "event": "PAUSE"},  # New transition
            {"from": ["paused"], "to": ["running"], "event": "RESUME"},  # New transition
            {"from": ["running"], "to": ["stopped"], "event": "STOP"},
        ]
    }

    # Generate diff
    generator = DiffGenerator()
    diff = generator.compute_diff(original, modified, "Add pause functionality")

    print(f"\nOriginal chart size: {chart_size(original)} chars")
    print(f"Modified chart size: {chart_size(modified)} chars")
    print(f"Diff size: {diff_size(diff)} chars")
    print(f"Compression ratio: {compression_ratio(original, diff):.2%}")

    print(f"\nDiff operations ({len(diff)} ops):")
    for op in diff.operations:
        print(f"  {op.op.value}: {op.path}")

    # Apply diff
    applier = DiffApplier()
    result = applier.apply(original, diff)

    print(f"\nApplied diff successfully")
    print(f"Result states: {[c['label'] for c in result['root_state']['children']]}")
    print(f"Result transitions: {len(result['transitions'])}")

    return generator, diff


if __name__ == "__main__":
    demo()
