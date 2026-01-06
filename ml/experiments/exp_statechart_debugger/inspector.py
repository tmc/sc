"""
Inspector - Configuration and State Viewer for Debugging

Provides inspection capabilities:
1. Active configuration - Current active states
2. Context variables - Current variable bindings
3. State hierarchy - Nested state structure
4. Transition graph - Available transitions
5. History timeline - Execution history visualization
6. Diff view - Compare configurations

Think of it as the "Variables" and "Watch" panels in a traditional debugger.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import json


# =============================================================================
# Configuration View
# =============================================================================

@dataclass
class StateInfo:
    """Information about a state for display."""
    name: str
    is_active: bool
    is_initial: bool
    is_final: bool
    is_composite: bool
    depth: int = 0
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    entry_count: int = 0
    exit_count: int = 0
    total_time_in_state: float = 0.0


@dataclass
class TransitionInfo:
    """Information about a transition for display."""
    label: str
    source: str
    target: str
    event: str
    guard: Optional[str] = None
    is_enabled: bool = False
    fire_count: int = 0


@dataclass
class ConfigurationSnapshot:
    """Snapshot of complete configuration."""
    sequence: int
    timestamp: float
    active_states: List[str]
    context: Dict[str, Any]
    enabled_transitions: List[str]
    last_event: Optional[str] = None
    last_transition: Optional[str] = None


# =============================================================================
# Inspector Views
# =============================================================================

class ConfigurationInspector:
    """
    Inspect the current configuration of a statechart.
    """

    def __init__(self):
        self.state_info: Dict[str, StateInfo] = {}
        self.transition_info: Dict[str, TransitionInfo] = {}
        self.snapshots: List[ConfigurationSnapshot] = []
        self.max_snapshots = 100

    def register_state(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        is_composite: bool = False,
        parent: Optional[str] = None,
    ):
        """Register a state for inspection."""
        depth = 0
        if parent and parent in self.state_info:
            depth = self.state_info[parent].depth + 1

        self.state_info[name] = StateInfo(
            name=name,
            is_active=False,
            is_initial=is_initial,
            is_final=is_final,
            is_composite=is_composite,
            depth=depth,
            parent=parent,
        )

        if parent and parent in self.state_info:
            self.state_info[parent].children.append(name)

    def register_transition(
        self,
        label: str,
        source: str,
        target: str,
        event: str,
        guard: Optional[str] = None,
    ):
        """Register a transition for inspection."""
        self.transition_info[label] = TransitionInfo(
            label=label,
            source=source,
            target=target,
            event=event,
            guard=guard,
        )

    def update_active_states(self, active: Set[str]):
        """Update which states are active."""
        for name, info in self.state_info.items():
            was_active = info.is_active
            info.is_active = name in active

            if not was_active and info.is_active:
                info.entry_count += 1
            elif was_active and not info.is_active:
                info.exit_count += 1

    def update_enabled_transitions(
        self,
        active_states: Set[str],
        context: Dict[str, Any]
    ):
        """Update which transitions are enabled."""
        for label, info in self.transition_info.items():
            # Check if source is active
            if info.source not in active_states:
                info.is_enabled = False
                continue

            # Check guard if present
            if info.guard:
                try:
                    info.is_enabled = bool(
                        eval(info.guard, {"ctx": context, **context})
                    )
                except Exception:
                    info.is_enabled = False
            else:
                info.is_enabled = True

    def record_transition_fired(self, label: str):
        """Record that a transition fired."""
        if label in self.transition_info:
            self.transition_info[label].fire_count += 1

    def take_snapshot(
        self,
        sequence: int,
        timestamp: float,
        active_states: Set[str],
        context: Dict[str, Any],
        last_event: Optional[str] = None,
        last_transition: Optional[str] = None,
    ):
        """Take a snapshot of current configuration."""
        enabled = [
            label for label, info in self.transition_info.items()
            if info.is_enabled
        ]

        snapshot = ConfigurationSnapshot(
            sequence=sequence,
            timestamp=timestamp,
            active_states=list(active_states),
            context=context.copy(),
            enabled_transitions=enabled,
            last_event=last_event,
            last_transition=last_transition,
        )

        if len(self.snapshots) >= self.max_snapshots:
            self.snapshots = self.snapshots[1:]

        self.snapshots.append(snapshot)

    def get_active_states(self) -> List[StateInfo]:
        """Get information about active states."""
        return [info for info in self.state_info.values() if info.is_active]

    def get_enabled_transitions(self) -> List[TransitionInfo]:
        """Get information about enabled transitions."""
        return [info for info in self.transition_info.values() if info.is_enabled]

    def get_state_hierarchy(self) -> Dict[str, Any]:
        """Get state hierarchy as nested dict."""
        def build_tree(state_name: str) -> Dict[str, Any]:
            info = self.state_info.get(state_name)
            if not info:
                return {}

            node = {
                "name": state_name,
                "active": info.is_active,
                "initial": info.is_initial,
                "final": info.is_final,
            }

            if info.children:
                node["children"] = [
                    build_tree(child) for child in info.children
                ]

            return node

        # Find root states
        roots = [
            name for name, info in self.state_info.items()
            if info.parent is None
        ]

        return {"states": [build_tree(root) for root in roots]}


class ContextInspector:
    """
    Inspect context variables and their changes.
    """

    def __init__(self):
        self.current_context: Dict[str, Any] = {}
        self.history: List[Tuple[int, str, Any, Any]] = []  # (seq, var, old, new)
        self.watches: Set[str] = set()

    def update_context(self, sequence: int, new_context: Dict[str, Any]):
        """Update context and track changes."""
        for key in set(self.current_context.keys()) | set(new_context.keys()):
            old_val = self.current_context.get(key)
            new_val = new_context.get(key)

            if old_val != new_val:
                self.history.append((sequence, key, old_val, new_val))

        self.current_context = new_context.copy()

    def add_watch(self, variable: str):
        """Add a variable to watch."""
        self.watches.add(variable)

    def remove_watch(self, variable: str):
        """Remove a variable from watch."""
        self.watches.discard(variable)

    def get_watched_values(self) -> Dict[str, Any]:
        """Get current values of watched variables."""
        return {
            var: self.current_context.get(var)
            for var in self.watches
        }

    def get_variable_history(self, variable: str) -> List[Tuple[int, Any, Any]]:
        """Get change history for a variable."""
        return [
            (seq, old, new)
            for seq, var, old, new in self.history
            if var == variable
        ]

    def format_context(self, indent: int = 2) -> str:
        """Format context for display."""
        return json.dumps(self.current_context, indent=indent, default=str)


class HistoryInspector:
    """
    Inspect execution history and visualize timeline.
    """

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.max_entries = 1000

    def add_entry(
        self,
        sequence: int,
        timestamp: float,
        event: Optional[str],
        transition: Optional[str],
        source_states: List[str],
        target_states: List[str],
        context_changes: Dict[str, Tuple[Any, Any]],
    ):
        """Add an entry to history."""
        entry = {
            "sequence": sequence,
            "timestamp": timestamp,
            "event": event,
            "transition": transition,
            "source_states": source_states,
            "target_states": target_states,
            "context_changes": context_changes,
        }

        if len(self.entries) >= self.max_entries:
            self.entries = self.entries[1:]

        self.entries.append(entry)

    def get_timeline(self, start_seq: int = 0, end_seq: int = -1) -> List[Dict]:
        """Get timeline of execution."""
        if end_seq < 0:
            end_seq = len(self.entries)

        return self.entries[start_seq:end_seq]

    def get_state_timeline(self, state: str) -> List[Tuple[int, int, str]]:
        """
        Get timeline for a specific state.

        Returns: [(enter_seq, exit_seq, event)]
        """
        timeline = []
        enter_seq = None
        enter_event = None

        for entry in self.entries:
            if state in entry["target_states"] and state not in entry["source_states"]:
                # Entered state
                enter_seq = entry["sequence"]
                enter_event = entry["event"]
            elif state in entry["source_states"] and state not in entry["target_states"]:
                # Exited state
                if enter_seq is not None:
                    timeline.append((enter_seq, entry["sequence"], enter_event))
                    enter_seq = None

        # Handle still-active state
        if enter_seq is not None:
            timeline.append((enter_seq, -1, enter_event))

        return timeline

    def format_ascii_timeline(self, width: int = 60) -> str:
        """Format history as ASCII timeline."""
        if not self.entries:
            return "No history"

        lines = []
        lines.append("=" * width)
        lines.append("EXECUTION TIMELINE")
        lines.append("=" * width)

        for entry in self.entries[-20:]:  # Show last 20
            seq = entry["sequence"]
            event = entry["event"] or "-"
            trans = entry["transition"] or "-"
            src = ", ".join(entry["source_states"])
            tgt = ", ".join(entry["target_states"])

            lines.append(f"[{seq:03d}] {event:15} | {src} -> {tgt}")

        lines.append("=" * width)
        return "\n".join(lines)


class DiffInspector:
    """
    Compare configurations and show differences.
    """

    def __init__(self):
        pass

    def diff_states(
        self,
        states1: Set[str],
        states2: Set[str]
    ) -> Dict[str, List[str]]:
        """Diff two state sets."""
        return {
            "added": list(states2 - states1),
            "removed": list(states1 - states2),
            "unchanged": list(states1 & states2),
        }

    def diff_context(
        self,
        ctx1: Dict[str, Any],
        ctx2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Diff two contexts."""
        all_keys = set(ctx1.keys()) | set(ctx2.keys())

        diff = {
            "added": {},
            "removed": {},
            "changed": {},
            "unchanged": {},
        }

        for key in all_keys:
            if key not in ctx1:
                diff["added"][key] = ctx2[key]
            elif key not in ctx2:
                diff["removed"][key] = ctx1[key]
            elif ctx1[key] != ctx2[key]:
                diff["changed"][key] = {"old": ctx1[key], "new": ctx2[key]}
            else:
                diff["unchanged"][key] = ctx1[key]

        return diff

    def diff_snapshots(
        self,
        snap1: ConfigurationSnapshot,
        snap2: ConfigurationSnapshot
    ) -> Dict[str, Any]:
        """Diff two configuration snapshots."""
        return {
            "sequence": {"from": snap1.sequence, "to": snap2.sequence},
            "timestamp": {"from": snap1.timestamp, "to": snap2.timestamp},
            "states": self.diff_states(
                set(snap1.active_states),
                set(snap2.active_states)
            ),
            "context": self.diff_context(snap1.context, snap2.context),
            "enabled_transitions": {
                "added": list(set(snap2.enabled_transitions) - set(snap1.enabled_transitions)),
                "removed": list(set(snap1.enabled_transitions) - set(snap2.enabled_transitions)),
            },
        }

    def format_diff(self, diff: Dict[str, Any], indent: int = 0) -> str:
        """Format diff for display."""
        prefix = "  " * indent
        lines = []

        if "states" in diff:
            states = diff["states"]
            if states.get("added"):
                lines.append(f"{prefix}+ States: {', '.join(states['added'])}")
            if states.get("removed"):
                lines.append(f"{prefix}- States: {', '.join(states['removed'])}")

        if "context" in diff:
            ctx = diff["context"]
            for key, value in ctx.get("added", {}).items():
                lines.append(f"{prefix}+ {key} = {value}")
            for key, value in ctx.get("removed", {}).items():
                lines.append(f"{prefix}- {key} = {value}")
            for key, info in ctx.get("changed", {}).items():
                lines.append(f"{prefix}~ {key}: {info['old']} -> {info['new']}")

        return "\n".join(lines) if lines else f"{prefix}(no changes)"


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the inspector."""
    print("=" * 60)
    print("Configuration Inspector Demo")
    print("=" * 60)

    # Create inspectors
    config_inspector = ConfigurationInspector()
    context_inspector = ContextInspector()
    history_inspector = HistoryInspector()
    diff_inspector = DiffInspector()

    # Register states
    config_inspector.register_state("IDLE", is_initial=True)
    config_inspector.register_state("RUNNING")
    config_inspector.register_state("PROCESSING", parent="RUNNING", is_composite=True)
    config_inspector.register_state("WAITING", parent="RUNNING")
    config_inspector.register_state("STOPPED", is_final=True)

    # Register transitions
    config_inspector.register_transition("t1", "IDLE", "RUNNING", "start")
    config_inspector.register_transition("t2", "RUNNING", "STOPPED", "stop")
    config_inspector.register_transition("t3", "RUNNING", "RUNNING", "tick", "ctx.get('count', 0) < 10")

    # Simulate execution
    active = {"IDLE"}
    context = {"count": 0}

    print("\n--- Initial Configuration ---")
    config_inspector.update_active_states(active)
    config_inspector.update_enabled_transitions(active, context)

    print(f"Active states: {[s.name for s in config_inspector.get_active_states()]}")
    print(f"Enabled transitions: {[t.label for t in config_inspector.get_enabled_transitions()]}")

    # Take snapshot
    config_inspector.take_snapshot(0, 0.0, active, context)

    # Simulate transition
    print("\n--- After 'start' event ---")
    active = {"RUNNING", "PROCESSING"}
    context = {"count": 1, "status": "active"}

    config_inspector.update_active_states(active)
    config_inspector.update_enabled_transitions(active, context)
    config_inspector.record_transition_fired("t1")
    context_inspector.update_context(1, context)
    context_inspector.add_watch("count")

    history_inspector.add_entry(
        sequence=1,
        timestamp=0.1,
        event="start",
        transition="t1",
        source_states=["IDLE"],
        target_states=["RUNNING", "PROCESSING"],
        context_changes={"count": (0, 1), "status": (None, "active")},
    )

    print(f"Active states: {[s.name for s in config_inspector.get_active_states()]}")
    print(f"Enabled transitions: {[t.label for t in config_inspector.get_enabled_transitions()]}")
    print(f"Watched values: {context_inspector.get_watched_values()}")

    # Take another snapshot
    config_inspector.take_snapshot(1, 0.1, active, context, "start", "t1")

    # Show diff
    print("\n--- Configuration Diff ---")
    snap1 = config_inspector.snapshots[0]
    snap2 = config_inspector.snapshots[1]
    diff = diff_inspector.diff_snapshots(snap1, snap2)
    print(diff_inspector.format_diff(diff))

    # Show hierarchy
    print("\n--- State Hierarchy ---")
    hierarchy = config_inspector.get_state_hierarchy()
    print(json.dumps(hierarchy, indent=2))

    # Show timeline
    print("\n--- Execution Timeline ---")
    print(history_inspector.format_ascii_timeline())

    return config_inspector, context_inspector, history_inspector


if __name__ == "__main__":
    demo()
