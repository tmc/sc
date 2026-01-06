"""
Schema Diff: Detect Changes Between Statechart Versions

Implements diffing algorithms to detect:
- State changes (add, remove, rename, move, modify)
- Transition changes
- Event changes
- Breaking vs non-breaking changes

Based on proto/statecharts/v1/evolution.proto ChartDiff semantics.

NO HARDCODING: Patterns are discovered through structural analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import copy


# =============================================================================
# CHANGE TYPES (mirrors proto/statecharts/v1/evolution.proto)
# =============================================================================

class ChangeType(Enum):
    """Type of schema change."""
    UNSPECIFIED = 0
    ADDED = 1       # Element exists in new but not old
    REMOVED = 2     # Element exists in old but not new
    MODIFIED = 3    # Element exists in both with different attributes
    RENAMED = 4     # Identity preserved, label changed
    MOVED = 5       # Position in hierarchy changed


class BreakingChangeType(Enum):
    """Specific breaking change patterns."""
    UNSPECIFIED = 0
    STATE_REMOVED = 1
    STATE_TYPE_CHANGED = 2
    INITIAL_STATE_CHANGED = 3
    TRANSITION_REMOVED = 4
    TRANSITION_SOURCE_CHANGED = 5
    GUARD_ADDED = 6
    EVENT_REMOVED = 7
    EVENT_PAYLOAD_CHANGED = 8


class CompatibilityLevel(Enum):
    """Overall compatibility classification."""
    FULL = 0        # No changes required
    BACKWARD = 1    # Old clients work with new version
    FORWARD = 2     # New clients work with old version
    BREAKING = 3    # Requires migration


class MigrationRequirement(Enum):
    """Complexity of required migration."""
    NONE = 0
    STATE_MAPPING = 1
    MANUAL = 2


# =============================================================================
# STATECHART REPRESENTATION
# =============================================================================

@dataclass
class State:
    """Represents a state in a statechart."""
    label: str
    state_type: str = "BASIC"  # BASIC, NORMAL, PARALLEL
    children: List['State'] = field(default_factory=list)
    is_initial: bool = False
    parent_label: Optional[str] = None

    def get_all_states(self) -> Dict[str, 'State']:
        """Get all states including nested children."""
        result = {self.label: self}
        for child in self.children:
            child.parent_label = self.label
            result.update(child.get_all_states())
        return result

    def copy(self) -> 'State':
        """Deep copy."""
        return State(
            label=self.label,
            state_type=self.state_type,
            children=[c.copy() for c in self.children],
            is_initial=self.is_initial,
            parent_label=self.parent_label,
        )


@dataclass
class Transition:
    """Represents a transition in a statechart."""
    label: str
    from_states: List[str]
    to_states: List[str]
    event: str = ""
    guard: str = ""
    actions: List[str] = field(default_factory=list)


@dataclass
class Event:
    """Represents an event in a statechart."""
    name: str
    payload_type: str = ""


@dataclass
class Statechart:
    """Complete statechart representation."""
    version: str = "1.0.0"
    root_state: Optional[State] = None
    transitions: List[Transition] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)

    def get_all_states(self) -> Dict[str, State]:
        """Get all states in the chart."""
        if self.root_state is None:
            return {}
        return self.root_state.get_all_states()

    def get_state_labels(self) -> Set[str]:
        """Get all state labels."""
        return set(self.get_all_states().keys())

    def get_transition_key(self, t: Transition) -> Tuple:
        """Get unique key for a transition."""
        return (tuple(sorted(t.from_states)), tuple(sorted(t.to_states)), t.event)

    def copy(self) -> 'Statechart':
        """Deep copy."""
        return Statechart(
            version=self.version,
            root_state=self.root_state.copy() if self.root_state else None,
            transitions=[copy.deepcopy(t) for t in self.transitions],
            events=[copy.deepcopy(e) for e in self.events],
        )


# =============================================================================
# DIFF TYPES
# =============================================================================

@dataclass
class StateDiff:
    """Change to a single state."""
    change_type: ChangeType
    state_label: str
    old_state: Optional[State] = None
    new_state: Optional[State] = None
    modified_fields: List[str] = field(default_factory=list)
    new_label: Optional[str] = None  # For renames


@dataclass
class TransitionDiff:
    """Change to a single transition."""
    change_type: ChangeType
    transition_label: str
    old_transition: Optional[Transition] = None
    new_transition: Optional[Transition] = None
    modified_fields: List[str] = field(default_factory=list)


@dataclass
class EventDiff:
    """Change to a single event."""
    change_type: ChangeType
    event_name: str
    old_event: Optional[Event] = None
    new_event: Optional[Event] = None


@dataclass
class BreakingChange:
    """A change that requires migration attention."""
    change_type: BreakingChangeType
    description: str
    affected_element: str
    migration_required: MigrationRequirement


@dataclass
class ChartDiff:
    """
    Complete diff between two statechart versions.

    Formal: D(SC₁, SC₂) = (V₁, V₂, ΔS, Δδ, ΔE, C, B)
    """
    from_version: str
    to_version: str
    state_changes: List[StateDiff] = field(default_factory=list)
    transition_changes: List[TransitionDiff] = field(default_factory=list)
    event_changes: List[EventDiff] = field(default_factory=list)
    compatibility: CompatibilityLevel = CompatibilityLevel.FULL
    breaking_changes: List[BreakingChange] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        """Check if diff contains breaking changes."""
        return len(self.breaking_changes) > 0

    @property
    def requires_migration(self) -> bool:
        """Check if diff requires migration."""
        return any(
            bc.migration_required != MigrationRequirement.NONE
            for bc in self.breaking_changes
        )

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Diff: {self.from_version} → {self.to_version}",
            f"Compatibility: {self.compatibility.name}",
            f"State changes: {len(self.state_changes)}",
            f"Transition changes: {len(self.transition_changes)}",
            f"Event changes: {len(self.event_changes)}",
            f"Breaking changes: {len(self.breaking_changes)}",
        ]
        return "\n".join(lines)


# =============================================================================
# DIFF COMPUTATION
# =============================================================================

class SchemaDiffer:
    """
    Computes diffs between statechart versions.

    Implements structural matching to detect:
    - Added/removed states
    - Renamed states (heuristic based on structure similarity)
    - Moved states (parent changed)
    - Modified states (type/initial changed)
    """

    def __init__(self, rename_threshold: float = 0.7):
        """
        Args:
            rename_threshold: Similarity threshold for rename detection
        """
        self.rename_threshold = rename_threshold

    def diff(self, old_chart: Statechart, new_chart: Statechart) -> ChartDiff:
        """
        Compute diff between two statechart versions.

        Algorithm:
        1. Match states by label
        2. Detect renames via structure similarity
        3. Classify remaining as added/removed
        4. Diff transitions and events
        5. Identify breaking changes
        """
        result = ChartDiff(
            from_version=old_chart.version,
            to_version=new_chart.version,
        )

        # Get all states
        old_states = old_chart.get_all_states()
        new_states = new_chart.get_all_states()

        # Find state changes
        self._diff_states(old_states, new_states, result)

        # Find transition changes
        self._diff_transitions(old_chart.transitions, new_chart.transitions, result)

        # Find event changes
        self._diff_events(old_chart.events, new_chart.events, result)

        # Identify breaking changes
        self._identify_breaking_changes(result)

        # Classify compatibility
        result.compatibility = self._classify_compatibility(result)

        return result

    def _diff_states(
        self,
        old_states: Dict[str, State],
        new_states: Dict[str, State],
        result: ChartDiff
    ):
        """Diff states between versions."""
        old_labels = set(old_states.keys())
        new_labels = set(new_states.keys())

        # Direct matches (same label)
        matched_old = set()
        matched_new = set()

        for label in old_labels & new_labels:
            old_state = old_states[label]
            new_state = new_states[label]

            # Check for modifications
            modified_fields = self._get_modified_fields(old_state, new_state)
            if modified_fields:
                result.state_changes.append(StateDiff(
                    change_type=ChangeType.MODIFIED,
                    state_label=label,
                    old_state=old_state,
                    new_state=new_state,
                    modified_fields=modified_fields,
                ))
            elif old_state.parent_label != new_state.parent_label:
                result.state_changes.append(StateDiff(
                    change_type=ChangeType.MOVED,
                    state_label=label,
                    old_state=old_state,
                    new_state=new_state,
                ))

            matched_old.add(label)
            matched_new.add(label)

        # Unmatched states
        unmatched_old = old_labels - matched_old
        unmatched_new = new_labels - matched_new

        # Try to detect renames
        renames = self._detect_renames(
            {l: old_states[l] for l in unmatched_old},
            {l: new_states[l] for l in unmatched_new},
        )

        for old_label, new_label in renames:
            result.state_changes.append(StateDiff(
                change_type=ChangeType.RENAMED,
                state_label=old_label,
                old_state=old_states[old_label],
                new_state=new_states[new_label],
                new_label=new_label,
            ))
            unmatched_old.discard(old_label)
            unmatched_new.discard(new_label)

        # Remaining unmatched = added/removed
        for label in unmatched_old:
            result.state_changes.append(StateDiff(
                change_type=ChangeType.REMOVED,
                state_label=label,
                old_state=old_states[label],
            ))

        for label in unmatched_new:
            result.state_changes.append(StateDiff(
                change_type=ChangeType.ADDED,
                state_label=label,
                new_state=new_states[label],
            ))

    def _get_modified_fields(self, old: State, new: State) -> List[str]:
        """Get list of modified fields between states."""
        modified = []
        if old.state_type != new.state_type:
            modified.append("state_type")
        if old.is_initial != new.is_initial:
            modified.append("is_initial")
        return modified

    def _detect_renames(
        self,
        old_states: Dict[str, State],
        new_states: Dict[str, State],
    ) -> List[Tuple[str, str]]:
        """
        Detect renamed states using structure similarity.

        Heuristic: States with same type, similar children count,
        and similar position are likely renames.
        """
        renames = []

        for old_label, old_state in old_states.items():
            best_match = None
            best_score = 0.0

            for new_label, new_state in new_states.items():
                score = self._state_similarity(old_state, new_state)
                if score > best_score and score >= self.rename_threshold:
                    best_score = score
                    best_match = new_label

            if best_match:
                renames.append((old_label, best_match))

        return renames

    def _state_similarity(self, s1: State, s2: State) -> float:
        """
        Compute similarity between two states.

        Factors:
        - Same type: +0.4
        - Same children count: +0.3
        - Same parent type: +0.3
        """
        score = 0.0

        if s1.state_type == s2.state_type:
            score += 0.4

        if len(s1.children) == len(s2.children):
            score += 0.3
        elif abs(len(s1.children) - len(s2.children)) <= 1:
            score += 0.15

        if s1.is_initial == s2.is_initial:
            score += 0.3

        return score

    def _diff_transitions(
        self,
        old_transitions: List[Transition],
        new_transitions: List[Transition],
        result: ChartDiff
    ):
        """Diff transitions between versions."""
        old_by_key = {
            (tuple(t.from_states), tuple(t.to_states), t.event): t
            for t in old_transitions
        }
        new_by_key = {
            (tuple(t.from_states), tuple(t.to_states), t.event): t
            for t in new_transitions
        }

        old_keys = set(old_by_key.keys())
        new_keys = set(new_by_key.keys())

        # Modified transitions
        for key in old_keys & new_keys:
            old_t = old_by_key[key]
            new_t = new_by_key[key]

            modified = []
            if old_t.guard != new_t.guard:
                modified.append("guard")
            if old_t.actions != new_t.actions:
                modified.append("actions")

            if modified:
                result.transition_changes.append(TransitionDiff(
                    change_type=ChangeType.MODIFIED,
                    transition_label=old_t.label or str(key),
                    old_transition=old_t,
                    new_transition=new_t,
                    modified_fields=modified,
                ))

        # Removed transitions
        for key in old_keys - new_keys:
            t = old_by_key[key]
            result.transition_changes.append(TransitionDiff(
                change_type=ChangeType.REMOVED,
                transition_label=t.label or str(key),
                old_transition=t,
            ))

        # Added transitions
        for key in new_keys - old_keys:
            t = new_by_key[key]
            result.transition_changes.append(TransitionDiff(
                change_type=ChangeType.ADDED,
                transition_label=t.label or str(key),
                new_transition=t,
            ))

    def _diff_events(
        self,
        old_events: List[Event],
        new_events: List[Event],
        result: ChartDiff
    ):
        """Diff events between versions."""
        old_by_name = {e.name: e for e in old_events}
        new_by_name = {e.name: e for e in new_events}

        old_names = set(old_by_name.keys())
        new_names = set(new_by_name.keys())

        # Modified events
        for name in old_names & new_names:
            old_e = old_by_name[name]
            new_e = new_by_name[name]

            if old_e.payload_type != new_e.payload_type:
                result.event_changes.append(EventDiff(
                    change_type=ChangeType.MODIFIED,
                    event_name=name,
                    old_event=old_e,
                    new_event=new_e,
                ))

        # Removed events
        for name in old_names - new_names:
            result.event_changes.append(EventDiff(
                change_type=ChangeType.REMOVED,
                event_name=name,
                old_event=old_by_name[name],
            ))

        # Added events
        for name in new_names - old_names:
            result.event_changes.append(EventDiff(
                change_type=ChangeType.ADDED,
                event_name=name,
                new_event=new_by_name[name],
            ))

    def _identify_breaking_changes(self, result: ChartDiff):
        """Identify breaking changes in the diff."""
        for change in result.state_changes:
            if change.change_type == ChangeType.REMOVED:
                result.breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.STATE_REMOVED,
                    description=f"State '{change.state_label}' was removed",
                    affected_element=change.state_label,
                    migration_required=MigrationRequirement.STATE_MAPPING,
                ))

            elif change.change_type == ChangeType.MODIFIED:
                if "state_type" in change.modified_fields:
                    result.breaking_changes.append(BreakingChange(
                        change_type=BreakingChangeType.STATE_TYPE_CHANGED,
                        description=f"State '{change.state_label}' type changed",
                        affected_element=change.state_label,
                        migration_required=MigrationRequirement.MANUAL,
                    ))

                if "is_initial" in change.modified_fields:
                    result.breaking_changes.append(BreakingChange(
                        change_type=BreakingChangeType.INITIAL_STATE_CHANGED,
                        description=f"Initial state changed for '{change.state_label}'",
                        affected_element=change.state_label,
                        migration_required=MigrationRequirement.STATE_MAPPING,
                    ))

        for change in result.transition_changes:
            if change.change_type == ChangeType.REMOVED:
                result.breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.TRANSITION_REMOVED,
                    description=f"Transition '{change.transition_label}' was removed",
                    affected_element=change.transition_label,
                    migration_required=MigrationRequirement.NONE,
                ))

            elif change.change_type == ChangeType.MODIFIED:
                if "guard" in change.modified_fields:
                    old_guard = change.old_transition.guard if change.old_transition else ""
                    new_guard = change.new_transition.guard if change.new_transition else ""
                    if not old_guard and new_guard:
                        result.breaking_changes.append(BreakingChange(
                            change_type=BreakingChangeType.GUARD_ADDED,
                            description=f"Guard added to transition '{change.transition_label}'",
                            affected_element=change.transition_label,
                            migration_required=MigrationRequirement.NONE,
                        ))

        for change in result.event_changes:
            if change.change_type == ChangeType.REMOVED:
                result.breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.EVENT_REMOVED,
                    description=f"Event '{change.event_name}' was removed",
                    affected_element=change.event_name,
                    migration_required=MigrationRequirement.MANUAL,
                ))

            elif change.change_type == ChangeType.MODIFIED:
                result.breaking_changes.append(BreakingChange(
                    change_type=BreakingChangeType.EVENT_PAYLOAD_CHANGED,
                    description=f"Event '{change.event_name}' payload changed",
                    affected_element=change.event_name,
                    migration_required=MigrationRequirement.MANUAL,
                ))

    def _classify_compatibility(self, result: ChartDiff) -> CompatibilityLevel:
        """Classify overall compatibility level."""
        if not result.state_changes and not result.transition_changes and not result.event_changes:
            return CompatibilityLevel.FULL

        if result.breaking_changes:
            return CompatibilityLevel.BREAKING

        # Only additions = forward compatible
        all_additions = all(
            c.change_type == ChangeType.ADDED
            for c in result.state_changes + result.transition_changes + result.event_changes
        )
        if all_additions:
            return CompatibilityLevel.FORWARD

        return CompatibilityLevel.BACKWARD


# =============================================================================
# TESTING
# =============================================================================

def test_schema_diff():
    """Test schema diffing."""
    print("=" * 60)
    print("SCHEMA DIFF TEST")
    print("=" * 60)

    # Create old chart
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active", children=[
                    State(label="Processing"),
                    State(label="Waiting"),
                ]),
                State(label="Done"),
            ]
        ),
        transitions=[
            Transition(label="t1", from_states=["Idle"], to_states=["Active"], event="START"),
            Transition(label="t2", from_states=["Active"], to_states=["Done"], event="FINISH"),
        ],
        events=[
            Event(name="START"),
            Event(name="FINISH"),
        ]
    )

    # Create new chart with changes
    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active", children=[
                    State(label="Running"),  # Renamed from Processing
                    State(label="Waiting"),
                    State(label="Paused"),  # Added
                ]),
                # Done removed
                State(label="Complete"),  # Added
            ]
        ),
        transitions=[
            Transition(label="t1", from_states=["Idle"], to_states=["Active"], event="START"),
            Transition(label="t3", from_states=["Active"], to_states=["Complete"], event="FINISH", guard="count > 0"),  # Modified
            Transition(label="t4", from_states=["Active"], to_states=["Active"], event="PAUSE"),  # Added
        ],
        events=[
            Event(name="START"),
            Event(name="FINISH"),
            Event(name="PAUSE"),  # Added
        ]
    )

    differ = SchemaDiffer()
    diff = differ.diff(old_chart, new_chart)

    print("\n" + diff.summary())

    print("\nState changes:")
    for change in diff.state_changes:
        print(f"  {change.change_type.name}: {change.state_label}")
        if change.new_label:
            print(f"    -> {change.new_label}")

    print("\nTransition changes:")
    for change in diff.transition_changes:
        print(f"  {change.change_type.name}: {change.transition_label}")

    print("\nBreaking changes:")
    for bc in diff.breaking_changes:
        print(f"  {bc.change_type.name}: {bc.description}")
        print(f"    Migration: {bc.migration_required.name}")

    print("\n" + "=" * 60)
    print("SCHEMA DIFF TEST COMPLETE")
    print("=" * 60)

    return diff


if __name__ == "__main__":
    test_schema_diff()
