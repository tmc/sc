"""
Conflict Resolver for Statechart Merging.

Detects and resolves conflicts when merging statecharts:
1. STATE CONFLICTS: Same state name, different semantics
2. EVENT CONFLICTS: Same event name, different triggers
3. TRANSITION CONFLICTS: Conflicting transitions from same state
4. INITIAL STATE CONFLICTS: Both SCs have different initial states

Resolution strategies:
- RENAME: Add prefix/suffix to disambiguate
- MERGE: Combine into single entity if semantically similar
- NAMESPACE: Keep separate with namespace prefixes
- LLM: Use Qwen to decide best resolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum, auto

# Try to import mlx_lm
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class ConflictType(Enum):
    """Types of conflicts that can occur during merge."""
    STATE_NAME = auto()      # Same state name in both SCs
    EVENT_NAME = auto()      # Same event name in both SCs
    TRANSITION = auto()      # Conflicting transitions
    INITIAL_STATE = auto()   # Different initial states
    HIERARCHY = auto()       # Incompatible hierarchies


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    RENAME_FIRST = auto()    # Rename entity from first SC
    RENAME_SECOND = auto()   # Rename entity from second SC
    RENAME_BOTH = auto()     # Rename both with prefixes
    MERGE = auto()           # Merge into single entity
    KEEP_FIRST = auto()      # Keep first SC's version
    KEEP_SECOND = auto()     # Keep second SC's version
    LLM_DECIDE = auto()      # Let LLM decide


@dataclass
class Conflict:
    """Represents a detected conflict."""
    conflict_type: ConflictType
    entity_name: str
    sc1_value: Dict          # Value/definition from SC1
    sc2_value: Dict          # Value/definition from SC2
    severity: str = "medium"  # low, medium, high
    description: str = ""


@dataclass
class Resolution:
    """Resolution for a conflict."""
    conflict: Conflict
    strategy: ResolutionStrategy
    result: Dict             # Resolved value
    renamed_from: Optional[str] = None
    renamed_to: Optional[str] = None


class ConflictResolver:
    """
    Detects and resolves conflicts between statecharts.
    """

    def __init__(self, use_llm: bool = False, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.use_llm = use_llm and HAS_MLX
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if self.use_llm:
            self._load_model()

    def _load_model(self):
        """Load LLM for intelligent resolution."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.use_llm = False

    def detect_conflicts(self, sc1: Dict, sc2: Dict) -> List[Conflict]:
        """
        Detect all conflicts between two statecharts.

        Args:
            sc1: First statechart JSON
            sc2: Second statechart JSON

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Extract components
        states1 = self._extract_states(sc1)
        states2 = self._extract_states(sc2)
        events1 = self._extract_events(sc1)
        events2 = self._extract_events(sc2)
        transitions1 = self._extract_transitions(sc1)
        transitions2 = self._extract_transitions(sc2)

        # Detect state name conflicts
        common_states = set(states1.keys()) & set(states2.keys())
        for state_name in common_states:
            conflicts.append(Conflict(
                conflict_type=ConflictType.STATE_NAME,
                entity_name=state_name,
                sc1_value=states1[state_name],
                sc2_value=states2[state_name],
                severity="high" if state_name != "__root__" else "low",
                description=f"State '{state_name}' exists in both statecharts"
            ))

        # Detect event name conflicts
        common_events = events1 & events2
        for event_name in common_events:
            conflicts.append(Conflict(
                conflict_type=ConflictType.EVENT_NAME,
                entity_name=event_name,
                sc1_value={"event": event_name, "source": "sc1"},
                sc2_value={"event": event_name, "source": "sc2"},
                severity="medium",
                description=f"Event '{event_name}' exists in both statecharts"
            ))

        # Detect initial state conflicts
        initial1 = self._find_initial(sc1)
        initial2 = self._find_initial(sc2)
        if initial1 and initial2 and initial1 != initial2:
            conflicts.append(Conflict(
                conflict_type=ConflictType.INITIAL_STATE,
                entity_name="initial",
                sc1_value={"initial": initial1},
                sc2_value={"initial": initial2},
                severity="high",
                description=f"Different initial states: '{initial1}' vs '{initial2}'"
            ))

        # Detect transition conflicts (same source+event, different target)
        for (src1, evt1), tgt1 in transitions1.items():
            if (src1, evt1) in transitions2:
                tgt2 = transitions2[(src1, evt1)]
                if tgt1 != tgt2:
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.TRANSITION,
                        entity_name=f"{src1}+{evt1}",
                        sc1_value={"source": src1, "event": evt1, "target": tgt1},
                        sc2_value={"source": src1, "event": evt1, "target": tgt2},
                        severity="high",
                        description=f"Conflicting transition from '{src1}' on '{evt1}'"
                    ))

        return conflicts

    def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        default_strategy: ResolutionStrategy = ResolutionStrategy.RENAME_SECOND
    ) -> List[Resolution]:
        """
        Resolve all conflicts.

        Args:
            conflicts: List of detected conflicts
            default_strategy: Default resolution strategy

        Returns:
            List of resolutions
        """
        resolutions = []

        for conflict in conflicts:
            if self.use_llm and conflict.severity == "high":
                resolution = self._resolve_with_llm(conflict)
            else:
                resolution = self._resolve_heuristic(conflict, default_strategy)

            resolutions.append(resolution)

        return resolutions

    def _resolve_heuristic(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy
    ) -> Resolution:
        """Resolve conflict using heuristic rules."""

        if conflict.conflict_type == ConflictType.STATE_NAME:
            if conflict.entity_name == "__root__":
                # Root state: merge
                return Resolution(
                    conflict=conflict,
                    strategy=ResolutionStrategy.MERGE,
                    result={"label": "__root__", "merged": True}
                )
            elif strategy == ResolutionStrategy.RENAME_SECOND:
                new_name = f"{conflict.entity_name}_2"
                return Resolution(
                    conflict=conflict,
                    strategy=strategy,
                    result={"original": conflict.entity_name, "renamed": new_name},
                    renamed_from=conflict.entity_name,
                    renamed_to=new_name
                )
            elif strategy == ResolutionStrategy.RENAME_BOTH:
                name1 = f"SC1_{conflict.entity_name}"
                name2 = f"SC2_{conflict.entity_name}"
                return Resolution(
                    conflict=conflict,
                    strategy=strategy,
                    result={"sc1_name": name1, "sc2_name": name2},
                    renamed_from=conflict.entity_name,
                    renamed_to=name2
                )

        elif conflict.conflict_type == ConflictType.EVENT_NAME:
            # Events can often be shared if semantically similar
            return Resolution(
                conflict=conflict,
                strategy=ResolutionStrategy.MERGE,
                result={"event": conflict.entity_name, "shared": True}
            )

        elif conflict.conflict_type == ConflictType.INITIAL_STATE:
            # Create a new composite initial state
            return Resolution(
                conflict=conflict,
                strategy=ResolutionStrategy.MERGE,
                result={
                    "new_initial": "Initial",
                    "sc1_initial": conflict.sc1_value["initial"],
                    "sc2_initial": conflict.sc2_value["initial"],
                }
            )

        elif conflict.conflict_type == ConflictType.TRANSITION:
            # Keep first by default, could create parallel paths
            return Resolution(
                conflict=conflict,
                strategy=ResolutionStrategy.KEEP_FIRST,
                result=conflict.sc1_value
            )

        # Default: rename second
        return Resolution(
            conflict=conflict,
            strategy=ResolutionStrategy.RENAME_SECOND,
            result=conflict.sc2_value
        )

    def _resolve_with_llm(self, conflict: Conflict) -> Resolution:
        """Use LLM to decide resolution strategy."""
        if not self.model:
            return self._resolve_heuristic(conflict, ResolutionStrategy.RENAME_SECOND)

        prompt = f"""You are merging two statecharts. There is a conflict:

Type: {conflict.conflict_type.name}
Entity: {conflict.entity_name}
SC1 value: {conflict.sc1_value}
SC2 value: {conflict.sc2_value}

How should this be resolved? Choose one:
1. MERGE - Combine into single entity
2. RENAME_SECOND - Rename the second statechart's entity
3. KEEP_FIRST - Use first statechart's version

Answer with just the number (1, 2, or 3):"""

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=10,
                temp=0.1,
            )

            response = response.strip()
            if "1" in response or "MERGE" in response.upper():
                strategy = ResolutionStrategy.MERGE
            elif "3" in response or "KEEP_FIRST" in response.upper():
                strategy = ResolutionStrategy.KEEP_FIRST
            else:
                strategy = ResolutionStrategy.RENAME_SECOND

            return self._resolve_heuristic(conflict, strategy)

        except Exception as e:
            print(f"LLM resolution failed: {e}")
            return self._resolve_heuristic(conflict, ResolutionStrategy.RENAME_SECOND)

    def _extract_states(self, sc: Dict) -> Dict[str, Dict]:
        """Extract states as name -> definition dict."""
        states = {}
        root = sc.get('root_state', {})
        self._collect_states(root, states)
        return states

    def _collect_states(self, node: Dict, states: Dict):
        """Recursively collect states."""
        label = node.get('label', '')
        if label:
            states[label] = node
        for child in node.get('children', []):
            self._collect_states(child, states)

    def _extract_events(self, sc: Dict) -> Set[str]:
        """Extract all event names."""
        events = set()
        for t in sc.get('transitions', []):
            event = t.get('event', '')
            if event:
                events.add(event)
        return events

    def _extract_transitions(self, sc: Dict) -> Dict[Tuple[str, str], str]:
        """Extract transitions as (source, event) -> target."""
        transitions = {}
        for t in sc.get('transitions', []):
            src = t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', '')
            tgt = t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', '')
            event = t.get('event', '')
            if src and event:
                transitions[(src, event)] = tgt
        return transitions

    def _find_initial(self, sc: Dict) -> Optional[str]:
        """Find initial state."""
        root = sc.get('root_state', {})
        return self._find_initial_recursive(root)

    def _find_initial_recursive(self, node: Dict) -> Optional[str]:
        """Find initial state recursively."""
        if node.get('is_initial') and node.get('label') != '__root__':
            return node.get('label')
        for child in node.get('children', []):
            result = self._find_initial_recursive(child)
            if result:
                return result
        return None

    def apply_resolutions(
        self,
        sc: Dict,
        resolutions: List[Resolution],
        is_second: bool = True
    ) -> Dict:
        """
        Apply resolutions to a statechart.

        Args:
            sc: Statechart to modify
            resolutions: List of resolutions
            is_second: Whether this is SC2 (the one being renamed)

        Returns:
            Modified statechart
        """
        import copy
        modified = copy.deepcopy(sc)

        # Build rename map
        rename_map = {}
        for res in resolutions:
            if res.strategy in (ResolutionStrategy.RENAME_SECOND, ResolutionStrategy.RENAME_BOTH):
                if is_second and res.renamed_from and res.renamed_to:
                    rename_map[res.renamed_from] = res.renamed_to

        if not rename_map:
            return modified

        # Apply renames to states
        self._rename_states(modified.get('root_state', {}), rename_map)

        # Apply renames to transitions
        for t in modified.get('transitions', []):
            # Rename source
            if isinstance(t.get('from'), list):
                t['from'] = [rename_map.get(s, s) for s in t['from']]
            elif t.get('from') in rename_map:
                t['from'] = rename_map[t['from']]

            # Rename target
            if isinstance(t.get('to'), list):
                t['to'] = [rename_map.get(s, s) for s in t['to']]
            elif t.get('to') in rename_map:
                t['to'] = rename_map[t['to']]

        return modified

    def _rename_states(self, node: Dict, rename_map: Dict[str, str]):
        """Recursively rename states."""
        label = node.get('label', '')
        if label in rename_map:
            node['label'] = rename_map[label]

        for child in node.get('children', []):
            self._rename_states(child, rename_map)


def detect_conflicts(sc1: Dict, sc2: Dict) -> List[Conflict]:
    """Convenience function to detect conflicts."""
    resolver = ConflictResolver(use_llm=False)
    return resolver.detect_conflicts(sc1, sc2)


def resolve_conflicts(
    conflicts: List[Conflict],
    use_llm: bool = False
) -> List[Resolution]:
    """Convenience function to resolve conflicts."""
    resolver = ConflictResolver(use_llm=use_llm)
    return resolver.resolve_conflicts(conflicts)


def test_conflict_resolver():
    """Test conflict resolution."""
    print("=" * 60)
    print("Testing Conflict Resolver")
    print("=" * 60)

    # Two statecharts with conflicts
    sc1 = {
        "root_state": {
            "label": "__root__",
            "children": [
                {"label": "Idle", "is_initial": True},
                {"label": "Active"},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Active"], "event": "START"},
            {"from": ["Active"], "to": ["Idle"], "event": "STOP"},
        ]
    }

    sc2 = {
        "root_state": {
            "label": "__root__",
            "children": [
                {"label": "Idle", "is_initial": True},  # Conflict!
                {"label": "Running"},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Running"], "event": "RUN"},
            {"from": ["Running"], "to": ["Idle"], "event": "STOP"},  # Shared event
        ]
    }

    resolver = ConflictResolver(use_llm=False)

    print("\n1. Detecting conflicts:")
    conflicts = resolver.detect_conflicts(sc1, sc2)
    for c in conflicts:
        print(f"  [{c.severity}] {c.conflict_type.name}: {c.description}")

    print(f"\n  Total conflicts: {len(conflicts)}")

    print("\n2. Resolving conflicts:")
    resolutions = resolver.resolve_conflicts(conflicts)
    for r in resolutions:
        print(f"  {r.conflict.entity_name}: {r.strategy.name}")
        if r.renamed_to:
            print(f"    Renamed: {r.renamed_from} -> {r.renamed_to}")

    print("\n3. Applying resolutions to SC2:")
    modified_sc2 = resolver.apply_resolutions(sc2, resolutions, is_second=True)

    # Show modified states
    def show_states(node, indent=0):
        label = node.get('label', '')
        if label:
            print("  " * indent + f"- {label}")
        for child in node.get('children', []):
            show_states(child, indent + 1)

    print("  Modified SC2 states:")
    show_states(modified_sc2['root_state'], 1)

    print("\n  Modified SC2 transitions:")
    for t in modified_sc2.get('transitions', []):
        src = t.get('from', ['?'])[0] if isinstance(t.get('from'), list) else t.get('from', '?')
        tgt = t.get('to', ['?'])[0] if isinstance(t.get('to'), list) else t.get('to', '?')
        print(f"    {src} --{t.get('event', '?')}--> {tgt}")

    print("\n" + "=" * 60)
    print("Conflict resolver tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_conflict_resolver()
