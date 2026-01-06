"""
Statechart Merger.

Merges two statecharts into a unified statechart that preserves
the behaviors of both original statecharts.

Merge strategies:
1. PARALLEL: Create parallel regions (AND-composition)
2. SEQUENTIAL: Chain statecharts (first completes, then second)
3. UNION: Combine states and transitions (OR-composition)
4. LLM: Use Qwen to intelligently merge
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum, auto

from .conflict_resolver import (
    ConflictResolver, Conflict, Resolution,
    ResolutionStrategy, detect_conflicts
)

# Try to import mlx_lm
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class MergeStrategy(Enum):
    """Strategies for merging statecharts."""
    PARALLEL = auto()     # Create parallel regions
    SEQUENTIAL = auto()   # Chain: SC1 then SC2
    UNION = auto()        # Union of states/transitions
    LLM = auto()          # LLM-guided merge


@dataclass
class MergeConfig:
    """Configuration for statechart merging."""
    strategy: MergeStrategy = MergeStrategy.UNION
    use_llm: bool = False
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    preserve_hierarchy: bool = True
    auto_resolve_conflicts: bool = True
    merged_name: str = "MergedStatechart"


@dataclass
class MergeResult:
    """Result of merging two statecharts."""
    merged: Dict                    # The merged statechart
    strategy_used: MergeStrategy
    conflicts_found: int
    conflicts_resolved: int
    states_merged: int
    transitions_merged: int
    warnings: List[str] = field(default_factory=list)
    sc1_preservation: float = 1.0   # How much of SC1 behavior preserved
    sc2_preservation: float = 1.0   # How much of SC2 behavior preserved


class StatechartMerger:
    """
    Merges two statecharts into one.
    """

    def __init__(self, config: MergeConfig = None):
        self.config = config or MergeConfig()
        self.resolver = ConflictResolver(use_llm=self.config.use_llm)
        self.model = None
        self.tokenizer = None

        if self.config.use_llm and HAS_MLX:
            self._load_model()

    def _load_model(self):
        """Load LLM for intelligent merging."""
        try:
            self.model, self.tokenizer = load(self.config.model_name)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def merge(self, sc1: Dict, sc2: Dict) -> MergeResult:
        """
        Merge two statecharts.

        Args:
            sc1: First statechart
            sc2: Second statechart

        Returns:
            MergeResult with merged statechart and metadata
        """
        # Step 1: Detect conflicts
        conflicts = self.resolver.detect_conflicts(sc1, sc2)

        # Step 2: Resolve conflicts
        resolutions = []
        if self.config.auto_resolve_conflicts and conflicts:
            resolutions = self.resolver.resolve_conflicts(conflicts)

        # Step 3: Apply resolutions
        sc1_resolved = copy.deepcopy(sc1)
        sc2_resolved = self.resolver.apply_resolutions(sc2, resolutions, is_second=True)

        # Step 4: Merge based on strategy
        if self.config.strategy == MergeStrategy.PARALLEL:
            merged = self._merge_parallel(sc1_resolved, sc2_resolved)
        elif self.config.strategy == MergeStrategy.SEQUENTIAL:
            merged = self._merge_sequential(sc1_resolved, sc2_resolved)
        elif self.config.strategy == MergeStrategy.LLM and self.model:
            merged = self._merge_with_llm(sc1_resolved, sc2_resolved)
        else:
            merged = self._merge_union(sc1_resolved, sc2_resolved)

        # Count merged entities
        states_merged = self._count_states(merged)
        transitions_merged = len(merged.get('transitions', []))

        return MergeResult(
            merged=merged,
            strategy_used=self.config.strategy,
            conflicts_found=len(conflicts),
            conflicts_resolved=len(resolutions),
            states_merged=states_merged,
            transitions_merged=transitions_merged,
        )

    def _merge_union(self, sc1: Dict, sc2: Dict) -> Dict:
        """
        Merge using union strategy.

        Combines all states and transitions from both SCs.
        """
        merged = {
            "root_state": {
                "label": "__root__",
                "type": 2,  # Normal (OR)
                "children": []
            },
            "transitions": []
        }

        # Collect states from both
        states1 = self._get_top_level_states(sc1)
        states2 = self._get_top_level_states(sc2)

        # Combine states
        merged["root_state"]["children"] = states1 + states2

        # Mark initial state (prefer SC1's initial)
        initial_marked = False
        for child in merged["root_state"]["children"]:
            if child.get('is_initial') and not initial_marked:
                initial_marked = True
            elif child.get('is_initial'):
                child['is_initial'] = False  # Only one initial

        if not initial_marked and merged["root_state"]["children"]:
            merged["root_state"]["children"][0]['is_initial'] = True

        # Combine transitions
        merged["transitions"] = (
            copy.deepcopy(sc1.get('transitions', [])) +
            copy.deepcopy(sc2.get('transitions', []))
        )

        return merged

    def _merge_parallel(self, sc1: Dict, sc2: Dict) -> Dict:
        """
        Merge using parallel strategy.

        Creates an AND-composition where both SCs run simultaneously.
        """
        merged = {
            "root_state": {
                "label": "__root__",
                "type": 3,  # Parallel (AND)
                "children": []
            },
            "transitions": []
        }

        # Create regions for each SC
        region1 = {
            "label": "Region1",
            "type": 2,  # Normal
            "children": self._get_top_level_states(sc1)
        }

        region2 = {
            "label": "Region2",
            "type": 2,  # Normal
            "children": self._get_top_level_states(sc2)
        }

        merged["root_state"]["children"] = [region1, region2]

        # Prefix transitions with region
        transitions1 = copy.deepcopy(sc1.get('transitions', []))
        transitions2 = copy.deepcopy(sc2.get('transitions', []))

        merged["transitions"] = transitions1 + transitions2

        return merged

    def _merge_sequential(self, sc1: Dict, sc2: Dict) -> Dict:
        """
        Merge using sequential strategy.

        SC2 starts when SC1 reaches a final state.
        """
        merged = {
            "root_state": {
                "label": "__root__",
                "type": 2,  # Normal
                "children": []
            },
            "transitions": []
        }

        # Get states
        states1 = self._get_top_level_states(sc1)
        states2 = self._get_top_level_states(sc2)

        # Clear is_initial from SC2
        for state in states2:
            state['is_initial'] = False

        merged["root_state"]["children"] = states1 + states2

        # Copy transitions
        merged["transitions"] = copy.deepcopy(sc1.get('transitions', []))
        merged["transitions"].extend(copy.deepcopy(sc2.get('transitions', [])))

        # Find "final" states in SC1 (states with no outgoing transitions)
        sc1_states = {s['label'] for s in states1}
        states_with_outgoing = set()
        for t in sc1.get('transitions', []):
            src = t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', '')
            if src in sc1_states:
                states_with_outgoing.add(src)

        final_states = sc1_states - states_with_outgoing

        # Find initial state of SC2
        initial2 = None
        for state in self._get_top_level_states(sc2):
            if state.get('is_initial'):
                initial2 = state['label']
                break
        if not initial2 and states2:
            initial2 = states2[0]['label']

        # Add transitions from SC1 finals to SC2 initial
        if final_states and initial2:
            for final in final_states:
                merged["transitions"].append({
                    "from": [final],
                    "to": [initial2],
                    "event": "CONTINUE"
                })

        return merged

    def _merge_with_llm(self, sc1: Dict, sc2: Dict) -> Dict:
        """
        Use LLM to intelligently merge statecharts.
        """
        if not self.model:
            return self._merge_union(sc1, sc2)

        # Extract info for prompt
        states1 = [s['label'] for s in self._get_top_level_states(sc1)]
        states2 = [s['label'] for s in self._get_top_level_states(sc2)]

        prompt = f"""Merge these two statecharts:

SC1 states: {states1}
SC1 transitions: {sc1.get('transitions', [])}

SC2 states: {states2}
SC2 transitions: {sc2.get('transitions', [])}

How should they be combined? Output one of:
1. PARALLEL - Run both simultaneously
2. SEQUENTIAL - SC1 then SC2
3. UNION - Combine all states

Answer with number only:"""

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=10,
                temp=0.1,
            )

            if "1" in response or "PARALLEL" in response.upper():
                return self._merge_parallel(sc1, sc2)
            elif "2" in response or "SEQUENTIAL" in response.upper():
                return self._merge_sequential(sc1, sc2)
            else:
                return self._merge_union(sc1, sc2)

        except Exception as e:
            print(f"LLM merge failed: {e}")
            return self._merge_union(sc1, sc2)

    def _get_top_level_states(self, sc: Dict) -> List[Dict]:
        """Get top-level state children from root."""
        root = sc.get('root_state', {})
        return copy.deepcopy(root.get('children', []))

    def _count_states(self, sc: Dict) -> int:
        """Count total states in statechart."""
        root = sc.get('root_state', {})
        return self._count_recursive(root)

    def _count_recursive(self, node: Dict) -> int:
        """Recursively count states."""
        count = 1 if node.get('label') and node.get('label') != '__root__' else 0
        for child in node.get('children', []):
            count += self._count_recursive(child)
        return count


def merge_statecharts(
    sc1: Dict,
    sc2: Dict,
    strategy: MergeStrategy = MergeStrategy.UNION,
    use_llm: bool = False
) -> MergeResult:
    """
    Convenience function to merge statecharts.

    Args:
        sc1: First statechart
        sc2: Second statechart
        strategy: Merge strategy
        use_llm: Whether to use LLM assistance

    Returns:
        MergeResult with merged statechart
    """
    config = MergeConfig(strategy=strategy, use_llm=use_llm)
    merger = StatechartMerger(config)
    return merger.merge(sc1, sc2)


def test_merger():
    """Test statechart merger."""
    print("=" * 60)
    print("Testing Statechart Merger")
    print("=" * 60)

    # Two simple statecharts
    sc1 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
            {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"},
        ]
    }

    sc2 = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Locked", "type": 1, "is_initial": True},
                {"label": "Unlocked", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Locked"], "to": ["Unlocked"], "event": "UNLOCK"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
        ]
    }

    def show_statechart(sc: Dict, name: str):
        """Display statechart structure."""
        print(f"\n  {name}:")
        root = sc.get('root_state', {})
        print(f"    Root type: {root.get('type', '?')}")
        print("    States:")
        for child in root.get('children', []):
            label = child.get('label', '?')
            initial = " (initial)" if child.get('is_initial') else ""
            print(f"      - {label}{initial}")
            for subchild in child.get('children', []):
                print(f"        - {subchild.get('label', '?')}")
        print("    Transitions:")
        for t in sc.get('transitions', []):
            src = t.get('from', ['?'])[0] if isinstance(t.get('from'), list) else t.get('from', '?')
            tgt = t.get('to', ['?'])[0] if isinstance(t.get('to'), list) else t.get('to', '?')
            print(f"      {src} --{t.get('event', '?')}--> {tgt}")

    print("\n1. Input statecharts:")
    show_statechart(sc1, "SC1 (Light)")
    show_statechart(sc2, "SC2 (Lock)")

    print("\n2. Union merge:")
    result_union = merge_statecharts(sc1, sc2, MergeStrategy.UNION)
    print(f"  Conflicts: {result_union.conflicts_found}")
    print(f"  States: {result_union.states_merged}")
    print(f"  Transitions: {result_union.transitions_merged}")
    show_statechart(result_union.merged, "Merged (UNION)")

    print("\n3. Parallel merge:")
    result_parallel = merge_statecharts(sc1, sc2, MergeStrategy.PARALLEL)
    print(f"  Strategy: {result_parallel.strategy_used.name}")
    show_statechart(result_parallel.merged, "Merged (PARALLEL)")

    print("\n4. Sequential merge:")
    result_seq = merge_statecharts(sc1, sc2, MergeStrategy.SEQUENTIAL)
    print(f"  Strategy: {result_seq.strategy_used.name}")
    show_statechart(result_seq.merged, "Merged (SEQUENTIAL)")

    # Test with conflicts
    print("\n5. Merge with conflicts:")
    sc3 = {
        "root_state": {
            "label": "__root__",
            "children": [
                {"label": "Off", "is_initial": True},  # Conflict with SC1!
                {"label": "Standby"},
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["Standby"], "event": "WAKE"},
        ]
    }

    result_conflict = merge_statecharts(sc1, sc3, MergeStrategy.UNION)
    print(f"  Conflicts found: {result_conflict.conflicts_found}")
    print(f"  Conflicts resolved: {result_conflict.conflicts_resolved}")
    show_statechart(result_conflict.merged, "Merged (with conflict resolution)")

    print("\n" + "=" * 60)
    print("Merger tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_merger()
