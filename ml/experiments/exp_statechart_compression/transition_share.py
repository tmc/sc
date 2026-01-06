"""
Transition Sharing Compression

Compresses statecharts by identifying and deduplicating common transition patterns.

APPROACHES:
1. Guard Canonicalization - Normalize guard expressions to find equivalents
2. Action Factoring - Extract common action sequences into shared subroutines
3. Transition Template - Parameterized transitions that can be instantiated
4. Subgraph Sharing - Identify repeated subgraph patterns

KEY INSIGHT:
Many statecharts have repeated patterns like:
- Same guard on multiple transitions
- Same action sequence in multiple places
- Repeated state/transition subgraphs (e.g., error handling)

By sharing these patterns, we reduce the representation size.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, FrozenSet
from collections import defaultdict
import hashlib
import copy

from .bisimulation import Statechart, State, Transition, StateType


# =============================================================================
# Transition Pattern Representation
# =============================================================================

@dataclass(frozen=True)
class TransitionPattern:
    """
    A canonical representation of a transition pattern.

    Captures the essential structure that can be shared.
    """
    event: str
    guard: Optional[str]
    action: Optional[str]

    # Structural features (source/target are parameterized)
    source_type: StateType = StateType.BASIC
    target_type: StateType = StateType.BASIC

    def signature(self) -> str:
        """Get hashable signature for pattern matching."""
        return f"{self.event}|{self.guard}|{self.action}|{self.source_type}|{self.target_type}"


@dataclass
class TransitionTemplate:
    """
    A parameterized transition template.

    Can be instantiated with specific source/target states.
    """
    pattern: TransitionPattern
    id: int

    # Statistics
    usage_count: int = 0
    instances: List[Tuple[str, str]] = field(default_factory=list)  # (source, target)

    def instantiate(self, source: str, target: str) -> Transition:
        """Create a concrete transition from this template."""
        return Transition(
            source=source,
            target=target,
            event=self.pattern.event,
            guard=self.pattern.guard,
            action=self.pattern.action,
        )


# =============================================================================
# Guard Canonicalization
# =============================================================================

class GuardCanonicalizer:
    """
    Normalizes guard expressions to canonical form.

    Enables detection of semantically equivalent guards.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def canonicalize(self, guard: Optional[str]) -> Optional[str]:
        """Convert guard to canonical form."""
        if guard is None:
            return None

        if guard in self._cache:
            return self._cache[guard]

        # Normalize whitespace
        canonical = " ".join(guard.split())

        # Sort commutative operations (AND, OR)
        canonical = self._normalize_commutative(canonical, " && ")
        canonical = self._normalize_commutative(canonical, " || ")

        # Normalize comparison operators
        canonical = self._normalize_comparisons(canonical)

        self._cache[guard] = canonical
        return canonical

    def _normalize_commutative(self, expr: str, op: str) -> str:
        """Sort operands of commutative operators."""
        if op not in expr:
            return expr

        parts = expr.split(op)
        sorted_parts = sorted(parts)
        return op.join(sorted_parts)

    def _normalize_comparisons(self, expr: str) -> str:
        """Normalize comparison operators to consistent form."""
        # Convert != to !(==)
        # Convert > to !(<= )
        # etc.
        return expr  # Simplified for now

    def are_equivalent(self, g1: Optional[str], g2: Optional[str]) -> bool:
        """Check if two guards are equivalent."""
        return self.canonicalize(g1) == self.canonicalize(g2)


# =============================================================================
# Action Factoring
# =============================================================================

@dataclass
class ActionSequence:
    """A sequence of actions that can be shared."""
    actions: List[str]
    id: int
    usage_count: int = 0

    def signature(self) -> str:
        return ";".join(self.actions)


class ActionFactorizer:
    """
    Extracts common action subsequences for sharing.

    Uses suffix tree or similar for finding repeated patterns.
    """

    def __init__(self):
        self.sequences: Dict[str, ActionSequence] = {}
        self._next_id = 0

    def _split_actions(self, action_str: Optional[str]) -> List[str]:
        """Split action string into individual actions."""
        if not action_str:
            return []
        return [a.strip() for a in action_str.split(";") if a.strip()]

    def _get_or_create_sequence(self, actions: List[str]) -> ActionSequence:
        """Get existing sequence or create new one."""
        sig = ";".join(actions)
        if sig not in self.sequences:
            self.sequences[sig] = ActionSequence(
                actions=actions,
                id=self._next_id,
            )
            self._next_id += 1
        seq = self.sequences[sig]
        seq.usage_count += 1
        return seq

    def factor_actions(
        self,
        statechart: Statechart,
        min_length: int = 2,
        min_occurrences: int = 2,
    ) -> Dict[str, List[ActionSequence]]:
        """
        Find repeated action patterns.

        Returns mapping from original action string to factored sequences.
        """
        # Collect all action strings
        all_actions = []
        for t in statechart.transitions:
            if t.action:
                all_actions.append(t.action)

        for state in statechart.states.values():
            if state.entry_action:
                all_actions.append(state.entry_action)
            if state.exit_action:
                all_actions.append(state.exit_action)

        # Count subsequences
        subsequence_count: Dict[str, int] = defaultdict(int)
        for action_str in all_actions:
            actions = self._split_actions(action_str)
            for length in range(min_length, len(actions) + 1):
                for start in range(len(actions) - length + 1):
                    subseq = ";".join(actions[start:start + length])
                    subsequence_count[subseq] += 1

        # Find frequent subsequences
        frequent = {
            sig: count for sig, count in subsequence_count.items()
            if count >= min_occurrences
        }

        # Create sequences for frequent patterns
        result: Dict[str, List[ActionSequence]] = {}
        for action_str in set(all_actions):
            actions = self._split_actions(action_str)
            sequences = []

            # Greedy matching of frequent subsequences
            i = 0
            while i < len(actions):
                # Find longest matching frequent subsequence
                best_len = 1
                for length in range(len(actions) - i, 0, -1):
                    subseq = ";".join(actions[i:i + length])
                    if subseq in frequent:
                        seq = self._get_or_create_sequence(actions[i:i + length])
                        sequences.append(seq)
                        best_len = length
                        break
                else:
                    # No match, use single action
                    seq = self._get_or_create_sequence([actions[i]])
                    sequences.append(seq)

                i += best_len

            result[action_str] = sequences

        return result

    def get_compression_stats(self) -> Dict[str, any]:
        """Get statistics about factorization."""
        total_uses = sum(s.usage_count for s in self.sequences.values())
        shared = sum(1 for s in self.sequences.values() if s.usage_count > 1)

        return {
            "total_sequences": len(self.sequences),
            "shared_sequences": shared,
            "total_uses": total_uses,
            "sharing_ratio": shared / len(self.sequences) if self.sequences else 0,
        }


# =============================================================================
# Transition Pattern Sharing
# =============================================================================

class TransitionSharer:
    """
    Compresses statecharts by sharing transition patterns.
    """

    def __init__(self, statechart: Statechart):
        self.sc = statechart
        self.guard_canon = GuardCanonicalizer()
        self.templates: Dict[str, TransitionTemplate] = {}
        self._next_template_id = 0

    def _get_pattern(self, t: Transition) -> TransitionPattern:
        """Extract pattern from transition."""
        source_type = self.sc.states[t.source].state_type if t.source in self.sc.states else StateType.BASIC
        target_type = self.sc.states[t.target].state_type if t.target in self.sc.states else StateType.BASIC

        return TransitionPattern(
            event=t.event,
            guard=self.guard_canon.canonicalize(t.guard),
            action=t.action,
            source_type=source_type,
            target_type=target_type,
        )

    def _get_or_create_template(self, pattern: TransitionPattern) -> TransitionTemplate:
        """Get existing template or create new one."""
        sig = pattern.signature()
        if sig not in self.templates:
            self.templates[sig] = TransitionTemplate(
                pattern=pattern,
                id=self._next_template_id,
            )
            self._next_template_id += 1
        return self.templates[sig]

    def analyze_sharing(self) -> Dict[str, any]:
        """Analyze potential for transition sharing."""
        # Group transitions by pattern
        pattern_groups: Dict[str, List[Transition]] = defaultdict(list)

        for t in self.sc.transitions:
            pattern = self._get_pattern(t)
            sig = pattern.signature()
            pattern_groups[sig].append(t)

        # Count patterns
        unique_patterns = len(pattern_groups)
        total_transitions = len(self.sc.transitions)

        # Shared patterns (used more than once)
        shared = sum(1 for group in pattern_groups.values() if len(group) > 1)
        sharing_instances = sum(len(g) for g in pattern_groups.values() if len(g) > 1)

        return {
            "total_transitions": total_transitions,
            "unique_patterns": unique_patterns,
            "shared_patterns": shared,
            "sharing_instances": sharing_instances,
            "compression_potential": 1 - unique_patterns / total_transitions if total_transitions > 0 else 0,
        }

    def create_shared_representation(self) -> Tuple[List[TransitionTemplate], List[Tuple[int, str, str]]]:
        """
        Create shared representation of transitions.

        Returns:
            (templates, instances) where instances are (template_id, source, target)
        """
        # Create templates
        for t in self.sc.transitions:
            pattern = self._get_pattern(t)
            template = self._get_or_create_template(pattern)
            template.usage_count += 1
            template.instances.append((t.source, t.target))

        templates = list(self.templates.values())
        instances = [
            (template.id, src, tgt)
            for template in templates
            for src, tgt in template.instances
        ]

        return templates, instances

    def get_compressed_size(self) -> Tuple[int, int]:
        """
        Estimate compressed size.

        Returns:
            (original_size, compressed_size) in abstract units
        """
        # Original: each transition has all fields
        # Assume: event(4) + guard(8) + action(8) + source(4) + target(4) = 28 bytes per trans
        original_per_trans = 28
        original_size = len(self.sc.transitions) * original_per_trans

        # Compressed: templates + instances
        # Template: event(4) + guard(8) + action(8) + types(2) = 22 bytes
        # Instance: template_id(2) + source(4) + target(4) = 10 bytes
        templates, instances = self.create_shared_representation()
        compressed_size = len(templates) * 22 + len(instances) * 10

        return original_size, compressed_size

    def compress(self) -> Statechart:
        """
        Create compressed statechart with shared transitions.

        Note: The compressed form is conceptual - actual storage would
        use templates + instances format.
        """
        # For now, just deduplicate equivalent transitions
        compressed = Statechart(name=self.sc.name + "_shared")

        # Copy states
        for label, state in self.sc.states.items():
            compressed.add_state(State(
                label=state.label,
                state_type=state.state_type,
                is_initial=state.is_initial,
                is_final=state.is_final,
                parent=state.parent,
                entry_action=state.entry_action,
                exit_action=state.exit_action,
            ))

        # Add transitions with canonicalized guards
        seen = set()
        for t in self.sc.transitions:
            canon_guard = self.guard_canon.canonicalize(t.guard)
            key = (t.source, t.target, t.event, canon_guard, t.action)
            if key not in seen:
                compressed.add_transition(Transition(
                    source=t.source,
                    target=t.target,
                    event=t.event,
                    guard=canon_guard,
                    action=t.action,
                ))
                seen.add(key)

        compressed.initial_state = self.sc.initial_state
        return compressed


# =============================================================================
# Subgraph Sharing
# =============================================================================

@dataclass(frozen=True)
class SubgraphSignature:
    """Signature of a state subgraph for matching."""
    n_states: int
    n_transitions: int
    state_types: Tuple[StateType, ...]
    transition_events: Tuple[str, ...]


class SubgraphSharer:
    """
    Identifies and shares repeated subgraph patterns.
    """

    def __init__(self, statechart: Statechart):
        self.sc = statechart

    def _get_subgraph(self, root: str, depth: int = 2) -> Tuple[Set[str], List[Transition]]:
        """Extract subgraph rooted at state up to given depth."""
        states = {root}
        frontier = {root}

        for _ in range(depth):
            new_frontier = set()
            for state in frontier:
                for t in self.sc.get_outgoing(state):
                    if t.target not in states:
                        states.add(t.target)
                        new_frontier.add(t.target)
            frontier = new_frontier

        # Get internal transitions
        transitions = [
            t for t in self.sc.transitions
            if t.source in states and t.target in states
        ]

        return states, transitions

    def _get_signature(self, states: Set[str], transitions: List[Transition]) -> SubgraphSignature:
        """Get signature for subgraph matching."""
        state_types = tuple(sorted(
            self.sc.states[s].state_type for s in states
        ))
        trans_events = tuple(sorted(t.event for t in transitions))

        return SubgraphSignature(
            n_states=len(states),
            n_transitions=len(transitions),
            state_types=state_types,
            transition_events=trans_events,
        )

    def find_repeated_subgraphs(
        self,
        min_size: int = 3,
        depth: int = 2,
    ) -> Dict[SubgraphSignature, List[str]]:
        """
        Find subgraphs that appear multiple times.

        Returns mapping from signature to list of root states.
        """
        sig_to_roots: Dict[SubgraphSignature, List[str]] = defaultdict(list)

        for state_label in self.sc.states:
            states, transitions = self._get_subgraph(state_label, depth)
            if len(states) >= min_size:
                sig = self._get_signature(states, transitions)
                sig_to_roots[sig].append(state_label)

        # Filter to repeated subgraphs
        return {
            sig: roots for sig, roots in sig_to_roots.items()
            if len(roots) > 1
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate transition sharing compression."""
    print("=" * 60)
    print("Transition Sharing Compression")
    print("=" * 60)

    # Create a statechart with repeated patterns
    sc = Statechart(name="repeated_patterns")

    # Add states
    for i in range(8):
        sc.add_state(State(f"s{i}", is_initial=(i == 0), is_final=(i == 7)))

    # Add transitions with repeated patterns
    # Pattern 1: guard "x > 0" with action "inc(y)"
    sc.add_transition(Transition("s0", "s1", "e1", "x > 0", "inc(y)"))
    sc.add_transition(Transition("s2", "s3", "e1", "x > 0", "inc(y)"))  # Same pattern
    sc.add_transition(Transition("s4", "s5", "e1", "x > 0", "inc(y)"))  # Same pattern

    # Pattern 2: guard "y == 0" with action "reset(x)"
    sc.add_transition(Transition("s1", "s2", "e2", "y == 0", "reset(x)"))
    sc.add_transition(Transition("s3", "s4", "e2", "y == 0", "reset(x)"))  # Same pattern

    # Unique transitions
    sc.add_transition(Transition("s5", "s6", "e3", "z > 5", "special()"))
    sc.add_transition(Transition("s6", "s7", "e4", None, "finish()"))

    print(f"\nOriginal statechart:")
    print(f"  States: {len(sc.states)}")
    print(f"  Transitions: {len(sc.transitions)}")

    # Analyze sharing
    sharer = TransitionSharer(sc)
    analysis = sharer.analyze_sharing()

    print(f"\nPattern analysis:")
    print(f"  Unique patterns: {analysis['unique_patterns']}")
    print(f"  Shared patterns: {analysis['shared_patterns']}")
    print(f"  Compression potential: {analysis['compression_potential']:.1%}")

    # Get compressed size
    orig_size, comp_size = sharer.get_compressed_size()
    reduction = (1 - comp_size / orig_size) * 100

    print(f"\nSize comparison:")
    print(f"  Original size: {orig_size} bytes")
    print(f"  Compressed size: {comp_size} bytes")
    print(f"  Reduction: {reduction:.1f}%")

    # Action factoring demo
    print("\n" + "-" * 40)
    print("Action Factoring Demo:")

    sc2 = Statechart(name="action_demo")
    sc2.add_state(State("s0", is_initial=True))
    sc2.add_state(State("s1"))
    sc2.add_state(State("s2"))

    # Repeated action sequences
    sc2.add_transition(Transition("s0", "s1", "e1", action="a(); b(); c()"))
    sc2.add_transition(Transition("s1", "s2", "e2", action="a(); b(); d()"))
    sc2.add_transition(Transition("s2", "s0", "e3", action="a(); b(); c()"))

    factorizer = ActionFactorizer()
    factored = factorizer.factor_actions(sc2)
    stats = factorizer.get_compression_stats()

    print(f"  Factorization result:")
    print(f"    Total sequences: {stats['total_sequences']}")
    print(f"    Shared sequences: {stats['shared_sequences']}")
    print(f"    Sharing ratio: {stats['sharing_ratio']:.1%}")

    return sharer


if __name__ == "__main__":
    demo()
