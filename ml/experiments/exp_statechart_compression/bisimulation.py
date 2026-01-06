"""
Bisimulation-Based State Compression

Compresses statecharts by merging bisimilar states - states that have
identical observable behavior and cannot be distinguished by any sequence
of events.

KEY CONCEPT:
Two states s1 and s2 are bisimilar if:
1. They have the same output (accepting/rejecting, actions)
2. For every transition s1 --e--> t1, there exists s2 --e--> t2
   where t1 and t2 are also bisimilar (and vice versa)

ALGORITHM:
1. Start with partition by output (accepting vs non-accepting)
2. Iteratively refine: split blocks where states have different successors
3. Merge states within same final block

This is the standard minimization algorithm for DFAs.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, FrozenSet
from enum import IntEnum
import copy


# =============================================================================
# Core Types
# =============================================================================

class StateType(IntEnum):
    """State types in statechart."""
    BASIC = 0
    OR = 1      # XOR composition
    AND = 2     # Parallel composition


@dataclass
class State:
    """A state in the statechart."""
    label: str
    state_type: StateType = StateType.BASIC
    is_initial: bool = False
    is_final: bool = False
    parent: Optional[str] = None

    # Output/actions (for bisimulation equivalence)
    entry_action: Optional[str] = None
    exit_action: Optional[str] = None

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.label == other.label
        return False


@dataclass
class Transition:
    """A transition in the statechart."""
    source: str
    target: str
    event: str
    guard: Optional[str] = None
    action: Optional[str] = None

    def __hash__(self):
        return hash((self.source, self.target, self.event, self.guard))


@dataclass
class Statechart:
    """A statechart with states and transitions."""
    name: str
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    initial_state: Optional[str] = None

    def add_state(self, state: State):
        """Add a state to the statechart."""
        self.states[state.label] = state
        if state.is_initial and self.initial_state is None:
            self.initial_state = state.label

    def add_transition(self, transition: Transition):
        """Add a transition to the statechart."""
        self.transitions.append(transition)

    def get_outgoing(self, state_label: str) -> List[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.source == state_label]

    def get_incoming(self, state_label: str) -> List[Transition]:
        """Get all transitions to a state."""
        return [t for t in self.transitions if t.target == state_label]

    def get_events(self) -> Set[str]:
        """Get all events used in transitions."""
        return {t.event for t in self.transitions}

    def size(self) -> Tuple[int, int]:
        """Return (n_states, n_transitions)."""
        return len(self.states), len(self.transitions)

    def copy(self) -> "Statechart":
        """Create a deep copy."""
        new_sc = Statechart(name=self.name + "_copy")
        for label, state in self.states.items():
            new_state = State(
                label=state.label,
                state_type=state.state_type,
                is_initial=state.is_initial,
                is_final=state.is_final,
                parent=state.parent,
                entry_action=state.entry_action,
                exit_action=state.exit_action,
            )
            new_sc.add_state(new_state)
        for t in self.transitions:
            new_sc.add_transition(Transition(
                source=t.source,
                target=t.target,
                event=t.event,
                guard=t.guard,
                action=t.action,
            ))
        new_sc.initial_state = self.initial_state
        return new_sc


# =============================================================================
# Bisimulation Partition Refinement
# =============================================================================

class BisimulationCompressor:
    """
    Compresses statecharts using bisimulation-based state merging.

    Uses partition refinement algorithm (Hopcroft's algorithm variant).
    """

    def __init__(self, statechart: Statechart):
        self.sc = statechart
        self.partition: List[Set[str]] = []
        self.state_to_block: Dict[str, int] = {}

    def _get_signature(self, state_label: str) -> Tuple:
        """
        Get signature of a state for partitioning.

        Signature includes:
        - Output properties (is_final, actions)
        - For each event, the block(s) of successor states
        """
        state = self.sc.states[state_label]

        # Output signature
        output = (
            state.is_final,
            state.entry_action,
            state.exit_action,
            state.state_type,
        )

        # Transition signatures: event -> frozenset of target blocks
        trans_sig = {}
        for t in self.sc.get_outgoing(state_label):
            key = (t.event, t.guard, t.action)
            target_block = self.state_to_block.get(t.target, -1)
            if key not in trans_sig:
                trans_sig[key] = set()
            trans_sig[key].add(target_block)

        # Convert to hashable
        trans_tuple = tuple(
            (k, frozenset(v)) for k, v in sorted(trans_sig.items())
        )

        return (output, trans_tuple)

    def _initial_partition(self):
        """Create initial partition based on output properties."""
        # Group by output signature (is_final, actions, type)
        groups: Dict[Tuple, Set[str]] = {}

        for label, state in self.sc.states.items():
            sig = (
                state.is_final,
                state.entry_action,
                state.exit_action,
                state.state_type,
            )
            if sig not in groups:
                groups[sig] = set()
            groups[sig].add(label)

        self.partition = list(groups.values())
        self._update_state_to_block()

    def _update_state_to_block(self):
        """Update state -> block index mapping."""
        self.state_to_block = {}
        for i, block in enumerate(self.partition):
            for state in block:
                self.state_to_block[state] = i

    def _refine_partition(self) -> bool:
        """
        Refine partition by splitting blocks with different signatures.

        Returns True if partition changed.
        """
        new_partition = []
        changed = False

        for block in self.partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue

            # Group states by signature
            sig_groups: Dict[Tuple, Set[str]] = {}
            for state in block:
                sig = self._get_signature(state)
                if sig not in sig_groups:
                    sig_groups[sig] = set()
                sig_groups[sig].add(state)

            # If more than one group, we split
            if len(sig_groups) > 1:
                changed = True
                for group in sig_groups.values():
                    new_partition.append(group)
            else:
                new_partition.append(block)

        self.partition = new_partition
        self._update_state_to_block()
        return changed

    def compute_bisimulation(self) -> List[Set[str]]:
        """
        Compute bisimulation equivalence classes.

        Returns list of state sets where states in same set are bisimilar.
        """
        self._initial_partition()

        # Refine until fixed point
        max_iterations = len(self.sc.states) + 1
        for _ in range(max_iterations):
            if not self._refine_partition():
                break

        return self.partition

    def compress(self) -> Statechart:
        """
        Create compressed statechart by merging bisimilar states.

        Returns new statechart with merged states.
        """
        # Compute equivalence classes
        partition = self.compute_bisimulation()

        # Create representative for each class
        class_rep: Dict[int, str] = {}  # block_idx -> representative label
        state_to_rep: Dict[str, str] = {}  # state -> representative

        for i, block in enumerate(partition):
            # Use first state as representative, prefer initial state
            sorted_block = sorted(block)
            rep = sorted_block[0]
            for s in sorted_block:
                if self.sc.states[s].is_initial:
                    rep = s
                    break

            class_rep[i] = rep
            for s in block:
                state_to_rep[s] = rep

        # Build compressed statechart
        compressed = Statechart(name=self.sc.name + "_compressed")

        # Add representative states
        added_states = set()
        for i, block in enumerate(partition):
            rep = class_rep[i]
            if rep not in added_states:
                old_state = self.sc.states[rep]
                new_state = State(
                    label=rep,
                    state_type=old_state.state_type,
                    is_initial=old_state.is_initial or any(
                        self.sc.states[s].is_initial for s in block
                    ),
                    is_final=old_state.is_final,
                    parent=state_to_rep.get(old_state.parent) if old_state.parent else None,
                    entry_action=old_state.entry_action,
                    exit_action=old_state.exit_action,
                )
                compressed.add_state(new_state)
                added_states.add(rep)

        # Add transitions (deduplicated)
        added_transitions = set()
        for t in self.sc.transitions:
            new_source = state_to_rep[t.source]
            new_target = state_to_rep[t.target]

            trans_key = (new_source, new_target, t.event, t.guard, t.action)
            if trans_key not in added_transitions:
                compressed.add_transition(Transition(
                    source=new_source,
                    target=new_target,
                    event=t.event,
                    guard=t.guard,
                    action=t.action,
                ))
                added_transitions.add(trans_key)

        # Set initial state
        if self.sc.initial_state:
            compressed.initial_state = state_to_rep[self.sc.initial_state]

        return compressed

    def get_compression_ratio(self) -> float:
        """Get compression ratio (compressed_size / original_size)."""
        partition = self.compute_bisimulation()
        original_states = len(self.sc.states)
        compressed_states = len(partition)
        return compressed_states / original_states if original_states > 0 else 1.0


# =============================================================================
# Utilities
# =============================================================================

def create_sample_statechart_with_redundancy() -> Statechart:
    """Create a sample statechart with redundant states for testing."""
    sc = Statechart(name="redundant_sample")

    # States with redundancy: s1 and s2 are bisimilar, s3 and s4 are bisimilar
    sc.add_state(State("s0", is_initial=True))
    sc.add_state(State("s1"))  # These two are bisimilar
    sc.add_state(State("s2"))  #
    sc.add_state(State("s3", is_final=True))  # These two are bisimilar
    sc.add_state(State("s4", is_final=True))  #
    sc.add_state(State("s5"))

    # Transitions that make s1≈s2 and s3≈s4
    sc.add_transition(Transition("s0", "s1", "a"))
    sc.add_transition(Transition("s0", "s2", "b"))
    sc.add_transition(Transition("s1", "s3", "c"))
    sc.add_transition(Transition("s1", "s5", "d"))
    sc.add_transition(Transition("s2", "s4", "c"))  # Same as s1
    sc.add_transition(Transition("s2", "s5", "d"))  # Same as s1
    sc.add_transition(Transition("s3", "s0", "e"))
    sc.add_transition(Transition("s4", "s0", "e"))  # Same as s3
    sc.add_transition(Transition("s5", "s0", "f"))

    return sc


def create_dfa_statechart(
    n_states: int = 10,
    n_events: int = 3,
    redundancy_factor: float = 0.3,
) -> Statechart:
    """
    Create a random DFA-style statechart with controllable redundancy.

    Args:
        n_states: Number of states
        n_events: Number of distinct events
        redundancy_factor: Fraction of states that are redundant copies
    """
    import random

    sc = Statechart(name=f"dfa_{n_states}_{n_events}")
    events = [f"e{i}" for i in range(n_events)]

    # Create states
    n_unique = int(n_states * (1 - redundancy_factor))
    n_redundant = n_states - n_unique

    # Unique states
    for i in range(n_unique):
        is_final = random.random() < 0.3
        sc.add_state(State(f"s{i}", is_initial=(i == 0), is_final=is_final))

    # Redundant copies (will be bisimilar to existing states)
    for i in range(n_redundant):
        template_idx = random.randint(0, n_unique - 1)
        template = sc.states[f"s{template_idx}"]
        sc.add_state(State(
            f"s{n_unique + i}",
            is_final=template.is_final,
        ))

    # Add transitions - ensure redundant states have same transitions as templates
    for state_label in sc.states:
        for event in events:
            # Random target
            target_idx = random.randint(0, n_unique - 1)
            sc.add_transition(Transition(
                source=state_label,
                target=f"s{target_idx}",
                event=event,
            ))

    return sc


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate bisimulation compression."""
    print("=" * 60)
    print("Bisimulation-Based State Compression")
    print("=" * 60)

    # Create sample with redundancy
    sc = create_sample_statechart_with_redundancy()
    print(f"\nOriginal statechart:")
    print(f"  States: {len(sc.states)}")
    print(f"  Transitions: {len(sc.transitions)}")

    # Compress
    compressor = BisimulationCompressor(sc)
    partition = compressor.compute_bisimulation()

    print(f"\nBisimulation classes:")
    for i, block in enumerate(partition):
        print(f"  Block {i}: {block}")

    compressed = compressor.compress()
    print(f"\nCompressed statechart:")
    print(f"  States: {len(compressed.states)}")
    print(f"  Transitions: {len(compressed.transitions)}")

    ratio = compressor.get_compression_ratio()
    reduction = (1 - ratio) * 100
    print(f"\nCompression: {reduction:.1f}% state reduction")

    # Test on larger random statechart
    print("\n" + "-" * 40)
    print("Random DFA with 30% redundancy:")

    sc2 = create_dfa_statechart(n_states=20, n_events=3, redundancy_factor=0.3)
    print(f"  Original: {len(sc2.states)} states, {len(sc2.transitions)} transitions")

    compressor2 = BisimulationCompressor(sc2)
    compressed2 = compressor2.compress()
    print(f"  Compressed: {len(compressed2.states)} states, {len(compressed2.transitions)} transitions")

    ratio2 = len(compressed2.states) / len(sc2.states)
    reduction2 = (1 - ratio2) * 100
    print(f"  Reduction: {reduction2:.1f}%")

    return compressor, compressed


if __name__ == "__main__":
    demo()
