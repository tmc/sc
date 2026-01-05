"""
Topology Transfer: Extract and Adapt Statechart Structure Across Games

KEY INSIGHT:
The STRUCTURE of a statechart (AND/OR states, hierarchy depth, history usage)
captures game-agnostic patterns that can transfer across different games.

Process:
1. EXTRACT: Pull topology from trained source genome
   - State hierarchy (parent relationships)
   - State types (BASIC, OR, AND)
   - History flags
   - Transition structure (src/tgt pairs)
   - Discard: guard parameters, event indices

2. ADAPT: Resize for target game
   - Map events: 9 (TicTacToe) -> 7 (Connect4) -> 64 (Othello)
   - Scale transitions proportionally
   - Preserve structural ratios

3. INITIALIZE: Create target genome with transferred topology
   - Re-randomize guards (to be fine-tuned)
   - Map events to target game space
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Type, Optional
from enum import IntEnum

# Import from topology evolution framework
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_topology_evolution.environments import (
    GameEnvironment, TicTacToeEnv, Connect4Env, OthelloEnv
)
from exp_topology_evolution.evolve import (
    StatechartGenome, StateType, Transition, create_random_genome
)


# =============================================================================
# Topology Representation (Game-Agnostic)
# =============================================================================

@dataclass
class AbstractTopology:
    """
    Game-agnostic statechart topology.

    Contains only structural information, no game-specific parameters.
    """
    n_states: int
    parent: List[int]              # Hierarchy structure
    state_type: List[StateType]    # BASIC, OR, AND
    has_history: List[bool]        # History flags
    initial_state: int

    # Transition structure (normalized)
    # Each is (src_idx, tgt_idx, relative_event, guard_class)
    # relative_event in [0, 1] - will be scaled to target game
    transition_structure: List[Tuple[int, int, float, int]]

    # Structural statistics (for analysis)
    hierarchy_depth: int = 0
    n_parallel_states: int = 0
    n_history_states: int = 0
    n_or_states: int = 0

    def __post_init__(self):
        """Compute structural statistics."""
        self.n_parallel_states = sum(1 for t in self.state_type if t == StateType.AND)
        self.n_history_states = sum(self.has_history)
        self.n_or_states = sum(1 for t in self.state_type if t == StateType.OR)

        # Compute max depth
        def get_depth(idx):
            if self.parent[idx] == -1:
                return 0
            return 1 + get_depth(self.parent[idx])

        self.hierarchy_depth = max(get_depth(i) for i in range(self.n_states))

    def summary(self) -> str:
        """Human-readable topology summary."""
        return (
            f"AbstractTopology:\n"
            f"  States: {self.n_states}\n"
            f"  Hierarchy Depth: {self.hierarchy_depth}\n"
            f"  Parallel (AND): {self.n_parallel_states}\n"
            f"  Exclusive (OR): {self.n_or_states}\n"
            f"  History States: {self.n_history_states}\n"
            f"  Transitions: {len(self.transition_structure)}"
        )


# =============================================================================
# Topology Extraction
# =============================================================================

def extract_topology(genome: StatechartGenome, source_n_events: int) -> AbstractTopology:
    """
    Extract game-agnostic topology from a trained genome.

    Args:
        genome: Trained statechart genome
        source_n_events: Number of events in source game (for normalization)

    Returns:
        AbstractTopology with structural information only
    """
    # Normalize transition events to [0, 1] range
    transition_structure = []
    for t in genome.transitions:
        relative_event = t.event / max(1, source_n_events - 1)
        transition_structure.append((
            t.src,
            t.tgt,
            relative_event,
            t.guard_idx  # Keep guard class, not parameters
        ))

    return AbstractTopology(
        n_states=genome.n_states,
        parent=genome.parent.copy(),
        state_type=genome.state_type.copy(),
        has_history=genome.has_history.copy(),
        initial_state=genome.initial_state,
        transition_structure=transition_structure
    )


# =============================================================================
# Topology Adaptation
# =============================================================================

def adapt_topology(
    topology: AbstractTopology,
    target_env: Type[GameEnvironment],
    scale_transitions: bool = True
) -> StatechartGenome:
    """
    Adapt extracted topology to a target game.

    Args:
        topology: Extracted abstract topology
        target_env: Target game environment class
        scale_transitions: Whether to scale transition count to target game

    Returns:
        New StatechartGenome adapted for target game
    """
    target_n_events = target_env.config.n_positions
    target_n_guards = 4  # Default guard count

    # Scale events to target game
    transitions = []
    for src, tgt, rel_event, guard_class in topology.transition_structure:
        # Map relative event to target game's event space
        target_event = int(rel_event * (target_n_events - 1))
        target_event = min(target_event, target_n_events - 1)

        transitions.append(Transition(
            src=src,
            tgt=tgt,
            event=target_event,
            guard_idx=guard_class % target_n_guards  # Keep guard class
        ))

    # If scaling, add transitions proportional to board size ratio
    if scale_transitions:
        source_size = len(topology.transition_structure)
        target_ratio = target_n_events / 9  # Relative to TicTacToe

        if target_ratio > 1.5:
            # Add more transitions for larger games
            n_extra = int(source_size * (target_ratio - 1) * 0.3)
            leaves = [i for i in range(topology.n_states)
                      if topology.state_type[i] == StateType.BASIC]

            for _ in range(n_extra):
                if len(leaves) >= 2:
                    src = random.choice(leaves)
                    tgt = random.choice(leaves)
                    event = random.randint(0, target_n_events - 1)
                    guard = random.randint(0, target_n_guards - 1)
                    transitions.append(Transition(src, tgt, event, guard))

    return StatechartGenome(
        n_states=topology.n_states,
        parent=topology.parent.copy(),
        state_type=topology.state_type.copy(),
        has_history=topology.has_history.copy(),
        transitions=transitions,
        initial_state=topology.initial_state,
        n_events=target_n_events,
        n_guards=target_n_guards,
        max_states=max(topology.n_states + 5, 15)
    )


# =============================================================================
# Topology Transfer Class
# =============================================================================

class TopologyTransfer:
    """
    Complete topology transfer pipeline.

    Usage:
        transfer = TopologyTransfer(source_genome, TicTacToeEnv, Connect4Env)
        target_genome = transfer.transfer()
    """

    def __init__(
        self,
        source_genome: StatechartGenome,
        source_env: Type[GameEnvironment],
        target_env: Type[GameEnvironment]
    ):
        self.source_genome = source_genome
        self.source_env = source_env
        self.target_env = target_env

        # Extract topology immediately
        self.topology = extract_topology(
            source_genome,
            source_env.config.n_positions
        )

    def transfer(
        self,
        scale_transitions: bool = True,
        randomize_guards: bool = True
    ) -> StatechartGenome:
        """
        Transfer topology to target game.

        Args:
            scale_transitions: Add transitions for larger games
            randomize_guards: Re-randomize guard indices

        Returns:
            New genome with transferred topology
        """
        genome = adapt_topology(
            self.topology,
            self.target_env,
            scale_transitions=scale_transitions
        )

        # Optionally randomize guards for fresh fine-tuning
        if randomize_guards:
            for t in genome.transitions:
                t.guard_idx = random.randint(0, genome.n_guards - 1)

        return genome

    def analyze_transfer(self) -> Dict:
        """Analyze the topology being transferred."""
        return {
            "source_game": self.source_env.config.name,
            "target_game": self.target_env.config.name,
            "topology": {
                "n_states": self.topology.n_states,
                "hierarchy_depth": self.topology.hierarchy_depth,
                "n_parallel": self.topology.n_parallel_states,
                "n_history": self.topology.n_history_states,
                "n_transitions": len(self.topology.transition_structure)
            },
            "scaling": {
                "source_events": self.source_env.config.n_positions,
                "target_events": self.target_env.config.n_positions,
                "ratio": self.target_env.config.n_positions / self.source_env.config.n_positions
            }
        }


# =============================================================================
# Transfer Chain Support
# =============================================================================

def transfer_chain(
    source_genome: StatechartGenome,
    env_chain: List[Type[GameEnvironment]]
) -> List[StatechartGenome]:
    """
    Transfer topology through a chain of games.

    Example: TicTacToe -> Connect4 -> Othello

    Args:
        source_genome: Initial trained genome
        env_chain: List of environment classes (source first)

    Returns:
        List of genomes, one per game in chain
    """
    genomes = [source_genome]

    for i in range(len(env_chain) - 1):
        transfer = TopologyTransfer(
            genomes[-1],
            env_chain[i],
            env_chain[i + 1]
        )
        genomes.append(transfer.transfer())

    return genomes


# =============================================================================
# Testing
# =============================================================================

def test_topology_transfer():
    """Test topology transfer functionality."""
    print("=" * 60)
    print("TOPOLOGY TRANSFER TEST")
    print("=" * 60)

    # Create a "trained" TicTacToe genome (simulated)
    source = create_random_genome(
        max_states=8,
        n_events=9,  # TicTacToe
        n_guards=4
    )

    # Ensure it's valid
    while not source.is_valid():
        source = create_random_genome(max_states=8, n_events=9, n_guards=4)

    print(f"\nSource genome (TicTacToe):")
    print(f"  States: {source.n_states}")
    print(f"  Transitions: {len(source.transitions)}")
    print(f"  Events: {source.n_events}")

    # Extract topology
    topology = extract_topology(source, 9)
    print(f"\n{topology.summary()}")

    # Transfer to Connect4
    print("\n" + "-" * 60)
    print("Transferring to Connect4...")

    transfer_c4 = TopologyTransfer(source, TicTacToeEnv, Connect4Env)
    c4_genome = transfer_c4.transfer()

    print(f"\nTarget genome (Connect4):")
    print(f"  States: {c4_genome.n_states}")
    print(f"  Transitions: {len(c4_genome.transitions)}")
    print(f"  Events: {c4_genome.n_events}")
    print(f"  Valid: {c4_genome.is_valid()}")

    # Analyze transfer
    analysis = transfer_c4.analyze_transfer()
    print(f"\nTransfer analysis:")
    print(f"  Source: {analysis['source_game']}")
    print(f"  Target: {analysis['target_game']}")
    print(f"  Event scaling: {analysis['scaling']['source_events']} -> {analysis['scaling']['target_events']}")

    # Test chain transfer
    print("\n" + "-" * 60)
    print("Testing chain transfer: TicTacToe -> Connect4 -> Othello")

    chain = transfer_chain(source, [TicTacToeEnv, Connect4Env, OthelloEnv])
    for i, (env, genome) in enumerate(zip([TicTacToeEnv, Connect4Env, OthelloEnv], chain)):
        print(f"  {env.config.name}: {genome.n_states} states, {len(genome.transitions)} transitions")

    print("\nTopology transfer test complete!")
    return topology, c4_genome


if __name__ == "__main__":
    test_topology_transfer()
