"""
Scale Generator: Generate Statecharts at Various Scales

Creates statecharts with 10 to 1,000,000 states for benchmarking.

Topologies:
1. Flat - All states at same level (stress test for large OR-states)
2. Deep - Linear chain of depth N (stress test for hierarchy)
3. Wide - Shallow tree with high branching factor
4. Balanced - Balanced tree with moderate depth and width
5. Random - Random hierarchical structure
6. Parallel - Heavy use of AND-states (orthogonal regions)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Generator, Any
from enum import Enum, auto
import random
import math
import time
import gc


class Topology(Enum):
    """Statechart topology types."""
    FLAT = "flat"           # All states at root level
    DEEP = "deep"           # Linear chain
    WIDE = "wide"           # Shallow, high branching
    BALANCED = "balanced"   # Balanced tree
    RANDOM = "random"       # Random hierarchy
    PARALLEL = "parallel"   # Heavy AND-states


# Standard scale levels for testing
SCALE_LEVELS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


@dataclass
class ScaleConfig:
    """Configuration for scale generation."""
    n_states: int
    topology: Topology = Topology.BALANCED
    transition_density: float = 1.5  # Transitions per state
    guard_probability: float = 0.3   # Probability of guarded transition
    parallel_regions: int = 0        # Number of AND-state regions (0 = auto)
    max_depth: int = 0               # Max hierarchy depth (0 = auto)
    seed: int = 42


@dataclass
class GeneratedState:
    """A generated state."""
    id: int
    name: str
    parent_id: Optional[int] = None
    is_initial: bool = False
    is_final: bool = False
    is_parallel: bool = False  # AND-state
    children: List[int] = field(default_factory=list)
    depth: int = 0


@dataclass
class GeneratedTransition:
    """A generated transition."""
    id: int
    source_id: int
    target_id: int
    event: str = ""
    guard: str = ""
    priority: int = 0


@dataclass
class GeneratedStatechart:
    """Complete generated statechart."""
    name: str
    n_states: int
    n_transitions: int
    topology: Topology
    states: Dict[int, GeneratedState] = field(default_factory=dict)
    transitions: List[GeneratedTransition] = field(default_factory=list)
    root_id: int = 0
    max_depth: int = 0
    generation_time: float = 0.0
    memory_bytes: int = 0

    def get_state_names(self) -> List[str]:
        """Get all state names."""
        return [s.name for s in self.states.values()]

    def get_leaf_states(self) -> List[GeneratedState]:
        """Get all leaf (basic) states."""
        return [s for s in self.states.values() if not s.children]

    def get_states_at_depth(self, depth: int) -> List[GeneratedState]:
        """Get states at specific depth."""
        return [s for s in self.states.values() if s.depth == depth]


class StatechartGenerator:
    """
    Generate statecharts at various scales.

    Optimized for memory efficiency at large scales.
    """

    def __init__(self, config: ScaleConfig):
        self.config = config
        random.seed(config.seed)
        self.state_counter = 0
        self.transition_counter = 0

    def generate(self) -> GeneratedStatechart:
        """Generate statechart based on configuration."""
        t0 = time.perf_counter()

        # Select generator based on topology
        generators = {
            Topology.FLAT: self._generate_flat,
            Topology.DEEP: self._generate_deep,
            Topology.WIDE: self._generate_wide,
            Topology.BALANCED: self._generate_balanced,
            Topology.RANDOM: self._generate_random,
            Topology.PARALLEL: self._generate_parallel,
        }

        generator = generators.get(self.config.topology, self._generate_balanced)
        sc = generator()

        sc.generation_time = time.perf_counter() - t0

        # Estimate memory
        sc.memory_bytes = self._estimate_memory(sc)

        return sc

    def _create_state(
        self,
        parent_id: Optional[int] = None,
        is_initial: bool = False,
        is_final: bool = False,
        is_parallel: bool = False,
        depth: int = 0,
    ) -> GeneratedState:
        """Create a new state."""
        state = GeneratedState(
            id=self.state_counter,
            name=f"s{self.state_counter}",
            parent_id=parent_id,
            is_initial=is_initial,
            is_final=is_final,
            is_parallel=is_parallel,
            depth=depth,
        )
        self.state_counter += 1
        return state

    def _create_transition(
        self,
        source_id: int,
        target_id: int,
        with_guard: bool = False,
    ) -> GeneratedTransition:
        """Create a new transition."""
        guard = ""
        if with_guard:
            var = random.choice(["x", "y", "count", "timer"])
            op = random.choice(["<", ">", "==", ">=", "<="])
            val = random.randint(0, 100)
            guard = f"{var} {op} {val}"

        trans = GeneratedTransition(
            id=self.transition_counter,
            source_id=source_id,
            target_id=target_id,
            event=f"e{self.transition_counter % 20}",
            guard=guard,
        )
        self.transition_counter += 1
        return trans

    def _generate_flat(self) -> GeneratedStatechart:
        """Generate flat statechart (all states at root)."""
        sc = GeneratedStatechart(
            name=f"flat_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.FLAT,
        )

        # Create root
        root = self._create_state(is_parallel=False, depth=0)
        sc.states[root.id] = root
        sc.root_id = root.id

        # Create all child states
        for i in range(self.config.n_states - 1):
            state = self._create_state(
                parent_id=root.id,
                is_initial=(i == 0),
                is_final=(i == self.config.n_states - 2),
                depth=1,
            )
            sc.states[state.id] = state
            root.children.append(state.id)

        sc.max_depth = 1

        # Generate transitions
        self._generate_transitions(sc)

        return sc

    def _generate_deep(self) -> GeneratedStatechart:
        """Generate deep statechart (linear chain)."""
        sc = GeneratedStatechart(
            name=f"deep_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.DEEP,
        )

        # Create linear chain
        prev_id = None
        for i in range(self.config.n_states):
            state = self._create_state(
                parent_id=prev_id,
                is_initial=(i == 0),
                is_final=(i == self.config.n_states - 1),
                depth=i,
            )
            sc.states[state.id] = state

            if prev_id is not None:
                sc.states[prev_id].children.append(state.id)

            if i == 0:
                sc.root_id = state.id

            prev_id = state.id

        sc.max_depth = self.config.n_states - 1

        # Generate transitions (mostly parent-child)
        self._generate_transitions(sc, prefer_local=True)

        return sc

    def _generate_wide(self) -> GeneratedStatechart:
        """Generate wide statechart (shallow, high branching)."""
        sc = GeneratedStatechart(
            name=f"wide_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.WIDE,
        )

        # Root
        root = self._create_state(is_parallel=False, depth=0)
        sc.states[root.id] = root
        sc.root_id = root.id

        # Calculate branching for 2 levels
        remaining = self.config.n_states - 1
        n_level1 = min(remaining, int(math.sqrt(remaining)) + 1)
        per_level1 = remaining // n_level1 if n_level1 > 0 else 0

        created = 1
        level1_states = []

        # Level 1
        for i in range(n_level1):
            if created >= self.config.n_states:
                break
            state = self._create_state(
                parent_id=root.id,
                is_initial=(i == 0),
                depth=1,
            )
            sc.states[state.id] = state
            root.children.append(state.id)
            level1_states.append(state)
            created += 1

        # Level 2
        level2_per_parent = per_level1
        for parent in level1_states:
            for j in range(level2_per_parent):
                if created >= self.config.n_states:
                    break
                state = self._create_state(
                    parent_id=parent.id,
                    is_final=(created == self.config.n_states - 1),
                    depth=2,
                )
                sc.states[state.id] = state
                parent.children.append(state.id)
                created += 1

        sc.max_depth = 2

        self._generate_transitions(sc)

        return sc

    def _generate_balanced(self) -> GeneratedStatechart:
        """Generate balanced tree statechart."""
        sc = GeneratedStatechart(
            name=f"balanced_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.BALANCED,
        )

        # Calculate depth and branching factor
        if self.config.max_depth > 0:
            max_depth = self.config.max_depth
        else:
            # Aim for depth = log_b(n) where b is branching factor
            max_depth = max(2, int(math.log(self.config.n_states, 4)))

        branching = int(math.pow(self.config.n_states, 1.0 / max_depth)) + 1

        # BFS to create balanced tree
        root = self._create_state(is_initial=True, depth=0)
        sc.states[root.id] = root
        sc.root_id = root.id

        queue = [(root, 0)]
        created = 1

        while queue and created < self.config.n_states:
            parent, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            children_to_create = min(branching, self.config.n_states - created)

            for i in range(children_to_create):
                if created >= self.config.n_states:
                    break

                state = self._create_state(
                    parent_id=parent.id,
                    is_final=(created == self.config.n_states - 1),
                    depth=depth + 1,
                )
                sc.states[state.id] = state
                parent.children.append(state.id)
                queue.append((state, depth + 1))
                created += 1

        sc.max_depth = max(s.depth for s in sc.states.values())

        self._generate_transitions(sc)

        return sc

    def _generate_random(self) -> GeneratedStatechart:
        """Generate random hierarchical statechart."""
        sc = GeneratedStatechart(
            name=f"random_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.RANDOM,
        )

        # Create root
        root = self._create_state(is_initial=True, depth=0)
        sc.states[root.id] = root
        sc.root_id = root.id

        parents = [root]
        created = 1

        while created < self.config.n_states:
            # Pick random parent
            parent = random.choice(parents)

            state = self._create_state(
                parent_id=parent.id,
                is_final=(created == self.config.n_states - 1),
                depth=parent.depth + 1,
            )
            sc.states[state.id] = state
            parent.children.append(state.id)

            # Sometimes add as potential parent
            if random.random() < 0.7:
                parents.append(state)

            created += 1

        sc.max_depth = max(s.depth for s in sc.states.values())

        self._generate_transitions(sc)

        return sc

    def _generate_parallel(self) -> GeneratedStatechart:
        """Generate statechart with many parallel (AND) regions."""
        sc = GeneratedStatechart(
            name=f"parallel_{self.config.n_states}",
            n_states=self.config.n_states,
            n_transitions=0,
            topology=Topology.PARALLEL,
        )

        # Calculate regions
        if self.config.parallel_regions > 0:
            n_regions = self.config.parallel_regions
        else:
            n_regions = max(2, int(math.sqrt(self.config.n_states / 10)))

        states_per_region = (self.config.n_states - 1) // n_regions

        # Create AND root
        root = self._create_state(is_parallel=True, depth=0)
        sc.states[root.id] = root
        sc.root_id = root.id

        created = 1

        # Create each region
        for r in range(n_regions):
            if created >= self.config.n_states:
                break

            # Region root (OR state)
            region = self._create_state(
                parent_id=root.id,
                is_initial=True,  # All regions active in AND
                depth=1,
            )
            sc.states[region.id] = region
            root.children.append(region.id)
            created += 1

            # States in this region
            for i in range(states_per_region):
                if created >= self.config.n_states:
                    break

                state = self._create_state(
                    parent_id=region.id,
                    is_initial=(i == 0),
                    depth=2,
                )
                sc.states[state.id] = state
                region.children.append(state.id)
                created += 1

        sc.max_depth = 2

        self._generate_transitions(sc)

        return sc

    def _generate_transitions(
        self,
        sc: GeneratedStatechart,
        prefer_local: bool = False,
    ):
        """Generate transitions for the statechart."""
        leaf_states = sc.get_leaf_states()

        if not leaf_states:
            return

        n_transitions = int(len(leaf_states) * self.config.transition_density)

        for _ in range(n_transitions):
            if prefer_local:
                # Prefer transitions between siblings
                source = random.choice(leaf_states)
                siblings = [s for s in leaf_states if s.parent_id == source.parent_id and s.id != source.id]
                if siblings:
                    target = random.choice(siblings)
                else:
                    target = random.choice(leaf_states)
            else:
                source = random.choice(leaf_states)
                target = random.choice(leaf_states)

            with_guard = random.random() < self.config.guard_probability
            trans = self._create_transition(source.id, target.id, with_guard)
            sc.transitions.append(trans)

        sc.n_transitions = len(sc.transitions)

    def _estimate_memory(self, sc: GeneratedStatechart) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimates
        bytes_per_state = 200  # State object overhead
        bytes_per_transition = 100
        bytes_per_child_ref = 8

        state_bytes = len(sc.states) * bytes_per_state
        transition_bytes = len(sc.transitions) * bytes_per_transition
        child_ref_bytes = sum(len(s.children) for s in sc.states.values()) * bytes_per_child_ref

        return state_bytes + transition_bytes + child_ref_bytes


def generate_at_scale(
    n_states: int,
    topology: Topology = Topology.BALANCED,
    **kwargs,
) -> GeneratedStatechart:
    """Convenience function to generate at specific scale."""
    config = ScaleConfig(n_states=n_states, topology=topology, **kwargs)
    generator = StatechartGenerator(config)
    return generator.generate()


def generate_all_scales(
    topology: Topology = Topology.BALANCED,
    scales: List[int] = None,
) -> Generator[GeneratedStatechart, None, None]:
    """Generate statecharts at all standard scales."""
    if scales is None:
        scales = SCALE_LEVELS

    for n in scales:
        gc.collect()  # Clean up before each scale
        yield generate_at_scale(n, topology)


def demo():
    """Demonstrate scale generation."""
    print("=" * 60)
    print("SCALE GENERATOR: Generate Large Statecharts")
    print("=" * 60)

    for topology in [Topology.FLAT, Topology.BALANCED, Topology.PARALLEL]:
        print(f"\n--- {topology.value.upper()} Topology ---")

        for n in [10, 100, 1000]:
            sc = generate_at_scale(n, topology)
            print(f"  n={n:,}: {sc.n_transitions} transitions, "
                  f"depth={sc.max_depth}, "
                  f"time={sc.generation_time*1000:.1f}ms, "
                  f"mem={sc.memory_bytes/1024:.1f}KB")

    return sc


if __name__ == "__main__":
    demo()
