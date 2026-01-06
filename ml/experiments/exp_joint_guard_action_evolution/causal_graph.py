"""
Causal Dependency Graph

Models the causal relationships between guards and actions:
- Action A writes variable X → Guard G reads variable X
- This creates edge: A → G (A causally enables G)

The graph helps:
1. Identify coherent guard-action clusters
2. Detect broken dependencies (guard reads unwritten variable)
3. Guide mutation (mutate causally-related pairs together)
4. Measure causal coverage (are all guards satisfied by some action?)

Key insight: Good statecharts have DENSE causal graphs where
guards are satisfied by prior actions, not external inputs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from collections import defaultdict
import random

from .joint_genome import JointGenome, GuardActionPair, Guard, Action


# =============================================================================
# Causal Edge
# =============================================================================

@dataclass(frozen=True)
class CausalEdge:
    """Edge from action to guard via shared variable."""
    source_pair_idx: int      # Index of pair whose action writes
    target_pair_idx: int      # Index of pair whose guard reads
    variable: str             # The variable creating the dependency
    strength: float = 1.0     # How strong the dependency is

    def __repr__(self):
        return f"({self.source_pair_idx})-[{self.variable}]->({self.target_pair_idx})"


# =============================================================================
# Causal Graph
# =============================================================================

@dataclass
class CausalGraph:
    """Graph of causal dependencies between guard-action pairs."""

    # Core structure
    n_pairs: int
    edges: List[CausalEdge] = field(default_factory=list)

    # Adjacency lists
    outgoing: Dict[int, List[CausalEdge]] = field(default_factory=lambda: defaultdict(list))
    incoming: Dict[int, List[CausalEdge]] = field(default_factory=lambda: defaultdict(list))

    # Variable mappings
    writers: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    readers: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, edge: CausalEdge):
        """Add causal edge."""
        self.edges.append(edge)
        self.outgoing[edge.source_pair_idx].append(edge)
        self.incoming[edge.target_pair_idx].append(edge)

    def get_causal_ancestors(self, pair_idx: int, depth: int = -1) -> Set[int]:
        """Get all pairs that causally influence this one."""
        ancestors = set()
        frontier = {pair_idx}
        current_depth = 0

        while frontier and (depth < 0 or current_depth < depth):
            next_frontier = set()
            for idx in frontier:
                for edge in self.incoming[idx]:
                    if edge.source_pair_idx not in ancestors:
                        ancestors.add(edge.source_pair_idx)
                        next_frontier.add(edge.source_pair_idx)
            frontier = next_frontier
            current_depth += 1

        return ancestors

    def get_causal_descendants(self, pair_idx: int, depth: int = -1) -> Set[int]:
        """Get all pairs causally influenced by this one."""
        descendants = set()
        frontier = {pair_idx}
        current_depth = 0

        while frontier and (depth < 0 or current_depth < depth):
            next_frontier = set()
            for idx in frontier:
                for edge in self.outgoing[idx]:
                    if edge.target_pair_idx not in descendants:
                        descendants.add(edge.target_pair_idx)
                        next_frontier.add(edge.target_pair_idx)
            frontier = next_frontier
            current_depth += 1

        return descendants

    def get_connected_component(self, pair_idx: int) -> Set[int]:
        """Get all pairs in same causal cluster."""
        return self.get_causal_ancestors(pair_idx) | \
               self.get_causal_descendants(pair_idx) | \
               {pair_idx}

    def find_all_components(self) -> List[Set[int]]:
        """Find all connected components."""
        visited = set()
        components = []

        for i in range(self.n_pairs):
            if i not in visited:
                component = self.get_connected_component(i)
                components.append(component)
                visited |= component

        return components

    def get_unsatisfied_guards(self) -> Set[int]:
        """Find guards that read variables no action writes."""
        unsatisfied = set()
        for pair_idx in range(self.n_pairs):
            if not self.incoming[pair_idx]:
                # No incoming edges = no causal support
                unsatisfied.add(pair_idx)
        return unsatisfied

    def get_unused_actions(self) -> Set[int]:
        """Find actions that write variables no guard reads."""
        unused = set()
        for pair_idx in range(self.n_pairs):
            if not self.outgoing[pair_idx]:
                unused.add(pair_idx)
        return unused

    def causal_density(self) -> float:
        """Ratio of actual edges to possible edges."""
        max_edges = self.n_pairs * (self.n_pairs - 1)
        if max_edges == 0:
            return 1.0
        return len(self.edges) / max_edges

    def causal_coverage(self) -> float:
        """Fraction of pairs with at least one incoming edge."""
        if self.n_pairs == 0:
            return 1.0
        covered = sum(1 for i in range(self.n_pairs) if self.incoming[i])
        return covered / self.n_pairs


# =============================================================================
# Graph Builder
# =============================================================================

def build_causal_graph(genome: JointGenome) -> CausalGraph:
    """Build causal graph from joint genome.

    Creates edges where:
    - Pair i's action writes variable X
    - Pair j's guard reads variable X
    → Edge: i → j
    """
    graph = CausalGraph(n_pairs=len(genome.pairs))

    # Index which pairs write/read which variables
    for i, pair in enumerate(genome.pairs):
        for var in pair.action_writes():
            graph.writers[var].add(i)
        for var in pair.guard_reads():
            graph.readers[var].add(i)

    # Create edges
    for var in graph.writers:
        writers = graph.writers[var]
        readers = graph.readers.get(var, set())

        for writer_idx in writers:
            for reader_idx in readers:
                if writer_idx != reader_idx:  # No self-loops
                    edge = CausalEdge(
                        source_pair_idx=writer_idx,
                        target_pair_idx=reader_idx,
                        variable=var,
                        strength=1.0
                    )
                    graph.add_edge(edge)

    return graph


# =============================================================================
# Causal Analysis
# =============================================================================

@dataclass
class CausalAnalysis:
    """Analysis results for a causal graph."""
    n_pairs: int
    n_edges: int
    n_components: int
    largest_component_size: int
    causal_density: float
    causal_coverage: float
    unsatisfied_guards: Set[int]
    unused_actions: Set[int]
    variable_connectivity: Dict[str, int]  # How many edges per variable

    def coherence_score(self) -> float:
        """Overall coherence: coverage + density + connectedness."""
        # Penalize isolated pairs
        connectedness = self.largest_component_size / self.n_pairs if self.n_pairs > 0 else 0

        # Penalize unsatisfied guards (bad!) and unused actions (wasteful)
        satisfaction = 1.0 - (len(self.unsatisfied_guards) / max(1, self.n_pairs))
        utilization = 1.0 - (len(self.unused_actions) / max(1, self.n_pairs))

        return (self.causal_coverage * 0.4 +
                connectedness * 0.3 +
                satisfaction * 0.2 +
                utilization * 0.1)


def analyze_causal_graph(graph: CausalGraph, genome: JointGenome) -> CausalAnalysis:
    """Analyze causal properties of graph."""
    components = graph.find_all_components()
    largest = max(len(c) for c in components) if components else 0

    # Variable connectivity
    var_connectivity = defaultdict(int)
    for edge in graph.edges:
        var_connectivity[edge.variable] += 1

    return CausalAnalysis(
        n_pairs=graph.n_pairs,
        n_edges=len(graph.edges),
        n_components=len(components),
        largest_component_size=largest,
        causal_density=graph.causal_density(),
        causal_coverage=graph.causal_coverage(),
        unsatisfied_guards=graph.get_unsatisfied_guards(),
        unused_actions=graph.get_unused_actions(),
        variable_connectivity=dict(var_connectivity)
    )


# =============================================================================
# Causal Mutation Guidance
# =============================================================================

def get_mutation_cluster(
    graph: CausalGraph,
    pair_idx: int,
    depth: int = 1
) -> Set[int]:
    """Get pairs that should be mutated together.

    When mutating pair i, we should also consider mutating:
    - Pairs that depend on i (so they stay coherent)
    - Pairs that i depends on (so the dependency stays valid)
    """
    ancestors = graph.get_causal_ancestors(pair_idx, depth=depth)
    descendants = graph.get_causal_descendants(pair_idx, depth=depth)
    return ancestors | descendants | {pair_idx}


def suggest_coherent_mutation(
    genome: JointGenome,
    graph: CausalGraph,
    pair_idx: int,
    variables: List[str],
    possible_values: Dict[str, List]
) -> Tuple[int, str]:
    """Suggest a mutation that maintains causal coherence.

    Returns: (pair_to_mutate, mutation_type)
    """
    pair = genome.pairs[pair_idx]

    # If guard reads unwritten variable, suggest adding that write to an ancestor
    guard_vars = pair.guard_reads()
    written_vars = genome.all_variables_written()
    unwritten = guard_vars - written_vars

    if unwritten:
        # Suggest action mutation to write missing variable
        var = random.choice(list(unwritten))
        ancestors = graph.get_causal_ancestors(pair_idx, depth=1)
        if ancestors:
            target = random.choice(list(ancestors))
            return (target, f"add_write:{var}")
        else:
            return (pair_idx, f"add_write:{var}")

    # If action writes unused variable, suggest guard that reads it
    action_vars = pair.action_writes()
    read_vars = genome.all_variables_read()
    unread = action_vars - read_vars

    if unread:
        var = random.choice(list(unread))
        descendants = graph.get_causal_descendants(pair_idx, depth=1)
        if descendants:
            target = random.choice(list(descendants))
            return (target, f"add_read:{var}")
        else:
            return (pair_idx, f"add_read:{var}")

    # Default: random mutation
    return (pair_idx, "random")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate causal graph analysis."""
    print("=" * 60)
    print("CAUSAL DEPENDENCY GRAPH DEMO")
    print("=" * 60)

    from .joint_genome import random_joint_genome

    variables = ["turn", "phase", "score", "ready"]
    possible_values = {
        "turn": [0, 1, 2],
        "phase": ["init", "play", "end"],
        "score": [0, 1, 2, 3],
        "ready": [True, False]
    }

    # Generate genome
    genome = random_joint_genome(
        n_states=4,
        n_events=3,
        n_pairs=8,
        variables=variables,
        possible_values=possible_values
    )

    print(f"\nGenome: {len(genome.pairs)} pairs")

    # Build causal graph
    graph = build_causal_graph(genome)

    print(f"\n--- Causal Graph ---")
    print(f"Edges: {len(graph.edges)}")
    for edge in graph.edges[:10]:
        print(f"  {edge}")
    if len(graph.edges) > 10:
        print(f"  ... and {len(graph.edges) - 10} more")

    # Analyze
    analysis = analyze_causal_graph(graph, genome)

    print(f"\n--- Analysis ---")
    print(f"Components: {analysis.n_components}")
    print(f"Largest component: {analysis.largest_component_size} pairs")
    print(f"Causal density: {analysis.causal_density:.2%}")
    print(f"Causal coverage: {analysis.causal_coverage:.2%}")
    print(f"Unsatisfied guards: {analysis.unsatisfied_guards}")
    print(f"Unused actions: {analysis.unused_actions}")
    print(f"Coherence score: {analysis.coherence_score():.3f}")

    print(f"\n--- Variable Connectivity ---")
    for var, count in sorted(analysis.variable_connectivity.items(),
                             key=lambda x: -x[1]):
        print(f"  {var}: {count} edges")

    # Test mutation guidance
    print(f"\n--- Mutation Guidance ---")
    for i in range(min(3, len(genome.pairs))):
        cluster = get_mutation_cluster(graph, i, depth=1)
        print(f"Pair {i} mutation cluster: {cluster}")

        suggestion = suggest_coherent_mutation(genome, graph, i,
                                               variables, possible_values)
        print(f"  Suggested: mutate pair {suggestion[0]}, type={suggestion[1]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
