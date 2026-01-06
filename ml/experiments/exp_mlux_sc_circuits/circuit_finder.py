"""
Circuit Finder for Statechart Validity

Identifies circuits (connected sets of components) responsible for
specific aspects of statechart validity.

APPROACH:
1. Use ablation results to identify critical components
2. Analyze activation patterns to find connected components
3. Build circuit graph showing information flow
4. Map circuits to validity aspects (names, transitions, hierarchy)

CIRCUIT TYPES:
- State Name Memory Circuit: Maintains state name consistency
- Transition Validity Circuit: Ensures valid state references
- Hierarchy Circuit: Maintains parent-child relationships
- Structural Circuit: Produces valid JSON structure
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

from .validity_measurer import ValidityMeasurer, ValidityMetrics
from .ablation_runner import (
    AblationRunner,
    AblationResult,
    AblationStudy,
    ComponentSpec,
    AblationType,
)


class CircuitType(Enum):
    """Types of circuits for SC validity."""
    STATE_NAME_MEMORY = "state_name_memory"
    TRANSITION_VALIDITY = "transition_validity"
    HIERARCHY = "hierarchy"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass
class CircuitNode:
    """A node in the circuit graph."""
    component: ComponentSpec
    importance: float  # How much ablation affects validity
    circuit_types: Set[CircuitType] = field(default_factory=set)

    # Connections
    upstream: List["CircuitNode"] = field(default_factory=list)
    downstream: List["CircuitNode"] = field(default_factory=list)

    def __hash__(self):
        return hash(str(self.component))

    def __eq__(self, other):
        return str(self.component) == str(other.component)


@dataclass
class Circuit:
    """A circuit (connected component set) for a validity aspect."""
    circuit_type: CircuitType
    nodes: List[CircuitNode] = field(default_factory=list)
    total_importance: float = 0.0

    @property
    def layers(self) -> Set[int]:
        return {n.component.layer for n in self.nodes}

    @property
    def layer_span(self) -> Tuple[int, int]:
        layers = self.layers
        return (min(layers), max(layers)) if layers else (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.circuit_type.value,
            "n_nodes": len(self.nodes),
            "total_importance": self.total_importance,
            "layers": sorted(self.layers),
            "layer_span": self.layer_span,
            "nodes": [
                {
                    "component": str(n.component),
                    "importance": n.importance,
                }
                for n in sorted(self.nodes, key=lambda x: -x.importance)
            ],
        }


@dataclass
class CircuitGraph:
    """Complete circuit graph for a model."""
    circuits: Dict[CircuitType, Circuit] = field(default_factory=dict)
    all_nodes: Dict[str, CircuitNode] = field(default_factory=dict)
    n_layers: int = 0
    n_heads: int = 0

    def get_critical_layers(
        self,
        circuit_type: Optional[CircuitType] = None,
        threshold: float = 0.1,
    ) -> List[int]:
        """Get layers critical for a circuit type."""
        if circuit_type and circuit_type in self.circuits:
            nodes = self.circuits[circuit_type].nodes
        else:
            nodes = list(self.all_nodes.values())

        critical = [n for n in nodes if n.importance >= threshold]
        return sorted(set(n.component.layer for n in critical))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_total_nodes": len(self.all_nodes),
            "circuits": {
                ct.value: c.to_dict()
                for ct, c in self.circuits.items()
            },
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of circuit graph."""
        lines = ["graph TD"]

        # Add nodes grouped by layer
        for layer in range(self.n_layers):
            layer_nodes = [
                n for n in self.all_nodes.values()
                if n.component.layer == layer and n.importance >= 0.05
            ]
            if layer_nodes:
                for node in layer_nodes:
                    node_id = str(node.component).replace(".", "_")
                    label = str(node.component)
                    importance = node.importance

                    # Color by circuit type
                    if CircuitType.STATE_NAME_MEMORY in node.circuit_types:
                        style = "fill:#f9f,stroke:#333"
                    elif CircuitType.TRANSITION_VALIDITY in node.circuit_types:
                        style = "fill:#9ff,stroke:#333"
                    elif CircuitType.HIERARCHY in node.circuit_types:
                        style = "fill:#ff9,stroke:#333"
                    else:
                        style = "fill:#999,stroke:#333"

                    lines.append(f"    {node_id}[{label}<br/>imp:{importance:.0%}]")
                    lines.append(f"    style {node_id} {style}")

        # Add edges (layer connections)
        for layer in range(self.n_layers - 1):
            current_nodes = [
                n for n in self.all_nodes.values()
                if n.component.layer == layer and n.importance >= 0.05
            ]
            next_nodes = [
                n for n in self.all_nodes.values()
                if n.component.layer == layer + 1 and n.importance >= 0.05
            ]

            for curr in current_nodes:
                for next_n in next_nodes:
                    # Connect if same circuit type
                    if curr.circuit_types & next_n.circuit_types:
                        curr_id = str(curr.component).replace(".", "_")
                        next_id = str(next_n.component).replace(".", "_")
                        lines.append(f"    {curr_id} --> {next_id}")

        return "\n".join(lines)


class CircuitFinder:
    """
    Finds circuits responsible for statechart validity.

    Uses ablation results to identify components and their roles,
    then builds a circuit graph showing information flow.
    """

    # Thresholds for circuit membership
    STATE_NAME_THRESHOLD = 0.05
    TRANSITION_THRESHOLD = 0.05
    HIERARCHY_THRESHOLD = 0.05
    IMPORTANCE_THRESHOLD = 0.02

    def __init__(
        self,
        runner: Optional[AblationRunner] = None,
        verbose: bool = True,
    ):
        self.runner = runner or AblationRunner()
        self.verbose = verbose

    def find_circuits(
        self,
        study: AblationStudy,
    ) -> CircuitGraph:
        """
        Find circuits from ablation study results.

        Args:
            study: Completed ablation study

        Returns:
            CircuitGraph with identified circuits
        """
        graph = CircuitGraph(
            n_layers=study.n_layers,
            n_heads=study.n_heads,
        )

        # Create nodes from ablation results
        for result in study.results:
            if result.validity_drop < self.IMPORTANCE_THRESHOLD:
                continue

            node = CircuitNode(
                component=result.component,
                importance=result.validity_drop,
            )

            # Determine circuit membership based on what drops
            if result.state_name_drop >= self.STATE_NAME_THRESHOLD:
                node.circuit_types.add(CircuitType.STATE_NAME_MEMORY)

            if result.transition_drop >= self.TRANSITION_THRESHOLD:
                node.circuit_types.add(CircuitType.TRANSITION_VALIDITY)

            if result.hierarchy_drop >= self.HIERARCHY_THRESHOLD:
                node.circuit_types.add(CircuitType.HIERARCHY)

            # Structural if high overall but low specific drops
            if (result.validity_drop >= 0.1 and
                result.state_name_drop < 0.05 and
                result.transition_drop < 0.05):
                node.circuit_types.add(CircuitType.STRUCTURAL)

            graph.all_nodes[str(node.component)] = node

        # Build circuits for each type
        for circuit_type in CircuitType:
            circuit = self._build_circuit(graph, circuit_type)
            if circuit.nodes:
                graph.circuits[circuit_type] = circuit

        # Connect nodes within circuits
        self._connect_nodes(graph)

        return graph

    def _build_circuit(
        self,
        graph: CircuitGraph,
        circuit_type: CircuitType,
    ) -> Circuit:
        """Build a circuit for a specific validity aspect."""
        circuit = Circuit(circuit_type=circuit_type)

        for node in graph.all_nodes.values():
            if circuit_type in node.circuit_types:
                circuit.nodes.append(node)
                circuit.total_importance += node.importance

        # Sort by layer then importance
        circuit.nodes.sort(key=lambda n: (n.component.layer, -n.importance))

        return circuit

    def _connect_nodes(self, graph: CircuitGraph):
        """Connect nodes based on layer adjacency and circuit membership."""
        nodes_by_layer: Dict[int, List[CircuitNode]] = defaultdict(list)

        for node in graph.all_nodes.values():
            nodes_by_layer[node.component.layer].append(node)

        # Connect adjacent layers
        for layer in range(graph.n_layers - 1):
            current = nodes_by_layer.get(layer, [])
            next_layer = nodes_by_layer.get(layer + 1, [])

            for curr_node in current:
                for next_node in next_layer:
                    # Connect if they share circuit types
                    if curr_node.circuit_types & next_node.circuit_types:
                        curr_node.downstream.append(next_node)
                        next_node.upstream.append(curr_node)

    def analyze_layer_roles(
        self,
        graph: CircuitGraph,
    ) -> Dict[str, Any]:
        """
        Analyze what role each layer plays in validity.

        Returns analysis of layer functions.
        """
        analysis = {
            "early_layers": {"range": (0, graph.n_layers // 3), "roles": []},
            "middle_layers": {"range": (graph.n_layers // 3, 2 * graph.n_layers // 3), "roles": []},
            "late_layers": {"range": (2 * graph.n_layers // 3, graph.n_layers), "roles": []},
        }

        # Analyze each layer range
        for key, info in analysis.items():
            start, end = info["range"]
            layer_nodes = [
                n for n in graph.all_nodes.values()
                if start <= n.component.layer < end
            ]

            if not layer_nodes:
                continue

            # Count circuit type occurrences
            type_counts = defaultdict(float)
            for node in layer_nodes:
                for ct in node.circuit_types:
                    type_counts[ct] += node.importance

            # Identify dominant roles
            sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
            for ct, importance in sorted_types:
                if importance >= 0.1:
                    info["roles"].append({
                        "type": ct.value,
                        "importance": importance,
                    })

        return analysis

    def find_critical_paths(
        self,
        graph: CircuitGraph,
        circuit_type: CircuitType,
    ) -> List[List[CircuitNode]]:
        """
        Find critical paths through a circuit.

        Returns list of paths from early to late layers.
        """
        if circuit_type not in graph.circuits:
            return []

        circuit = graph.circuits[circuit_type]
        if not circuit.nodes:
            return []

        # Find start nodes (earliest layers)
        min_layer = min(n.component.layer for n in circuit.nodes)
        start_nodes = [n for n in circuit.nodes if n.component.layer == min_layer]

        # Find end nodes (latest layers)
        max_layer = max(n.component.layer for n in circuit.nodes)
        end_nodes = [n for n in circuit.nodes if n.component.layer == max_layer]

        # DFS to find paths
        paths = []

        def dfs(node: CircuitNode, path: List[CircuitNode]):
            path = path + [node]

            if node in end_nodes:
                paths.append(path)
                return

            for downstream in node.downstream:
                if downstream in circuit.nodes and downstream not in path:
                    dfs(downstream, path)

        for start in start_nodes:
            dfs(start, [])

        # Sort by total importance
        paths.sort(key=lambda p: -sum(n.importance for n in p))

        return paths[:5]  # Return top 5 paths


# =============================================================================
# High-level analysis
# =============================================================================

def summarize_circuits(graph: CircuitGraph) -> str:
    """Generate human-readable circuit summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("CIRCUIT ANALYSIS SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nModel: {graph.n_layers} layers, {graph.n_heads} heads")
    lines.append(f"Total critical components: {len(graph.all_nodes)}")

    for ct, circuit in graph.circuits.items():
        lines.append(f"\n{ct.value.upper()} CIRCUIT:")
        lines.append(f"  Nodes: {len(circuit.nodes)}")
        lines.append(f"  Layer span: {circuit.layer_span[0]}-{circuit.layer_span[1]}")
        lines.append(f"  Total importance: {circuit.total_importance:.2f}")

        # Top components
        top_nodes = sorted(circuit.nodes, key=lambda n: -n.importance)[:3]
        lines.append("  Top components:")
        for node in top_nodes:
            lines.append(f"    - {node.component}: {node.importance:.1%}")

    return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate circuit finding."""
    print("=" * 60)
    print("Circuit Finder for SC Validity")
    print("=" * 60)

    # Create mock ablation study
    from .ablation_runner import default_parser

    runner = AblationRunner(model=None, verbose=True)

    prompts = [
        "Generate a traffic light statechart:",
        "Create a login state machine:",
    ]

    # Run study on sample components
    components = runner.enumerate_components(
        include_attention=True,
        include_mlp=True,
        layers=list(range(0, 24, 2)),  # Every other layer
    )

    study = runner.run_study(
        prompts=prompts,
        parse_fn=default_parser,
        components=components,
    )

    # Find circuits
    finder = CircuitFinder(runner=runner)
    graph = finder.find_circuits(study)

    # Print summary
    print(summarize_circuits(graph))

    # Analyze layer roles
    print("\n" + "-" * 40)
    print("LAYER ROLE ANALYSIS:")
    analysis = finder.analyze_layer_roles(graph)
    for key, info in analysis.items():
        print(f"\n{key} ({info['range'][0]}-{info['range'][1]}):")
        for role in info["roles"]:
            print(f"  - {role['type']}: importance={role['importance']:.2f}")

    # Critical paths
    print("\n" + "-" * 40)
    print("CRITICAL PATHS:")
    for ct in CircuitType:
        paths = finder.find_critical_paths(graph, ct)
        if paths:
            print(f"\n{ct.value}:")
            for i, path in enumerate(paths[:2]):
                path_str = " -> ".join(str(n.component) for n in path)
                importance = sum(n.importance for n in path)
                print(f"  Path {i+1}: {path_str} (imp={importance:.2f})")

    return graph, finder


if __name__ == "__main__":
    demo()
