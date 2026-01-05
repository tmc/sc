"""
State Discovery from Execution Traces

Automatically discover statechart states from program execution patterns.

Approaches:
1. Line Clustering - Group lines that always execute together
2. Variable Patterns - States defined by variable value combinations
3. Control Flow Phases - Identify distinct execution phases
4. SAE Features - Use sparse autoencoder to find monosemantic states

The discovered states form the foundation for statechart synthesis.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, FrozenSet
from collections import defaultdict
from enum import Enum, auto
import json

from .coverage_collector import CoverageTriple
from .dataset import DatasetExample, Dataset


class StateType(Enum):
    """Types of discovered states."""
    ENTRY = auto()          # Program entry point
    EXIT = auto()           # Program exit point
    BASIC = auto()          # Basic block state
    BRANCH = auto()         # Branch decision point
    LOOP_HEADER = auto()    # Loop entry
    LOOP_BODY = auto()      # Inside loop
    ERROR = auto()          # Error/exception state
    COMPOSITE = auto()      # Contains substates


@dataclass
class DiscoveredState:
    """A state discovered from execution traces."""
    id: str
    state_type: StateType
    lines: FrozenSet[int]           # Source lines in this state
    entry_count: int = 0            # How often entered
    exit_count: int = 0             # How often exited
    variables: Dict[str, Set[Any]] = field(default_factory=dict)  # Variable values seen
    predecessors: Set[str] = field(default_factory=set)  # States that lead here
    successors: Set[str] = field(default_factory=set)    # States reachable from here
    is_initial: bool = False
    is_final: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.state_type.name,
            'lines': sorted(self.lines),
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'predecessors': sorted(self.predecessors),
            'successors': sorted(self.successors),
            'is_initial': self.is_initial,
            'is_final': self.is_final,
        }


@dataclass
class ExecutionSegment:
    """A segment of execution trace."""
    lines: Tuple[int, ...]
    count: int = 1
    inputs: List[str] = field(default_factory=list)


class LineClusterDiscovery:
    """
    Discover states by clustering lines that execute together.

    Key insight: Lines that ALWAYS execute together form a basic block.
    Lines that SOMETIMES execute together are in the same branch.
    """

    def __init__(self):
        self.line_cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        self.line_counts: Dict[int, int] = defaultdict(int)
        self.total_traces = 0

    def add_trace(self, covered_lines: Set[int]):
        """Add an execution trace."""
        self.total_traces += 1
        lines = sorted(covered_lines)

        for line in lines:
            self.line_counts[line] += 1

        # Track co-occurrence
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                key = (min(line1, line2), max(line1, line2))
                self.line_cooccurrence[key] += 1

    def get_cooccurrence_ratio(self, line1: int, line2: int) -> float:
        """Get how often two lines appear together."""
        key = (min(line1, line2), max(line1, line2))
        together = self.line_cooccurrence[key]
        either = max(self.line_counts[line1], self.line_counts[line2])
        return together / either if either > 0 else 0.0

    def cluster_lines(self, threshold: float = 0.95) -> List[FrozenSet[int]]:
        """
        Cluster lines that always (>threshold) appear together.

        Returns list of line sets, each forming a potential state.
        """
        all_lines = set(self.line_counts.keys())
        if not all_lines:
            return []

        # Union-find for clustering
        parent = {line: line for line in all_lines}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Cluster lines with high co-occurrence
        for (line1, line2), count in self.line_cooccurrence.items():
            ratio = self.get_cooccurrence_ratio(line1, line2)
            if ratio >= threshold:
                union(line1, line2)

        # Build clusters
        clusters = defaultdict(set)
        for line in all_lines:
            clusters[find(line)].add(line)

        return [frozenset(c) for c in clusters.values()]


class VariablePatternDiscovery:
    """
    Discover states based on variable value patterns.

    States are defined by combinations of variable values.
    Similar to abstract interpretation domains.
    """

    def __init__(self):
        self.patterns: Dict[FrozenSet[Tuple[str, Any]], int] = defaultdict(int)
        self.variable_values: Dict[str, Set[Any]] = defaultdict(set)

    def add_state(self, variables: Dict[str, Any]):
        """Add a variable state observation."""
        # Create hashable pattern
        pattern = frozenset((k, self._hashable(v)) for k, v in variables.items())
        self.patterns[pattern] += 1

        for k, v in variables.items():
            self.variable_values[k].add(self._hashable(v))

    def _hashable(self, v: Any) -> Any:
        """Convert value to hashable form."""
        if isinstance(v, (list, set)):
            return tuple(sorted(self._hashable(x) for x in v))
        elif isinstance(v, dict):
            return tuple(sorted((k, self._hashable(v)) for k, v in v.items()))
        return v

    def get_significant_patterns(self, min_count: int = 2) -> List[Dict[str, Any]]:
        """Get patterns that appear multiple times."""
        result = []
        for pattern, count in self.patterns.items():
            if count >= min_count:
                result.append({
                    'variables': dict(pattern),
                    'count': count,
                })
        return sorted(result, key=lambda x: -x['count'])


class ControlFlowPhaseDiscovery:
    """
    Discover execution phases from trace sequences.

    Phases are contiguous segments of execution that represent
    distinct program states (e.g., initialization, processing, cleanup).
    """

    def __init__(self):
        self.segments: Dict[Tuple[int, ...], ExecutionSegment] = {}
        self.transitions: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = defaultdict(int)

    def add_trace(self, execution_trace: List[int], input_repr: str = ""):
        """Add an execution trace and identify segments."""
        if not execution_trace:
            return

        # Find segments (runs of consecutive or repeated lines)
        segments = self._find_segments(execution_trace)

        for i, seg in enumerate(segments):
            key = tuple(sorted(set(seg)))
            if key not in self.segments:
                self.segments[key] = ExecutionSegment(lines=key)
            self.segments[key].count += 1
            self.segments[key].inputs.append(input_repr)

            # Track transitions between segments
            if i > 0:
                prev_key = tuple(sorted(set(segments[i-1])))
                self.transitions[(prev_key, key)] += 1

    def _find_segments(self, trace: List[int], window: int = 3) -> List[List[int]]:
        """Split trace into segments based on line patterns."""
        if len(trace) <= window:
            return [trace]

        segments = []
        current = []

        for i, line in enumerate(trace):
            current.append(line)

            # Segment boundary: significant line number jump or repeated pattern
            if i > 0:
                jump = abs(line - trace[i-1])
                if jump > 5:  # Significant jump
                    if len(current) > 1:
                        segments.append(current[:-1])
                        current = [line]

        if current:
            segments.append(current)

        return segments

    def get_phases(self, min_count: int = 2) -> List[ExecutionSegment]:
        """Get significant execution phases."""
        return [seg for seg in self.segments.values() if seg.count >= min_count]


class StateDiscoverer:
    """
    Main state discovery engine.

    Combines multiple discovery strategies to find program states
    from execution traces.
    """

    def __init__(self):
        self.line_clusterer = LineClusterDiscovery()
        self.variable_patterns = VariablePatternDiscovery()
        self.phase_discovery = ControlFlowPhaseDiscovery()
        self.discovered_states: Dict[str, DiscoveredState] = {}
        self.state_transitions: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_example(self, example: DatasetExample):
        """Add a training example."""
        covered = set(example.covered_lines)
        self.line_clusterer.add_trace(covered)
        self.phase_discovery.add_trace(example.execution_trace, example.input_repr)

    def add_triple(self, triple: CoverageTriple):
        """Add a coverage triple."""
        self.line_clusterer.add_trace(triple.covered_lines)
        self.phase_discovery.add_trace(triple.execution_trace, triple.input_repr)

    def discover_states(
        self,
        cluster_threshold: float = 0.9,
        min_phase_count: int = 2,
    ) -> Dict[str, DiscoveredState]:
        """
        Discover states from accumulated traces.

        Returns dict of state_id -> DiscoveredState
        """
        self.discovered_states = {}

        # 1. Discover states from line clusters
        clusters = self.line_clusterer.cluster_lines(threshold=cluster_threshold)

        for i, cluster in enumerate(clusters):
            state_id = f"cluster_{i}"
            lines = cluster

            # Determine state type based on lines
            state_type = self._infer_state_type(lines)

            state = DiscoveredState(
                id=state_id,
                state_type=state_type,
                lines=lines,
                entry_count=sum(self.line_clusterer.line_counts.get(l, 0) for l in lines),
            )

            # Check if initial (contains line 1 or 2)
            if 1 in lines or 2 in lines:
                state.is_initial = True

            self.discovered_states[state_id] = state

        # 2. Discover phases and merge with clusters
        phases = self.phase_discovery.get_phases(min_count=min_phase_count)

        for i, phase in enumerate(phases):
            phase_lines = frozenset(phase.lines)

            # Check if this phase matches an existing cluster
            matched = False
            for state in self.discovered_states.values():
                overlap = len(state.lines & phase_lines)
                if overlap > 0.8 * len(phase_lines):
                    # Merge phase info into cluster
                    state.entry_count = max(state.entry_count, phase.count)
                    matched = True
                    break

            if not matched and len(phase_lines) > 1:
                # Create new state from phase
                state_id = f"phase_{i}"
                state = DiscoveredState(
                    id=state_id,
                    state_type=StateType.BASIC,
                    lines=phase_lines,
                    entry_count=phase.count,
                )
                self.discovered_states[state_id] = state

        # 3. Discover transitions between states
        self._discover_transitions()

        # 4. Identify final states (no successors)
        for state in self.discovered_states.values():
            if not state.successors:
                state.is_final = True

        return self.discovered_states

    def _infer_state_type(self, lines: FrozenSet[int]) -> StateType:
        """Infer state type from line characteristics."""
        if len(lines) == 1:
            line = next(iter(lines))
            if line == 1:
                return StateType.ENTRY
            return StateType.BASIC

        # Check line patterns
        lines_list = sorted(lines)
        if lines_list[0] <= 2:
            return StateType.ENTRY

        # Consecutive lines suggest basic block
        gaps = [lines_list[i+1] - lines_list[i] for i in range(len(lines_list)-1)]
        if all(g <= 2 for g in gaps):
            return StateType.BASIC

        return StateType.COMPOSITE

    def _discover_transitions(self):
        """Discover transitions between states from phase transitions."""
        for (from_lines, to_lines), count in self.phase_discovery.transitions.items():
            # Find matching states
            from_state = self._find_state_for_lines(set(from_lines))
            to_state = self._find_state_for_lines(set(to_lines))

            if from_state and to_state and from_state != to_state:
                self.state_transitions[(from_state.id, to_state.id)] += count
                from_state.successors.add(to_state.id)
                to_state.predecessors.add(from_state.id)

    def _find_state_for_lines(self, lines: Set[int]) -> Optional[DiscoveredState]:
        """Find the state that best matches given lines."""
        best_state = None
        best_overlap = 0

        for state in self.discovered_states.values():
            overlap = len(state.lines & lines)
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = state

        return best_state

    def to_statechart_proto(self) -> Dict:
        """Convert discovered states to statechart proto format."""
        children = []
        for state in self.discovered_states.values():
            child = {
                'label': state.id,
                'type': 1,  # STATE_TYPE_BASIC
                'is_initial': state.is_initial,
            }
            if state.lines:
                child['metadata'] = {'lines': sorted(state.lines)}
            children.append(child)

        root_state = {
            'label': '__root__',
            'type': 2,  # STATE_TYPE_NORMAL
            'children': children,
        }

        transitions = []
        for (from_id, to_id), count in self.state_transitions.items():
            transitions.append({
                'from': [from_id],
                'to': [to_id],
                'metadata': {'count': count},
            })

        return {
            'root_state': root_state,
            'transitions': transitions,
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of discovered states."""
        lines = ['stateDiagram-v2']

        for state in self.discovered_states.values():
            label = state.id
            line_info = f"L{min(state.lines)}" if state.lines else "?"

            if state.is_initial:
                lines.append(f"    [*] --> {label}")
            if state.is_final:
                lines.append(f"    {label} --> [*]")

            lines.append(f"    {label}: {label} ({line_info}, n={state.entry_count})")

        for (from_id, to_id), count in self.state_transitions.items():
            lines.append(f"    {from_id} --> {to_id}: {count}x")

        return '\n'.join(lines)


def demo():
    """Demonstrate state discovery."""
    print("=" * 60)
    print("STATE DISCOVERY DEMO")
    print("=" * 60)

    # Create discoverer
    discoverer = StateDiscoverer()

    # Generate some execution traces
    from .dataset import DatasetGenerator
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate_dataset()

    print(f"\nAnalyzing {len(dataset)} execution traces...")

    # Add examples
    for example in dataset.examples[:50]:  # Use subset for demo
        discoverer.add_example(example)

    # Discover states
    states = discoverer.discover_states(cluster_threshold=0.85)

    print(f"\nDiscovered {len(states)} states:")
    for state in sorted(states.values(), key=lambda s: min(s.lines) if s.lines else 999):
        print(f"  {state.id}: lines={sorted(state.lines)[:5]}{'...' if len(state.lines) > 5 else ''}, "
              f"type={state.state_type.name}, entries={state.entry_count}")

    print(f"\nTransitions: {len(discoverer.state_transitions)}")
    for (from_id, to_id), count in sorted(discoverer.state_transitions.items(), key=lambda x: -x[1])[:10]:
        print(f"  {from_id} -> {to_id}: {count}x")

    print("\nMermaid diagram (first 20 lines):")
    mermaid = discoverer.to_mermaid()
    for line in mermaid.split('\n')[:20]:
        print(f"  {line}")

    return discoverer


if __name__ == "__main__":
    demo()
