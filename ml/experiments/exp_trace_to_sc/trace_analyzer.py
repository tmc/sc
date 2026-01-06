"""
Trace Analyzer: Extract patterns from event traces.

Provides algorithmic analysis of traces to extract:
- Unique events (potential states)
- Transition patterns (event -> next event)
- Initial state (first event in traces)
- Cycle detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter


@dataclass
class TraceAnalysis:
    """Analysis results from traces."""
    unique_events: Set[str] = field(default_factory=set)
    transitions: Set[Tuple[str, str]] = field(default_factory=set)
    transition_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    initial_events: Counter = field(default_factory=Counter)
    terminal_events: Counter = field(default_factory=Counter)
    self_loops: Set[str] = field(default_factory=set)
    cycles: List[List[str]] = field(default_factory=list)

    @property
    def num_states(self) -> int:
        return len(self.unique_events)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    @property
    def most_likely_initial(self) -> Optional[str]:
        if not self.initial_events:
            return None
        return self.initial_events.most_common(1)[0][0]


class TraceAnalyzer:
    """
    Analyzes traces to extract patterns for statechart induction.
    """

    def analyze(self, traces: List[List[str]]) -> TraceAnalysis:
        """
        Analyze traces and extract patterns.

        Args:
            traces: List of event sequences

        Returns:
            TraceAnalysis with extracted patterns
        """
        result = TraceAnalysis()

        for trace in traces:
            if not trace:
                continue

            # Record all events
            result.unique_events.update(trace)

            # Record initial event
            result.initial_events[trace[0]] += 1

            # Record terminal event
            result.terminal_events[trace[-1]] += 1

            # Record transitions
            for i in range(len(trace) - 1):
                src = trace[i]
                tgt = trace[i + 1]
                trans = (src, tgt)

                result.transitions.add(trans)
                result.transition_counts[trans] = \
                    result.transition_counts.get(trans, 0) + 1

                # Self-loops
                if src == tgt:
                    result.self_loops.add(src)

        # Detect cycles
        result.cycles = self._detect_cycles(result.transitions)

        return result

    def _detect_cycles(
        self,
        transitions: Set[Tuple[str, str]]
    ) -> List[List[str]]:
        """Detect cycles in transition graph."""
        cycles = []

        # Build adjacency list
        adj: Dict[str, Set[str]] = {}
        for src, tgt in transitions:
            if src not in adj:
                adj[src] = set()
            adj[src].add(tgt)

        # Find simple cycles using DFS
        visited = set()
        rec_stack = []

        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found cycle
                idx = rec_stack.index(node)
                cycle = rec_stack[idx:] + [node]
                if len(cycle) > 2:  # Ignore self-loops
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in adj.get(node, []):
                dfs(neighbor, path + [neighbor])

            rec_stack.pop()

        for start in adj:
            visited.clear()
            rec_stack.clear()
            dfs(start, [start])

        return cycles

    def format_analysis(self, analysis: TraceAnalysis) -> str:
        """
        Format analysis as a human-readable string for CoT prompt.
        """
        lines = []

        # Step 1: Unique states
        sorted_events = sorted(analysis.unique_events)
        lines.append(f"1. Unique states seen: {', '.join(sorted_events)} ({analysis.num_states} states)")

        # Step 2: Transitions
        lines.append("2. Transitions observed:")
        sorted_trans = sorted(analysis.transitions)
        for src, tgt in sorted_trans:
            count = analysis.transition_counts.get((src, tgt), 0)
            lines.append(f"   - {src} -> {tgt} (seen {count}x)")

        # Step 3: Initial state
        if analysis.most_likely_initial:
            initial = analysis.most_likely_initial
            count = analysis.initial_events[initial]
            lines.append(f"3. Initial state: {initial} (starts {count}/{sum(analysis.initial_events.values())} traces)")

        # Note self-loops
        if analysis.self_loops:
            lines.append(f"   Self-loops: {', '.join(sorted(analysis.self_loops))}")

        # Note cycles
        if analysis.cycles:
            cycles_str = [" -> ".join(c) for c in analysis.cycles[:3]]
            lines.append(f"   Cycles: {'; '.join(cycles_str)}")

        return "\n".join(lines)


def analyze_traces(traces: List[List[str]]) -> TraceAnalysis:
    """Convenience function to analyze traces."""
    analyzer = TraceAnalyzer()
    return analyzer.analyze(traces)


def format_trace_analysis(traces: List[List[str]]) -> str:
    """Convenience function to get formatted analysis."""
    analyzer = TraceAnalyzer()
    analysis = analyzer.analyze(traces)
    return analyzer.format_analysis(analysis)


if __name__ == "__main__":
    print("Trace Analyzer Test")
    print("=" * 60)

    # Test 1: Toggle
    traces1 = [
        ["TURN_ON", "TURN_OFF", "TURN_ON"],
        ["TURN_ON", "TURN_OFF", "TURN_ON", "TURN_OFF"],
    ]
    print("\n1. Toggle:")
    print(format_trace_analysis(traces1))

    # Test 2: Traffic light
    traces2 = [
        ["RED", "GREEN", "YELLOW", "RED"],
        ["RED", "GREEN", "YELLOW", "RED", "GREEN"],
    ]
    print("\n2. Traffic light:")
    print(format_trace_analysis(traces2))

    # Test 3: Player
    traces3 = [
        ["PLAY", "PAUSE", "PLAY", "STOP"],
        ["PLAY", "STOP"],
        ["PLAY", "PAUSE", "STOP"],
    ]
    print("\n3. Player:")
    print(format_trace_analysis(traces3))
