#!/usr/bin/env python3
"""
ASCII Graph Visualization for Statecharts.

Creates text-based graph representations to help LLMs visualize paths.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class StatechartPath:
    """A statechart with path to analyze."""
    states: List[str]
    transitions: List[Dict[str, Any]]
    initial: str
    terminal: List[str]
    context: Dict[str, Any] = None
    expected_length: int = -1


def create_ascii_graph(sc: StatechartPath) -> str:
    """
    Create ASCII representation of statechart.

    Example output:
    [A*] --e1--> [B] --e2--> [C] --e3--> [[D]]

    * = initial, [[ ]] = terminal
    """
    # Build adjacency list
    adj = {s: [] for s in sc.states}
    for t in sc.transitions:
        adj[t['from']].append((t['to'], t.get('event', ''), t.get('guard', '')))

    # Try to create linear visualization
    lines = []

    # Check if it's a simple linear chain
    if _is_linear_chain(sc):
        return _create_linear_ascii(sc)

    # Otherwise create multi-line format
    return _create_multiline_ascii(sc)


def _is_linear_chain(sc: StatechartPath) -> bool:
    """Check if the graph is a simple linear chain."""
    # Each state should have at most one outgoing transition (except self-loops)
    adj = {s: [] for s in sc.states}
    for t in sc.transitions:
        if t['from'] != t['to']:  # Ignore self-loops for this check
            adj[t['from']].append(t['to'])

    for targets in adj.values():
        if len(targets) > 1:
            return False
    return True


def _create_linear_ascii(sc: StatechartPath) -> str:
    """Create linear ASCII for simple chain."""
    # Build path from initial
    visited = set()
    path = []
    current = sc.initial

    while current and current not in visited:
        visited.add(current)
        path.append(current)

        # Find next state
        next_state = None
        for t in sc.transitions:
            if t['from'] == current and t['to'] != current:
                next_state = t['to']
                break

        if next_state and next_state not in visited:
            current = next_state
        else:
            break

    # Format path
    parts = []
    for i, state in enumerate(path):
        # Format state
        if state == sc.initial and state in sc.terminal:
            state_str = f"[[{state}*]]"
        elif state == sc.initial:
            state_str = f"[{state}*]"
        elif state in sc.terminal:
            state_str = f"[[{state}]]"
        else:
            state_str = f"[{state}]"

        parts.append(state_str)

        # Add arrow to next
        if i < len(path) - 1:
            # Find transition
            for t in sc.transitions:
                if t['from'] == state and t['to'] == path[i + 1]:
                    arrow = f" --{t.get('event', 'e')}"
                    if t.get('guard'):
                        arrow += f"[{t['guard']}]"
                    arrow += "--> "
                    parts.append(arrow)
                    break

    return "".join(parts)


def _create_multiline_ascii(sc: StatechartPath) -> str:
    """Create multi-line ASCII for complex graphs."""
    lines = []
    lines.append("Graph structure:")

    # Group by source state
    by_source = {}
    for t in sc.transitions:
        src = t['from']
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(t)

    # Format each source
    for state in sc.states:
        # Format state
        if state == sc.initial and state in sc.terminal:
            state_str = f"[[{state}*]]"
        elif state == sc.initial:
            state_str = f"[{state}*]"
        elif state in sc.terminal:
            state_str = f"[[{state}]]"
        else:
            state_str = f"[{state}]"

        if state in by_source:
            for t in by_source[state]:
                arrow = f"  {state_str} --{t.get('event', 'e')}"
                if t.get('guard'):
                    arrow += f"[{t['guard']}]"

                # Format target
                tgt = t['to']
                if tgt in sc.terminal:
                    tgt_str = f"[[{tgt}]]"
                else:
                    tgt_str = f"[{tgt}]"

                arrow += f"--> {tgt_str}"
                lines.append(arrow)
        elif state in sc.terminal:
            lines.append(f"  {state_str} (terminal)")

    return "\n".join(lines)


def create_transition_list(sc: StatechartPath) -> str:
    """Create simple numbered transition list."""
    lines = ["Transitions:"]

    for i, t in enumerate(sc.transitions, 1):
        line = f"  {i}. {t['from']} → {t['to']}"
        if t.get('event'):
            line += f" (on {t['event']})"
        if t.get('guard'):
            line += f" [when {t['guard']}]"
        lines.append(line)

    return "\n".join(lines)


def create_distance_table(sc: StatechartPath) -> str:
    """Create distance table showing steps from initial to each state."""

    # BFS to compute distances
    from collections import deque

    distances = {sc.initial: 0}
    queue = deque([sc.initial])

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        for t in sc.transitions:
            if t['from'] == current:
                next_state = t['to']
                if next_state not in distances:
                    distances[next_state] = current_dist + 1
                    queue.append(next_state)

    lines = ["Distance from initial:"]
    for state in sc.states:
        if state in distances:
            marker = "*" if state == sc.initial else ("T" if state in sc.terminal else "")
            lines.append(f"  {state}{marker}: {distances[state]} steps")
        else:
            lines.append(f"  {state}: unreachable")

    return "\n".join(lines)


def create_path_trace(sc: StatechartPath) -> str:
    """Create a trace showing the path from initial to terminal."""

    # Simple BFS to find path
    from collections import deque

    queue = deque([(sc.initial, [sc.initial], 0)])
    visited = {sc.initial}

    while queue:
        current, path, steps = queue.popleft()

        if current in sc.terminal:
            # Found terminal - format trace
            lines = ["Path trace:"]
            lines.append(f"  Start: {path[0]}")
            for i in range(1, len(path)):
                prev = path[i - 1]
                curr = path[i]
                # Find transition
                for t in sc.transitions:
                    if t['from'] == prev and t['to'] == curr:
                        event = t.get('event', 'e')
                        lines.append(f"  Step {i}: {prev} --{event}--> {curr}")
                        break
            lines.append(f"  Total: {steps} steps")
            return "\n".join(lines)

        # Explore neighbors
        for t in sc.transitions:
            if t['from'] == current and t['to'] not in visited:
                visited.add(t['to'])
                queue.append((t['to'], path + [t['to']], steps + 1))

    return "Path trace: No path to terminal found (infinite)"


class PathVisualizer:
    """Helper class for creating various visualizations."""

    def __init__(self, sc: StatechartPath):
        self.sc = sc

    def ascii_graph(self) -> str:
        return create_ascii_graph(self.sc)

    def transition_list(self) -> str:
        return create_transition_list(self.sc)

    def distance_table(self) -> str:
        return create_distance_table(self.sc)

    def path_trace(self) -> str:
        return create_path_trace(self.sc)

    def all_formats(self) -> str:
        """Return all visualization formats."""
        parts = [
            self.ascii_graph(),
            "",
            self.transition_list(),
            "",
            self.distance_table(),
            "",
            self.path_trace(),
        ]
        return "\n".join(parts)


if __name__ == "__main__":
    # Test visualizations
    print("Path Visualizer Test")
    print("=" * 60)

    # Linear chain
    sc1 = StatechartPath(
        states=["A", "B", "C", "D"],
        transitions=[
            {"from": "A", "to": "B", "event": "e1"},
            {"from": "B", "to": "C", "event": "e2"},
            {"from": "C", "to": "D", "event": "e3"},
        ],
        initial="A",
        terminal=["D"],
        expected_length=3,
    )

    print("\nLinear Chain:")
    viz = PathVisualizer(sc1)
    print(viz.all_formats())

    # Branching
    sc2 = StatechartPath(
        states=["Check", "Pass", "Fail", "Done"],
        transitions=[
            {"from": "Check", "to": "Pass", "event": "EVAL", "guard": "x>=50"},
            {"from": "Check", "to": "Fail", "event": "EVAL", "guard": "x<50"},
            {"from": "Pass", "to": "Done", "event": "NEXT"},
            {"from": "Fail", "to": "Done", "event": "NEXT"},
        ],
        initial="Check",
        terminal=["Done"],
        expected_length=2,
    )

    print("\n" + "=" * 60)
    print("Branching:")
    viz2 = PathVisualizer(sc2)
    print(viz2.all_formats())
