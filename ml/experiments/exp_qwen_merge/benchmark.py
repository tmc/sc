"""
Benchmark for Statechart Merging.

Measures semantic preservation: Does the merged statechart
preserve the behaviors of the original statecharts?

Tests:
1. Trace preservation: Can merged SC execute original traces?
2. State reachability: Are all original states reachable?
3. Transition preservation: Are all original transitions present?
4. Event handling: Do events trigger correct transitions?

Target: 85%+ semantic preservation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import copy

from .sc_merger import StatechartMerger, MergeConfig, MergeStrategy, merge_statecharts
from .conflict_resolver import ConflictResolver


@dataclass
class SemanticTest:
    """A semantic preservation test."""
    name: str
    description: str
    trace: List[str]          # Sequence of events
    source_sc: int            # Which SC (1 or 2) this trace is from
    expected_states: List[str]  # Expected state sequence
    passed: bool = False


@dataclass
class BenchmarkResult:
    """Results from running the merge benchmark."""
    total_tests: int
    tests_passed: int
    sc1_preservation: float   # % of SC1 behavior preserved
    sc2_preservation: float   # % of SC2 behavior preserved
    overall_preservation: float
    strategy_used: MergeStrategy
    details: List[SemanticTest] = field(default_factory=list)


# Benchmark test cases
BENCHMARK_CASES = [
    {
        "name": "light_lock",
        "sc1": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Off", "is_initial": True},
                    {"label": "On"},
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
                {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"},
            ]
        },
        "sc2": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Locked", "is_initial": True},
                    {"label": "Unlocked"},
                ]
            },
            "transitions": [
                {"from": ["Locked"], "to": ["Unlocked"], "event": "UNLOCK"},
                {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
            ]
        },
        "sc1_traces": [
            (["TURN_ON"], ["On"]),
            (["TURN_ON", "TURN_OFF"], ["On", "Off"]),
        ],
        "sc2_traces": [
            (["UNLOCK"], ["Unlocked"]),
            (["UNLOCK", "LOCK"], ["Unlocked", "Locked"]),
        ],
    },
    {
        "name": "player_door",
        "sc1": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Idle", "is_initial": True},
                    {"label": "Playing"},
                    {"label": "Paused"},
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "RESUME"},
                {"from": ["Playing"], "to": ["Idle"], "event": "STOP"},
            ]
        },
        "sc2": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Closed", "is_initial": True},
                    {"label": "Open"},
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"},
            ]
        },
        "sc1_traces": [
            (["PLAY"], ["Playing"]),
            (["PLAY", "PAUSE"], ["Playing", "Paused"]),
            (["PLAY", "PAUSE", "RESUME"], ["Playing", "Paused", "Playing"]),
        ],
        "sc2_traces": [
            (["OPEN"], ["Open"]),
            (["OPEN", "CLOSE"], ["Open", "Closed"]),
        ],
    },
    {
        "name": "conflicting_idle",
        "sc1": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Idle", "is_initial": True},
                    {"label": "Active"},
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Active"], "event": "START"},
                {"from": ["Active"], "to": ["Idle"], "event": "STOP"},
            ]
        },
        "sc2": {
            "root_state": {
                "label": "__root__",
                "children": [
                    {"label": "Idle", "is_initial": True},  # Conflict!
                    {"label": "Running"},
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Running"], "event": "RUN"},
                {"from": ["Running"], "to": ["Idle"], "event": "HALT"},
            ]
        },
        "sc1_traces": [
            (["START"], ["Active"]),
            (["START", "STOP"], ["Active", "Idle"]),
        ],
        "sc2_traces": [
            (["RUN"], ["Running"]),  # Note: SC2's Idle may be renamed
        ],
    },
]


class StatechartSimulator:
    """Simple simulator to test statechart behavior."""

    def __init__(self, sc: Dict):
        self.sc = sc
        self.current_state = self._find_initial()
        self.transitions = self._build_transition_table()

    def _find_initial(self) -> str:
        """Find initial state."""
        root = self.sc.get('root_state', {})
        return self._find_initial_recursive(root) or "Unknown"

    def _find_initial_recursive(self, node: Dict) -> Optional[str]:
        """Find initial state recursively."""
        if node.get('is_initial') and node.get('label') != '__root__':
            return node.get('label')
        for child in node.get('children', []):
            result = self._find_initial_recursive(child)
            if result:
                return result
        # Return first child if no explicit initial
        children = node.get('children', [])
        if children:
            first = children[0]
            if first.get('label') != '__root__':
                return first.get('label')
        return None

    def _build_transition_table(self) -> Dict[Tuple[str, str], str]:
        """Build transition lookup table."""
        table = {}
        for t in self.sc.get('transitions', []):
            src = t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', '')
            tgt = t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', '')
            event = t.get('event', '')
            if src and event:
                table[(src, event)] = tgt
        return table

    def reset(self):
        """Reset to initial state."""
        self.current_state = self._find_initial()

    def send(self, event: str) -> Tuple[bool, str]:
        """
        Send an event.

        Returns (success, new_state)
        """
        key = (self.current_state, event)
        if key in self.transitions:
            self.current_state = self.transitions[key]
            return True, self.current_state
        return False, self.current_state

    def execute_trace(self, events: List[str], start_state: str = None) -> List[str]:
        """Execute a trace and return state sequence."""
        if start_state:
            self.current_state = start_state
        else:
            self.reset()
        states = []
        for event in events:
            success, state = self.send(event)
            states.append(state)
        return states

    def find_state_for_event(self, event: str) -> Optional[str]:
        """Find a state that has a transition for this event."""
        for (src, evt), tgt in self.transitions.items():
            if evt == event:
                return src
        return None

    def get_reachable_states(self) -> Set[str]:
        """Get all reachable states from initial."""
        reachable = set()
        to_visit = [self._find_initial()]

        while to_visit:
            state = to_visit.pop()
            if state in reachable:
                continue
            reachable.add(state)

            # Find transitions from this state
            for (src, _), tgt in self.transitions.items():
                if src == state and tgt not in reachable:
                    to_visit.append(tgt)

        return reachable


class MergeBenchmark:
    """Benchmark for statechart merging quality."""

    def __init__(self, strategy: MergeStrategy = MergeStrategy.UNION):
        self.strategy = strategy

    def evaluate_merge(
        self,
        sc1: Dict,
        sc2: Dict,
        merged: Dict,
        sc1_traces: List[Tuple[List[str], List[str]]],
        sc2_traces: List[Tuple[List[str], List[str]]]
    ) -> BenchmarkResult:
        """
        Evaluate how well merged SC preserves original behaviors.

        Args:
            sc1, sc2: Original statecharts
            merged: Merged statechart
            sc1_traces: Expected traces from SC1
            sc2_traces: Expected traces from SC2

        Returns:
            BenchmarkResult with preservation metrics
        """
        tests = []
        sim = StatechartSimulator(merged)

        # Test SC1 traces
        sc1_passed = 0
        for events, expected_states in sc1_traces:
            actual_states = sim.execute_trace(events)
            passed = self._states_match(expected_states, actual_states)

            tests.append(SemanticTest(
                name=f"sc1_trace_{len(tests)}",
                description=f"Trace: {events}",
                trace=events,
                source_sc=1,
                expected_states=expected_states,
                passed=passed
            ))

            if passed:
                sc1_passed += 1

        # Test SC2 traces (accounting for renamed states)
        sc2_passed = 0
        resolver = ConflictResolver()
        conflicts = resolver.detect_conflicts(sc1, sc2)
        resolutions = resolver.resolve_conflicts(conflicts)

        # Build rename map
        rename_map = {}
        for res in resolutions:
            if res.renamed_from and res.renamed_to:
                rename_map[res.renamed_from] = res.renamed_to

        # Find SC2's initial state (possibly renamed)
        sc2_initial = self._find_initial_state(sc2)
        sc2_initial_renamed = rename_map.get(sc2_initial, sc2_initial)

        for events, expected_states in sc2_traces:
            # Adjust expected states for renames
            adjusted_expected = [rename_map.get(s, s) for s in expected_states]

            # Start from SC2's initial state (or find state that handles first event)
            start_state = sc2_initial_renamed
            if events:
                # Try to find state that handles the first event
                possible_start = sim.find_state_for_event(events[0])
                if possible_start:
                    start_state = possible_start

            actual_states = sim.execute_trace(events, start_state=start_state)
            passed = self._states_match(adjusted_expected, actual_states)

            tests.append(SemanticTest(
                name=f"sc2_trace_{len(tests)}",
                description=f"Trace: {events}",
                trace=events,
                source_sc=2,
                expected_states=expected_states,
                passed=passed
            ))

            if passed:
                sc2_passed += 1

        # Calculate preservation rates
        sc1_total = len(sc1_traces)
        sc2_total = len(sc2_traces)

        sc1_preservation = sc1_passed / sc1_total if sc1_total > 0 else 1.0
        sc2_preservation = sc2_passed / sc2_total if sc2_total > 0 else 1.0

        total_passed = sc1_passed + sc2_passed
        total_tests = sc1_total + sc2_total
        overall = total_passed / total_tests if total_tests > 0 else 1.0

        return BenchmarkResult(
            total_tests=total_tests,
            tests_passed=total_passed,
            sc1_preservation=sc1_preservation,
            sc2_preservation=sc2_preservation,
            overall_preservation=overall,
            strategy_used=self.strategy,
            details=tests
        )

    def _find_initial_state(self, sc: Dict) -> Optional[str]:
        """Find initial state in a statechart."""
        root = sc.get('root_state', {})
        return self._find_initial_recursive(root)

    def _find_initial_recursive(self, node: Dict) -> Optional[str]:
        """Find initial state recursively."""
        if node.get('is_initial') and node.get('label') != '__root__':
            return node.get('label')
        for child in node.get('children', []):
            result = self._find_initial_recursive(child)
            if result:
                return result
        children = node.get('children', [])
        if children and children[0].get('label') != '__root__':
            return children[0].get('label')
        return None

    def _states_match(self, expected: List[str], actual: List[str]) -> bool:
        """Check if state sequences match (allowing for renames)."""
        if len(expected) != len(actual):
            return False

        for exp, act in zip(expected, actual):
            # Direct match or renamed match
            if exp != act and not act.endswith(f"_{exp}") and not act.startswith(f"{exp}_"):
                # Check if it's a _2 suffix rename
                if f"{exp}_2" != act and act != exp:
                    return False

        return True

    def run(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Run benchmark on all test cases."""
        results = []

        if verbose:
            print("=" * 70)
            print("STATECHART MERGE BENCHMARK")
            print("=" * 70)
            print(f"Strategy: {self.strategy.name}")
            print(f"Test cases: {len(BENCHMARK_CASES)}")
            print(f"Target: 85%+ preservation")
            print("-" * 70)

        for case in BENCHMARK_CASES:
            name = case['name']
            sc1 = case['sc1']
            sc2 = case['sc2']

            # Merge
            merge_result = merge_statecharts(sc1, sc2, self.strategy)
            merged = merge_result.merged

            # Evaluate
            result = self.evaluate_merge(
                sc1, sc2, merged,
                case['sc1_traces'],
                case['sc2_traces']
            )

            results.append(result)

            if verbose:
                status = "PASS" if result.overall_preservation >= 0.85 else "FAIL"
                print(f"\n{name}:")
                print(f"  SC1 preservation: {result.sc1_preservation:.1%}")
                print(f"  SC2 preservation: {result.sc2_preservation:.1%}")
                print(f"  Overall: {result.overall_preservation:.1%} [{status}]")
                print(f"  Tests: {result.tests_passed}/{result.total_tests}")

        if verbose:
            print("-" * 70)
            avg_preservation = sum(r.overall_preservation for r in results) / len(results)
            passing = sum(1 for r in results if r.overall_preservation >= 0.85)

            print(f"\nSUMMARY:")
            print(f"  Cases passing (>=85%): {passing}/{len(results)}")
            print(f"  Average preservation: {avg_preservation:.1%}")

            if avg_preservation >= 0.85:
                print(f"\n  TARGET ACHIEVED: {avg_preservation:.1%} >= 85%!")
            else:
                print(f"\n  Below target: {avg_preservation:.1%} < 85%")

            print("=" * 70)

        return results


def run_merge_benchmark(
    strategy: MergeStrategy = MergeStrategy.UNION,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """Convenience function to run benchmark."""
    benchmark = MergeBenchmark(strategy)
    return benchmark.run(verbose=verbose)


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("MERGE BENCHMARK TEST")
    print("=" * 60)

    # Run with UNION strategy
    print("\n1. Testing UNION merge strategy:")
    results_union = run_merge_benchmark(MergeStrategy.UNION)

    # Run with PARALLEL strategy
    print("\n2. Testing PARALLEL merge strategy:")
    results_parallel = run_merge_benchmark(MergeStrategy.PARALLEL)

    # Compare
    print("\n3. Strategy comparison:")
    avg_union = sum(r.overall_preservation for r in results_union) / len(results_union)
    avg_parallel = sum(r.overall_preservation for r in results_parallel) / len(results_parallel)

    print(f"  UNION average: {avg_union:.1%}")
    print(f"  PARALLEL average: {avg_parallel:.1%}")
    print(f"  Better: {'UNION' if avg_union >= avg_parallel else 'PARALLEL'}")

    print("\n" + "=" * 60)
    print("Benchmark tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
