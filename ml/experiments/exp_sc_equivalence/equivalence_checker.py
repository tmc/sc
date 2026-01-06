"""
Equivalence Checker: Determine if two statecharts are behaviorally equivalent.

Two statecharts are equivalent if they accept the same traces (event sequences
leading to same final states).

Methods:
1. Trace comparison - enumerate and compare traces
2. Structural isomorphism - check if same structure
3. Bisimulation - formal equivalence via simulation relation
4. LLM-assisted - semantic understanding for complex cases
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum

from .trace_generator import TraceGenerator, TraceSet

# MLX imports
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class EquivalenceMethod(Enum):
    """Methods for checking equivalence."""
    TRACE = "trace"           # Compare traces
    STRUCTURAL = "structural"  # Check isomorphism
    BISIMULATION = "bisimulation"  # Formal bisimulation
    COMBINED = "combined"     # Use all methods


@dataclass
class EquivalenceConfig:
    """Configuration for equivalence checking."""
    method: EquivalenceMethod = EquivalenceMethod.COMBINED
    max_trace_depth: int = 6
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    use_llm: bool = True


@dataclass
class EquivalenceResult:
    """Result of equivalence check."""
    equivalent: bool
    confidence: float
    method_used: str
    evidence: List[str] = field(default_factory=list)
    counterexample: Optional[Tuple[str, ...]] = None  # Event sequence that differs


class EquivalenceChecker:
    """
    Check behavioral equivalence of two statecharts.
    """

    def __init__(self, config: Optional[EquivalenceConfig] = None):
        self.config = config or EquivalenceConfig()
        self.trace_gen = TraceGenerator(max_depth=self.config.max_trace_depth)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load LLM for semantic analysis."""
        if MLX_AVAILABLE and self.config.use_llm:
            try:
                self.model, self.tokenizer = load(self.config.model_name)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def check_equivalence(
        self,
        sc1: Dict[str, Any],
        sc2: Dict[str, Any],
    ) -> EquivalenceResult:
        """
        Check if two statecharts are behaviorally equivalent.

        Args:
            sc1: First statechart
            sc2: Second statechart

        Returns:
            EquivalenceResult with verdict and evidence
        """
        method = self.config.method

        if method == EquivalenceMethod.TRACE:
            return self._check_trace_equivalence(sc1, sc2)
        elif method == EquivalenceMethod.STRUCTURAL:
            return self._check_structural_equivalence(sc1, sc2)
        elif method == EquivalenceMethod.BISIMULATION:
            return self._check_bisimulation(sc1, sc2)
        else:  # COMBINED
            return self._check_combined(sc1, sc2)

    def _check_trace_equivalence(
        self,
        sc1: Dict[str, Any],
        sc2: Dict[str, Any],
    ) -> EquivalenceResult:
        """Check equivalence by comparing traces."""
        # First check: initial states must be "equivalent" (same outgoing behavior)
        init1 = self._get_initial_state(sc1)
        init2 = self._get_initial_state(sc2)

        if not init1 or not init2:
            return EquivalenceResult(
                equivalent=False,
                confidence=0.5,
                method_used="trace",
                evidence=["Could not determine initial states"],
            )

        traces1 = self.trace_gen.generate(sc1)
        traces2 = self.trace_gen.generate(sc2)

        events1 = traces1.event_sequences
        events2 = traces2.event_sequences

        # Find event sequence differences
        only_in_1 = events1 - events2
        only_in_2 = events2 - events1

        if only_in_1 or only_in_2:
            # Find counterexample
            if only_in_1:
                counterexample = next(iter(only_in_1))
                evidence = [f"SC1 accepts {counterexample} but SC2 does not"]
            else:
                counterexample = next(iter(only_in_2))
                evidence = [f"SC2 accepts {counterexample} but SC1 does not"]

            return EquivalenceResult(
                equivalent=False,
                confidence=1.0,
                method_used="trace",
                evidence=evidence,
                counterexample=counterexample,
            )

        # Same event sequences - check structural differences

        # Check: if state names are IDENTICAL but initial differs, NOT equivalent
        states1 = self._get_states(sc1)
        states2 = self._get_states(sc2)

        if states1 == states2 and init1 != init2:
            return EquivalenceResult(
                equivalent=False,
                confidence=0.95,
                method_used="trace",
                evidence=[
                    f"Identical state names but different initial states",
                    f"SC1 initial: {init1}, SC2 initial: {init2}",
                ],
            )

        # 0. Check initial state behavior (what events are accepted initially?)
        init_events1 = self.trace_gen.get_accepted_events(sc1, init1)
        init_events2 = self.trace_gen.get_accepted_events(sc2, init2)

        # Also check: what is the FIRST transition target for same events?
        trans1 = self._build_transition_map(sc1)
        trans2 = self._build_transition_map(sc2)

        for event in init_events1 & init_events2:
            targets1 = trans1.get(init1, {}).get(event, set())
            targets2 = trans2.get(init2, {}).get(event, set())

            # If both have single target, check if they produce same trace structure
            if len(targets1) == 1 and len(targets2) == 1:
                t1 = next(iter(targets1))
                t2 = next(iter(targets2))

                # Check if from those targets, the behavior continues equivalently
                next_events1 = self.trace_gen.get_accepted_events(sc1, t1)
                next_events2 = self.trace_gen.get_accepted_events(sc2, t2)

                if next_events1 != next_events2:
                    return EquivalenceResult(
                        equivalent=False,
                        confidence=0.85,
                        method_used="trace",
                        evidence=[
                            f"Different behavior after event '{event}'",
                            f"SC1: {init1}->({event})->{t1} then accepts {next_events1}",
                            f"SC2: {init2}->({event})->{t2} then accepts {next_events2}",
                        ],
                    )

        # 1. Check cycle length (when do states repeat?)
        cycle1 = self._detect_cycle_length(sc1)
        cycle2 = self._detect_cycle_length(sc2)

        if cycle1 != cycle2 and cycle1 > 0 and cycle2 > 0:
            return EquivalenceResult(
                equivalent=False,
                confidence=0.9,
                method_used="trace",
                evidence=[
                    f"Different cycle lengths: {cycle1} vs {cycle2}",
                    "State machines have different periodic behavior",
                ],
            )

        # 2. Check number of reachable states
        reach1 = self.trace_gen.get_reachable_states(sc1)
        reach2 = self.trace_gen.get_reachable_states(sc2)

        if len(reach1) != len(reach2):
            return EquivalenceResult(
                equivalent=False,
                confidence=0.85,
                method_used="trace",
                evidence=[
                    f"Different reachable state counts: {len(reach1)} vs {len(reach2)}",
                ],
            )

        # 3. Check trace structure patterns
        # Get distinct state sequences at each length
        state_seqs1 = {t.states for t in traces1.traces}
        state_seqs2 = {t.states for t in traces2.traces}

        # Compare number of distinct paths
        if len(state_seqs1) != len(state_seqs2):
            return EquivalenceResult(
                equivalent=False,
                confidence=0.75,
                method_used="trace",
                evidence=[
                    f"Different number of distinct paths: {len(state_seqs1)} vs {len(state_seqs2)}",
                ],
            )

        # 4. Check state sequence lengths match at each depth
        # This catches different initial states in cycles
        for depth in range(1, self.config.max_trace_depth + 1):
            seqs1_at_depth = {t.states for t in traces1.traces if len(t.steps) == depth}
            seqs2_at_depth = {t.states for t in traces2.traces if len(t.steps) == depth}

            lens1 = {len(s) for s in seqs1_at_depth}
            lens2 = {len(s) for s in seqs2_at_depth}

            if lens1 != lens2:
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.8,
                    method_used="trace",
                    evidence=[
                        f"Different state sequence structure at depth {depth}",
                    ],
                )

        # 5. For renamed states, check if state visit patterns match
        # Count unique states visited at each step
        visits1 = {}  # step -> set of states
        visits2 = {}

        for t in traces1.traces:
            for i, state in enumerate(t.states):
                if i not in visits1:
                    visits1[i] = set()
                visits1[i].add(state)

        for t in traces2.traces:
            for i, state in enumerate(t.states):
                if i not in visits2:
                    visits2[i] = set()
                visits2[i].add(state)

        # Compare cardinality of states at each position
        for i in set(visits1.keys()) | set(visits2.keys()):
            n1 = len(visits1.get(i, set()))
            n2 = len(visits2.get(i, set()))
            if n1 != n2:
                return EquivalenceResult(
                    equivalent=False,
                    confidence=0.8,
                    method_used="trace",
                    evidence=[
                        f"Different number of possible states at step {i}: {n1} vs {n2}",
                    ],
                )

        return EquivalenceResult(
            equivalent=True,
            confidence=0.9,
            method_used="trace",
            evidence=[
                f"Both accept {len(events1)} event sequences",
                f"Trace structure matches",
                f"Checked traces up to depth {self.config.max_trace_depth}",
            ],
        )

    def _check_structural_equivalence(
        self,
        sc1: Dict[str, Any],
        sc2: Dict[str, Any],
    ) -> EquivalenceResult:
        """Check structural isomorphism."""
        # Get state sets
        states1 = self._get_states(sc1)
        states2 = self._get_states(sc2)

        # Get transition signatures
        trans1 = self._get_transition_signatures(sc1)
        trans2 = self._get_transition_signatures(sc2)

        # Get event alphabets
        events1 = {t[1] for t in trans1}
        events2 = {t[1] for t in trans2}

        evidence = []

        # Check state count
        if len(states1) != len(states2):
            evidence.append(f"Different state counts: {len(states1)} vs {len(states2)}")
            return EquivalenceResult(
                equivalent=False,
                confidence=0.8,
                method_used="structural",
                evidence=evidence,
            )

        # Check event alphabets
        if events1 != events2:
            diff = events1.symmetric_difference(events2)
            evidence.append(f"Different event alphabets: {diff}")
            return EquivalenceResult(
                equivalent=False,
                confidence=0.9,
                method_used="structural",
                evidence=evidence,
            )

        # Check transition count
        if len(trans1) != len(trans2):
            evidence.append(f"Different transition counts: {len(trans1)} vs {len(trans2)}")
            return EquivalenceResult(
                equivalent=False,
                confidence=0.7,
                method_used="structural",
                evidence=evidence,
            )

        # If same counts, might be isomorphic
        evidence.append(f"Same state count ({len(states1)})")
        evidence.append(f"Same event alphabet ({events1})")
        evidence.append(f"Same transition count ({len(trans1)})")

        return EquivalenceResult(
            equivalent=True,
            confidence=0.6,  # Lower confidence - structure match != behavioral match
            method_used="structural",
            evidence=evidence,
        )

    def _check_bisimulation(
        self,
        sc1: Dict[str, Any],
        sc2: Dict[str, Any],
    ) -> EquivalenceResult:
        """
        Check bisimulation equivalence.

        Two states are bisimilar if:
        1. They have same outgoing transitions on same events
        2. Target states are also bisimilar
        """
        # Build transition maps
        trans1 = self._build_transition_map(sc1)
        trans2 = self._build_transition_map(sc2)

        # Get initial states
        init1 = self._get_initial_state(sc1)
        init2 = self._get_initial_state(sc2)

        if not init1 or not init2:
            return EquivalenceResult(
                equivalent=False,
                confidence=0.5,
                method_used="bisimulation",
                evidence=["Could not determine initial states"],
            )

        # Check if initial states are bisimilar
        bisim_relation = {}  # (s1, s2) -> bool
        result, evidence = self._check_bisimilar(
            init1, init2, trans1, trans2, bisim_relation
        )

        return EquivalenceResult(
            equivalent=result,
            confidence=0.95 if result else 1.0,
            method_used="bisimulation",
            evidence=evidence,
        )

    def _check_bisimilar(
        self,
        s1: str,
        s2: str,
        trans1: Dict,
        trans2: Dict,
        memo: Dict,
    ) -> Tuple[bool, List[str]]:
        """Check if two states are bisimilar."""
        key = (s1, s2)
        if key in memo:
            return memo[key], []

        # Assume true (for cycles)
        memo[key] = True
        evidence = []

        # Get outgoing events
        events1 = set(trans1.get(s1, {}).keys())
        events2 = set(trans2.get(s2, {}).keys())

        if events1 != events2:
            memo[key] = False
            diff = events1.symmetric_difference(events2)
            evidence.append(f"States {s1}/{s2} have different events: {diff}")
            return False, evidence

        # Check each event leads to bisimilar targets
        for event in events1:
            targets1 = trans1.get(s1, {}).get(event, set())
            targets2 = trans2.get(s2, {}).get(event, set())

            # For each target in sc1, find bisimilar target in sc2
            for t1 in targets1:
                found_match = False
                for t2 in targets2:
                    if self._check_bisimilar(t1, t2, trans1, trans2, memo)[0]:
                        found_match = True
                        break
                if not found_match:
                    memo[key] = False
                    evidence.append(f"No bisimilar match for {t1} from {s1} on {event}")
                    return False, evidence

        evidence.append(f"States {s1} and {s2} are bisimilar")
        return True, evidence

    def _check_combined(
        self,
        sc1: Dict[str, Any],
        sc2: Dict[str, Any],
    ) -> EquivalenceResult:
        """Use multiple methods for robust checking."""
        # Try trace equivalence first (most reliable)
        trace_result = self._check_trace_equivalence(sc1, sc2)

        if not trace_result.equivalent:
            # Definite non-equivalence
            return trace_result

        # Supplement with structural check
        struct_result = self._check_structural_equivalence(sc1, sc2)

        # Combine evidence
        evidence = trace_result.evidence + struct_result.evidence

        # If both agree on equivalence, high confidence
        if trace_result.equivalent and struct_result.equivalent:
            confidence = max(trace_result.confidence, struct_result.confidence)
        else:
            confidence = trace_result.confidence * 0.8  # Reduce if mismatch

        return EquivalenceResult(
            equivalent=trace_result.equivalent,
            confidence=confidence,
            method_used="combined",
            evidence=evidence,
        )

    def _detect_cycle_length(self, sc: Dict) -> int:
        """Detect the cycle length of the statechart (0 if no cycle)."""
        init = self._get_initial_state(sc)
        if not init:
            return 0

        trans_map = self._build_transition_map(sc)
        visited = []
        current = init

        # Follow transitions until we revisit a state
        for _ in range(20):  # Max iterations
            visited.append(current)

            # Get any outgoing transition (deterministic case)
            next_states = []
            for event, targets in trans_map.get(current, {}).items():
                next_states.extend(targets)

            if not next_states:
                return 0  # Dead end, no cycle

            next_state = next(iter(next_states))

            if next_state in visited:
                # Found cycle - length is from first occurrence to now
                cycle_start = visited.index(next_state)
                return len(visited) - cycle_start

            current = next_state

        return 0  # No cycle found within limit

    def _get_states(self, sc: Dict) -> Set[str]:
        """Get all state labels."""
        states = set()

        def collect(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for child in state.get("children", []):
                collect(child)

        collect(sc.get("root_state", {}))
        return states

    def _get_transition_signatures(
        self,
        sc: Dict,
    ) -> Set[Tuple[str, str, str]]:
        """Get transition signatures (from, event, to)."""
        sigs = set()
        for trans in sc.get("transitions", []):
            event = trans.get("event", "")
            for src in trans.get("from", []):
                for tgt in trans.get("to", []):
                    sigs.add((src, event, tgt))
        return sigs

    def _get_initial_state(self, sc: Dict) -> Optional[str]:
        """Get initial state label."""
        def find(state):
            if state.get("is_initial") and not state.get("label", "").startswith("__"):
                return state.get("label")
            for child in state.get("children", []):
                result = find(child)
                if result:
                    return result
            return None

        return find(sc.get("root_state", {}))

    def _build_transition_map(
        self,
        sc: Dict,
    ) -> Dict[str, Dict[str, Set[str]]]:
        """Build map: state -> event -> {target states}."""
        trans_map = {}
        for trans in sc.get("transitions", []):
            event = trans.get("event", "")
            for src in trans.get("from", []):
                if src not in trans_map:
                    trans_map[src] = {}
                if event not in trans_map[src]:
                    trans_map[src][event] = set()
                for tgt in trans.get("to", []):
                    trans_map[src][event].add(tgt)
        return trans_map


def check_equivalence(
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
) -> EquivalenceResult:
    """Convenience function for equivalence checking."""
    checker = EquivalenceChecker()
    return checker.check_equivalence(sc1, sc2)


def demo():
    """Demonstrate equivalence checking."""
    print("=" * 60)
    print("EQUIVALENCE CHECKER: Behavioral Equivalence")
    print("=" * 60)

    # Two equivalent traffic lights (different state names)
    tl1 = {
        "name": "Traffic Light 1",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    tl2 = {
        "name": "Traffic Light 2",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Stop", "type": 1, "is_initial": True},
                {"label": "Go", "type": 1},
                {"label": "Caution", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Stop"], "to": ["Go"], "event": "TIMER"},
            {"from": ["Go"], "to": ["Caution"], "event": "TIMER"},
            {"from": ["Caution"], "to": ["Stop"], "event": "TIMER"},
        ]
    }

    checker = EquivalenceChecker()

    print("\n--- Same Structure, Different Names ---")
    result1 = checker.check_equivalence(tl1, tl2)
    print(f"Equivalent: {result1.equivalent} (conf: {result1.confidence:.2f})")
    print(f"Method: {result1.method_used}")
    for e in result1.evidence:
        print(f"  - {e}")

    # Non-equivalent (different behavior)
    tl3 = {
        "name": "Traffic Light 3",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    print("\n--- Different Behavior (missing Yellow) ---")
    result2 = checker.check_equivalence(tl1, tl3)
    print(f"Equivalent: {result2.equivalent} (conf: {result2.confidence:.2f})")
    print(f"Method: {result2.method_used}")
    for e in result2.evidence:
        print(f"  - {e}")
    if result2.counterexample:
        print(f"  Counterexample: {result2.counterexample}")

    return checker


if __name__ == "__main__":
    demo()
