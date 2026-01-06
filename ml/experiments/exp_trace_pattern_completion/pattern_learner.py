"""
Pattern Learner: Extract repeating patterns from example traces.

Detects pattern types:
- Cycle: [A,B,C,A,B,C,...] - repeating sequence
- Alternating: [A,B,A,B,...] - 2-state toggle
- Growth: [A, A,B, A,B,C,...] - incrementally growing
- Nested: [A,B,C,A,B,C,...] with sub-patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter
import math


@dataclass
class Pattern:
    """Detected pattern from traces."""
    pattern_type: str  # "cycle", "alternating", "growth", "nested", "unknown"
    sequence: List[str] = field(default_factory=list)  # The repeating unit
    period: int = 0  # Length of repeating unit
    confidence: float = 0.0  # 0-1 confidence score
    transitions: Dict[str, str] = field(default_factory=dict)  # state -> next state
    scratchpad: str = ""


@dataclass
class PatternAnalysis:
    """Full analysis of trace set."""
    patterns: List[Pattern] = field(default_factory=list)
    primary_pattern: Optional[Pattern] = None
    unique_events: Set[str] = field(default_factory=set)
    avg_trace_length: float = 0.0
    scratchpad: str = ""


class PatternLearner:
    """
    Learns patterns from example traces using algorithmic analysis.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def learn(self, traces: List[List[str]]) -> PatternAnalysis:
        """
        Learn patterns from example traces.

        Args:
            traces: List of example event sequences

        Returns:
            PatternAnalysis with detected patterns
        """
        result = PatternAnalysis()
        scratchpad_lines = []

        if not traces:
            return result

        # Gather stats
        result.unique_events = set(e for t in traces for e in t)
        result.avg_trace_length = sum(len(t) for t in traces) / len(traces)

        scratchpad_lines.append(f"Step 1 - Analyze example traces:")
        for i, trace in enumerate(traces[:5]):
            trace_str = "→".join(trace[:10])
            if len(trace) > 10:
                trace_str += f"... (len={len(trace)})"
            scratchpad_lines.append(f"  Trace {i+1}: {trace_str}")

        # Try different pattern detectors
        patterns = []

        # 1. Detect cycle patterns
        cycle_pattern = self._detect_cycle(traces)
        if cycle_pattern and cycle_pattern.confidence > 0.5:
            patterns.append(cycle_pattern)
            scratchpad_lines.append(f"\nStep 2 - Cycle pattern detected:")
            scratchpad_lines.append(f"  Sequence: {cycle_pattern.sequence}")
            scratchpad_lines.append(f"  Period: {cycle_pattern.period}")
            scratchpad_lines.append(f"  Confidence: {cycle_pattern.confidence:.2f}")

        # 2. Detect alternating patterns
        alt_pattern = self._detect_alternating(traces)
        if alt_pattern and alt_pattern.confidence > 0.5:
            patterns.append(alt_pattern)
            scratchpad_lines.append(f"\nStep 2 - Alternating pattern detected:")
            scratchpad_lines.append(f"  Sequence: {alt_pattern.sequence}")
            scratchpad_lines.append(f"  Confidence: {alt_pattern.confidence:.2f}")

        # 3. Detect growth patterns
        growth_pattern = self._detect_growth(traces)
        if growth_pattern and growth_pattern.confidence > 0.5:
            patterns.append(growth_pattern)
            scratchpad_lines.append(f"\nStep 2 - Growth pattern detected:")
            scratchpad_lines.append(f"  Type: {growth_pattern.pattern_type}")
            scratchpad_lines.append(f"  Confidence: {growth_pattern.confidence:.2f}")

        # Select primary pattern (highest confidence)
        result.patterns = patterns
        if patterns:
            result.primary_pattern = max(patterns, key=lambda p: p.confidence)
            scratchpad_lines.append(f"\nStep 3 - Primary pattern: {result.primary_pattern.pattern_type}")
            scratchpad_lines.append(f"  Transitions: {result.primary_pattern.transitions}")
        else:
            # Fallback: build transition map from data
            fallback = self._build_transition_map(traces)
            fallback.pattern_type = "unknown"
            result.primary_pattern = fallback
            result.patterns = [fallback]
            scratchpad_lines.append(f"\nStep 3 - No clear pattern, using transition map")

        result.scratchpad = "\n".join(scratchpad_lines)
        return result

    def _detect_cycle(self, traces: List[List[str]]) -> Optional[Pattern]:
        """Detect cyclic patterns like [A,B,C,A,B,C,...]."""
        if not traces:
            return None

        # Try different cycle lengths
        best_pattern = None
        best_confidence = 0.0

        for period in range(2, min(10, min(len(t) for t in traces if t) // 2 + 1)):
            confidence, sequence = self._check_cycle_period(traces, period)
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = Pattern(
                    pattern_type="cycle",
                    sequence=sequence,
                    period=period,
                    confidence=confidence,
                )

        if best_pattern and best_confidence > 0.6:
            # Build transition map
            seq = best_pattern.sequence
            best_pattern.transitions = {
                seq[i]: seq[(i + 1) % len(seq)]
                for i in range(len(seq))
            }
            return best_pattern

        return None

    def _check_cycle_period(
        self,
        traces: List[List[str]],
        period: int
    ) -> Tuple[float, List[str]]:
        """Check if traces follow a cycle with given period."""
        if not traces:
            return 0.0, []

        # Extract candidate sequence from first trace
        first_trace = traces[0]
        if len(first_trace) < period:
            return 0.0, []

        candidate = first_trace[:period]

        # Count matches across all traces
        total_positions = 0
        matches = 0

        for trace in traces:
            for i, event in enumerate(trace):
                expected = candidate[i % period]
                if event == expected:
                    matches += 1
                total_positions += 1

        confidence = matches / total_positions if total_positions > 0 else 0.0
        return confidence, candidate

    def _detect_alternating(self, traces: List[List[str]]) -> Optional[Pattern]:
        """Detect alternating patterns like [A,B,A,B,...]."""
        # Special case of period-2 cycle
        if not traces:
            return None

        # Check for period 2
        confidence, sequence = self._check_cycle_period(traces, 2)
        if confidence > 0.8 and len(sequence) == 2:
            return Pattern(
                pattern_type="alternating",
                sequence=sequence,
                period=2,
                confidence=confidence,
                transitions={
                    sequence[0]: sequence[1],
                    sequence[1]: sequence[0],
                },
            )
        return None

    def _detect_growth(self, traces: List[List[str]]) -> Optional[Pattern]:
        """Detect growth patterns like [A, A,B, A,B,C, ...]."""
        if not traces or len(traces) < 2:
            return None

        # Check if trace lengths are increasing
        lengths = [len(t) for t in traces]
        if lengths != sorted(lengths):
            return None

        # Check for prefix relationship
        is_growth = True
        for i in range(1, len(traces)):
            prev = traces[i - 1]
            curr = traces[i]
            if not curr[:len(prev)] == prev:
                is_growth = False
                break

        if is_growth:
            # Build transition map from longest trace
            longest = max(traces, key=len)
            transitions = {}
            for i in range(len(longest) - 1):
                transitions[longest[i]] = longest[i + 1]

            return Pattern(
                pattern_type="growth",
                sequence=longest,
                period=len(longest),
                confidence=0.9,
                transitions=transitions,
            )

        return None

    def _build_transition_map(self, traces: List[List[str]]) -> Pattern:
        """Build transition map from observed transitions."""
        transitions: Dict[str, Counter] = {}

        for trace in traces:
            for i in range(len(trace) - 1):
                src = trace[i]
                tgt = trace[i + 1]
                if src not in transitions:
                    transitions[src] = Counter()
                transitions[src][tgt] += 1

        # Pick most common transition for each state
        transition_map = {}
        for src, targets in transitions.items():
            if targets:
                transition_map[src] = targets.most_common(1)[0][0]

        # Detect sequence from transitions
        sequence = []
        if transition_map:
            # Find a starting point
            start = list(transition_map.keys())[0]
            current = start
            seen = set()
            while current not in seen and current in transition_map:
                sequence.append(current)
                seen.add(current)
                current = transition_map[current]
            if current == start:  # Completed cycle
                pass
            elif current not in seen:
                sequence.append(current)

        return Pattern(
            pattern_type="transition_map",
            sequence=sequence,
            period=len(sequence) if sequence else 0,
            confidence=0.5,
            transitions=transition_map,
        )

    def format_analysis(self, analysis: PatternAnalysis) -> str:
        """Format analysis as human-readable string."""
        lines = [analysis.scratchpad]

        if analysis.primary_pattern:
            p = analysis.primary_pattern
            lines.append(f"\nPrimary Pattern: {p.pattern_type}")
            lines.append(f"  Sequence: {p.sequence}")
            lines.append(f"  Period: {p.period}")
            lines.append(f"  Confidence: {p.confidence:.2f}")

        return "\n".join(lines)


def learn_patterns(traces: List[List[str]]) -> PatternAnalysis:
    """Convenience function to learn patterns."""
    learner = PatternLearner()
    return learner.learn(traces)


if __name__ == "__main__":
    print("Pattern Learner Test")
    print("=" * 60)

    # Test 1: Cycle
    traces1 = [
        ["A", "B", "C", "A", "B", "C"],
        ["A", "B", "C", "A", "B", "C", "A"],
        ["A", "B", "C", "A"],
    ]
    print("\n1. Cycle pattern:")
    analysis1 = learn_patterns(traces1)
    print(analysis1.scratchpad)
    if analysis1.primary_pattern:
        print(f"  Type: {analysis1.primary_pattern.pattern_type}")
        print(f"  Sequence: {analysis1.primary_pattern.sequence}")

    # Test 2: Alternating
    traces2 = [
        ["ON", "OFF", "ON", "OFF", "ON"],
        ["ON", "OFF", "ON", "OFF"],
        ["ON", "OFF", "ON"],
    ]
    print("\n2. Alternating pattern:")
    analysis2 = learn_patterns(traces2)
    if analysis2.primary_pattern:
        print(f"  Type: {analysis2.primary_pattern.pattern_type}")
        print(f"  Sequence: {analysis2.primary_pattern.sequence}")

    # Test 3: Growth
    traces3 = [
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
    ]
    print("\n3. Growth pattern:")
    analysis3 = learn_patterns(traces3)
    if analysis3.primary_pattern:
        print(f"  Type: {analysis3.primary_pattern.pattern_type}")
        print(f"  Sequence: {analysis3.primary_pattern.sequence}")
