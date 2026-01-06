"""
Topology Completer: Analyze and fix incomplete statecharts.

Uses scratchpad enumeration to identify gaps:
1. Reachability check - find unreachable states
2. Terminal check - find dead ends
3. Completeness check - states without transitions
4. Propose fixes to make SC valid
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class TopologyGap:
    """Represents a gap/issue in SC topology."""
    gap_type: str  # "unreachable", "no_terminal", "orphan", "dead_end", "no_initial"
    affected_states: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TopologyFix:
    """Proposed fix for a topology gap."""
    fix_type: str  # "add_transition", "mark_terminal", "mark_initial", "remove_state"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyAnalysis:
    """Analysis results for a partial statechart."""
    gaps: List[TopologyGap] = field(default_factory=list)
    fixes: List[TopologyFix] = field(default_factory=list)
    reachable_states: Set[str] = field(default_factory=set)
    unreachable_states: Set[str] = field(default_factory=set)
    dead_end_states: Set[str] = field(default_factory=set)
    initial_state: Optional[str] = None
    terminal_states: Set[str] = field(default_factory=set)
    scratchpad: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if SC is valid (no gaps)."""
        return len(self.gaps) == 0


@dataclass
class CompletionResult:
    """Result of topology completion."""
    original_sc: Dict = field(default_factory=dict)
    completed_sc: Optional[Dict] = None
    analysis: Optional[TopologyAnalysis] = None
    is_valid: bool = False
    generation_time_ms: float = 0.0


class TopologyAnalyzer:
    """Analyzes statechart topology for gaps."""

    def analyze(self, sc: Dict) -> TopologyAnalysis:
        """
        Analyze SC topology and identify gaps.

        Args:
            sc: Statechart dict with root_state and transitions

        Returns:
            TopologyAnalysis with identified gaps
        """
        result = TopologyAnalysis()
        scratchpad_lines = []

        # Extract states
        states = self._extract_states(sc)
        transitions = sc.get("transitions", [])

        scratchpad_lines.append(f"States found: {sorted(states)}")

        # Build transition graph
        outgoing: Dict[str, Set[str]] = {s: set() for s in states}
        incoming: Dict[str, Set[str]] = {s: set() for s in states}

        for t in transitions:
            froms = t.get("from", [])
            tos = t.get("to", [])
            for src in froms:
                for tgt in tos:
                    if src in outgoing:
                        outgoing[src].add(tgt)
                    if tgt in incoming:
                        incoming[tgt].add(src)

        # Step 1: Find initial state
        result.initial_state = self._find_initial(sc)
        scratchpad_lines.append(f"\nStep 1 - Initial state: {result.initial_state}")

        if not result.initial_state:
            result.gaps.append(TopologyGap(
                gap_type="no_initial",
                description="No initial state marked"
            ))
            # Guess: state with no incoming
            candidates = [s for s in states if not incoming[s]]
            if candidates:
                result.fixes.append(TopologyFix(
                    fix_type="mark_initial",
                    details={"state": candidates[0]}
                ))
                result.initial_state = candidates[0]  # Use for further analysis

        # Step 2: Reachability check
        scratchpad_lines.append("\nStep 2 - Reachability check:")
        if result.initial_state:
            result.reachable_states = self._find_reachable(result.initial_state, outgoing)
            result.unreachable_states = states - result.reachable_states
            scratchpad_lines.append(f"  Reachable from {result.initial_state}: {sorted(result.reachable_states)}")

            if result.unreachable_states:
                scratchpad_lines.append(f"  UNREACHABLE: {sorted(result.unreachable_states)}")
                result.gaps.append(TopologyGap(
                    gap_type="unreachable",
                    affected_states=list(result.unreachable_states),
                    description=f"States {result.unreachable_states} unreachable from initial"
                ))
                # Fix: add transitions from reachable to unreachable
                for unreachable in result.unreachable_states:
                    # Find closest reachable state to connect from
                    closest = self._find_closest_connectable(unreachable, result.reachable_states, incoming)
                    if closest:
                        result.fixes.append(TopologyFix(
                            fix_type="add_transition",
                            details={"from": closest, "to": unreachable, "event": f"TO_{unreachable.upper()}"}
                        ))

        # Step 3: Terminal check
        scratchpad_lines.append("\nStep 3 - Terminal check:")
        result.terminal_states = self._find_terminals(sc)
        result.dead_end_states = {s for s in states if not outgoing[s] and s not in result.terminal_states}

        scratchpad_lines.append(f"  Marked terminals: {sorted(result.terminal_states)}")
        scratchpad_lines.append(f"  Dead ends (no outgoing, not terminal): {sorted(result.dead_end_states)}")

        if result.dead_end_states and not result.terminal_states:
            result.gaps.append(TopologyGap(
                gap_type="no_terminal",
                affected_states=list(result.dead_end_states),
                description="No terminal states, but states have no outgoing"
            ))
            # Fix: mark a dead end as terminal
            for dead_end in result.dead_end_states:
                result.fixes.append(TopologyFix(
                    fix_type="mark_terminal",
                    details={"state": dead_end}
                ))
                break  # Just mark one

        elif result.dead_end_states:
            result.gaps.append(TopologyGap(
                gap_type="dead_end",
                affected_states=list(result.dead_end_states),
                description=f"States {result.dead_end_states} have no outgoing transitions"
            ))
            # Fix: either mark as terminal or add transition to terminal
            for dead_end in result.dead_end_states:
                if result.terminal_states:
                    result.fixes.append(TopologyFix(
                        fix_type="add_transition",
                        details={"from": dead_end, "to": list(result.terminal_states)[0], "event": "COMPLETE"}
                    ))
                else:
                    result.fixes.append(TopologyFix(
                        fix_type="mark_terminal",
                        details={"state": dead_end}
                    ))

        # Step 4: Orphan check
        scratchpad_lines.append("\nStep 4 - Orphan check:")
        orphans = {s for s in states if not incoming[s] and not outgoing[s] and s != result.initial_state}
        scratchpad_lines.append(f"  Orphan states: {sorted(orphans)}")

        if orphans:
            result.gaps.append(TopologyGap(
                gap_type="orphan",
                affected_states=list(orphans),
                description=f"States {orphans} have no connections"
            ))
            # Fix: remove orphan or connect to graph
            for orphan in orphans:
                result.fixes.append(TopologyFix(
                    fix_type="remove_state",
                    details={"state": orphan}
                ))

        # Summary
        scratchpad_lines.append(f"\nGaps found: {len(result.gaps)}")
        scratchpad_lines.append(f"Fixes proposed: {len(result.fixes)}")
        result.scratchpad = "\n".join(scratchpad_lines)

        return result

    def _extract_states(self, sc: Dict) -> Set[str]:
        """Extract all state labels from SC."""
        states = set()

        def recurse(state):
            if not isinstance(state, dict):
                return
            label = state.get("label")
            if label and label != "__root__":
                states.add(label)
            for child in state.get("children", []):
                recurse(child)

        root = sc.get("root_state", {})
        if isinstance(root, dict):
            recurse(root)
        return states

    def _find_initial(self, sc: Dict) -> Optional[str]:
        """Find initial state."""
        root = sc.get("root_state", {})
        if not isinstance(root, dict):
            return None

        for child in root.get("children", []):
            if isinstance(child, dict) and child.get("is_initial"):
                return child.get("label")
        return None

    def _find_terminals(self, sc: Dict) -> Set[str]:
        """Find terminal states."""
        terminals = set()
        root = sc.get("root_state", {})
        if not isinstance(root, dict):
            return terminals

        def recurse(state):
            if not isinstance(state, dict):
                return
            if state.get("is_terminal") or state.get("type") == 4:  # TERMINAL type
                label = state.get("label")
                if label:
                    terminals.add(label)
            for child in state.get("children", []):
                recurse(child)

        recurse(root)
        return terminals

    def _find_reachable(self, start: str, outgoing: Dict[str, Set[str]]) -> Set[str]:
        """Find all states reachable from start."""
        reachable = {start}
        frontier = [start]

        while frontier:
            current = frontier.pop()
            for neighbor in outgoing.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)

        return reachable

    def _find_closest_connectable(
        self,
        target: str,
        reachable: Set[str],
        incoming: Dict[str, Set[str]]
    ) -> Optional[str]:
        """Find a reachable state that could connect to target."""
        # If target has expected incoming, use that source
        expected_sources = incoming.get(target, set())
        for src in expected_sources:
            if src in reachable:
                return src

        # Otherwise pick any reachable state
        if reachable:
            return sorted(reachable)[-1]  # Pick last alphabetically (often makes sense)
        return None


class TopologyCompleter:
    """
    Completes partial statecharts using LLM with scratchpad reasoning.
    """

    def __init__(self, model=None, tokenizer=None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.analyzer = TopologyAnalyzer()

    def complete(self, partial_sc: Dict) -> CompletionResult:
        """
        Complete a partial statechart.

        Args:
            partial_sc: Incomplete statechart

        Returns:
            CompletionResult with completed SC
        """
        result = CompletionResult(original_sc=partial_sc)
        start = time.time()

        # First analyze algorithmically
        analysis = self.analyzer.analyze(partial_sc)
        result.analysis = analysis

        if analysis.is_valid:
            # Already valid
            result.completed_sc = partial_sc
            result.is_valid = True
            result.generation_time_ms = (time.time() - start) * 1000
            return result

        # Build prompt with scratchpad
        prompt = self._build_prompt(partial_sc, analysis)

        # Generate completion
        output = self._generate(prompt)

        # Parse result
        completed = self._parse_completion(output, partial_sc, analysis)
        if completed:
            result.completed_sc = completed

            # Validate completed SC
            final_analysis = self.analyzer.analyze(completed)
            result.is_valid = final_analysis.is_valid

        result.generation_time_ms = (time.time() - start) * 1000
        return result

    def complete_algorithmic(self, partial_sc: Dict) -> CompletionResult:
        """
        Complete using only algorithmic fixes (no LLM).
        """
        result = CompletionResult(original_sc=partial_sc)
        start = time.time()

        # Analyze
        analysis = self.analyzer.analyze(partial_sc)
        result.analysis = analysis

        if analysis.is_valid:
            result.completed_sc = partial_sc
            result.is_valid = True
            result.generation_time_ms = (time.time() - start) * 1000
            return result

        # Apply fixes
        completed = self._apply_fixes(partial_sc, analysis.fixes)
        result.completed_sc = completed

        # Validate
        final_analysis = self.analyzer.analyze(completed)
        result.is_valid = final_analysis.is_valid

        result.generation_time_ms = (time.time() - start) * 1000
        return result

    def _build_prompt(self, partial_sc: Dict, analysis: TopologyAnalysis) -> str:
        """Build prompt with scratchpad analysis."""
        sc_json = json.dumps(partial_sc, indent=2)

        gaps_desc = "\n".join([
            f"- {g.gap_type}: {g.description}"
            for g in analysis.gaps
        ])

        fixes_desc = "\n".join([
            f"- {f.fix_type}: {f.details}"
            for f in analysis.fixes
        ])

        prompt = f"""Complete this partial statechart by fixing the identified issues.

PARTIAL STATECHART:
{sc_json}

SCRATCHPAD ANALYSIS:
{analysis.scratchpad}

GAPS IDENTIFIED:
{gaps_desc}

PROPOSED FIXES:
{fixes_desc}

Apply the fixes and output the COMPLETE statechart JSON.
Ensure:
- All states are reachable from initial
- Dead ends are either marked terminal or have outgoing transitions
- Exactly one initial state exists

COMPLETED JSON:"""

        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=800,
                sampler=sampler,
            )

            if self.verbose:
                print(f"[Completer] Generated: {output[:200]}...")

            return output
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return "{}"

    def _parse_completion(
        self,
        output: str,
        original: Dict,
        analysis: TopologyAnalysis
    ) -> Optional[Dict]:
        """Parse completed SC from LLM output."""
        try:
            json_str = output.strip()
            if '{' not in json_str:
                # Fallback to algorithmic
                return self._apply_fixes(original, analysis.fixes)

            start = json_str.find('{')
            depth = 0
            end = len(json_str)

            for i, c in enumerate(json_str[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                if depth == 0:
                    end = i + 1
                    break

            json_str = json_str[start:end]
            return json.loads(json_str)

        except json.JSONDecodeError:
            # Fallback to algorithmic
            return self._apply_fixes(original, analysis.fixes)

    def _apply_fixes(self, sc: Dict, fixes: List[TopologyFix]) -> Dict:
        """Apply fixes to SC algorithmically."""
        import copy
        completed = copy.deepcopy(sc)

        for fix in fixes:
            if fix.fix_type == "mark_initial":
                state_label = fix.details.get("state")
                self._set_initial(completed, state_label)

            elif fix.fix_type == "mark_terminal":
                state_label = fix.details.get("state")
                self._set_terminal(completed, state_label)

            elif fix.fix_type == "add_transition":
                from_state = fix.details.get("from")
                to_state = fix.details.get("to")
                event = fix.details.get("event", "GO")
                if "transitions" not in completed:
                    completed["transitions"] = []
                completed["transitions"].append({
                    "from": [from_state],
                    "to": [to_state],
                    "event": event
                })

            elif fix.fix_type == "remove_state":
                state_label = fix.details.get("state")
                self._remove_state(completed, state_label)

        return completed

    def _set_initial(self, sc: Dict, label: str):
        """Mark a state as initial."""
        root = sc.get("root_state", {})
        if not isinstance(root, dict):
            return

        for child in root.get("children", []):
            if isinstance(child, dict):
                if child.get("label") == label:
                    child["is_initial"] = True
                else:
                    child["is_initial"] = False

    def _set_terminal(self, sc: Dict, label: str):
        """Mark a state as terminal."""
        root = sc.get("root_state", {})
        if not isinstance(root, dict):
            return

        for child in root.get("children", []):
            if isinstance(child, dict) and child.get("label") == label:
                child["is_terminal"] = True

    def _remove_state(self, sc: Dict, label: str):
        """Remove a state from SC."""
        root = sc.get("root_state", {})
        if not isinstance(root, dict):
            return

        children = root.get("children", [])
        root["children"] = [c for c in children if isinstance(c, dict) and c.get("label") != label]

        # Also remove transitions involving this state
        transitions = sc.get("transitions", [])
        sc["transitions"] = [
            t for t in transitions
            if label not in t.get("from", []) and label not in t.get("to", [])
        ]


if __name__ == "__main__":
    print("Topology Completer Test")
    print("=" * 60)

    # Test incomplete SC
    partial = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "A", "type": 1, "is_initial": True},
                {"label": "B", "type": 1},
                {"label": "C", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["A"], "to": ["B"], "event": "GO"}
        ]
    }

    print("\nPartial SC:")
    print(json.dumps(partial, indent=2))

    analyzer = TopologyAnalyzer()
    analysis = analyzer.analyze(partial)

    print("\nScratchpad:")
    print(analysis.scratchpad)

    print(f"\nGaps: {len(analysis.gaps)}")
    for gap in analysis.gaps:
        print(f"  - {gap.gap_type}: {gap.description}")

    print(f"\nFixes: {len(analysis.fixes)}")
    for fix in analysis.fixes:
        print(f"  - {fix.fix_type}: {fix.details}")

    # Complete algorithmically
    completer = TopologyCompleter()
    result = completer.complete_algorithmic(partial)

    print(f"\nCompleted SC valid: {result.is_valid}")
    print(json.dumps(result.completed_sc, indent=2))
