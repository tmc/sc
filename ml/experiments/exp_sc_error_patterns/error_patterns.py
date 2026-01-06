"""
Error Patterns: Common Statechart Error Patterns

Catalogs frequently occurring error patterns from:
- LLM-generated statecharts
- Manual authoring mistakes
- Import/export format issues
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable
from enum import Enum, auto
import re

from .error_taxonomy import ErrorCategory, ErrorSeverity


class PatternSource(Enum):
    """Source of error pattern."""
    LLM_GENERATION = "llm_generation"
    MANUAL_AUTHORING = "manual_authoring"
    FORMAT_CONVERSION = "format_conversion"
    REFACTORING = "refactoring"


@dataclass
class ErrorPattern:
    """A common error pattern with detection and examples."""
    name: str
    category: ErrorCategory
    description: str
    source: PatternSource
    frequency: float  # 0-1, how common
    examples: List[Dict[str, Any]]
    detection_hint: str
    prevention_tip: str

    def matches(self, statechart: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if pattern matches in statechart. Returns match details."""
        # Override in specific patterns
        return []


@dataclass
class PatternMatch:
    """A match of a pattern in a statechart."""
    pattern: ErrorPattern
    location: str
    details: Dict[str, Any]
    confidence: float  # 0-1


# Common error patterns catalog
COMMON_PATTERNS: List[ErrorPattern] = []


# Pattern: Typo in state name
class TypoStatePattern(ErrorPattern):
    """Detects likely typos in state names."""

    def __init__(self):
        super().__init__(
            name="state_name_typo",
            category=ErrorCategory.MISSING_STATE,
            description="State name appears to be a typo of an existing state",
            source=PatternSource.MANUAL_AUTHORING,
            frequency=0.25,
            examples=[
                {"error": "Procesing", "likely_meant": "Processing"},
                {"error": "Compelted", "likely_meant": "Completed"},
            ],
            detection_hint="Look for state names with edit distance 1-2 from valid states",
            prevention_tip="Use autocomplete or validation during authoring",
        )

    def matches(self, sc: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []
        defined = self._get_defined_states(sc)
        referenced = self._get_referenced_states(sc)

        for ref in referenced:
            if ref not in defined:
                # Check for similar states
                similar = self._find_similar(ref, defined)
                if similar:
                    matches.append({
                        "pattern": self.name,
                        "typo": ref,
                        "likely_meant": similar,
                        "confidence": 0.8,
                    })

        return matches

    def _get_defined_states(self, sc: Dict) -> Set[str]:
        states = set()
        def collect(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for child in state.get("children", []):
                collect(child)
        collect(sc.get("root_state", {}))
        return states

    def _get_referenced_states(self, sc: Dict) -> Set[str]:
        refs = set()
        for trans in sc.get("transitions", []):
            refs.update(trans.get("from", []))
            refs.update(trans.get("to", []))
        return refs

    def _find_similar(self, target: str, candidates: Set[str]) -> Optional[str]:
        for c in candidates:
            if self._edit_distance(target.lower(), c.lower()) <= 2:
                return c
        return None

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


COMMON_PATTERNS.append(TypoStatePattern())


# Pattern: Missing initial state
class MissingInitialPattern(ErrorPattern):
    """Detects composite states without initial child."""

    def __init__(self):
        super().__init__(
            name="missing_initial_state",
            category=ErrorCategory.HIERARCHY_VIOLATION,
            description="Composite state has no initial child state",
            source=PatternSource.LLM_GENERATION,
            frequency=0.35,
            examples=[
                {"composite": "Main", "children": ["A", "B", "C"], "issue": "none marked initial"},
            ],
            detection_hint="Check type=2 states for is_initial child",
            prevention_tip="Always specify which child is entered first",
        )

    def matches(self, sc: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []

        def check_state(state, path=""):
            label = state.get("label", "")
            state_type = state.get("type", 1)
            children = state.get("children", [])

            if state_type == 2 and children:  # Composite
                has_initial = any(c.get("is_initial") for c in children)
                if not has_initial:
                    child_names = [c.get("label", "?") for c in children]
                    matches.append({
                        "pattern": self.name,
                        "composite": label,
                        "children": child_names,
                        "path": path,
                        "confidence": 0.95,
                    })

            for child in children:
                check_state(child, f"{path}/{label}" if label else path)

        check_state(sc.get("root_state", {}))
        return matches


COMMON_PATTERNS.append(MissingInitialPattern())


# Pattern: Orphan transition (no source)
class OrphanTransitionPattern(ErrorPattern):
    """Detects transitions with empty or invalid source."""

    def __init__(self):
        super().__init__(
            name="orphan_transition",
            category=ErrorCategory.DANGLING_TRANSITION,
            description="Transition has no valid source state",
            source=PatternSource.LLM_GENERATION,
            frequency=0.15,
            examples=[
                {"event": "RESET", "from": [], "to": ["Start"]},
            ],
            detection_hint="Check transitions where 'from' is empty or all invalid",
            prevention_tip="Ensure every transition has at least one valid source",
        )

    def matches(self, sc: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []
        defined = self._get_defined_states(sc)

        for trans in sc.get("transitions", []):
            sources = trans.get("from", [])
            valid_sources = [s for s in sources if s in defined]

            if not sources or not valid_sources:
                matches.append({
                    "pattern": self.name,
                    "event": trans.get("event", "?"),
                    "from": sources,
                    "to": trans.get("to", []),
                    "confidence": 0.9,
                })

        return matches

    def _get_defined_states(self, sc: Dict) -> Set[str]:
        states = set()
        def collect(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for child in state.get("children", []):
                collect(child)
        collect(sc.get("root_state", {}))
        return states


COMMON_PATTERNS.append(OrphanTransitionPattern())


# Pattern: Invalid guard syntax
class InvalidGuardSyntaxPattern(ErrorPattern):
    """Detects common guard syntax errors."""

    def __init__(self):
        super().__init__(
            name="invalid_guard_syntax",
            category=ErrorCategory.INVALID_GUARD,
            description="Guard expression has syntax error",
            source=PatternSource.LLM_GENERATION,
            frequency=0.20,
            examples=[
                {"guard": "count >> 5", "issue": ">> instead of >"},
                {"guard": "x = 10", "issue": "= instead of =="},
                {"guard": "flag = true", "issue": "= instead of =="},
            ],
            detection_hint="Look for common operator typos",
            prevention_tip="Validate guard syntax before saving",
        )
        self.bad_patterns = [
            (r'([^<>=!])=([^=])', 'assignment instead of comparison'),
            (r'>>', 'bitshift instead of greater-than'),
            (r'<<', 'bitshift instead of less-than'),
            (r'&[^&]', 'single & instead of &&'),
            (r'[^|]\|[^|]', 'single | instead of ||'),
        ]

    def matches(self, sc: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []

        for trans in sc.get("transitions", []):
            guard = trans.get("guard", "")
            if not guard:
                continue

            for pattern, issue in self.bad_patterns:
                if re.search(pattern, guard):
                    matches.append({
                        "pattern": self.name,
                        "guard": guard,
                        "event": trans.get("event", "?"),
                        "issue": issue,
                        "confidence": 0.85,
                    })
                    break

        return matches


COMMON_PATTERNS.append(InvalidGuardSyntaxPattern())


# Pattern: Cross-boundary transition
class CrossBoundaryPattern(ErrorPattern):
    """Detects transitions crossing parallel region boundaries."""

    def __init__(self):
        super().__init__(
            name="cross_boundary_transition",
            category=ErrorCategory.HIERARCHY_VIOLATION,
            description="Transition crosses parallel state boundaries incorrectly",
            source=PatternSource.MANUAL_AUTHORING,
            frequency=0.10,
            examples=[
                {"from": "Region1.StateA", "to": "Region2.StateB", "issue": "crosses parallel regions"},
            ],
            detection_hint="Check if source and target are in different parallel regions",
            prevention_tip="Use proper fork/join semantics for parallel transitions",
        )

    def matches(self, sc: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []

        # Build region map
        regions = self._find_parallel_regions(sc)
        if not regions:
            return matches

        for trans in sc.get("transitions", []):
            sources = trans.get("from", [])
            targets = trans.get("to", [])

            for src in sources:
                for tgt in targets:
                    src_region = self._get_region(src, regions)
                    tgt_region = self._get_region(tgt, regions)

                    if src_region and tgt_region and src_region != tgt_region:
                        matches.append({
                            "pattern": self.name,
                            "from": src,
                            "to": tgt,
                            "source_region": src_region,
                            "target_region": tgt_region,
                            "event": trans.get("event", "?"),
                            "confidence": 0.75,
                        })

        return matches

    def _find_parallel_regions(self, sc: Dict) -> Dict[str, str]:
        """Map states to their parallel region."""
        regions = {}

        def scan(state, current_region=None):
            label = state.get("label", "")
            state_type = state.get("type", 1)
            children = state.get("children", [])

            if state_type == 3:  # Parallel
                for i, child in enumerate(children):
                    region_name = f"{label}_region{i}"
                    scan(child, region_name)
            else:
                if current_region and label and not label.startswith("__"):
                    regions[label] = current_region
                for child in children:
                    scan(child, current_region)

        scan(sc.get("root_state", {}))
        return regions

    def _get_region(self, state: str, regions: Dict[str, str]) -> Optional[str]:
        return regions.get(state)


COMMON_PATTERNS.append(CrossBoundaryPattern())


class PatternMatcher:
    """Match error patterns against statecharts."""

    def __init__(self, patterns: Optional[List[ErrorPattern]] = None):
        self.patterns = patterns or COMMON_PATTERNS

    def find_matches(self, statechart: Dict[str, Any]) -> List[PatternMatch]:
        """Find all pattern matches in statechart."""
        all_matches = []

        for pattern in self.patterns:
            matches = pattern.matches(statechart)
            for match in matches:
                all_matches.append(PatternMatch(
                    pattern=pattern,
                    location=match.get("pattern", ""),
                    details=match,
                    confidence=match.get("confidence", 0.5),
                ))

        # Sort by confidence
        all_matches.sort(key=lambda m: -m.confidence)
        return all_matches

    def summarize_matches(self, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Summarize pattern matches."""
        by_category = {}
        for match in matches:
            cat = match.pattern.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(match)

        return {
            "total": len(matches),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "high_confidence": len([m for m in matches if m.confidence >= 0.8]),
        }


def demo():
    """Demonstrate pattern matching."""
    print("=" * 60)
    print("ERROR PATTERNS: Common Statechart Errors")
    print("=" * 60)

    # Statechart with multiple pattern matches
    test_sc = {
        "name": "Pattern Test",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {
                    "label": "Working",
                    "type": 2,  # Composite without initial!
                    "children": [
                        {"label": "Step1", "type": 1},
                        {"label": "Step2", "type": 1},
                    ]
                },
                {"label": "Done", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Working"], "event": "START"},
            {"from": ["Wrking"], "to": ["Done"], "event": "FINISH"},  # Typo
            {"from": [], "to": ["Idle"], "event": "RESET"},  # Orphan
            {"from": ["Working"], "to": ["Done"], "event": "COMPLETE",
             "guard": "count = 5"},  # Bad guard
        ]
    }

    matcher = PatternMatcher()
    matches = matcher.find_matches(test_sc)

    print(f"\nFound {len(matches)} pattern matches:\n")
    for match in matches:
        print(f"Pattern: {match.pattern.name}")
        print(f"  Category: {match.pattern.category.value}")
        print(f"  Confidence: {match.confidence:.0%}")
        print(f"  Details: {match.details}")
        print(f"  Prevention: {match.pattern.prevention_tip}")
        print()

    summary = matcher.summarize_matches(matches)
    print(f"Summary: {summary}")

    return matches


if __name__ == "__main__":
    demo()
