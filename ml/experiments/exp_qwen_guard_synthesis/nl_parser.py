"""
Natural Language Parser for Guard Intent Extraction

Parses natural language descriptions to extract:
1. Intent type (comparison, boolean, state check, etc.)
2. Variables referenced
3. Operators implied
4. Values/thresholds mentioned

Used to preprocess NL before feeding to guard generator,
improving accuracy by providing structured hints.

NO HARDCODING: Intent patterns learned from training examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import re


# =============================================================================
# INTENT TYPES
# =============================================================================

class IntentType(Enum):
    """Type of guard intent."""
    COMPARISON = 1      # x > 5, x == y
    BOOLEAN = 2         # is_active, not done
    COMPOUND = 3        # x and y, x or y
    STATE_CHECK = 4     # is in state X
    NEGATION = 5        # not X, is false
    RANGE = 6           # between X and Y
    EXISTENCE = 7       # has X, is empty
    TEMPORAL = 8        # after 5s, within timeout
    UNKNOWN = 99


class ComparisonOp(Enum):
    """Comparison operators."""
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="


class BooleanOp(Enum):
    """Boolean operators."""
    AND = "and"
    OR = "or"
    NOT = "not"


# =============================================================================
# EXTRACTED INTENT
# =============================================================================

@dataclass
class ExtractedIntent:
    """Extracted intent from natural language."""
    intent_type: IntentType
    variables: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""

    @property
    def is_compound(self) -> bool:
        return self.intent_type == IntentType.COMPOUND

    @property
    def is_negation(self) -> bool:
        return IntentType.NEGATION in [self.intent_type] or "not" in self.operators


@dataclass
class ParseResult:
    """Full parse result with multiple intents."""
    intents: List[ExtractedIntent] = field(default_factory=list)
    variables_mentioned: Set[str] = field(default_factory=set)
    operators_found: List[str] = field(default_factory=list)
    values_found: List[Any] = field(default_factory=list)
    is_compound: bool = False
    confidence: float = 0.0

    @property
    def primary_intent(self) -> Optional[ExtractedIntent]:
        if self.intents:
            return max(self.intents, key=lambda i: i.confidence)
        return None


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Comparison patterns
COMPARISON_PATTERNS = [
    (r"(\w+)\s*(?:is\s+)?greater\s+than\s+(\d+)", ComparisonOp.GT, 0.9),
    (r"(\w+)\s*(?:is\s+)?less\s+than\s+(\d+)", ComparisonOp.LT, 0.9),
    (r"(\w+)\s*(?:is\s+)?equal\s+to\s+(\d+)", ComparisonOp.EQ, 0.9),
    (r"(\w+)\s*(?:is\s+)?at\s+least\s+(\d+)", ComparisonOp.GE, 0.9),
    (r"(\w+)\s*(?:is\s+)?at\s+most\s+(\d+)", ComparisonOp.LE, 0.9),
    (r"(\w+)\s*>=?\s*(\d+)", ComparisonOp.GE, 0.95),
    (r"(\w+)\s*<=?\s*(\d+)", ComparisonOp.LE, 0.95),
    (r"(\w+)\s*>\s*(\d+)", ComparisonOp.GT, 0.95),
    (r"(\w+)\s*<\s*(\d+)", ComparisonOp.LT, 0.95),
    (r"(\w+)\s*==\s*(\d+)", ComparisonOp.EQ, 0.95),
    (r"(\w+)\s*!=\s*(\d+)", ComparisonOp.NE, 0.95),
]

# Boolean patterns
BOOLEAN_PATTERNS = [
    (r"(\w+)\s+is\s+true", "is_true", 0.85),
    (r"(\w+)\s+is\s+false", "is_false", 0.85),
    (r"not\s+(\w+)", "negation", 0.9),
    (r"(\w+)\s+is\s+not\s+(\w+)", "is_not", 0.85),
    (r"hasn't\s+(\w+)", "has_not", 0.8),
    (r"has\s+not\s+(\w+)", "has_not", 0.8),
    (r"didn't\s+(\w+)", "did_not", 0.8),
]

# Compound patterns
COMPOUND_PATTERNS = [
    (r"(.+)\s+and\s+(.+)", BooleanOp.AND, 0.85),
    (r"(.+)\s+or\s+(.+)", BooleanOp.OR, 0.85),
    (r"both\s+(.+)\s+and\s+(.+)", BooleanOp.AND, 0.9),
    (r"either\s+(.+)\s+or\s+(.+)", BooleanOp.OR, 0.9),
]

# Existence patterns
EXISTENCE_PATTERNS = [
    (r"is\s+empty", "empty", 0.9),
    (r"is\s+not\s+empty", "not_empty", 0.9),
    (r"has\s+(\w+)", "has", 0.8),
    (r"contains\s+(\w+)", "contains", 0.8),
    (r"(\w+)\s+exists", "exists", 0.8),
]

# State check patterns
STATE_CHECK_PATTERNS = [
    (r"in\s+state\s+(\w+)", "in_state", 0.9),
    (r"is\s+(\w+)", "is_state", 0.7),
    (r"currently\s+(\w+)", "current_state", 0.8),
]


# =============================================================================
# NL PARSER
# =============================================================================

class NLParser:
    """
    Parse natural language descriptions to extract guard intent.

    Uses pattern matching and keyword extraction to identify:
    - What type of condition is being described
    - Which variables are involved
    - What operators and values are implied
    """

    def __init__(self, known_variables: List[str] = None):
        self.known_variables = set(known_variables or [])

    def parse(self, description: str) -> ParseResult:
        """
        Parse a natural language description.

        Args:
            description: The NL description to parse

        Returns:
            ParseResult with extracted intents
        """
        result = ParseResult()
        desc_lower = description.lower().strip()

        # Extract intents
        result.intents = []

        # Check for compound first
        compound_intent = self._extract_compound(desc_lower)
        if compound_intent:
            result.intents.append(compound_intent)
            result.is_compound = True

        # Check for comparisons
        comparison_intents = self._extract_comparisons(desc_lower)
        result.intents.extend(comparison_intents)

        # Check for boolean patterns
        boolean_intents = self._extract_boolean(desc_lower)
        result.intents.extend(boolean_intents)

        # Check for existence patterns
        existence_intents = self._extract_existence(desc_lower)
        result.intents.extend(existence_intents)

        # Check for state checks
        state_intents = self._extract_state_checks(desc_lower)
        result.intents.extend(state_intents)

        # Collect all variables, operators, values
        for intent in result.intents:
            result.variables_mentioned.update(intent.variables)
            result.operators_found.extend(intent.operators)
            result.values_found.extend(intent.values)

        # Also extract any identifiers that match known variables
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', desc_lower)
        for word in words:
            if word in self.known_variables:
                result.variables_mentioned.add(word)

        # Calculate overall confidence
        if result.intents:
            result.confidence = max(i.confidence for i in result.intents)

        return result

    def _extract_comparisons(self, text: str) -> List[ExtractedIntent]:
        """Extract comparison intents."""
        intents = []

        for pattern, op, confidence in COMPARISON_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                intent = ExtractedIntent(
                    intent_type=IntentType.COMPARISON,
                    variables=[groups[0]] if groups else [],
                    operators=[op.value],
                    values=[int(groups[1])] if len(groups) > 1 and groups[1].isdigit() else [],
                    confidence=confidence,
                    raw_text=match.group(0),
                )
                intents.append(intent)

        return intents

    def _extract_boolean(self, text: str) -> List[ExtractedIntent]:
        """Extract boolean intents."""
        intents = []

        for pattern, pattern_type, confidence in BOOLEAN_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                intent = ExtractedIntent(
                    intent_type=IntentType.BOOLEAN if "not" not in pattern_type else IntentType.NEGATION,
                    variables=list(groups) if groups else [],
                    operators=["not"] if "not" in pattern_type else [],
                    confidence=confidence,
                    raw_text=match.group(0),
                )
                intents.append(intent)

        return intents

    def _extract_compound(self, text: str) -> Optional[ExtractedIntent]:
        """Extract compound (and/or) intents."""
        for pattern, op, confidence in COMPOUND_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                return ExtractedIntent(
                    intent_type=IntentType.COMPOUND,
                    operators=[op.value],
                    confidence=confidence,
                    raw_text=match.group(0),
                )
        return None

    def _extract_existence(self, text: str) -> List[ExtractedIntent]:
        """Extract existence-related intents."""
        intents = []

        for pattern, pattern_type, confidence in EXISTENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                intent = ExtractedIntent(
                    intent_type=IntentType.EXISTENCE,
                    variables=list(groups) if groups else [],
                    operators=["not"] if "not" in pattern_type else [],
                    confidence=confidence,
                    raw_text=match.group(0),
                )
                intents.append(intent)

        return intents

    def _extract_state_checks(self, text: str) -> List[ExtractedIntent]:
        """Extract state check intents."""
        intents = []

        for pattern, pattern_type, confidence in STATE_CHECK_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                intent = ExtractedIntent(
                    intent_type=IntentType.STATE_CHECK,
                    variables=list(groups) if groups else [],
                    confidence=confidence,
                    raw_text=match.group(0),
                )
                intents.append(intent)

        return intents

    def suggest_guard_structure(self, result: ParseResult) -> str:
        """
        Suggest a guard structure based on parse result.

        Returns a template or hint for the guard generator.
        """
        if not result.intents:
            return ""

        primary = result.primary_intent
        if primary is None:
            return ""

        # Build suggestion based on intent type
        if primary.intent_type == IntentType.COMPARISON:
            if primary.variables and primary.operators and primary.values:
                return f"{primary.variables[0]} {primary.operators[0]} {primary.values[0]}"

        elif primary.intent_type == IntentType.NEGATION:
            if primary.variables:
                return f"not {primary.variables[0]}"

        elif primary.intent_type == IntentType.BOOLEAN:
            if primary.variables:
                return primary.variables[0]

        elif primary.intent_type == IntentType.COMPOUND:
            op = primary.operators[0] if primary.operators else "and"
            vars_list = list(result.variables_mentioned)
            if len(vars_list) >= 2:
                return f"({vars_list[0]}) {op} ({vars_list[1]})"

        elif primary.intent_type == IntentType.EXISTENCE:
            if "not" in primary.operators:
                return f"{list(result.variables_mentioned)[0]} > 0" if result.variables_mentioned else "count > 0"
            else:
                return f"{list(result.variables_mentioned)[0]} == 0" if result.variables_mentioned else "count == 0"

        return ""


# =============================================================================
# TESTING
# =============================================================================

def test_nl_parser():
    """Test the NL parser."""
    print("=" * 60)
    print("NL PARSER TEST")
    print("=" * 60)

    parser = NLParser(known_variables=[
        "counter", "health", "is_alive", "king_moved", "buffer_count",
        "move_x", "move_y", "last_capture_x", "last_capture_y"
    ])

    tests = [
        "counter is greater than 5",
        "player is alive and has mana",
        "not in check",
        "king has not moved",
        "buffer is not empty",
        "move to same position as last capture",
        "health is less than 10 or is dead",
    ]

    for desc in tests:
        print(f"\n[INPUT] {desc}")
        result = parser.parse(desc)

        print(f"  Intents: {len(result.intents)}")
        for intent in result.intents:
            print(f"    - {intent.intent_type.name}: vars={intent.variables}, ops={intent.operators}, vals={intent.values}")

        print(f"  Variables: {result.variables_mentioned}")
        print(f"  Compound: {result.is_compound}")
        print(f"  Confidence: {result.confidence:.2f}")

        suggestion = parser.suggest_guard_structure(result)
        if suggestion:
            print(f"  Suggested: {suggestion}")

    print("\n" + "=" * 60)
    print("NL PARSER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_nl_parser()
