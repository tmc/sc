"""
Pattern Extractor - Extract FSM patterns using Qwen2.5-Coder.

Uses MLX LM with Qwen2.5-Coder-0.5B-Instruct to:
1. Identify state machine patterns in code
2. Extract states, events, and transitions
3. Generate structured JSON representation

Patterns detected:
- Switch/case state machines
- If/else state handling
- Event-driven patterns
- State transition tables
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple

try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

from .code_analyzer import CodeAnalysis, StateTransition, MultiLanguageAnalyzer


@dataclass
class ExtractedPattern:
    """An FSM pattern extracted from code."""
    pattern_type: str  # "switch_case", "if_else", "event_handler"
    states: List[str] = field(default_factory=list)
    initial_state: Optional[str] = None
    final_states: List[str] = field(default_factory=list)
    transitions: List[Dict[str, str]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    guards: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'states': self.states,
            'initial_state': self.initial_state,
            'final_states': self.final_states,
            'transitions': self.transitions,
            'events': self.events,
            'guards': self.guards,
            'actions': self.actions,
            'confidence': self.confidence,
        }


@dataclass
class ExtractionResult:
    """Complete extraction result."""
    source_code: str
    patterns: List[ExtractedPattern] = field(default_factory=list)
    combined_states: Set[str] = field(default_factory=set)
    combined_transitions: List[Dict] = field(default_factory=list)
    raw_llm_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)


# Prompts for Qwen extraction
EXTRACTION_PROMPT = '''Analyze this code and extract the state machine structure.

Code:
```
{code}
```

Extract:
1. All states (including initial and final)
2. All transitions (from_state, to_state, event, guard, action)
3. All events that trigger transitions

Return JSON format:
{{
  "states": ["state1", "state2", ...],
  "initial_state": "initial",
  "final_states": ["final1", ...],
  "transitions": [
    {{"from": "state1", "to": "state2", "event": "event1", "guard": "condition", "action": "do_something"}}
  ],
  "events": ["event1", "event2", ...]
}}

JSON:'''

VALIDATION_PROMPT = '''Given this code and extracted state machine, verify correctness.

Code:
```
{code}
```

Extracted:
{extracted}

Is this correct? List any missing states, transitions, or errors.
Return JSON: {{"correct": true/false, "missing_states": [], "missing_transitions": [], "errors": []}}

JSON:'''


class QwenPatternExtractor:
    """Extract FSM patterns using Qwen2.5-Coder."""

    MODEL_NAME = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"

    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path or self.MODEL_NAME
        self.analyzer = MultiLanguageAnalyzer()

    def load_model(self):
        """Load Qwen model."""
        if not HAS_MLX_LM:
            raise ImportError("mlx_lm not installed. Run: pip install mlx-lm")

        if self.model is None:
            print(f"Loading model: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path)
            print("Model loaded.")

    def extract(self, code: str, use_llm: bool = True) -> ExtractionResult:
        """
        Extract FSM patterns from code.

        Args:
            code: Source code to analyze
            use_llm: Whether to use LLM (False = AST only)
        """
        result = ExtractionResult(source_code=code)

        # Phase 1: AST-based analysis
        ast_analysis = self.analyzer.analyze(code)
        ast_pattern = self._analysis_to_pattern(ast_analysis)
        if ast_pattern.states:
            result.patterns.append(ast_pattern)

        # Phase 2: LLM-based extraction (if available and requested)
        if use_llm and HAS_MLX_LM:
            try:
                llm_pattern = self._extract_with_llm(code)
                if llm_pattern.states:
                    result.patterns.append(llm_pattern)
            except Exception as e:
                result.errors.append(f"LLM extraction error: {e}")

        # Combine results
        for pattern in result.patterns:
            result.combined_states.update(pattern.states)
            result.combined_transitions.extend(pattern.transitions)

        return result

    def _analysis_to_pattern(self, analysis: CodeAnalysis) -> ExtractedPattern:
        """Convert CodeAnalysis to ExtractedPattern."""
        pattern = ExtractedPattern(
            pattern_type="ast_analysis",
            states=list(analysis.states),
            events=list(analysis.events),
            confidence=0.8,  # AST is reliable
        )

        # Find initial state
        for sv in analysis.state_variables:
            if sv.initial_value:
                pattern.initial_state = sv.initial_value
                break

        # Convert transitions
        for t in analysis.transitions:
            pattern.transitions.append({
                'from': t.from_state or '*',
                'to': t.to_state,
                'event': t.event or '',
                'guard': t.condition or '',
                'action': t.action or '',
            })

        return pattern

    def _extract_with_llm(self, code: str) -> ExtractedPattern:
        """Extract pattern using Qwen LLM."""
        self.load_model()

        prompt = EXTRACTION_PROMPT.format(code=code[:2000])  # Truncate long code

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=500,
            verbose=False,
        )

        # Parse JSON from response
        pattern = self._parse_llm_response(response)
        pattern.pattern_type = "llm_extraction"
        pattern.confidence = 0.7  # LLM may hallucinate

        return pattern

    def _parse_llm_response(self, response: str) -> ExtractedPattern:
        """Parse LLM JSON response."""
        pattern = ExtractedPattern()

        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                pattern.states = data.get('states', [])
                pattern.initial_state = data.get('initial_state')
                pattern.final_states = data.get('final_states', [])
                pattern.transitions = data.get('transitions', [])
                pattern.events = data.get('events', [])
        except json.JSONDecodeError:
            pass

        return pattern


class RuleBasedExtractor:
    """Extract FSM patterns using rule-based heuristics (no LLM)."""

    def __init__(self):
        self.analyzer = MultiLanguageAnalyzer()

    def extract(self, code: str) -> ExtractionResult:
        """Extract FSM patterns using rules only."""
        result = ExtractionResult(source_code=code)

        analysis = self.analyzer.analyze(code)

        # Create pattern from analysis
        pattern = ExtractedPattern(
            pattern_type=self._detect_pattern_type(code),
            states=list(analysis.states),
            events=list(analysis.events),
            confidence=0.9,
        )

        # Determine initial state
        for sv in analysis.state_variables:
            if sv.initial_value:
                pattern.initial_state = sv.initial_value
                break

        # Detect final states (heuristic: states with "final", "done", "end")
        for state in pattern.states:
            if any(kw in state.lower() for kw in ['final', 'done', 'end', 'stop', 'complete']):
                pattern.final_states.append(state)

        # Convert transitions
        for t in analysis.transitions:
            pattern.transitions.append({
                'from': t.from_state or '*',
                'to': t.to_state,
                'event': t.event or '',
                'guard': t.condition or '',
                'action': t.action or '',
            })

        # Extract events from transitions
        for t in pattern.transitions:
            if t.get('event'):
                pattern.events.append(t['event'])
        pattern.events = list(set(pattern.events))

        result.patterns.append(pattern)
        result.combined_states = set(pattern.states)
        result.combined_transitions = pattern.transitions

        return result

    def _detect_pattern_type(self, code: str) -> str:
        """Detect FSM pattern type from code."""
        if 'switch' in code and 'case' in code:
            return "switch_case"
        elif 'match ' in code:
            return "match_statement"
        elif 'if ' in code and ('state' in code.lower() or 'status' in code.lower()):
            return "if_else"
        else:
            return "unknown"


class HybridExtractor:
    """Combine rule-based and LLM extraction."""

    def __init__(self, use_llm: bool = True):
        self.rule_extractor = RuleBasedExtractor()
        self.qwen_extractor = QwenPatternExtractor() if use_llm and HAS_MLX_LM else None

    def extract(self, code: str) -> ExtractionResult:
        """Extract using both methods and merge."""
        # Rule-based extraction (always)
        rule_result = self.rule_extractor.extract(code)

        # LLM extraction (if available)
        if self.qwen_extractor:
            try:
                llm_result = self.qwen_extractor.extract(code, use_llm=True)
                return self._merge_results(rule_result, llm_result)
            except Exception:
                pass

        return rule_result

    def _merge_results(
        self,
        rule_result: ExtractionResult,
        llm_result: ExtractionResult,
    ) -> ExtractionResult:
        """Merge results from different extractors."""
        merged = ExtractionResult(source_code=rule_result.source_code)

        # Combine patterns
        merged.patterns.extend(rule_result.patterns)
        merged.patterns.extend(llm_result.patterns)

        # Union of states (trust rule-based more)
        merged.combined_states = rule_result.combined_states | llm_result.combined_states

        # Combine transitions (deduplicate)
        seen = set()
        for t in rule_result.combined_transitions + llm_result.combined_transitions:
            key = (t.get('from'), t.get('to'), t.get('event'))
            if key not in seen:
                seen.add(key)
                merged.combined_transitions.append(t)

        return merged


def demo():
    """Demonstrate pattern extraction."""
    print("=" * 60)
    print("PATTERN EXTRACTOR")
    print("=" * 60)

    code = '''
def traffic_light():
    state = "red"

    while True:
        if state == "red":
            wait(30)
            state = "green"
        elif state == "green":
            wait(25)
            state = "yellow"
        elif state == "yellow":
            wait(5)
            state = "red"
'''

    print("\nCode:")
    print(code)

    # Rule-based extraction
    print("\n--- Rule-Based Extraction ---")
    extractor = RuleBasedExtractor()
    result = extractor.extract(code)

    print(f"Pattern type: {result.patterns[0].pattern_type}")
    print(f"States: {result.combined_states}")
    print(f"Initial: {result.patterns[0].initial_state}")
    print(f"Transitions: {len(result.combined_transitions)}")
    for t in result.combined_transitions:
        print(f"  {t['from']} -> {t['to']}")

    # C-like code
    c_code = '''
switch (light_state) {
    case RED:
        if (timer_expired) {
            light_state = GREEN;
            start_timer(25);
        }
        break;
    case GREEN:
        if (timer_expired) {
            light_state = YELLOW;
            start_timer(5);
        }
        break;
    case YELLOW:
        if (timer_expired) {
            light_state = RED;
            start_timer(30);
        }
        break;
}
'''

    print("\n--- C-like Code Extraction ---")
    result2 = extractor.extract(c_code)
    print(f"Pattern type: {result2.patterns[0].pattern_type}")
    print(f"States: {result2.combined_states}")
    print(f"Transitions: {len(result2.combined_transitions)}")
    for t in result2.combined_transitions:
        print(f"  {t['from']} -> {t['to']}")

    return result, result2


if __name__ == "__main__":
    demo()
