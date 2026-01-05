"""
SC Debugger using LLM reasoning.

Given a statechart and failing trace, identifies the bug.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from mlx_lm import load, generate

from .test_cases import DebugTestCase, BugType


@dataclass
class DebugResult:
    """Result of debugging analysis."""
    test_name: str
    identified_bug_type: Optional[str]
    identified_location: Optional[str]
    suggested_fix: str
    correct_bug_type: bool
    correct_location: bool
    raw_response: str


DEBUG_PROMPT_TEMPLATE = """Statechart bug analysis.

Statechart: {statechart_oneline}

Trace: {trace_oneline}

Expected: {expected}
Actual: {actual}

What's wrong? Answer with:
BUG_TYPE: missing_transition OR wrong_target OR wrong_event OR missing_guard
BUG_LOCATION: <which transition>
FIX: <how to fix>

Answer:
BUG_TYPE:"""


class SCDebugger:
    """LLM-based statechart debugger."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model if not already loaded."""
        if self.model is None:
            print(f"Loading model: {self.model_id}")
            self.model, self.tokenizer = load(self.model_id)

    def format_trace(self, trace: List[Dict[str, Any]]) -> str:
        """Format execution trace for prompt."""
        lines = []
        for i, step in enumerate(trace, 1):
            state = step.get("state", "?")
            event = step.get("event", "?")
            result = step.get("result", "?")
            context = step.get("context", {})

            line = f"{i}. State: {state}, Event: {event}"
            if context:
                line += f", Context: {context}"
            line += f" -> Result: {result}"
            lines.append(line)

        return "\n".join(lines)

    def create_prompt(self, test_case: DebugTestCase) -> str:
        """Create debugging prompt from test case."""
        # Compact JSON for smaller prompt
        statechart_oneline = json.dumps(test_case.statechart, separators=(',', ':'))
        trace_oneline = "; ".join(
            f"{s['state']}+{s['event']}->{s['result']}" for s in test_case.trace
        )
        return DEBUG_PROMPT_TEMPLATE.format(
            statechart_oneline=statechart_oneline,
            trace_oneline=trace_oneline,
            expected=test_case.expected_behavior,
            actual=test_case.actual_behavior,
        )

    def parse_response(self, response: str) -> Dict[str, Optional[str]]:
        """Parse LLM response to extract bug analysis."""
        result = {
            "bug_type": None,
            "bug_location": None,
            "fix": None,
        }

        # The response starts right after "BUG_TYPE:" in prompt
        # So first word/line should be the bug type
        lines = response.strip().split('\n')

        # First line should be bug type (possibly with extra text)
        if lines:
            first_line = lines[0].strip()
            # Extract first word that matches a known type
            for bug_type in ['missing_transition', 'wrong_target', 'wrong_event',
                            'missing_guard', 'wrong_guard', 'unreachable_state']:
                if bug_type in first_line.lower():
                    result["bug_type"] = bug_type
                    break

            # If no match, try first word
            if not result["bug_type"]:
                first_word = first_line.split()[0] if first_line.split() else ""
                result["bug_type"] = first_word.lower().strip('.,;:')

        # Extract BUG_LOCATION
        loc_match = re.search(r'BUG_LOCATION:\s*(.+?)(?:\n|FIX:|$)', response, re.IGNORECASE)
        if loc_match:
            result["bug_location"] = loc_match.group(1).strip()

        # Extract FIX
        fix_match = re.search(r'FIX:\s*(.+?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        if fix_match:
            result["fix"] = fix_match.group(1).strip()

        return result

    def debug(self, test_case: DebugTestCase, max_tokens: int = 300) -> DebugResult:
        """Run debugging on a test case."""
        self.load_model()

        prompt = self.create_prompt(test_case)

        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Parse response
        parsed = self.parse_response(response)

        # Check correctness
        correct_type = False
        if parsed["bug_type"]:
            # Normalize and compare
            identified = parsed["bug_type"].replace("-", "_").lower()
            expected = test_case.bug_type.value.lower()
            correct_type = identified == expected or identified in expected or expected in identified

        correct_location = False
        if parsed["bug_location"]:
            # Check if key terms from expected location appear in identified
            expected_terms = test_case.bug_location.lower().split()
            identified_loc = parsed["bug_location"].lower()
            matches = sum(1 for term in expected_terms if term in identified_loc)
            correct_location = matches >= len(expected_terms) // 2 + 1

        return DebugResult(
            test_name=test_case.name,
            identified_bug_type=parsed["bug_type"],
            identified_location=parsed["bug_location"],
            suggested_fix=parsed["fix"] or "",
            correct_bug_type=correct_type,
            correct_location=correct_location,
            raw_response=response,
        )


def test_debugger():
    """Test the debugger on a few cases."""
    from .test_cases import generate_test_cases

    print("=" * 70)
    print("Testing SC Debugger")
    print("=" * 70)

    debugger = SCDebugger()
    cases = generate_test_cases()[:3]  # Just first 3 for quick test

    for case in cases:
        print(f"\nCase: {case.name}")
        print(f"  Bug type: {case.bug_type.value}")

        result = debugger.debug(case)

        print(f"  Identified: {result.identified_bug_type}")
        print(f"  Location: {result.identified_location}")
        print(f"  Correct type: {result.correct_bug_type}")
        print(f"  Correct location: {result.correct_location}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_debugger()
