"""
Action executor using LLM for statechart action execution.

Uses template-based prompting to:
1. Execute actions (entry/exit/transition)
2. Update context with mutations
3. Maintain correct execution order
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class ActionType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    TRANSITION = "transition"


@dataclass
class ExecutionResult:
    """Result of executing a statechart scenario."""
    success: bool
    final_state: str
    final_context: Dict[str, Any]
    action_order: List[str]
    errors: List[str] = field(default_factory=list)

    # Detailed scores
    context_correct: bool = False
    order_correct: bool = False
    state_correct: bool = False


EXECUTION_PROMPT = '''STATECHART ACTION EXECUTION

Given a statechart with actions, execute a sequence of events and track:
1. Action execution order (Harel semantics)
2. Context mutations from each action
3. Final state

HAREL SEMANTICS:
- On transition from A to B:
  1. Execute A's exit actions (bottom-up for nested states)
  2. Execute transition action (if any)
  3. Execute B's entry actions (top-down for nested states)

STATECHART:
States: {states}
Hierarchy: {hierarchy}
Initial state: {initial}
Entry actions: {entry_actions}
Exit actions: {exit_actions}
Transition actions: {trans_actions}
Transitions: {transitions}

INITIAL CONTEXT: {context}

EVENTS TO PROCESS: {events}

EXECUTE step by step, tracking each action:

SCRATCHPAD:
'''

EXECUTION_EXAMPLES = '''EXAMPLE 1:
States: [A, B]
Entry actions: {"B": ["count = 0"]}
Exit actions: {"A": ["cleaned = true"]}
Transition actions: {"A->B": "step = 1"}
Transitions: {"A,GO": "B"}
Initial context: {}
Events: [GO]

SCRATCHPAD:
Event GO in state A:
- Transition: A -> B (via A,GO)
- Execute A.exit: cleaned = true → context = {"cleaned": true}
- Execute trans:A->B: step = 1 → context = {"cleaned": true, "step": 1}
- Execute B.entry: count = 0 → context = {"cleaned": true, "step": 1, "count": 0}
- Now in state: B

RESULT:
{"final_state": "B", "context": {"cleaned": true, "step": 1, "count": 0}, "order": ["A.exit", "trans:A->B", "B.entry"]}

EXAMPLE 2:
States: [Parent, Child, Other]
Hierarchy: {"Parent": ["Child"]}
Entry actions: {}
Exit actions: {"Child": ["order = order + 'c,'"], "Parent": ["order = order + 'p,'"]}
Transitions: {"Child,LEAVE": "Other"}
Initial context: {"order": ""}
Events: [LEAVE]

SCRATCHPAD:
Event LEAVE in state Child (inside Parent):
- Transition: Child -> Other
- Exit bottom-up: Child first, then Parent
- Execute Child.exit: order = "" + "c," → context = {"order": "c,"}
- Execute Parent.exit: order = "c," + "p," → context = {"order": "c,p,"}
- No transition action
- No entry for Other
- Now in state: Other

RESULT:
{"final_state": "Other", "context": {"order": "c,p,"}, "order": ["Child.exit", "Parent.exit"]}
'''


class ActionExecutor:
    """Execute statechart actions using LLM."""

    def __init__(self, model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the MLX model."""
        if not HAS_MLX:
            raise ImportError("mlx_lm not available")
        if self.model is None:
            self.model, self.tokenizer = load(self.model_path)

    def execute(
        self,
        states: List[str],
        hierarchy: Dict[str, List[str]],
        initial_state: str,
        entry_actions: Dict[str, List[str]],
        exit_actions: Dict[str, List[str]],
        transition_actions: Dict[str, str],
        transitions: Dict[str, str],
        initial_context: Dict[str, Any],
        events: List[str],
        guards: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute events and return the result."""
        self.load_model()

        prompt = EXECUTION_EXAMPLES + "\n\n" + EXECUTION_PROMPT.format(
            states=states,
            hierarchy=hierarchy if hierarchy else "{}",
            initial=initial_state,
            entry_actions=json.dumps(entry_actions),
            exit_actions=json.dumps(exit_actions),
            trans_actions=json.dumps(transition_actions),
            transitions=json.dumps(transitions),
            context=json.dumps(initial_context) if initial_context else "{}",
            events=events,
        )

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False,
        )

        return self._parse_response(response, initial_context)

    def _parse_response(self, response: str, initial_context: Dict[str, Any]) -> ExecutionResult:
        """Parse the LLM response to extract execution result."""
        errors = []

        # Extract RESULT JSON
        result_match = re.search(r'RESULT:\s*(\{[^}]+\})', response, re.DOTALL)
        if not result_match:
            # Try to find any JSON object
            json_match = re.search(r'\{[^{}]*"final_state"[^{}]*\}', response)
            if json_match:
                result_match = json_match

        if result_match:
            try:
                result_str = result_match.group(1) if hasattr(result_match, 'group') else result_match.group(0)
                # Clean up common JSON issues
                result_str = result_str.replace("'", '"')
                result_str = re.sub(r'(\w+):', r'"\1":', result_str)  # Quote keys
                result_str = result_str.replace('true', 'True').replace('false', 'False')

                result = eval(result_str)  # Use eval for Python-style booleans

                return ExecutionResult(
                    success=True,
                    final_state=result.get("final_state", ""),
                    final_context=result.get("context", {}),
                    action_order=result.get("order", []),
                )
            except Exception as e:
                errors.append(f"JSON parse error: {e}")

        # Fallback: try to extract from scratchpad
        final_state = self._extract_final_state(response)
        context = self._extract_context(response, initial_context)
        order = self._extract_action_order(response)

        return ExecutionResult(
            success=bool(final_state),
            final_state=final_state,
            final_context=context,
            action_order=order,
            errors=errors,
        )

    def _extract_final_state(self, response: str) -> str:
        """Extract final state from response."""
        patterns = [
            r'Now in state:\s*(\w+)',
            r'final_state["\']?\s*:\s*["\']?(\w+)',
            r'state:\s*(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_context(self, response: str, initial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from response."""
        context = dict(initial)

        # Look for context assignments
        patterns = [
            r'context\s*=\s*(\{[^}]+\})',
            r'→\s*context\s*=\s*(\{[^}]+\})',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    # Take the last context update
                    ctx_str = matches[-1].replace("'", '"')
                    ctx_str = ctx_str.replace('true', 'True').replace('false', 'False')
                    context = eval(ctx_str)
                except:
                    pass

        return context

    def _extract_action_order(self, response: str) -> List[str]:
        """Extract action execution order from response."""
        order = []

        # Look for action executions in scratchpad
        patterns = [
            r'Execute\s+(\w+\.(entry|exit))',
            r'Execute\s+(trans:\w+->\w+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if isinstance(match, tuple):
                    action = match[0]
                else:
                    action = match
                if action not in order:
                    order.append(action)

        return order


def compare_contexts(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Compare two contexts, handling nested objects."""
    if set(expected.keys()) != set(actual.keys()):
        return False

    for key in expected:
        exp_val = expected[key]
        act_val = actual.get(key)

        if isinstance(exp_val, dict):
            if not isinstance(act_val, dict):
                return False
            if not compare_contexts(exp_val, act_val):
                return False
        elif exp_val != act_val:
            return False

    return True


def compare_action_order(expected: List[str], actual: List[str]) -> bool:
    """Compare action execution orders.

    Returns True if expected is a subsequence of actual (allows extra actions).
    Also checks that relative order of expected actions is preserved.
    """
    # Normalize action names
    def normalize(actions):
        result = []
        for a in actions:
            a = a.replace(" ", "").lower()
            result.append(a)
        return result

    exp_norm = normalize(expected)
    act_norm = normalize(actual)

    # Exact match is always good
    if exp_norm == act_norm:
        return True

    # Check if expected is a subsequence of actual (preserving order)
    exp_idx = 0
    for act in act_norm:
        if exp_idx < len(exp_norm) and act == exp_norm[exp_idx]:
            exp_idx += 1

    return exp_idx == len(exp_norm)
