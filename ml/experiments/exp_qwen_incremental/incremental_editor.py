"""
Incremental Statechart Editor using Qwen LLM

Given a statechart and a natural language change request,
generates minimal diffs to apply the requested changes.

STRATEGY:
1. Parse change request to identify operation type
2. Use Qwen to generate structured diff
3. Validate the diff can be applied
4. Apply and validate result

TARGET: 90%+ valid edits
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

from .diff_generator import (
    DiffOp,
    DiffOperation,
    StatechartDiff,
    DiffGenerator,
    DiffApplier,
)


class ChangeType(Enum):
    """Types of changes that can be requested."""
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    RENAME_STATE = "rename_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    MODIFY_TRANSITION = "modify_transition"
    ADD_GUARD = "add_guard"
    ADD_ACTION = "add_action"
    SET_INITIAL = "set_initial"
    ADD_HIERARCHY = "add_hierarchy"
    UNKNOWN = "unknown"


@dataclass
class ChangeRequest:
    """Parsed change request."""
    raw_text: str
    change_type: ChangeType = ChangeType.UNKNOWN
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class EditResult:
    """Result of an incremental edit."""
    success: bool
    original_chart: Dict[str, Any]
    modified_chart: Optional[Dict[str, Any]] = None
    diff: Optional[StatechartDiff] = None
    error: Optional[str] = None
    valid: bool = False
    llm_response: str = ""


class ChangeParser:
    """
    Parses natural language change requests into structured form.
    """

    # Pattern matchers for common change types
    PATTERNS = {
        ChangeType.ADD_STATE: [
            r"add\s+(?:a\s+)?(?:new\s+)?state\s+(?:called\s+|named\s+)?['\"]?(\w+)['\"]?",
            r"create\s+(?:a\s+)?(?:new\s+)?state\s+['\"]?(\w+)['\"]?",
            r"insert\s+(?:a\s+)?(?:new\s+)?state\s+['\"]?(\w+)['\"]?",
            r"add\s+(?:a\s+)?state\s+['\"]?(\w+)['\"]?\s+and\s+connect",
        ],
        ChangeType.REMOVE_STATE: [
            r"remove\s+(?:the\s+)?state\s+['\"]?(\w+)['\"]?",
            r"delete\s+(?:the\s+)?state\s+['\"]?(\w+)['\"]?",
        ],
        ChangeType.RENAME_STATE: [
            r"rename\s+(?:state\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?",
            r"change\s+(?:state\s+)?name\s+(?:from\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?",
        ],
        ChangeType.ADD_TRANSITION: [
            r"add\s+(?:a\s+)?transition\s+from\s+['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?\s+(?:on|with|for)\s+(?:event\s+)?['\"]?(\w+)['\"]?",
            r"connect\s+['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?\s+(?:on|with)\s+['\"]?(\w+)['\"]?",
            r"create\s+(?:a\s+)?(?:retry|transition)\s+.+from\s+['\"]?(\w+)['\"]?\s+(?:back\s+)?to\s+['\"]?(\w+)['\"]?",
            r"adding\s+(?:a\s+)?transition\s+from\s+['\"]?(\w+)['\"]?\s+(?:back\s+)?to\s+['\"]?(\w+)['\"]?",
        ],
        ChangeType.REMOVE_TRANSITION: [
            r"remove\s+(?:the\s+)?transition\s+(?:from\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?",
            r"delete\s+(?:the\s+)?transition\s+['\"]?(\w+)['\"]?\s*->\s*['\"]?(\w+)['\"]?",
        ],
        ChangeType.ADD_GUARD: [
            r"add\s+(?:a\s+)?guard\s+['\"]?(.+?)['\"]?\s+to\s+(?:the\s+)?(?:transition\s+)?(?:from\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?",
            r"guard\s+(?:the\s+)?transition\s+(?:from\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?\s+with\s+['\"]?(.+?)['\"]?",
            r"add\s+guard\s+['\"]?(.+?)['\"]?\s+to\s+(?:the\s+)?first\s+transition",
        ],
        ChangeType.ADD_ACTION: [
            r"add\s+(?:an?\s+)?action\s+['\"]?(.+?)['\"]?\s+to\s+(?:the\s+)?transition\s+(?:from\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?",
        ],
        ChangeType.SET_INITIAL: [
            r"(?:set|make)\s+['\"]?(\w+)['\"]?\s+(?:as\s+)?(?:the\s+)?initial\s+state",
            r"initial\s+state\s+(?:should\s+be|is)\s+['\"]?(\w+)['\"]?",
        ],
    }

    def parse(self, text: str) -> ChangeRequest:
        """Parse change request from natural language."""
        text = text.lower().strip()

        for change_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = self._extract_entities(change_type, match)
                    return ChangeRequest(
                        raw_text=text,
                        change_type=change_type,
                        entities=entities,
                        confidence=0.9,
                    )

        return ChangeRequest(raw_text=text, change_type=ChangeType.UNKNOWN, confidence=0.1)

    def _extract_entities(self, change_type: ChangeType, match: re.Match) -> Dict[str, Any]:
        """Extract entities from regex match."""
        groups = match.groups()

        if change_type == ChangeType.ADD_STATE:
            return {"state_name": groups[0]}
        elif change_type == ChangeType.REMOVE_STATE:
            return {"state_name": groups[0]}
        elif change_type == ChangeType.RENAME_STATE:
            return {"old_name": groups[0], "new_name": groups[1]}
        elif change_type == ChangeType.ADD_TRANSITION:
            # Handle variable group counts
            if len(groups) >= 3 and groups[2]:
                return {"from_state": groups[0], "to_state": groups[1], "event": groups[2]}
            elif len(groups) >= 2:
                return {"from_state": groups[0], "to_state": groups[1], "event": "EVENT"}
            return {}
        elif change_type == ChangeType.REMOVE_TRANSITION:
            return {"from_state": groups[0], "to_state": groups[1]}
        elif change_type == ChangeType.ADD_GUARD:
            # Handle "first transition" case
            if len(groups) == 1:
                return {"guard": groups[0], "first_transition": True}
            return {"guard": groups[0], "from_state": groups[1], "to_state": groups[2]}
        elif change_type == ChangeType.ADD_ACTION:
            return {"action": groups[0], "from_state": groups[1], "to_state": groups[2]}
        elif change_type == ChangeType.SET_INITIAL:
            return {"state_name": groups[0]}

        return {}


class IncrementalEditor:
    """
    Incremental statechart editor using Qwen LLM.

    Generates minimal diffs based on natural language change requests.
    """

    SYSTEM_PROMPT = """You are an expert statechart editor. Given a statechart JSON and a change request,
output ONLY a JSON diff that applies the requested change.

DIFF FORMAT:
{
  "description": "Brief description of change",
  "operations": [
    {"op": "add_state", "path": "root_state.children[N]", "value": {"label": "name", "type": 1}},
    {"op": "remove_state", "path": "root_state.children[N]"},
    {"op": "modify_state", "path": "path.to.property", "value": new_value},
    {"op": "add_transition", "path": "transitions", "value": {"from": ["s1"], "to": ["s2"], "event": "E"}},
    {"op": "remove_transition", "path": "transitions[N]"},
    {"op": "modify_transition", "path": "transitions[N].property", "value": new_value},
    {"op": "rename_state", "path": "states", "value": "new_name", "old": "old_name"}
  ]
}

State types: 1=BASIC, 2=NORMAL (compound), 3=PARALLEL

IMPORTANT:
- Output ONLY valid JSON diff
- Keep changes minimal
- Preserve existing structure
- Update all references when renaming"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.2,
        use_deterministic: bool = True,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_deterministic = use_deterministic
        self.verbose = verbose

        self.parser = ChangeParser()
        self.generator = DiffGenerator()
        self.applier = DiffApplier()

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load Qwen model lazily."""
        if self._model is None and MLX_AVAILABLE:
            if self.verbose:
                print(f"Loading model: {self.model_name}")
            self._model, self._tokenizer = load(self.model_name)

    def edit(
        self,
        chart: Dict[str, Any],
        change_request: str,
    ) -> EditResult:
        """
        Apply incremental edit based on change request.

        Args:
            chart: Original statechart JSON
            change_request: Natural language change description

        Returns:
            EditResult with success status and modified chart
        """
        # Parse the change request
        parsed = self.parser.parse(change_request)

        # Try deterministic edit first
        if self.use_deterministic and parsed.change_type != ChangeType.UNKNOWN:
            result = self._deterministic_edit(chart, parsed)
            if result.success:
                return result

        # Fall back to LLM
        return self._llm_edit(chart, change_request, parsed)

    def _deterministic_edit(
        self,
        chart: Dict[str, Any],
        parsed: ChangeRequest,
    ) -> EditResult:
        """Apply deterministic edit based on parsed request."""
        try:
            diff = StatechartDiff(description=parsed.raw_text)

            if parsed.change_type == ChangeType.ADD_STATE:
                state_name = parsed.entities.get("state_name", "new_state")
                # Find the root's children count
                n_children = len(chart.get("root_state", {}).get("children", []))
                diff.operations.append(DiffOperation(
                    op=DiffOp.ADD_STATE,
                    path=f"root_state.children[{n_children}]",
                    value={"label": state_name, "type": 1, "is_initial": False},
                ))

            elif parsed.change_type == ChangeType.REMOVE_STATE:
                state_name = parsed.entities.get("state_name")
                idx = self._find_state_index(chart, state_name)
                if idx >= 0:
                    diff.operations.append(DiffOperation(
                        op=DiffOp.REMOVE_STATE,
                        path=f"root_state.children[{idx}]",
                        old_value={"label": state_name},
                    ))
                    # Also remove transitions referencing this state
                    for i, t in reversed(list(enumerate(chart.get("transitions", [])))):
                        if state_name in t.get("from", []) or state_name in t.get("to", []):
                            diff.operations.append(DiffOperation(
                                op=DiffOp.REMOVE_TRANSITION,
                                path=f"transitions[{i}]",
                                old_value=t,
                            ))
                else:
                    return EditResult(success=False, original_chart=chart,
                                     error=f"State '{state_name}' not found")

            elif parsed.change_type == ChangeType.RENAME_STATE:
                old_name = parsed.entities.get("old_name")
                new_name = parsed.entities.get("new_name")
                diff.operations.append(DiffOperation(
                    op=DiffOp.RENAME_STATE,
                    path="states",
                    value=new_name,
                    old_value=old_name,
                ))

            elif parsed.change_type == ChangeType.ADD_TRANSITION:
                diff.operations.append(DiffOperation(
                    op=DiffOp.ADD_TRANSITION,
                    path="transitions",
                    value={
                        "from": [parsed.entities.get("from_state")],
                        "to": [parsed.entities.get("to_state")],
                        "event": parsed.entities.get("event", "EVENT"),
                    },
                ))

            elif parsed.change_type == ChangeType.REMOVE_TRANSITION:
                from_state = parsed.entities.get("from_state")
                to_state = parsed.entities.get("to_state")
                idx = self._find_transition_index(chart, from_state, to_state)
                if idx >= 0:
                    diff.operations.append(DiffOperation(
                        op=DiffOp.REMOVE_TRANSITION,
                        path=f"transitions[{idx}]",
                    ))
                else:
                    return EditResult(success=False, original_chart=chart,
                                     error=f"Transition {from_state}->{to_state} not found")

            elif parsed.change_type == ChangeType.SET_INITIAL:
                state_name = parsed.entities.get("state_name")
                idx = self._find_state_index(chart, state_name)
                if idx >= 0:
                    # Clear other initials
                    for i, child in enumerate(chart.get("root_state", {}).get("children", [])):
                        if child.get("is_initial"):
                            diff.operations.append(DiffOperation(
                                op=DiffOp.MODIFY_STATE,
                                path=f"root_state.children[{i}].is_initial",
                                value=False,
                                old_value=True,
                            ))
                    # Set new initial
                    diff.operations.append(DiffOperation(
                        op=DiffOp.MODIFY_STATE,
                        path=f"root_state.children[{idx}].is_initial",
                        value=True,
                        old_value=False,
                    ))
                else:
                    return EditResult(success=False, original_chart=chart,
                                     error=f"State '{state_name}' not found")

            elif parsed.change_type == ChangeType.ADD_GUARD:
                guard = parsed.entities.get("guard")
                if parsed.entities.get("first_transition"):
                    # Add guard to first transition
                    if chart.get("transitions"):
                        diff.operations.append(DiffOperation(
                            op=DiffOp.MODIFY_TRANSITION,
                            path="transitions[0].guard",
                            value=guard,
                        ))
                    else:
                        return EditResult(success=False, original_chart=chart,
                                         error="No transitions to add guard to")
                else:
                    from_state = parsed.entities.get("from_state")
                    to_state = parsed.entities.get("to_state")
                    idx = self._find_transition_index(chart, from_state, to_state)
                    if idx >= 0:
                        diff.operations.append(DiffOperation(
                            op=DiffOp.MODIFY_TRANSITION,
                            path=f"transitions[{idx}].guard",
                            value=guard,
                        ))
                    else:
                        return EditResult(success=False, original_chart=chart,
                                         error=f"Transition {from_state}->{to_state} not found")

            elif parsed.change_type == ChangeType.ADD_ACTION:
                from_state = parsed.entities.get("from_state")
                to_state = parsed.entities.get("to_state")
                action = parsed.entities.get("action")
                idx = self._find_transition_index(chart, from_state, to_state)
                if idx >= 0:
                    diff.operations.append(DiffOperation(
                        op=DiffOp.MODIFY_TRANSITION,
                        path=f"transitions[{idx}].action",
                        value=action,
                    ))
                else:
                    return EditResult(success=False, original_chart=chart,
                                     error=f"Transition {from_state}->{to_state} not found")

            else:
                return EditResult(success=False, original_chart=chart,
                                 error="Cannot handle deterministically")

            # Apply the diff
            if diff.is_empty():
                return EditResult(success=False, original_chart=chart,
                                 error="No operations generated")

            modified = self.applier.apply(chart, diff)
            valid = self._validate_chart(modified)

            return EditResult(
                success=True,
                original_chart=chart,
                modified_chart=modified,
                diff=diff,
                valid=valid,
            )

        except Exception as e:
            return EditResult(success=False, original_chart=chart, error=str(e))

    def _llm_edit(
        self,
        chart: Dict[str, Any],
        change_request: str,
        parsed: ChangeRequest,
    ) -> EditResult:
        """Use LLM to generate diff."""
        if not MLX_AVAILABLE:
            return EditResult(success=False, original_chart=chart,
                             error="MLX not available for LLM inference")

        self._load_model()

        # Build prompt
        prompt = self._build_prompt(chart, change_request)

        try:
            # Generate response
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )

            # Parse response
            diff = self._parse_llm_response(response)
            if diff is None:
                return EditResult(
                    success=False,
                    original_chart=chart,
                    error="Failed to parse LLM response",
                    llm_response=response,
                )

            # Apply diff
            modified = self.applier.apply(chart, diff)
            valid = self._validate_chart(modified)

            return EditResult(
                success=True,
                original_chart=chart,
                modified_chart=modified,
                diff=diff,
                valid=valid,
                llm_response=response,
            )

        except Exception as e:
            return EditResult(success=False, original_chart=chart, error=str(e))

    def _build_prompt(self, chart: Dict[str, Any], change_request: str) -> str:
        """Build prompt for LLM."""
        chart_json = json.dumps(chart, indent=2)
        return f"""{self.SYSTEM_PROMPT}

CURRENT STATECHART:
{chart_json}

CHANGE REQUEST: {change_request}

OUTPUT (JSON diff only):"""

    def _parse_llm_response(self, response: str) -> Optional[StatechartDiff]:
        """Parse LLM response into diff."""
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
            return StatechartDiff.from_dict(data)
        except json.JSONDecodeError:
            return None

    def _find_state_index(self, chart: Dict[str, Any], state_name: str) -> int:
        """Find index of state in root's children."""
        for i, child in enumerate(chart.get("root_state", {}).get("children", [])):
            if child.get("label") == state_name:
                return i
        return -1

    def _find_transition_index(
        self, chart: Dict[str, Any], from_state: str, to_state: str
    ) -> int:
        """Find index of transition."""
        for i, t in enumerate(chart.get("transitions", [])):
            if from_state in t.get("from", []) and to_state in t.get("to", []):
                return i
        return -1

    def _validate_chart(self, chart: Dict[str, Any]) -> bool:
        """Validate chart using basic structural validation.

        Note: sc validate requires protobuf format, so we use basic validation
        for our internal chart format.
        """
        return self._basic_validate(chart)

    def _basic_validate(self, chart: Dict[str, Any]) -> bool:
        """Basic structural validation."""
        if "root_state" not in chart:
            return False

        root = chart["root_state"]
        if "label" not in root:
            return False

        # Check states are properly formed
        def check_state(state):
            if "label" not in state:
                return False
            for child in state.get("children", []):
                if not check_state(child):
                    return False
            return True

        if not check_state(root):
            return False

        # Check transitions reference valid states
        all_states = self._collect_state_labels(root)
        for t in chart.get("transitions", []):
            # Check state references
            for s in t.get("from", []) + t.get("to", []):
                if s not in all_states:
                    return False
            # Guards and actions are optional - just check they're strings if present
            if t.get("guard") is not None and not isinstance(t.get("guard"), str):
                return False
            if t.get("action") is not None and not isinstance(t.get("action"), str):
                return False

        return True

    def _collect_state_labels(self, state: Dict[str, Any]) -> Set[str]:
        """Collect all state labels."""
        labels = {state.get("label", "")}
        for child in state.get("children", []):
            labels.update(self._collect_state_labels(child))
        return labels


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate incremental editing."""
    print("=" * 60)
    print("Incremental Statechart Editor")
    print("=" * 60)

    # Test chart
    chart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "idle", "type": 1, "is_initial": True},
                {"label": "running", "type": 1},
                {"label": "stopped", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["idle"], "to": ["running"], "event": "START"},
            {"from": ["running"], "to": ["stopped"], "event": "STOP"},
        ]
    }

    editor = IncrementalEditor(verbose=True)

    # Test cases
    test_requests = [
        "Add a new state called 'paused'",
        "Add transition from running to paused on PAUSE",
        "Rename state 'stopped' to 'terminated'",
        "Set paused as the initial state",
        "Remove the state 'idle'",
    ]

    results = []
    current_chart = chart

    for request in test_requests:
        print(f"\n{'='*40}")
        print(f"Request: {request}")

        result = editor.edit(current_chart, request)
        results.append(result)

        if result.success:
            print(f"Success: {result.valid}")
            if result.diff:
                print(f"Operations: {len(result.diff)}")
                for op in result.diff.operations:
                    print(f"  - {op.op.value}: {op.path}")
            if result.modified_chart:
                current_chart = result.modified_chart
        else:
            print(f"Failed: {result.error}")

    # Summary
    successful = sum(1 for r in results if r.success)
    valid = sum(1 for r in results if r.valid)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"Valid edits: {valid}/{len(results)} ({100*valid/len(results):.1f}%)")

    return editor, results


if __name__ == "__main__":
    demo()
