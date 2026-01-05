"""
FIM Generator - Fill-in-Middle generation with structure awareness.

Provides structure-aware prompting for statechart hole filling.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any


class HoleType(Enum):
    """Types of holes that can be filled in a statechart."""
    STATE = "state"           # Insert child state
    TRANSITION = "transition" # Insert transition
    GUARD = "guard"           # Insert guard expression
    ACTION = "action"         # Insert entry/exit action


@dataclass
class FIMResult:
    """Result of a FIM generation attempt."""
    hole_type: HoleType
    prefix: str
    suffix: str
    generated: str
    is_valid: bool
    error: Optional[str] = None
    generation_time_ms: float = 0.0


@dataclass
class StructureContext:
    """Structural context for informed hole filling."""
    existing_states: List[str] = field(default_factory=list)
    existing_events: List[str] = field(default_factory=list)
    parent_state: Optional[str] = None
    sibling_states: List[str] = field(default_factory=list)
    available_targets: List[str] = field(default_factory=list)


class FIMGenerator:
    """
    Structure-aware Fill-in-Middle generator for statecharts.

    Uses MLX with Qwen2.5-Coder to generate valid statechart fragments.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load the MLX model and tokenizer."""
        try:
            from mlx_lm import load, generate
            self.model, self.tokenizer = load(self.model_name)
            self._generate_fn = generate
            self._loaded = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def generate_fim(
        self,
        prefix: str,
        suffix: str,
        hole_type: HoleType,
        context: Optional[StructureContext] = None,
        max_tokens: int = 100
    ) -> FIMResult:
        """
        Generate content to fill the hole between prefix and suffix.

        Args:
            prefix: Content before the hole
            suffix: Content after the hole
            hole_type: Type of content to generate
            context: Structural context for informed generation
            max_tokens: Maximum tokens to generate

        Returns:
            FIMResult with generated content and validation status
        """
        import time

        if not self._loaded:
            if not self.load_model():
                return FIMResult(
                    hole_type=hole_type,
                    prefix=prefix,
                    suffix=suffix,
                    generated="",
                    is_valid=False,
                    error="Model not loaded"
                )

        # Build structure-aware prompt
        prompt = self._build_prompt(prefix, suffix, hole_type, context)

        # Generate
        start = time.time()
        try:
            generated = self._generate_fn(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            generation_time = (time.time() - start) * 1000

            # Extract just the middle content
            middle = self._extract_middle(generated, hole_type)

            # Validate the generated content
            is_valid, error = self._validate(middle, hole_type, context)

            return FIMResult(
                hole_type=hole_type,
                prefix=prefix,
                suffix=suffix,
                generated=middle,
                is_valid=is_valid,
                error=error,
                generation_time_ms=generation_time
            )

        except Exception as e:
            return FIMResult(
                hole_type=hole_type,
                prefix=prefix,
                suffix=suffix,
                generated="",
                is_valid=False,
                error=str(e),
                generation_time_ms=(time.time() - start) * 1000
            )

    def _build_prompt(
        self,
        prefix: str,
        suffix: str,
        hole_type: HoleType,
        context: Optional[StructureContext]
    ) -> str:
        """Build a structure-aware FIM prompt."""

        # Base instruction based on hole type
        instructions = {
            HoleType.STATE: "Generate a valid JSON state object with label, type (1=BASIC, 2=NORMAL), and is_initial fields.",
            HoleType.TRANSITION: "Generate a valid JSON transition object with from, to (arrays), and event fields.",
            HoleType.GUARD: "Generate a valid guard expression string for a statechart transition.",
            HoleType.ACTION: "Generate a valid action object with name field for a state entry/exit action."
        }

        instruction = instructions[hole_type]

        # Add context if available
        context_str = ""
        if context:
            if context.existing_states:
                context_str += f"\nExisting states: {context.existing_states}"
            if context.existing_events:
                context_str += f"\nExisting events: {context.existing_events}"
            if context.parent_state:
                context_str += f"\nParent state: {context.parent_state}"
            if context.available_targets:
                context_str += f"\nValid transition targets: {context.available_targets}"

        # Build the prompt
        prompt = f"""You are a statechart JSON generator. {instruction}
{context_str}

Complete the MIDDLE section to create valid JSON:

<PREFIX>
{prefix}
</PREFIX>

<SUFFIX>
{suffix}
</SUFFIX>

<MIDDLE>"""

        return prompt

    def _extract_middle(self, generated: str, hole_type: HoleType) -> str:
        """Extract the middle content from generation."""
        # Try to extract JSON object or string
        text = generated.strip()

        # Remove any trailing tags
        for tag in ['</MIDDLE>', '</PREFIX>', '</SUFFIX>', '<SUFFIX>']:
            if tag in text:
                text = text.split(tag)[0]

        # For guard type, might be a simple string
        if hole_type == HoleType.GUARD:
            # Extract quoted string or expression
            match = re.search(r'"([^"]*)"', text)
            if match:
                return match.group(1)
            return text.strip().strip('"').strip("'")

        # For other types, try to extract JSON object
        try:
            # Find JSON object boundaries
            start = text.find('{')
            if start == -1:
                return text

            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            return text[start:end]
        except Exception:
            return text

    def _validate(
        self,
        content: str,
        hole_type: HoleType,
        context: Optional[StructureContext]
    ) -> Tuple[bool, Optional[str]]:
        """Validate generated content against structural constraints."""

        if not content.strip():
            return False, "Empty content"

        if hole_type == HoleType.STATE:
            return self._validate_state(content, context)
        elif hole_type == HoleType.TRANSITION:
            return self._validate_transition(content, context)
        elif hole_type == HoleType.GUARD:
            return self._validate_guard(content, context)
        elif hole_type == HoleType.ACTION:
            return self._validate_action(content, context)

        return False, "Unknown hole type"

    def _validate_state(
        self, content: str, context: Optional[StructureContext]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a state object."""
        try:
            obj = json.loads(content)

            # Must have label
            if 'label' not in obj:
                return False, "Missing label field"

            # Label must be non-empty string
            if not isinstance(obj['label'], str) or not obj['label']:
                return False, "Invalid label"

            # Must have type (1=BASIC, 2=NORMAL, 3=PARALLEL)
            if 'type' not in obj:
                return False, "Missing type field"

            if obj['type'] not in [1, 2, 3]:
                return False, f"Invalid type: {obj['type']}"

            # is_initial should be boolean if present
            if 'is_initial' in obj and not isinstance(obj['is_initial'], bool):
                return False, "is_initial must be boolean"

            # Check for duplicate labels in context
            if context and context.sibling_states:
                if obj['label'] in context.sibling_states:
                    return False, f"Duplicate label: {obj['label']}"

            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def _validate_transition(
        self, content: str, context: Optional[StructureContext]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a transition object."""
        try:
            obj = json.loads(content)

            # Must have from and to arrays
            if 'from' not in obj or 'to' not in obj:
                return False, "Missing from/to fields"

            if not isinstance(obj['from'], list) or not isinstance(obj['to'], list):
                return False, "from/to must be arrays"

            if not obj['from'] or not obj['to']:
                return False, "Empty from/to arrays"

            # Check that states exist in context
            if context and context.existing_states:
                for state in obj['from']:
                    if state not in context.existing_states:
                        return False, f"Unknown source state: {state}"
                for state in obj['to']:
                    if state not in context.existing_states:
                        return False, f"Unknown target state: {state}"

            # Should have event
            if 'event' not in obj:
                return False, "Missing event field"

            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def _validate_guard(
        self, content: str, context: Optional[StructureContext]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a guard expression."""
        content = content.strip()

        if not content:
            return False, "Empty guard"

        # Basic syntax check - should be a condition
        # Valid patterns: "x > 0", "state == 'active'", "count < 10"
        valid_patterns = [
            r'\w+\s*[<>=!]+\s*\w+',  # comparison
            r'\w+\s*&&\s*\w+',       # AND
            r'\w+\s*\|\|\s*\w+',     # OR
            r'!\w+',                  # NOT
            r'\w+',                   # simple variable
        ]

        for pattern in valid_patterns:
            if re.match(pattern, content):
                return True, None

        # Allow any non-empty alphanumeric expression
        if re.match(r'^[\w\s<>=!&|()"\'.]+$', content):
            return True, None

        return False, f"Invalid guard expression: {content}"

    def _validate_action(
        self, content: str, context: Optional[StructureContext]
    ) -> Tuple[bool, Optional[str]]:
        """Validate an action object."""
        try:
            obj = json.loads(content)

            # Must have name field
            if 'name' not in obj:
                return False, "Missing name field"

            if not isinstance(obj['name'], str) or not obj['name']:
                return False, "Invalid action name"

            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"


def create_test_cases() -> List[Tuple[str, str, HoleType, StructureContext]]:
    """Create test cases for FIM evaluation."""
    cases = []

    # STATE hole
    cases.append((
        '{"root_state": {"label": "__root__", "type": 2, "children": [',
        ', {"label": "Off", "type": 1}]}}',
        HoleType.STATE,
        StructureContext(
            existing_states=["__root__", "Off"],
            parent_state="__root__",
            sibling_states=["Off"]
        )
    ))

    # TRANSITION hole
    cases.append((
        '{"transitions": [',
        ', {"from": ["Off"], "to": ["On"], "event": "TURN_OFF"}]}',
        HoleType.TRANSITION,
        StructureContext(
            existing_states=["On", "Off"],
            existing_events=["TURN_ON", "TURN_OFF"]
        )
    ))

    # GUARD hole
    cases.append((
        '{"from": ["Idle"], "to": ["Active"], "event": "START", "guard": {"expression": "',
        '"}}',
        HoleType.GUARD,
        StructureContext(
            existing_states=["Idle", "Active"]
        )
    ))

    # ACTION hole
    cases.append((
        '{"label": "Loading", "type": 1, "entry_actions": [',
        ']}',
        HoleType.ACTION,
        StructureContext(
            existing_states=["Loading", "Ready"]
        )
    ))

    return cases
