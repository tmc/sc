"""
Meta-Constrained Sampler with Introspection Tokens.

Injects grammar state information into the generation context,
allowing the model to "see" its constraint state.

Introspection tokens:
- SC:STATE=<state_name> - Current grammar state
- SC:VALID=<token1,token2,...> - Valid next tokens
- SC:DEPTH=<n> - Current nesting depth
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any, Tuple
from enum import Enum, auto

import mlx.core as mx
from mlx_lm import load, generate


class IntrospectionMode(Enum):
    """Modes for introspection token injection."""
    NONE = auto()           # No introspection (baseline)
    STATE_ONLY = auto()     # Only inject SC:STATE
    VALID_ONLY = auto()     # Only inject SC:VALID tokens
    FULL = auto()           # Inject all introspection info


@dataclass
class GrammarState:
    """Current state of SC grammar parsing."""
    state_name: str = "START"
    valid_tokens: Set[str] = field(default_factory=set)
    depth: int = 0
    context: str = ""  # e.g., "in_root", "in_state", "in_transition"

    def to_introspection_string(self, mode: IntrospectionMode) -> str:
        """Convert state to introspection token string."""
        if mode == IntrospectionMode.NONE:
            return ""

        parts = []

        if mode in (IntrospectionMode.STATE_ONLY, IntrospectionMode.FULL):
            parts.append(f"[SC:STATE={self.state_name}]")

        if mode in (IntrospectionMode.VALID_ONLY, IntrospectionMode.FULL):
            valid_str = ",".join(sorted(list(self.valid_tokens)[:5]))  # Limit to 5
            parts.append(f"[SC:VALID={valid_str}]")

        if mode == IntrospectionMode.FULL:
            parts.append(f"[SC:DEPTH={self.depth}]")

        return " ".join(parts)


class SCGrammarTracker:
    """Tracks SC grammar state during generation."""

    ROOT_FIELDS = {"root_state", "transitions"}
    STATE_FIELDS = {"label", "type", "children", "is_initial"}
    TRANS_FIELDS = {"from", "to", "event", "guard", "action"}

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "START"
        self.depth = 0
        self.context_stack = []
        self.current_field = None

    def get_state(self) -> GrammarState:
        valid = self._get_valid_tokens()
        context = self.context_stack[-1] if self.context_stack else "root"
        return GrammarState(
            state_name=self.state,
            valid_tokens=valid,
            depth=self.depth,
            context=context,
        )

    def _get_valid_tokens(self) -> Set[str]:
        if self.state == "START":
            return {"{"}
        elif self.state == "ROOT_OBJ":
            return {"root_state", "transitions", "}"}
        elif self.state == "IN_STATE_OBJ":
            return {"label", "type", "children", "is_initial", "}"}
        elif self.state == "IN_TRANS_OBJ":
            return {"from", "to", "event", "guard", "action", "}"}
        elif self.state == "EXPECT_VALUE":
            ctx = self.context_stack[-1] if self.context_stack else ""
            if ctx == "type":
                return {"1", "2", "3"}
            elif ctx == "is_initial":
                return {"true", "false"}
            return {"<value>"}
        elif self.state == "AFTER_VALUE":
            return {",", "}", "]"}
        return {"<any>"}


class MetaConstrainedSampler:
    """Sampler that injects introspection tokens during generation."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        mode: IntrospectionMode = IntrospectionMode.FULL,
    ):
        self.model_id = model_id
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.grammar = SCGrammarTracker()

    def load_model(self):
        if self.model is None:
            self.model, self.tokenizer = load(self.model_id)

    def generate_with_introspection(
        self,
        base_prompt: str,
        max_tokens: int = 150,
    ) -> Tuple[str, str]:
        """Generate with introspection tokens in prompt."""
        self.load_model()
        self.grammar.reset()

        # Add introspection context to prompt
        state = self.grammar.get_state()
        intro_str = state.to_introspection_string(self.mode)

        enhanced_prompt = base_prompt
        if intro_str:
            enhanced_prompt = f"{base_prompt}\n\nConstraint state: {intro_str}\nGenerate:"

        output = generate(
            self.model,
            self.tokenizer,
            prompt=enhanced_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        return output, intro_str

    def generate_baseline(
        self,
        base_prompt: str,
        max_tokens: int = 150,
    ) -> str:
        """Generate without introspection."""
        self.load_model()

        return generate(
            self.model,
            self.tokenizer,
            prompt=base_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )


def test_meta_sampler():
    print("=" * 70)
    print("Testing Meta-Constrained Sampler")
    print("=" * 70)

    sampler = MetaConstrainedSampler(mode=IntrospectionMode.FULL)
    prompt = "Generate a valid statechart JSON:"

    print("\n1. Baseline:")
    baseline = sampler.generate_baseline(prompt, max_tokens=100)
    print(f"   {baseline[:150]}...")

    print("\n2. With introspection:")
    output, intro = sampler.generate_with_introspection(prompt, max_tokens=100)
    print(f"   Intro: {intro}")
    print(f"   {output[:150]}...")

    print("=" * 70)


if __name__ == "__main__":
    test_meta_sampler()
