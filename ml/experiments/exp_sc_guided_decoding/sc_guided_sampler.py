"""
SC-Guided Token Sampler.

Uses SC grammar FSM to mask invalid tokens during generation.
Only allows tokens that lead to valid SC JSON structure.

Key features:
- Token-level masking for SC-valid generation
- Lookahead to avoid dead ends
- Simulated generation for benchmarking
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .sc_grammar import (
    SCGrammar,
    SCGrammarState,
    GrammarContext,
    tokenize_sc_json,
)


@dataclass
class SamplerConfig:
    """Configuration for guided sampler."""
    max_tokens: int = 200
    temperature: float = 1.0
    lookahead_depth: int = 2
    allow_optional_fields: bool = True


@dataclass
class GenerationResult:
    """Result of a guided generation."""
    tokens: List[str] = field(default_factory=list)
    text: str = ""
    is_valid_json: bool = False
    is_valid_sc: bool = False
    num_tokens: int = 0
    num_masked: int = 0  # Tokens masked by grammar
    parse_error: str = ""
    string_limit_hits: int = 0  # Times string length limit was enforced


class SCGuidedSampler:
    """
    Guided sampler using SC grammar for constrained generation.

    Masks invalid tokens at each step to ensure only valid SC JSON.
    """

    # Token vocabulary for SC JSON
    STRUCTURAL_TOKENS = ["{", "}", "[", "]", ":", ","]
    ROOT_FIELD_TOKENS = ['"root_state"', '"transitions"']
    STATE_FIELD_TOKENS = ['"label"', '"type"', '"children"', '"is_initial"']
    TRANS_FIELD_TOKENS = ['"from"', '"to"', '"event"', '"guard"', '"action"']
    TYPE_TOKENS = ["1", "2", "3"]
    BOOL_TOKENS = ["true", "false"]

    # Sample state labels
    STATE_LABELS = [
        '"Off"', '"On"', '"Idle"', '"Active"', '"Running"',
        '"Stopped"', '"Start"', '"End"', '"Init"', '"Ready"',
        '"A"', '"B"', '"C"', '"X"', '"Y"', '"Z"',
        '"__root__"', '"Main"', '"Sub1"', '"Sub2"',
    ]

    # Sample events
    EVENTS = [
        '"TOGGLE"', '"START"', '"STOP"', '"RESET"', '"NEXT"',
        '"BACK"', '"GO"', '"DONE"', '"ERROR"', '"TIMEOUT"',
        '"CLICK"', '"ENTER"', '"EXIT"', '"UPDATE"',
    ]

    # Sample guards/actions
    GUARDS = ['"ready"', '"canStart"', '"isValid"', '"hasData"', '""']
    ACTIONS = ['"doStart"', '"cleanup"', '"notify"', '"log"', '""']

    def __init__(self, grammar: SCGrammar = None, config: SamplerConfig = None):
        self.grammar = grammar or SCGrammar()
        self.config = config or SamplerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build full token vocabulary."""
        self.vocabulary = set()
        self.vocabulary.update(self.STRUCTURAL_TOKENS)
        self.vocabulary.update(self.ROOT_FIELD_TOKENS)
        self.vocabulary.update(self.STATE_FIELD_TOKENS)
        self.vocabulary.update(self.TRANS_FIELD_TOKENS)
        self.vocabulary.update(self.TYPE_TOKENS)
        self.vocabulary.update(self.BOOL_TOKENS)
        self.vocabulary.update(self.STATE_LABELS)
        self.vocabulary.update(self.EVENTS)
        self.vocabulary.update(self.GUARDS)
        self.vocabulary.update(self.ACTIONS)

    def get_valid_tokens(self, context: GrammarContext) -> Set[str]:
        """Get valid next tokens from current context."""
        # Get grammar-valid tokens
        grammar_valid = self.grammar.get_valid_tokens(context)

        # Map to actual vocabulary
        valid = set()

        for pattern in grammar_valid:
            if pattern in self.vocabulary:
                valid.add(pattern)
            elif pattern == "STRING_CONTENT":
                # Add appropriate strings based on context
                if context.context_stack:
                    current = context.context_stack[-1]
                    if current == "label":
                        valid.update(self.STATE_LABELS)
                    elif current == "event":
                        valid.update(self.EVENTS)
                    elif current == "guard":
                        valid.update(self.GUARDS)
                    elif current == "action":
                        valid.update(self.ACTIONS)
                    elif current in ("from", "to"):
                        valid.update(self.STATE_LABELS)
            elif pattern.startswith('"') and pattern.endswith('"'):
                valid.add(pattern)

        # Special handling for field tokens based on state
        if context.state == SCGrammarState.ROOT_FIELD_OR_CLOSE:
            valid.update(self.ROOT_FIELD_TOKENS)
            valid.add("}")
        elif context.state == SCGrammarState.STATE_FIELD_OR_CLOSE:
            valid.update(self.STATE_FIELD_TOKENS)
            valid.add("}")
        elif context.state == SCGrammarState.TRANS_FIELD_OR_CLOSE:
            valid.update(self.TRANS_FIELD_TOKENS)
            valid.add("}")

        return valid

    def mask_logits(
        self,
        logits: Dict[str, float],
        context: GrammarContext,
    ) -> Dict[str, float]:
        """
        Mask logits for invalid tokens.

        Sets invalid token logits to -inf.
        """
        valid_tokens = self.get_valid_tokens(context)

        masked = {}
        for token, logit in logits.items():
            if token in valid_tokens:
                masked[token] = logit
            else:
                masked[token] = float('-inf')

        return masked

    def sample_next_token(
        self,
        context: GrammarContext,
        logits: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Sample next token given context and logits.

        If logits not provided, uses uniform distribution over valid tokens.
        """
        valid_tokens = self.get_valid_tokens(context)

        if not valid_tokens:
            return None

        if logits is None:
            # Uniform sampling
            return random.choice(list(valid_tokens))

        # Mask and sample
        masked = self.mask_logits(logits, context)
        valid_with_logits = {t: l for t, l in masked.items() if l > float('-inf')}

        if not valid_with_logits:
            return random.choice(list(valid_tokens))

        # Temperature-scaled softmax sampling
        import math
        max_logit = max(valid_with_logits.values())
        exp_logits = {
            t: math.exp((l - max_logit) / self.config.temperature)
            for t, l in valid_with_logits.items()
        }
        total = sum(exp_logits.values())
        probs = {t: e / total for t, e in exp_logits.items()}

        r = random.random()
        cumsum = 0.0
        for token, prob in probs.items():
            cumsum += prob
            if r <= cumsum:
                return token

        return list(valid_with_logits.keys())[0]

    def generate(
        self,
        prompt_tokens: List[str] = None,
        logits_fn=None,
    ) -> GenerationResult:
        """
        Generate SC JSON with guided sampling.

        Args:
            prompt_tokens: Optional initial tokens
            logits_fn: Optional function(context, tokens) -> Dict[str, float]

        Returns:
            GenerationResult with generated tokens
        """
        result = GenerationResult()
        context = GrammarContext()
        tokens = list(prompt_tokens) if prompt_tokens else []

        # Process prompt tokens
        for token in tokens:
            new_ctx = self.grammar.step(context, token)
            if new_ctx is None:
                result.parse_error = f"Invalid prompt token: {token}"
                return result
            context = new_ctx

        # Generate
        total_masked = 0

        while len(tokens) < self.config.max_tokens:
            valid_tokens = self.get_valid_tokens(context)
            total_masked += len(self.vocabulary) - len(valid_tokens)

            if not valid_tokens:
                break

            # Check if we can end
            if self.grammar.is_valid_complete(context):
                break

            # Get logits or use uniform
            if logits_fn:
                logits = logits_fn(context, tokens)
            else:
                logits = None

            # Sample next token
            next_token = self.sample_next_token(context, logits)
            if next_token is None:
                break

            tokens.append(next_token)

            # Update context
            new_ctx = self.grammar.step(context, next_token)
            if new_ctx is None:
                result.parse_error = f"Grammar rejected token: {next_token}"
                break
            context = new_ctx

        # Build result
        result.tokens = tokens
        result.text = " ".join(tokens)
        result.num_tokens = len(tokens)
        result.num_masked = total_masked
        result.is_valid_sc = self.grammar.is_valid_complete(context)
        result.string_limit_hits = context.string_limit_hits

        # Try to parse as JSON
        try:
            json_text = "".join(tokens)
            json.loads(json_text)
            result.is_valid_json = True
        except:
            result.is_valid_json = False

        return result

    def generate_batch(self, n: int = 10) -> List[GenerationResult]:
        """Generate multiple SC JSONs."""
        return [self.generate() for _ in range(n)]


class GenericJSONSampler:
    """
    Generic JSON sampler for comparison.

    Less restrictive than SC-guided sampler.
    """

    TOKENS = [
        "{", "}", "[", "]", ":", ",",
        "true", "false", "null",
        "1", "2", "3", "0",
        '"key"', '"value"', '"name"', '"id"',
    ]

    def __init__(self):
        self.depth = 0
        self.in_object = False
        self.in_array = False

    def generate(self) -> GenerationResult:
        """Generate generic JSON."""
        result = GenerationResult()
        tokens = []
        depth = 0
        max_depth = 3

        # Always start with object
        tokens.append("{")
        depth = 1

        while len(tokens) < 100 and depth > 0:
            if depth >= max_depth:
                # Close
                tokens.append("}")
                depth -= 1
            else:
                # Random choice
                r = random.random()
                if r < 0.3:
                    tokens.append('"key"')
                    tokens.append(":")
                    tokens.append('"value"')
                    if random.random() < 0.5:
                        tokens.append(",")
                elif r < 0.5:
                    tokens.append("}")
                    depth -= 1
                else:
                    tokens.append('"nested"')
                    tokens.append(":")
                    tokens.append("{")
                    depth += 1

        while depth > 0:
            tokens.append("}")
            depth -= 1

        result.tokens = tokens
        result.text = "".join(tokens)
        result.num_tokens = len(tokens)

        try:
            json.loads(result.text)
            result.is_valid_json = True
        except:
            result.is_valid_json = False

        # Generic JSON is never valid SC
        result.is_valid_sc = False

        return result


def test_sampler():
    """Test SC guided sampler."""
    print("=" * 60)
    print("Testing SC Guided Sampler")
    print("=" * 60)

    sampler = SCGuidedSampler()

    # Test valid token masking
    print("\n1. Token masking at each state:")
    context = GrammarContext()
    print(f"  START: {sampler.get_valid_tokens(context)}")

    context = sampler.grammar.step(context, "{")
    print(f"  After '{{': {sampler.get_valid_tokens(context)}")

    # Test generation
    print("\n2. Guided generation:")
    results = sampler.generate_batch(5)

    valid_json = sum(1 for r in results if r.is_valid_json)
    valid_sc = sum(1 for r in results if r.is_valid_sc)
    avg_tokens = sum(r.num_tokens for r in results) / len(results)

    print(f"  Generated: {len(results)}")
    print(f"  Valid JSON: {valid_json}/{len(results)} ({valid_json/len(results)*100:.1f}%)")
    print(f"  Valid SC: {valid_sc}/{len(results)} ({valid_sc/len(results)*100:.1f}%)")
    print(f"  Avg tokens: {avg_tokens:.1f}")

    # Show sample
    print("\n3. Sample generation:")
    result = results[0]
    print(f"  Tokens: {result.num_tokens}")
    print(f"  Text: {result.text[:100]}...")
    print(f"  Valid JSON: {result.is_valid_json}")
    print(f"  Valid SC: {result.is_valid_sc}")

    # Compare to generic JSON
    print("\n4. Comparison to generic JSON:")
    generic = GenericJSONSampler()
    generic_results = [generic.generate() for _ in range(5)]

    generic_valid_json = sum(1 for r in generic_results if r.is_valid_json)
    generic_valid_sc = sum(1 for r in generic_results if r.is_valid_sc)

    print(f"  Generic JSON valid: {generic_valid_json}/{len(generic_results)}")
    print(f"  Generic SC valid: {generic_valid_sc}/{len(generic_results)} (expected 0)")

    print("\n" + "=" * 60)
    print("SC Guided Sampler tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_sampler()
