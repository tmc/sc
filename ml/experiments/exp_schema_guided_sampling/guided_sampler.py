#!/usr/bin/env python3
"""
Guided Sampler: Use statechart to guide token selection during generation.

Implements constrained decoding by:
1. Tracking JSON state with statechart
2. Building token mask based on valid next tokens
3. Applying mask to logits before sampling
4. Force-closing structures when limits reached
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False

from .json_schema_statechart import (
    JSONSchemaStatechart,
    TokenType,
    JSONState,
)


@dataclass
class TokenMask:
    """Mask for valid/invalid tokens."""
    # Token IDs that are allowed
    allowed_ids: Set[int]
    # Token IDs that are forbidden
    forbidden_ids: Set[int]
    # Bias to add to allowed tokens (positive = more likely)
    bias: float = 0.0


class GuidedSampler:
    """
    Sampler that uses statechart to guide token selection.

    Works by:
    1. Maintaining JSON parsing state
    2. Computing valid token types for current state
    3. Building mask over vocabulary
    4. Applying mask to logits before sampling
    """

    def __init__(
        self,
        tokenizer: Any,
        max_depth: int = 8,
        max_states: int = 15,
        force_close_depth: int = 6,
    ):
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.max_states = max_states
        self.force_close_depth = force_close_depth

        # Initialize statechart
        self.statechart = JSONSchemaStatechart(
            max_depth=max_depth,
            max_states=max_states,
        )

        # Build token type mapping
        self._build_token_maps()

    def _build_token_maps(self):
        """Build mapping from token types to token IDs."""
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        # Maps token type -> set of token IDs
        self.type_to_ids: Dict[TokenType, Set[int]] = {t: set() for t in TokenType}

        # Special token patterns
        brace_tokens = {"{", " {", "{\n", "\n{"}
        rbrace_tokens = {"}", " }", "}\n", "\n}", "}}", "}}}", "}]"}
        bracket_tokens = {"[", " [", "[\n", "\n["}
        rbracket_tokens = {"]", " ]", "]\n", "\n]", "]]", "}]"}
        colon_tokens = {":", ": ", " :"}
        comma_tokens = {",", ", ", " ,", ",\n"}
        quote_tokens = {'"', ' "', '"\n'}

        for token_str, token_id in vocab.items():
            # Classify each vocabulary token
            token_clean = token_str.strip()

            if token_clean == "{" or token_str in brace_tokens:
                self.type_to_ids[TokenType.LBRACE].add(token_id)
            if token_clean == "}" or any(c == "}" for c in token_str):
                self.type_to_ids[TokenType.RBRACE].add(token_id)
            if token_clean == "[" or token_str in bracket_tokens:
                self.type_to_ids[TokenType.LBRACKET].add(token_id)
            if token_clean == "]" or any(c == "]" for c in token_str):
                self.type_to_ids[TokenType.RBRACKET].add(token_id)
            if ":" in token_str:
                self.type_to_ids[TokenType.COLON].add(token_id)
            if "," in token_str:
                self.type_to_ids[TokenType.COMMA].add(token_id)
            if '"' in token_str:
                self.type_to_ids[TokenType.QUOTE].add(token_id)
            if token_clean in ("true", "false"):
                self.type_to_ids[TokenType.TRUE if token_clean == "true" else TokenType.FALSE].add(token_id)
            if token_clean == "null":
                self.type_to_ids[TokenType.NULL].add(token_id)
            if token_str.strip() == "" or token_str in (" ", "\n", "\t", "  "):
                self.type_to_ids[TokenType.WHITESPACE].add(token_id)

            # Numbers
            try:
                float(token_clean)
                self.type_to_ids[TokenType.NUMBER].add(token_id)
            except ValueError:
                pass

            # String content (alphanumeric, common words)
            if token_clean.isalnum() or token_clean.replace("_", "").isalnum():
                self.type_to_ids[TokenType.STRING].add(token_id)

        # Add common JSON key tokens as strings
        common_keys = ["label", "type", "children", "root_state", "is_initial",
                       "transitions", "event", "from", "to", "Idle", "Active",
                       "Running", "Stopped", "On", "Off", "Start", "End"]
        for key in common_keys:
            for token_str, token_id in vocab.items():
                if key.lower() in token_str.lower():
                    self.type_to_ids[TokenType.STRING].add(token_id)

    def reset(self):
        """Reset statechart to initial state."""
        self.statechart.reset()

    def get_token_mask(self) -> TokenMask:
        """Get mask for valid tokens based on current state."""
        valid_types = self.statechart.get_valid_token_types()

        allowed = set()
        for token_type in valid_types:
            allowed.update(self.type_to_ids.get(token_type, set()))

        # If should force close, only allow closing tokens
        if self.statechart.should_force_close():
            closing_ids = set()
            closing_ids.update(self.type_to_ids[TokenType.RBRACE])
            closing_ids.update(self.type_to_ids[TokenType.RBRACKET])
            closing_ids.update(self.type_to_ids[TokenType.QUOTE])
            return TokenMask(allowed_ids=closing_ids, forbidden_ids=set(), bias=10.0)

        return TokenMask(allowed_ids=allowed, forbidden_ids=set())

    def apply_mask(self, logits: Any, mask: TokenMask) -> Any:
        """Apply token mask to logits."""
        if not MLX_AVAILABLE:
            return logits

        # Create mask array - start with zeros (no penalty)
        vocab_size = logits.shape[-1]

        # Convert allowed_ids to list and create indices
        allowed_list = [tid for tid in mask.allowed_ids if tid < vocab_size]

        if not allowed_list:
            # No mask to apply
            return logits

        # Create penalty mask: -inf everywhere except allowed tokens
        # Use numpy for efficient mask creation, then convert to mlx
        mask_np = np.full(vocab_size, -100.0)  # Large negative penalty
        for tid in allowed_list:
            mask_np[tid] = mask.bias  # No penalty + optional bias for allowed

        mask_array = mx.array(mask_np)

        # Apply mask (additive)
        return logits + mask_array

    def process_generated_token(self, token_id: int) -> bool:
        """
        Process a generated token and update statechart.

        Returns True if token was valid, False otherwise.
        """
        token_str = self.tokenizer.decode([token_id])

        # Process each character
        for char in token_str:
            if char.strip() == "":
                continue
            if not self.statechart.process_token(char):
                return False

        return True

    def is_complete(self) -> bool:
        """Check if generation is complete."""
        return self.statechart.is_complete()

    def is_error(self) -> bool:
        """Check if in error state."""
        return self.statechart.is_error()

    def get_closing_tokens(self) -> List[int]:
        """Get token IDs needed to close current structure."""
        closing_seq = self.statechart.get_closing_sequence()
        if not closing_seq:
            return []

        # Try to find tokens for closing sequence
        token_ids = []
        for char in closing_seq:
            # Find token ID for this character
            char_tokens = self.tokenizer.encode(char)
            if char_tokens:
                token_ids.extend(char_tokens)

        return token_ids


class GuidedGenerator:
    """
    Generator that uses guided sampling for constrained decoding.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_depth: int = 8,
        max_states: int = 15,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = GuidedSampler(
            tokenizer,
            max_depth=max_depth,
            max_states=max_states,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3,
        use_guidance: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with optional schema guidance.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_guidance: Whether to use statechart guidance

        Returns:
            (generated_text, metadata)
        """
        if not MLX_AVAILABLE:
            return "[MOCK] Guided generation", {"mock": True}

        from mlx_lm.sample_utils import make_sampler

        # Reset statechart
        self.sampler.reset()

        # Process prompt through statechart to establish initial state
        self._process_prompt(prompt)

        # Encode prompt
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        base_sampler = make_sampler(temp=temperature)

        generated_tokens = []
        metadata = {
            "use_guidance": use_guidance,
            "forced_closes": 0,
            "mask_applications": 0,
        }

        for _ in range(max_tokens):
            # Forward pass
            logits = self.model(tokens)
            next_logits = logits[:, -1, :]

            if use_guidance:
                # Get and apply mask
                mask = self.sampler.get_token_mask()
                if mask.allowed_ids:
                    next_logits = self.sampler.apply_mask(next_logits, mask)
                    metadata["mask_applications"] += 1

                # Check if should force close
                if self.sampler.statechart.should_force_close():
                    # Build proper closing sequence based on actual brace counts
                    if generated_tokens:
                        current_output = self.tokenizer.decode(generated_tokens)
                        full_text = prompt + current_output

                        # Strip trailing comma
                        if current_output.rstrip().endswith(','):
                            # Remove trailing comma tokens
                            while generated_tokens:
                                last = self.tokenizer.decode([generated_tokens[-1]])
                                if ',' in last:
                                    generated_tokens.pop()
                                    break
                                elif last.strip() == '':
                                    generated_tokens.pop()
                                else:
                                    break
                            current_output = self.tokenizer.decode(generated_tokens)
                            full_text = prompt + current_output

                        # Count actual braces to determine what's needed
                        open_braces = full_text.count('{') - full_text.count('}')
                        open_brackets = full_text.count('[') - full_text.count(']')

                        # Check for unclosed string
                        quote_count = full_text.count('"')
                        if quote_count % 2 == 1:
                            closing_seq = '"'
                        else:
                            closing_seq = ''

                        # Add closing braces/brackets in correct order (brackets first for arrays)
                        # Typical structure: {...[{...}]}  so close: }]}
                        closing_seq += '}' * max(0, open_braces - open_brackets)  # Close inner objects
                        closing_seq += ']' * open_brackets  # Close arrays
                        closing_seq += '}' * min(open_braces, open_brackets)  # Close outer objects

                        # Simpler: just close everything
                        closing_seq = ''
                        if quote_count % 2 == 1:
                            closing_seq += '"'
                        closing_seq += '}' * open_braces + ']' * open_brackets

                        if closing_seq:
                            closing_tokens = self.tokenizer.encode(closing_seq)
                            generated_tokens.extend(closing_tokens)
                            metadata["forced_closes"] += 1
                        break
                    else:
                        break

            # Sample next token
            next_token = base_sampler(next_logits)
            token_id = next_token.item()

            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id') and token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)

            # Update statechart
            if use_guidance:
                self.sampler.process_generated_token(token_id)

                # Check if complete
                if self.sampler.is_complete():
                    break

            # Append to sequence
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

        # Decode
        output = self.tokenizer.decode(generated_tokens)

        metadata["tokens_generated"] = len(generated_tokens)
        metadata["final_depth"] = self.sampler.statechart.state.depth
        metadata["state_count"] = self.sampler.statechart.state.state_count

        return output, metadata

    def _process_prompt(self, prompt: str):
        """Process prompt through statechart to establish state."""
        for char in prompt:
            self.sampler.statechart.process_token(char)


def demo():
    """Demo guided sampling."""
    print("=" * 60)
    print("GUIDED SAMPLER DEMO")
    print("=" * 60)

    try:
        from mlx_lm import load

        model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")

        generator = GuidedGenerator(model, tokenizer, max_depth=6, max_states=10)

        prompt = '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label":'

        print(f"\nPrompt: {prompt}")

        # Unguided
        print("\n--- Unguided ---")
        output, meta = generator.generate(prompt, max_tokens=100, use_guidance=False)
        print(f"Output: {output[:200]}")
        print(f"Meta: {meta}")

        # Guided
        print("\n--- Guided ---")
        output, meta = generator.generate(prompt, max_tokens=100, use_guidance=True)
        print(f"Output: {output[:200]}")
        print(f"Meta: {meta}")

    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo()
