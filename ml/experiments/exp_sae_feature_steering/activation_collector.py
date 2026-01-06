"""
Activation Collector for SC Generation.

Collects hidden state activations during statechart JSON generation
for SAE training. Labels activations with semantic context:
- STATE: Inside state definition
- TRANSITION: Inside transition definition
- GUARD: Inside guard expression
- EVENT: Event name
- STRUCTURAL: Brackets, commas, etc.

Builds on exp_grammar_induction token collection patterns.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from collections import defaultdict

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


class SemanticContext(Enum):
    """Semantic context during SC generation."""
    ROOT = auto()           # At root level
    STATE_DEF = auto()      # Defining a state
    STATE_LABEL = auto()    # State label/name
    STATE_TYPE = auto()     # State type
    STATE_CHILDREN = auto() # State children array
    TRANSITION = auto()     # Inside transition
    TRANS_FROM = auto()     # Transition source
    TRANS_TO = auto()       # Transition target
    TRANS_EVENT = auto()    # Transition event
    TRANS_GUARD = auto()    # Transition guard
    TRANS_ACTION = auto()   # Transition action
    STRUCTURAL = auto()     # Brackets, commas
    UNKNOWN = auto()


@dataclass
class TokenActivation:
    """Activation for a single token."""
    token: str
    position: int
    context: SemanticContext
    activation: Optional[Any] = None  # mx.array or numpy
    layer: int = -1

    # Metadata
    json_path: str = ""  # e.g., "root_state.children[0].label"
    nesting_depth: int = 0

    def to_dict(self) -> Dict:
        return {
            'token': self.token,
            'position': self.position,
            'context': self.context.name,
            'layer': self.layer,
            'json_path': self.json_path,
            'nesting_depth': self.nesting_depth,
        }


@dataclass
class ActivationCache:
    """Cache of collected activations."""
    activations: List[TokenActivation] = field(default_factory=list)
    context_counts: Dict[SemanticContext, int] = field(default_factory=lambda: defaultdict(int))

    # Aggregated tensors for training
    hidden_states: Optional[Any] = None  # (N, dim)
    labels: List[SemanticContext] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.activations)

    def add(self, activation: TokenActivation):
        self.activations.append(activation)
        self.context_counts[activation.context] += 1
        self.labels.append(activation.context)

    def get_by_context(self, context: SemanticContext) -> List[TokenActivation]:
        return [a for a in self.activations if a.context == context]

    def build_tensor(self, dim: int = 256):
        """Build tensor from activations for training."""
        if not self.activations:
            return

        if HAS_MLX:
            # Collect real activations if available
            real_acts = [
                a.activation for a in self.activations
                if a.activation is not None
            ]
            if real_acts:
                self.hidden_states = mx.stack(real_acts)
            else:
                # Generate synthetic activations for testing
                self.hidden_states = mx.random.normal((len(self.activations), dim))
        else:
            import numpy as np
            self.hidden_states = np.random.randn(len(self.activations), dim).astype(np.float32)

    def summary(self) -> str:
        lines = [f"ActivationCache: {self.total} activations"]
        for ctx, count in sorted(self.context_counts.items(), key=lambda x: -x[1]):
            pct = count / self.total * 100 if self.total > 0 else 0
            lines.append(f"  {ctx.name}: {count} ({pct:.1f}%)")
        return "\n".join(lines)


@dataclass
class CollectionConfig:
    """Configuration for activation collection."""
    layers_to_collect: List[int] = field(default_factory=lambda: [-1])  # -1 = last
    include_attention: bool = False
    max_tokens: int = 10000
    hidden_dim: int = 256


class ActivationCollector:
    """
    Collects activations from SC generation.

    Labels each token with semantic context based on JSON structure.
    """

    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        self.cache = ActivationCache()

        # JSON parsing state
        self._reset_parser()

    def _reset_parser(self):
        """Reset JSON parsing state."""
        self.path_stack: List[str] = []
        self.context_stack: List[SemanticContext] = [SemanticContext.ROOT]
        self.in_string = False
        self.current_key = ""
        self.array_indices: Dict[str, int] = defaultdict(int)

    def collect(
        self,
        json_text: str,
        activations: Optional[Any] = None,
        layer: int = -1,
    ) -> ActivationCache:
        """
        Collect activations from SC JSON.

        Args:
            json_text: Statechart JSON string
            activations: Optional activation tensor (seq_len, dim)
            layer: Layer index

        Returns:
            ActivationCache with labeled activations
        """
        self._reset_parser()

        tokens = self._tokenize(json_text)

        for i, token in enumerate(tokens):
            if i >= self.config.max_tokens:
                break

            context = self._determine_context(token)

            act = None
            if activations is not None and i < len(activations):
                act = activations[i]

            token_act = TokenActivation(
                token=token,
                position=i,
                context=context,
                activation=act,
                layer=layer,
                json_path=self._current_path(),
                nesting_depth=len(self.path_stack),
            )

            self.cache.add(token_act)
            self._update_parser_state(token)

        return self.cache

    def collect_batch(
        self,
        json_texts: List[str],
        batch_activations: Optional[List[Any]] = None,
    ) -> ActivationCache:
        """Collect from multiple SC JSON strings."""
        for i, text in enumerate(json_texts):
            acts = batch_activations[i] if batch_activations else None
            self.collect(text, acts)
        return self.cache

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize JSON into semantic chunks."""
        tokens = []
        current = ""
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                current += char
                escape_next = False
                continue

            if char == '\\' and in_string:
                current += char
                escape_next = True
                continue

            if char == '"':
                if in_string:
                    current += char
                    tokens.append(current)
                    current = ""
                    in_string = False
                else:
                    if current.strip():
                        tokens.append(current.strip())
                    current = char
                    in_string = True
                continue

            if in_string:
                current += char
            elif char in '{}[]:,':
                if current.strip():
                    tokens.append(current.strip())
                tokens.append(char)
                current = ""
            elif char in ' \t\n\r':
                if current.strip():
                    tokens.append(current.strip())
                    current = ""
            else:
                current += char

        if current.strip():
            tokens.append(current.strip())

        return tokens

    def _determine_context(self, token: str) -> SemanticContext:
        """Determine semantic context for token."""
        if not self.context_stack:
            return SemanticContext.UNKNOWN

        current = self.context_stack[-1]

        # Structural tokens
        if token in '{}[]:,':
            return SemanticContext.STRUCTURAL

        # Check based on current key
        key = self.current_key.strip('"').lower()

        # State-related contexts
        if key == 'root_state' or key == 'children':
            if token.startswith('"'):
                return SemanticContext.STATE_LABEL
            return SemanticContext.STATE_DEF

        if key == 'label':
            return SemanticContext.STATE_LABEL

        if key == 'type':
            if current in (SemanticContext.STATE_DEF, SemanticContext.ROOT):
                return SemanticContext.STATE_TYPE
            return SemanticContext.UNKNOWN

        if key == 'is_initial':
            return SemanticContext.STATE_DEF

        # Transition-related contexts
        if key == 'transitions':
            return SemanticContext.TRANSITION

        if key == 'from':
            return SemanticContext.TRANS_FROM

        if key == 'to':
            return SemanticContext.TRANS_TO

        if key == 'event':
            return SemanticContext.TRANS_EVENT

        if key == 'guard':
            return SemanticContext.TRANS_GUARD

        if key == 'action' or key == 'actions':
            return SemanticContext.TRANS_ACTION

        # Default based on stack
        return current

    def _update_parser_state(self, token: str):
        """Update parser state after processing token."""
        if token == '{':
            self.path_stack.append(self.current_key or "object")
            if 'transition' in self.current_key.lower():
                self.context_stack.append(SemanticContext.TRANSITION)
            elif 'state' in self.current_key.lower() or self.current_key == '':
                self.context_stack.append(SemanticContext.STATE_DEF)
            else:
                self.context_stack.append(self.context_stack[-1] if self.context_stack else SemanticContext.ROOT)
            self.current_key = ""

        elif token == '}':
            if self.path_stack:
                self.path_stack.pop()
            if self.context_stack:
                self.context_stack.pop()

        elif token == '[':
            path = self.current_key or "array"
            self.path_stack.append(path)
            self.array_indices[self._current_path()] = 0
            if 'children' in path.lower():
                self.context_stack.append(SemanticContext.STATE_CHILDREN)
            elif 'transition' in path.lower():
                self.context_stack.append(SemanticContext.TRANSITION)
            else:
                self.context_stack.append(self.context_stack[-1] if self.context_stack else SemanticContext.ROOT)
            self.current_key = ""

        elif token == ']':
            if self.path_stack:
                self.path_stack.pop()
            if self.context_stack:
                self.context_stack.pop()

        elif token == ',':
            path = self._current_path()
            if path in self.array_indices:
                self.array_indices[path] += 1

        elif token == ':':
            pass

        elif token.startswith('"') and token.endswith('"'):
            # Could be key or value
            if not self.current_key or self.current_key == ':':
                self.current_key = token
            else:
                self.current_key = ""

        else:
            # Value token
            self.current_key = ""

    def _current_path(self) -> str:
        """Get current JSON path."""
        return ".".join(self.path_stack)


def test_collector():
    """Test activation collector."""
    print("=" * 60)
    print("Testing Activation Collector")
    print("=" * 60)

    collector = ActivationCollector()

    # Sample statechart JSON
    sc_json = '''{
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": true},
                {"label": "On", "type": 1}
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
            {"from": ["On"], "to": ["Off"], "event": "TURN_OFF", "guard": "canTurnOff"}
        ]
    }'''

    print("\n1. Collecting activations from SC JSON:")
    cache = collector.collect(sc_json)

    print(f"\n{cache.summary()}")

    # Show sample activations
    print("\n2. Sample activations by context:")
    for ctx in [SemanticContext.STATE_LABEL, SemanticContext.TRANS_EVENT, SemanticContext.TRANS_GUARD]:
        acts = cache.get_by_context(ctx)
        if acts:
            tokens = [a.token for a in acts[:5]]
            print(f"  {ctx.name}: {tokens}")

    # Build tensor
    print("\n3. Building activation tensor:")
    cache.build_tensor(dim=64)
    if HAS_MLX:
        print(f"  Shape: {cache.hidden_states.shape}")
    else:
        print(f"  Shape: {cache.hidden_states.shape}")

    # Test batch collection
    print("\n4. Batch collection:")
    collector2 = ActivationCollector()
    jsons = [
        '{"root_state": {"label": "__root__", "children": [{"label": "A"}]}, "transitions": []}',
        '{"root_state": {"label": "__root__", "children": [{"label": "B"}, {"label": "C"}]}, "transitions": [{"from": ["B"], "to": ["C"], "event": "GO"}]}',
    ]
    cache2 = collector2.collect_batch(jsons)
    print(f"  Collected {cache2.total} activations from {len(jsons)} JSONs")

    print("\n" + "=" * 60)
    print("Activation collector tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_collector()
