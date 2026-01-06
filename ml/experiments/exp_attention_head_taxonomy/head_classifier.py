"""
Attention Head Taxonomy: Classify heads by function during SC generation.

CATEGORIES:
- state-tracking: Attends to state labels, maintains state memory
- transition-following: Connects from/to states, tracks transition flow
- hierarchy-aware: Tracks parent-child nesting, compound states
- syntax-checking: Focuses on JSON structure, brackets, quotes

APPROACH:
1. Generate statecharts with attention caching
2. Analyze attention patterns for each head
3. Classify based on what tokens each head attends to
4. Build taxonomy of head functions
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    CacheConfig,
    MLUX_AVAILABLE,
)


class HeadFunction(Enum):
    """Categories of attention head functions."""
    STATE_TRACKING = "state_tracking"
    TRANSITION_FOLLOWING = "transition_following"
    HIERARCHY_AWARE = "hierarchy_aware"
    SYNTAX_CHECKING = "syntax_checking"
    COPY_SUPPRESS = "copy_suppress"
    UNKNOWN = "unknown"


@dataclass
class HeadProfile:
    """Profile of an attention head's behavior."""
    layer: int
    head: int
    function: HeadFunction
    confidence: float
    patterns: Dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"L{self.layer}H{self.head}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "head": self.head,
            "function": self.function.value,
            "confidence": self.confidence,
            "patterns": self.patterns,
        }


@dataclass
class TaxonomyResult:
    """Result of head taxonomy analysis."""
    heads: List[HeadProfile]
    function_counts: Dict[HeadFunction, int]
    layer_distribution: Dict[int, Dict[HeadFunction, int]]

    def get_heads_by_function(self, function: HeadFunction) -> List[HeadProfile]:
        return [h for h in self.heads if h.function == function]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_heads": len(self.heads),
            "function_counts": {f.value: c for f, c in self.function_counts.items()},
            "heads": [h.to_dict() for h in self.heads],
        }


class TokenClassifier:
    """Classify tokens by their role in statechart generation."""

    # Patterns for token classification
    STATE_PATTERNS = [
        r'"label":\s*"',  # State label
        r'"from":\s*\[',  # Transition from
        r'"to":\s*\[',    # Transition to
    ]

    TRANSITION_PATTERNS = [
        r'"event":\s*"',
        r'"guard":\s*',
        r'"action":\s*',
    ]

    HIERARCHY_PATTERNS = [
        r'"children":\s*\[',
        r'"type":\s*2',  # Compound state
        r'"parent":\s*',
    ]

    SYNTAX_PATTERNS = [
        r'[\{\}\[\]]',  # Brackets
        r'[:,]',        # JSON delimiters
    ]

    def classify_token(self, token: str, context: str) -> str:
        """Classify a token based on its context."""
        # Check each pattern category
        for pattern in self.STATE_PATTERNS:
            if re.search(pattern, context):
                return "state"

        for pattern in self.TRANSITION_PATTERNS:
            if re.search(pattern, context):
                return "transition"

        for pattern in self.HIERARCHY_PATTERNS:
            if re.search(pattern, context):
                return "hierarchy"

        for pattern in self.SYNTAX_PATTERNS:
            if re.search(pattern, token):
                return "syntax"

        return "other"

    def classify_positions(
        self,
        tokens: List[str],
    ) -> Dict[int, str]:
        """Classify each token position."""
        classifications = {}
        context = ""

        for i, token in enumerate(tokens):
            context += token
            classifications[i] = self.classify_token(token, context)

        return classifications


class HeadAnalyzer:
    """Analyze attention patterns of individual heads."""

    def __init__(self, token_classifier: Optional[TokenClassifier] = None):
        self.token_classifier = token_classifier or TokenClassifier()

    def analyze_attention(
        self,
        attention: Any,  # [seq, seq] attention matrix
        tokens: List[str],
    ) -> Dict[str, float]:
        """Analyze what token types a head attends to."""
        # Classify token positions
        classifications = self.token_classifier.classify_positions(tokens)

        # Compute attention distribution over token types
        type_attention = defaultdict(float)

        if not MLX_AVAILABLE or attention is None:
            # Mock analysis
            return {
                "state": 0.3,
                "transition": 0.2,
                "hierarchy": 0.1,
                "syntax": 0.3,
                "other": 0.1,
            }

        # Convert to numpy-like for analysis
        attn_array = attention.tolist() if hasattr(attention, 'tolist') else attention

        n_tokens = len(tokens)
        for i in range(n_tokens):
            for j in range(n_tokens):
                if i < len(attn_array) and j < len(attn_array[i]):
                    weight = attn_array[i][j]
                    token_type = classifications.get(j, "other")
                    type_attention[token_type] += weight

        # Normalize
        total = sum(type_attention.values())
        if total > 0:
            for k in type_attention:
                type_attention[k] /= total

        return dict(type_attention)

    def classify_head(
        self,
        attention_patterns: Dict[str, float],
    ) -> Tuple[HeadFunction, float]:
        """Classify head function based on attention patterns."""
        scores = {
            HeadFunction.STATE_TRACKING: attention_patterns.get("state", 0),
            HeadFunction.TRANSITION_FOLLOWING: attention_patterns.get("transition", 0),
            HeadFunction.HIERARCHY_AWARE: attention_patterns.get("hierarchy", 0),
            HeadFunction.SYNTAX_CHECKING: attention_patterns.get("syntax", 0),
        }

        # Find dominant function
        if not scores:
            return HeadFunction.UNKNOWN, 0.0

        best_function = max(scores, key=scores.get)
        confidence = scores[best_function]

        # Require minimum confidence
        if confidence < 0.25:
            return HeadFunction.UNKNOWN, confidence

        return best_function, confidence


class HeadTaxonomy:
    """
    Build taxonomy of attention heads by function.
    """

    def __init__(
        self,
        model: Optional[HookedModelWrapper] = None,
        n_layers: int = 24,
        n_heads: int = 16,
        verbose: bool = True,
    ):
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.verbose = verbose

        self.analyzer = HeadAnalyzer()

    def analyze_generation(
        self,
        prompt: str,
        output: str,
        attention_cache: Dict[str, Any],
    ) -> List[HeadProfile]:
        """Analyze heads from a single generation."""
        heads = []

        # Tokenize output for analysis
        tokens = list(output)  # Simple character-level for now

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                cache_key = f"model.layers.{layer}.self_attn.head_{head}"

                attention = attention_cache.get(cache_key)

                # Analyze attention patterns
                patterns = self.analyzer.analyze_attention(attention, tokens)

                # Classify head
                function, confidence = self.analyzer.classify_head(patterns)

                profile = HeadProfile(
                    layer=layer,
                    head=head,
                    function=function,
                    confidence=confidence,
                    patterns=patterns,
                )
                heads.append(profile)

        return heads

    def build_taxonomy(
        self,
        prompts: List[str],
    ) -> TaxonomyResult:
        """Build taxonomy from multiple generations."""
        # Aggregate head profiles
        head_functions: Dict[str, List[HeadFunction]] = defaultdict(list)
        head_confidences: Dict[str, List[float]] = defaultdict(list)
        head_patterns: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"Analyzing prompt {i+1}/{len(prompts)}...")

            if self.model is not None and self.model.has_interpretability:
                output, cache = self.model.generate_with_cache(
                    prompt,
                    cache_config=CacheConfig(
                        hooks=[f"model.layers.*.self_attn.head_*"],
                    ),
                )
            else:
                # Mock for testing
                output = self._mock_generate(prompt)
                cache = {}

            heads = self.analyze_generation(prompt, output, cache)

            for head in heads:
                head_id = head.id
                head_functions[head_id].append(head.function)
                head_confidences[head_id].append(head.confidence)
                head_patterns[head_id].append(head.patterns)

        # Aggregate into final profiles
        final_heads = []
        function_counts = defaultdict(int)
        layer_dist = defaultdict(lambda: defaultdict(int))

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_id = f"L{layer}H{head}"

                if head_id in head_functions:
                    functions = head_functions[head_id]
                    # Majority vote
                    from collections import Counter
                    function = Counter(functions).most_common(1)[0][0]

                    confidences = head_confidences[head_id]
                    avg_confidence = sum(confidences) / len(confidences)

                    # Average patterns
                    patterns_list = head_patterns[head_id]
                    avg_patterns = {}
                    if patterns_list:
                        all_keys = set()
                        for p in patterns_list:
                            all_keys.update(p.keys())
                        for k in all_keys:
                            values = [p.get(k, 0) for p in patterns_list]
                            avg_patterns[k] = sum(values) / len(values)
                else:
                    # Generate mock profile
                    function, avg_confidence, avg_patterns = self._mock_profile(layer, head)

                profile = HeadProfile(
                    layer=layer,
                    head=head,
                    function=function,
                    confidence=avg_confidence,
                    patterns=avg_patterns,
                )
                final_heads.append(profile)
                function_counts[function] += 1
                layer_dist[layer][function] += 1

        return TaxonomyResult(
            heads=final_heads,
            function_counts=dict(function_counts),
            layer_distribution=dict(layer_dist),
        )

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        return json.dumps({
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "s0", "type": 1, "is_initial": True},
                    {"label": "s1", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["s0"], "to": ["s1"], "event": "E1"},
            ]
        })

    def _mock_profile(
        self,
        layer: int,
        head: int,
    ) -> Tuple[HeadFunction, float, Dict[str, float]]:
        """Generate mock profile based on layer."""
        import random
        random.seed(layer * 100 + head)

        # Layer-based function distribution (based on circuit findings)
        if layer < 8:
            # Early layers: syntax and structure
            functions = [HeadFunction.SYNTAX_CHECKING] * 3 + [HeadFunction.HIERARCHY_AWARE] * 2 + [HeadFunction.STATE_TRACKING]
        elif layer < 16:
            # Middle layers: state tracking and transitions
            functions = [HeadFunction.STATE_TRACKING] * 3 + [HeadFunction.TRANSITION_FOLLOWING] * 2 + [HeadFunction.HIERARCHY_AWARE]
        else:
            # Late layers: transitions and output
            functions = [HeadFunction.TRANSITION_FOLLOWING] * 3 + [HeadFunction.STATE_TRACKING] * 2 + [HeadFunction.SYNTAX_CHECKING]

        function = random.choice(functions)
        confidence = random.uniform(0.4, 0.9)

        # Generate patterns matching function
        patterns = {
            "state": 0.1,
            "transition": 0.1,
            "hierarchy": 0.1,
            "syntax": 0.1,
            "other": 0.1,
        }

        if function == HeadFunction.STATE_TRACKING:
            patterns["state"] = 0.5
        elif function == HeadFunction.TRANSITION_FOLLOWING:
            patterns["transition"] = 0.5
        elif function == HeadFunction.HIERARCHY_AWARE:
            patterns["hierarchy"] = 0.5
        elif function == HeadFunction.SYNTAX_CHECKING:
            patterns["syntax"] = 0.5

        # Normalize
        total = sum(patterns.values())
        patterns = {k: v/total for k, v in patterns.items()}

        return function, confidence, patterns


# Test prompts
TAXONOMY_PROMPTS = [
    "Generate a JSON statechart for a traffic light:",
    "Create a JSON statechart for user authentication:",
    "Design a JSON statechart with nested compound states:",
    "Build a JSON statechart for an order workflow:",
]


def build_taxonomy(
    model: Optional[HookedModelWrapper] = None,
    prompts: Optional[List[str]] = None,
    n_layers: int = 24,
    n_heads: int = 16,
    verbose: bool = True,
) -> TaxonomyResult:
    """
    Build attention head taxonomy.

    Args:
        model: Model to analyze (mock if None)
        prompts: Test prompts
        n_layers: Number of layers
        n_heads: Heads per layer
        verbose: Print progress

    Returns:
        TaxonomyResult with head classifications
    """
    if prompts is None:
        prompts = TAXONOMY_PROMPTS

    taxonomy = HeadTaxonomy(
        model=model,
        n_layers=n_layers,
        n_heads=n_heads,
        verbose=verbose,
    )

    return taxonomy.build_taxonomy(prompts)


def demo():
    """Demonstrate head taxonomy."""
    print("=" * 60)
    print("Attention Head Taxonomy")
    print("=" * 60)

    result = build_taxonomy(model=None, verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nFunction Distribution:")
    for func, count in sorted(result.function_counts.items(), key=lambda x: -x[1]):
        pct = count / len(result.heads) * 100
        print(f"  {func.value:<25} {count:>4} ({pct:.1f}%)")

    print("\nTop Heads by Function:")
    for func in HeadFunction:
        if func == HeadFunction.UNKNOWN:
            continue
        heads = result.get_heads_by_function(func)
        top_heads = sorted(heads, key=lambda h: -h.confidence)[:3]
        if top_heads:
            print(f"\n  {func.value}:")
            for h in top_heads:
                print(f"    {h.id}: confidence={h.confidence:.2f}")

    print("\nLayer Distribution:")
    for layer in range(0, 24, 8):
        layer_range = list(range(layer, min(layer+8, 24)))
        func_counts = defaultdict(int)
        for l in layer_range:
            for func, count in result.layer_distribution.get(l, {}).items():
                func_counts[func] += count

        dominant = max(func_counts.items(), key=lambda x: x[1]) if func_counts else (HeadFunction.UNKNOWN, 0)
        print(f"  L{layer}-{layer+7}: {dominant[0].value} dominant")

    return result


if __name__ == "__main__":
    demo()
