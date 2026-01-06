#!/usr/bin/env python3
"""
Attention Visualization: Extract and analyze attention patterns.

Demonstrates:
1. Extracting attention patterns from statechart generation
2. Identifying key token relationships
3. Visualizing attention heatmaps (text-based)
4. Analyzing what tokens influence structure decisions
"""

import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

# Add utils to path
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    MLUX_AVAILABLE,
)


@dataclass
class AttentionStats:
    """Statistics about attention patterns."""
    layer: int
    head: int
    avg_attention: float
    max_attention: float
    entropy: float  # Higher = more distributed attention


@dataclass
class TokenAttention:
    """Attention from one token to others."""
    token: str
    position: int
    top_attended: List[Tuple[str, int, float]]  # (token, position, attention)


@dataclass
class AttentionAnalysis:
    """Full attention analysis result."""
    prompt: str
    tokens: List[str]
    layer_stats: Dict[int, List[AttentionStats]]
    token_attention: List[TokenAttention]
    key_relationships: List[str]


class AttentionAnalyzer:
    """
    Analyzes attention patterns in statechart generation.

    Uses mlux when available, provides mock analysis otherwise.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        backend: Optional[ModelBackend] = None,
    ):
        """Initialize analyzer."""
        print("=" * 60)
        print("ATTENTION ANALYZER INITIALIZATION")
        print("=" * 60)
        print(f"MLUX available: {MLUX_AVAILABLE}")

        self.model = load_model(model_name, backend=backend)
        print(f"Backend: {self.model.backend.name}")
        print(f"Has interpretability: {self.model.has_interpretability}")
        print("=" * 60)

    def analyze_prompt(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> AttentionAnalysis:
        """
        Analyze attention patterns for a prompt.

        Args:
            prompt: Input prompt
            layers: Specific layers to analyze (None = all)

        Returns:
            AttentionAnalysis with patterns and statistics
        """
        if not self.model.has_interpretability:
            return self._mock_analysis(prompt)

        # Get attention patterns
        attention = self.model.get_attention_patterns(prompt, layers=layers)

        # Tokenize for reference
        tokens = self._tokenize(prompt)

        # Compute statistics per layer/head
        layer_stats = {}
        for layer_idx, attn_matrix in attention.items():
            layer_stats[layer_idx] = self._compute_layer_stats(attn_matrix)

        # Find top attended tokens per position
        token_attention = self._analyze_token_attention(tokens, attention)

        # Identify key relationships
        key_relationships = self._find_key_relationships(tokens, attention)

        return AttentionAnalysis(
            prompt=prompt,
            tokens=tokens,
            layer_stats=layer_stats,
            token_attention=token_attention,
            key_relationships=key_relationships,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.model.tokenizer:
            ids = self.model.tokenizer.encode(text)
            # Try to decode individual tokens
            try:
                return [self.model.tokenizer.decode([i]) for i in ids]
            except:
                return [str(i) for i in ids]
        return list(text)  # Fallback: character-level

    def _compute_layer_stats(self, attn_matrix: Any) -> List[AttentionStats]:
        """Compute statistics for a layer's attention."""
        stats = []

        try:
            import numpy as np
            attn = np.array(attn_matrix)

            # Shape: [heads, seq, seq]
            num_heads = attn.shape[0]

            for head in range(num_heads):
                head_attn = attn[head]
                avg = float(np.mean(head_attn))
                max_val = float(np.max(head_attn))

                # Compute entropy
                flat = head_attn.flatten()
                flat = flat[flat > 0]  # Remove zeros
                if len(flat) > 0:
                    entropy = float(-np.sum(flat * np.log(flat + 1e-10)))
                else:
                    entropy = 0.0

                stats.append(AttentionStats(
                    layer=0,  # Will be set by caller
                    head=head,
                    avg_attention=avg,
                    max_attention=max_val,
                    entropy=entropy,
                ))
        except ImportError:
            # No numpy - return mock stats
            stats.append(AttentionStats(
                layer=0, head=0,
                avg_attention=0.1,
                max_attention=0.5,
                entropy=2.0,
            ))

        return stats

    def _analyze_token_attention(
        self,
        tokens: List[str],
        attention: Dict[int, Any],
    ) -> List[TokenAttention]:
        """Analyze what each token attends to."""
        results = []

        if not attention:
            return results

        try:
            import numpy as np

            # Use middle layer for analysis
            layer_idx = list(attention.keys())[len(attention) // 2]
            attn = np.array(attention[layer_idx])

            # Average across heads
            avg_attn = np.mean(attn, axis=0)  # [seq, seq]

            for pos, token in enumerate(tokens[:min(len(tokens), avg_attn.shape[0])]):
                # Get attention from this position
                attn_from = avg_attn[pos]

                # Find top attended positions
                top_k = min(3, len(attn_from))
                top_indices = np.argsort(attn_from)[-top_k:][::-1]

                top_attended = []
                for idx in top_indices:
                    if idx < len(tokens):
                        top_attended.append((
                            tokens[idx],
                            int(idx),
                            float(attn_from[idx]),
                        ))

                results.append(TokenAttention(
                    token=token,
                    position=pos,
                    top_attended=top_attended,
                ))
        except (ImportError, IndexError, KeyError):
            pass

        return results

    def _find_key_relationships(
        self,
        tokens: List[str],
        attention: Dict[int, Any],
    ) -> List[str]:
        """Find key attention relationships for statechart generation."""
        relationships = []

        if not attention:
            return relationships

        try:
            import numpy as np

            # Keywords to look for in statechart context
            keywords = ['state', 'transition', 'from', 'to', 'event', 'initial', 'label']

            # Find keyword positions
            keyword_positions = {}
            for pos, token in enumerate(tokens):
                token_lower = token.lower().strip()
                for kw in keywords:
                    if kw in token_lower:
                        keyword_positions[pos] = kw
                        break

            if not keyword_positions:
                return ["No statechart keywords found in tokens"]

            # Analyze attention between keywords
            layer_idx = list(attention.keys())[len(attention) // 2]
            attn = np.array(attention[layer_idx])
            avg_attn = np.mean(attn, axis=0)

            for pos, kw in list(keyword_positions.items())[:5]:
                if pos < avg_attn.shape[0]:
                    # Find what this keyword attends to
                    attn_from = avg_attn[pos]
                    top_idx = np.argmax(attn_from)
                    if top_idx < len(tokens):
                        relationships.append(
                            f"'{kw}' (pos {pos}) -> '{tokens[top_idx]}' (pos {top_idx}): {attn_from[top_idx]:.3f}"
                        )
        except (ImportError, IndexError, KeyError) as e:
            relationships.append(f"Analysis error: {e}")

        return relationships

    def _mock_analysis(self, prompt: str) -> AttentionAnalysis:
        """Provide mock analysis when mlux unavailable."""
        tokens = list(prompt[:50]) + ["..."]

        return AttentionAnalysis(
            prompt=prompt,
            tokens=tokens,
            layer_stats={
                0: [AttentionStats(0, 0, 0.1, 0.5, 2.0)],
                6: [AttentionStats(6, 0, 0.1, 0.4, 2.2)],
                12: [AttentionStats(12, 0, 0.1, 0.3, 2.5)],
            },
            token_attention=[
                TokenAttention("G", 0, [("e", 1, 0.3), ("n", 2, 0.2)]),
                TokenAttention("e", 1, [("G", 0, 0.4), ("n", 2, 0.2)]),
            ],
            key_relationships=[
                "[MOCK] No real attention data available",
                "[MOCK] Install mlux for actual analysis",
            ],
        )

    def print_analysis(self, analysis: AttentionAnalysis):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print("ATTENTION ANALYSIS")
        print("=" * 60)

        print(f"\nPrompt: {analysis.prompt[:80]}...")
        print(f"Tokens: {len(analysis.tokens)}")

        print("\n--- Layer Statistics ---")
        for layer, stats in list(analysis.layer_stats.items())[:3]:
            print(f"\nLayer {layer}:")
            for s in stats[:4]:  # First 4 heads
                print(f"  Head {s.head}: avg={s.avg_attention:.4f}, "
                      f"max={s.max_attention:.4f}, entropy={s.entropy:.2f}")

        print("\n--- Token Attention (first 5 tokens) ---")
        for ta in analysis.token_attention[:5]:
            attended = ", ".join(
                f"'{t}'@{p}:{a:.2f}" for t, p, a in ta.top_attended[:2]
            )
            print(f"  '{ta.token}' -> [{attended}]")

        print("\n--- Key Relationships ---")
        for rel in analysis.key_relationships[:5]:
            print(f"  {rel}")


def visualize_attention_heatmap(
    attention: Dict[int, Any],
    tokens: List[str],
    layer: int = 0,
    head: int = 0,
    max_display: int = 20,
):
    """
    Print text-based attention heatmap.

    Args:
        attention: Attention patterns from model
        tokens: Token list
        layer: Layer to visualize
        head: Head to visualize
        max_display: Max tokens to display
    """
    print(f"\n=== Attention Heatmap (Layer {layer}, Head {head}) ===")

    if layer not in attention:
        print(f"Layer {layer} not in attention data")
        return

    try:
        import numpy as np
        attn = np.array(attention[layer])

        if head >= attn.shape[0]:
            print(f"Head {head} not available (only {attn.shape[0]} heads)")
            return

        head_attn = attn[head]
        n = min(max_display, head_attn.shape[0], len(tokens))

        # Header
        header = "    |" + "".join(f"{t[:3]:>4}" for t in tokens[:n])
        print(header)
        print("-" * len(header))

        # Rows
        for i in range(n):
            row = f"{tokens[i][:3]:>3} |"
            for j in range(n):
                val = head_attn[i, j]
                # Use ASCII blocks for intensity
                if val > 0.5:
                    char = "##"
                elif val > 0.3:
                    char = "**"
                elif val > 0.1:
                    char = "++"
                elif val > 0.05:
                    char = ".."
                else:
                    char = "  "
                row += f"{char:>4}"
            print(row)

        print("\nLegend: ## (>0.5), ** (>0.3), ++ (>0.1), .. (>0.05)")

    except ImportError:
        print("[Requires numpy for heatmap visualization]")


def demo_attention_analysis():
    """Demo: Analyze attention in statechart generation."""
    print("=" * 60)
    print("DEMO: ATTENTION ANALYSIS")
    print("=" * 60)

    analyzer = AttentionAnalyzer()

    prompt = """Generate a statechart JSON for a traffic light:
```json
{"root_state": {"label": "__root__", "type": 2, "children": ["""

    analysis = analyzer.analyze_prompt(prompt, layers=[0, 6, 12])
    analyzer.print_analysis(analysis)

    return analysis


def demo_token_relationships():
    """Demo: Find key token relationships."""
    print("\n" + "=" * 60)
    print("DEMO: TOKEN RELATIONSHIPS")
    print("=" * 60)

    analyzer = AttentionAnalyzer()

    # Prompt with explicit statechart structure
    prompt = """{"root_state": {"label": "Root", "children": [{"label": "Idle", "is_initial": true}, {"label": "Active"}]}, "transitions": [{"from": ["Idle"], "to": ["Active"], "event": "START"}]}"""

    analysis = analyzer.analyze_prompt(prompt)

    print(f"\nAnalyzing: {prompt[:60]}...")
    print(f"\nKey relationships found:")
    for rel in analysis.key_relationships:
        print(f"  {rel}")

    return analysis


def run_all_demos():
    """Run all attention demos."""
    print("=" * 60)
    print("ATTENTION VISUALIZATION DEMOS")
    print("=" * 60)
    print(f"MLUX available: {MLUX_AVAILABLE}")

    results = []

    # Demo 1: Basic attention analysis
    results.append(("attention_analysis", demo_attention_analysis()))

    # Demo 2: Token relationships
    results.append(("token_relationships", demo_token_relationships()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, analysis in results:
        status = "OK" if analysis.tokens else "FAIL"
        print(f"  [{status}] {name}: {len(analysis.tokens)} tokens, "
              f"{len(analysis.layer_stats)} layers analyzed")

    return results


if __name__ == "__main__":
    run_all_demos()
