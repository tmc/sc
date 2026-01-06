#!/usr/bin/env python3
"""
Structure-Aware Head Detection: Find attention heads that focus on SC structure.

Identifies heads that:
1. Attend to structural tokens (braces, brackets, colons)
2. Track key-value relationships in JSON
3. Focus on statechart keywords (state, transition, event, label)
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import json

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False

from utils.mlux_loader import load_model, ModelBackend, MLUX_AVAILABLE


# Structural token categories
STRUCTURE_TOKENS = {
    'braces': ['{', '}'],
    'brackets': ['[', ']'],
    'punctuation': [':', ',', '"'],
    'keywords': ['root_state', 'label', 'type', 'children', 'transitions',
                 'from', 'to', 'event', 'is_initial', 'guard', 'action'],
}


@dataclass
class HeadScore:
    """Score for a single attention head."""
    layer: int
    head: int
    structure_score: float  # How much it attends to structural tokens
    keyword_score: float    # How much it attends to SC keywords
    brace_score: float      # Specifically brace matching
    overall_score: float    # Combined score


@dataclass
class StructureHeadAnalysis:
    """Results of structure head analysis."""
    prompt: str
    tokens: List[str]
    head_scores: List[HeadScore]
    top_structure_heads: List[Tuple[int, int, float]]  # (layer, head, score)
    top_keyword_heads: List[Tuple[int, int, float]]
    top_brace_heads: List[Tuple[int, int, float]]


class StructureHeadDetector:
    """
    Detects attention heads that focus on structural elements.

    Uses mlux to extract attention patterns and score each head
    based on how much it attends to structural tokens.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("STRUCTURE HEAD DETECTOR")
        print("=" * 60)

        if not MLUX_AVAILABLE:
            print("WARNING: mlux not available, using mock analysis")
            self.model = None
            self.tokenizer = None
        else:
            from mlux import HookedModel
            print(f"Loading model: {model_name}")
            self.model = HookedModel.from_pretrained(model_name)
            self.tokenizer = self.model.tokenizer
            config = self.model.config
            num_layers = config.get('num_hidden_layers', config.get('n_layers', 24)) if isinstance(config, dict) else getattr(config, 'num_hidden_layers', 24)
            print(f"Model loaded. Layers: {num_layers}")

        print("=" * 60)

    def _tokenize(self, text: str) -> Tuple[List[str], Any]:
        """Tokenize text and return tokens + ids."""
        if self.tokenizer is None:
            return list(text), list(range(len(text)))

        ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([i]) for i in ids]
        return tokens, ids

    def _get_token_categories(self, tokens: List[str]) -> Dict[str, List[int]]:
        """Categorize token positions."""
        categories = {cat: [] for cat in STRUCTURE_TOKENS}

        for pos, token in enumerate(tokens):
            token_clean = token.strip().lower()

            for cat, patterns in STRUCTURE_TOKENS.items():
                for pattern in patterns:
                    if pattern.lower() in token_clean:
                        categories[cat].append(pos)
                        break

        return categories

    def analyze(self, prompt: str) -> StructureHeadAnalysis:
        """
        Analyze attention patterns to find structure-aware heads.

        Args:
            prompt: Statechart JSON or generation prompt

        Returns:
            StructureHeadAnalysis with head scores
        """
        if not MLUX_AVAILABLE or self.model is None:
            return self._mock_analysis(prompt)

        tokens, token_ids = self._tokenize(prompt)
        categories = self._get_token_categories(tokens)

        # Get attention patterns from model
        # Use get_attention_patterns with layer indices
        config = self.model.config
        num_layers = config.get('num_hidden_layers', config.get('n_layers', 24)) if isinstance(config, dict) else getattr(config, 'num_hidden_layers', 24)

        try:
            attention_cache = self.model.get_attention_patterns(
                prompt,
                layers=list(range(num_layers)),
            )
        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return self._mock_analysis(prompt)

        # Score each head
        head_scores = []
        num_layers = len(attention_cache) if attention_cache else 24

        for layer_idx in range(num_layers):
            layer_key = f"model.layers.{layer_idx}.self_attn"
            if layer_key not in attention_cache and layer_idx not in attention_cache:
                continue

            attn = attention_cache.get(layer_key, attention_cache.get(layer_idx))
            if attn is None:
                continue

            # Convert to numpy for analysis
            if hasattr(attn, 'tolist'):
                attn_np = np.array(attn.tolist())
            else:
                attn_np = np.array(attn)

            # Shape: [batch, heads, seq, seq] or [heads, seq, seq]
            if len(attn_np.shape) == 4:
                attn_np = attn_np[0]  # Remove batch dim

            num_heads = attn_np.shape[0]
            seq_len = attn_np.shape[1]

            for head_idx in range(num_heads):
                head_attn = attn_np[head_idx]  # [seq, seq]

                scores = self._score_head(head_attn, categories, seq_len)

                head_scores.append(HeadScore(
                    layer=layer_idx,
                    head=head_idx,
                    structure_score=scores['structure'],
                    keyword_score=scores['keyword'],
                    brace_score=scores['brace'],
                    overall_score=scores['overall'],
                ))

        # Sort to find top heads
        by_structure = sorted(head_scores, key=lambda h: h.structure_score, reverse=True)
        by_keyword = sorted(head_scores, key=lambda h: h.keyword_score, reverse=True)
        by_brace = sorted(head_scores, key=lambda h: h.brace_score, reverse=True)

        return StructureHeadAnalysis(
            prompt=prompt,
            tokens=tokens,
            head_scores=head_scores,
            top_structure_heads=[(h.layer, h.head, h.structure_score) for h in by_structure[:10]],
            top_keyword_heads=[(h.layer, h.head, h.keyword_score) for h in by_keyword[:10]],
            top_brace_heads=[(h.layer, h.head, h.brace_score) for h in by_brace[:10]],
        )

    def _score_head(
        self,
        head_attn: Any,
        categories: Dict[str, List[int]],
        seq_len: int,
    ) -> Dict[str, float]:
        """Score a single attention head."""
        scores = {'structure': 0.0, 'keyword': 0.0, 'brace': 0.0, 'overall': 0.0}

        # Structure score: attention to braces + brackets + punctuation
        struct_positions = (
            categories.get('braces', []) +
            categories.get('brackets', []) +
            categories.get('punctuation', [])
        )
        if struct_positions:
            valid_pos = [p for p in struct_positions if p < seq_len]
            if valid_pos:
                # Average attention TO structural tokens
                struct_attn = head_attn[:, valid_pos].mean()
                scores['structure'] = float(struct_attn)

        # Keyword score: attention to SC keywords
        keyword_positions = categories.get('keywords', [])
        if keyword_positions:
            valid_pos = [p for p in keyword_positions if p < seq_len]
            if valid_pos:
                keyword_attn = head_attn[:, valid_pos].mean()
                scores['keyword'] = float(keyword_attn)

        # Brace score: specifically brace matching
        brace_positions = categories.get('braces', [])
        if len(brace_positions) >= 2:
            valid_pos = [p for p in brace_positions if p < seq_len]
            if len(valid_pos) >= 2:
                # Attention between braces
                brace_attn = head_attn[valid_pos][:, valid_pos].mean()
                scores['brace'] = float(brace_attn)

        # Overall score
        scores['overall'] = (
            scores['structure'] * 0.4 +
            scores['keyword'] * 0.4 +
            scores['brace'] * 0.2
        )

        return scores

    def _mock_analysis(self, prompt: str) -> StructureHeadAnalysis:
        """Mock analysis when mlux unavailable."""
        tokens = list(prompt[:100])

        # Generate mock scores
        mock_heads = []
        for layer in range(24):
            for head in range(16):
                score = 0.1 + (layer / 24) * 0.2 + (head / 16) * 0.1
                mock_heads.append(HeadScore(
                    layer=layer,
                    head=head,
                    structure_score=score,
                    keyword_score=score * 0.8,
                    brace_score=score * 0.6,
                    overall_score=score,
                ))

        return StructureHeadAnalysis(
            prompt=prompt,
            tokens=tokens,
            head_scores=mock_heads,
            top_structure_heads=[(23, 15, 0.3), (23, 14, 0.29), (22, 15, 0.28)],
            top_keyword_heads=[(23, 15, 0.24), (23, 14, 0.23), (22, 15, 0.22)],
            top_brace_heads=[(23, 15, 0.18), (23, 14, 0.17), (22, 15, 0.16)],
        )

    def print_analysis(self, analysis: StructureHeadAnalysis):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print("STRUCTURE HEAD ANALYSIS")
        print("=" * 60)

        print(f"\nPrompt length: {len(analysis.prompt)} chars")
        print(f"Tokens: {len(analysis.tokens)}")
        print(f"Heads analyzed: {len(analysis.head_scores)}")

        print("\n--- Top Structure-Aware Heads ---")
        for layer, head, score in analysis.top_structure_heads[:5]:
            print(f"  Layer {layer:2d}, Head {head:2d}: {score:.4f}")

        print("\n--- Top Keyword-Focused Heads ---")
        for layer, head, score in analysis.top_keyword_heads[:5]:
            print(f"  Layer {layer:2d}, Head {head:2d}: {score:.4f}")

        print("\n--- Top Brace-Matching Heads ---")
        for layer, head, score in analysis.top_brace_heads[:5]:
            print(f"  Layer {layer:2d}, Head {head:2d}: {score:.4f}")


def demo():
    """Demo structure head detection."""
    print("=" * 60)
    print("STRUCTURE HEAD DETECTION DEMO")
    print("=" * 60)

    detector = StructureHeadDetector()

    # Test with a statechart JSON
    sc_json = '''{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle", "is_initial": true}, {"label": "Active"}]}, "transitions": [{"from": ["Idle"], "to": ["Active"], "event": "START"}]}'''

    analysis = detector.analyze(sc_json)
    detector.print_analysis(analysis)

    return analysis


if __name__ == "__main__":
    demo()
