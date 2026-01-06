#!/usr/bin/env python3
"""
Hierarchy Attention Analysis: Find heads that track parent-child relationships.

Identifies attention patterns that:
1. Connect parent states to their children
2. Track nesting depth (root -> composite -> leaf)
3. Maintain scope awareness in nested structures
"""

import sys
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from utils.mlux_loader import load_model, MLUX_AVAILABLE


@dataclass
class HierarchyNode:
    """A node in the statechart hierarchy."""
    label: str
    token_start: int
    token_end: int
    depth: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class HierarchyAttentionScore:
    """Score for parent-child attention patterns."""
    layer: int
    head: int
    parent_to_child: float    # Attention from parent to children
    child_to_parent: float    # Attention from child to parent
    sibling_attention: float  # Attention between siblings
    depth_awareness: float    # Correlation with nesting depth
    overall_score: float


@dataclass
class HierarchyAnalysis:
    """Results of hierarchy attention analysis."""
    statechart: Dict[str, Any]
    hierarchy: List[HierarchyNode]
    tokens: List[str]
    head_scores: List[HierarchyAttentionScore]
    top_parent_child_heads: List[Tuple[int, int, float]]
    top_depth_aware_heads: List[Tuple[int, int, float]]
    avg_parent_child_attention: float
    avg_sibling_attention: float


class HierarchyAttentionAnalyzer:
    """
    Analyzes attention patterns for hierarchical structure awareness.

    Looks for heads that:
    - Track parent-child relationships between states
    - Are aware of nesting depth
    - Maintain scope during generation
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("HIERARCHY ATTENTION ANALYZER")
        print("=" * 60)

        if not MLUX_AVAILABLE:
            print("WARNING: mlux not available, using mock analysis")
            self.model = None
            self.tokenizer = None
            self.num_layers = 24
        else:
            from mlux import HookedModel
            print(f"Loading model: {model_name}")
            self.model = HookedModel.from_pretrained(model_name)
            self.tokenizer = self.model.tokenizer
            config = self.model.config
            self.num_layers = config.get('num_hidden_layers', config.get('n_layers', 24)) if isinstance(config, dict) else getattr(config, 'num_hidden_layers', 24)
            print(f"Model loaded. Layers: {self.num_layers}")

        print("=" * 60)

    def _tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        """Tokenize text."""
        if self.tokenizer is None:
            return list(text), list(range(len(text)))

        ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([i]) for i in ids]
        return tokens, ids

    def _extract_hierarchy(
        self,
        sc: Dict[str, Any],
        tokens: List[str],
    ) -> List[HierarchyNode]:
        """Extract hierarchy from statechart with token positions."""
        hierarchy = []
        sc_str = json.dumps(sc)

        def find_token_range(label: str) -> Tuple[int, int]:
            """Find token range for a label."""
            # Find label in original string
            pattern = f'"{label}"'
            pos = sc_str.find(pattern)
            if pos == -1:
                return (0, 0)

            # Map character position to token position
            char_count = 0
            start_token = 0
            for i, tok in enumerate(tokens):
                if char_count >= pos:
                    start_token = i
                    break
                char_count += len(tok)

            end_token = min(start_token + len(label.split()), len(tokens))
            return (start_token, end_token)

        def process_state(
            state: Dict[str, Any],
            depth: int,
            parent: Optional[str],
        ):
            label = state.get('label', 'unknown')
            start, end = find_token_range(label)

            node = HierarchyNode(
                label=label,
                token_start=start,
                token_end=end,
                depth=depth,
                parent=parent,
                children=[],
            )

            # Process children
            children = state.get('children', [])
            for child in children:
                child_label = child.get('label', '')
                node.children.append(child_label)
                process_state(child, depth + 1, label)

            hierarchy.append(node)

        root = sc.get('root_state', {})
        process_state(root, 0, None)

        return hierarchy

    def analyze(self, sc: Dict[str, Any]) -> HierarchyAnalysis:
        """
        Analyze attention patterns for hierarchy awareness.

        Args:
            sc: Statechart dictionary

        Returns:
            HierarchyAnalysis with head scores
        """
        sc_str = json.dumps(sc)
        tokens, _ = self._tokenize(sc_str)
        hierarchy = self._extract_hierarchy(sc, tokens)

        if not MLUX_AVAILABLE or self.model is None:
            return self._mock_analysis(sc, hierarchy, tokens)

        # Get attention patterns
        try:
            attention = self.model.get_attention_patterns(
                sc_str,
                layers=list(range(self.num_layers)),
            )
        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return self._mock_analysis(sc, hierarchy, tokens)

        # Score each head for hierarchy awareness
        head_scores = []

        for layer_idx, attn in attention.items():
            if not NUMPY_AVAILABLE:
                continue

            attn_np = np.array(attn) if not isinstance(attn, np.ndarray) else attn

            # Handle shape
            if len(attn_np.shape) == 4:
                attn_np = attn_np[0]

            num_heads = attn_np.shape[0]

            for head_idx in range(num_heads):
                head_attn = attn_np[head_idx]
                scores = self._score_hierarchy_head(head_attn, hierarchy)

                head_scores.append(HierarchyAttentionScore(
                    layer=layer_idx,
                    head=head_idx,
                    parent_to_child=scores['parent_to_child'],
                    child_to_parent=scores['child_to_parent'],
                    sibling_attention=scores['sibling'],
                    depth_awareness=scores['depth'],
                    overall_score=scores['overall'],
                ))

        # Sort to find top heads
        by_parent_child = sorted(
            head_scores,
            key=lambda h: h.parent_to_child + h.child_to_parent,
            reverse=True
        )
        by_depth = sorted(head_scores, key=lambda h: h.depth_awareness, reverse=True)

        # Compute averages
        avg_pc = np.mean([h.parent_to_child for h in head_scores]) if head_scores else 0
        avg_sib = np.mean([h.sibling_attention for h in head_scores]) if head_scores else 0

        return HierarchyAnalysis(
            statechart=sc,
            hierarchy=hierarchy,
            tokens=tokens,
            head_scores=head_scores,
            top_parent_child_heads=[
                (h.layer, h.head, h.parent_to_child + h.child_to_parent)
                for h in by_parent_child[:10]
            ],
            top_depth_aware_heads=[
                (h.layer, h.head, h.depth_awareness)
                for h in by_depth[:10]
            ],
            avg_parent_child_attention=float(avg_pc),
            avg_sibling_attention=float(avg_sib),
        )

    def _score_hierarchy_head(
        self,
        head_attn: Any,
        hierarchy: List[HierarchyNode],
    ) -> Dict[str, float]:
        """Score a head for hierarchy awareness."""
        scores = {
            'parent_to_child': 0.0,
            'child_to_parent': 0.0,
            'sibling': 0.0,
            'depth': 0.0,
            'overall': 0.0,
        }

        if not NUMPY_AVAILABLE or len(hierarchy) < 2:
            return scores

        seq_len = head_attn.shape[0]

        # Build parent-child pairs
        parent_child_pairs = []
        for node in hierarchy:
            if node.parent:
                parent_node = next((n for n in hierarchy if n.label == node.parent), None)
                if parent_node:
                    parent_child_pairs.append((parent_node, node))

        # Score parent-to-child attention
        pc_scores = []
        cp_scores = []
        for parent, child in parent_child_pairs:
            p_start, p_end = parent.token_start, parent.token_end
            c_start, c_end = child.token_start, child.token_end

            if p_end <= seq_len and c_end <= seq_len:
                # Parent attending to child
                if p_start < seq_len and c_start < seq_len:
                    pc_attn = head_attn[p_start:min(p_end, seq_len), c_start:min(c_end, seq_len)]
                    if pc_attn.size > 0:
                        pc_scores.append(float(np.mean(pc_attn)))

                    # Child attending to parent
                    cp_attn = head_attn[c_start:min(c_end, seq_len), p_start:min(p_end, seq_len)]
                    if cp_attn.size > 0:
                        cp_scores.append(float(np.mean(cp_attn)))

        if pc_scores:
            scores['parent_to_child'] = np.mean(pc_scores)
        if cp_scores:
            scores['child_to_parent'] = np.mean(cp_scores)

        # Score sibling attention
        sibling_scores = []
        for node in hierarchy:
            siblings = [n for n in hierarchy if n.parent == node.parent and n.label != node.label]
            for sib in siblings:
                n_start = node.token_start
                s_start = sib.token_start
                if n_start < seq_len and s_start < seq_len:
                    sib_attn = head_attn[n_start, s_start]
                    sibling_scores.append(float(sib_attn))

        if sibling_scores:
            scores['sibling'] = np.mean(sibling_scores)

        # Depth awareness: correlation between attention and depth
        depth_scores = []
        depths = [n.depth for n in hierarchy]
        if len(set(depths)) > 1:  # Need variation in depth
            for node in hierarchy:
                if node.token_start < seq_len:
                    # Attention from this node to all others
                    for other in hierarchy:
                        if other.token_start < seq_len and other.label != node.label:
                            attn_val = head_attn[node.token_start, other.token_start]
                            depth_diff = abs(node.depth - other.depth)
                            # Higher attention to closer depths = higher score
                            if depth_diff > 0:
                                depth_scores.append(float(attn_val) / depth_diff)

        if depth_scores:
            scores['depth'] = np.mean(depth_scores)

        # Overall score
        scores['overall'] = (
            scores['parent_to_child'] * 0.3 +
            scores['child_to_parent'] * 0.3 +
            scores['sibling'] * 0.2 +
            scores['depth'] * 0.2
        )

        return scores

    def _mock_analysis(
        self,
        sc: Dict[str, Any],
        hierarchy: List[HierarchyNode],
        tokens: List[str],
    ) -> HierarchyAnalysis:
        """Mock analysis when mlux unavailable."""
        mock_heads = []
        for layer in range(24):
            for head in range(16):
                score = 0.05 + (layer / 24) * 0.15
                mock_heads.append(HierarchyAttentionScore(
                    layer=layer,
                    head=head,
                    parent_to_child=score,
                    child_to_parent=score * 0.8,
                    sibling_attention=score * 0.5,
                    depth_awareness=score * 0.6,
                    overall_score=score,
                ))

        return HierarchyAnalysis(
            statechart=sc,
            hierarchy=hierarchy,
            tokens=tokens,
            head_scores=mock_heads,
            top_parent_child_heads=[(23, 15, 0.35), (23, 14, 0.33), (22, 15, 0.31)],
            top_depth_aware_heads=[(23, 15, 0.12), (22, 14, 0.11), (21, 15, 0.10)],
            avg_parent_child_attention=0.15,
            avg_sibling_attention=0.08,
        )

    def print_analysis(self, analysis: HierarchyAnalysis):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print("HIERARCHY ATTENTION ANALYSIS")
        print("=" * 60)

        print(f"\nHierarchy nodes: {len(analysis.hierarchy)}")
        for node in analysis.hierarchy:
            indent = "  " * node.depth
            parent_info = f" (parent: {node.parent})" if node.parent else " (root)"
            print(f"{indent}- {node.label}{parent_info}")

        print(f"\nHeads analyzed: {len(analysis.head_scores)}")

        print("\n--- Top Parent-Child Attention Heads ---")
        for layer, head, score in analysis.top_parent_child_heads[:5]:
            print(f"  Layer {layer:2d}, Head {head:2d}: {score:.4f}")

        print("\n--- Top Depth-Aware Heads ---")
        for layer, head, score in analysis.top_depth_aware_heads[:5]:
            print(f"  Layer {layer:2d}, Head {head:2d}: {score:.4f}")

        print(f"\n--- Averages ---")
        print(f"  Parent-child attention: {analysis.avg_parent_child_attention:.4f}")
        print(f"  Sibling attention: {analysis.avg_sibling_attention:.4f}")


def demo():
    """Demo hierarchy attention analysis."""
    print("=" * 60)
    print("HIERARCHY ATTENTION ANALYSIS DEMO")
    print("=" * 60)

    analyzer = HierarchyAttentionAnalyzer()

    # Test with hierarchical statechart
    sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {
                    "label": "Idle",
                    "is_initial": True,
                },
                {
                    "label": "Active",
                    "type": 2,
                    "children": [
                        {"label": "Running"},
                        {"label": "Paused"},
                    ]
                },
                {"label": "Done"},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Active"], "event": "START"},
            {"from": ["Active"], "to": ["Done"], "event": "FINISH"},
        ]
    }

    analysis = analyzer.analyze(sc)
    analyzer.print_analysis(analysis)

    return analysis


if __name__ == "__main__":
    demo()
