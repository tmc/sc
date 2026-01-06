#!/usr/bin/env python3
"""
Validity Difference Analysis: Compare attention patterns for valid vs invalid SCs.

Identifies attention differences that correlate with:
1. Missing required fields (no initial state, missing labels)
2. Invalid transitions (dangling references)
3. Structural errors (unclosed braces, malformed JSON)
"""

import sys
import json
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from utils.mlux_loader import load_model, MLUX_AVAILABLE


@dataclass
class AttentionDiff:
    """Difference in attention patterns between valid and invalid."""
    layer: int
    head: int
    valid_avg: float
    invalid_avg: float
    difference: float  # valid - invalid
    significance: float  # abs(difference) / std


@dataclass
class ValidityPattern:
    """Pattern that distinguishes valid from invalid."""
    error_type: str
    description: str
    attention_signature: Dict[str, float]  # Which heads show difference
    discriminative_power: float  # How well this pattern separates


@dataclass
class ValidityAnalysis:
    """Results of validity difference analysis."""
    valid_sc: Dict[str, Any]
    invalid_sc: Dict[str, Any]
    error_type: str
    attention_diffs: List[AttentionDiff]
    top_discriminative_heads: List[Tuple[int, int, float]]
    patterns: List[ValidityPattern]
    classification_accuracy: float  # If we use attention to classify


class ValidityDiffAnalyzer:
    """
    Compares attention patterns between valid and invalid statecharts.

    Finds heads that reliably distinguish valid from invalid structures,
    potentially useful for:
    - Early error detection during generation
    - Steering away from invalid patterns
    - Understanding model's implicit validity checking
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("VALIDITY DIFF ANALYZER")
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

    def _get_attention(self, text: str) -> Dict[int, Any]:
        """Get attention patterns for text."""
        if not MLUX_AVAILABLE or self.model is None:
            return {}

        try:
            return self.model.get_attention_patterns(text, layers=list(range(self.num_layers)))
        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return {}

    def analyze_pair(
        self,
        valid_sc: Dict[str, Any],
        invalid_sc: Dict[str, Any],
        error_type: str = "unknown",
    ) -> ValidityAnalysis:
        """
        Analyze attention differences between valid and invalid SC.

        Args:
            valid_sc: Valid statechart
            invalid_sc: Invalid statechart (with known error)
            error_type: Type of error in invalid SC

        Returns:
            ValidityAnalysis with attention differences
        """
        valid_str = json.dumps(valid_sc)
        invalid_str = json.dumps(invalid_sc)

        if not MLUX_AVAILABLE or self.model is None:
            return self._mock_analysis(valid_sc, invalid_sc, error_type)

        # Get attention for both
        valid_attn = self._get_attention(valid_str)
        invalid_attn = self._get_attention(invalid_str)

        # Compute differences
        attention_diffs = self._compute_diffs(valid_attn, invalid_attn)

        # Find discriminative heads
        by_discrimination = sorted(
            attention_diffs,
            key=lambda d: abs(d.difference),
            reverse=True
        )

        # Build patterns
        patterns = self._build_patterns(attention_diffs, error_type)

        # Estimate classification accuracy
        accuracy = self._estimate_accuracy(attention_diffs)

        return ValidityAnalysis(
            valid_sc=valid_sc,
            invalid_sc=invalid_sc,
            error_type=error_type,
            attention_diffs=attention_diffs,
            top_discriminative_heads=[
                (d.layer, d.head, d.difference)
                for d in by_discrimination[:10]
            ],
            patterns=patterns,
            classification_accuracy=accuracy,
        )

    def _compute_diffs(
        self,
        valid_attn: Dict[int, Any],
        invalid_attn: Dict[int, Any],
    ) -> List[AttentionDiff]:
        """Compute attention differences."""
        diffs = []

        if not NUMPY_AVAILABLE:
            return diffs

        common_layers = set(valid_attn.keys()) & set(invalid_attn.keys())

        for layer in common_layers:
            v_attn = np.array(valid_attn[layer])
            i_attn = np.array(invalid_attn[layer])

            # Handle shape differences
            if v_attn.shape != i_attn.shape:
                continue

            if len(v_attn.shape) == 4:
                v_attn = v_attn[0]
                i_attn = i_attn[0]

            num_heads = v_attn.shape[0]

            for head in range(num_heads):
                v_head = v_attn[head]
                i_head = i_attn[head]

                v_avg = float(np.mean(v_head))
                i_avg = float(np.mean(i_head))
                diff = v_avg - i_avg

                # Significance based on pooled std
                pooled_std = np.sqrt((np.std(v_head)**2 + np.std(i_head)**2) / 2)
                significance = abs(diff) / (pooled_std + 1e-8)

                diffs.append(AttentionDiff(
                    layer=layer,
                    head=head,
                    valid_avg=v_avg,
                    invalid_avg=i_avg,
                    difference=diff,
                    significance=float(significance),
                ))

        return diffs

    def _build_patterns(
        self,
        diffs: List[AttentionDiff],
        error_type: str,
    ) -> List[ValidityPattern]:
        """Build validity patterns from differences."""
        if not diffs:
            return []

        # Group by significance
        significant = [d for d in diffs if d.significance > 1.0]

        if not significant:
            return [ValidityPattern(
                error_type=error_type,
                description="No significant attention differences found",
                attention_signature={},
                discriminative_power=0.0,
            )]

        # Build signature from top discriminative heads
        signature = {}
        for d in significant[:5]:
            key = f"L{d.layer}H{d.head}"
            signature[key] = d.difference

        avg_significance = np.mean([d.significance for d in significant])

        return [ValidityPattern(
            error_type=error_type,
            description=f"Pattern for {error_type}: {len(significant)} discriminative heads",
            attention_signature=signature,
            discriminative_power=float(avg_significance),
        )]

    def _estimate_accuracy(self, diffs: List[AttentionDiff]) -> float:
        """Estimate how well attention can classify valid vs invalid."""
        if not diffs or not NUMPY_AVAILABLE:
            return 0.5  # Random chance

        # Use top 5 most discriminative heads
        top_diffs = sorted(diffs, key=lambda d: d.significance, reverse=True)[:5]

        # Simple threshold-based classification estimate
        # If avg significance > 1.5, assume good separation
        avg_sig = np.mean([d.significance for d in top_diffs])

        # Map to accuracy estimate (rough heuristic)
        if avg_sig > 2.0:
            return 0.9
        elif avg_sig > 1.5:
            return 0.8
        elif avg_sig > 1.0:
            return 0.7
        else:
            return 0.5 + avg_sig * 0.1

    def _mock_analysis(
        self,
        valid_sc: Dict[str, Any],
        invalid_sc: Dict[str, Any],
        error_type: str,
    ) -> ValidityAnalysis:
        """Mock analysis when mlux unavailable."""
        mock_diffs = []
        for layer in range(24):
            for head in range(16):
                diff = 0.02 * (1 - layer/24)  # Later layers less different
                mock_diffs.append(AttentionDiff(
                    layer=layer,
                    head=head,
                    valid_avg=0.1,
                    invalid_avg=0.1 - diff,
                    difference=diff,
                    significance=diff * 20,
                ))

        return ValidityAnalysis(
            valid_sc=valid_sc,
            invalid_sc=invalid_sc,
            error_type=error_type,
            attention_diffs=mock_diffs,
            top_discriminative_heads=[
                (0, 0, 0.02), (0, 1, 0.019), (1, 0, 0.018)
            ],
            patterns=[ValidityPattern(
                error_type=error_type,
                description=f"[MOCK] Pattern for {error_type}",
                attention_signature={"L0H0": 0.02, "L0H1": 0.019},
                discriminative_power=0.4,
            )],
            classification_accuracy=0.65,
        )

    def analyze_error_types(
        self,
        valid_base: Dict[str, Any],
    ) -> Dict[str, ValidityAnalysis]:
        """
        Analyze multiple error types against a valid base.

        Creates invalid variants and analyzes each.
        """
        results = {}

        # Error type 1: Missing initial state
        invalid_no_initial = copy.deepcopy(valid_base)
        for child in invalid_no_initial.get('root_state', {}).get('children', []):
            child.pop('is_initial', None)
        results['no_initial'] = self.analyze_pair(
            valid_base, invalid_no_initial, "no_initial_state"
        )

        # Error type 2: Invalid transition target
        invalid_target = copy.deepcopy(valid_base)
        if invalid_target.get('transitions'):
            invalid_target['transitions'][0]['to'] = ['NonexistentState']
        results['invalid_target'] = self.analyze_pair(
            valid_base, invalid_target, "invalid_transition_target"
        )

        # Error type 3: Duplicate state label
        invalid_dup = copy.deepcopy(valid_base)
        children = invalid_dup.get('root_state', {}).get('children', [])
        if len(children) >= 2:
            children[1]['label'] = children[0]['label']
        results['duplicate_label'] = self.analyze_pair(
            valid_base, invalid_dup, "duplicate_state_label"
        )

        # Error type 4: Missing label
        invalid_no_label = copy.deepcopy(valid_base)
        children = invalid_no_label.get('root_state', {}).get('children', [])
        if children:
            children[0].pop('label', None)
        results['missing_label'] = self.analyze_pair(
            valid_base, invalid_no_label, "missing_label"
        )

        return results

    def print_analysis(self, analysis: ValidityAnalysis):
        """Print analysis results."""
        print("\n" + "=" * 60)
        print(f"VALIDITY DIFF ANALYSIS: {analysis.error_type}")
        print("=" * 60)

        print(f"\nTop discriminative heads:")
        for layer, head, diff in analysis.top_discriminative_heads[:5]:
            direction = "+" if diff > 0 else "-"
            print(f"  Layer {layer:2d}, Head {head:2d}: {direction}{abs(diff):.4f}")

        print(f"\nPatterns found: {len(analysis.patterns)}")
        for pattern in analysis.patterns:
            print(f"  - {pattern.description}")
            print(f"    Power: {pattern.discriminative_power:.3f}")

        print(f"\nEstimated classification accuracy: {analysis.classification_accuracy:.1%}")


def demo():
    """Demo validity difference analysis."""
    print("=" * 60)
    print("VALIDITY DIFF ANALYSIS DEMO")
    print("=" * 60)

    analyzer = ValidityDiffAnalyzer()

    # Valid base statechart
    valid_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "is_initial": True},
                {"label": "Active"},
                {"label": "Done"},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Active"], "event": "START"},
            {"from": ["Active"], "to": ["Done"], "event": "FINISH"},
        ]
    }

    # Analyze all error types
    results = analyzer.analyze_error_types(valid_sc)

    for error_type, analysis in results.items():
        analyzer.print_analysis(analysis)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for error_type, analysis in results.items():
        print(f"  {error_type}: accuracy={analysis.classification_accuracy:.1%}, "
              f"patterns={len(analysis.patterns)}")

    return results


if __name__ == "__main__":
    demo()
