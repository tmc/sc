#!/usr/bin/env python3
"""
Attention Pattern Analysis

Analyzes attention patterns in the trained transformer to understand
how the model tracks syntactic structure.

Key analyses:
1. Attention head specialization (which heads focus on what)
2. Positional patterns (local vs long-range dependencies)
3. Token-type attention (how different tokens attend to each other)
4. Syntax-aware patterns (bracket matching, function boundaries, etc.)

Usage:
    python -m experiments.exp_grammar_induction.analyze_attention --run-dir runs/run_20260104_154119
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .token_collector import GoTokenCollector, TOKEN_VOCAB
from .sequence_model import GoSyntaxTransformer, TransformerConfig

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


@dataclass
class AttentionPattern:
    """A discovered attention pattern."""
    name: str
    description: str
    head_indices: List[Tuple[int, int]]  # (layer, head)
    score: float  # 0-1 indicating strength
    examples: List[str]


def analyze_attention(run_dir: str, max_sequences: int = 100):
    """Analyze attention patterns in the trained model."""

    print("=" * 70)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    print(f"\nRun directory: {run_dir}")
    print(f"MLX available: {HAS_MLX}")

    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nModel config:")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_heads']}")
    print(f"  Hidden dim: {config['hidden_dim']}")

    # Load sequences
    sequences_path = os.path.join(run_dir, 'sequences.json')
    collector = GoTokenCollector()
    collector.load(sequences_path)

    print(f"\nLoaded {len(collector.sequences)} sequences")

    if not HAS_MLX:
        print("\nERROR: MLX required for attention analysis")
        return None

    # Create transformer
    transformer_config = TransformerConfig(
        vocab_size=len(TOKEN_VOCAB),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    )

    transformer = GoSyntaxTransformer(transformer_config)
    print(f"\nCreated transformer for attention extraction")

    # Collect attention statistics
    print(f"\n{'=' * 70}")
    print("COLLECTING ATTENTION STATISTICS")
    print("=" * 70)

    # Track attention patterns
    head_specialization = defaultdict(lambda: defaultdict(float))
    position_patterns = defaultdict(lambda: defaultdict(float))
    token_pair_attention = defaultdict(lambda: defaultdict(float))
    bracket_matching = []

    sequences = collector.sequences[:min(max_sequences, 30)]  # Reduced for speed
    print(f"\nAnalyzing {len(sequences)} sequences...", flush=True)

    for seq_idx, seq in enumerate(sequences):
        if seq_idx % 5 == 0:
            import sys
            print(f"  Processing sequence {seq_idx+1}/{len(sequences)}...", flush=True)
            sys.stdout.flush()
        token_ids = seq.to_type_ids()
        if len(token_ids) < 10:
            continue

        token_ids = token_ids[:config['max_seq_len']]
        batch = mx.array([token_ids])

        # Get attention weights
        attention_weights = transformer.get_attention_weights(batch)

        if not attention_weights:
            continue

        token_types = [t.type_name for t in seq.tokens[:len(token_ids)]]

        # Analyze each layer and head
        for layer_idx, layer_attn in enumerate(attention_weights):
            # layer_attn shape: (batch, heads, seq, seq)
            for head_idx in range(config['num_heads']):
                head_attn = layer_attn[0, head_idx]  # (seq, seq)

                # 1. Position pattern analysis
                seq_len = head_attn.shape[0]
                for i in range(seq_len):
                    for j in range(i + 1):  # Only look at causal positions
                        distance = i - j
                        attn_val = float(head_attn[i, j])

                        if distance == 0:
                            position_patterns[(layer_idx, head_idx)]['self'] += attn_val
                        elif distance == 1:
                            position_patterns[(layer_idx, head_idx)]['prev'] += attn_val
                        elif distance <= 5:
                            position_patterns[(layer_idx, head_idx)]['local'] += attn_val
                        else:
                            position_patterns[(layer_idx, head_idx)]['distant'] += attn_val

                # 2. Token-type attention patterns
                for i in range(len(token_types)):
                    for j in range(i + 1):
                        src_type = token_types[j]
                        dst_type = token_types[i]
                        attn_val = float(head_attn[i, j])

                        token_pair_attention[(layer_idx, head_idx)][(src_type, dst_type)] += attn_val

                # 3. Bracket matching detection
                open_brackets = {'{': '}', '(': ')', '[': ']'}
                close_brackets = {'}': '{', ')': '(', ']': '['}

                bracket_stack = []
                for i, tok_type in enumerate(token_types):
                    if tok_type in open_brackets:
                        bracket_stack.append((i, tok_type))
                    elif tok_type in close_brackets and bracket_stack:
                        open_idx, open_tok = bracket_stack.pop()
                        if open_tok == close_brackets[tok_type]:
                            # Check if this head attends to matching bracket
                            attn_to_open = float(head_attn[i, open_idx])
                            if attn_to_open > 0.1:  # Threshold
                                bracket_matching.append({
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'open_type': open_tok,
                                    'close_type': tok_type,
                                    'attention': attn_to_open,
                                    'distance': i - open_idx,
                                })

        if (seq_idx + 1) % 20 == 0:
            print(f"  Processed {seq_idx + 1}/{len(sequences)} sequences")

    # Analyze results
    print(f"\n{'=' * 70}")
    print("ATTENTION PATTERN ANALYSIS RESULTS")
    print("=" * 70)

    discovered_patterns = []

    # 1. Head Specialization
    print("\n1. HEAD POSITION SPECIALIZATION")
    print("-" * 50)

    head_roles = {}
    for (layer, head), patterns in position_patterns.items():
        total = sum(patterns.values())
        if total == 0:
            continue

        normalized = {k: v / total for k, v in patterns.items()}

        # Determine specialization
        if normalized.get('self', 0) > 0.5:
            role = 'self-attention (identity)'
        elif normalized.get('prev', 0) > 0.3:
            role = 'previous token (bigram)'
        elif normalized.get('local', 0) > 0.4:
            role = 'local context (n-gram)'
        elif normalized.get('distant', 0) > 0.3:
            role = 'long-range dependencies'
        else:
            role = 'mixed'

        head_roles[(layer, head)] = {
            'role': role,
            'patterns': normalized,
        }

    # Print by layer
    for layer in range(config['num_layers']):
        print(f"\nLayer {layer}:")
        for head in range(config['num_heads']):
            if (layer, head) in head_roles:
                info = head_roles[(layer, head)]
                print(f"  Head {head}: {info['role']}")

    # 2. Token-Type Patterns
    print("\n2. TOKEN-TYPE ATTENTION PATTERNS")
    print("-" * 50)

    # Aggregate across heads
    token_attention_totals = defaultdict(float)
    for patterns in token_pair_attention.values():
        for pair, value in patterns.items():
            token_attention_totals[pair] += value

    # Find strongest patterns
    sorted_pairs = sorted(
        token_attention_totals.items(),
        key=lambda x: -x[1]
    )[:20]

    print("\nTop token-type attention pairs:")
    for (src, dst), score in sorted_pairs:
        print(f"  {src:15} -> {dst:15}: {score:.2f}")

    # 3. Bracket Matching
    print("\n3. BRACKET MATCHING HEADS")
    print("-" * 50)

    if bracket_matching:
        # Find heads that consistently match brackets
        head_bracket_scores = defaultdict(list)
        for match in bracket_matching:
            head_bracket_scores[(match['layer'], match['head'])].append(match['attention'])

        print("\nHeads with bracket matching behavior:")
        for (layer, head), scores in sorted(head_bracket_scores.items()):
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.15 and len(scores) > 5:
                print(f"  Layer {layer}, Head {head}: avg_attn={avg_score:.3f}, count={len(scores)}")

                discovered_patterns.append(AttentionPattern(
                    name=f"bracket_matcher_L{layer}H{head}",
                    description=f"Attends to matching open brackets",
                    head_indices=[(layer, head)],
                    score=avg_score,
                    examples=[f"distance={m['distance']}" for m in bracket_matching[:3]],
                ))
    else:
        print("  No strong bracket matching patterns detected")

    # 4. Syntax-Aware Patterns
    print("\n4. SYNTAX-AWARE PATTERNS")
    print("-" * 50)

    # Check for function definition patterns
    func_patterns = []
    for (layer, head), patterns in token_pair_attention.items():
        func_score = patterns.get(('func', 'IDENT'), 0) + patterns.get(('IDENT', 'func'), 0)
        if func_score > 10:
            func_patterns.append((layer, head, func_score))

    if func_patterns:
        print("\nFunction definition tracking:")
        for layer, head, score in sorted(func_patterns, key=lambda x: -x[2])[:5]:
            print(f"  Layer {layer}, Head {head}: score={score:.2f}")
            discovered_patterns.append(AttentionPattern(
                name=f"func_tracker_L{layer}H{head}",
                description="Tracks function definition context",
                head_indices=[(layer, head)],
                score=min(1.0, score / 100),
                examples=["func -> IDENT attention"],
            ))

    # Check for statement boundary patterns
    stmt_patterns = []
    for (layer, head), patterns in token_pair_attention.items():
        stmt_score = (
            patterns.get((';', 'if'), 0) +
            patterns.get((';', 'for'), 0) +
            patterns.get((';', 'return'), 0) +
            patterns.get(('}', 'if'), 0) +
            patterns.get(('}', 'for'), 0)
        )
        if stmt_score > 5:
            stmt_patterns.append((layer, head, stmt_score))

    if stmt_patterns:
        print("\nStatement boundary tracking:")
        for layer, head, score in sorted(stmt_patterns, key=lambda x: -x[2])[:5]:
            print(f"  Layer {layer}, Head {head}: score={score:.2f}")
            discovered_patterns.append(AttentionPattern(
                name=f"stmt_boundary_L{layer}H{head}",
                description="Tracks statement boundaries (;, })",
                head_indices=[(layer, head)],
                score=min(1.0, score / 50),
                examples=["semicolon/brace -> keyword attention"],
            ))

    # Save results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print("=" * 70)

    # Save head roles
    head_roles_serializable = {
        f"L{layer}H{head}": info
        for (layer, head), info in head_roles.items()
    }
    roles_path = os.path.join(run_dir, 'attention_head_roles.json')
    with open(roles_path, 'w') as f:
        json.dump(head_roles_serializable, f, indent=2)
    print(f"\nHead roles: {roles_path}")

    # Save token attention patterns
    token_patterns_serializable = {
        f"{src}->{dst}": score
        for (src, dst), score in sorted_pairs
    }
    patterns_path = os.path.join(run_dir, 'attention_token_patterns.json')
    with open(patterns_path, 'w') as f:
        json.dump(token_patterns_serializable, f, indent=2)
    print(f"Token patterns: {patterns_path}")

    # Save discovered patterns
    discovered_serializable = [
        {
            'name': p.name,
            'description': p.description,
            'heads': [f"L{l}H{h}" for l, h in p.head_indices],
            'score': p.score,
            'examples': p.examples,
        }
        for p in discovered_patterns
    ]
    discovered_path = os.path.join(run_dir, 'attention_discovered_patterns.json')
    with open(discovered_path, 'w') as f:
        json.dump(discovered_serializable, f, indent=2)
    print(f"Discovered patterns: {discovered_path}")

    # Summary report
    report_lines = [
        "=" * 70,
        "ATTENTION PATTERN ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Model: {config['num_layers']} layers x {config['num_heads']} heads",
        f"Sequences analyzed: {len(sequences)}",
        "",
        "HEAD SPECIALIZATION",
        "-" * 50,
    ]

    for layer in range(config['num_layers']):
        report_lines.append(f"\nLayer {layer}:")
        for head in range(config['num_heads']):
            if (layer, head) in head_roles:
                info = head_roles[(layer, head)]
                report_lines.append(f"  Head {head}: {info['role']}")

    report_lines.extend([
        "",
        "DISCOVERED PATTERNS",
        "-" * 50,
    ])

    for p in discovered_patterns:
        report_lines.append(f"\n{p.name}:")
        report_lines.append(f"  {p.description}")
        report_lines.append(f"  Score: {p.score:.3f}")
        report_lines.append(f"  Heads: {', '.join(f'L{l}H{h}' for l, h in p.head_indices)}")

    report_path = os.path.join(run_dir, 'attention_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Report: {report_path}")

    print(f"\n{'=' * 70}")
    print("ATTENTION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Heads analyzed: {config['num_layers'] * config['num_heads']}")
    print(f"  Patterns discovered: {len(discovered_patterns)}")

    return discovered_patterns


def main():
    parser = argparse.ArgumentParser(
        description='Analyze attention patterns in trained model'
    )
    parser.add_argument(
        '--run-dir', type=str, required=True,
        help='Run directory containing config.json and sequences.json'
    )
    parser.add_argument(
        '--max-sequences', type=int, default=100,
        help='Maximum sequences to analyze'
    )

    args = parser.parse_args()
    analyze_attention(args.run_dir, args.max_sequences)


if __name__ == '__main__':
    main()
