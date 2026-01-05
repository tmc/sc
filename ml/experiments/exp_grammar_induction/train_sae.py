#!/usr/bin/env python3
"""
SAE Training on Transformer Hidden States

Trains a TopK Sparse Autoencoder on the hidden states from the
trained transformer to discover discrete syntax contexts.

Usage:
    python -m experiments.exp_grammar_induction.train_sae --run-dir runs/run_20260104_154119
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .token_collector import GoTokenCollector, TOKEN_VOCAB
from .sequence_model import GoSyntaxTransformer, TransformerConfig, SyntaxTrainer
from .sae_syntax import SyntaxSAE, SAEConfig, GoProductionRule

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


def train_sae(run_dir: str, sae_epochs: int = 10):
    """Train SAE on saved transformer hidden states."""

    print("=" * 70)
    print("SAE TRAINING ON TRANSFORMER REPRESENTATIONS")
    print("=" * 70)
    print(f"\nRun directory: {run_dir}")
    print(f"MLX available: {HAS_MLX}")

    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nLoaded config:")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  SAE expansion: {config['sae_expansion']}")
    print(f"  SAE k_active: {config['sae_k_active']}")

    # Load sequences
    sequences_path = os.path.join(run_dir, 'sequences.json')
    collector = GoTokenCollector()
    collector.load(sequences_path)

    print(f"\nLoaded {len(collector.sequences)} sequences")

    train_seqs, val_seqs, _ = collector.split()
    print(f"Training on {len(train_seqs)} sequences")

    if not HAS_MLX:
        print("\nERROR: MLX required for SAE training")
        return None

    # Create transformer (to get hidden states)
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
    print(f"\nCreated transformer for hidden state extraction")

    # Create SAE
    sae_config = SAEConfig(
        input_dim=config['hidden_dim'],
        expansion_factor=config['sae_expansion'],
        k_active=config['sae_k_active'],
    )

    sae = SyntaxSAE(sae_config)
    print(f"\nCreated SAE:")
    print(f"  Input dim: {sae_config.input_dim}")
    print(f"  Latent dim: {sae_config.input_dim * sae_config.expansion_factor}")
    print(f"  K active: {sae_config.k_active}")

    # Create optimizer for SAE (using the underlying TopKSAE module)
    optimizer = optim.Adam(learning_rate=config.get('sae_lr', 1e-3))

    # Define loss function for gradient computation
    def loss_fn(sae_model, x):
        loss, _ = sae_model.compute_loss(x)
        return loss

    # Create value_and_grad function
    loss_and_grad_fn = nn.value_and_grad(sae.sae, loss_fn)

    # Training loop
    print(f"\n{'=' * 70}")
    print(f"TRAINING SAE FOR {sae_epochs} EPOCHS")
    print("=" * 70)

    training_log = []

    for epoch in range(sae_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_recon = 0.0
        total_sparsity = 0.0
        n_batches = 0

        for seq_idx, seq in enumerate(train_seqs[:50]):  # Reduced for speed
            if seq_idx % 10 == 0:
                import sys
                print(f"    Processing sequence {seq_idx+1}/50...", flush=True)
                sys.stdout.flush()
            token_ids = seq.to_type_ids()
            if len(token_ids) < 10:
                continue

            token_ids = token_ids[:config['max_seq_len']]
            batch = mx.array([token_ids])

            # Get hidden states from transformer
            hidden = transformer.get_hidden_states(batch)
            token_types = [t.type_name for t in seq.tokens[:hidden.shape[1]]]

            # Flatten for SAE
            batch_size, seq_len, dim = hidden.shape
            x = hidden.reshape(-1, dim)

            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(sae.sae, x)

            # Update SAE weights
            optimizer.update(sae.sae, grads)
            mx.eval(sae.sae.parameters())

            # Get metrics
            _, metrics = sae.sae.compute_loss(x)

            # Track feature associations (without gradient tracking)
            _, acts, indices = sae.sae(x)
            for b in range(min(x.shape[0], 100)):  # Limit for speed
                active_indices = [int(indices[b, i].item()) for i in range(sae_config.k_active)]
                for idx in active_indices:
                    sae.features[idx].activation_count += 1
                if token_types:
                    token_idx = b % len(token_types)
                    token = token_types[token_idx]
                    for idx in active_indices:
                        sae.features[idx].associated_tokens.add(token)

            total_loss += float(loss)
            total_recon += metrics.get('reconstruction_loss', 0)
            total_sparsity += metrics.get('sparsity_loss', 0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_sparsity = total_sparsity / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        log_entry = {
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'reconstruction_loss': avg_recon,
            'sparsity_loss': avg_sparsity,
            'time': epoch_time,
        }
        training_log.append(log_entry)

        # Count active features
        active_features = sum(1 for f in sae.features.values() if f.activation_count > 0)

        print(f"Epoch {epoch+1:2d}/{sae_epochs}: "
              f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, "
              f"sparsity={avg_sparsity:.4f}, active_feats={active_features}, "
              f"time={epoch_time:.1f}s")

    # Discover states
    print(f"\n{'=' * 70}")
    print("DISCOVERING SYNTAX STATES")
    print("=" * 70)

    states = sae.discover_states(min_activation_count=50)
    print(f"\nDiscovered {len(states)} syntax states:")

    for state_name, features in sorted(states.items()):
        feature_tokens = []
        for fid in list(features)[:3]:
            tokens = list(sae.features[fid].associated_tokens)[:3]
            feature_tokens.extend(tokens)
        print(f"  {state_name}: {len(features)} features, tokens: {feature_tokens[:5]}")

    # Map to production rules
    token_rule_map = {
        'func': GoProductionRule.FUNC_DECL,
        'type': GoProductionRule.TYPE_DECL,
        'const': GoProductionRule.CONST_DECL,
        'var': GoProductionRule.VAR_DECL,
        'if': GoProductionRule.IF_STMT,
        'for': GoProductionRule.FOR_STMT,
        'switch': GoProductionRule.SWITCH_STMT,
        'select': GoProductionRule.SELECT_STMT,
        'return': GoProductionRule.RETURN_STMT,
        'go': GoProductionRule.GO_STMT,
        'defer': GoProductionRule.DEFER_STMT,
        '{': GoProductionRule.BLOCK,
        '}': GoProductionRule.BLOCK,
        'struct': GoProductionRule.STRUCT_TYPE,
        'interface': GoProductionRule.INTERFACE_TYPE,
        'map': GoProductionRule.MAP_TYPE,
        'chan': GoProductionRule.CHANNEL_TYPE,
        'package': GoProductionRule.PACKAGE_CLAUSE,
        'import': GoProductionRule.IMPORT_DECL,
    }

    feature_rules = sae.map_to_production_rules(token_rule_map)
    print(f"\nMapped {len(feature_rules)} features to production rules")

    # Save outputs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Save training log
    log_path = os.path.join(run_dir, 'sae_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nSAE training log: {log_path}")

    # Save feature descriptions
    features_path = os.path.join(run_dir, 'sae_features.txt')
    with open(features_path, 'w') as f:
        f.write(sae.describe_features(50))
    print(f"SAE features: {features_path}")

    # Save states
    states_data = {
        name: list(features)
        for name, features in states.items()
    }
    states_path = os.path.join(run_dir, 'sae_states.json')
    with open(states_path, 'w') as f:
        json.dump(states_data, f, indent=2)
    print(f"SAE states: {states_path}")

    # Summary statistics
    total_activations = sum(f.activation_count for f in sae.features.values())
    active_features = sum(1 for f in sae.features.values() if f.activation_count > 0)
    dead_features = sum(1 for f in sae.features.values() if f.activation_count == 0)

    summary = {
        'total_features': len(sae.features),
        'active_features': active_features,
        'dead_features': dead_features,
        'total_activations': total_activations,
        'num_states': len(states),
        'features_mapped_to_rules': len(feature_rules),
    }

    summary_path = os.path.join(run_dir, 'sae_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"SAE summary: {summary_path}")

    print(f"\n{'=' * 70}")
    print("SAE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total features: {summary['total_features']}")
    print(f"  Active features: {summary['active_features']} ({100*active_features/len(sae.features):.1f}%)")
    print(f"  Dead features: {summary['dead_features']} ({100*dead_features/len(sae.features):.1f}%)")
    print(f"  Discovered states: {summary['num_states']}")

    return sae


def main():
    parser = argparse.ArgumentParser(
        description='Train SAE on transformer hidden states'
    )
    parser.add_argument(
        '--run-dir', type=str, required=True,
        help='Run directory containing config.json and sequences.json'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of SAE training epochs'
    )

    args = parser.parse_args()
    train_sae(args.run_dir, args.epochs)


if __name__ == '__main__':
    main()
