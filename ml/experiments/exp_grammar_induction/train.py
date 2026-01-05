#!/usr/bin/env python3
"""
Full Training Run for Grammar Induction

This script runs the complete pipeline:
1. Collect tokens from Go stdlib (200+ files)
2. Train transformer on token sequences
3. Apply SAE to discover syntax states
4. Build statechart from discovered patterns
5. Benchmark against go/parser oracle

Usage:
    python -m experiments.exp_grammar_induction.train
    python -m experiments.exp_grammar_induction.train --quick  # Fast test run
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
from .grammar_inducer import GrammarInducer, InducedGrammar
from .go_parser_oracle import GoParserOracle
from .benchmark import GrammarBenchmark, BenchmarkType

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# Default hyperparameters
DEFAULT_CONFIG = {
    # Data collection
    'max_files': 500,
    'include_stdlib': True,
    'stdlib_packages': [
        'fmt', 'io', 'os', 'net', 'http', 'strings', 'bytes',
        'strconv', 'sort', 'sync', 'context', 'errors', 'time',
        'encoding/json', 'encoding/xml', 'path/filepath',
        'regexp', 'bufio', 'container/list', 'container/heap',
        'crypto', 'database/sql', 'flag', 'log', 'math',
        'reflect', 'runtime', 'testing', 'text/template',
    ],
    
    # Transformer
    'hidden_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'ff_dim': 1024,
    'max_seq_len': 512,
    'dropout': 0.1,
    
    # Training
    'transformer_epochs': 20,
    'transformer_lr': 1e-4,
    'batch_size': 32,
    
    # SAE
    'sae_expansion': 16,
    'sae_k_active': 32,
    'sae_epochs': 10,
    'sae_lr': 1e-3,
    
    # Benchmark thresholds
    'threshold_state_coverage': 0.70,
    'threshold_transition_accuracy': 0.80,
    'threshold_accept_reject': 0.95,
    'threshold_edge_cases': 0.85,
}

# Quick test config
QUICK_CONFIG = {
    'max_files': 50,
    'include_stdlib': True,
    'stdlib_packages': ['fmt', 'io', 'os', 'strings'],
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'ff_dim': 256,
    'max_seq_len': 128,
    'dropout': 0.1,
    'transformer_epochs': 5,
    'transformer_lr': 1e-3,
    'batch_size': 16,
    'sae_expansion': 8,
    'sae_k_active': 16,
    'sae_epochs': 3,
    'sae_lr': 1e-3,
    'threshold_state_coverage': 0.50,
    'threshold_transition_accuracy': 0.60,
    'threshold_accept_reject': 0.80,
    'threshold_edge_cases': 0.75,
}


def run_training(config: dict, output_dir: str):
    """Run the full training pipeline."""
    
    start_time = time.time()
    
    print("=" * 70)
    print("GRAMMAR INDUCTION - FULL TRAINING RUN")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Output dir: {output_dir}")
    print(f"MLX available: {HAS_MLX}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")
    
    # =========================================================================
    # STEP 1: Token Collection
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: TOKEN COLLECTION")
    print("=" * 70)
    
    collector = GoTokenCollector()
    
    print(f"\nCollecting from Go stdlib ({len(config['stdlib_packages'])} packages)...")
    print(f"Max files: {config['max_files']}")
    
    collector.collect(
        sources=[],  # Just stdlib for now
        include_stdlib=config['include_stdlib'],
        max_total_files=config['max_files'],
    )
    
    print(f"\nCollected:")
    print(f"  Sequences: {len(collector.sequences)}")
    print(f"  Total tokens: {sum(len(s) for s in collector.sequences):,}")
    
    # Save token stats
    token_stats = collector.get_token_stats()
    stats_path = os.path.join(output_dir, 'token_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(token_stats, f, indent=2)
    print(f"  Token stats saved to: {stats_path}")
    
    # Save sequences for reproducibility
    sequences_path = os.path.join(output_dir, 'sequences.json')
    collector.save(sequences_path)
    print(f"  Sequences saved to: {sequences_path}")
    
    # Split data
    train_seqs, val_seqs, test_seqs = collector.split()
    print(f"\nData split:")
    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Val:   {len(val_seqs)} sequences")
    print(f"  Test:  {len(test_seqs)} sequences")
    
    # =========================================================================
    # STEP 2: Transformer Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TRANSFORMER TRAINING")
    print("=" * 70)
    
    if not HAS_MLX:
        print("\nWARNING: MLX not available - skipping transformer training")
        transformer = None
        trainer = None
    else:
        # Create transformer config
        transformer_config = TransformerConfig(
            vocab_size=len(TOKEN_VOCAB),
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
        )
        
        print(f"\nTransformer config:")
        print(f"  Vocab size: {transformer_config.vocab_size}")
        print(f"  Hidden dim: {transformer_config.hidden_dim}")
        print(f"  Layers: {transformer_config.num_layers}")
        print(f"  Heads: {transformer_config.num_heads}")
        print(f"  FF dim: {transformer_config.ff_dim}")
        
        # Create model
        transformer = GoSyntaxTransformer(transformer_config)
        trainer = SyntaxTrainer(transformer, learning_rate=config['transformer_lr'])
        
        # Training loop
        print(f"\nTraining for {config['transformer_epochs']} epochs...")
        
        training_log = []
        best_val_loss = float('inf')
        
        for epoch in range(config['transformer_epochs']):
            epoch_start = time.time()
            epoch_loss = 0.0
            n_batches = 0
            
            for seq in train_seqs:
                token_ids = seq.to_type_ids()
                if len(token_ids) < 10:
                    continue
                
                # Truncate to max_seq_len
                token_ids = token_ids[:config['max_seq_len']]
                batch = mx.array([token_ids])
                
                loss = trainer.train_step(batch)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            
            # Validation
            val_loss = 0.0
            val_batches = 0
            for seq in val_seqs[:50]:  # Sample for speed
                token_ids = seq.to_type_ids()
                if len(token_ids) < 10:
                    continue
                token_ids = token_ids[:config['max_seq_len']]
                batch = mx.array([token_ids])
                
                logits, _ = transformer(batch[:, :-1])
                loss = float(transformer.compute_loss(batch))
                val_loss += loss
                val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            epoch_time = time.time() - epoch_start
            
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'time': epoch_time,
            }
            training_log.append(log_entry)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best model weights would go here
            
            print(f"  Epoch {epoch+1:2d}/{config['transformer_epochs']}: "
                  f"train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                  f"time={epoch_time:.1f}s")
        
        # Save training log
        log_path = os.path.join(output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"\nTraining log saved to: {log_path}")
    
    # =========================================================================
    # STEP 3: SAE State Discovery
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SAE STATE DISCOVERY")
    print("=" * 70)
    
    sae_config = SAEConfig(
        input_dim=config['hidden_dim'],
        expansion_factor=config['sae_expansion'],
        k_active=config['sae_k_active'],
    )
    
    print(f"\nSAE config:")
    print(f"  Input dim: {sae_config.input_dim}")
    print(f"  Expansion: {sae_config.expansion_factor}")
    print(f"  K active: {sae_config.k_active}")
    print(f"  Total features: {sae_config.input_dim * sae_config.expansion_factor}")
    
    sae = SyntaxSAE(sae_config)
    
    if HAS_MLX and transformer:
        print(f"\nTraining SAE for {config['sae_epochs']} epochs...")
        
        for epoch in range(config['sae_epochs']):
            epoch_loss = 0.0
            n_samples = 0
            
            for seq in train_seqs[:100]:  # Use subset for speed
                token_ids = seq.to_type_ids()
                if len(token_ids) < 10:
                    continue
                
                token_ids = token_ids[:config['max_seq_len']]
                batch = mx.array([token_ids])
                
                # Get hidden states from transformer
                hidden = transformer.get_hidden_states(batch)
                token_types = [t.type_name for t in seq.tokens[:hidden.shape[1]]]
                
                metrics = sae.train_step(hidden, token_types)
                epoch_loss += metrics.get('total_loss', 0)
                n_samples += 1
            
            avg_loss = epoch_loss / max(n_samples, 1)
            print(f"  Epoch {epoch+1}/{config['sae_epochs']}: loss={avg_loss:.4f}")
    else:
        print("\nRunning SAE in simulation mode...")
        # Simulate feature associations from token patterns
        for seq in train_seqs[:100]:
            for token in seq.tokens:
                feature_id = hash(token.type_name) % (sae_config.input_dim * sae_config.expansion_factor)
                sae.features[feature_id].activation_count += 1
                sae.features[feature_id].associated_tokens.add(token.type_name)
    
    # Discover states
    print("\nDiscovering syntax states...")
    states = sae.discover_states(min_activation_count=10)
    print(f"  Found {len(states)} states")
    
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
    print(f"  Mapped {len(feature_rules)} features to production rules")
    
    # Save feature descriptions
    features_path = os.path.join(output_dir, 'features.txt')
    with open(features_path, 'w') as f:
        f.write(sae.describe_features(50))
    print(f"  Features saved to: {features_path}")
    
    # =========================================================================
    # STEP 4: Build Statechart
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: BUILD STATECHART")
    print("=" * 70)
    
    # Use the GrammarInducer to build the statechart
    inducer = GrammarInducer()
    inducer.collector = collector
    inducer.sae = sae
    
    print("\nBuilding statechart from discovered states...")
    inducer.build_statechart()
    
    grammar = inducer.grammar
    grammar.name = "induced_go_grammar"
    grammar.description = f"Go grammar induced from {len(train_seqs)} files"
    grammar.total_sequences = len(collector.sequences)
    grammar.total_tokens = sum(len(s) for s in collector.sequences)
    
    print(f"\nStatechart:")
    print(f"  States: {len(grammar.root_state.children) if grammar.root_state else 0}")
    print(f"  Transitions: {len(grammar.transitions)}")
    print(f"  Events: {len(grammar.events)}")
    
    # Save statechart
    statechart_path = os.path.join(output_dir, 'grammar.json')
    with open(statechart_path, 'w') as f:
        f.write(grammar.to_json())
    print(f"  Statechart saved to: {statechart_path}")
    
    # Save mermaid diagram
    mermaid_path = os.path.join(output_dir, 'grammar.mermaid')
    with open(mermaid_path, 'w') as f:
        f.write(grammar.to_mermaid())
    print(f"  Mermaid diagram saved to: {mermaid_path}")
    
    # =========================================================================
    # STEP 5: Benchmark
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: BENCHMARK")
    print("=" * 70)
    
    benchmark = GrammarBenchmark(grammar)
    
    thresholds = {
        BenchmarkType.STATE_COVERAGE: config['threshold_state_coverage'],
        BenchmarkType.TRANSITION_ACCURACY: config['threshold_transition_accuracy'],
        BenchmarkType.ACCEPT_REJECT: config['threshold_accept_reject'],
        BenchmarkType.EDGE_CASES: config['threshold_edge_cases'],
    }
    
    results = benchmark.run_all(thresholds)
    
    # Save benchmark results
    results_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nBenchmark results saved to: {results_path}")
    
    # Save report
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(benchmark.report())
    print(f"Report saved to: {report_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}: {size:,} bytes")
    
    # Overall score
    overall_score = sum(r.score for r in results) / len(results)
    all_passed = all(r.passed for r in results)
    
    print(f"\nFinal Results:")
    print(f"  Overall Score: {overall_score:.2%}")
    print(f"  All Benchmarks Passed: {'Yes' if all_passed else 'No'}")
    
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.benchmark_type.value}: {r.score:.2%}")
    
    return grammar, results


def main():
    parser = argparse.ArgumentParser(
        description='Train grammar induction model on Go code'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test run with reduced parameters'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: runs/run_TIMESTAMP)'
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum files to process'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Transformer training epochs'
    )
    
    args = parser.parse_args()
    
    # Select config
    if args.quick:
        config = QUICK_CONFIG.copy()
        print("Using QUICK config for fast test run")
    else:
        config = DEFAULT_CONFIG.copy()
        print("Using DEFAULT config for full training")
    
    # Override with CLI args
    if args.max_files:
        config['max_files'] = args.max_files
    if args.epochs:
        config['transformer_epochs'] = args.epochs
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'runs',
            f'run_{timestamp}'
        )
    
    # Run training
    run_training(config, output_dir)


if __name__ == '__main__':
    main()
