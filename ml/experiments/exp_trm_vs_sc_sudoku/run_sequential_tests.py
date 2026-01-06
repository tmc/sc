"""
Sequential test runner: Run Vanilla TRM and SC-TRM one at a time.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import argparse
import gc
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sc_trm_hybrid import SCTRMHybrid, SCTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sc_trm_simple import SimpleSCTRM, SimpleSCConfig
from experiments.exp_trm_vs_sc_sudoku.sc_trm_enhanced import EnhancedSCTRM, EnhancedSCConfig
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku


def get_memory_mb():
    """Get current MLX memory usage in MB."""
    try:
        # Use new API if available, fall back to deprecated
        if hasattr(mx, 'get_active_memory'):
            active = mx.get_active_memory() / (1024 * 1024)
            peak = mx.get_peak_memory() / (1024 * 1024)
        else:
            active = mx.metal.get_active_memory() / (1024 * 1024)
            peak = mx.metal.get_peak_memory() / (1024 * 1024)
        return active, peak
    except Exception:
        return 0, 0


def clear_cache():
    """Clear MLX cache using new or old API."""
    try:
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def print_memory(label=""):
    """Print current memory usage."""
    active, peak = get_memory_mb()
    print(f"  [MEM {label}] Active: {active:.1f}MB, Peak: {peak:.1f}MB")


def clear_memory():
    """Clear MLX cache and run garbage collection."""
    clear_cache()
    gc.collect()


def count_params(params):
    """Count parameters recursively."""
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += count_params(v)
    elif isinstance(params, mx.array):
        total += params.size
    elif isinstance(params, (list, tuple)):
        for v in params:
            total += count_params(v)
    return total


def train_model(model, train_p, train_s, test_p, test_s, epochs=30, batch_size=32, lr=1e-4):
    """Train a model and return metrics."""
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print_memory("before training")

    start = time.time()
    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, train_p.shape[0], batch_size):
            p = train_p_shuf[i:i+batch_size]
            s = train_s_shuf[i:i+batch_size]

            # Print memory before forward pass
            if n_batches == 0:
                print_memory(f"epoch {epoch} start")

            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss.item())
            n_batches += 1

            # Print and clear memory every 3 batches
            if n_batches % 3 == 0:
                active, _ = get_memory_mb()
                print(f"    batch {n_batches}: loss={float(loss.item()):.4f}, mem={active:.0f}MB")
                clear_memory()

        # Print memory at each epoch
        active, peak = get_memory_mb()

        if epoch % 5 == 0:
            # Test accuracy on subset
            result = model.solve(test_p[:50])
            pred = result['predictions']
            mx.eval(pred)
            correct = mx.all(pred == test_s[:50], axis=1)
            acc = float(mx.mean(correct.astype(mx.float32)).item())
            cell_acc = float(mx.mean((pred == test_s[:50]).astype(mx.float32)).item())
            print(f'  Epoch {epoch}: loss={total_loss/n_batches:.4f}, '
                  f'test_acc={acc:.1%}, cell_acc={cell_acc:.1%}, mem={active:.0f}MB')
            clear_memory()
        else:
            print(f'  Epoch {epoch}: loss={total_loss/n_batches:.4f}, mem={active:.0f}MB')
            clear_memory()

    elapsed = time.time() - start

    # Final evaluation
    result = model.solve(test_p)
    pred = result['predictions']
    correct = mx.all(pred == test_s, axis=1)
    exact_acc = float(mx.mean(correct.astype(mx.float32)).item())
    cell_acc = float(mx.mean((pred == test_s).astype(mx.float32)).item())

    return {
        'exact_acc': exact_acc,
        'cell_acc': cell_acc,
        'time': elapsed,
    }


def test_vanilla_trm(train_p, train_s, test_p, test_s, epochs=30):
    """Test vanilla TRM."""
    print("\n" + "=" * 60)
    print("TEST 1: VANILLA TRM")
    print("=" * 60)

    config = VanillaTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
    )

    model = VanillaTRM(config)
    params = count_params(model.parameters())
    print(f"Parameters: {params:,}")

    metrics = train_model(model, train_p, train_s, test_p, test_s, epochs=epochs)

    print(f"\nFINAL RESULTS:")
    print(f"  Exact accuracy: {metrics['exact_acc']:.1%}")
    print(f"  Cell accuracy: {metrics['cell_acc']:.1%}")
    print(f"  Training time: {metrics['time']:.1f}s")

    return metrics


def test_sc_trm(train_p, train_s, test_p, test_s, epochs=30):
    """Test SC-TRM hybrid."""
    print("\n" + "=" * 60)
    print("TEST 2: SC-TRM HYBRID")
    print("=" * 60)

    config = SCTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        use_constraint_guards=True,  # Enable guards
        guard_temperature=0.5,       # Softer temperature for learning
    )

    model = SCTRMHybrid(config)
    params = count_params(model.parameters())
    print(f"Parameters: {params:,}")

    metrics = train_model(model, train_p, train_s, test_p, test_s, epochs=epochs)

    print(f"\nFINAL RESULTS:")
    print(f"  Exact accuracy: {metrics['exact_acc']:.1%}")
    print(f"  Cell accuracy: {metrics['cell_acc']:.1%}")
    print(f"  Training time: {metrics['time']:.1f}s")

    return metrics


def test_simple_sc_trm(train_p, train_s, test_p, test_s, epochs=30, constraint_weight=0.1,
                       use_inference_guards=False):
    """Test simplified SC-TRM (vanilla TRM + constraint loss)."""
    print("\n" + "=" * 60)
    print(f"TEST 3: SIMPLE SC-TRM (weight={constraint_weight}, guards={use_inference_guards})")
    print("=" * 60)

    config = SimpleSCConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        constraint_loss_weight=constraint_weight,
        use_inference_guards=use_inference_guards,
    )

    model = SimpleSCTRM(config)
    params = count_params(model.parameters())
    print(f"Parameters: {params:,}")

    metrics = train_model(model, train_p, train_s, test_p, test_s, epochs=epochs)

    print(f"\nFINAL RESULTS:")
    print(f"  Exact accuracy: {metrics['exact_acc']:.1%}")
    print(f"  Cell accuracy: {metrics['cell_acc']:.1%}")
    print(f"  Training time: {metrics['time']:.1f}s")

    return metrics


def test_enhanced_sc_trm(train_p, train_s, test_p, test_s, epochs=30, method='iterative'):
    """Test enhanced SC-TRM with iterative/multi-pass methods."""
    print("\n" + "=" * 60)
    print(f"TEST 4: ENHANCED SC-TRM (method={method})")
    print("=" * 60)

    config = EnhancedSCConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        iterative_guards=(method == 'iterative'),
        multi_pass=3 if method == 'multi_pass' else 1,
        constraint_propagation=True,
        confidence_threshold=0.8,
    )

    model = EnhancedSCTRM(config)
    params = count_params(model.parameters())
    print(f"Parameters: {params:,}")
    print(f"Method: {method}")

    # Train with standard loss (no guards during training)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def loss_fn(m, p, s):
        # Use model's loss (which uses raw solve without guards)
        loss, _ = m.loss(p, s)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print_memory("before training")

    start = time.time()
    batch_size = 32
    for epoch in range(1, epochs + 1):
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, train_p.shape[0], batch_size):
            p = train_p_shuf[i:i+batch_size]
            s = train_s_shuf[i:i+batch_size]

            if n_batches == 0:
                print_memory(f"epoch {epoch} start")

            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss.item())
            n_batches += 1

            if n_batches % 3 == 0:
                active, _ = get_memory_mb()
                print(f"    batch {n_batches}: loss={float(loss.item()):.4f}, mem={active:.0f}MB")
                clear_memory()

        active, peak = get_memory_mb()

        if epoch % 5 == 0:
            # Test with enhanced solve
            result = model.solve(test_p[:50], method=method)
            pred = result['predictions']
            mx.eval(pred)
            correct = mx.all(pred == test_s[:50], axis=1)
            acc = float(mx.mean(correct.astype(mx.float32)).item())
            cell_acc = float(mx.mean((pred == test_s[:50]).astype(mx.float32)).item())
            print(f'  Epoch {epoch}: loss={total_loss/n_batches:.4f}, '
                  f'test_acc={acc:.1%}, cell_acc={cell_acc:.1%}, mem={active:.0f}MB')
            clear_memory()
        else:
            print(f'  Epoch {epoch}: loss={total_loss/n_batches:.4f}, mem={active:.0f}MB')
            clear_memory()

    elapsed = time.time() - start

    # Final evaluation with enhanced solve
    result = model.solve(test_p, method=method)
    pred = result['predictions']
    correct = mx.all(pred == test_s, axis=1)
    exact_acc = float(mx.mean(correct.astype(mx.float32)).item())
    cell_acc = float(mx.mean((pred == test_s).astype(mx.float32)).item())

    print(f"\nFINAL RESULTS:")
    print(f"  Exact accuracy: {exact_acc:.1%}")
    print(f"  Cell accuracy: {cell_acc:.1%}")
    print(f"  Training time: {elapsed:.1f}s")

    return {
        'exact_acc': exact_acc,
        'cell_acc': cell_acc,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Run TRM vs SC tests sequentially')
    parser.add_argument('--test', choices=['vanilla', 'sc', 'simple', 'enhanced', 'iterative', 'multipass', 'all'], default='all',
                       help='Which test to run')
    parser.add_argument('--train-size', type=int, default=1000,
                       help='Number of training puzzles')
    parser.add_argument('--test-size', type=int, default=200,
                       help='Number of test puzzles')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--constraint-weight', type=float, default=0.1,
                       help='Constraint loss weight for simple SC-TRM')
    parser.add_argument('--inference-guards', action='store_true',
                       help='Enable inference-time guards for simple SC-TRM')
    args = parser.parse_args()

    print("=" * 60)
    print("TRM vs SC-TRM SEQUENTIAL TESTS")
    print("=" * 60)
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Epochs: {args.epochs}")

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(
        args.train_size, args.test_size, seed=args.seed
    )

    results = {}

    if args.test in ['vanilla', 'all']:
        results['vanilla'] = test_vanilla_trm(train_p, train_s, test_p, test_s, epochs=args.epochs)

    if args.test in ['sc', 'all']:
        results['sc'] = test_sc_trm(train_p, train_s, test_p, test_s, epochs=args.epochs)

    if args.test in ['simple', 'all']:
        results['simple'] = test_simple_sc_trm(train_p, train_s, test_p, test_s, epochs=args.epochs,
                                               constraint_weight=args.constraint_weight,
                                               use_inference_guards=args.inference_guards)

    if args.test in ['enhanced', 'iterative']:
        results['iterative'] = test_enhanced_sc_trm(train_p, train_s, test_p, test_s,
                                                    epochs=args.epochs, method='iterative')

    if args.test in ['enhanced', 'multipass']:
        results['multipass'] = test_enhanced_sc_trm(train_p, train_s, test_p, test_s,
                                                    epochs=args.epochs, method='multi_pass')

    # Summary
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        # Build header dynamically
        header = f"{'Metric':<20}"
        for name in results.keys():
            header += f" {name:>12}"
        print(header)
        print("-" * (20 + 13 * len(results)))

        # Exact accuracy
        row = f"{'Exact Accuracy':<20}"
        for r in results.values():
            row += f" {r['exact_acc']:>11.1%}"
        print(row)

        # Cell accuracy
        row = f"{'Cell Accuracy':<20}"
        for r in results.values():
            row += f" {r['cell_acc']:>11.1%}"
        print(row)

        # Training time
        row = f"{'Training Time':<20}"
        for r in results.values():
            row += f" {r['time']:>10.1f}s"
        print(row)

        # Improvements vs vanilla
        if 'vanilla' in results:
            print(f"\nImprovements vs Vanilla TRM:")
            for name, r in results.items():
                if name != 'vanilla':
                    diff = r['cell_acc'] - results['vanilla']['cell_acc']
                    print(f"  {name}: {diff:+.1%} cell accuracy")

    print("\nDone!")


if __name__ == "__main__":
    main()
