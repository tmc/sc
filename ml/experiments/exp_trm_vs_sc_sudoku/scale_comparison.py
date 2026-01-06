"""
Scaling experiment: Test SC methods at different scales.

Compare vanilla TRM vs SC-TRM at:
1. Small scale (current): 1K samples, 128 hidden, 30 epochs
2. Medium scale: 5K samples, 256 hidden, 50 epochs
3. Larger scale: 10K samples, 256 hidden, 100 epochs

Hypothesis: SC inference guards provide consistent improvement across scales.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import argparse
import gc
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sc_trm_simple import SimpleSCTRM, SimpleSCConfig
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku


def clear_cache():
    """Clear MLX cache."""
    try:
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def train_and_evaluate(model, train_p, train_s, test_p, test_s, epochs, batch_size, lr,
                       use_guards=False, print_every=10):
    """Train model and return metrics."""
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    start = time.time()
    best_cell_acc = 0.0

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

            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += float(loss.item())
            n_batches += 1

        # Periodic evaluation
        if epoch % print_every == 0 or epoch == epochs:
            # Evaluate
            if hasattr(model, 'solve'):
                if use_guards:
                    result = model.solve(test_p[:100], use_guards=True)
                else:
                    result = model.solve(test_p[:100])
            else:
                result = model.solve(test_p[:100])

            pred = result['predictions']
            mx.eval(pred)

            cell_acc = float(mx.mean((pred == test_s[:100]).astype(mx.float32)).item())
            exact_acc = float(mx.mean(mx.all(pred == test_s[:100], axis=1).astype(mx.float32)).item())
            best_cell_acc = max(best_cell_acc, cell_acc)

            print(f"  Epoch {epoch:4d}: loss={total_loss/n_batches:.4f}, "
                  f"cell_acc={cell_acc:.1%}, exact_acc={exact_acc:.1%}")

            clear_cache()

    elapsed = time.time() - start

    # Final evaluation on full test set
    if hasattr(model, 'solve'):
        if use_guards:
            result = model.solve(test_p, use_guards=True)
        else:
            result = model.solve(test_p)
    else:
        result = model.solve(test_p)

    pred = result['predictions']
    mx.eval(pred)

    final_cell_acc = float(mx.mean((pred == test_s).astype(mx.float32)).item())
    final_exact_acc = float(mx.mean(mx.all(pred == test_s, axis=1).astype(mx.float32)).item())

    return {
        'cell_acc': final_cell_acc,
        'exact_acc': final_exact_acc,
        'best_cell_acc': best_cell_acc,
        'time': elapsed,
    }


def run_scale_experiment(scale='small'):
    """Run experiment at specified scale."""
    scales = {
        'small': {
            'train_size': 1000,
            'test_size': 200,
            'hidden_dim': 128,
            'num_layers': 3,
            'epochs': 30,
            'batch_size': 32,
            'lr': 1e-4,
        },
        'medium': {
            'train_size': 5000,
            'test_size': 500,
            'hidden_dim': 256,
            'num_layers': 4,
            'epochs': 50,
            'batch_size': 64,
            'lr': 1e-4,
        },
        'large': {
            'train_size': 10000,
            'test_size': 1000,
            'hidden_dim': 256,
            'num_layers': 4,
            'epochs': 100,
            'batch_size': 64,
            'lr': 1e-4,
        },
    }

    config = scales[scale]

    print("=" * 70)
    print(f"SCALE EXPERIMENT: {scale.upper()}")
    print("=" * 70)
    print(f"Train size: {config['train_size']}")
    print(f"Test size: {config['test_size']}")
    print(f"Hidden dim: {config['hidden_dim']}")
    print(f"Num layers: {config['num_layers']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(
        config['train_size'], config['test_size'], seed=42
    )
    print(f"Generated {train_p.shape[0]} train, {test_p.shape[0]} test puzzles")

    results = {}

    # 1. Vanilla TRM
    print("\n" + "-" * 70)
    print("1. VANILLA TRM")
    print("-" * 70)

    vanilla_config = VanillaTRMConfig(
        hidden_dim=config['hidden_dim'],
        num_heads=4,  # Match earlier successful tests
        num_layers=config['num_layers'],
        ff_dim=config['hidden_dim'] * 2,
        H_cycles=3,
        L_cycles=4,  # Match earlier successful tests
    )
    vanilla_model = VanillaTRM(vanilla_config)

    def count_params(m):
        total = 0
        for p in m.parameters().values():
            if isinstance(p, mx.array):
                total += p.size
            elif isinstance(p, dict):
                for v in p.values():
                    if isinstance(v, mx.array):
                        total += v.size
            elif isinstance(p, list):
                for v in p:
                    if isinstance(v, mx.array):
                        total += v.size
        return total

    print(f"Parameters: ~{count_params(vanilla_model):,}")

    results['vanilla'] = train_and_evaluate(
        vanilla_model, train_p, train_s, test_p, test_s,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        use_guards=False,
        print_every=max(1, config['epochs'] // 10),
    )

    clear_cache()
    del vanilla_model

    # 2. Simple SC-TRM without guards (baseline)
    print("\n" + "-" * 70)
    print("2. SIMPLE SC-TRM (no guards)")
    print("-" * 70)

    sc_config = SimpleSCConfig(
        hidden_dim=config['hidden_dim'],
        num_heads=4,
        num_layers=config['num_layers'],
        ff_dim=config['hidden_dim'] * 2,
        H_cycles=3,
        L_cycles=4,
        constraint_loss_weight=0.0,
        use_inference_guards=False,
    )
    sc_model_no_guards = SimpleSCTRM(sc_config)

    results['sc_no_guards'] = train_and_evaluate(
        sc_model_no_guards, train_p, train_s, test_p, test_s,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        use_guards=False,
        print_every=max(1, config['epochs'] // 10),
    )

    clear_cache()
    del sc_model_no_guards

    # 3. Simple SC-TRM with inference guards
    print("\n" + "-" * 70)
    print("3. SIMPLE SC-TRM (with inference guards)")
    print("-" * 70)

    sc_config_guards = SimpleSCConfig(
        hidden_dim=config['hidden_dim'],
        num_heads=4,
        num_layers=config['num_layers'],
        ff_dim=config['hidden_dim'] * 2,
        H_cycles=3,
        L_cycles=4,
        constraint_loss_weight=0.0,  # No constraint loss (proven to hurt)
        use_inference_guards=True,   # Only inference-time guards
    )
    sc_model_guards = SimpleSCTRM(sc_config_guards)

    results['sc_guards'] = train_and_evaluate(
        sc_model_guards, train_p, train_s, test_p, test_s,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        use_guards=True,
        print_every=max(1, config['epochs'] // 10),
    )

    clear_cache()
    del sc_model_guards

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY ({scale.upper()} SCALE)")
    print("=" * 70)
    print(f"{'Model':<25} {'Cell Acc':>10} {'Exact Acc':>10} {'Time':>10}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{name:<25} {r['cell_acc']:>9.1%} {r['exact_acc']:>10.1%} {r['time']:>9.1f}s")

    print("-" * 70)

    # Improvement from guards
    if 'vanilla' in results and 'sc_guards' in results:
        improvement = results['sc_guards']['cell_acc'] - results['vanilla']['cell_acc']
        print(f"\nSC Guards improvement: {improvement:+.1%} cell accuracy")

    return results


def main():
    parser = argparse.ArgumentParser(description='Scale comparison experiment')
    parser.add_argument('--scale', choices=['small', 'medium', 'large', 'all'], default='small',
                       help='Scale to test')
    args = parser.parse_args()

    if args.scale == 'all':
        all_results = {}
        for scale in ['small', 'medium', 'large']:
            all_results[scale] = run_scale_experiment(scale)
            print("\n\n")

        # Final comparison
        print("=" * 70)
        print("CROSS-SCALE COMPARISON")
        print("=" * 70)
        print(f"{'Scale':<10} {'Vanilla':>12} {'SC Guards':>12} {'Improvement':>12}")
        print("-" * 70)
        for scale, results in all_results.items():
            vanilla = results['vanilla']['cell_acc']
            sc_guards = results['sc_guards']['cell_acc']
            improvement = sc_guards - vanilla
            print(f"{scale:<10} {vanilla:>11.1%} {sc_guards:>12.1%} {improvement:>+11.1%}")
    else:
        run_scale_experiment(args.scale)


if __name__ == "__main__":
    main()
