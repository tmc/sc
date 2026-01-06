"""
Gradient flow diagnostics for TRM models.

Identifies where gradients vanish or explode.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sc_trm_hybrid import SCTRMHybrid, SCTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku


def compute_gradient_norms(grads, prefix=""):
    """Recursively compute gradient norms for all parameters."""
    norms = {}
    if isinstance(grads, dict):
        for k, v in grads.items():
            sub_norms = compute_gradient_norms(v, f"{prefix}{k}.")
            norms.update(sub_norms)
    elif isinstance(grads, mx.array):
        norm = float(mx.sqrt(mx.sum(grads * grads)).item())
        norms[prefix.rstrip('.')] = norm
    elif isinstance(grads, (list, tuple)):
        for i, v in enumerate(grads):
            sub_norms = compute_gradient_norms(v, f"{prefix}{i}.")
            norms.update(sub_norms)
    return norms


def diagnose_model(model, name, puzzle, solution):
    """Run gradient diagnostics on a model."""
    print(f"\n{'='*60}")
    print(f"GRADIENT DIAGNOSTICS: {name}")
    print(f"{'='*60}")

    def loss_fn(m, p, s):
        loss, metrics = m.loss(p, s)
        return loss

    # Compute loss and gradients
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, puzzle, solution)
    mx.eval(loss, grads)

    print(f"\nLoss: {float(loss.item()):.4f}")

    # Get gradient norms
    norms = compute_gradient_norms(grads)

    # Sort by norm (ascending to find vanishing gradients)
    sorted_norms = sorted(norms.items(), key=lambda x: x[1])

    print(f"\nGradient norms (smallest first):")
    print("-" * 50)

    # Show smallest 10
    print("\nSMALLEST (potential vanishing):")
    for name, norm in sorted_norms[:10]:
        status = "⚠️ VANISHING" if norm < 1e-6 else ""
        print(f"  {norm:12.2e}  {name} {status}")

    # Show largest 10
    print("\nLARGEST (potential exploding):")
    for name, norm in sorted_norms[-10:]:
        status = "⚠️ EXPLODING" if norm > 100 else ""
        print(f"  {norm:12.2e}  {name} {status}")

    # Summary statistics
    all_norms = list(norms.values())
    print(f"\nSUMMARY:")
    print(f"  Total parameters: {len(all_norms)}")
    print(f"  Mean gradient norm: {np.mean(all_norms):.2e}")
    print(f"  Std gradient norm: {np.std(all_norms):.2e}")
    print(f"  Min gradient norm: {np.min(all_norms):.2e}")
    print(f"  Max gradient norm: {np.max(all_norms):.2e}")

    # Count problematic gradients
    vanishing = sum(1 for n in all_norms if n < 1e-6)
    exploding = sum(1 for n in all_norms if n > 100)
    print(f"  Vanishing (<1e-6): {vanishing}")
    print(f"  Exploding (>100): {exploding}")

    return norms


def main():
    print("="*60)
    print("GRADIENT FLOW DIAGNOSTICS")
    print("="*60)

    # Generate small test batch
    print("\nGenerating test data...")
    train_p, train_s, _, _ = generate_random_sudoku(32, 8, seed=42)
    puzzle = train_p[:8]
    solution = train_s[:8]

    # Test Vanilla TRM
    print("\nInitializing Vanilla TRM...")
    vanilla_config = VanillaTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=2,
        L_cycles=2,
    )
    vanilla_model = VanillaTRM(vanilla_config)
    vanilla_norms = diagnose_model(vanilla_model, "Vanilla TRM", puzzle, solution)

    # Test SC-TRM with guards disabled
    print("\nInitializing SC-TRM (guards disabled)...")
    sc_config = SCTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=2,
        L_cycles=2,
        use_constraint_guards=False,
    )
    sc_model = SCTRMHybrid(sc_config)
    sc_norms = diagnose_model(sc_model, "SC-TRM (no guards)", puzzle, solution)

    # Test SC-TRM with guards enabled
    print("\nInitializing SC-TRM (guards enabled)...")
    sc_config_guards = SCTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=2,
        L_cycles=2,
        use_constraint_guards=True,
    )
    sc_model_guards = SCTRMHybrid(sc_config_guards)
    sc_norms_guards = diagnose_model(sc_model_guards, "SC-TRM (with guards)", puzzle, solution)

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    vanilla_mean = np.mean(list(vanilla_norms.values()))
    sc_mean = np.mean(list(sc_norms.values()))
    sc_guards_mean = np.mean(list(sc_norms_guards.values()))

    print(f"\nMean gradient norms:")
    print(f"  Vanilla TRM:        {vanilla_mean:.2e}")
    print(f"  SC-TRM (no guards): {sc_mean:.2e}")
    print(f"  SC-TRM (guards):    {sc_guards_mean:.2e}")

    if sc_mean < vanilla_mean * 0.1:
        print("\n⚠️  SC-TRM gradients are 10x+ smaller than Vanilla TRM!")
        print("    This explains why SC-TRM isn't learning.")

    print("\nDone!")


if __name__ == "__main__":
    main()
