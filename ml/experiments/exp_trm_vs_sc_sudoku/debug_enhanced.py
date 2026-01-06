"""Debug enhanced SC-TRM training."""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.exp_trm_vs_sc_sudoku.sc_trm_enhanced import EnhancedSCTRM, EnhancedSCConfig
from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku


def main():
    print("=" * 60)
    print("Debug Enhanced SC-TRM Training")
    print("=" * 60)

    # Small test
    train_p, train_s, test_p, test_s = generate_random_sudoku(500, 100, seed=42)

    print("\n1. Testing Vanilla TRM (baseline)...")
    vanilla_config = VanillaTRMConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
    )
    vanilla_model = VanillaTRM(vanilla_config)

    def vanilla_loss_fn(m, p, s):
        l, _ = m.loss(p, s)
        return l

    vanilla_loss_and_grad = nn.value_and_grad(vanilla_model, vanilla_loss_fn)
    vanilla_optimizer = optim.AdamW(learning_rate=1e-4)

    for epoch in range(10):
        loss, grads = vanilla_loss_and_grad(vanilla_model, train_p[:32], train_s[:32])
        vanilla_optimizer.update(vanilla_model, grads)
        mx.eval(vanilla_model.parameters())
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    print("\n2. Testing Enhanced SC-TRM...")
    enhanced_config = EnhancedSCConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
    )
    enhanced_model = EnhancedSCTRM(enhanced_config)

    # Test raw solve
    print("   Testing raw solve...")
    result = enhanced_model.solve(train_p[:4], method="raw")
    print(f"   Raw logits shape: {result['logits'].shape}")

    # Test loss
    print("   Testing loss...")
    loss, metrics = enhanced_model.loss(train_p[:4], train_s[:4])
    mx.eval(loss)
    print(f"   Loss: {float(loss.item()):.4f}")

    def enhanced_loss_fn(m, p, s):
        l, _ = m.loss(p, s)
        return l

    enhanced_loss_and_grad = nn.value_and_grad(enhanced_model, enhanced_loss_fn)
    enhanced_optimizer = optim.AdamW(learning_rate=1e-4)

    print("   Training for 10 epochs...")
    for epoch in range(10):
        loss, grads = enhanced_loss_and_grad(enhanced_model, train_p[:32], train_s[:32])
        enhanced_optimizer.update(enhanced_model, grads)
        mx.eval(enhanced_model.parameters())
        if epoch % 2 == 0:
            print(f"   Epoch {epoch+1}: loss={float(loss.item()):.4f}")

    # Compare gradient norms
    print("\n3. Comparing gradient norms...")

    vanilla_l, vanilla_g = vanilla_loss_and_grad(vanilla_model, train_p[:8], train_s[:8])
    enhanced_l, enhanced_g = enhanced_loss_and_grad(enhanced_model, train_p[:8], train_s[:8])
    mx.eval(vanilla_l, vanilla_g, enhanced_l, enhanced_g)

    def count_grad_norm(grads):
        total = 0.0
        count = 0
        if isinstance(grads, dict):
            for v in grads.values():
                t, c = count_grad_norm(v)
                total += t
                count += c
        elif isinstance(grads, mx.array):
            total = float(mx.sqrt(mx.sum(grads * grads)).item())
            count = 1
        elif isinstance(grads, (list, tuple)):
            for v in grads:
                t, c = count_grad_norm(v)
                total += t
                count += c
        return total, count

    v_total, v_count = count_grad_norm(vanilla_g)
    e_total, e_count = count_grad_norm(enhanced_g)

    print(f"   Vanilla TRM: mean grad norm = {v_total/v_count:.4e}")
    print(f"   Enhanced SC-TRM: mean grad norm = {e_total/e_count:.4e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
