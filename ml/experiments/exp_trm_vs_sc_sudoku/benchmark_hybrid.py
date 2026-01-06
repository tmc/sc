"""
Benchmark Hybrid TRM.

Target: Beat 61% accuracy (Attention Bias pure) using Hybrid (Bias + Grad Truncation).
"""

import sys
# sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
import time

from experiments.exp_trm_vs_sc_sudoku.sc_trm_hybrid import SCTRMHybrid, SCTRMConfig
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

def train_hybrid_benchmark():
    print("=" * 70)
    print("HYBRID TRM BENCHMARK (Bias + Grad Truncation)")
    print("=" * 70)

    # 1. Generate Data (Standard size for quick benchmark)
    print("\nGenerating data...")
    # 2000 train, 200 test
    train_p, train_s, test_p, test_s = generate_random_sudoku(2000, 200, seed=42)
    
    # 2. Configure Hybrid Model
    config = SCTRMConfig(
        hidden_size=128,
        num_heads=4,
        num_layers=2,       # L_layers
        H_cycles=4,         # More cycles to leverage stability
        L_cycles=4,
        use_constraint_bias=True,
        initial_bias_scale=1.0, 
        final_bias_scale=5.0,   # Strong annealing
        dropout=0.1
    )
    
    model = SCTRMHybrid(config)
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4) # Slightly higher LR allowed by stable grad
    
    print(f"\nModel Configuration:")
    print(f"  Hidden: {config.hidden_size}, Heads: {config.num_heads}")
    print(f"  H_cycles: {config.H_cycles}, L_cycles: {config.L_cycles}")
    print(f"  Bias Annealing: {config.initial_bias_scale} -> {config.final_bias_scale}")
    
    # 3. Setup Logging
    import os
    import datetime
    
    log_dir = "experiments/exp_trm_vs_sc_sudoku/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hybrid_benchmark_{timestamp}.log")
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            
    header = f"Hybrid TRM Benchmark - {timestamp}\nConfig: {config}\n"
    log(header)
    
    # 3. Training Loop
    def step_fn(m, p, s):
        loss, metrics = m.loss(p, s, truncate_grad=True)
        return loss, metrics["cell_accuracy"]

    loss_and_grad = nn.value_and_grad(model, step_fn)
    
    batch_size = 32
    epochs = 20
    
    log(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        perm = mx.random.permutation(train_p.shape[0])
        train_p = train_p[perm]
        train_s = train_s[perm]
        
        epoch_loss = 0
        epoch_acc = 0
        steps = 0
        
        for i in range(0, train_p.shape[0], batch_size):
            batch_p = train_p[i:i+batch_size]
            batch_s = train_s[i:i+batch_size]
            
            (loss, acc), grads = loss_and_grad(model, batch_p, batch_s)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            steps += 1
            
        avg_loss = epoch_loss / steps
        avg_acc = epoch_acc / steps
        log(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f} | Cell Acc={avg_acc:.2%}")
        
    duration = time.time() - start_time
    log(f"\nTraining finished in {duration:.1f}s")
    
    # 4. Evaluation
    log("\nEvaluating on Test Set...")
    result = model.solve(test_p)
    acc = mx.mean((result['predictions'] == test_s).astype(mx.float32)).item()
    
    log("-" * 70)
    log(f"Final Test Accuracy: {acc:.2%}")
    log("-" * 70)
    
    # Comparison Baseline
    log("Baselines (approximate):")
    log("  Vanilla TRM (30 epochs): ~13-15%")
    log("  Faithful v3 (100 epochs): ~41%")
    log("  Attn Bias (30 epochs):   ~60%")
    log("If Hybrid > 60% quickly, we have a winner.")

if __name__ == "__main__":
    train_hybrid_benchmark()
