"""
Unified Benchmark Runner for Long Training Runs (100 Epochs).

Usage:
    python3 -m experiments.exp_trm_vs_sc_sudoku.run_long_benchmark --model [hybrid|attention|faithful|hierarchical|vanilla] --epochs 100
"""

import argparse
import sys
import os
import time
import datetime
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Import Models
from experiments.exp_trm_vs_sc_sudoku.sc_trm_hybrid import SCTRMHybrid, SCTRMConfig
from experiments.exp_trm_vs_sc_sudoku.attention_bias import AttentionBiasTRM, AttentionBiasConfig
from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v3 import FaithfulTRMv3, FaithfulTRMv3Config
from experiments.exp_trm_vs_sc_sudoku.hierarchical_sc_trm import HierarchicalSCTRM, HierarchicalConfig
from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
from experiments.exp_trm_vs_sc_sudoku.hybrid_models import (
    AttentionBiasGradTrunc, AttentionBiasGradTruncConfig,
    HierarchicalAttention, HierarchicalAttentionConfig,
    FaithfulAttention, FaithfulAttentionConfig,
    DeepAttention, DeepAttentionConfig,
)
from experiments.exp_trm_vs_sc_sudoku.sc_neural_models import (
    StateHeadTRM, StateHeadConfig,
    GuardGatedTRM, GuardGatedConfig,
    OrthogonalTRM, OrthogonalConfig,
    HistoryTRM, HistoryConfig,
    EvolvedConstraintTRM, EvolvedConstraintConfig,
)
from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

def setup_logging(model_name):
    log_dir = "experiments/exp_trm_vs_sc_sudoku/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_{model_name}_{timestamp}.log")
    
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    return log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=[
        "hybrid", "attention", "faithful", "hierarchical", "vanilla",
        "attn_grad_trunc", "hier_attn", "faithful_attn", "deep_attn",
        # SC-inspired models
        "state_head", "guard_gated", "orthogonal", "history", "evolved_constraint"
    ])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    log = setup_logging(args.model)
    log(f"=" * 60)
    log(f"Starting Benchmark: {args.model.upper()}")
    log(f"Epochs: {args.epochs}")
    log(f"Seed: {args.seed}")
    log(f"=" * 60)
    
    # 1. Data
    log("\nGenerating 2200 Sudoku puzzles (2000 train, 200 test)...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(2000, 200, seed=args.seed)
    
    # 2. Model Setup
    if args.model == "hybrid":
        config = SCTRMConfig(
            hidden_size=128, num_heads=4, num_layers=2, 
            H_cycles=4, L_cycles=4,
            use_constraint_bias=True, 
            initial_bias_scale=1.0, final_bias_scale=5.0,
            init_std=0.02
        )
        model = SCTRMHybrid(config)
        truncate_grad = True
        
    elif args.model == "attention":
        config = AttentionBiasConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            use_constraint_bias=True,
            initial_bias_scale=0.5, final_bias_scale=3.0,
            learnable_bias=True
        )
        model = AttentionBiasTRM(config)
        truncate_grad = False # Not used in this class
        
    elif args.model == "faithful":
        config = FaithfulTRMv3Config(
            hidden_size=128, num_heads=4, 
            H_cycles=3, L_cycles=6, L_layers=2,
            init_std=1.0 # Faithful uses 1.0 (Samsung default)
        )
        model = FaithfulTRMv3(config)
        truncate_grad = True
        
    elif args.model == "hierarchical":
        config = HierarchicalConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            unit_hidden_dim=64, board_hidden_dim=32
        )
        model = HierarchicalSCTRM(config)
        truncate_grad = False
        
    elif args.model == "vanilla":
        config = VanillaTRMConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4
        )
        model = VanillaTRM(config)
        truncate_grad = False

    # === NEW HYBRID MODELS ===
    elif args.model == "attn_grad_trunc":
        config = AttentionBiasGradTruncConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            initial_bias_scale=0.5, final_bias_scale=3.0
        )
        model = AttentionBiasGradTrunc(config)
        truncate_grad = True

    elif args.model == "hier_attn":
        config = HierarchicalAttentionConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            unit_hidden_dim=64, board_hidden_dim=32,
            initial_bias_scale=0.5, final_bias_scale=3.0
        )
        model = HierarchicalAttention(config)
        truncate_grad = False

    elif args.model == "faithful_attn":
        config = FaithfulAttentionConfig(
            hidden_size=128, num_heads=4,
            H_cycles=3, L_cycles=6, L_layers=2,
            init_std=1.0, use_constraint_bias=True,
            initial_bias_scale=0.5, final_bias_scale=3.0
        )
        model = FaithfulAttention(config)
        truncate_grad = True

    elif args.model == "deep_attn":
        config = DeepAttentionConfig(
            hidden_dim=192, num_heads=6, num_layers=4, ff_dim=384,
            H_cycles=4, L_cycles=4,
            initial_bias_scale=0.5, final_bias_scale=4.0
        )
        model = DeepAttention(config)
        truncate_grad = True

    # === SC-INSPIRED NEURAL MODELS ===
    elif args.model == "state_head":
        config = StateHeadConfig(
            hidden_dim=128, num_state_heads=9, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4, state_temperature=1.0
        )
        model = StateHeadTRM(config)
        truncate_grad = False

    elif args.model == "guard_gated":
        config = GuardGatedConfig(
            hidden_dim=128, num_heads=4, num_guards=27, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4, guard_threshold=0.5
        )
        model = GuardGatedTRM(config)
        truncate_grad = False

    elif args.model == "orthogonal":
        config = OrthogonalConfig(
            hidden_dim=128, num_regions=3, region_dim=42, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4, sync_every=2
        )
        model = OrthogonalTRM(config)
        truncate_grad = False

    elif args.model == "history":
        config = HistoryConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4, history_depth=2, use_deep_history=True
        )
        model = HistoryTRM(config)
        truncate_grad = False

    elif args.model == "evolved_constraint":
        config = EvolvedConstraintConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=4, L_cycles=4, constraint_dim=64
        )
        model = EvolvedConstraintTRM(config)
        truncate_grad = False

    else:
        raise ValueError(f"Unknown model: {args.model}")

    log(f"\nModel Config: {config}")
    
    # 3. Optimizer
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    # 4. Training Loop (with History Tracking)
    history = []
    
    def loss_fn(m, p, s):
        # Handle different signatures
        if args.model == "hybrid":
            loss, metrics = m.loss(p, s, truncate_grad=True)
        elif args.model == "faithful":
            loss, metrics = m.loss(p, s, truncate_grad=True)
        elif args.model == "attention":
            loss, metrics = m.loss(p, s, bias_anneal=True)
        elif args.model == "hierarchical":
             loss, metrics = m.loss(p, s)
        elif args.model == "vanilla":
            loss, metrics = m.loss(p, s)
        # New hybrid models
        elif args.model in ["attn_grad_trunc", "faithful_attn", "deep_attn"]:
            loss, metrics = m.loss(p, s, truncate_grad=True)
        elif args.model == "hier_attn":
            loss, metrics = m.loss(p, s)
        # SC-inspired models
        elif args.model in ["state_head", "guard_gated", "orthogonal", "history", "evolved_constraint"]:
            loss, metrics = m.loss(p, s)
        else:
            return 0, 0

        # Standardize metric keys - prioritizing cell_accuracy
        acc = metrics.get("cell_accuracy", metrics.get("accuracy", 0))
        return loss, acc

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    batch_size = 32
    log(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    
    # Fixed validation set for consistent metrics
    val_p = test_p[:100]
    val_s = test_s[:100]

    for epoch in range(args.epochs):
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
            if hasattr(acc, "item"):
                epoch_acc += acc.item()
            else:
                epoch_acc += acc
            steps += 1
            
        avg_loss = epoch_loss / steps
        avg_acc = epoch_acc / steps

        # Validation Step (Exact Accuracy)
        if args.model == "attention":
            val_out = model.solve(val_p, bias_anneal=True)
        else:
            val_out = model.solve(val_p)
        
        val_preds = val_out['predictions']
        val_cell_acc = mx.mean((val_preds == val_s).astype(mx.float32)).item()
        
        # Exact match: all 81 cells must match
        # predictions: [B, 81], solution: [B, 81]
        row_matches = mx.all(val_preds == val_s, axis=1) # [B]
        val_exact_acc = mx.mean(row_matches.astype(mx.float32)).item()
        
        # Log to console
        log(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f} | TrainAcc={avg_acc:.2%} | ValCell={val_cell_acc:.2%} | ValExact={val_exact_acc:.2%}")
        
        # Track history
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cell_accuracy": val_cell_acc, # Use validation metrics for cleaner graphs
            "exact_accuracy": val_exact_acc,
            "violation_rate": 0.0,
        }
        history.append(epoch_metrics)

        # Incremental Save
        result_dir = "experiments/exp_trm_vs_sc_sudoku/benchmark_results"
        os.makedirs(result_dir, exist_ok=True)
        interim_result = {
            "model_name": args.model,
            "epochs": args.epochs,
            "seed": args.seed,
            "status": "running",
            "current_epoch": epoch + 1,
            "train_history": history,
            "final_cell_accuracy": val_cell_acc,
            "final_exact_accuracy": val_exact_acc,
            "final_violation_rate": 0.0,
            "training_time_seconds": time.time() - start_time,
            "epochs_to_50_cell": next((x["epoch"] for x in history if x["cell_accuracy"] >= 0.5), None),
            "epochs_to_70_cell": next((x["epoch"] for x in history if x["cell_accuracy"] >= 0.7), None),
        }
        json_path = os.path.join(result_dir, f"results_{args.model}_e{args.epochs}_s{args.seed}.json")
        import json
        with open(json_path, "w") as f:
            json.dump(interim_result, f, indent=2)
        
    duration = time.time() - start_time
    log(f"\nTraining finished in {duration:.1f}s")
    
    # 5. Evaluation & Saving Results
    log("\nEvaluating on Full Test Set...")
    
    # Handle solve signature differences
    if args.model == "attention":
        result = model.solve(test_p, bias_anneal=True)
    else:
        result = model.solve(test_p)
        
    test_preds = result['predictions']
    test_acc = mx.mean((test_preds == test_s).astype(mx.float32)).item()
    test_exact = mx.mean(mx.all(test_preds == test_s, axis=1).astype(mx.float32)).item()
    
    log("=" * 60)
    log(f"FINAL TEST CELL ACCURACY:  {test_acc:.2%}")
    log(f"FINAL TEST EXACT ACCURACY: {test_exact:.2%}")
    log("=" * 60)
    
    # Save standard result JSON
    result_dir = "experiments/exp_trm_vs_sc_sudoku/benchmark_results"
    os.makedirs(result_dir, exist_ok=True)
    
    experiment_result = {
        "model_name": args.model,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_history": history,
        "final_cell_accuracy": test_acc,
        "final_exact_accuracy": test_exact,
        "final_violation_rate": 0.0, # Placeholder for now
        "training_time_seconds": duration,
        "epochs_to_50_cell": next((x["epoch"] for x in history if x["cell_accuracy"] >= 0.5), None),
        "epochs_to_70_cell": next((x["epoch"] for x in history if x["cell_accuracy"] >= 0.7), None),
    }
    
    json_path = os.path.join(result_dir, f"results_{args.model}_e{args.epochs}_s{args.seed}.json")
    import json
    with open(json_path, "w") as f:
        json.dump(experiment_result, f, indent=2)
    log(f"Saved standard results to {json_path}")

if __name__ == "__main__":
    main()
