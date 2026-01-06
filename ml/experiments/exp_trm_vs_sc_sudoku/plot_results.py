#!/usr/bin/env python3
"""
Simple visualization: Learning curves + Time vs Accuracy.
"""

import json
import os
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "experiments/exp_trm_vs_sc_sudoku/benchmark_results"
FIGURES_DIR = f"{RESULTS_DIR}/figures"

MODEL_COLORS = {
    'vanilla': '#808080',
    'faithful': '#000000',
    'ste': '#2196F3',
    'confidence': '#4CAF50',
    'attention_bias': '#FF9800',
    'hierarchical': '#9C27B0',
    'learned': '#E91E63',
    'nas': '#00BCD4',
    # New models
    'iterative': '#3F51B5',      # Indigo
    'iterative_mlp': '#673AB7',  # Deep Purple
    'faithful_v2': '#009688',    # Teal
    'faithful_v2_attn': '#00796B',
    'hard_mask': '#F44336',      # Red
    'hard_mask_iter': '#D32F2F', # Dark Red
    'nas_pretrained': '#607D8B', # Blue Grey
    'large': '#795548',          # Brown
    'xlarge': '#5D4037',         # Dark Brown
    # SAE+Diff models
    'sae_diff': '#8BC34A',       # Light Green
    'sae_diff_sudoku': '#689F38', # Green
    # Faithful v3 (corrected)
    'faithful_v3': '#1E88E5',    # Blue
    'faithful_v3_attn': '#1565C0', # Dark Blue
    # Faithful v4 (StableMax)
    'faithful_v4': '#43A047',    # Green
    'faithful_v4_attn': '#2E7D32', # Dark Green
    'faithful_v4_rope': '#1B5E20', # Darker Green
    # ACT (Adaptive Computation Time)
    'faithful_act': '#FF5722',   # Deep Orange
    'faithful_act_attn': '#E64A19', # Dark Deep Orange
}

MODEL_LABELS = {
    'vanilla': 'Vanilla TRM',
    'faithful': 'Faithful TRM',
    'ste': 'STE Guards',
    'confidence': 'Confidence Guards',
    'attention_bias': 'Attention Bias',
    'hierarchical': 'Hierarchical SC',
    'learned': 'Learned SC',
    'nas': 'NAS-SC',
    # New models
    'iterative': 'Iterative (Attn)',
    'iterative_mlp': 'Iterative (MLP)',
    'faithful_v2': 'Faithful v2 (MLP)',
    'faithful_v2_attn': 'Faithful v2 (Attn)',
    'hard_mask': 'Hard Mask',
    'hard_mask_iter': 'Hard Mask + Iter',
    'nas_pretrained': 'NAS Pretrained',
    'large': 'Large (256d)',
    'xlarge': 'XLarge (512d)',
    # SAE+Diff models
    'sae_diff': 'SAE+Diff (cold)',
    'sae_diff_sudoku': 'SAE+Diff (warm)',
    # Faithful v3 (corrected)
    'faithful_v3': 'Faithful v3 (MLP)',
    'faithful_v3_attn': 'Faithful v3 (Attn)',
    # Faithful v4 (StableMax)
    'faithful_v4': 'Faithful v4 (MLP)',
    'faithful_v4_attn': 'Faithful v4 (Attn)',
    'faithful_v4_rope': 'Faithful v4 (RoPE)',
    # ACT
    'faithful_act': 'ACT (MLP)',
    'faithful_act_attn': 'ACT (Attn)',
}

def load_results():
    """Load all results from JSON files."""
    results = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith('results_') and filename.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, filename)) as f:
                results.append(json.load(f))
    return results

def plot_learning_curves(results, output_path):
    """Plot cell accuracy over epochs for all models."""
    plt.figure(figsize=(10, 6))

    for result in sorted(results, key=lambda r: r.get('final_cell_accuracy', 0), reverse=True):
        model = result['model_name']
        history = result.get('train_history', [])

        if not history:
            continue

        epochs = [h['epoch'] for h in history]
        cell_acc = [h['cell_accuracy'] * 100 for h in history]

        color = MODEL_COLORS.get(model, '#333333')
        label = MODEL_LABELS.get(model, model)

        plt.plot(epochs, cell_acc, color=color, label=label, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cell Accuracy (%)', fontsize=12)
    plt.title('SC-TRM Learning Curves: Cell Accuracy Over Training', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_time_vs_accuracy(results, output_path):
    """Plot training time vs final accuracy."""
    plt.figure(figsize=(10, 6))

    for result in results:
        model = result['model_name']
        time_s = result.get('training_time_seconds', 0)
        cell_acc = result.get('final_cell_accuracy', 0) * 100

        color = MODEL_COLORS.get(model, '#333333')
        label = MODEL_LABELS.get(model, model)

        plt.scatter(time_s, cell_acc, color=color, s=150, label=label,
                   edgecolors='white', linewidths=2, zorder=5)

        # Add label next to point
        plt.annotate(label, (time_s, cell_acc),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=9, va='center')

    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Final Cell Accuracy (%)', fontsize=12)
    plt.title('Training Efficiency: Time vs Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_results()
    print(f"Loaded {len(results)} results")

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x.get('final_cell_accuracy', 0), reverse=True):
        model = r['model_name']
        acc = r.get('final_cell_accuracy', 0) * 100
        time_s = r.get('training_time_seconds', 0)
        print(f"  {MODEL_LABELS.get(model, model):20s}: {acc:5.1f}% ({time_s:.1f}s)")
    print("-" * 50)

    # Generate plots
    plot_learning_curves(results, f"{FIGURES_DIR}/learning_curves.png")
    plot_time_vs_accuracy(results, f"{FIGURES_DIR}/time_vs_accuracy.png")

    print(f"\nPlots saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
