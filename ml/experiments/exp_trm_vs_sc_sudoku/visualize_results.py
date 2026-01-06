"""
Visualization Generator for SC-TRM Benchmark Results.

Generates publication-quality plots comparing 5 SC-TRM approaches.

Usage:
    python -m experiments.exp_trm_vs_sc_sudoku.visualize_results --input benchmark_results/
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Color scheme for models
MODEL_COLORS = {
    'vanilla': '#808080',       # Gray - Our baseline
    'faithful': '#000000',      # Black - Faithful to upstream
    'ste': '#2196F3',           # Blue
    'confidence': '#4CAF50',    # Green
    'attention_bias': '#FF9800', # Orange
    'hierarchical': '#9C27B0',  # Purple
}

MODEL_LABELS = {
    'vanilla': 'Vanilla TRM',
    'faithful': 'Faithful TRM',
    'ste': 'STE Guards',
    'confidence': 'Confidence Guards',
    'attention_bias': 'Attention Bias',
    'hierarchical': 'Hierarchical SC',
}


def load_results(results_dir: str) -> List[Dict]:
    """Load all experiment results from directory."""
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
            return summary.get('results', [])

    # Fallback: load individual files
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith('results_') and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath) as f:
                results.append(json.load(f))
    return results


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_learning_curves(results: List[Dict], output_path: str):
    """
    Plot learning curves for all approaches.

    X-axis: Epoch
    Y-axis: Cell Accuracy (%)
    Lines: 5 approaches with error bands
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group results by model
    model_histories = defaultdict(list)
    for r in results:
        model_name = r['model_name']
        history = r['train_history']
        model_histories[model_name].append(history)

    # Plot each model
    for model_name in MODEL_COLORS.keys():
        histories = model_histories.get(model_name, [])
        if not histories:
            continue

        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]

        # Align histories by epoch
        all_epochs = set()
        for h in histories:
            for entry in h:
                all_epochs.add(entry['epoch'])
        epochs = sorted(all_epochs)

        # Collect values per epoch
        epoch_values = defaultdict(list)
        for h in histories:
            epoch_to_acc = {e['epoch']: e['cell_accuracy'] for e in h}
            for epoch in epochs:
                if epoch in epoch_to_acc:
                    epoch_values[epoch].append(epoch_to_acc[epoch])

        # Compute mean and range
        epochs_list = []
        means = []
        mins = []
        maxs = []

        for epoch in epochs:
            vals = epoch_values[epoch]
            if vals:
                epochs_list.append(epoch)
                means.append(np.mean(vals))
                mins.append(np.min(vals))
                maxs.append(np.max(vals))

        if epochs_list:
            epochs_arr = np.array(epochs_list)
            means_arr = np.array(means) * 100
            mins_arr = np.array(mins) * 100
            maxs_arr = np.array(maxs) * 100

            ax.plot(epochs_arr, means_arr, color=color, label=label, linewidth=2)
            ax.fill_between(epochs_arr, mins_arr, maxs_arr, color=color, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cell Accuracy (%)')
    ax.set_title('SC-TRM Learning Curves: Cell Accuracy Over Training')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_epoch_scaling(results: List[Dict], output_path: str):
    """
    Plot epoch scaling comparison as grouped bar chart.

    X-axis: Epochs trained [30, 50, 100, ...]
    Y-axis: Final Cell Accuracy (%)
    Bars: 5 approaches grouped
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by (model, epochs)
    grouped = defaultdict(list)
    for r in results:
        key = (r['model_name'], r['epochs'])
        grouped[key].append(r['final_cell_accuracy'])

    # Get unique epochs and models
    all_epochs = sorted(set(r['epochs'] for r in results))
    models = list(MODEL_COLORS.keys())

    x = np.arange(len(all_epochs))
    width = 0.15
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    for i, model in enumerate(models):
        means = []
        stds = []
        for epochs in all_epochs:
            vals = grouped.get((model, epochs), [])
            if vals:
                means.append(np.mean(vals) * 100)
                stds.append(np.std(vals) * 100 if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + offsets[i], means, width, label=MODEL_LABELS[model],
               color=MODEL_COLORS[model], yerr=stds, capsize=3)

    ax.set_xlabel('Epochs Trained')
    ax.set_ylabel('Final Cell Accuracy (%)')
    ax.set_title('SC-TRM Epoch Scaling: How Accuracy Improves with More Training')
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in all_epochs])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_pareto(results: List[Dict], output_path: str):
    """
    Plot efficiency comparison (Pareto frontier).

    X-axis: Training Time (seconds)
    Y-axis: Cell Accuracy (%)
    Points: Each (model, epochs) config
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODEL_COLORS.keys():
        model_results = [r for r in results if r['model_name'] == model]
        if not model_results:
            continue

        times = [r['training_time_seconds'] for r in model_results]
        accs = [r['final_cell_accuracy'] * 100 for r in model_results]
        epochs = [r['epochs'] for r in model_results]

        ax.scatter(times, accs, c=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

        # Annotate with epoch count
        for t, a, e in zip(times, accs, epochs):
            ax.annotate(f'{e}', (t, a), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, alpha=0.7)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Cell Accuracy (%)')
    ax.set_title('SC-TRM Efficiency: Accuracy vs Training Time')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_speed(results: List[Dict], output_path: str):
    """
    Plot convergence speed (epochs to reach 70% cell accuracy).

    X-axis: Model
    Y-axis: Epochs to reach 70% cell accuracy
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model, collect epochs_to_70_cell
    model_epochs = defaultdict(list)
    for r in results:
        if r.get('epochs_to_70_cell') is not None:
            model_epochs[r['model_name']].append(r['epochs_to_70_cell'])

    models = list(MODEL_COLORS.keys())
    x = np.arange(len(models))
    means = []
    stds = []
    colors = []

    for model in models:
        vals = model_epochs.get(model, [])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        else:
            means.append(float('nan'))
            stds.append(0)
        colors.append(MODEL_COLORS[model])

    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(means)
    valid_x = x[valid_mask]
    valid_means = np.array(means)[valid_mask]
    valid_stds = np.array(stds)[valid_mask]
    valid_colors = [c for c, v in zip(colors, valid_mask) if v]
    valid_labels = [MODEL_LABELS[models[i]] for i in range(len(models)) if valid_mask[i]]

    if len(valid_x) > 0:
        bars = ax.bar(valid_x, valid_means, color=valid_colors, yerr=valid_stds, capsize=5)
        ax.set_xticks(valid_x)
        ax.set_xticklabels(valid_labels, rotation=15, ha='right')
    else:
        ax.text(0.5, 0.5, 'No models reached 70% cell accuracy',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Model')
    ax.set_ylabel('Epochs to Reach 70% Cell Accuracy')
    ax.set_title('SC-TRM Convergence Speed: How Fast Models Learn')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_violations_vs_accuracy(results: List[Dict], output_path: str):
    """
    Plot constraint violations vs accuracy.

    X-axis: Cell Accuracy (%)
    Y-axis: Constraint Violations per Puzzle
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODEL_COLORS.keys():
        model_results = [r for r in results if r['model_name'] == model]
        if not model_results:
            continue

        accs = [r['final_cell_accuracy'] * 100 for r in model_results]
        violations = [r['final_violation_rate'] for r in model_results]

        ax.scatter(accs, violations, c=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Cell Accuracy (%)')
    ax.set_ylabel('Constraint Violations per Puzzle')
    ax.set_title('SC-TRM: Accuracy vs Constraint Satisfaction')
    ax.legend(loc='upper right')

    # Ideal point annotation
    ax.annotate('Ideal: High accuracy,\nlow violations', xy=(95, 1),
                fontsize=9, ha='center', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison_summary(results: List[Dict], output_path: str):
    """
    Plot summary comparison: best result per model.

    Horizontal bar chart showing final accuracy.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get best result per model
    best_per_model = {}
    for r in results:
        model = r['model_name']
        if model not in best_per_model or r['final_cell_accuracy'] > best_per_model[model]['final_cell_accuracy']:
            best_per_model[model] = r

    models = list(MODEL_COLORS.keys())
    y_pos = np.arange(len(models))
    accs = [best_per_model.get(m, {}).get('final_cell_accuracy', 0) * 100 for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    # Sort by accuracy
    sorted_indices = np.argsort(accs)
    accs = [accs[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    bars = ax.barh(y_pos, accs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Best Cell Accuracy (%)')
    ax.set_title('SC-TRM Model Comparison: Best Results')
    ax.set_xlim(0, 100)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.svg'))
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(results_dir: str):
    """Generate all visualization plots."""
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\nGenerating plots from {len(results)} experiment results...\n")

    plot_learning_curves(
        results,
        os.path.join(figures_dir, 'learning_curves_cell_accuracy.png')
    )

    plot_epoch_scaling(
        results,
        os.path.join(figures_dir, 'epoch_scaling_bar.png')
    )

    plot_efficiency_pareto(
        results,
        os.path.join(figures_dir, 'efficiency_pareto.png')
    )

    plot_convergence_speed(
        results,
        os.path.join(figures_dir, 'convergence_speed.png')
    )

    plot_violations_vs_accuracy(
        results,
        os.path.join(figures_dir, 'violations_vs_accuracy.png')
    )

    plot_model_comparison_summary(
        results,
        os.path.join(figures_dir, 'model_comparison_summary.png')
    )

    print(f"\nAll plots saved to {figures_dir}/")


def main():
    parser = argparse.ArgumentParser(description='SC-TRM Results Visualizer')
    parser.add_argument('--input', type=str,
                        default='experiments/exp_trm_vs_sc_sudoku/benchmark_results',
                        help='Results directory')

    args = parser.parse_args()
    generate_all_plots(args.input)


if __name__ == "__main__":
    main()
