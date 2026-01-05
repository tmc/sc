"""
Experiment 2: LoRA vs Constrained LoRA

Goal: Compare standard LoRA vs statechart-constrained LoRA training.

Metrics:
- Validation loss curves
- Syntax validity at checkpoints
- Semantic correctness (execution success)
- Sample efficiency (samples to reach X% validity)
"""

from typing import List, Dict, Any
from .lora_trainer import LoRAConfig, LoRATrainer, load_starlark_corpus, load_starlark_eval
from .constrained_lora import ConstrainedLoRAConfig, ConstrainedLoRATrainer, train_and_eval


def run_experiment_2(
    epochs: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare LoRA vs Constrained LoRA training."""

    train_data = load_starlark_corpus()
    eval_data = load_starlark_eval()

    results = {'baseline': None, 'constrained': None}

    # Baseline LoRA
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE LORA TRAINING")
        print("=" * 60)

    baseline_config = LoRAConfig(r=8, alpha=16, batch_size=2)
    baseline = LoRATrainer(baseline_config)
    baseline.load_model()

    baseline_metrics = []
    for epoch in range(epochs):
        metrics = baseline.train_epoch(train_data, eval_data)
        baseline_metrics.append(metrics)
        if verbose:
            print(f"Epoch {epoch+1}: loss={metrics.get('train_loss', 0):.4f}")

    results['baseline'] = {'epochs': baseline_metrics}

    # Constrained LoRA
    if verbose:
        print("\n" + "=" * 60)
        print("CONSTRAINED LORA TRAINING")
        print("=" * 60)

    constrained_config = ConstrainedLoRAConfig(r=8, alpha=16, batch_size=2, constraint_weight=0.5)
    constrained = ConstrainedLoRATrainer(constrained_config)
    constrained.load_model()

    constrained_results = train_and_eval(constrained, train_data, eval_data, epochs=epochs)
    results['constrained'] = constrained_results

    # Summary
    results['comparison'] = {
        'baseline_final_loss': baseline_metrics[-1].get('train_loss', 0) if baseline_metrics else 0,
        'constrained_final_loss': constrained_results.get('final_metrics', {}).get('train_loss', 0),
        'validity_improvement': constrained_results.get('final_metrics', {}).get('validity_improvement', 0),
    }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 2: LoRA vs Constrained LoRA")
    print("=" * 60)

    try:
        results = run_experiment_2(epochs=2, verbose=True)
        print("\nComparison:")
        for k, v in results['comparison'].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error (MLX may not be available): {e}")
