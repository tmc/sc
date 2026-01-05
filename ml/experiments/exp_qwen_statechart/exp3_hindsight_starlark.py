"""
Experiment 3: Hindsight Relabeling for Starlark

Goal: Test hindsight relabeling for Starlark generation sample efficiency.

Hypothesis: Hindsight relabeling provides 2-3x sample efficiency improvement
by converting failed program executions into valid training data.
"""

from typing import List, Dict, Any
from .hindsight_trainer import HindsightConfig, HindsightTrainer
from .lora_trainer import load_starlark_corpus, load_starlark_eval


def run_experiment_3(
    epochs: int = 5,
    generate_per_epoch: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test hindsight relabeling for sample efficiency."""

    train_data = load_starlark_corpus()
    eval_data = load_starlark_eval()

    config = HindsightConfig(
        r=8, alpha=16, batch_size=2,
        min_complexity=0.1,
        diversity_threshold=0.3,
        generations_per_epoch=generate_per_epoch,
    )

    trainer = HindsightTrainer(config)
    trainer.load_model()

    results = trainer.compare_with_baseline(train_data, eval_data, epochs=epochs)

    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3 RESULTS: Hindsight Relabeling")
        print("=" * 60)
        print(f"\nBaseline final loss: {results['baseline']['final_loss']:.4f}")
        print(f"Hindsight final loss: {results['hindsight']['final_loss']:.4f}")
        print(f"Loss reduction: {results['improvement']['loss_reduction']:.4f}")
        print(f"Data amplification: {results['improvement']['data_amplification']:.2f}x")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 3: Hindsight Relabeling")
    print("=" * 60)

    try:
        results = run_experiment_3(epochs=2, generate_per_epoch=20, verbose=True)
    except Exception as e:
        print(f"Error (MLX may not be available): {e}")

        # Demo the relabeler
        from .hindsight_trainer import HindsightRelabeler
        relabeler = HindsightRelabeler()
        relabeler.collect("def foo():", "def foo():\n    return 1", True, False, "test")
        print(f"\nRelabeler demo: {relabeler.get_statistics()}")
