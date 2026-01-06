"""
Training script for Sudoku Statechart 9x9.

Usage:
    python -m ml.experiments.exp_trm_sudoku_9x9.train --max_samples 1000 --epochs 10
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .sudoku_statechart_9x9 import SudokuStatechart9x9
from .data_loader import SudokuExtremeDataset
from .losses import stablemax_cross_entropy, sudoku_loss


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    max_samples: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_augmentations: int = 0
    batch_size: int = 64

    # Model
    hidden_dim: int = 128
    H_cycles: int = 3
    L_cycles: int = 6
    learned_guards: bool = True

    # Training
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1.0
    warmup_steps: int = 1000
    use_stablemax: bool = True
    constraint_weight: float = 0.0

    # Checkpointing
    save_dir: str = "checkpoints/exp_trm_sudoku_9x9"
    save_every: int = 10
    eval_every: int = 5

    # Misc
    seed: int = 42


def train_step(
    model: SudokuStatechart9x9,
    optimizer: optim.Optimizer,
    questions: mx.array,
    answers: mx.array,
    config: TrainConfig,
) -> dict:
    """
    Single training step.

    Args:
        model: The model
        optimizer: The optimizer
        questions: [B, 81] input puzzles
        answers: [B, 81] target solutions
        config: Training config

    Returns:
        Dict with loss and metrics
    """
    def loss_fn(model):
        # Forward pass
        logits, _ = model.forward(questions)

        # Compute loss
        loss = sudoku_loss(
            logits,
            answers,
            use_stablemax=config.use_stablemax,
            constraint_weight=config.constraint_weight,
        )

        return loss, logits

    # Compute gradients
    (loss, logits), grads = nn.value_and_grad(model, loss_fn)(model)

    # Update model
    optimizer.update(model, grads)

    # Compute metrics
    predictions = mx.argmax(logits, axis=-1)
    cell_correct = (predictions == answers).astype(mx.float32)
    cell_accuracy = mx.mean(cell_correct).item()

    # Exact match (all cells correct)
    board_correct = mx.all(predictions == answers, axis=-1).astype(mx.float32)
    exact_accuracy = mx.mean(board_correct).item()

    return {
        "loss": loss.item(),
        "cell_accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
    }


def evaluate(
    model: SudokuStatechart9x9,
    dataset: SudokuExtremeDataset,
    config: TrainConfig,
    max_batches: int = 50,
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: The model
        dataset: Evaluation dataset
        config: Training config
        max_batches: Maximum batches to evaluate

    Returns:
        Dict with evaluation metrics
    """
    total_loss = 0.0
    total_cell_correct = 0
    total_exact_correct = 0
    total_cells = 0
    total_boards = 0
    total_violations = 0

    for i, (questions, answers) in enumerate(dataset.iter_batches(config.batch_size, shuffle=False)):
        if i >= max_batches:
            break

        # Forward pass
        logits, _ = model.forward(questions)

        # Loss
        loss = sudoku_loss(logits, answers, use_stablemax=config.use_stablemax)
        total_loss += loss.item()

        # Predictions
        predictions = mx.argmax(logits, axis=-1)

        # Cell accuracy
        cell_correct = mx.sum((predictions == answers).astype(mx.float32)).item()
        total_cell_correct += cell_correct
        total_cells += predictions.size

        # Exact accuracy
        board_correct = mx.sum(mx.all(predictions == answers, axis=-1).astype(mx.float32)).item()
        total_exact_correct += board_correct
        total_boards += predictions.shape[0]

        # Constraint violations
        valid, _ = model.is_valid(predictions)
        total_violations += mx.sum((~valid).astype(mx.float32)).item()

    return {
        "loss": total_loss / max(i + 1, 1),
        "cell_accuracy": total_cell_correct / max(total_cells, 1),
        "exact_accuracy": total_exact_correct / max(total_boards, 1),
        "violation_rate": total_violations / max(total_boards, 1),
    }


def train(config: TrainConfig):
    """Main training loop."""
    print("=" * 60)
    print("Training Sudoku Statechart 9x9")
    print("=" * 60)
    print(f"\nConfig: {json.dumps(asdict(config), indent=2)}")

    # Set seed
    mx.random.seed(config.seed)

    # Create datasets
    print("\nLoading datasets...")
    try:
        train_dataset = SudokuExtremeDataset(
            split="train",
            max_samples=config.max_samples,
            min_difficulty=config.min_difficulty,
            num_augmentations=config.num_augmentations,
            seed=config.seed,
        )
        test_dataset = SudokuExtremeDataset(
            split="test",
            max_samples=min(1000, config.max_samples or 1000),
            seed=config.seed + 1,
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data for testing...")
        # Fallback to synthetic data
        train_dataset = None
        test_dataset = None

    # Create model
    print("\nCreating model...")
    model = SudokuStatechart9x9(
        hidden_dim=config.hidden_dim,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        learned_guards=config.learned_guards,
    )

    # Count parameters
    num_params = sum(p.size for p in nn.utils.tree_flatten(model.parameters())[0])
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    best_exact_accuracy = 0.0
    step = 0
    history = []

    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        if train_dataset is None:
            # Use synthetic data
            for batch_idx in range(100):
                questions = mx.random.randint(0, 10, (config.batch_size, 81))
                answers = mx.random.randint(1, 10, (config.batch_size, 81))

                metrics = train_step(model, optimizer, questions, answers, config)
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                step += 1

                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}: loss={metrics['loss']:.4f}, "
                          f"cell_acc={metrics['cell_accuracy']:.4f}, "
                          f"exact_acc={metrics['exact_accuracy']:.4f}")
        else:
            for questions, answers in train_dataset.iter_batches(config.batch_size):
                metrics = train_step(model, optimizer, questions, answers, config)
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                step += 1

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_steps, 1)

        print(f"\nEpoch {epoch + 1}/{config.epochs}: "
              f"loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        # Evaluate
        if (epoch + 1) % config.eval_every == 0:
            print("  Evaluating...")
            if test_dataset is not None:
                eval_metrics = evaluate(model, test_dataset, config)
            else:
                eval_metrics = {"loss": 0, "cell_accuracy": 0, "exact_accuracy": 0, "violation_rate": 0}

            print(f"  Eval: loss={eval_metrics['loss']:.4f}, "
                  f"cell_acc={eval_metrics['cell_accuracy']:.4f}, "
                  f"exact_acc={eval_metrics['exact_accuracy']:.4f}, "
                  f"violations={eval_metrics['violation_rate']:.4f}")

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **eval_metrics,
            })

            # Save best model
            if eval_metrics["exact_accuracy"] > best_exact_accuracy:
                best_exact_accuracy = eval_metrics["exact_accuracy"]
                save_path = os.path.join(config.save_dir, "best_model.safetensors")
                nn.utils.save(save_path, dict(nn.utils.tree_flatten(model.parameters())))
                print(f"  Saved best model (exact_acc={best_exact_accuracy:.4f})")

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_path = os.path.join(config.save_dir, f"checkpoint_epoch{epoch + 1}.safetensors")
            nn.utils.save(save_path, dict(nn.utils.tree_flatten(model.parameters())))

    # Save final model
    save_path = os.path.join(config.save_dir, "final_model.safetensors")
    nn.utils.save(save_path, dict(nn.utils.tree_flatten(model.parameters())))

    # Save history
    history_path = os.path.join(config.save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete. Best exact accuracy: {best_exact_accuracy:.4f}")
    print(f"Models saved to: {config.save_dir}")
    print("=" * 60)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Sudoku Statechart 9x9")

    # Data args
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_difficulty", type=int, default=None)
    parser.add_argument("--num_augmentations", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)

    # Model args
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--no_learned_guards", action="store_true")

    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--no_stablemax", action="store_true")
    parser.add_argument("--constraint_weight", type=float, default=0.0)

    # Misc args
    parser.add_argument("--save_dir", type=str, default="checkpoints/exp_trm_sudoku_9x9")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainConfig(
        max_samples=args.max_samples,
        min_difficulty=args.min_difficulty,
        num_augmentations=args.num_augmentations,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        learned_guards=not args.no_learned_guards,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_stablemax=not args.no_stablemax,
        constraint_weight=args.constraint_weight,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    train(config)


if __name__ == "__main__":
    main()
