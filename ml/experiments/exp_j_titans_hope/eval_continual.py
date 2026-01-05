"""
Continual Learning Evaluation for HOPE/Nested Learning Architectures.

Implements class-incremental learning benchmark:
1. Train on task 0 (classes 0-1)
2. Train on task 1 (classes 2-3)
3. ... continue adding tasks
4. Measure accuracy on all seen classes after each task

Key metrics:
- Accuracy per task after training
- Backward transfer (forgetting): accuracy drop on old tasks
- Forward transfer: initial accuracy on new tasks
- Average accuracy across all tasks
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Dict
import time

try:
    from .continuum_memory import HOPEStatechart, ContinuumMemorySystem
except ImportError:
    from continuum_memory import HOPEStatechart, ContinuumMemorySystem


class SyntheticClassDataset:
    """Synthetic dataset for class-incremental learning.

    Each class is a Gaussian distribution in hidden_dim space.
    Classes are well-separated to make the task learnable.
    """

    def __init__(self, hidden_dim: int, num_classes: int, samples_per_class: int = 100):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        # Generate class centers (well-separated)
        mx.random.seed(42)
        self.class_centers = mx.random.normal((num_classes, hidden_dim)) * 3.0

        # Store samples per class for easy retrieval
        self.class_samples = {}
        self.class_labels = {}

        for c in range(num_classes):
            center = self.class_centers[c]
            # Samples around the center with some noise
            samples = center + mx.random.normal((samples_per_class, hidden_dim)) * 0.5
            labels = mx.full((samples_per_class,), c, dtype=mx.int32)
            self.class_samples[c] = samples
            self.class_labels[c] = labels

        # Also store concatenated for convenience
        self.all_samples = mx.concatenate([self.class_samples[c] for c in range(num_classes)], axis=0)
        self.all_labels = mx.concatenate([self.class_labels[c] for c in range(num_classes)], axis=0)

    def get_task_data(self, task_id: int, classes_per_task: int = 2) -> Tuple[mx.array, mx.array]:
        """Get samples for a specific task (subset of classes)."""
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        samples_list = [self.class_samples[c] for c in range(start_class, end_class)]
        labels_list = [self.class_labels[c] for c in range(start_class, end_class)]

        samples = mx.concatenate(samples_list, axis=0)
        labels = mx.concatenate(labels_list, axis=0)

        return samples, labels

    def get_all_seen_data(self, num_tasks: int, classes_per_task: int = 2) -> Tuple[mx.array, mx.array]:
        """Get all data from tasks 0 to num_tasks-1."""
        end_class = num_tasks * classes_per_task

        samples_list = [self.class_samples[c] for c in range(end_class)]
        labels_list = [self.class_labels[c] for c in range(end_class)]

        samples = mx.concatenate(samples_list, axis=0)
        labels = mx.concatenate(labels_list, axis=0)

        return samples, labels


class ContinualClassifier(nn.Module):
    """Classifier using HOPE architecture for continual learning."""

    def __init__(self, hidden_dim: int, max_classes: int, num_levels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_classes = max_classes

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # HOPE backbone (statechart with continuum memory)
        self.hope = HOPEStatechart(hidden_dim=hidden_dim, num_levels=num_levels, num_heads=4)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, max_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, hidden_dim] input features

        Returns:
            [B, max_classes] logits
        """
        # Add sequence dimension for HOPE
        x = self.input_proj(x)
        x = x[:, None, :]  # [B, 1, hidden_dim]

        # Process through HOPE
        out, _ = self.hope(x)

        # Remove sequence dimension and classify
        out = out[:, 0, :]  # [B, hidden_dim]
        logits = self.classifier(out)

        return logits

    def reset(self):
        """Reset HOPE memory state."""
        self.hope.reset()


class BaselineClassifier(nn.Module):
    """Simple MLP baseline (no memory mechanism)."""

    def __init__(self, hidden_dim: int, max_classes: int):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, max_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.l1(x), 0)  # ReLU
        x = mx.maximum(self.l2(x), 0)  # ReLU
        return self.classifier(x)

    def reset(self):
        pass


class MemoryClassifier(nn.Module):
    """Classifier with exp_b/exp_i memory backend for comparison."""

    def __init__(self, hidden_dim: int, max_classes: int, memory_type: str = "attention"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_classes = max_classes
        self.memory_type = memory_type

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize memory based on type
        self.memory = None
        self._init_memory(memory_type, hidden_dim)

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, max_classes)

        # Memory state
        self._memory_state = None

    def _init_memory(self, memory_type: str, hidden_dim: int):
        """Initialize memory module."""
        try:
            if memory_type == "attention":
                from ml.differentiable.exp_b_memory import SoftAttentionMemory
                self.memory = SoftAttentionMemory(hidden_dim, d_memory=hidden_dim // 2)
            elif memory_type == "recurrent":
                from ml.differentiable.exp_b_memory import RecurrentMemory
                self.memory = RecurrentMemory(hidden_dim, hidden_size=hidden_dim)
            elif memory_type == "htm":
                from ml.differentiable.exp_i_htm_memory import HTMMemory
                self.memory = HTMMemory(hidden_dim, sdr_size=hidden_dim, d_pooled=hidden_dim // 2)
        except ImportError as e:
            print(f"Warning: Could not import {memory_type} memory: {e}")
            self.memory = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with memory."""
        batch_size = x.shape[0]

        # Project input
        h = mx.tanh(self.input_proj(x))

        # Apply memory if available
        if self.memory is not None:
            if self._memory_state is None:
                self._memory_state = self.memory.init_state(batch_size)
            retrieved, self._memory_state, _ = self.memory.step(h, self._memory_state)
            h = h + retrieved  # Residual connection

        # Classify
        return self.classifier(h)

    def reset(self):
        """Reset memory state."""
        self._memory_state = None


def compute_accuracy(model: nn.Module, samples: mx.array, labels: mx.array) -> float:
    """Compute classification accuracy."""
    logits = model(samples)
    predictions = mx.argmax(logits, axis=1)
    correct = mx.sum(predictions == labels)
    return float(correct) / len(labels)


def train_on_task(
    model: nn.Module,
    optimizer: optim.Optimizer,
    samples: mx.array,
    labels: mx.array,
    num_epochs: int = 50,
    batch_size: int = 32,
) -> List[float]:
    """Train model on a single task."""
    losses = []
    num_samples = len(samples)

    def loss_fn(model, x, y):
        logits = model(x)
        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        loss = -mx.mean(mx.sum(mx.eye(model.max_classes)[y] * log_probs, axis=-1))
        return loss

    for epoch in range(num_epochs):
        # Shuffle data
        perm = mx.random.permutation(num_samples)
        samples_shuffled = samples[perm]
        labels_shuffled = labels[perm]

        epoch_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            batch_x = samples_shuffled[i:i+batch_size]
            batch_y = labels_shuffled[i:i+batch_size]

            loss, grads = nn.value_and_grad(model, loss_fn)(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)

        losses.append(epoch_loss / num_batches)

    return losses


def evaluate_continual_learning(
    model: nn.Module,
    dataset: SyntheticClassDataset,
    num_tasks: int,
    classes_per_task: int = 2,
    epochs_per_task: int = 30,
) -> Dict:
    """Run full continual learning evaluation.

    Returns dict with:
    - task_accuracies: accuracy matrix [num_tasks, num_tasks]
      task_accuracies[i][j] = accuracy on task j after training on task i
    - forgetting: per-task forgetting metrics
    - avg_accuracy: average accuracy after all tasks
    """
    results = {
        "task_accuracies": [],
        "forgetting": [],
        "avg_accuracy": 0.0,
    }

    # Learning rate for optimizer
    optimizer = optim.Adam(learning_rate=0.001)

    for task_id in range(num_tasks):
        print(f"\n--- Training on Task {task_id} ---")

        # Get task data
        task_samples, task_labels = dataset.get_task_data(task_id, classes_per_task)
        print(f"Task {task_id}: {len(task_samples)} samples, classes {task_id * classes_per_task}-{(task_id + 1) * classes_per_task - 1}")

        # Train on current task
        model.reset()
        losses = train_on_task(model, optimizer, task_samples, task_labels, epochs_per_task)
        print(f"Final training loss: {losses[-1]:.4f}")

        # Evaluate on all seen tasks
        task_accs = []
        for eval_task in range(task_id + 1):
            eval_samples, eval_labels = dataset.get_task_data(eval_task, classes_per_task)
            acc = compute_accuracy(model, eval_samples, eval_labels)
            task_accs.append(acc)
            print(f"  Accuracy on Task {eval_task}: {acc:.3f}")

        results["task_accuracies"].append(task_accs)

    # Compute forgetting for each task
    for task_id in range(num_tasks - 1):
        # Accuracy right after training on task
        peak_acc = results["task_accuracies"][task_id][task_id]
        # Accuracy after training on all tasks
        final_acc = results["task_accuracies"][-1][task_id]
        forgetting = peak_acc - final_acc
        results["forgetting"].append(forgetting)

    # Compute average accuracy on all tasks after training
    final_accs = results["task_accuracies"][-1]
    results["avg_accuracy"] = sum(final_accs) / len(final_accs)

    return results


def run_evaluation():
    """Run the complete continual learning evaluation."""
    print("=" * 60)
    print("Continual Learning Evaluation (HOPE vs Baseline)")
    print("=" * 60)

    # Configuration
    hidden_dim = 64
    num_classes = 10
    num_tasks = 5  # 2 classes per task
    classes_per_task = 2
    samples_per_class = 100

    # Create dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticClassDataset(hidden_dim, num_classes, samples_per_class)
    print(f"Dataset: {num_classes} classes, {samples_per_class} samples each")
    print(f"Total samples: {len(dataset.all_samples)}")

    # Evaluate HOPE model
    print("\n" + "=" * 60)
    print("Evaluating HOPE Statechart (with Continuum Memory)")
    print("=" * 60)

    hope_model = ContinualClassifier(hidden_dim, num_classes, num_levels=4)
    start_time = time.time()
    hope_results = evaluate_continual_learning(
        hope_model, dataset, num_tasks, classes_per_task, epochs_per_task=30
    )
    hope_time = time.time() - start_time

    # Evaluate baseline model
    print("\n" + "=" * 60)
    print("Evaluating Baseline MLP (no memory)")
    print("=" * 60)

    baseline_model = BaselineClassifier(hidden_dim, num_classes)
    baseline_model.max_classes = num_classes  # Add for loss function
    start_time = time.time()
    baseline_results = evaluate_continual_learning(
        baseline_model, dataset, num_tasks, classes_per_task, epochs_per_task=30
    )
    baseline_time = time.time() - start_time

    # Evaluate memory-augmented models (exp_b/exp_i)
    all_results = {
        "HOPE": hope_results,
        "Baseline": baseline_results,
    }
    all_times = {
        "HOPE": hope_time,
        "Baseline": baseline_time,
    }

    for mem_type in ["attention", "recurrent", "htm"]:
        print("\n" + "=" * 60)
        print(f"Evaluating {mem_type.upper()} Memory Classifier")
        print("=" * 60)

        try:
            mem_model = MemoryClassifier(hidden_dim, num_classes, memory_type=mem_type)
            if mem_model.memory is not None:
                start_time = time.time()
                mem_results = evaluate_continual_learning(
                    mem_model, dataset, num_tasks, classes_per_task, epochs_per_task=30
                )
                mem_time = time.time() - start_time
                all_results[mem_type.upper()] = mem_results
                all_times[mem_type.upper()] = mem_time
            else:
                print(f"  Skipped (memory not available)")
        except Exception as e:
            print(f"  Skipped due to error: {e}")

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    models = list(all_results.keys())
    header = f"{'Task':>6} |" + "".join(f" {m:>10} |" for m in models)
    print("\nFinal Accuracy per Task:")
    print(header)
    print("-" * len(header))
    for task_id in range(num_tasks):
        row = f"{task_id:>6} |"
        for m in models:
            acc = all_results[m]["task_accuracies"][-1][task_id]
            row += f" {acc:>10.3f} |"
        print(row)

    print("\nForgetting (accuracy drop on old tasks):")
    print(header.replace("Task", "Task"))
    print("-" * len(header))
    for task_id in range(num_tasks - 1):
        row = f"{task_id:>6} |"
        for m in models:
            fgt = all_results[m]["forgetting"][task_id]
            row += f" {fgt:>10.3f} |"
        print(row)

    print("\nSummary:")
    print(f"{'Model':<12} | {'Avg Acc':>8} | {'Avg Fgt':>8} | {'Time':>8}")
    print("-" * 46)
    for m in models:
        avg_acc = all_results[m]["avg_accuracy"]
        avg_fgt = sum(all_results[m]["forgetting"]) / len(all_results[m]["forgetting"])
        t = all_times[m]
        print(f"{m:<12} | {avg_acc:>8.3f} | {avg_fgt:>8.3f} | {t:>7.1f}s")

    # Find best model for forgetting
    best_fgt = min(models, key=lambda m: sum(all_results[m]["forgetting"]))
    print(f"\nLowest forgetting: {best_fgt}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_evaluation()
