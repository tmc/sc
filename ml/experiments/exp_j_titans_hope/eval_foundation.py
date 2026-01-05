"""
Foundation Capabilities Evaluation for HOPE/Nested Learning.

Measures:
1. Perplexity on synthetic sequences (pattern prediction)
2. Simple reasoning tasks (counting, copying, reversal)
3. State tracking (finite state machine simulation)

These are baseline capabilities that any memory-augmented model should have.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Dict
import math
import time

try:
    from .continuum_memory import HOPEStatechart, ContinuumMemorySystem
except ImportError:
    from continuum_memory import HOPEStatechart, ContinuumMemorySystem


# =============================================================================
# Perplexity Evaluation
# =============================================================================


class SyntheticSequenceDataset:
    """Generates synthetic sequences with learnable patterns."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        mx.random.seed(42)
        self.embeddings = mx.random.normal((vocab_size, hidden_dim)) * 0.1

    def generate_repetition_sequence(self, length: int, period: int = 5) -> mx.array:
        """Generate repeating pattern: ABCDE ABCDE ABCDE..."""
        pattern = mx.random.randint(0, self.vocab_size, (period,))
        repeats = (length + period - 1) // period
        seq = mx.tile(pattern, (repeats,))[:length]
        return seq

    def generate_counting_sequence(self, length: int, modulo: int = 10) -> mx.array:
        """Generate counting pattern: 0 1 2 3 ... mod N"""
        return mx.arange(length) % modulo

    def generate_copying_sequence(self, prefix_len: int, copy_len: int) -> Tuple[mx.array, mx.array]:
        """Generate copy task: [prefix] [separator] [copy of prefix]."""
        prefix = mx.random.randint(0, self.vocab_size - 1, (prefix_len,))
        separator = mx.array([self.vocab_size - 1])  # Special separator token

        input_seq = mx.concatenate([prefix, separator, mx.zeros((copy_len,), dtype=mx.int32)])
        target_seq = mx.concatenate([prefix, separator, prefix[:copy_len]])

        return input_seq, target_seq

    def generate_reversal_sequence(self, length: int) -> Tuple[mx.array, mx.array]:
        """Generate reversal task: [seq] [sep] [reversed seq]."""
        seq = mx.random.randint(0, self.vocab_size - 1, (length,))
        separator = mx.array([self.vocab_size - 1])

        # Reverse using list slicing (MLX doesn't have direct reverse)
        reversed_list = seq.tolist()[::-1]
        reversed_seq = mx.array(reversed_list)

        input_seq = mx.concatenate([seq, separator, mx.zeros((length,), dtype=mx.int32)])
        target_seq = mx.concatenate([seq, separator, reversed_seq])

        return input_seq, target_seq

    def embed(self, token_ids: mx.array) -> mx.array:
        """Convert token IDs to embeddings."""
        return self.embeddings[token_ids]


class SequenceModel(nn.Module):
    """HOPE-based sequence model for perplexity evaluation."""

    def __init__(self, hidden_dim: int, vocab_size: int, num_levels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hope = HOPEStatechart(hidden_dim=hidden_dim, num_levels=num_levels)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, L, hidden_dim] input embeddings
        Returns:
            [B, L, vocab_size] logits
        """
        h = self.input_proj(x)
        out, _ = self.hope(h)
        return self.output_head(out)

    def reset(self):
        self.hope.reset()


class BaselineSequenceModel(nn.Module):
    """Simple MLP baseline (no temporal structure)."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.tanh(self.l1(x))
        h = mx.tanh(self.l2(h))
        return self.output_head(h)

    def reset(self):
        pass


def compute_perplexity(
    model: nn.Module,
    sequences: List[mx.array],
    embeddings: mx.array,
) -> float:
    """Compute perplexity on a list of sequences."""
    total_loss = 0.0
    total_tokens = 0

    for seq in sequences:
        if len(seq) < 2:
            continue

        # Input: all tokens except last
        # Target: all tokens except first (next-token prediction)
        input_ids = seq[:-1]
        target_ids = seq[1:]

        input_emb = embeddings[input_ids][None, :, :]  # [1, L-1, D]
        logits = model(input_emb)  # [1, L-1, vocab]

        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits[0], axis=-1) + 1e-10)

        # Gather log probs for target tokens
        L = len(target_ids)
        target_log_probs = sum(
            float(log_probs[i, int(target_ids[i].item())])
            for i in range(L)
        )

        total_loss -= target_log_probs
        total_tokens += L

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


# =============================================================================
# Simple Reasoning Tasks
# =============================================================================


def evaluate_counting_task(
    model: nn.Module,
    dataset: SyntheticSequenceDataset,
    num_samples: int = 20,
    seq_length: int = 50,
) -> float:
    """Evaluate next-token prediction on counting sequences."""
    correct = 0
    total = 0

    for _ in range(num_samples):
        modulo = mx.random.randint(5, 15, ()).item()
        seq = dataset.generate_counting_sequence(seq_length, modulo)

        # Predict next token at each position
        input_emb = dataset.embed(seq[:-1])[None, :, :]
        logits = model(input_emb)
        preds = mx.argmax(logits[0], axis=-1)

        # Check predictions
        targets = seq[1:]
        matches = mx.sum(preds == targets)
        correct += int(matches.item())
        total += len(targets)

    return correct / total if total > 0 else 0.0


def evaluate_repetition_task(
    model: nn.Module,
    dataset: SyntheticSequenceDataset,
    num_samples: int = 20,
    seq_length: int = 50,
) -> float:
    """Evaluate pattern repetition prediction."""
    correct = 0
    total = 0

    for _ in range(num_samples):
        period = mx.random.randint(3, 8, ()).item()
        seq = dataset.generate_repetition_sequence(seq_length, period)

        input_emb = dataset.embed(seq[:-1])[None, :, :]
        logits = model(input_emb)
        preds = mx.argmax(logits[0], axis=-1)

        targets = seq[1:]
        # Only count predictions after seeing at least one full period
        start_idx = period
        if start_idx < len(targets):
            matches = mx.sum(preds[start_idx:] == targets[start_idx:])
            correct += int(matches.item())
            total += len(targets) - start_idx

    return correct / total if total > 0 else 0.0


def evaluate_copy_task(
    model: nn.Module,
    dataset: SyntheticSequenceDataset,
    num_samples: int = 20,
    prefix_len: int = 5,
) -> float:
    """Evaluate copying task (memorize and reproduce prefix)."""
    correct = 0
    total = 0

    for _ in range(num_samples):
        input_seq, target_seq = dataset.generate_copying_sequence(prefix_len, prefix_len)

        input_emb = dataset.embed(input_seq)[None, :, :]
        logits = model(input_emb)
        preds = mx.argmax(logits[0], axis=-1)

        # Only check the copy region (after separator)
        copy_start = prefix_len + 1
        if copy_start < len(target_seq):
            copy_preds = preds[copy_start:]
            copy_targets = target_seq[copy_start:]

            min_len = min(len(copy_preds), len(copy_targets))
            if min_len > 0:
                matches = mx.sum(copy_preds[:min_len] == copy_targets[:min_len])
                correct += int(matches.item())
                total += min_len

    return correct / total if total > 0 else 0.0


# =============================================================================
# State Tracking Task (FSM Simulation)
# =============================================================================


def generate_fsm_sequence(length: int, num_states: int = 4) -> Tuple[mx.array, mx.array]:
    """Generate a sequence from a simple FSM.

    FSM: State i transitions to state (i + input) % num_states
    Input tokens are 0 or 1.
    Target is the resulting state after each transition.
    """
    state = 0
    inputs = mx.random.randint(0, 2, (length,))
    states = []

    for i in range(length):
        inp = int(inputs[i].item())
        state = (state + inp) % num_states
        states.append(state)

    return inputs, mx.array(states)


def evaluate_fsm_tracking(
    model: nn.Module,
    dataset: SyntheticSequenceDataset,
    num_samples: int = 20,
    seq_length: int = 30,
    num_states: int = 4,
) -> float:
    """Evaluate state tracking in FSM simulation."""
    correct = 0
    total = 0

    for _ in range(num_samples):
        inputs, target_states = generate_fsm_sequence(seq_length, num_states)

        # Embed inputs (use first num_states tokens as state representations)
        input_emb = dataset.embed(inputs)[None, :, :]
        logits = model(input_emb)

        # Predict states (only use first num_states logits)
        preds = mx.argmax(logits[0, :, :num_states], axis=-1)

        matches = mx.sum(preds == target_states)
        correct += int(matches.item())
        total += len(target_states)

    return correct / total if total > 0 else 0.0


# =============================================================================
# Main Evaluation
# =============================================================================


def run_evaluation():
    """Run all foundation capability evaluations."""
    print("=" * 60)
    print("Foundation Capabilities Evaluation")
    print("=" * 60)

    # Configuration
    hidden_dim = 64
    vocab_size = 100

    # Create dataset
    dataset = SyntheticSequenceDataset(vocab_size, hidden_dim)

    # Create models
    print("\nCreating models...")
    hope_model = SequenceModel(hidden_dim, vocab_size, num_levels=4)
    baseline_model = BaselineSequenceModel(hidden_dim, vocab_size)

    models = {
        "HOPE": hope_model,
        "Baseline": baseline_model,
    }

    results = {name: {} for name in models}

    # 1. Perplexity on synthetic sequences
    print("\n" + "=" * 60)
    print("1. Perplexity Evaluation")
    print("=" * 60)

    # Generate test sequences
    test_seqs = []
    for _ in range(10):
        test_seqs.append(dataset.generate_repetition_sequence(50, period=5))
        test_seqs.append(dataset.generate_counting_sequence(50, modulo=10))

    for name, model in models.items():
        model.reset()
        ppl = compute_perplexity(model, test_seqs, dataset.embeddings)
        results[name]["perplexity"] = ppl
        print(f"  {name}: perplexity = {ppl:.2f}")

    # 2. Counting task
    print("\n" + "=" * 60)
    print("2. Counting Task")
    print("=" * 60)

    for name, model in models.items():
        model.reset()
        acc = evaluate_counting_task(model, dataset)
        results[name]["counting"] = acc
        print(f"  {name}: accuracy = {acc:.3f}")

    # 3. Repetition task
    print("\n" + "=" * 60)
    print("3. Repetition Task")
    print("=" * 60)

    for name, model in models.items():
        model.reset()
        acc = evaluate_repetition_task(model, dataset)
        results[name]["repetition"] = acc
        print(f"  {name}: accuracy = {acc:.3f}")

    # 4. Copy task
    print("\n" + "=" * 60)
    print("4. Copy Task")
    print("=" * 60)

    for name, model in models.items():
        model.reset()
        acc = evaluate_copy_task(model, dataset, prefix_len=5)
        results[name]["copy"] = acc
        print(f"  {name}: accuracy = {acc:.3f}")

    # 5. FSM state tracking
    print("\n" + "=" * 60)
    print("5. FSM State Tracking")
    print("=" * 60)

    for name, model in models.items():
        model.reset()
        acc = evaluate_fsm_tracking(model, dataset)
        results[name]["fsm"] = acc
        print(f"  {name}: accuracy = {acc:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Task':<15} | {'HOPE':>10} | {'Baseline':>10} | {'Delta':>10}")
    print("-" * 55)

    tasks = ["perplexity", "counting", "repetition", "copy", "fsm"]
    task_labels = ["Perplexity", "Counting", "Repetition", "Copy", "FSM Track"]

    for task, label in zip(tasks, task_labels):
        hope_val = results["HOPE"][task]
        base_val = results["Baseline"][task]

        if task == "perplexity":
            # Lower is better for perplexity
            delta = base_val - hope_val
            print(f"{label:<15} | {hope_val:>10.2f} | {base_val:>10.2f} | {delta:>+10.2f}")
        else:
            # Higher is better for accuracy
            delta = hope_val - base_val
            print(f"{label:<15} | {hope_val:>10.3f} | {base_val:>10.3f} | {delta:>+10.3f}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_evaluation()
