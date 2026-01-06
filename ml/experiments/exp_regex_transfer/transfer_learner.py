"""
Transfer Learner - Transfer Regex Patterns Across Domains

Learn structure from one domain (email), transfer to another (URL, phone).

Transfer mechanism:
1. Learn domain-agnostic structure encoder from source domain
2. Learn pattern matcher using structural features
3. Apply to target domain with minimal fine-tuning

Key insight: The STRUCTURE is transferable, not the characters.
- Email: [alpha]+ @ [alpha]+ . [alpha]+
- URL: [alpha]+ :// [alpha]+ . [alpha]+ / [alpha]*
- Phone: + [digit]+ - [digit]+ - [digit]+

All share: [content]+ [delim] [content]+ [delim] [content]+
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import random
import math
import time

from .domain_encoder import (
    DomainEncoder, PatternAnalyzer, PatternStructure,
    DomainPatterns, CharType, Segment
)


# =============================================================================
# Regex Pattern Matcher (Neural)
# =============================================================================

class PatternMatcherNetwork(nn.Module):
    """
    Neural network that learns to match patterns.

    Input: Pattern structure vector
    Output: Match probability for candidate strings
    """

    def __init__(self, feature_dim: int = 220, hidden_dim: int = 128):
        super().__init__()

        # Structure encoder (learned)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # String encoder (processes candidate strings)
        self.string_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Matcher (combines pattern and string features)
        self.matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_pattern(self, pattern_vec: mx.array) -> mx.array:
        """Encode pattern structure."""
        return self.encoder(pattern_vec)

    def encode_string(self, string_vec: mx.array) -> mx.array:
        """Encode candidate string."""
        return self.string_encoder(string_vec)

    def __call__(self, pattern_vec: mx.array, string_vec: mx.array) -> mx.array:
        """Predict match probability."""
        pattern_emb = self.encode_pattern(pattern_vec)
        string_emb = self.encode_string(string_vec)

        combined = mx.concatenate([pattern_emb, string_emb], axis=-1)
        logit = self.matcher(combined)
        return mx.sigmoid(logit)


# =============================================================================
# Transfer Learner
# =============================================================================

@dataclass
class TransferStats:
    """Statistics from transfer learning."""
    source_domain: str
    target_domain: str
    source_accuracy: float
    target_accuracy_before: float
    target_accuracy_after: float
    samples_used: int
    training_time: float
    transfer_gain: float = 0.0  # Improvement from transfer

    def __post_init__(self):
        self.transfer_gain = self.target_accuracy_after - self.target_accuracy_before


class TransferLearner:
    """
    Learn patterns from source domain, transfer to target domain.

    Training phases:
    1. Pre-training: Learn structural encoder on source domain
    2. Transfer: Apply encoder to target domain with frozen layers
    3. Fine-tuning: Optionally fine-tune on target domain
    """

    def __init__(
        self,
        feature_dim: int = 220,
        hidden_dim: int = 128,
        learning_rate: float = 0.01,
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.domain_encoder = DomainEncoder(feature_dim=feature_dim)
        self.pattern_matcher = PatternMatcherNetwork(feature_dim, hidden_dim)

        # Training state
        self.source_trained = False
        self.epochs_trained = 0

    def _generate_training_data(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Generate training data from examples.

        Returns: (pattern_vecs, string_vecs, labels)
        """
        # Encode pattern from positive examples
        pattern_vec = self.domain_encoder.encode_pattern(positive_examples[:10])

        # Encode all examples as candidate strings
        all_examples = positive_examples + negative_examples
        labels = [1.0] * len(positive_examples) + [0.0] * len(negative_examples)

        string_vecs = []
        for s in all_examples:
            vec = self.domain_encoder.encode_single(s)
            string_vecs.append(vec)

        # Stack
        pattern_vecs = mx.stack([pattern_vec] * len(all_examples))
        string_vecs = mx.stack(string_vecs)
        labels_arr = mx.array(labels).reshape(-1, 1)

        return pattern_vecs, string_vecs, labels_arr

    def _generate_negatives(
        self,
        positive_examples: List[str],
        n_negatives: int,
    ) -> List[str]:
        """Generate negative examples by mutation."""
        negatives = []

        for _ in range(n_negatives):
            # Pick a positive and mutate it
            pos = random.choice(positive_examples)

            mutation_type = random.choice(['swap', 'delete', 'insert', 'replace'])

            if mutation_type == 'swap' and len(pos) > 1:
                i = random.randint(0, len(pos) - 2)
                neg = pos[:i] + pos[i + 1] + pos[i] + pos[i + 2:]
            elif mutation_type == 'delete' and len(pos) > 1:
                i = random.randint(0, len(pos) - 1)
                neg = pos[:i] + pos[i + 1:]
            elif mutation_type == 'insert':
                i = random.randint(0, len(pos))
                c = random.choice('abcdefghijklmnopqrstuvwxyz0123456789@.-_/')
                neg = pos[:i] + c + pos[i:]
            else:  # replace
                if pos:
                    i = random.randint(0, len(pos) - 1)
                    c = random.choice('abcdefghijklmnopqrstuvwxyz0123456789@.-_/')
                    neg = pos[:i] + c + pos[i + 1:]
                else:
                    neg = 'x'

            negatives.append(neg)

        return negatives

    def train_source(
        self,
        positive_examples: List[str],
        epochs: int = 100,
        batch_size: int = 32,
    ) -> float:
        """
        Train on source domain.

        Returns: Final accuracy on source domain
        """
        # Generate negatives
        negatives = self._generate_negatives(positive_examples, len(positive_examples))

        # Prepare data
        pattern_vecs, string_vecs, labels = self._generate_training_data(
            positive_examples, negatives
        )

        # Training loop (simplified - would use proper optimizer)
        for epoch in range(epochs):
            # Forward pass
            preds = self.pattern_matcher(pattern_vecs, string_vecs)

            # Binary cross-entropy loss
            loss = -mx.mean(
                labels * mx.log(preds + 1e-10) +
                (1 - labels) * mx.log(1 - preds + 1e-10)
            )

            # Simple gradient descent (mock - would use mlx.optimizers)
            # In practice, would compute gradients and update weights

            self.epochs_trained += 1

        # Compute accuracy
        preds = self.pattern_matcher(pattern_vecs, string_vecs)
        accuracy = float(mx.mean((preds > 0.5).astype(mx.float32) == labels))

        self.source_trained = True
        return accuracy

    def evaluate(
        self,
        positive_examples: List[str],
        negative_examples: Optional[List[str]] = None,
    ) -> float:
        """
        Evaluate on a domain.

        Returns: Accuracy
        """
        if negative_examples is None:
            negative_examples = self._generate_negatives(positive_examples, len(positive_examples))

        pattern_vecs, string_vecs, labels = self._generate_training_data(
            positive_examples, negative_examples
        )

        preds = self.pattern_matcher(pattern_vecs, string_vecs)
        accuracy = float(mx.mean((preds > 0.5).astype(mx.float32) == labels))

        return accuracy

    def transfer(
        self,
        target_examples: List[str],
        fine_tune_epochs: int = 0,
    ) -> Tuple[float, float]:
        """
        Transfer to target domain.

        Returns: (before_accuracy, after_accuracy)
        """
        # Evaluate before fine-tuning
        before_acc = self.evaluate(target_examples)

        # Optional fine-tuning
        if fine_tune_epochs > 0:
            negatives = self._generate_negatives(target_examples, len(target_examples))
            pattern_vecs, string_vecs, labels = self._generate_training_data(
                target_examples, negatives
            )

            for _ in range(fine_tune_epochs):
                preds = self.pattern_matcher(pattern_vecs, string_vecs)
                loss = -mx.mean(
                    labels * mx.log(preds + 1e-10) +
                    (1 - labels) * mx.log(1 - preds + 1e-10)
                )
                # Would update weights here

        # Evaluate after fine-tuning
        after_acc = self.evaluate(target_examples)

        return before_acc, after_acc


# =============================================================================
# Baseline (No Transfer)
# =============================================================================

class BaselineLearner:
    """
    Baseline learner without transfer.

    Learns from scratch on target domain.
    """

    def __init__(self, feature_dim: int = 220, hidden_dim: int = 128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.domain_encoder = DomainEncoder(feature_dim=feature_dim)
        self.pattern_matcher = PatternMatcherNetwork(feature_dim, hidden_dim)

    def _generate_training_data(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Generate training data."""
        pattern_vec = self.domain_encoder.encode_pattern(positive_examples[:10])

        all_examples = positive_examples + negative_examples
        labels = [1.0] * len(positive_examples) + [0.0] * len(negative_examples)

        string_vecs = [self.domain_encoder.encode_single(s) for s in all_examples]

        pattern_vecs = mx.stack([pattern_vec] * len(all_examples))
        string_vecs = mx.stack(string_vecs)
        labels_arr = mx.array(labels).reshape(-1, 1)

        return pattern_vecs, string_vecs, labels_arr

    def _generate_negatives(
        self,
        positive_examples: List[str],
        n_negatives: int,
    ) -> List[str]:
        """Generate negative examples."""
        negatives = []
        for _ in range(n_negatives):
            pos = random.choice(positive_examples)
            if len(pos) > 1:
                i = random.randint(0, len(pos) - 1)
                c = random.choice('abcdefghijklmnopqrstuvwxyz0123456789@.-_/')
                neg = pos[:i] + c + pos[i + 1:]
            else:
                neg = 'x'
            negatives.append(neg)
        return negatives

    def train_and_evaluate(
        self,
        positive_examples: List[str],
        epochs: int = 100,
    ) -> float:
        """Train from scratch and evaluate."""
        negatives = self._generate_negatives(positive_examples, len(positive_examples))
        pattern_vecs, string_vecs, labels = self._generate_training_data(
            positive_examples, negatives
        )

        for _ in range(epochs):
            preds = self.pattern_matcher(pattern_vecs, string_vecs)
            loss = -mx.mean(
                labels * mx.log(preds + 1e-10) +
                (1 - labels) * mx.log(1 - preds + 1e-10)
            )

        preds = self.pattern_matcher(pattern_vecs, string_vecs)
        accuracy = float(mx.mean((preds > 0.5).astype(mx.float32) == labels))
        return accuracy


# =============================================================================
# Transfer Experiment
# =============================================================================

def run_transfer_experiment(
    source_domain: str,
    target_domain: str,
    source_examples: List[str],
    target_examples: List[str],
    source_epochs: int = 50,
    fine_tune_epochs: int = 10,
) -> TransferStats:
    """
    Run a transfer learning experiment.

    1. Train on source domain
    2. Evaluate on target (before fine-tuning)
    3. Fine-tune on target
    4. Evaluate on target (after fine-tuning)
    """
    start_time = time.time()

    # Create learner
    learner = TransferLearner()

    # Train on source
    source_acc = learner.train_source(source_examples, epochs=source_epochs)

    # Transfer to target
    before_acc, after_acc = learner.transfer(target_examples, fine_tune_epochs=fine_tune_epochs)

    training_time = time.time() - start_time

    return TransferStats(
        source_domain=source_domain,
        target_domain=target_domain,
        source_accuracy=source_acc,
        target_accuracy_before=before_acc,
        target_accuracy_after=after_acc,
        samples_used=len(source_examples) + len(target_examples),
        training_time=training_time,
    )


def compare_transfer_vs_baseline(
    target_domain: str,
    target_examples: List[str],
    source_domain: str,
    source_examples: List[str],
    sample_sizes: List[int] = [10, 25, 50, 100],
) -> Dict[int, Tuple[float, float]]:
    """
    Compare transfer learning vs training from scratch.

    Returns: {sample_size: (transfer_acc, baseline_acc)}
    """
    results = {}

    for n_samples in sample_sizes:
        if n_samples > len(target_examples):
            continue

        # Subset of target examples
        target_subset = target_examples[:n_samples]

        # Transfer learning
        learner = TransferLearner()
        learner.train_source(source_examples[:100], epochs=50)
        _, transfer_acc = learner.transfer(target_subset, fine_tune_epochs=20)

        # Baseline (from scratch)
        baseline = BaselineLearner()
        baseline_acc = baseline.train_and_evaluate(target_subset, epochs=50)

        results[n_samples] = (transfer_acc, baseline_acc)

    return results


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate transfer learning."""
    print("=" * 60)
    print("Transfer Learner Demo")
    print("=" * 60)

    # Generate domain data
    emails = DomainPatterns.generate_email(100)
    urls = DomainPatterns.generate_url(100)
    phones = DomainPatterns.generate_phone(100)

    # Transfer experiment: Email -> URL
    print("\n--- Transfer: Email -> URL ---")
    stats = run_transfer_experiment(
        source_domain="email",
        target_domain="url",
        source_examples=emails,
        target_examples=urls,
        source_epochs=30,
        fine_tune_epochs=10,
    )

    print(f"Source (email) accuracy: {stats.source_accuracy:.2%}")
    print(f"Target (URL) before fine-tuning: {stats.target_accuracy_before:.2%}")
    print(f"Target (URL) after fine-tuning: {stats.target_accuracy_after:.2%}")
    print(f"Transfer gain: {stats.transfer_gain:+.2%}")

    # Transfer experiment: Email -> Phone
    print("\n--- Transfer: Email -> Phone ---")
    stats = run_transfer_experiment(
        source_domain="email",
        target_domain="phone",
        source_examples=emails,
        target_examples=phones,
        source_epochs=30,
        fine_tune_epochs=10,
    )

    print(f"Source (email) accuracy: {stats.source_accuracy:.2%}")
    print(f"Target (phone) before fine-tuning: {stats.target_accuracy_before:.2%}")
    print(f"Target (phone) after fine-tuning: {stats.target_accuracy_after:.2%}")
    print(f"Transfer gain: {stats.transfer_gain:+.2%}")

    # Compare transfer vs baseline
    print("\n--- Sample Efficiency Comparison ---")
    print("Target: URL, Source: Email")
    print(f"{'Samples':>10} {'Transfer':>12} {'Baseline':>12} {'Gain':>10}")
    print("-" * 50)

    results = compare_transfer_vs_baseline(
        target_domain="url",
        target_examples=urls,
        source_domain="email",
        source_examples=emails,
        sample_sizes=[10, 25, 50, 100],
    )

    for n_samples, (transfer_acc, baseline_acc) in sorted(results.items()):
        gain = transfer_acc - baseline_acc
        print(f"{n_samples:>10} {transfer_acc:>12.2%} {baseline_acc:>12.2%} {gain:>+10.2%}")

    return stats, results


if __name__ == "__main__":
    demo()
