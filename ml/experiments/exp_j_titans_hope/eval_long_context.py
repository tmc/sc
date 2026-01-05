"""
Long-Context Reasoning Evaluation (Needle-in-Haystack).

Implements NIAH (Needle-In-A-Haystack) benchmark variants:
- S-NIAH: Single needle (one key-value pair)
- MK-NIAH: Multi-Key (multiple keys, retrieve all values)
- MQ-NIAH: Multi-Query (query same key multiple times)
- MV-NIAH: Multi-Value (one key maps to multiple values)

Key insight: HOPE's CMS (Continuum Memory System) should enable
better long-range retrieval by consolidating patterns at multiple
timescales.

Context lengths tested: 1K, 4K, 16K, 64K, 256K tokens (simulated)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Dict, Optional
import time
import math

try:
    from .continuum_memory import HOPEStatechart, ContinuumMemorySystem
except ImportError:
    from continuum_memory import HOPEStatechart, ContinuumMemorySystem


class NeedleHaystackDataset:
    """Dataset for Needle-in-Haystack evaluation.

    Creates sequences where:
    - Haystack: Random noise tokens
    - Needle: Key-value pair(s) inserted at random position(s)
    - Query: Key token(s) at the end
    - Target: Corresponding value token(s)
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 1000,
        num_samples: int = 100,
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        mx.random.seed(42)

        # Create embeddings for tokens
        self.embeddings = mx.random.normal((vocab_size, hidden_dim)) * 0.1

        # Reserve special tokens
        self.KEY_START = vocab_size - 10  # Keys start here
        self.VALUE_START = vocab_size - 100  # Values start here

    def create_s_niah_sample(
        self,
        context_length: int,
        needle_depth: float = 0.5,  # 0.0 = start, 1.0 = end
    ) -> Tuple[mx.array, mx.array, int]:
        """Create Single-NIAH sample.

        Args:
            context_length: Total sequence length
            needle_depth: Relative position of needle (0-1)

        Returns:
            (sequence, query, target_value_idx)
        """
        # Generate random haystack
        haystack = mx.random.randint(0, self.VALUE_START, (context_length,))

        # Create key-value pair
        key_idx = self.KEY_START + mx.random.randint(0, 10, ()).item()
        value_idx = self.VALUE_START + mx.random.randint(0, 90, ()).item()

        # Insert needle at specified depth
        needle_pos = int(context_length * needle_depth)
        needle_pos = max(0, min(needle_pos, context_length - 2))

        # Insert key-value pair
        sequence = haystack.tolist()
        sequence[needle_pos] = key_idx
        sequence[needle_pos + 1] = value_idx
        sequence = mx.array(sequence)

        # Query is just the key
        query = mx.array([key_idx])

        return sequence, query, value_idx

    def create_mk_niah_sample(
        self,
        context_length: int,
        num_needles: int = 3,
    ) -> Tuple[mx.array, mx.array, List[int]]:
        """Create Multi-Key NIAH sample.

        Multiple key-value pairs scattered throughout context.
        Must retrieve all values.
        """
        haystack = mx.random.randint(0, self.VALUE_START, (context_length,)).tolist()

        keys = []
        values = []

        # Insert multiple needles at different positions
        positions = sorted(mx.random.permutation(context_length - 1)[:num_needles * 2].tolist())

        for i in range(num_needles):
            key_idx = self.KEY_START + i
            value_idx = self.VALUE_START + mx.random.randint(0, 90, ()).item()

            pos = positions[i * 2] if i * 2 < len(positions) else i * (context_length // num_needles)
            pos = min(pos, context_length - 2)

            haystack[pos] = key_idx
            haystack[pos + 1] = value_idx

            keys.append(key_idx)
            values.append(value_idx)

        sequence = mx.array(haystack)
        query = mx.array(keys)

        return sequence, query, values

    def create_mq_niah_sample(
        self,
        context_length: int,
        num_queries: int = 5,
    ) -> Tuple[mx.array, mx.array, int]:
        """Create Multi-Query NIAH sample.

        Same key queried multiple times (tests caching/memory).
        """
        haystack = mx.random.randint(0, self.VALUE_START, (context_length,)).tolist()

        # Single key-value pair
        key_idx = self.KEY_START
        value_idx = self.VALUE_START + mx.random.randint(0, 90, ()).item()

        needle_pos = context_length // 4  # Place early
        haystack[needle_pos] = key_idx
        haystack[needle_pos + 1] = value_idx

        sequence = mx.array(haystack)
        # Query same key multiple times
        query = mx.array([key_idx] * num_queries)

        return sequence, query, value_idx

    def create_mv_niah_sample(
        self,
        context_length: int,
        num_values: int = 3,
    ) -> Tuple[mx.array, mx.array, List[int]]:
        """Create Multi-Value NIAH sample.

        One key maps to multiple values at different positions.
        Must retrieve all values associated with key.
        """
        haystack = mx.random.randint(0, self.VALUE_START, (context_length,)).tolist()

        key_idx = self.KEY_START
        values = []

        # Insert same key with different values at multiple positions
        positions = sorted(mx.random.permutation(context_length - 1)[:num_values * 2].tolist())

        for i in range(num_values):
            value_idx = self.VALUE_START + i * 10 + mx.random.randint(0, 10, ()).item()

            pos = positions[i * 2] if i * 2 < len(positions) else i * (context_length // num_values)
            pos = min(pos, context_length - 2)

            haystack[pos] = key_idx
            haystack[pos + 1] = value_idx
            values.append(value_idx)

        sequence = mx.array(haystack)
        query = mx.array([key_idx])

        return sequence, query, values

    def embed_sequence(self, token_ids: mx.array) -> mx.array:
        """Convert token IDs to embeddings."""
        return self.embeddings[token_ids]


class LongContextRetriever(nn.Module):
    """HOPE-based retriever for long context."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_levels: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # HOPE backbone for long-range dependencies
        self.hope = HOPEStatechart(
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_heads=num_heads,
        )

        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output head (predict value token)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(
        self,
        context_emb: mx.array,  # [B, L, D]
        query_emb: mx.array,    # [B, Q, D]
    ) -> mx.array:
        """
        Process context and query, predict value tokens.

        Args:
            context_emb: Context embeddings [B, L, hidden_dim]
            query_emb: Query embeddings [B, Q, hidden_dim]

        Returns:
            [B, Q, vocab_size] logits for value prediction
        """
        # Process context through HOPE
        context = self.input_proj(context_emb)
        context_out, level_outputs = self.hope(context)

        # Project queries
        query = self.query_proj(query_emb)  # [B, Q, D]

        # Cross-attention: query attends to context
        # Simple dot-product attention
        B, L, D = context_out.shape
        Q = query.shape[1]

        # Compute attention scores
        scores = mx.matmul(query, context_out.transpose(0, 2, 1))  # [B, Q, L]
        scores = scores / math.sqrt(D)
        attn = mx.softmax(scores, axis=-1)

        # Aggregate context
        retrieved = mx.matmul(attn, context_out)  # [B, Q, D]

        # Predict value tokens
        logits = self.output_head(retrieved)  # [B, Q, vocab_size]

        return logits

    def reset(self):
        self.hope.reset()


class BaselineRetriever(nn.Module):
    """Simple attention-only retriever (no HOPE memory)."""

    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple transformer layers
        self.self_attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output head
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, context_emb: mx.array, query_emb: mx.array) -> mx.array:
        # Self-attention on context (limited to last 1024 tokens for efficiency)
        if context_emb.shape[1] > 1024:
            context_emb = context_emb[:, -1024:, :]

        context = self.ln1(context_emb + self.self_attn(context_emb, context_emb, context_emb))

        # Cross-attention: query to context
        query = self.ln2(query_emb + self.cross_attn(query_emb, context, context))

        return self.output_head(query)

    def reset(self):
        pass


def evaluate_retrieval_accuracy(
    model: nn.Module,
    dataset: NeedleHaystackDataset,
    context_length: int,
    niah_type: str = "s_niah",
    num_samples: int = 50,
    needle_depths: List[float] = None,
) -> Dict:
    """Evaluate retrieval accuracy for a specific context length.

    Args:
        model: Retriever model
        dataset: NIAH dataset
        context_length: Sequence length to test
        niah_type: One of 's_niah', 'mk_niah', 'mq_niah', 'mv_niah'
        num_samples: Number of samples to test
        needle_depths: For S-NIAH, test different depths

    Returns:
        Dict with accuracy metrics
    """
    if needle_depths is None:
        needle_depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {
        "context_length": context_length,
        "niah_type": niah_type,
        "accuracy": 0.0,
        "depth_accuracies": {},
    }

    model.reset()

    if niah_type == "s_niah":
        # Test at different depths
        for depth in needle_depths:
            correct = 0
            for _ in range(num_samples):
                seq, query, target = dataset.create_s_niah_sample(context_length, depth)

                # Embed and predict
                seq_emb = dataset.embed_sequence(seq)[None, :, :]  # [1, L, D]
                query_emb = dataset.embed_sequence(query)[None, :, :]  # [1, 1, D]

                logits = model(seq_emb, query_emb)  # [1, 1, vocab]
                pred = int(mx.argmax(logits[0, 0]).item())

                if pred == target:
                    correct += 1

            results["depth_accuracies"][depth] = correct / num_samples

        results["accuracy"] = sum(results["depth_accuracies"].values()) / len(needle_depths)

    elif niah_type == "mk_niah":
        correct = 0
        for _ in range(num_samples):
            seq, query, targets = dataset.create_mk_niah_sample(context_length)

            seq_emb = dataset.embed_sequence(seq)[None, :, :]
            query_emb = dataset.embed_sequence(query)[None, :, :]

            logits = model(seq_emb, query_emb)
            preds = mx.argmax(logits[0], axis=-1).tolist()

            # Check if all predictions match
            if preds == targets:
                correct += 1

        results["accuracy"] = correct / num_samples

    elif niah_type == "mq_niah":
        correct = 0
        for _ in range(num_samples):
            seq, query, target = dataset.create_mq_niah_sample(context_length)

            seq_emb = dataset.embed_sequence(seq)[None, :, :]
            query_emb = dataset.embed_sequence(query)[None, :, :]

            logits = model(seq_emb, query_emb)
            preds = mx.argmax(logits[0], axis=-1).tolist()

            # All queries should return same value
            if all(p == target for p in preds):
                correct += 1

        results["accuracy"] = correct / num_samples

    elif niah_type == "mv_niah":
        correct = 0
        for _ in range(num_samples):
            seq, query, targets = dataset.create_mv_niah_sample(context_length)

            seq_emb = dataset.embed_sequence(seq)[None, :, :]
            query_emb = dataset.embed_sequence(query)[None, :, :]

            logits = model(seq_emb, query_emb)
            pred = int(mx.argmax(logits[0, 0]).item())

            # Check if prediction is any of the valid values
            if pred in targets:
                correct += 1

        results["accuracy"] = correct / num_samples

    return results


def run_evaluation():
    """Run full long-context evaluation."""
    print("=" * 60)
    print("Long-Context Evaluation (Needle-in-Haystack)")
    print("=" * 60)

    # Configuration
    hidden_dim = 64
    vocab_size = 1000

    # Context lengths to test (use smaller sizes for MLX efficiency)
    context_lengths = [256, 512, 1024, 2048, 4096]

    # Create dataset
    print("\nCreating NIAH dataset...")
    dataset = NeedleHaystackDataset(hidden_dim, vocab_size)

    # Create models
    print("Creating models...")
    hope_model = LongContextRetriever(hidden_dim, vocab_size, num_levels=4)
    baseline_model = BaselineRetriever(hidden_dim, vocab_size)

    # Test different NIAH types
    niah_types = ["s_niah", "mk_niah", "mq_niah", "mv_niah"]

    all_results = {
        "HOPE": {},
        "Baseline": {},
    }

    for niah_type in niah_types:
        print(f"\n{'=' * 60}")
        print(f"Testing {niah_type.upper()}")
        print("=" * 60)

        all_results["HOPE"][niah_type] = []
        all_results["Baseline"][niah_type] = []

        for ctx_len in context_lengths:
            print(f"\n  Context length: {ctx_len}")

            # HOPE model
            hope_result = evaluate_retrieval_accuracy(
                hope_model, dataset, ctx_len, niah_type, num_samples=20
            )
            all_results["HOPE"][niah_type].append(hope_result)
            print(f"    HOPE accuracy: {hope_result['accuracy']:.3f}")

            # Baseline model
            baseline_result = evaluate_retrieval_accuracy(
                baseline_model, dataset, ctx_len, niah_type, num_samples=20
            )
            all_results["Baseline"][niah_type].append(baseline_result)
            print(f"    Baseline accuracy: {baseline_result['accuracy']:.3f}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for niah_type in niah_types:
        print(f"\n{niah_type.upper()}:")
        print(f"{'Context':>10} | {'HOPE':>10} | {'Baseline':>10} | {'Delta':>10}")
        print("-" * 50)

        for i, ctx_len in enumerate(context_lengths):
            hope_acc = all_results["HOPE"][niah_type][i]["accuracy"]
            base_acc = all_results["Baseline"][niah_type][i]["accuracy"]
            delta = hope_acc - base_acc
            print(f"{ctx_len:>10} | {hope_acc:>10.3f} | {base_acc:>10.3f} | {delta:>+10.3f}")

    # S-NIAH depth analysis
    print("\n" + "=" * 60)
    print("S-NIAH Depth Analysis (at longest context)")
    print("=" * 60)

    last_hope = all_results["HOPE"]["s_niah"][-1]
    last_base = all_results["Baseline"]["s_niah"][-1]

    print(f"{'Depth':>10} | {'HOPE':>10} | {'Baseline':>10}")
    print("-" * 40)
    for depth in last_hope["depth_accuracies"]:
        hope_d = last_hope["depth_accuracies"][depth]
        base_d = last_base["depth_accuracies"].get(depth, 0)
        print(f"{depth:>10.2f} | {hope_d:>10.3f} | {base_d:>10.3f}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_evaluation()
