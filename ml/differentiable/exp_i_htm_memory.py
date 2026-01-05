"""
Experiment I: HTM-Inspired Statechart Memory

Hierarchical Temporal Memory (HTM) concepts applied to statechart history:

1. Sparse Distributed Representations (SDR): Sparse state activations
2. Temporal Pooling: Aggregate temporal patterns (like SoftAttentionMemory)
3. Sequence Memory: Predict next configuration based on history
4. Column Structure: Group related states into computational columns

This builds on exp_b_memory.py's SoftAttentionMemory with HTM-specific enhancements.
"""

import mlx.core as mx
import mlx.nn as nn
import math


class SDREncoder(nn.Module):
    """
    Sparse Distributed Representation encoder for state configurations.

    Maps soft state distributions to sparse binary-like representations.
    Uses top-k activation to enforce sparsity.
    """

    def __init__(self, num_states: int, sdr_size: int = 64, sparsity: float = 0.05):
        super().__init__()
        self.num_states = num_states
        self.sdr_size = sdr_size
        self.k = max(1, int(sdr_size * sparsity))  # Number of active bits

        # Project state config to SDR space
        self.encoder = nn.Sequential(
            nn.Linear(num_states, sdr_size * 2),
            nn.ReLU(),
            nn.Linear(sdr_size * 2, sdr_size),
        )

    def __call__(self, config: mx.array) -> mx.array:
        """
        Encode configuration to sparse representation.

        Args:
            config: [B, num_states] state distribution

        Returns:
            sdr: [B, sdr_size] sparse representation (soft top-k)
        """
        # Project to SDR space
        logits = self.encoder(config)  # [B, sdr_size]

        # Soft top-k: temperature-scaled softmax encourages sparsity
        temperature = 0.1  # Low temp = more sparse
        soft_sdr = mx.softmax(logits / temperature, axis=-1)

        # Rescale to have roughly k active "bits"
        # This maintains differentiability while approximating sparsity
        return soft_sdr * self.sdr_size / self.k

    def overlap(self, sdr1: mx.array, sdr2: mx.array) -> mx.array:
        """
        Compute overlap between two SDRs (HTM similarity metric).

        Args:
            sdr1, sdr2: [B, sdr_size] sparse representations

        Returns:
            overlap: [B] similarity scores
        """
        # Element-wise min gives overlap (for sparse binary vectors)
        # For soft SDRs, this approximates overlap
        return mx.sum(mx.minimum(sdr1, sdr2), axis=-1)


class TemporalPooler(nn.Module):
    """
    HTM Temporal Pooler - learns stable representations over time.

    Aggregates sequences of SDRs into temporal patterns that are
    more stable than individual timestep representations.
    """

    def __init__(self, sdr_size: int, d_pooled: int = 32, window_size: int = 8):
        super().__init__()
        self.sdr_size = sdr_size
        self.d_pooled = d_pooled
        self.window_size = window_size

        # Temporal convolution over SDR history
        # Uses 1D conv to aggregate temporal patterns
        self.temporal_conv = nn.Conv1d(
            in_channels=sdr_size,
            out_channels=d_pooled,
            kernel_size=min(3, window_size),
            padding=1,
        )

        # Pooling attention
        self.query = nn.Linear(sdr_size, d_pooled)
        self.key = nn.Linear(sdr_size, d_pooled)
        self.scale = 1.0 / math.sqrt(d_pooled)

    def __call__(self, sdr_history: mx.array, current_sdr: mx.array) -> mx.array:
        """
        Pool temporal patterns from SDR history.

        Args:
            sdr_history: [B, T, sdr_size] history of SDRs
            current_sdr: [B, sdr_size] current SDR

        Returns:
            pooled: [B, d_pooled] temporally pooled representation
        """
        B, T, _ = sdr_history.shape

        # Query from current, keys from history
        q = self.query(current_sdr)  # [B, d_pooled]
        k = self.key(sdr_history)     # [B, T, d_pooled]

        # Attention scores
        q = mx.expand_dims(q, axis=1)  # [B, 1, d_pooled]
        scores = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * self.scale  # [B, 1, T]
        attn = mx.softmax(scores, axis=-1)  # [B, 1, T]

        # Attend over history
        context = mx.matmul(attn, sdr_history)  # [B, 1, sdr_size]
        context = mx.squeeze(context, axis=1)   # [B, sdr_size]

        # Project to pooled dimension
        pooled = self.query(context)  # Reuse query projection

        return pooled


class SequenceMemory(nn.Module):
    """
    HTM Sequence Memory - predicts next state based on context.

    Learns temporal patterns to predict upcoming configurations
    given the current state and temporal context.
    """

    def __init__(self, sdr_size: int, d_pooled: int, num_states: int, hidden_dim: int = 64):
        super().__init__()
        self.sdr_size = sdr_size
        self.d_pooled = d_pooled
        self.num_states = num_states
        self.hidden_dim = hidden_dim

        # Prediction network layers
        self.l1 = nn.Linear(sdr_size + d_pooled, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, num_states)

        # Confidence estimator
        self.confidence = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, current_sdr: mx.array, pooled_context: mx.array) -> tuple[mx.array, mx.array]:
        """
        Predict next configuration.

        Args:
            current_sdr: [B, sdr_size] current state SDR
            pooled_context: [B, d_pooled] temporal context

        Returns:
            prediction: [B, num_states] predicted next configuration
            confidence: [B, 1] prediction confidence
        """
        combined = mx.concatenate([current_sdr, pooled_context], axis=-1)

        # Get hidden representation
        x = mx.tanh(self.l1(combined))
        x = mx.tanh(self.l2(x))

        # Predict next config
        pred_logits = self.l3(x)
        prediction = mx.softmax(pred_logits, axis=-1)

        # Confidence in prediction
        conf = mx.sigmoid(self.confidence(x))

        return prediction, conf


class HTMMemory(nn.Module):
    """
    Complete HTM-inspired memory for statecharts.

    Combines:
    - SDR encoding for sparse state representation
    - Temporal pooling for pattern aggregation
    - Sequence memory for prediction

    This provides a biologically-inspired alternative to
    the attention/NTM/GRU memories in exp_b.
    """

    def __init__(
        self,
        num_states: int,
        sdr_size: int = 64,
        sparsity: float = 0.05,
        d_pooled: int = 32,
        max_history: int = 64,
    ):
        super().__init__()
        self.num_states = num_states
        self.sdr_size = sdr_size
        self.max_history = max_history

        # HTM components
        self.sdr_encoder = SDREncoder(num_states, sdr_size, sparsity)
        self.temporal_pooler = TemporalPooler(sdr_size, d_pooled)
        self.sequence_memory = SequenceMemory(sdr_size, d_pooled, num_states)

        # Output projection (to match exp_b interface)
        self.out_proj = nn.Linear(num_states, num_states)

    def __call__(
        self,
        current_config: mx.array,
        sdr_history: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Process current config through HTM memory.

        Args:
            current_config: [B, num_states] current state distribution
            sdr_history: [B, T, sdr_size] history of SDRs

        Returns:
            retrieved: [B, num_states] memory-informed retrieval
            prediction: [B, num_states] predicted next config
            confidence: [B, 1] prediction confidence
            current_sdr: [B, sdr_size] current SDR for history update
        """
        # Encode current config to SDR
        current_sdr = self.sdr_encoder(current_config)  # [B, sdr_size]

        # Pool temporal context
        pooled = self.temporal_pooler(sdr_history, current_sdr)  # [B, d_pooled]

        # Predict next configuration
        prediction, confidence = self.sequence_memory(current_sdr, pooled)

        # Retrieved representation blends current with prediction
        # (weighted by confidence)
        retrieved = current_config * (1 - confidence) + prediction * confidence
        retrieved = self.out_proj(retrieved)

        return retrieved, prediction, confidence, current_sdr

    def update_history(self, sdr_history: mx.array, new_sdr: mx.array) -> mx.array:
        """
        Update SDR history buffer.

        Args:
            sdr_history: [B, T, sdr_size]
            new_sdr: [B, sdr_size]

        Returns:
            updated_history: [B, T', sdr_size] where T' <= max_history
        """
        new_sdr = mx.expand_dims(new_sdr, axis=1)  # [B, 1, sdr_size]
        updated = mx.concatenate([sdr_history, new_sdr], axis=1)

        if updated.shape[1] > self.max_history:
            updated = updated[:, -self.max_history:, :]

        return updated

    def init_state(self, batch_size: int) -> dict:
        """Initialize HTM memory state."""
        return {
            "sdr_history": mx.zeros((batch_size, 1, self.sdr_size))
        }

    def step(self, current_config: mx.array, state: dict) -> tuple[mx.array, dict, dict]:
        """
        Single memory step (matches UnifiedMemory interface).

        Args:
            current_config: [B, num_states]
            state: Memory state dict

        Returns:
            retrieved: [B, num_states]
            new_state: Updated memory state
            info: dict with current_sdr, prediction, confidence for integration
        """
        retrieved, prediction, confidence, current_sdr = self(
            current_config, state["sdr_history"]
        )

        new_history = self.update_history(state["sdr_history"], current_sdr)

        info = {
            "current_sdr": current_sdr,      # [B, sdr_size] - for guard context
            "prediction": prediction,         # [B, num_states] - predicted next config
            "confidence": confidence,         # [B, 1] - prediction confidence
        }

        return retrieved, {"sdr_history": new_history}, info


class ColumnMemory(nn.Module):
    """
    HTM Column-based memory organization.

    Groups related states into "columns" (like HTM minicolumns).
    Each column learns patterns within its group of states,
    enabling hierarchical pattern learning.
    """

    def __init__(
        self,
        num_states: int,
        num_columns: int = 4,
        states_per_column: int = None,
        d_column: int = 16,
    ):
        super().__init__()
        self.num_states = num_states
        self.num_columns = num_columns

        if states_per_column is None:
            states_per_column = (num_states + num_columns - 1) // num_columns
        self.states_per_column = states_per_column

        # Column-level processing
        self.column_proj = nn.Linear(states_per_column, d_column)
        self.column_interact = nn.Linear(num_columns * d_column, num_columns * d_column)
        self.column_out = nn.Linear(d_column, states_per_column)

    def __call__(self, config: mx.array) -> mx.array:
        """
        Process configuration through column structure.

        Args:
            config: [B, num_states] state distribution

        Returns:
            enhanced: [B, num_states] column-enhanced representation
        """
        B = config.shape[0]

        # Pad config to fit columns evenly
        padded_size = self.num_columns * self.states_per_column
        if self.num_states < padded_size:
            padding = mx.zeros((B, padded_size - self.num_states))
            config_padded = mx.concatenate([config, padding], axis=-1)
        else:
            config_padded = config

        # Reshape into columns: [B, num_columns, states_per_column]
        columns = config_padded.reshape(B, self.num_columns, self.states_per_column)

        # Process each column
        col_hidden = self.column_proj(columns)  # [B, num_columns, d_column]

        # Flatten for inter-column interaction
        flat = col_hidden.reshape(B, -1)  # [B, num_columns * d_column]
        interacted = mx.tanh(self.column_interact(flat))

        # Reshape back
        col_hidden = interacted.reshape(B, self.num_columns, -1)

        # Project back to states
        col_out = self.column_out(col_hidden)  # [B, num_columns, states_per_column]

        # Flatten and trim to original size
        enhanced = col_out.reshape(B, -1)[:, :self.num_states]

        return enhanced


# =============================================================================
# Test Script
# =============================================================================


def test_htm_memory():
    """Test HTM-inspired memory mechanisms."""
    print("=" * 60)
    print("Testing HTM-Inspired Statechart Memory")
    print("=" * 60)

    num_states = 4
    batch_size = 2
    seq_len = 10

    # Create HTM memory
    htm = HTMMemory(
        num_states=num_states,
        sdr_size=32,
        sparsity=0.1,
        d_pooled=16,
        max_history=32,
    )

    print(f"\n1. HTMMemory Configuration:")
    print(f"   num_states={num_states}, sdr_size=32, sparsity=0.1")

    # Generate test sequence
    mx.random.seed(42)
    configs = []
    for t in range(seq_len):
        # Cycle through states
        state_idx = t % num_states
        config = mx.zeros((batch_size, num_states))
        # Soft one-hot
        probs = [0.1] * num_states
        probs[state_idx] = 0.7
        config = mx.array([probs for _ in range(batch_size)])
        configs.append(config)

    # Run through memory
    print("\n2. Running sequence...")
    state = htm.init_state(batch_size)

    for t, config in enumerate(configs):
        retrieved, state, info = htm.step(config, state)

        if t < 3 or t == seq_len - 1:
            conf_val = float(info["confidence"][0, 0])
            print(f"   t={t}: input={[f'{x:.2f}' for x in config[0].tolist()]}, "
                  f"conf={conf_val:.3f}")

    # Test gradient flow
    print("\n3. Testing gradient flow...")

    def loss_fn(model, config, mem_state):
        retrieved, _, _ = model.step(config, mem_state)
        target = mx.array([[1.0, 0.0, 0.0, 0.0]] * batch_size)
        return mx.mean((retrieved - target) ** 2)

    fresh_state = htm.init_state(batch_size)
    loss_and_grad = nn.value_and_grad(htm, loss_fn)
    loss, grads = loss_and_grad(htm, configs[0], fresh_state)

    print(f"   Loss: {float(loss):.4f}")

    # Count gradients
    def count_grads(g):
        if isinstance(g, dict):
            return sum(count_grads(v) for v in g.values())
        elif isinstance(g, mx.array):
            return 1 if mx.any(g != 0).item() else 0
        return 0

    nonzero = count_grads(grads)
    print(f"   Non-zero gradient tensors: {nonzero}")

    if nonzero > 0:
        print("   Gradients flow correctly!")

    # Test SDR sparsity
    print("\n4. Testing SDR sparsity...")
    test_config = mx.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    sdr = htm.sdr_encoder(test_config)

    # Check effective sparsity (how concentrated the activations are)
    sorted_sdr = mx.sort(sdr, axis=-1)
    top_10_pct = sorted_sdr[:, -3:]  # Top 10% of 32 = ~3
    top_energy = mx.sum(top_10_pct, axis=-1) / mx.sum(sdr, axis=-1)

    print(f"   SDR shape: {sdr.shape}")
    print(f"   Energy in top 10%: {[f'{x:.2f}' for x in top_energy.tolist()]}")

    # Test column memory
    print("\n5. Testing ColumnMemory...")
    col_mem = ColumnMemory(num_states=num_states, num_columns=2, d_column=8)
    enhanced = col_mem(test_config)
    print(f"   Input:    {[f'{x:.2f}' for x in test_config[0].tolist()]}")
    print(f"   Enhanced: {[f'{x:.2f}' for x in enhanced[0].tolist()]}")

    print("\n" + "=" * 60)
    print("HTM Memory tests complete!")
    print("=" * 60)


def benchmark_htm_vs_attention():
    """Compare HTM memory with attention memory."""
    from .exp_b_memory import SoftAttentionMemory

    print("\n--- Benchmark: HTM vs Attention Memory ---")

    import time

    num_states = 8
    batch_size = 32
    seq_len = 100

    # Create memories
    htm = HTMMemory(num_states, sdr_size=64, d_pooled=32)
    attn = SoftAttentionMemory(num_states, d_memory=32, max_history=64)

    # Generate sequence
    configs = [mx.softmax(mx.random.normal((batch_size, num_states)), axis=-1)
               for _ in range(seq_len)]

    # Benchmark HTM
    htm_state = htm.init_state(batch_size)
    start = time.perf_counter()
    for config in configs:
        _, htm_state, _ = htm.step(config, htm_state)
    mx.eval(htm_state)
    htm_time = time.perf_counter() - start

    # Benchmark Attention
    attn_buffer = mx.zeros((batch_size, 1, num_states))
    start = time.perf_counter()
    for config in configs:
        _, _ = attn(config, attn_buffer)
        attn_buffer = attn.update_buffer(attn_buffer, config)
    mx.eval(attn_buffer)
    attn_time = time.perf_counter() - start

    print(f"  HTM:       {htm_time*1000:.2f}ms ({htm_time/seq_len*1000:.3f}ms/step)")
    print(f"  Attention: {attn_time*1000:.2f}ms ({attn_time/seq_len*1000:.3f}ms/step)")


if __name__ == "__main__":
    test_htm_memory()
    try:
        benchmark_htm_vs_attention()
    except ImportError:
        print("\n[Skipping benchmark - exp_b_memory not importable standalone]")
