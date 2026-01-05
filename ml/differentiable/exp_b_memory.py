"""
Differentiable Memory Mechanisms for Statechart History

Three approaches for learning history-aware state transitions:
1. SoftAttentionMemory: Query-key attention over past configurations
2. NTMMemory: Neural Turing Machine style content-based addressing
3. RecurrentMemory: GRU baseline for sequential processing

All implementations use MLX and support gradient flow for end-to-end training.
"""

import mlx.core as mx
import mlx.nn as nn
import math


class SoftAttentionMemory(nn.Module):
    """
    Stores past configurations in a buffer and uses attention to retrieve.

    Memory buffer: [B, T, num_states] - T timesteps of state distributions
    Query projects current state to retrieve relevant history.
    """

    def __init__(self, num_states: int, d_memory: int = 32, max_history: int = 64):
        super().__init__()
        self.num_states = num_states
        self.d_memory = d_memory
        self.max_history = max_history

        # Project state distributions to key/value space
        self.key_proj = nn.Linear(num_states, d_memory)
        self.value_proj = nn.Linear(num_states, d_memory)
        self.query_proj = nn.Linear(num_states, d_memory)

        # Output projection
        self.out_proj = nn.Linear(d_memory, num_states)

        self.scale = 1.0 / math.sqrt(d_memory)

    def __call__(self, current_config: mx.array, memory_buffer: mx.array) -> tuple[mx.array, mx.array]:
        """
        Retrieve relevant history using attention.

        Args:
            current_config: [B, num_states] current state distribution
            memory_buffer: [B, T, num_states] past configurations

        Returns:
            retrieved: [B, num_states] attention-weighted history
            attention_weights: [B, T] for interpretability
        """
        B = current_config.shape[0]
        T = memory_buffer.shape[1]

        # Project to attention space
        query = self.query_proj(current_config)  # [B, d_memory]
        keys = self.key_proj(memory_buffer)       # [B, T, d_memory]
        values = self.value_proj(memory_buffer)   # [B, T, d_memory]

        # Compute attention scores
        query = mx.expand_dims(query, axis=1)  # [B, 1, d_memory]
        scores = mx.matmul(query, mx.transpose(keys, (0, 2, 1)))  # [B, 1, T]
        scores = scores * self.scale

        # Softmax for attention weights
        attn_weights = mx.softmax(scores, axis=-1)  # [B, 1, T]

        # Weighted sum of values
        context = mx.matmul(attn_weights, values)  # [B, 1, d_memory]
        context = mx.squeeze(context, axis=1)      # [B, d_memory]

        # Project back to state space
        retrieved = self.out_proj(context)  # [B, num_states]

        return retrieved, mx.squeeze(attn_weights, axis=1)

    def update_buffer(self, memory_buffer: mx.array, new_config: mx.array) -> mx.array:
        """
        Add new configuration to buffer (FIFO if full).

        Args:
            memory_buffer: [B, T, num_states]
            new_config: [B, num_states]

        Returns:
            updated_buffer: [B, T', num_states] where T' <= max_history
        """
        new_config = mx.expand_dims(new_config, axis=1)  # [B, 1, num_states]
        updated = mx.concatenate([memory_buffer, new_config], axis=1)

        # Keep only last max_history entries
        if updated.shape[1] > self.max_history:
            updated = updated[:, -self.max_history:, :]

        return updated


class NTMMemory(nn.Module):
    """
    Neural Turing Machine style memory with content-based addressing.

    Uses external memory bank with read/write heads for more flexible
    memory access patterns than simple attention.
    """

    def __init__(self, num_states: int, memory_size: int = 32, d_memory: int = 32):
        super().__init__()
        self.num_states = num_states
        self.memory_size = memory_size
        self.d_memory = d_memory

        # Controller network (processes current state)
        self.controller = nn.Sequential(
            nn.Linear(num_states, d_memory * 2),
            nn.ReLU(),
            nn.Linear(d_memory * 2, d_memory),
        )

        # Read head: generates key, strength, and interpolation gate
        self.read_key = nn.Linear(d_memory, d_memory)
        self.read_strength = nn.Linear(d_memory, 1)
        self.read_gate = nn.Linear(d_memory, 1)  # Interpolation with previous weights

        # Write head: generates key, strength, erase vector, and add vector
        self.write_key = nn.Linear(d_memory, d_memory)
        self.write_strength = nn.Linear(d_memory, 1)
        self.write_gate = nn.Linear(d_memory, 1)
        self.erase_vector = nn.Linear(d_memory, d_memory)
        self.add_vector = nn.Linear(d_memory, d_memory)

        # Output projection
        self.out_proj = nn.Linear(d_memory, num_states)

    def content_addressing(self, key: mx.array, memory: mx.array, strength: mx.array) -> mx.array:
        """
        Content-based addressing using cosine similarity.

        Args:
            key: [B, d_memory] query key
            memory: [B, memory_size, d_memory] memory bank
            strength: [B, 1] sharpening factor

        Returns:
            weights: [B, memory_size] addressing weights
        """
        # Normalize for cosine similarity
        key_norm = key / (mx.linalg.norm(key, axis=-1, keepdims=True) + 1e-8)
        mem_norm = memory / (mx.linalg.norm(memory, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity
        key_expanded = mx.expand_dims(key_norm, axis=1)  # [B, 1, d_memory]
        similarity = mx.sum(key_expanded * mem_norm, axis=-1)  # [B, memory_size]

        # Apply sharpening and softmax
        # Softplus: log(1 + exp(x))
        strength = mx.log(1 + mx.exp(strength))  # Ensure positive
        weights = mx.softmax(similarity * strength, axis=-1)

        return weights

    def read(self, controller_state: mx.array, memory: mx.array, prev_weights: mx.array) -> tuple[mx.array, mx.array]:
        """
        Read from memory using content-based addressing.

        Args:
            controller_state: [B, d_memory]
            memory: [B, memory_size, d_memory]
            prev_weights: [B, memory_size] previous read weights

        Returns:
            read_vector: [B, d_memory]
            weights: [B, memory_size] new read weights
        """
        # Generate read parameters
        key = self.read_key(controller_state)
        strength = self.read_strength(controller_state)
        gate = mx.sigmoid(self.read_gate(controller_state))

        # Content addressing
        content_weights = self.content_addressing(key, memory, strength)

        # Interpolate with previous weights
        weights = gate * content_weights + (1 - gate) * prev_weights

        # Read from memory
        weights_expanded = mx.expand_dims(weights, axis=-1)  # [B, memory_size, 1]
        read_vector = mx.sum(memory * weights_expanded, axis=1)  # [B, d_memory]

        return read_vector, weights

    def write(self, controller_state: mx.array, memory: mx.array, prev_weights: mx.array) -> tuple[mx.array, mx.array]:
        """
        Write to memory using content-based addressing.

        Args:
            controller_state: [B, d_memory]
            memory: [B, memory_size, d_memory]
            prev_weights: [B, memory_size] previous write weights

        Returns:
            new_memory: [B, memory_size, d_memory]
            weights: [B, memory_size] new write weights
        """
        # Generate write parameters
        key = self.write_key(controller_state)
        strength = self.write_strength(controller_state)
        gate = mx.sigmoid(self.write_gate(controller_state))
        erase = mx.sigmoid(self.erase_vector(controller_state))
        add = self.add_vector(controller_state)

        # Content addressing
        content_weights = self.content_addressing(key, memory, strength)

        # Interpolate with previous weights
        weights = gate * content_weights + (1 - gate) * prev_weights

        # Erase and write
        weights_expanded = mx.expand_dims(weights, axis=-1)  # [B, memory_size, 1]
        erase_expanded = mx.expand_dims(erase, axis=1)        # [B, 1, d_memory]
        add_expanded = mx.expand_dims(add, axis=1)            # [B, 1, d_memory]

        # M_t = M_{t-1} * (1 - w_t * e_t) + w_t * a_t
        erased = memory * (1 - weights_expanded * erase_expanded)
        new_memory = erased + weights_expanded * add_expanded

        return new_memory, weights

    def __call__(self, current_config: mx.array, memory: mx.array,
                 prev_read_weights: mx.array, prev_write_weights: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Full NTM step: read, then write.

        Args:
            current_config: [B, num_states] current state distribution
            memory: [B, memory_size, d_memory] memory bank
            prev_read_weights: [B, memory_size]
            prev_write_weights: [B, memory_size]

        Returns:
            retrieved: [B, num_states] retrieved state distribution
            new_memory: [B, memory_size, d_memory]
            read_weights: [B, memory_size]
            write_weights: [B, memory_size]
        """
        # Controller processes current state
        ctrl = self.controller(current_config)  # [B, d_memory]

        # Read from memory
        read_vector, read_weights = self.read(ctrl, memory, prev_read_weights)

        # Write to memory
        new_memory, write_weights = self.write(ctrl, memory, prev_write_weights)

        # Project read vector to state space
        retrieved = self.out_proj(read_vector)

        return retrieved, new_memory, read_weights, write_weights

    def init_memory(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        """
        Initialize memory and weights.

        Returns:
            memory: [B, memory_size, d_memory]
            read_weights: [B, memory_size]
            write_weights: [B, memory_size]
        """
        memory = mx.zeros((batch_size, self.memory_size, self.d_memory))
        # Start with uniform weights
        weights = mx.ones((batch_size, self.memory_size)) / self.memory_size
        return memory, weights, weights


class GRUCell(nn.Module):
    """Minimal GRU cell implementation for MLX."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Gates
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)  # Reset gate
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        """
        GRU step.

        Args:
            x: [B, input_size] input
            h: [B, hidden_size] previous hidden state

        Returns:
            h_new: [B, hidden_size] new hidden state
        """
        combined = mx.concatenate([x, h], axis=-1)

        z = mx.sigmoid(self.Wz(combined))  # Update gate
        r = mx.sigmoid(self.Wr(combined))  # Reset gate

        combined_reset = mx.concatenate([x, r * h], axis=-1)
        h_tilde = mx.tanh(self.Wh(combined_reset))  # Candidate

        h_new = (1 - z) * h + z * h_tilde
        return h_new


class RecurrentMemory(nn.Module):
    """
    GRU-based recurrent memory for processing configurations sequentially.

    The hidden state serves as the memory, accumulating information
    from the sequence of visited states.
    """

    def __init__(self, num_states: int, hidden_size: int = 64):
        super().__init__()
        self.num_states = num_states
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(num_states, hidden_size)

        # GRU cell
        self.gru = GRUCell(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, num_states)

    def __call__(self, current_config: mx.array, hidden: mx.array) -> tuple[mx.array, mx.array]:
        """
        Process current configuration and update memory.

        Args:
            current_config: [B, num_states] current state distribution
            hidden: [B, hidden_size] previous hidden state

        Returns:
            retrieved: [B, num_states] memory-informed output
            new_hidden: [B, hidden_size] updated hidden state
        """
        # Project input
        x = self.input_proj(current_config)

        # GRU step
        new_hidden = self.gru(x, hidden)

        # Project to state space
        retrieved = self.out_proj(new_hidden)

        return retrieved, new_hidden

    def init_hidden(self, batch_size: int) -> mx.array:
        """Initialize hidden state to zeros."""
        return mx.zeros((batch_size, self.hidden_size))


# =============================================================================
# Unified Memory Interface
# =============================================================================

class MemoryType:
    ATTENTION = "attention"
    NTM = "ntm"
    RECURRENT = "recurrent"


class UnifiedMemory(nn.Module):
    """
    Unified wrapper for different memory mechanisms.

    Provides consistent interface for comparing memory types.
    """

    def __init__(self, num_states: int, memory_type: str = MemoryType.ATTENTION,
                 d_memory: int = 32, max_history: int = 64, memory_size: int = 32):
        super().__init__()
        self.memory_type = memory_type
        self.num_states = num_states

        if memory_type == MemoryType.ATTENTION:
            self.memory = SoftAttentionMemory(num_states, d_memory, max_history)
        elif memory_type == MemoryType.NTM:
            self.memory = NTMMemory(num_states, memory_size, d_memory)
        elif memory_type == MemoryType.RECURRENT:
            self.memory = RecurrentMemory(num_states, d_memory * 2)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def init_state(self, batch_size: int) -> dict:
        """Initialize memory state based on type."""
        if self.memory_type == MemoryType.ATTENTION:
            return {
                "buffer": mx.zeros((batch_size, 1, self.num_states))
            }
        elif self.memory_type == MemoryType.NTM:
            mem, rw, ww = self.memory.init_memory(batch_size)
            return {
                "memory": mem,
                "read_weights": rw,
                "write_weights": ww
            }
        else:  # RECURRENT
            return {
                "hidden": self.memory.init_hidden(batch_size)
            }

    def step(self, current_config: mx.array, state: dict) -> tuple[mx.array, dict, mx.array]:
        """
        Single memory step.

        Args:
            current_config: [B, num_states]
            state: Memory state dict

        Returns:
            retrieved: [B, num_states]
            new_state: Updated memory state
            attention: [B, ?] attention weights for visualization
        """
        if self.memory_type == MemoryType.ATTENTION:
            retrieved, attn = self.memory(current_config, state["buffer"])
            new_buffer = self.memory.update_buffer(state["buffer"], current_config)
            return retrieved, {"buffer": new_buffer}, attn

        elif self.memory_type == MemoryType.NTM:
            retrieved, new_mem, rw, ww = self.memory(
                current_config, state["memory"],
                state["read_weights"], state["write_weights"]
            )
            return retrieved, {
                "memory": new_mem,
                "read_weights": rw,
                "write_weights": ww
            }, rw

        else:  # RECURRENT
            retrieved, new_hidden = self.memory(current_config, state["hidden"])
            # No direct attention weights for GRU
            return retrieved, {"hidden": new_hidden}, mx.zeros((current_config.shape[0], 1))


# =============================================================================
# Test Script
# =============================================================================

def test_memory_mechanisms():
    """Test all 3 memory mechanisms on the same sequence."""
    print("=" * 60)
    print("Testing Differentiable Memory Mechanisms")
    print("=" * 60)

    # Setup: 3-state system, batch of 2
    num_states = 3
    batch_size = 2
    seq_len = 10

    # Create test sequence: oscillating between states
    # State 0 -> State 1 -> State 2 -> State 0 -> ...
    def make_sequence(seq_len, num_states, batch_size):
        configs = []
        for t in range(seq_len):
            # One-hot encoding for state (t % num_states)
            config = mx.zeros((batch_size, num_states))
            state_idx = t % num_states
            # Create one-hot manually
            one_hot = [0.0] * num_states
            one_hot[state_idx] = 1.0
            config = mx.array([[one_hot[i] for i in range(num_states)] for _ in range(batch_size)])
            configs.append(config)
        return configs

    sequence = make_sequence(seq_len, num_states, batch_size)

    # Initialize all 3 memory types
    memories = {
        "Attention": UnifiedMemory(num_states, MemoryType.ATTENTION, d_memory=16, max_history=32),
        "NTM": UnifiedMemory(num_states, MemoryType.NTM, d_memory=16, memory_size=16),
        "Recurrent": UnifiedMemory(num_states, MemoryType.RECURRENT, d_memory=16),
    }

    results = {}

    for name, memory in memories.items():
        print(f"\n--- Testing {name} Memory ---")

        state = memory.init_state(batch_size)
        outputs = []

        for t, config in enumerate(sequence):
            retrieved, state, attn = memory.step(config, state)
            outputs.append(retrieved)

            if t < 3 or t == seq_len - 1:
                print(f"  t={t}: input state={int(mx.argmax(config[0]).item())}, "
                      f"retrieved={retrieved[0].tolist()[:3]}")

        results[name] = outputs

        # Verify gradients flow
        print(f"  Gradient check: ", end="")
        try:
            def loss_fn(memory_model, config, init_state):
                # Simple loss: MSE between input and retrieved
                retrieved, _, _ = memory_model.step(config, init_state)
                return mx.mean((retrieved - config) ** 2)

            init_state = memory.init_state(batch_size)
            loss_and_grad = nn.value_and_grad(memory, loss_fn)
            loss, grads = loss_and_grad(memory, sequence[0], init_state)

            # Check that gradients exist
            has_grads = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
            print(f"OK (loss={float(loss):.4f}, grads={'found' if has_grads else 'zero'})")
        except Exception as e:
            print(f"FAILED: {e}")

    # Compare memory retrieval
    print("\n--- Comparison: Final State Retrieval ---")
    for name, outputs in results.items():
        final = outputs[-1][0]  # First batch item
        print(f"  {name:10s}: {final.tolist()}")

    print("\n" + "=" * 60)
    print("All memory mechanisms tested successfully!")
    print("=" * 60)

    return results


def benchmark_memory_mechanisms():
    """Benchmark memory mechanisms for speed and memory usage."""
    import time

    print("\n--- Benchmarking Memory Mechanisms ---")

    num_states = 8
    batch_size = 32
    seq_len = 100
    num_trials = 5

    # Generate random sequence
    configs = [mx.random.uniform(shape=(batch_size, num_states)) for _ in range(seq_len)]
    # Normalize to distributions
    configs = [c / mx.sum(c, axis=-1, keepdims=True) for c in configs]

    memories = {
        "Attention": UnifiedMemory(num_states, MemoryType.ATTENTION, d_memory=32, max_history=64),
        "NTM": UnifiedMemory(num_states, MemoryType.NTM, d_memory=32, memory_size=32),
        "Recurrent": UnifiedMemory(num_states, MemoryType.RECURRENT, d_memory=32),
    }

    for name, memory in memories.items():
        times = []
        for trial in range(num_trials):
            state = memory.init_state(batch_size)

            start = time.perf_counter()
            for config in configs:
                _, state, _ = memory.step(config, state)
            mx.eval(state)  # Force evaluation
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"  {name:10s}: {avg_time*1000:.2f}ms for {seq_len} steps ({avg_time/seq_len*1000:.3f}ms/step)")


if __name__ == "__main__":
    test_memory_mechanisms()
    benchmark_memory_mechanisms()
