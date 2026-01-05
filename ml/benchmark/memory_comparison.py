"""
Memory Mechanism Comparison Benchmark

Compares all memory mechanisms on sequence recall task:
1. SoftAttentionMemory (exp_b)
2. NTMMemory (exp_b)
3. RecurrentMemory (exp_b)
4. HTMMemory (exp_i)
5. CMS/HOPE (exp_j)

Task: Remember N items, recall in order
Metrics: accuracy, latency, memory usage, gradient norm
"""

import mlx.core as mx
import mlx.nn as nn
import time
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from differentiable.exp_b_memory import UnifiedMemory, MemoryType
from differentiable.exp_i_htm_memory import HTMMemory
from experiments.exp_j_titans_hope.continuum_memory import ContinuumMemorySystem


class SequenceRecallTask:
    """
    Simple sequence recall benchmark.
    
    Task: Given sequence [a, b, c, ...], recall items in order.
    Each item is a one-hot vector over vocab_size.
    """
    
    def __init__(self, vocab_size: int = 8, seq_len: int = 5, batch_size: int = 16):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
    
    def generate_batch(self) -> tuple:
        """
        Generate batch of sequences to memorize.
        
        Returns:
            inputs: [batch, seq_len, vocab_size] one-hot sequences
            targets: [batch, seq_len] target indices for recall
        """
        # Random sequence of indices
        targets = mx.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Convert to one-hot
        inputs = mx.eye(self.vocab_size)[targets]  # [batch, seq_len, vocab_size]
        
        return inputs, targets


class MemoryWrapper(nn.Module):
    """
    Wraps different memory mechanisms with consistent interface.
    """
    
    def __init__(self, memory_type: str, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.memory_type = memory_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Initialize memory based on type
        self.memory = None
        self._init_memory(memory_type)
        
        # Output projection for recall
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def _init_memory(self, memory_type: str):
        if memory_type == "attention":
            self.memory = UnifiedMemory(
                self.hidden_dim, memory_type=MemoryType.ATTENTION,
                d_memory=self.hidden_dim // 2, max_history=32
            )
        elif memory_type == "ntm":
            self.memory = UnifiedMemory(
                self.hidden_dim, memory_type=MemoryType.NTM,
                memory_size=32, d_memory=self.hidden_dim // 2
            )
        elif memory_type == "recurrent":
            self.memory = UnifiedMemory(
                self.hidden_dim, memory_type=MemoryType.RECURRENT,
                d_memory=self.hidden_dim // 2  # hidden_size = d_memory * 2
            )
        elif memory_type == "htm":
            self.memory = HTMMemory(
                self.hidden_dim, sdr_size=self.hidden_dim, d_pooled=self.hidden_dim // 2
            )
        elif memory_type == "cms":
            # CMS expects [B, seq, hidden] format
            self.memory = ContinuumMemorySystem(
                self.hidden_dim, num_levels=3, base_frequency=1, frequency_multiplier=2
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def init_state(self, batch_size: int) -> dict:
        """Initialize memory state."""
        if self.memory_type == "cms":
            return {"dummy": mx.zeros((batch_size, 1))}  # CMS is stateless per call
        return self.memory.init_state(batch_size)
    
    def encode_sequence(self, inputs: mx.array, state: dict) -> tuple:
        """
        Encode sequence into memory.
        
        Args:
            inputs: [batch, seq_len, input_dim]
            state: memory state
        
        Returns:
            final_state: updated memory state
            outputs: [batch, seq_len, hidden_dim] per-step outputs
        """
        batch_size, seq_len, _ = inputs.shape
        outputs = []
        
        if self.memory_type == "cms":
            # CMS processes whole sequence at once
            h = mx.tanh(self.input_proj(inputs))  # [B, seq, hidden]
            out, _ = self.memory(h)
            return state, out
        
        for t in range(seq_len):
            x = inputs[:, t, :]  # [batch, input_dim]
            h = mx.tanh(self.input_proj(x))  # [batch, hidden_dim]
            
            retrieved, state, _ = self.memory.step(h, state)
            outputs.append(retrieved)
        
        return state, mx.stack(outputs, axis=1)
    
    def recall(self, state: dict, outputs: mx.array) -> mx.array:
        """
        Recall sequence from memory.
        
        Args:
            state: final memory state
            outputs: [batch, seq_len, hidden_dim] encoded outputs
        
        Returns:
            recalled: [batch, seq_len, input_dim] recalled sequence
        """
        return self.output_proj(outputs)


def compute_accuracy(logits: mx.array, targets: mx.array) -> float:
    """Compute recall accuracy."""
    preds = mx.argmax(logits, axis=-1)  # [batch, seq_len]
    correct = mx.sum(preds == targets)
    total = targets.size
    return float(correct) / total


def compute_gradient_norm(grads: dict) -> float:
    """Compute L2 norm of all gradients."""
    total_sq = 0.0
    for _, g in nn.utils.tree_flatten(grads):
        if isinstance(g, mx.array):
            total_sq += float(mx.sum(g * g))
    return total_sq ** 0.5


def benchmark_memory(
    memory_type: str,
    task: SequenceRecallTask,
    num_trials: int = 5,
) -> dict:
    """
    Benchmark a memory mechanism.
    
    Returns:
        dict with accuracy, latency_ms, gradient_norm
    """
    wrapper = MemoryWrapper(memory_type, task.vocab_size, hidden_dim=32)
    
    # Warmup
    inputs, targets = task.generate_batch()
    state = wrapper.init_state(task.batch_size)
    state, outputs = wrapper.encode_sequence(inputs, state)
    recalled = wrapper.recall(state, outputs)
    mx.eval(recalled)
    
    # Timing trials
    latencies = []
    accuracies = []
    
    for _ in range(num_trials):
        inputs, targets = task.generate_batch()
        state = wrapper.init_state(task.batch_size)
        
        start = time.perf_counter()
        state, outputs = wrapper.encode_sequence(inputs, state)
        recalled = wrapper.recall(state, outputs)
        mx.eval(recalled)
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed * 1000)  # ms
        acc = compute_accuracy(recalled, targets)
        accuracies.append(acc)
    
    # Gradient norm (single sample)
    def loss_fn(model, inp, tgt):
        st = model.init_state(inp.shape[0])
        st, out = model.encode_sequence(inp, st)
        rec = model.recall(st, out)
        # Cross-entropy
        log_probs = mx.log(mx.softmax(rec, axis=-1) + 1e-10)
        loss = -mx.mean(log_probs.reshape(-1, task.vocab_size)[
            mx.arange(tgt.size), tgt.reshape(-1)
        ])
        return loss
    
    inputs, targets = task.generate_batch()
    try:
        loss, grads = nn.value_and_grad(wrapper, loss_fn)(wrapper, inputs, targets)
        grad_norm = compute_gradient_norm(grads)
    except Exception as e:
        grad_norm = float('nan')
    
    # Parameter count
    param_count = sum(p.size for _, p in nn.utils.tree_flatten(wrapper.parameters()))
    
    return {
        "type": memory_type,
        "accuracy": sum(accuracies) / len(accuracies),
        "latency_ms": sum(latencies) / len(latencies),
        "gradient_norm": grad_norm,
        "params": param_count,
    }


def run_comparison():
    """Run full comparison of all memory mechanisms."""
    print("=" * 70)
    print("Memory Mechanism Comparison Benchmark")
    print("=" * 70)
    
    # Task configuration
    VOCAB_SIZE = 8
    SEQ_LEN = 6
    BATCH_SIZE = 16
    NUM_TRIALS = 5
    
    task = SequenceRecallTask(VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)
    
    print(f"\nTask: Sequence Recall")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Trials: {NUM_TRIALS}")
    
    # Memory types to compare
    memory_types = ["attention", "ntm", "recurrent", "htm", "cms"]
    
    results = []
    for mem_type in memory_types:
        print(f"\nBenchmarking: {mem_type.upper()}...")
        try:
            result = benchmark_memory(mem_type, task, NUM_TRIALS)
            results.append(result)
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            print(f"  Gradient norm: {result['gradient_norm']:.4f}")
            print(f"  Parameters: {result['params']:,}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "type": mem_type,
                "accuracy": 0.0,
                "latency_ms": float('nan'),
                "gradient_norm": float('nan'),
                "params": 0,
            })
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Memory Type':<15} {'Accuracy':>10} {'Latency':>12} {'Grad Norm':>12} {'Params':>10}")
    print("-" * 70)
    
    for r in results:
        acc = f"{r['accuracy']:.1%}" if r['accuracy'] > 0 else "N/A"
        lat = f"{r['latency_ms']:.2f}ms" if not (r['latency_ms'] != r['latency_ms']) else "N/A"
        grad = f"{r['gradient_norm']:.4f}" if not (r['gradient_norm'] != r['gradient_norm']) else "N/A"
        params = f"{r['params']:,}" if r['params'] > 0 else "N/A"
        print(f"{r['type'].upper():<15} {acc:>10} {lat:>12} {grad:>12} {params:>10}")
    
    # Rankings
    print("\n" + "-" * 70)
    valid_results = [r for r in results if r['accuracy'] > 0]
    
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['latency_ms'])
        most_accurate = max(valid_results, key=lambda x: x['accuracy'])
        smallest = min(valid_results, key=lambda x: x['params'])
        
        print(f"Fastest: {fastest['type'].upper()} ({fastest['latency_ms']:.2f}ms)")
        print(f"Most accurate: {most_accurate['type'].upper()} ({most_accurate['accuracy']:.1%})")
        print(f"Smallest: {smallest['type'].upper()} ({smallest['params']:,} params)")
    
    print("\n" + "=" * 70)
    print("Benchmark complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    mx.random.seed(42)
    run_comparison()
