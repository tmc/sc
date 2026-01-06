
import mlx.core as mx
import mlx.nn as nn
import argparse
from typing import Tuple

# Experiment: Titans History Memory
# Hypothesis: Surprise-based memory improves H* recall
#
# Logic: We compute "surprise" (loss) of current state. 
# Highly surprising states are stored in Long-Term Memory (Titans style).
# History State (H*) can query this memory.

class TitansHistoryModule(nn.Module):
    def __init__(self, d_model: int, memory_size: int = 100):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # Surprise Predictor (simple autoencoder or next-step predictor)
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # MemoryStore: Fixed size buffer [Size, D]
        # In MLX, we might model this as a state passed around or a mock
        self.memory_keys = mx.zeros((memory_size, d_model))
        self.memory_vals = mx.zeros((memory_size, d_model))
        self.ptr = 0

    def compute_surprise(self, state: mx.array) -> mx.array:
        """Calculate implementation surprise (reconstruction error)."""
        pred = self.surprise_net(state)
        # MSE
        surprise = mx.mean((state - pred) ** 2, axis=-1)
        return surprise

    def update_memory(self, state: mx.array, surprise: mx.array):
        """Update memory if surprise is high."""
        # Simple FIFO for now, or prioritized by surprise
        # Mock logic
        pass

    def retrieve(self, query: mx.array) -> mx.array:
        """Attention-based retrieval."""
        # query: [1, D]
        # keys: [M, D]
        scores = query @ self.memory_keys.T
        attn = mx.softmax(scores, axis=-1)
        retrieved = attn @ self.memory_vals
        return retrieved

    def forward(self, current_state: mx.array):
        """
        1. Compute Surprise
        2. Retrieve context
        3. Update Memory
        """
        surprise = self.compute_surprise(current_state)
        context = self.retrieve(current_state)
        # self.update_memory(current_state, surprise) (Stateful update skipped for pure func check)
        return context, surprise

def run_titans_benchmark():
    print("Running Titans History Memory Benchmark...")
    
    model = TitansHistoryModule(d_model=64)
    
    # Mock State Sequence
    states = mx.random.normal((10, 64))
    
    surprises = []
    for i in range(10):
        s = states[i:i+1]
        ctx, surp = model.forward(s) # Explicit forward
        surprises.append(surp.item())
        
    avg_surprise = sum(surprises) / len(surprises)
    print(f"  Average Surprise: {avg_surprise:.4f}")
    
    print("  Verification: Memory module constructs and executes.")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Initializing Titans History Experiment...")
    run_titans_benchmark()
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
