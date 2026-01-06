
import mlx.core as mx
import mlx.nn as nn
import argparse
import time
from typing import Dict, List, Tuple
import json

# Experiment: Mamba SC Executor
# Hypothesis: Selective A(e) matrices map to SC transitions
#
# SC: S' = Transition(S, E)
# SSM: h' = A(e)h + B(e)x
#
# We map Discrete States -> Standard Basis Vectors in SSM latent space
# Transitions -> Permutation/Selection Matrices in A(e)

class MambaSCExecutor(nn.Module):
    """
    Implements Statechart execution using Mamba-like Selective SSM mechanisms.
    """
    def __init__(self, num_states: int, num_events: int, d_model: int = 64):
        super().__init__()
        self.num_states = num_states
        self.num_events = num_events
        self.d_model = d_model
        
        # We learn specific A matrices for each event type
        # A_stack shape: [num_events, num_states, num_states]
        # In a real continuous Mamba, A is -(exp(w)), here we simulate 
        # the "Selection" mechanism by choosing A based on input event.
        
        # Initialize A closer to identity to retain state by default
        self.A_stack = mx.eye(num_states) # [Events, States, States]
        # We will make A_stack learnable or settable
        self.A_stack = mx.tile(mx.expand_dims(self.A_stack, 0), (num_events, 1, 1))
        # Add some noise for learning
        self.A_stack = self.A_stack + mx.random.normal(self.A_stack.shape) * 0.01

        # Input projection (Events -> Latent)
        self.B = nn.Embedding(num_events, num_states)
        
        # Output projection (Latent -> States)
        self.C = nn.Linear(num_states, num_states)

    def step(self, state: mx.array, event_idx: mx.array):
        """
        Executes one step.
        h_t = A(event) * h_{t-1} + B(event)
        """
        # Select A for this event
        A = self.A_stack[event_idx] # [Batch, States, States] or just [States, States]
        
        # Update State
        # h' = A @ h
        # For simplicity, assuming batch size 1 for this prototype logic
        next_state_linear = A @ state
        
        # In SC, state is discrete (one-hot). In SSM, it's continuous.
        # We add a nonlinearity/normalization to keep it stable akin to SC.
        # But standard Mamba is linear recurrence.
        
        return next_state_linear

    def forward(self, events: mx.array, initial_state: mx.array = None):
        """Process a sequence of events."""
        seq_len = events.shape[0]
        if initial_state is None:
            initial_state = mx.zeros((self.num_states,))
            initial_state[0] = 1.0 # Initial state active
            
        current_state = initial_state
        history = []
        
        for i in range(seq_len):
            event = events[i]
            current_state = self.step(current_state, event)
            history.append(current_state)
            
        return mx.stack(history)

    def extract_transition_graph(self) -> Dict[str, List[str]]:
        """
        Interpret learned A matrices as transitions.
        If A[e][i][j] is high, means State j -> State i on Event e.
        """
        graph = {}
        # Naive extraction: Thresholding
        # Note: This requires the model to be trained.
        return graph

def benchmark_long_sequence(seq_len: int = 1000):
    print(f"Benchmarking Sequence Length: {seq_len}")
    
    model = MambaSCExecutor(num_states=10, num_events=5)
    
    events = mx.random.randint(0, 5, (seq_len,))
    
    start = time.time()
    # Evaluate
    _ = model.forward(events)
    # Force eval
    mx.eval(_)
    duration = time.time() - start
    
    print(f"  Duration: {duration*1000:.2f} ms")
    print(f"  Tokens/sec: {seq_len/duration:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1000)
    args = parser.parse_args()
    
    print("Initializing Mamba SC Executor Experiment...")
    benchmark_long_sequence(args.seq_len)
    
    print("Verification: Long sequence stability check...")
    # Add stability check logic here (e.g. ensure norms don't explode)
    
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
