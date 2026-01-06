
import mlx.core as mx
import mlx.nn as nn
import argparse
from typing import List

# Experiment: Nested SC Hierarchy
# Hypothesis: Nested Learning levels map to SC depth
#
# Logic: We define an outer loop (High Level State) and inner loop (Low Level State).
# Outer loop updates slow variables. Inner loop updates fast variables.
# This maps to "Active State" hierarchy in Statecharts.

class HierarchyLevelModule(nn.Module):
    """
    Represents one level of the hierarchy.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim), # [Input; Hidden_Prev] -> Hidden_New
            nn.Tanh()
        )

    def forward(self, input_signal: mx.array, prev_hidden: mx.array):
        obs = mx.concatenate([input_signal, prev_hidden], axis=-1)
        new_hidden = self.net(obs)
        return new_hidden

class NestedSCModel(nn.Module):
    def __init__(self, num_levels=2, dim=32):
        super().__init__()
        self.levels = [
            HierarchyLevelModule(dim, dim) for _ in range(num_levels)
        ]
        # In MLX, we need to register modules in a list or manual traversal if list is not supported
        # nn.ModuleList is not explicitly in base MLX, so we might need to assign attributes
        self.level_0 = self.levels[0]
        self.level_1 = self.levels[1]
        self.dim = dim

    def forward_step(self, x: mx.array, hiddens: List[mx.array]):
        # Inner loop (Level 0) runs every step
        # Outer loop (Level 1) runs every K steps?
        # For valid differentiable "Nested Learning", usually Outer output conditions Inner.
        
        # Level 1 (Outer/Parent)
        # Conditioned on x
        h1_next = self.level_1.forward(x, hiddens[1])
        
        # Level 0 (Inner/Child)
        # Conditioned on x AND h1_next (Parent State)
        # We assume x contains both environment input and parent signal
        # For simplicity, we just add them
        
        h0_input = x + h1_next 
        h0_next = self.level_0.forward(h0_input, hiddens[0])
        
        return [h0_next, h1_next]

def run_nested_benchmark():
    print("Running Nested SC Benchmark...")
    
    dim = 16
    model = NestedSCModel(num_levels=2, dim=dim)
    
    # Mock inputs
    x = mx.random.normal((10, dim))
    
    hiddens = [mx.zeros((1, dim)), mx.zeros((1, dim))]
    
    # Run loop
    for i in range(10):
        step_x = x[i:i+1]
        hiddens = model.forward_step(step_x, hiddens)
        
        # Verify "State Persistence" logic (simulated)
        # In real exp, we'd check if h1 changes slower than h0
        pass
        
    print("  Nested loop execution successful.")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Initializing Nested SC Hierarchy Experiment...")
    run_nested_benchmark()
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
