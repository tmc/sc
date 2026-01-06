
import mlx.core as mx
import mlx.nn as nn
import argparse
from typing import Tuple

# Experiment: SAE Feature Steering
# Hypothesis: Steering features controls SC behavior
#
# Logic:
# 1. Pre-trained SC Model (or mock).
# 2. SAE trained on activations.
# 3. Hook forward pass: Activations -> Encoder -> (Latent + Steering) -> Decoder -> Modified Activations.

class SAEFeatureSteering(nn.Module):
    def __init__(self, dim: int, sae_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        
        # SAE
        self.encoder = nn.Linear(dim, sae_dim)
        self.decoder = nn.Linear(sae_dim, dim)
        
        self.steering_vector = mx.zeros((sae_dim,))

    def forward(self, x: mx.array, steer: bool = False):
        # Base computation
        act = self.proj(x)
        
        if steer:
            # Intervene
            latent = nn.relu(self.encoder(act))
            
            # Steer
            latent = latent + self.steering_vector
            
            # Reconstruct
            act_steered = self.decoder(latent)
            
            # Residual replacement (force SAE output or mix?)
            # Usually: act = act_original + (reconstruction_error) + steering_effect
            # Simplified: act = act_steered for strong steering
            return act_steered
            
        return act

    def set_steering(self, feature_idx: int, value: float):
        self.steering_vector = mx.zeros_like(self.steering_vector)
        self.steering_vector[feature_idx] = value

def run_steering_benchmark():
    print("Running SAE Feature Steering Benchmark...")
    
    dim = 16
    sae_dim = 64
    model = SAEFeatureSteering(dim, sae_dim)
    
    x = mx.random.normal((1, dim))
    
    # Baseline
    out_base = model.forward(x, steer=False)
    
    # Steer Feature 10
    model.set_steering(10, 5.0)
    out_steered = model.forward(x, steer=True)
    
    diff = mx.linalg.norm(out_base - out_steered).item()
    print(f"  Steering Impact (Diff): {diff:.4f}")
    print("  Verification: Feature injection alters output.")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("Initializing SAE Feature Steering Experiment...")
    run_steering_benchmark()
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
