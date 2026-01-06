
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Reusing Logic from exp_qwen_statechart/exp5_sae_probing.py
# Enhanced for "Reasoning States" specific experiment

@dataclass
class SyntaxFeatureAssociation:
    """Association between SAE feature and syntax state."""
    feature_id: int
    state: str
    activation_count: int
    lift: float

class SCSparseAutoencoder:
    """
    Sparse Autoencoder (Linear -> ReLU -> Linear)
    """
    def __init__(self, input_dim: int, expansion_factor: int = 8, l1_coeff: float = 3e-4):
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.l1_coeff = l1_coeff
        
        # Encoder: W_enc, b_enc
        self.W_enc = mx.random.normal((input_dim, self.latent_dim)) * 0.01
        self.b_enc = mx.zeros((self.latent_dim,))
        
        # Decoder: W_dec, b_dec
        self.W_dec = mx.random.normal((self.latent_dim, input_dim)) * 0.01
        self.b_dec = mx.zeros((input_dim,))

    def encode(self, x: mx.array) -> mx.array:
        """Forward pass to latent space."""
        pre_act = x @ self.W_enc + self.b_enc
        latent = mx.maximum(pre_act, 0) # ReLU
        return latent

    def decode(self, latent: mx.array) -> mx.array:
        """Reconstruct input."""
        return latent @ self.W_dec + self.b_dec

    def forward(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Returns (reconstruction, latent, loss)."""
        latent = self.encode(x)
        recon = self.decode(latent)
        
        # Loss: MSE + L1
        mse_loss = mx.mean((x - recon) ** 2)
        l1_loss = self.l1_coeff * mx.mean(mx.abs(latent))
        loss = mse_loss + l1_loss
        
        return recon, latent, loss

class SCStateSAEAnalyzer:
    """
    Analyzes which SAE features correspond to SC states.
    """
    def __init__(self, input_dim=256, expansion=8):
        self.sae = SCSparseAutoencoder(input_dim, expansion)
        self.feature_state_counts = defaultdict(int)
        self.feature_counts = defaultdict(int)
        self.state_counts = defaultdict(int)
        self.total_obs = 0

    def train_step(self, activations: mx.array):
        """Mock training step."""
        # Typically needs optimizer
        recon, latent, loss = self.sae.forward(activations)
        return loss

    def record_cooccurrence(self, activations: mx.array, state_label: str):
        """Record which features are active for this state."""
        latent = self.sae.encode(activations)
        # Get active indices (sparse)
        # Threshold or top-k
        
        # Simple threshold
        active_mask = latent > 0.01
        active_indices = [i for i, x in enumerate(active_mask.tolist()) if x]
        
        for idx in active_indices:
            self.feature_state_counts[(idx, state_label)] += 1
            self.feature_counts[idx] += 1
        
        self.state_counts[state_label] += 1
        self.total_obs += 1

    def compute_associations(self, min_lift: float = 2.0) -> Dict[str, List[SyntaxFeatureAssociation]]:
        """Compute Lift = P(State|Feature) / P(State)."""
        assocs = defaultdict(list)
        
        for (fid, state), count in self.feature_state_counts.items():
            if count < 5: continue
            
            p_state_given_feature = count / self.feature_counts[fid]
            p_state = self.state_counts[state] / self.total_obs
            
            lift = p_state_given_feature / p_state if p_state > 0 else 0
            
            if lift > min_lift:
                assocs[state].append(SyntaxFeatureAssociation(fid, state, count, lift))
        
        # Sort by lift
        for state in assocs:
            assocs[state].sort(key=lambda x: x.lift, reverse=True)
            
        return assocs

def main():
    print("Initializing SAE Reasoning States Experiment...")
    
    # Mock Data: Random activations and states
    analyzer = SCStateSAEAnalyzer(input_dim=64, expansion=4)
    
    states = ["Idle", "Processing", "Error"]
    
    print("Training SAE (Mock)...")
    for _ in range(100):
        dummy_state = states[np.random.randint(0, 3)]
        # Simulate that "Processing" has a specific feature active (feature 10)
        act = mx.random.normal((64,)) 
        if dummy_state == "Processing":
            # Inject signal
            pass 
        analyzer.train_step(act)
        analyzer.record_cooccurrence(act, dummy_state)
        
    print("Computing Associations...")
    associations = analyzer.compute_associations(min_lift=1.0)
    
    print("Top associations:")
    for state, feats in associations.items():
        if feats:
            top = feats[0]
            print(f"State: {state} -> Feature {top.feature_id} (Lift: {top.lift:.2f})")
    
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
