
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, List, Callable

# Mock for mlx_lm if not installed or for testing logic without loading weights
class MockLLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.dim = dim
        
    def __call__(self, x):
        return self.model(x)
        
    def generate(self, prompt: str, max_tokens: int = 10, callback: Optional[Callable] = None):
        # Simulated generation loop for testing hooks
        tokens = []
        # Initial state (mock)
        x = mx.random.normal((1, self.dim))
        
        for _ in range(max_tokens):
            # 1. Forward pass (Simulate)
            # If we were a real LLM, we'd get logits. 
            # Here we just get a hidden state to steer.
            
            # --- HOOK POINT START ---
            if callback:
                # Callback receives (layer_output, layer_index)
                # We mock layer_index = 0
                x = callback(x, 0)
            # --- HOOK POINT END ---
            
            x = self.model(x)
            
            # Mock token selection
            tokens.append("token")
            
        return " ".join(tokens)

class SteeredGenerator:
    """
    Wraps an LLM and applies SAE-based steering during generation.
    """
    def __init__(self, model, sae_encoder: nn.Linear, sae_decoder: nn.Linear):
        self.model = model
        self.encoder = sae_encoder
        self.decoder = sae_decoder
        self.steering_vectors: Dict[int, float] = {} # Feature Index -> Steering Strength
        
    def set_steering(self, feature_idx: int, strength: float):
        """Sets the steering strength for a specific SAE feature."""
        self.steering_vectors[feature_idx] = strength
        
    def clear_steering(self):
        self.steering_vectors = {}
        
    def _steering_hook(self, x: mx.array, layer_idx: int) -> mx.array:
        """
        The hook function applied during generation.
        x: [Batch, Dim] - Activations at the hook point.
        """
        if not self.steering_vectors:
            return x
            
        # 1. Encode to Latent Space
        # latent: [Batch, SAE_Dim]
        latent = nn.relu(self.encoder(x))
        
        # 2. Apply Steering
        # We iterate through active steering targets
        for feat_idx, strength in self.steering_vectors.items():
            # Add strength to the specific feature activation
            # latent[:, feat_idx] += strength
            
            # Update: In MLX, we need to be careful with in-place ops or indexing
            # Construct a steering vector
            steer = mx.zeros_like(latent)
            steer[0, feat_idx] = strength # Assuming batch size 1 for generation
            latent = latent + steer

        # 3. Decode back to Activation Space
        # reconstruction: [Batch, Dim]
        reconstruction = self.decoder(latent)
        
        # 4. Residual Integration
        # Commonly: output = original + (reconstruction - original_reconstruction?) 
        # Or simpler: output = reconstruction (if SAE is perfect)
        # Or: output = original + error + steering
        # Let's assume we replace the activation with the steered reconstruction
        # for maximum effect, or add the delta.
        # Steering Delta = Decoder(Steered_Latent) - Decoder(Original_Latent)
        
        # For simplicity in this experiment: Replace with reconstruction
        return reconstruction

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """
        Generates text with the steering hook active.
        """
        # We pass our hook to the model's generate function
        # Note: This assumes the underlying model supports a callback/hook mechanism
        # like the MockLLM above. For `mlx_lm`, we would need to patch the model generator.
        
        return self.model.generate(prompt, max_tokens=max_tokens, callback=self._steering_hook)

