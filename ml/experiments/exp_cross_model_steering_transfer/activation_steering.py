"""
Real Activation Steering in MLX.

Actually modifies model activations during forward pass,
not just prompt engineering.
"""

import json
import re
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    vector: np.ndarray
    layer: int
    alpha: float = 1.0


def is_valid_sc_json(text: str) -> bool:
    """Check if text contains valid SC JSON structure."""
    try:
        # Handle markdown code blocks
        json_match = re.search(r'```json?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            start = text.find('{')
            if start == -1:
                return False
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = text[start:end]

        data = json.loads(json_str)
        if not isinstance(data, dict):
            return False
        if 'root_state' in data:
            root = data['root_state']
            if isinstance(root, dict) and 'label' in root:
                return True
        return False
    except:
        return False


class SteeredModel:
    """
    Wrapper that applies steering to a model's forward pass.

    Steering is applied by adding alpha * steering_vector to the
    hidden states at the specified layer.
    """

    def __init__(self, model, tokenizer, steering_config: Optional[SteeringConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_config = steering_config
        self.hidden_dim = model.args.hidden_size
        self.num_layers = model.args.num_hidden_layers

    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Generate with steering applied."""
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = list(tokens)

        for _ in range(max_tokens):
            # Forward pass with steering
            logits = self._forward_with_steering(mx.array([generated]))

            # Sample next token
            next_logits = logits[0, -1, :]

            if temperature > 0:
                # Temperature sampling
                probs = mx.softmax(next_logits / temperature)
                next_token = int(mx.random.categorical(mx.log(probs)))
            else:
                next_token = int(mx.argmax(next_logits))

            generated.append(next_token)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[len(tokens):])

    def _forward_with_steering(self, input_ids: mx.array) -> mx.array:
        """Forward pass with steering vector added at specified layer."""
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)

        # Create attention mask with correct dtype (float16 for 4-bit models)
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(mx.float16)  # Match model dtype

        # Process through layers
        for i, layer in enumerate(self.model.model.layers):
            hidden_states = layer(hidden_states, mask=mask, cache=None)

            # Apply steering at target layer
            if self.steering_config and i == self.steering_config.layer:
                steering = mx.array(self.steering_config.vector).astype(hidden_states.dtype) * self.steering_config.alpha
                # Add steering to all positions
                hidden_states = hidden_states + steering.reshape(1, 1, -1)

        # Final norm
        hidden_states = self.model.model.norm(hidden_states)

        # LM head
        logits = self.model.lm_head(hidden_states)

        return logits


def test_activation_steering():
    """Test real activation steering."""
    print("=" * 70)
    print("Real Activation Steering Test")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    hidden_dim = model.args.hidden_size
    num_layers = model.args.num_hidden_layers
    print(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")

    # Test prompts
    test_prompts = [
        "Generate valid statechart JSON with root_state:",
        "Create statechart JSON for On/Off toggle:",
        "Write SC JSON with states and transitions:",
    ]

    # Load real steering vector
    import os
    vectors_dir = os.path.join(os.path.dirname(__file__), "vectors_0.5B")
    from .real_vector_computation import RealSteeringVector

    # Adapt 0.5B vector (896 dim) to 1.5B (1536 dim)
    source_vec = RealSteeringVector.load(os.path.join(vectors_dir, "SC_MIDDLE_L12.npz"))
    adapted_vec = np.interp(
        np.linspace(0, 1, hidden_dim),
        np.linspace(0, 1, source_vec.vector.shape[0]),
        source_vec.vector
    )
    # Normalize
    adapted_vec = adapted_vec / np.linalg.norm(adapted_vec)

    # Map layer: 12/24 -> 14/28
    target_layer = int((source_vec.layer / 24) * num_layers)

    print(f"\nSteering vector: {source_vec.name}")
    print(f"  Adapted: {source_vec.vector.shape[0]} -> {hidden_dim} dims")
    print(f"  Layer: {source_vec.layer} -> {target_layer}")

    # Test different alpha values
    alphas = [0.0, 0.5, 1.0, 2.0]

    print("\n" + "-" * 70)
    for alpha in alphas:
        print(f"\nAlpha = {alpha}:")

        if alpha == 0:
            steering_config = None
        else:
            steering_config = SteeringConfig(
                vector=adapted_vec,
                layer=target_layer,
                alpha=alpha,
            )

        steered_model = SteeredModel(model, tokenizer, steering_config)

        valid_count = 0
        for prompt in test_prompts[:2]:  # Just 2 for speed
            try:
                output = steered_model.generate(prompt, max_tokens=100)
                is_valid = is_valid_sc_json(output)
                valid_count += 1 if is_valid else 0
                status = "VALID" if is_valid else "INVALID"
                print(f"  [{status}] {prompt[:40]}...")
            except Exception as e:
                print(f"  [ERROR] {prompt[:40]}... - {e}")

        print(f"  Rate: {valid_count}/{2} = {valid_count/2:.0%}")

    print("\n" + "=" * 70)
    print("Activation steering test complete")
    print("=" * 70)


if __name__ == "__main__":
    test_activation_steering()
