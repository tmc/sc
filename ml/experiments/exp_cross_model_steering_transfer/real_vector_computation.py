"""
REAL Steering Vector Computation.

Computes steering vectors from actual model activations, NOT simulated data.
Uses contrastive activation collection on SC vs non-SC prompts.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate


@dataclass
class RealSteeringVector:
    """A steering vector computed from real activations."""
    name: str
    vector: np.ndarray
    layer: int
    source_model: str
    num_samples: int  # How many activation pairs used

    def save(self, path: str):
        """Save vector to disk."""
        np.savez(path,
                 vector=self.vector,
                 name=self.name,
                 layer=self.layer,
                 source_model=self.source_model,
                 num_samples=self.num_samples)

    @classmethod
    def load(cls, path: str) -> 'RealSteeringVector':
        """Load vector from disk."""
        data = np.load(path, allow_pickle=True)
        return cls(
            name=str(data['name']),
            vector=data['vector'],
            layer=int(data['layer']),
            source_model=str(data['source_model']),
            num_samples=int(data['num_samples']),
        )


# Prompts designed to elicit SC JSON generation
SC_PROMPTS = [
    "Generate a statechart JSON with root_state containing states Off and On:",
    "Create a state machine JSON with transitions between Idle and Active:",
    "Write statechart JSON for a traffic light with Red, Yellow, Green states:",
    "Generate SC JSON with hierarchical states and nested children:",
    "Create a statechart with parallel regions for a vending machine:",
]

# Prompts that generate non-SC JSON (control group)
NON_SC_PROMPTS = [
    "Generate a JSON config for a web server with host and port:",
    "Create a JSON user profile with name, email, and age:",
    "Write JSON for a product catalog with items and prices:",
    "Generate a JSON API response with status and data fields:",
    "Create a JSON configuration for database connection:",
]


class ActivationCollector:
    """Collects activations from model forward passes."""

    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.activations = {}

        # Get model config
        self.hidden_dim = model.args.hidden_size
        self.num_layers = model.args.num_hidden_layers

    def get_activations_at_layer(self, text: str, layer_idx: int) -> np.ndarray:
        """Get residual stream activations at specified layer."""
        # Tokenize
        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        # Forward pass with activation capture
        # We need to hook into the model's forward pass
        # MLX models typically have model.model.layers[i] structure

        hidden_states = self.model.model.embed_tokens(input_ids)

        for i, layer in enumerate(self.model.model.layers):
            hidden_states = layer(hidden_states, mask=None, cache=None)
            if i == layer_idx:
                # Capture activation at this layer
                # Take mean over sequence dimension for a single vector
                activation = hidden_states.mean(axis=1)  # [1, hidden_dim]
                return np.array(activation[0])

        # If we get here, return last layer
        return np.array(hidden_states.mean(axis=1)[0])

    def collect_contrastive_activations(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect activations for positive and negative prompts."""
        positive_acts = []
        negative_acts = []

        print(f"  Collecting {len(positive_prompts)} positive activations...")
        for prompt in positive_prompts:
            act = self.get_activations_at_layer(prompt, layer_idx)
            positive_acts.append(act)

        print(f"  Collecting {len(negative_prompts)} negative activations...")
        for prompt in negative_prompts:
            act = self.get_activations_at_layer(prompt, layer_idx)
            negative_acts.append(act)

        return np.array(positive_acts), np.array(negative_acts)

    def compute_steering_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_idx: int,
        name: str,
    ) -> RealSteeringVector:
        """Compute steering vector as mean difference."""
        pos_acts, neg_acts = self.collect_contrastive_activations(
            positive_prompts, negative_prompts, layer_idx
        )

        # Steering vector = mean(positive) - mean(negative)
        pos_mean = pos_acts.mean(axis=0)
        neg_mean = neg_acts.mean(axis=0)
        steering_vec = pos_mean - neg_mean

        # Normalize
        norm = np.linalg.norm(steering_vec)
        if norm > 0:
            steering_vec = steering_vec / norm

        return RealSteeringVector(
            name=name,
            vector=steering_vec,
            layer=layer_idx,
            source_model=self.model_name,
            num_samples=len(positive_prompts) + len(negative_prompts),
        )


def compute_real_steering_vectors(model_id: str, model_name: str) -> List[RealSteeringVector]:
    """
    Compute real steering vectors from model activations.

    Returns vectors at multiple layers.
    """
    print(f"\n{'='*60}")
    print(f"Computing REAL steering vectors for {model_name}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading {model_id}...")
    model, tokenizer = load(model_id)

    collector = ActivationCollector(model, tokenizer, model_name)
    print(f"  Hidden dim: {collector.hidden_dim}")
    print(f"  Num layers: {collector.num_layers}")

    vectors = []

    # Compute vectors at key layers (early, middle, late)
    key_layers = [
        (collector.num_layers // 4, "EARLY"),      # ~L6 for 24 layers
        (collector.num_layers // 2, "MIDDLE"),     # ~L12 for 24 layers
        (3 * collector.num_layers // 4, "LATE"),   # ~L18 for 24 layers
        (collector.num_layers - 2, "FINAL"),       # ~L22 for 24 layers
    ]

    for layer_idx, layer_name in key_layers:
        print(f"\nComputing {layer_name} vector at layer {layer_idx}...")
        vec = collector.compute_steering_vector(
            SC_PROMPTS,
            NON_SC_PROMPTS,
            layer_idx,
            f"SC_{layer_name}_L{layer_idx}",
        )
        vectors.append(vec)
        print(f"  Vector norm: {np.linalg.norm(vec.vector):.4f}")
        print(f"  Vector shape: {vec.vector.shape}")

    return vectors


def save_vectors(vectors: List[RealSteeringVector], output_dir: str):
    """Save vectors to disk."""
    os.makedirs(output_dir, exist_ok=True)

    for vec in vectors:
        path = os.path.join(output_dir, f"{vec.name}.npz")
        vec.save(path)
        print(f"Saved: {path}")


def test_real_vector_computation():
    """Test real vector computation."""
    print("="*60)
    print("REAL Steering Vector Computation")
    print("="*60)

    # Compute for 0.5B
    vectors_05b = compute_real_steering_vectors(
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "Qwen-0.5B"
    )

    # Save vectors
    output_dir = os.path.dirname(__file__)
    vectors_dir = os.path.join(output_dir, "vectors_0.5B")
    save_vectors(vectors_05b, vectors_dir)

    print("\n" + "="*60)
    print(f"Computed {len(vectors_05b)} REAL steering vectors")
    print("="*60)

    return vectors_05b


if __name__ == "__main__":
    test_real_vector_computation()
