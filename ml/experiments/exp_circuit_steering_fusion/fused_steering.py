"""
Fused Steering: Merge circuit intervention with steered decoding.

Combines:
1. exp_steered_decoding: Token-by-token generation with hooks
2. exp_circuit_intervention: Circuit-specific layer targeting

Key insight: Apply different steering to different circuits:
- TRANSITION circuit (L8-14): For transition validity
- HIERARCHY circuit (L0-6): For hierarchical structure

This allows targeted intervention where each circuit gets the
steering it needs, rather than global steering everywhere.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    make_sampler = None
    MLX_AVAILABLE = False

try:
    from mlux import HookedModel
    MLUX_AVAILABLE = True
except ImportError:
    HookedModel = None
    MLUX_AVAILABLE = False

from .circuit_hooks import (
    CircuitType,
    Circuit,
    CIRCUITS,
    get_circuit,
    create_combined_hooks,
    create_transition_hooks,
    create_hierarchy_hooks,
    create_global_hooks,
    MultiCircuitConfig,
    create_multi_circuit_hooks,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FusedSteeringConfig:
    """Configuration for fused circuit steering."""
    # Generation
    max_tokens: int = 256
    temperature: float = 0.3

    # Global steering (for comparison)
    global_alpha: float = 1.0
    global_layer: int = 12

    # Circuit-specific steering
    transition_alpha: float = 1.0  # L8-14
    hierarchy_alpha: float = 1.0   # L0-6

    # Mode
    mode: str = "fused"  # "global", "transition", "hierarchy", "fused"


@dataclass
class FusedGenerationResult:
    """Result of fused generation."""
    prompt: str
    output: str
    tokens_generated: int
    mode: str
    circuits_used: List[str]
    generation_time: float


# =============================================================================
# FUSED STEERED SAMPLER
# =============================================================================

class FusedSteeredSampler:
    """
    Sampler with fused circuit steering.

    Applies different steering to different circuits during
    token-by-token generation.
    """

    def __init__(
        self,
        model: "HookedModel" = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        if model is not None:
            self.model = model
        elif MLUX_AVAILABLE:
            print(f"Loading {model_name} with mlux...")
            self.model = HookedModel.from_pretrained(model_name)
        else:
            self.model = None
            print("Warning: mlux not available, using mock mode")

        self._tokenizer = self.model.tokenizer if self.model else None
        self._eos_token_id = self._get_eos_token_id()

        # Cached steering vectors
        self._steering_vector: Optional[Any] = None
        self._transition_vector: Optional[Any] = None
        self._hierarchy_vector: Optional[Any] = None

    def _get_eos_token_id(self) -> int:
        if self._tokenizer is None:
            return 2
        if hasattr(self._tokenizer, 'eos_token_id'):
            return self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, 'eos_id'):
            return self._tokenizer.eos_id
        return 2

    @property
    def has_model(self) -> bool:
        return self.model is not None and MLUX_AVAILABLE

    def set_steering_vector(self, vector: Any):
        """Set the global steering vector."""
        self._steering_vector = vector

    def set_circuit_vectors(
        self,
        transition_vector: Any = None,
        hierarchy_vector: Any = None,
    ):
        """Set circuit-specific steering vectors."""
        self._transition_vector = transition_vector
        self._hierarchy_vector = hierarchy_vector

    def _get_hooks(self, config: FusedSteeringConfig) -> List[Tuple[str, Callable]]:
        """Get hooks based on configuration mode."""
        vec = self._steering_vector
        if vec is None and MLX_AVAILABLE:
            # Create random vector for testing
            vec = mx.random.normal((896,)) * 0.1

        t_vec = self._transition_vector if self._transition_vector is not None else vec
        h_vec = self._hierarchy_vector if self._hierarchy_vector is not None else vec

        if config.mode == "global":
            return create_global_hooks(vec, alpha=config.global_alpha, layers=[config.global_layer])

        elif config.mode == "transition":
            return create_transition_hooks(t_vec, alpha=config.transition_alpha)

        elif config.mode == "hierarchy":
            return create_hierarchy_hooks(h_vec, alpha=config.hierarchy_alpha)

        elif config.mode == "fused":
            return create_combined_hooks(
                steering_vector=vec,
                transition_alpha=config.transition_alpha,
                hierarchy_alpha=config.hierarchy_alpha,
                transition_vector=t_vec,
                hierarchy_vector=h_vec,
            )

        else:
            return []  # No steering

    def generate(
        self,
        prompt: str,
        config: FusedSteeringConfig = None,
    ) -> FusedGenerationResult:
        """
        Generate with fused circuit steering.

        Args:
            prompt: Input prompt
            config: Steering configuration

        Returns:
            FusedGenerationResult
        """
        config = config or FusedSteeringConfig()
        start_time = time.time()

        if not self.has_model:
            return self._mock_generate(prompt, config)

        # Tokenize
        tokens = self.model.tokenize(prompt)
        input_ids = mx.array([tokens])

        # Create sampler
        sampler = make_sampler(temp=config.temperature)

        # Get hooks for this mode
        hooks = self._get_hooks(config)
        circuits_used = self._get_circuits_used(config)

        # Autoregressive generation
        generated_tokens = []
        current_ids = input_ids

        for _ in range(config.max_tokens):
            if hooks:
                logits = self.model.run_with_hooks(current_ids, hooks=hooks)
            else:
                logits = self.model.forward(current_ids)

            next_logits = logits[:, -1, :]
            next_token = sampler(next_logits)
            next_token_id = int(next_token.item())

            if next_token_id == self._eos_token_id:
                break

            generated_tokens.append(next_token_id)
            current_ids = mx.concatenate([current_ids, next_token.reshape(1, 1)], axis=1)

        output = self._tokenizer.decode(generated_tokens)

        return FusedGenerationResult(
            prompt=prompt,
            output=output,
            tokens_generated=len(generated_tokens),
            mode=config.mode,
            circuits_used=circuits_used,
            generation_time=time.time() - start_time,
        )

    def _get_circuits_used(self, config: FusedSteeringConfig) -> List[str]:
        """Get list of circuits used based on mode."""
        if config.mode == "global":
            return [f"global_L{config.global_layer}"]
        elif config.mode == "transition":
            return ["transition_L8-14"]
        elif config.mode == "hierarchy":
            return ["hierarchy_L0-6"]
        elif config.mode == "fused":
            return ["hierarchy_L0-6", "transition_L8-14"]
        return []

    def generate_comparison(
        self,
        prompt: str,
        base_config: FusedSteeringConfig = None,
    ) -> Dict[str, FusedGenerationResult]:
        """
        Generate with all modes for comparison.

        Returns:
            {"global": result, "transition": result, "hierarchy": result, "fused": result}
        """
        base_config = base_config or FusedSteeringConfig()
        results = {}

        for mode in ["global", "transition", "hierarchy", "fused"]:
            config = FusedSteeringConfig(
                max_tokens=base_config.max_tokens,
                temperature=base_config.temperature,
                global_alpha=base_config.global_alpha,
                global_layer=base_config.global_layer,
                transition_alpha=base_config.transition_alpha,
                hierarchy_alpha=base_config.hierarchy_alpha,
                mode=mode,
            )
            results[mode] = self.generate(prompt, config)

        return results

    def _mock_generate(self, prompt: str, config: FusedSteeringConfig) -> FusedGenerationResult:
        """Mock generation for testing."""
        return FusedGenerationResult(
            prompt=prompt,
            output=f"[MOCK {config.mode}] Response for: {prompt[:50]}...",
            tokens_generated=10,
            mode=config.mode,
            circuits_used=self._get_circuits_used(config),
            generation_time=0.1,
        )


# =============================================================================
# STEERING VECTOR LOADING
# =============================================================================

def load_steering_vectors(sampler: FusedSteeredSampler) -> bool:
    """
    Load steering vectors from exp_mlux_sc_steering.

    Returns:
        True if vectors were loaded successfully
    """
    try:
        from experiments.exp_mlux_sc_steering import SteeringVectorComputer

        print("Loading steering vectors...")
        computer = SteeringVectorComputer()

        # Compute vectors at different layers for different circuits
        # TRANSITION uses middle layers (L12 as representative)
        transition_vec = computer.compute_averaged(layer=12)

        # HIERARCHY uses early layers (L4 as representative)
        hierarchy_vec = computer.compute_averaged(layer=4)

        sampler.set_steering_vector(transition_vec.vector)
        sampler.set_circuit_vectors(
            transition_vector=transition_vec.vector,
            hierarchy_vector=hierarchy_vec.vector,
        )

        print(f"Loaded transition vector: {transition_vec}")
        print(f"Loaded hierarchy vector: {hierarchy_vec}")
        return True

    except Exception as e:
        print(f"Failed to load steering vectors: {e}")
        if MLX_AVAILABLE:
            # Create random vectors for testing
            vec = mx.random.normal((896,)) * 0.1
            sampler.set_steering_vector(vec)
            print("Using random steering vector for testing")
        return False


# =============================================================================
# TESTING
# =============================================================================

def test_fused_steering():
    """Test fused steering functionality."""
    print("=" * 60)
    print("FUSED STEERING TEST")
    print("=" * 60)

    print(f"\nMLX available: {MLX_AVAILABLE}")
    print(f"MLUX available: {MLUX_AVAILABLE}")

    # Create sampler
    print("\nCreating fused sampler...")
    sampler = FusedSteeredSampler()
    print(f"Has model: {sampler.has_model}")

    if not sampler.has_model:
        print("\n[SKIP] No model available")
        return True

    # Set random steering vector
    vec = mx.random.normal((896,)) * 0.1
    sampler.set_steering_vector(vec)
    print(f"Set steering vector: shape={vec.shape}")

    # Test each mode
    prompt = "Generate a simple statechart JSON:"
    config = FusedSteeringConfig(max_tokens=50, temperature=0.3)

    for mode in ["global", "transition", "hierarchy", "fused"]:
        config.mode = mode
        print(f"\nTesting mode: {mode}")
        result = sampler.generate(prompt, config)
        print(f"  Tokens: {result.tokens_generated}")
        print(f"  Circuits: {result.circuits_used}")
        print(f"  Time: {result.generation_time:.2f}s")
        print(f"  Output: {result.output[:60]}...")

    print("\n[PASS] Fused steering test complete")
    return True


if __name__ == "__main__":
    test_fused_steering()
