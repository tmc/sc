"""
Steered Sampler: Custom autoregressive decoding with steering hooks.

Implements token-by-token generation with steering vectors applied
at each forward pass step.

Key insight: Standard generation doesn't support hooks during autoregressive
decoding. This module provides custom decoding that applies steering at
every token generation step.

Uses mlux HookedModel.run_with_hooks for hook injection.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any, Tuple
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

try:
    from mlux import HookedModel
    from mlx_lm.sample_utils import make_sampler
    MLUX_AVAILABLE = True
except ImportError:
    HookedModel = None
    make_sampler = None
    MLUX_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SamplerConfig:
    """Configuration for steered sampling."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_tokens: List[int] = field(default_factory=list)
    steering_alpha: float = 1.0
    steering_layer: int = 12


@dataclass
class GenerationResult:
    """Result of steered generation."""
    prompt: str
    output: str
    tokens_generated: int
    steering_applied: bool
    layer: int
    alpha: float


# =============================================================================
# STEERED SAMPLER
# =============================================================================

class SteeredSampler:
    """
    Custom autoregressive sampler with steering vector injection.

    Unlike standard generation, this applies steering hooks at EVERY
    token generation step, not just the initial forward pass.
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

    def _get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if self._tokenizer is None:
            return 2  # Common default

        # Try different tokenizer attributes
        if hasattr(self._tokenizer, 'eos_token_id'):
            return self._tokenizer.eos_token_id
        if hasattr(self._tokenizer, 'eos_id'):
            return self._tokenizer.eos_id
        return 2

    @property
    def has_model(self) -> bool:
        return self.model is not None and MLUX_AVAILABLE

    def generate(
        self,
        prompt: str,
        steering_vector: Any = None,
        config: SamplerConfig = None,
    ) -> GenerationResult:
        """
        Generate text with optional steering.

        Args:
            prompt: Input prompt
            steering_vector: Vector to apply during generation (mlx array)
            config: Sampling configuration

        Returns:
            GenerationResult with generated text
        """
        config = config or SamplerConfig()

        if not self.has_model:
            return self._mock_generate(prompt, config)

        # Tokenize prompt
        tokens = self.model.tokenize(prompt)
        input_ids = mx.array([tokens])

        # Create sampler function
        sampler = make_sampler(temp=config.temperature, top_p=config.top_p)

        # Build hooks if steering vector provided
        hooks = None
        if steering_vector is not None:
            from .hook_injection import create_steering_hook
            hook_fn = create_steering_hook(
                steering_vector,
                alpha=config.steering_alpha
            )
            layer_name = f"model.layers.{config.steering_layer}"
            hooks = [(layer_name, hook_fn)]

        # Autoregressive generation loop
        generated_tokens = []
        current_ids = input_ids

        for _ in range(config.max_tokens):
            # Forward pass with hooks
            if hooks:
                logits = self.model.run_with_hooks(current_ids, hooks=hooks)
            else:
                logits = self.model.forward(current_ids)

            # Get logits for last token
            next_logits = logits[:, -1, :]

            # Sample next token
            next_token = sampler(next_logits)
            next_token_id = int(next_token.item())

            # Check for EOS
            if next_token_id == self._eos_token_id:
                break
            if config.stop_tokens and next_token_id in config.stop_tokens:
                break

            generated_tokens.append(next_token_id)

            # Update input for next iteration (KV cache would be more efficient)
            current_ids = mx.concatenate([current_ids, next_token.reshape(1, 1)], axis=1)

        # Decode generated tokens
        output = self._tokenizer.decode(generated_tokens)

        return GenerationResult(
            prompt=prompt,
            output=output,
            tokens_generated=len(generated_tokens),
            steering_applied=steering_vector is not None,
            layer=config.steering_layer,
            alpha=config.steering_alpha,
        )

    def generate_comparison(
        self,
        prompt: str,
        steering_vector: Any,
        config: SamplerConfig = None,
    ) -> Tuple[GenerationResult, GenerationResult]:
        """
        Generate both baseline and steered outputs for comparison.

        Returns:
            (baseline_result, steered_result)
        """
        config = config or SamplerConfig()

        # Baseline (no steering)
        baseline = self.generate(prompt, steering_vector=None, config=config)

        # Steered
        steered = self.generate(prompt, steering_vector=steering_vector, config=config)

        return baseline, steered

    def _mock_generate(self, prompt: str, config: SamplerConfig) -> GenerationResult:
        """Mock generation for testing without mlux."""
        mock_output = f"[MOCK] Generated response for: {prompt[:50]}..."
        return GenerationResult(
            prompt=prompt,
            output=mock_output,
            tokens_generated=10,
            steering_applied=False,
            layer=config.steering_layer,
            alpha=config.steering_alpha,
        )


# =============================================================================
# BATCH GENERATION
# =============================================================================

def batch_generate(
    sampler: SteeredSampler,
    prompts: List[str],
    steering_vector: Any = None,
    config: SamplerConfig = None,
) -> List[GenerationResult]:
    """Generate for multiple prompts."""
    results = []
    for prompt in prompts:
        result = sampler.generate(prompt, steering_vector, config)
        results.append(result)
    return results


def batch_comparison(
    sampler: SteeredSampler,
    prompts: List[str],
    steering_vector: Any,
    config: SamplerConfig = None,
) -> List[Tuple[GenerationResult, GenerationResult]]:
    """Generate baseline and steered for multiple prompts."""
    results = []
    for prompt in prompts:
        baseline, steered = sampler.generate_comparison(prompt, steering_vector, config)
        results.append((baseline, steered))
    return results


# =============================================================================
# TESTING
# =============================================================================

def test_steered_sampler():
    """Test steered sampler functionality."""
    print("=" * 60)
    print("STEERED SAMPLER TEST")
    print("=" * 60)

    print(f"\nMLX available: {MLX_AVAILABLE}")
    print(f"MLUX available: {MLUX_AVAILABLE}")

    # Create sampler
    print("\nCreating sampler...")
    sampler = SteeredSampler()
    print(f"Has model: {sampler.has_model}")

    # Test basic generation
    print("\nTesting basic generation...")
    config = SamplerConfig(max_tokens=50, temperature=0.3)
    result = sampler.generate("Generate a simple JSON:", config=config)
    print(f"Result: {result.output[:100]}...")
    print(f"Tokens: {result.tokens_generated}")

    # Test with steering (mock vector for now)
    if sampler.has_model:
        print("\nTesting with steering vector...")
        # Create a simple steering vector
        steering_vec = mx.zeros((896,))  # Placeholder
        result_steered = sampler.generate(
            "Generate a simple JSON:",
            steering_vector=steering_vec,
            config=config,
        )
        print(f"Steered result: {result_steered.output[:100]}...")
        print(f"Steering applied: {result_steered.steering_applied}")

    print("\n[PASS] Steered sampler test complete")
    return True


if __name__ == "__main__":
    test_steered_sampler()
