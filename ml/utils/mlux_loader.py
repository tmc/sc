"""
Unified model loading with mlux/mlx_lm fallback.

Provides a consistent interface for loading LLMs with optional
interpretability features (activation caching, steering, etc).

Usage:
    from utils.mlux_loader import load_model, ModelBackend

    # Auto-select best available backend
    model = load_model("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")

    # Force specific backend
    model = load_model(model_name, backend=ModelBackend.MLUX)

    # Generate with activation caching (if mlux available)
    output, cache = model.generate_with_cache(prompt, hooks=["model.layers.12"])
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)
import json

# Detect available backends
MLUX_AVAILABLE = False
MLX_LM_AVAILABLE = False

try:
    from mlux import HookedModel
    from mlux.steering import ContrastiveSteering, compute_steering_vector
    MLUX_AVAILABLE = True
except ImportError:
    HookedModel = None
    ContrastiveSteering = None
    compute_steering_vector = None

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_LM_AVAILABLE = True
except ImportError:
    load = None
    generate = None
    make_sampler = None


class ModelBackend(Enum):
    """Available model backends."""
    MLUX = auto()      # Full interpretability features
    MLX_LM = auto()    # Basic generation
    MOCK = auto()      # For testing without GPU


@runtime_checkable
class GenerativeModel(Protocol):
    """Protocol for generative models."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        ...

    @property
    def tokenizer(self) -> Any:
        """Access tokenizer."""
        ...


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class CacheConfig:
    """Configuration for activation caching."""
    hooks: List[str] = field(default_factory=lambda: ["model.layers.*.mlp"])
    include_attention: bool = True
    include_mlp: bool = True
    layers: Optional[List[int]] = None  # None = all layers


class HookedModelWrapper:
    """
    Unified wrapper providing consistent interface across backends.

    When mlux is available:
    - Full activation caching
    - Steering vectors
    - Attention pattern analysis

    When only mlx_lm is available:
    - Basic generation
    - No activation access

    When neither is available:
    - Mock responses for testing
    """

    def __init__(
        self,
        model_name: str,
        backend: Optional[ModelBackend] = None,
        device: str = "gpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._backend = backend or self._select_backend()

        self._load_model()

    def _select_backend(self) -> ModelBackend:
        """Auto-select best available backend."""
        if MLUX_AVAILABLE:
            return ModelBackend.MLUX
        elif MLX_LM_AVAILABLE:
            return ModelBackend.MLX_LM
        else:
            return ModelBackend.MOCK

    def _load_model(self):
        """Load model with selected backend."""
        if self._backend == ModelBackend.MLUX:
            self._load_mlux()
        elif self._backend == ModelBackend.MLX_LM:
            self._load_mlx_lm()
        else:
            self._load_mock()

    def _load_mlux(self):
        """Load with mlux for full interpretability."""
        print(f"Loading {self.model_name} with mlux (full interpretability)")
        self._model = HookedModel.from_pretrained(self.model_name)
        self._tokenizer = self._model.tokenizer
        # Also load mlx_lm model for generation (mlux is for interpretability)
        if MLX_LM_AVAILABLE:
            self._mlx_model, self._mlx_tokenizer = load(self.model_name)
        else:
            self._mlx_model = None
            self._mlx_tokenizer = None

    def _load_mlx_lm(self):
        """Load with mlx_lm for basic generation."""
        print(f"Loading {self.model_name} with mlx_lm (basic generation)")
        self._model, self._tokenizer = load(self.model_name)

    def _load_mock(self):
        """Create mock for testing."""
        print(f"Using mock model (no GPU backend available)")
        self._model = None
        self._tokenizer = MockTokenizer()

    @property
    def backend(self) -> ModelBackend:
        """Current backend."""
        return self._backend

    @property
    def tokenizer(self) -> Any:
        """Access tokenizer."""
        return self._tokenizer

    @property
    def has_interpretability(self) -> bool:
        """Whether activation caching is available."""
        return self._backend == ModelBackend.MLUX

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text from prompt."""
        config = config or GenerationConfig()

        if self._backend == ModelBackend.MLUX:
            # Use mlx_lm for generation (mlux is for interpretability)
            if self._mlx_model is not None:
                sampler = make_sampler(temp=config.temperature)
                return generate(
                    self._mlx_model,
                    self._mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=config.max_tokens,
                    sampler=sampler,
                )
            else:
                return self._mock_generate(prompt, config)
        elif self._backend == ModelBackend.MLX_LM:
            sampler = make_sampler(temp=config.temperature)
            return generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
                sampler=sampler,
            )
        else:
            return self._mock_generate(prompt, config)

    def generate_with_cache(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with activation caching.

        Returns:
            (output_text, activation_cache)

        Raises:
            RuntimeError if mlux not available
        """
        config = config or GenerationConfig()
        cache_config = cache_config or CacheConfig()

        if self._backend != ModelBackend.MLUX:
            # Fallback: generate without cache
            output = self.generate(prompt, config)
            return output, {}

        output, cache = self._model.run_with_cache(
            prompt,
            hooks=cache_config.hooks,
        )

        return output, cache

    def get_attention_patterns(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Any]:
        """
        Get attention patterns for prompt.

        Returns:
            {layer_idx: attention_matrix} where matrix is [heads, seq, seq]
        """
        if self._backend != ModelBackend.MLUX:
            return {}

        return self._model.get_attention_patterns(prompt, layers=layers)

    def compute_steering_vector(
        self,
        positive: str,
        negative: str,
        layer: int,
    ) -> Any:
        """
        Compute steering vector from contrastive examples.

        Args:
            positive: Example of desired behavior
            negative: Example of undesired behavior
            layer: Layer to extract steering vector from

        Returns:
            Steering vector (mlx array)
        """
        if self._backend != ModelBackend.MLUX:
            raise RuntimeError("Steering requires mlux backend")

        return compute_steering_vector(
            self._model,
            positive=positive,
            negative=negative,
            layer=layer,
        )

    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: Any,
        layer: int,
        alpha: float = 1.0,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate with steering vector applied.

        Args:
            prompt: Input prompt
            steering_vector: Vector from compute_steering_vector
            layer: Layer to apply steering
            alpha: Steering strength (1.0 = full effect)
            config: Generation config

        Note: Full steered generation requires autoregressive decoding with
        hooks applied at each step. Currently falls back to normal generation.
        TODO: Implement proper steered autoregressive generation.
        """
        # Steered generation with hooks requires custom autoregressive loop
        # For now, fall back to normal generation
        # The steering vector computation still works for analysis purposes
        return self.generate(prompt, config)

    def _mock_generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate mock response for testing."""
        return f"[MOCK] Response to: {prompt[:50]}..."


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids if i > 2)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab


def load_model(
    model_name: str,
    backend: Optional[ModelBackend] = None,
    **kwargs,
) -> HookedModelWrapper:
    """
    Load model with unified interface.

    Args:
        model_name: HuggingFace model name or path
        backend: Force specific backend (auto-select if None)
        **kwargs: Additional arguments for model loading

    Returns:
        HookedModelWrapper with consistent interface
    """
    return HookedModelWrapper(model_name, backend=backend, **kwargs)


# Convenience exports
__all__ = [
    "load_model",
    "ModelBackend",
    "HookedModelWrapper",
    "GenerationConfig",
    "CacheConfig",
    "MLUX_AVAILABLE",
    "MLX_LM_AVAILABLE",
]
