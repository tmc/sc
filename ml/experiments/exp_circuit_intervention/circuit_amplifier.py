"""
Circuit Amplifier for Targeted Interventions

Amplifies activations in specific circuits to improve statechart validity.

CIRCUITS (from exp_mlux_sc_circuits):
- TRANSITION_VALIDITY (L8-14): Amplify to fix transition errors
- HIERARCHY (L0-6): Boost for better nested state handling
- STRUCTURAL (L0-6): Strengthen for JSON structure

AMPLIFICATION METHODS:
1. Scaling: Multiply activations by factor > 1
2. Steering: Add direction vectors
3. Clamping: Ensure minimum activation magnitude
4. Attention boost: Increase attention weights
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from enum import Enum

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    nn = None
    MLX_AVAILABLE = False


class CircuitType(Enum):
    """Circuit types from exp_mlux_sc_circuits."""
    TRANSITION_VALIDITY = "transition_validity"
    HIERARCHY = "hierarchy"
    STRUCTURAL = "structural"
    STATE_NAME_MEMORY = "state_name_memory"


@dataclass
class CircuitSpec:
    """Specification of a circuit's location."""
    circuit_type: CircuitType
    layers: List[int]
    components: List[str]  # "attention", "mlp", or both
    importance: float = 1.0

    @classmethod
    def transition_validity(cls) -> "CircuitSpec":
        """TRANSITION_VALIDITY circuit: layers 8-14."""
        return cls(
            circuit_type=CircuitType.TRANSITION_VALIDITY,
            layers=list(range(8, 15)),
            components=["attention", "mlp"],
            importance=1.0,
        )

    @classmethod
    def hierarchy(cls) -> "CircuitSpec":
        """HIERARCHY circuit: layers 0-6."""
        return cls(
            circuit_type=CircuitType.HIERARCHY,
            layers=list(range(0, 7)),
            components=["attention", "mlp"],
            importance=1.0,
        )

    @classmethod
    def structural(cls) -> "CircuitSpec":
        """STRUCTURAL circuit: layers 0-6."""
        return cls(
            circuit_type=CircuitType.STRUCTURAL,
            layers=list(range(0, 7)),
            components=["attention", "mlp"],
            importance=1.0,
        )


class AmplificationType(Enum):
    """Types of circuit amplification."""
    SCALE = "scale"           # Multiply by factor
    STEER = "steer"           # Add steering vector
    CLAMP = "clamp"           # Ensure minimum magnitude
    ATTENTION_BOOST = "attention_boost"  # Boost attention weights


@dataclass
class AmplificationConfig:
    """Configuration for circuit amplification."""
    circuit: CircuitSpec
    amp_type: AmplificationType
    strength: float = 1.5  # Amplification factor
    steering_vector: Optional[Any] = None  # For STEER type
    min_magnitude: float = 0.1  # For CLAMP type
    attention_scale: float = 1.2  # For ATTENTION_BOOST


@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    original_output: str
    modified_output: str
    circuit_type: CircuitType
    amp_type: AmplificationType
    strength: float
    validity_before: float
    validity_after: float

    @property
    def improvement(self) -> float:
        return self.validity_after - self.validity_before


class CircuitAmplifier:
    """
    Amplifies specific circuits to improve statechart generation.

    Uses hooks to modify activations in targeted layers.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: HookedModelWrapper from mlux_loader
            verbose: Print progress
        """
        self.model = model
        self.verbose = verbose

        # Default circuit specs (from exp_mlux_sc_circuits)
        self.circuits = {
            CircuitType.TRANSITION_VALIDITY: CircuitSpec.transition_validity(),
            CircuitType.HIERARCHY: CircuitSpec.hierarchy(),
            CircuitType.STRUCTURAL: CircuitSpec.structural(),
        }

        # Cached steering vectors
        self._steering_cache: Dict[str, Any] = {}

    def create_scaling_hook(
        self,
        scale: float,
        component: str = "all",
    ) -> Callable:
        """Create hook that scales activations."""
        def hook(module, inputs, outputs):
            if MLX_AVAILABLE:
                return outputs * scale
            return outputs

        return hook

    def create_steering_hook(
        self,
        direction: Any,
        strength: float = 1.0,
    ) -> Callable:
        """Create hook that adds steering direction."""
        def hook(module, inputs, outputs):
            if MLX_AVAILABLE and direction is not None:
                # Broadcast direction to match output shape
                dir_expanded = mx.broadcast_to(direction, outputs.shape)
                return outputs + strength * dir_expanded
            return outputs

        return hook

    def create_clamp_hook(
        self,
        min_magnitude: float,
    ) -> Callable:
        """Create hook that ensures minimum activation magnitude."""
        def hook(module, inputs, outputs):
            if MLX_AVAILABLE:
                magnitude = mx.sqrt(mx.sum(outputs ** 2, axis=-1, keepdims=True))
                scale = mx.maximum(magnitude, min_magnitude) / (magnitude + 1e-8)
                return outputs * scale
            return outputs

        return hook

    def create_attention_boost_hook(
        self,
        scale: float,
    ) -> Callable:
        """Create hook that boosts attention weights."""
        def hook(module, inputs, outputs):
            if MLX_AVAILABLE:
                # Assuming outputs are attention weights
                # Sharpen by scaling then re-normalizing
                scaled = outputs * scale
                return scaled / (mx.sum(scaled, axis=-1, keepdims=True) + 1e-8)
            return outputs

        return hook

    def get_hooks_for_circuit(
        self,
        config: AmplificationConfig,
    ) -> List[Tuple[str, Callable]]:
        """Get all hooks needed for a circuit intervention."""
        hooks = []
        circuit = config.circuit

        # Create the appropriate hook function
        if config.amp_type == AmplificationType.SCALE:
            hook_fn = self.create_scaling_hook(config.strength)
        elif config.amp_type == AmplificationType.STEER:
            hook_fn = self.create_steering_hook(
                config.steering_vector, config.strength
            )
        elif config.amp_type == AmplificationType.CLAMP:
            hook_fn = self.create_clamp_hook(config.min_magnitude)
        elif config.amp_type == AmplificationType.ATTENTION_BOOST:
            hook_fn = self.create_attention_boost_hook(config.attention_scale)
        else:
            return hooks

        # Create hooks for each layer and component
        for layer in circuit.layers:
            for comp in circuit.components:
                if comp == "attention":
                    path = f"model.layers.{layer}.self_attn"
                elif comp == "mlp":
                    path = f"model.layers.{layer}.mlp"
                else:
                    continue

                hooks.append((path, hook_fn))

        return hooks

    def amplify(
        self,
        prompt: str,
        config: AmplificationConfig,
    ) -> str:
        """
        Generate with circuit amplification.

        Args:
            prompt: Input prompt
            config: Amplification configuration

        Returns:
            Generated output with amplified circuit
        """
        if self.model is None:
            return self._mock_amplify(prompt, config)

        hooks = self.get_hooks_for_circuit(config)

        if hasattr(self.model, 'has_interpretability') and self.model.has_interpretability:
            # Use mlux hooks
            output = self.model._model.run_with_hooks(prompt, hooks=hooks)
        else:
            # Fallback to normal generation
            output = self.model.generate(prompt)

        return output

    def compute_steering_vector(
        self,
        circuit_type: CircuitType,
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> Any:
        """
        Compute steering vector for a circuit.

        Args:
            circuit_type: Which circuit to compute vector for
            positive_examples: Examples of desired behavior
            negative_examples: Examples of undesired behavior

        Returns:
            Steering vector
        """
        cache_key = f"{circuit_type.value}"
        if cache_key in self._steering_cache:
            return self._steering_cache[cache_key]

        if self.model is None or not MLX_AVAILABLE:
            # Return mock vector
            return None

        circuit = self.circuits[circuit_type]

        # Compute mean activations for positive and negative
        pos_activations = []
        neg_activations = []

        for layer in circuit.layers:
            for ex in positive_examples[:5]:  # Limit samples
                _, cache = self.model.generate_with_cache(ex)
                if f"model.layers.{layer}" in cache:
                    pos_activations.append(cache[f"model.layers.{layer}"])

            for ex in negative_examples[:5]:
                _, cache = self.model.generate_with_cache(ex)
                if f"model.layers.{layer}" in cache:
                    neg_activations.append(cache[f"model.layers.{layer}"])

        if not pos_activations or not neg_activations:
            return None

        # Compute difference
        pos_mean = mx.mean(mx.stack(pos_activations), axis=0)
        neg_mean = mx.mean(mx.stack(neg_activations), axis=0)
        steering = pos_mean - neg_mean

        self._steering_cache[cache_key] = steering
        return steering

    def _mock_amplify(
        self,
        prompt: str,
        config: AmplificationConfig,
    ) -> str:
        """Mock amplification for testing."""
        # Simulate improved output based on circuit type
        circuit_type = config.circuit.circuit_type

        if circuit_type == CircuitType.TRANSITION_VALIDITY:
            # Better transitions
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "s0", "type": 1, "is_initial": True},
                        {"label": "s1", "type": 1},
                        {"label": "s2", "type": 1},
                    ]
                },
                "transitions": [
                    {"from": ["s0"], "to": ["s1"], "event": "E1"},
                    {"from": ["s1"], "to": ["s2"], "event": "E2"},
                    {"from": ["s2"], "to": ["s0"], "event": "E3"},
                ]
            })

        elif circuit_type == CircuitType.HIERARCHY:
            # Better hierarchy
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "inactive", "type": 1, "is_initial": True},
                        {
                            "label": "active",
                            "type": 2,
                            "children": [
                                {"label": "running", "type": 1, "is_initial": True},
                                {"label": "paused", "type": 1},
                            ]
                        },
                    ]
                },
                "transitions": [
                    {"from": ["inactive"], "to": ["active"], "event": "START"},
                    {"from": ["running"], "to": ["paused"], "event": "PAUSE"},
                ]
            })

        else:
            # Default valid output
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "s0", "type": 1, "is_initial": True},
                        {"label": "s1", "type": 1},
                    ]
                },
                "transitions": [
                    {"from": ["s0"], "to": ["s1"], "event": "E1"},
                ]
            })


# =============================================================================
# Preset amplification configs
# =============================================================================

def transition_fix_config(strength: float = 1.5) -> AmplificationConfig:
    """Config to fix transition validity issues."""
    return AmplificationConfig(
        circuit=CircuitSpec.transition_validity(),
        amp_type=AmplificationType.SCALE,
        strength=strength,
    )


def hierarchy_boost_config(strength: float = 1.3) -> AmplificationConfig:
    """Config to improve hierarchy handling."""
    return AmplificationConfig(
        circuit=CircuitSpec.hierarchy(),
        amp_type=AmplificationType.SCALE,
        strength=strength,
    )


def structural_fix_config(strength: float = 1.2) -> AmplificationConfig:
    """Config to improve structural validity."""
    return AmplificationConfig(
        circuit=CircuitSpec.structural(),
        amp_type=AmplificationType.CLAMP,
        min_magnitude=0.1,
        strength=strength,
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate circuit amplification."""
    print("=" * 60)
    print("Circuit Amplifier for Targeted Interventions")
    print("=" * 60)

    amplifier = CircuitAmplifier(model=None, verbose=True)

    # Test different amplification configs
    configs = [
        ("Transition Fix", transition_fix_config(1.5)),
        ("Hierarchy Boost", hierarchy_boost_config(1.3)),
        ("Structural Fix", structural_fix_config(1.2)),
    ]

    prompt = "Generate a state machine for a login flow:"

    print(f"\nPrompt: {prompt}")

    for name, config in configs:
        print(f"\n{'-'*40}")
        print(f"{name}:")
        print(f"  Circuit: {config.circuit.circuit_type.value}")
        print(f"  Layers: {config.circuit.layers}")
        print(f"  Type: {config.amp_type.value}")
        print(f"  Strength: {config.strength}")

        output = amplifier.amplify(prompt, config)
        chart = json.loads(output)

        print(f"  States: {len(chart['root_state'].get('children', []))}")
        print(f"  Transitions: {len(chart.get('transitions', []))}")

    return amplifier


if __name__ == "__main__":
    demo()
