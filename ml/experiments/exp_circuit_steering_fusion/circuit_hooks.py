"""
Circuit Hooks: Circuit-specific steering hooks for fused intervention.

Defines circuits and creates hooks that apply steering to specific layer ranges:
- TRANSITION circuit (L8-14): For transition validity
- HIERARCHY circuit (L0-6): For hierarchical structure

Uses exp_circuit_intervention patterns for circuit definitions
and exp_steered_decoding patterns for hook creation.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any, Tuple
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


# =============================================================================
# CIRCUIT DEFINITIONS
# =============================================================================

class CircuitType(Enum):
    """Circuit types for statechart generation."""
    TRANSITION = "transition"      # L8-14: Transition validity
    HIERARCHY = "hierarchy"        # L0-6: Hierarchical structure
    STRUCTURAL = "structural"      # L0-6: JSON structure
    GLOBAL = "global"              # All layers: General steering


@dataclass
class Circuit:
    """Definition of a circuit (layer range with specific function)."""
    circuit_type: CircuitType
    layers: List[int]
    description: str
    importance: float = 1.0

    def __repr__(self):
        return f"Circuit({self.circuit_type.value}, layers={self.layers})"


# Pre-defined circuits based on exp_mlux_sc_circuits analysis
CIRCUITS = {
    CircuitType.TRANSITION: Circuit(
        circuit_type=CircuitType.TRANSITION,
        layers=list(range(8, 15)),  # L8-14
        description="Transition validity: source/target state matching",
        importance=1.0,
    ),
    CircuitType.HIERARCHY: Circuit(
        circuit_type=CircuitType.HIERARCHY,
        layers=list(range(0, 7)),  # L0-6
        description="Hierarchical structure: nested states, parent-child",
        importance=1.0,
    ),
    CircuitType.STRUCTURAL: Circuit(
        circuit_type=CircuitType.STRUCTURAL,
        layers=list(range(0, 7)),  # L0-6
        description="JSON structure: braces, commas, syntax",
        importance=0.8,
    ),
    CircuitType.GLOBAL: Circuit(
        circuit_type=CircuitType.GLOBAL,
        layers=list(range(0, 24)),  # All layers
        description="Global steering across all layers",
        importance=1.0,
    ),
}


def get_circuit(circuit_type: CircuitType) -> Circuit:
    """Get circuit by type."""
    return CIRCUITS.get(circuit_type, CIRCUITS[CircuitType.GLOBAL])


# =============================================================================
# CIRCUIT HOOKS
# =============================================================================

@dataclass
class CircuitHook:
    """A hook configured for a specific circuit."""
    circuit: Circuit
    hook_fn: Callable
    alpha: float
    layer: int
    layer_name: str

    def __repr__(self):
        return f"CircuitHook({self.circuit.circuit_type.value}, L{self.layer}, alpha={self.alpha})"


def create_circuit_hook(
    steering_vector: Any,
    alpha: float = 1.0,
    position: str = "last",
) -> Callable:
    """
    Create a steering hook for a circuit layer.

    Same as exp_steered_decoding.create_steering_hook but duplicated
    here for independence.

    Args:
        steering_vector: Vector to add to hidden states
        alpha: Scaling factor
        position: Where to apply ("last", "all", or int)

    Returns:
        Hook function for mlux run_with_hooks
    """
    scaled_vec = steering_vector * alpha if MLX_AVAILABLE else steering_vector

    def hook(args: tuple, output: Any, wrapper: Any) -> Any:
        if output is None:
            return None

        if position == "last":
            modified = mx.concatenate([
                output[:, :-1, :],
                output[:, -1:, :] + scaled_vec.reshape(1, 1, -1)
            ], axis=1)
        elif position == "all":
            modified = output + scaled_vec.reshape(1, 1, -1)
        else:
            modified = output + scaled_vec.reshape(1, 1, -1)

        return modified

    return hook


def create_circuit_hooks(
    circuit: Circuit,
    steering_vector: Any,
    alpha: float = 1.0,
    decay: float = 1.0,
) -> List[Tuple[str, Callable]]:
    """
    Create hooks for all layers in a circuit.

    Args:
        circuit: Circuit definition
        steering_vector: Base steering vector
        alpha: Base steering strength
        decay: Decay factor per layer (1.0 = no decay)

    Returns:
        List of (layer_name, hook_fn) tuples for run_with_hooks
    """
    hooks = []

    for i, layer in enumerate(circuit.layers):
        # Apply decay: later layers get weaker steering
        layer_alpha = alpha * (decay ** i)
        hook_fn = create_circuit_hook(steering_vector, alpha=layer_alpha)
        layer_name = f"model.layers.{layer}"
        hooks.append((layer_name, hook_fn))

    return hooks


# =============================================================================
# MULTI-CIRCUIT HOOKS
# =============================================================================

@dataclass
class MultiCircuitConfig:
    """Configuration for multi-circuit steering."""
    circuits: List[CircuitType]
    alphas: Dict[CircuitType, float] = field(default_factory=dict)
    vectors: Dict[CircuitType, Any] = field(default_factory=dict)
    decay: float = 1.0

    def get_alpha(self, circuit_type: CircuitType) -> float:
        """Get alpha for circuit (default: 1.0)."""
        return self.alphas.get(circuit_type, 1.0)

    def get_vector(self, circuit_type: CircuitType) -> Optional[Any]:
        """Get steering vector for circuit."""
        return self.vectors.get(circuit_type)


def create_multi_circuit_hooks(
    config: MultiCircuitConfig,
    default_vector: Any = None,
) -> List[Tuple[str, Callable]]:
    """
    Create hooks for multiple circuits.

    Each circuit can have its own steering vector and alpha.
    If a circuit doesn't have a specific vector, uses default_vector.

    Args:
        config: Multi-circuit configuration
        default_vector: Default steering vector if circuit-specific not available

    Returns:
        List of all (layer_name, hook_fn) tuples
    """
    all_hooks = []

    for circuit_type in config.circuits:
        circuit = get_circuit(circuit_type)
        vector = config.get_vector(circuit_type) or default_vector

        if vector is None:
            continue

        alpha = config.get_alpha(circuit_type)
        hooks = create_circuit_hooks(circuit, vector, alpha, config.decay)
        all_hooks.extend(hooks)

    return all_hooks


# =============================================================================
# SPECIALIZED CIRCUIT HOOKS
# =============================================================================

def create_transition_hooks(
    steering_vector: Any,
    alpha: float = 1.0,
) -> List[Tuple[str, Callable]]:
    """Create hooks for TRANSITION circuit (L8-14)."""
    circuit = get_circuit(CircuitType.TRANSITION)
    return create_circuit_hooks(circuit, steering_vector, alpha)


def create_hierarchy_hooks(
    steering_vector: Any,
    alpha: float = 1.0,
) -> List[Tuple[str, Callable]]:
    """Create hooks for HIERARCHY circuit (L0-6)."""
    circuit = get_circuit(CircuitType.HIERARCHY)
    return create_circuit_hooks(circuit, steering_vector, alpha)


def create_global_hooks(
    steering_vector: Any,
    alpha: float = 1.0,
    layers: List[int] = None,
) -> List[Tuple[str, Callable]]:
    """
    Create hooks for global steering (all layers or specified).

    Args:
        steering_vector: Steering vector
        alpha: Steering strength
        layers: Specific layers (default: all)
    """
    if layers is None:
        circuit = get_circuit(CircuitType.GLOBAL)
    else:
        circuit = Circuit(
            circuit_type=CircuitType.GLOBAL,
            layers=layers,
            description="Custom global",
        )
    return create_circuit_hooks(circuit, steering_vector, alpha)


# =============================================================================
# COMBINED CIRCUIT HOOKS (TRANSITION + HIERARCHY)
# =============================================================================

def create_combined_hooks(
    steering_vector: Any,
    transition_alpha: float = 1.0,
    hierarchy_alpha: float = 1.0,
    transition_vector: Any = None,
    hierarchy_vector: Any = None,
) -> List[Tuple[str, Callable]]:
    """
    Create combined hooks for TRANSITION + HIERARCHY circuits.

    This is the main fusion function for the experiment.

    Args:
        steering_vector: Default vector if circuit-specific not provided
        transition_alpha: Alpha for TRANSITION circuit (L8-14)
        hierarchy_alpha: Alpha for HIERARCHY circuit (L0-6)
        transition_vector: Optional specific vector for TRANSITION
        hierarchy_vector: Optional specific vector for HIERARCHY

    Returns:
        Combined hooks for both circuits
    """
    t_vec = transition_vector if transition_vector is not None else steering_vector
    h_vec = hierarchy_vector if hierarchy_vector is not None else steering_vector

    transition_hooks = create_transition_hooks(t_vec, transition_alpha)
    hierarchy_hooks = create_hierarchy_hooks(h_vec, hierarchy_alpha)

    return hierarchy_hooks + transition_hooks  # Hierarchy first (lower layers)


# =============================================================================
# TESTING
# =============================================================================

def test_circuit_hooks():
    """Test circuit hook creation."""
    print("=" * 60)
    print("CIRCUIT HOOKS TEST")
    print("=" * 60)

    print(f"\nMLX available: {MLX_AVAILABLE}")

    # Show circuits
    print("\nDefined circuits:")
    for ct, circuit in CIRCUITS.items():
        print(f"  {ct.value}: {circuit}")

    if not MLX_AVAILABLE:
        print("\nSkipping hook tests (MLX not available)")
        return True

    # Create steering vector
    print("\nCreating steering vector...")
    vec = mx.random.normal((896,))
    print(f"Vector shape: {vec.shape}")

    # Test single circuit hooks
    print("\nCreating TRANSITION hooks (L8-14)...")
    t_hooks = create_transition_hooks(vec, alpha=1.0)
    print(f"Created {len(t_hooks)} hooks:")
    for name, _ in t_hooks[:3]:
        print(f"  {name}")

    print("\nCreating HIERARCHY hooks (L0-6)...")
    h_hooks = create_hierarchy_hooks(vec, alpha=1.0)
    print(f"Created {len(h_hooks)} hooks:")
    for name, _ in h_hooks[:3]:
        print(f"  {name}")

    # Test combined hooks
    print("\nCreating COMBINED hooks...")
    combined = create_combined_hooks(vec, transition_alpha=1.0, hierarchy_alpha=0.5)
    print(f"Created {len(combined)} total hooks")

    # Test hook with mock data
    print("\nTesting hook with mock data...")
    mock_output = mx.random.normal((1, 10, 896))
    hook = create_circuit_hook(vec, alpha=1.0)
    modified = hook((), mock_output, None)
    print(f"Input shape: {mock_output.shape}")
    print(f"Output shape: {modified.shape}")

    # Verify steering applied
    diff = mx.max(mx.abs(modified[:, -1, :] - mock_output[:, -1, :]))
    print(f"Max diff at last position: {diff.item():.4f}")

    print("\n[PASS] Circuit hooks test complete")
    return True


if __name__ == "__main__":
    test_circuit_hooks()
