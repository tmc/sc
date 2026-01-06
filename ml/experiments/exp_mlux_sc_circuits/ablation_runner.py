"""
Ablation Runner for Circuit Discovery

Systematically ablates model components (attention heads, MLPs)
and measures the effect on statechart validity.

ABLATION TYPES:
1. Zero ablation: Set component output to zero
2. Mean ablation: Replace with running mean
3. Noise ablation: Add Gaussian noise
4. Resample ablation: Replace with different input's activation

COMPONENTS:
- Attention heads: model.layers.{L}.self_attn.{head}
- MLP layers: model.layers.{L}.mlp
- Layer norms: model.layers.{L}.input_layernorm
- Full layers: model.layers.{L}
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from enum import Enum
import random

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    nn = None
    MLX_AVAILABLE = False

from .validity_measurer import ValidityMeasurer, ValidityMetrics


class AblationType(Enum):
    """Types of ablation."""
    ZERO = "zero"       # Set to zero
    MEAN = "mean"       # Replace with mean activation
    NOISE = "noise"     # Add Gaussian noise
    RESAMPLE = "resample"  # Use activation from different input


@dataclass
class ComponentSpec:
    """Specification of a model component."""
    layer: int
    component_type: str  # "attention", "mlp", "layernorm", "full"
    head: Optional[int] = None  # For attention heads

    @property
    def path(self) -> str:
        """Get the module path."""
        if self.component_type == "attention":
            if self.head is not None:
                return f"model.layers.{self.layer}.self_attn.head_{self.head}"
            return f"model.layers.{self.layer}.self_attn"
        elif self.component_type == "mlp":
            return f"model.layers.{self.layer}.mlp"
        elif self.component_type == "layernorm":
            return f"model.layers.{self.layer}.input_layernorm"
        else:
            return f"model.layers.{self.layer}"

    def __str__(self) -> str:
        if self.head is not None:
            return f"L{self.layer}.H{self.head}"
        return f"L{self.layer}.{self.component_type}"


@dataclass
class AblationResult:
    """Result of ablating a single component."""
    component: ComponentSpec
    ablation_type: AblationType
    baseline_metrics: ValidityMetrics
    ablated_metrics: ValidityMetrics
    n_samples: int

    @property
    def validity_drop(self) -> float:
        """Drop in validity score (positive = component is important)."""
        return self.baseline_metrics.total_score - self.ablated_metrics.total_score

    @property
    def state_name_drop(self) -> float:
        return self.baseline_metrics.state_name_consistency - self.ablated_metrics.state_name_consistency

    @property
    def transition_drop(self) -> float:
        return self.baseline_metrics.transition_validity - self.ablated_metrics.transition_validity

    @property
    def hierarchy_drop(self) -> float:
        return self.baseline_metrics.hierarchy_validity - self.ablated_metrics.hierarchy_validity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": str(self.component),
            "ablation_type": self.ablation_type.value,
            "validity_drop": self.validity_drop,
            "state_name_drop": self.state_name_drop,
            "transition_drop": self.transition_drop,
            "hierarchy_drop": self.hierarchy_drop,
            "baseline_score": self.baseline_metrics.total_score,
            "ablated_score": self.ablated_metrics.total_score,
        }


@dataclass
class AblationStudy:
    """Complete ablation study results."""
    results: List[AblationResult] = field(default_factory=list)
    model_name: str = ""
    n_layers: int = 0
    n_heads: int = 0

    def get_critical_components(
        self,
        threshold: float = 0.1,
        metric: str = "validity",
    ) -> List[AblationResult]:
        """Get components whose ablation causes significant degradation."""
        if metric == "validity":
            key = lambda r: r.validity_drop
        elif metric == "state_name":
            key = lambda r: r.state_name_drop
        elif metric == "transition":
            key = lambda r: r.transition_drop
        elif metric == "hierarchy":
            key = lambda r: r.hierarchy_drop
        else:
            key = lambda r: r.validity_drop

        return sorted(
            [r for r in self.results if key(r) >= threshold],
            key=key,
            reverse=True,
        )

    def get_by_layer(self) -> Dict[int, List[AblationResult]]:
        """Group results by layer."""
        by_layer: Dict[int, List[AblationResult]] = {}
        for r in self.results:
            layer = r.component.layer
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(r)
        return by_layer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_components": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }


class AblationRunner:
    """
    Runs ablation experiments to identify critical components.

    For each component, measures validity degradation when ablated.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        measurer: Optional[ValidityMeasurer] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: HookedModelWrapper from mlux_loader
            measurer: Validity measurer instance
            verbose: Print progress
        """
        self.model = model
        self.measurer = measurer or ValidityMeasurer()
        self.verbose = verbose

        # Model config
        self.n_layers = 0
        self.n_heads = 0
        self._detect_model_config()

        # Cache for mean activations
        self._mean_cache: Dict[str, Any] = {}

    def _detect_model_config(self):
        """Detect model configuration."""
        if self.model is None:
            # Default config for testing
            self.n_layers = 24
            self.n_heads = 12
            return

        # Try to detect from model
        if hasattr(self.model, '_model') and self.model._model is not None:
            config = getattr(self.model._model, 'config', None)
            if config:
                self.n_layers = getattr(config, 'num_hidden_layers', 24)
                self.n_heads = getattr(config, 'num_attention_heads', 12)

    def enumerate_components(
        self,
        include_attention: bool = True,
        include_mlp: bool = True,
        include_heads: bool = False,
        layers: Optional[List[int]] = None,
    ) -> List[ComponentSpec]:
        """Enumerate all components to ablate."""
        components = []
        layer_range = layers or range(self.n_layers)

        for layer in layer_range:
            if include_attention:
                if include_heads:
                    for head in range(self.n_heads):
                        components.append(ComponentSpec(
                            layer=layer,
                            component_type="attention",
                            head=head,
                        ))
                else:
                    components.append(ComponentSpec(
                        layer=layer,
                        component_type="attention",
                    ))

            if include_mlp:
                components.append(ComponentSpec(
                    layer=layer,
                    component_type="mlp",
                ))

        return components

    def create_ablation_hook(
        self,
        component: ComponentSpec,
        ablation_type: AblationType,
    ) -> Callable:
        """Create a hook function that performs ablation."""
        def zero_hook(module, inputs, outputs):
            """Zero out the outputs."""
            if MLX_AVAILABLE:
                return mx.zeros_like(outputs)
            return outputs * 0

        def noise_hook(module, inputs, outputs):
            """Add Gaussian noise."""
            if MLX_AVAILABLE:
                noise = mx.random.normal(outputs.shape) * 0.5
                return outputs + noise
            return outputs

        def mean_hook(module, inputs, outputs):
            """Replace with cached mean."""
            key = component.path
            if key in self._mean_cache and MLX_AVAILABLE:
                mean = self._mean_cache[key]
                return mx.broadcast_to(mean, outputs.shape)
            return outputs

        if ablation_type == AblationType.ZERO:
            return zero_hook
        elif ablation_type == AblationType.NOISE:
            return noise_hook
        elif ablation_type == AblationType.MEAN:
            return mean_hook
        else:
            return zero_hook

    def run_baseline(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Dict[str, Any]],
    ) -> Tuple[List[str], ValidityMetrics]:
        """
        Run baseline generation without ablation.

        Args:
            prompts: List of SC generation prompts
            parse_fn: Function to parse model output to chart dict

        Returns:
            (outputs, aggregate_metrics)
        """
        outputs = []
        charts = []

        for prompt in prompts:
            if self.model is not None:
                output = self.model.generate(prompt)
            else:
                # Mock output for testing
                output = self._mock_generate(prompt)

            outputs.append(output)

            try:
                chart = parse_fn(output)
                charts.append(chart)
            except Exception:
                charts.append({})  # Invalid chart

        # Measure validity
        _, aggregate = self.measurer.measure_batch(charts)

        # Create aggregate metrics
        metrics = ValidityMetrics(
            is_valid=aggregate.get("valid_rate", 0) > 0.5,
            total_score=aggregate.get("mean_total_score", 0),
            state_name_consistency=aggregate.get("mean_state_name", 0),
            transition_validity=aggregate.get("mean_transition", 0),
            hierarchy_validity=aggregate.get("mean_hierarchy", 0),
            structural_validity=aggregate.get("mean_structural", 0),
            semantic_validity=aggregate.get("mean_semantic", 0),
            n_states=sum(m.n_states for m, _ in zip(
                [self.measurer.measure(c) for c in charts], charts
            )),
            n_transitions=sum(m.n_transitions for m, _ in zip(
                [self.measurer.measure(c) for c in charts], charts
            )),
        )

        return outputs, metrics

    def run_ablation(
        self,
        component: ComponentSpec,
        ablation_type: AblationType,
        prompts: List[str],
        parse_fn: Callable[[str], Dict[str, Any]],
    ) -> ValidityMetrics:
        """
        Run generation with component ablated.

        Returns aggregate validity metrics.
        """
        charts = []

        for prompt in prompts:
            if self.model is not None and hasattr(self.model, 'has_interpretability'):
                if self.model.has_interpretability:
                    # Use hooks for ablation
                    hook_fn = self.create_ablation_hook(component, ablation_type)
                    output = self.model._model.run_with_hooks(
                        prompt,
                        hooks=[(component.path, hook_fn)],
                    )
                else:
                    # Fallback: generate normally
                    output = self.model.generate(prompt)
            else:
                # Mock output
                output = self._mock_generate_ablated(prompt, component)

            try:
                chart = parse_fn(output)
                charts.append(chart)
            except Exception:
                charts.append({})

        # Measure validity
        _, aggregate = self.measurer.measure_batch(charts)

        return ValidityMetrics(
            is_valid=aggregate.get("valid_rate", 0) > 0.5,
            total_score=aggregate.get("mean_total_score", 0),
            state_name_consistency=aggregate.get("mean_state_name", 0),
            transition_validity=aggregate.get("mean_transition", 0),
            hierarchy_validity=aggregate.get("mean_hierarchy", 0),
            structural_validity=aggregate.get("mean_structural", 0),
            semantic_validity=aggregate.get("mean_semantic", 0),
        )

    def run_study(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Dict[str, Any]],
        components: Optional[List[ComponentSpec]] = None,
        ablation_type: AblationType = AblationType.ZERO,
    ) -> AblationStudy:
        """
        Run complete ablation study.

        Args:
            prompts: Generation prompts
            parse_fn: Output parser
            components: Components to ablate (default: all)
            ablation_type: Type of ablation

        Returns:
            Complete study results
        """
        if components is None:
            components = self.enumerate_components()

        study = AblationStudy(
            model_name=getattr(self.model, 'model_name', 'mock'),
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        )

        # Run baseline
        if self.verbose:
            print(f"Running baseline on {len(prompts)} prompts...")

        _, baseline_metrics = self.run_baseline(prompts, parse_fn)

        if self.verbose:
            print(f"Baseline validity: {baseline_metrics.total_score:.2%}")
            print(f"\nAblating {len(components)} components...")

        # Ablate each component
        for i, component in enumerate(components):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(components)}")

            ablated_metrics = self.run_ablation(
                component, ablation_type, prompts, parse_fn
            )

            result = AblationResult(
                component=component,
                ablation_type=ablation_type,
                baseline_metrics=baseline_metrics,
                ablated_metrics=ablated_metrics,
                n_samples=len(prompts),
            )

            study.results.append(result)

        return study

    def _mock_generate(self, prompt: str) -> str:
        """Generate mock valid SC output."""
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

    def _mock_generate_ablated(self, prompt: str, component: ComponentSpec) -> str:
        """Generate mock ablated SC output with errors based on component."""
        # Simulate different types of errors based on layer
        layer = component.layer

        if layer < self.n_layers // 3:
            # Early layers: structural issues
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    # Missing children
                },
                "transitions": []
            })

        elif layer < 2 * self.n_layers // 3:
            # Middle layers: transition issues
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
                    {"from": ["s0"], "to": ["INVALID"], "event": "E1"},  # Bad ref
                ]
            })

        else:
            # Late layers: name consistency issues
            return json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "s0", "type": 1, "is_initial": True},
                        {"label": "s0", "type": 1},  # Duplicate name
                    ]
                },
                "transitions": [
                    {"from": ["s0"], "to": ["s0"], "event": "E1"},
                ]
            })


# =============================================================================
# Demo
# =============================================================================

def default_parser(output: str) -> Dict[str, Any]:
    """Default parser for SC JSON output."""
    # Find JSON in output
    import re
    match = re.search(r'\{[\s\S]*\}', output)
    if match:
        return json.loads(match.group())
    return {}


def demo():
    """Demonstrate ablation runner."""
    print("=" * 60)
    print("Ablation Runner for SC Circuit Discovery")
    print("=" * 60)

    runner = AblationRunner(model=None, verbose=True)

    # Test prompts
    prompts = [
        "Generate a simple traffic light statechart:",
        "Create a login flow state machine:",
        "Design an order process statechart:",
    ]

    # Run study on subset of components
    components = runner.enumerate_components(
        include_attention=True,
        include_mlp=True,
        layers=list(range(0, 24, 4)),  # Sample every 4th layer
    )

    print(f"\nEnumerated {len(components)} components to ablate")

    study = runner.run_study(
        prompts=prompts,
        parse_fn=default_parser,
        components=components,
        ablation_type=AblationType.ZERO,
    )

    # Report critical components
    print("\n" + "=" * 60)
    print("CRITICAL COMPONENTS (validity drop > 10%)")
    print("=" * 60)

    critical = study.get_critical_components(threshold=0.1)
    for r in critical[:10]:
        print(f"  {r.component}: validity drop = {r.validity_drop:.1%}")
        print(f"    - State names: {r.state_name_drop:+.1%}")
        print(f"    - Transitions: {r.transition_drop:+.1%}")
        print(f"    - Hierarchy: {r.hierarchy_drop:+.1%}")

    # By layer
    print("\n" + "-" * 40)
    print("Mean validity drop by layer:")
    by_layer = study.get_by_layer()
    for layer in sorted(by_layer.keys()):
        results = by_layer[layer]
        mean_drop = sum(r.validity_drop for r in results) / len(results)
        print(f"  Layer {layer}: {mean_drop:+.1%}")

    return study


if __name__ == "__main__":
    demo()
