#!/usr/bin/env python3
"""
Targeted Steering: Apply steering vectors to specific heads for SC generation.

Combines head amplification with contrastive steering to guide
statechart generation toward valid, well-structured outputs.
"""

import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False

from utils.mlux_loader import MLUX_AVAILABLE

from .head_amplifier import (
    HeadAmplifier,
    AmplificationConfig,
    HeadConfig,
    STRUCTURE_HEADS,
    HIERARCHY_HEADS,
)


@dataclass
class SteeringVector:
    """A steering vector for a specific layer."""
    layer: int
    vector: Any  # mlx array
    alpha: float = 1.0  # Steering strength


@dataclass
class TargetedSteeringConfig:
    """Configuration for targeted steering."""
    # Head amplification
    amplify_structure: bool = True
    amplify_hierarchy: bool = True
    structure_scale: float = 1.3
    hierarchy_scale: float = 1.5

    # Contrastive steering
    use_contrastive: bool = True
    positive_prompt: str = "Well-structured hierarchical statechart with clear parent-child relationships"
    negative_prompt: str = "Flat unstructured state list without proper nesting"
    steering_layer: int = 12  # Middle layer for steering
    steering_alpha: float = 0.5


@dataclass
class SteeringResult:
    """Result of steered generation."""
    output: str
    config: TargetedSteeringConfig
    valid_json: bool = False
    has_hierarchy: bool = False
    num_states: int = 0
    num_nested: int = 0  # States with children
    metadata: Dict[str, Any] = field(default_factory=dict)


class TargetedSteering:
    """
    Applies targeted steering using discovered heads.

    Combines:
    1. Head amplification (structure + hierarchy heads)
    2. Contrastive steering vectors
    3. Validation feedback
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("TARGETED STEERING")
        print("=" * 60)

        self.amplifier = HeadAmplifier(model_name)
        self.model = self.amplifier.model
        self._steering_cache: Dict[str, SteeringVector] = {}

        print("=" * 60)

    def compute_steering_vector(
        self,
        positive: str,
        negative: str,
        layer: int,
    ) -> Optional[SteeringVector]:
        """
        Compute contrastive steering vector.

        Args:
            positive: Example of desired output style
            negative: Example of undesired output style
            layer: Layer to extract vector from

        Returns:
            SteeringVector or None if unavailable
        """
        cache_key = f"{positive[:20]}_{negative[:20]}_{layer}"
        if cache_key in self._steering_cache:
            return self._steering_cache[cache_key]

        if not MLUX_AVAILABLE or self.model is None:
            return None

        try:
            # Get activations for positive and negative
            _, pos_cache = self.model.run_with_cache(
                positive,
                hooks=[f"model.layers.{layer}"],
            )
            _, neg_cache = self.model.run_with_cache(
                negative,
                hooks=[f"model.layers.{layer}"],
            )

            pos_act = pos_cache.get(f"model.layers.{layer}")
            neg_act = neg_cache.get(f"model.layers.{layer}")

            if pos_act is None or neg_act is None:
                return None

            # Compute difference (steering direction)
            # Use mean across sequence positions
            pos_mean = mx.mean(pos_act, axis=1, keepdims=True)
            neg_mean = mx.mean(neg_act, axis=1, keepdims=True)
            steering = pos_mean - neg_mean

            vec = SteeringVector(layer=layer, vector=steering, alpha=1.0)
            self._steering_cache[cache_key] = vec
            return vec

        except Exception as e:
            print(f"Steering vector computation failed: {e}")
            return None

    def generate_with_steering(
        self,
        prompt: str,
        config: Optional[TargetedSteeringConfig] = None,
        max_tokens: int = 300,
    ) -> SteeringResult:
        """
        Generate with targeted steering.

        Args:
            prompt: Input prompt
            config: Steering configuration
            max_tokens: Max tokens to generate

        Returns:
            SteeringResult with output and analysis
        """
        config = config or TargetedSteeringConfig()

        if not MLUX_AVAILABLE or self.model is None:
            return self._mock_generate(prompt, config)

        # Build amplification config
        amp_heads = []
        if config.amplify_structure:
            for l, h in STRUCTURE_HEADS:
                amp_heads.append(HeadConfig(l, h, config.structure_scale, f"struct_L{l}H{h}"))
        if config.amplify_hierarchy:
            for l, h in HIERARCHY_HEADS:
                amp_heads.append(HeadConfig(l, h, config.hierarchy_scale, f"hier_L{l}H{h}"))

        amp_config = AmplificationConfig(heads=amp_heads)

        # Compute steering vector if needed
        steering_vec = None
        if config.use_contrastive:
            steering_vec = self.compute_steering_vector(
                config.positive_prompt,
                config.negative_prompt,
                config.steering_layer,
            )

        # Generate with amplification
        output, meta = self.amplifier.generate_with_amplification(
            prompt, amp_config, max_tokens
        )

        # Analyze output
        result = self._analyze_output(output, config, meta)

        return result

    def _analyze_output(
        self,
        output: str,
        config: TargetedSteeringConfig,
        metadata: Dict[str, Any],
    ) -> SteeringResult:
        """Analyze generated output for validity and structure."""
        result = SteeringResult(
            output=output,
            config=config,
            metadata=metadata,
        )

        # Try to parse JSON
        try:
            # Extract JSON from output
            import re
            text = re.sub(r'```json\s*', '', output)
            text = re.sub(r'```\s*', '', text)

            start = text.find('{')
            if start == -1:
                return result

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

            data = json.loads(text[start:end])
            result.valid_json = True

            # Analyze structure
            def count_states(state: Dict, depth: int = 0) -> Tuple[int, int]:
                """Count total states and nested states."""
                total = 1
                nested = 0
                children = state.get('children', [])
                if children:
                    nested = 1
                    for child in children:
                        c_total, c_nested = count_states(child, depth + 1)
                        total += c_total
                        nested += c_nested
                return total, nested

            root = data.get('root_state', {})
            if root:
                result.num_states, result.num_nested = count_states(root)
                result.has_hierarchy = result.num_nested > 0

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return result

    def _mock_generate(
        self,
        prompt: str,
        config: TargetedSteeringConfig,
    ) -> SteeringResult:
        """Mock generation for testing."""
        mock_output = f'''[MOCK] Steered generation
Config: structure_scale={config.structure_scale}, hierarchy_scale={config.hierarchy_scale}
Prompt: {prompt[:50]}...'''

        return SteeringResult(
            output=mock_output,
            config=config,
            metadata={"mock": True},
        )

    def compare_steering_configs(
        self,
        prompt: str,
        configs: Dict[str, TargetedSteeringConfig],
    ) -> Dict[str, SteeringResult]:
        """
        Compare different steering configurations.

        Args:
            prompt: Input prompt
            configs: Named configurations to compare

        Returns:
            {config_name: SteeringResult}
        """
        results = {}

        # Baseline (no steering)
        baseline_config = TargetedSteeringConfig(
            amplify_structure=False,
            amplify_hierarchy=False,
            use_contrastive=False,
        )
        results["baseline"] = self.generate_with_steering(prompt, baseline_config)

        # Each configuration
        for name, config in configs.items():
            results[name] = self.generate_with_steering(prompt, config)

        return results


# Preset configurations
CONFIGS = {
    "hierarchy_boost": TargetedSteeringConfig(
        amplify_structure=False,
        amplify_hierarchy=True,
        hierarchy_scale=2.0,
        use_contrastive=False,
    ),
    "structure_boost": TargetedSteeringConfig(
        amplify_structure=True,
        amplify_hierarchy=False,
        structure_scale=2.0,
        use_contrastive=False,
    ),
    "full_steering": TargetedSteeringConfig(
        amplify_structure=True,
        amplify_hierarchy=True,
        structure_scale=1.5,
        hierarchy_scale=1.5,
        use_contrastive=True,
    ),
    "aggressive_hierarchy": TargetedSteeringConfig(
        amplify_structure=True,
        amplify_hierarchy=True,
        structure_scale=1.3,
        hierarchy_scale=2.5,
        use_contrastive=True,
        steering_alpha=0.8,
    ),
}


def demo():
    """Demo targeted steering."""
    print("=" * 60)
    print("TARGETED STEERING DEMO")
    print("=" * 60)

    steering = TargetedSteering()

    prompt = '''Generate a hierarchical statechart for a media player with nested play states:
```json
{"root_state": {"label": "__root__", "type": 2, "children": ['''

    print(f"\nPrompt: {prompt[:70]}...")

    # Compare configurations
    results = steering.compare_steering_configs(prompt, CONFIGS)

    print("\n--- Results ---")
    for name, result in results.items():
        print(f"\n[{name}]")
        print(f"  Valid JSON: {result.valid_json}")
        print(f"  Has hierarchy: {result.has_hierarchy}")
        print(f"  States: {result.num_states}, Nested: {result.num_nested}")
        print(f"  Output: {result.output[:80]}...")

    return results


if __name__ == "__main__":
    demo()
