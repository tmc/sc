#!/usr/bin/env python3
"""
Baseline Demo: mlux integration patterns for statecharts.

Demonstrates:
1. Loading model with HookedModelWrapper
2. Generating statecharts with activation caching
3. Basic steering vector computation
4. Fallback behavior when mlux unavailable
"""

import json
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

# Add utils to path
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    CacheConfig,
    MLUX_AVAILABLE,
    MLX_LM_AVAILABLE,
)


@dataclass
class StatechartPrompt:
    """Prompt for statechart generation."""
    description: str
    num_states: int = 3
    include_transitions: bool = True

    def to_prompt(self) -> str:
        """Convert to LLM prompt."""
        return f"""Generate a statechart JSON for: {self.description}

Requirements:
- {self.num_states} states
- Include transitions between states
- Use this exact format:

```json
{{
  "root_state": {{
    "label": "__root__",
    "type": 2,
    "children": [
      {{"label": "State1", "is_initial": true}},
      {{"label": "State2"}},
      ...
    ]
  }},
  "transitions": [
    {{"from": ["State1"], "to": ["State2"], "event": "EVENT_NAME"}}
  ]
}}
```

Output only the JSON, no explanation:
```json
"""


@dataclass
class GenerationResult:
    """Result of statechart generation."""
    prompt: str
    output: str
    cache: Dict[str, Any]
    statechart: Optional[Dict[str, Any]] = None
    valid: bool = False
    generation_time: float = 0.0
    backend_used: str = ""


class StatechartGenerator:
    """
    Generates statecharts using mlux/mlx_lm.

    Shows integration patterns with activation caching.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        backend: Optional[ModelBackend] = None,
    ):
        """Initialize with model."""
        print("=" * 60)
        print("STATECHART GENERATOR INITIALIZATION")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"MLUX available: {MLUX_AVAILABLE}")
        print(f"MLX_LM available: {MLX_LM_AVAILABLE}")
        print()

        self.model = load_model(model_name, backend=backend)
        print(f"Backend selected: {self.model.backend.name}")
        print(f"Has interpretability: {self.model.has_interpretability}")
        print("=" * 60)

    def generate(
        self,
        prompt: StatechartPrompt,
        with_cache: bool = True,
    ) -> GenerationResult:
        """
        Generate statechart from prompt.

        Args:
            prompt: Statechart description
            with_cache: Whether to capture activations

        Returns:
            GenerationResult with output and optional cache
        """
        start_time = time.time()
        prompt_text = prompt.to_prompt()

        config = GenerationConfig(
            max_tokens=512,
            temperature=0.3,
        )

        if with_cache and self.model.has_interpretability:
            cache_config = CacheConfig(
                hooks=["model.layers.*.mlp", "model.layers.*.attention"],
                include_attention=True,
            )
            output, cache = self.model.generate_with_cache(
                prompt_text, config, cache_config
            )
        else:
            output = self.model.generate(prompt_text, config)
            cache = {}

        # Parse output
        statechart = self._extract_json(output)
        valid = statechart is not None

        return GenerationResult(
            prompt=prompt_text,
            output=output,
            cache=cache,
            statechart=statechart,
            valid=valid,
            generation_time=time.time() - start_time,
            backend_used=self.model.backend.name,
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from generated text."""
        import re

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Find JSON object
        start = text.find('{')
        if start == -1:
            return None

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

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def generate_with_steering(
        self,
        prompt: StatechartPrompt,
        positive_example: str,
        negative_example: str,
        layer: int = 12,
        alpha: float = 1.0,
    ) -> GenerationResult:
        """
        Generate with steering vector for style control.

        Args:
            prompt: Statechart description
            positive_example: Example of desired style
            negative_example: Example of undesired style
            layer: Layer to apply steering
            alpha: Steering strength

        Returns:
            GenerationResult
        """
        start_time = time.time()
        prompt_text = prompt.to_prompt()

        config = GenerationConfig(
            max_tokens=512,
            temperature=0.3,
        )

        if not self.model.has_interpretability:
            # Fallback to normal generation
            output = self.model.generate(prompt_text, config)
            return GenerationResult(
                prompt=prompt_text,
                output=output,
                cache={},
                statechart=self._extract_json(output),
                valid=self._extract_json(output) is not None,
                generation_time=time.time() - start_time,
                backend_used=f"{self.model.backend.name} (no steering)",
            )

        # Compute steering vector
        steering_vec = self.model.compute_steering_vector(
            positive=positive_example,
            negative=negative_example,
            layer=layer,
        )

        # Generate with steering
        output = self.model.generate_with_steering(
            prompt_text,
            steering_vector=steering_vec,
            layer=layer,
            alpha=alpha,
            config=config,
        )

        statechart = self._extract_json(output)

        return GenerationResult(
            prompt=prompt_text,
            output=output,
            cache={"steering_vector_shape": str(steering_vec.shape) if hasattr(steering_vec, 'shape') else "N/A"},
            statechart=statechart,
            valid=statechart is not None,
            generation_time=time.time() - start_time,
            backend_used=f"{self.model.backend.name} (steered)",
        )


def demo_basic_generation():
    """Demo 1: Basic generation with HookedModelWrapper."""
    print("\n" + "=" * 60)
    print("DEMO 1: BASIC GENERATION")
    print("=" * 60)

    generator = StatechartGenerator()

    prompt = StatechartPrompt(
        description="A traffic light controller",
        num_states=3,
    )

    result = generator.generate(prompt, with_cache=False)

    print(f"\nPrompt: {prompt.description}")
    print(f"Backend: {result.backend_used}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Valid JSON: {result.valid}")

    if result.valid:
        print(f"\nGenerated statechart:")
        print(json.dumps(result.statechart, indent=2)[:500])

    return result


def demo_with_cache():
    """Demo 2: Generation with activation caching."""
    print("\n" + "=" * 60)
    print("DEMO 2: GENERATION WITH ACTIVATION CACHE")
    print("=" * 60)

    generator = StatechartGenerator()

    prompt = StatechartPrompt(
        description="A user authentication flow",
        num_states=4,
    )

    result = generator.generate(prompt, with_cache=True)

    print(f"\nPrompt: {prompt.description}")
    print(f"Backend: {result.backend_used}")
    print(f"Has interpretability: {generator.model.has_interpretability}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Valid JSON: {result.valid}")

    if result.cache:
        print(f"\nCache keys: {list(result.cache.keys())[:5]}")
        # Show cache shape info if available
        for key in list(result.cache.keys())[:3]:
            val = result.cache[key]
            if hasattr(val, 'shape'):
                print(f"  {key}: shape={val.shape}")

    if result.valid:
        print(f"\nGenerated statechart has {len(result.statechart.get('root_state', {}).get('children', []))} states")

    return result


def demo_steering():
    """Demo 3: Generation with steering vectors."""
    print("\n" + "=" * 60)
    print("DEMO 3: STEERING VECTOR DEMO")
    print("=" * 60)

    generator = StatechartGenerator()

    prompt = StatechartPrompt(
        description="An order processing workflow",
        num_states=5,
    )

    # Define contrastive examples for style
    positive = "Clean, minimal statechart with clear transitions"
    negative = "Complex, nested statechart with many conditions"

    print(f"\nPositive style: {positive}")
    print(f"Negative style: {negative}")

    result = generator.generate_with_steering(
        prompt,
        positive_example=positive,
        negative_example=negative,
        layer=12,
        alpha=0.5,
    )

    print(f"\nPrompt: {prompt.description}")
    print(f"Backend: {result.backend_used}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Valid JSON: {result.valid}")

    if result.valid:
        num_states = len(result.statechart.get('root_state', {}).get('children', []))
        num_trans = len(result.statechart.get('transitions', []))
        print(f"\nGenerated: {num_states} states, {num_trans} transitions")

    return result


def demo_fallback_behavior():
    """Demo 4: Fallback when mlux unavailable."""
    print("\n" + "=" * 60)
    print("DEMO 4: FALLBACK BEHAVIOR")
    print("=" * 60)

    # Force MLX_LM backend even if mlux available
    generator = StatechartGenerator(
        backend=ModelBackend.MLX_LM if MLX_LM_AVAILABLE else ModelBackend.MOCK
    )

    prompt = StatechartPrompt(
        description="A simple toggle switch",
        num_states=2,
    )

    # Try to use cache (should return empty)
    result = generator.generate(prompt, with_cache=True)

    print(f"\nForced backend: {result.backend_used}")
    print(f"Has interpretability: {generator.model.has_interpretability}")
    print(f"Cache available: {bool(result.cache)}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Valid JSON: {result.valid}")

    return result


def run_all_demos():
    """Run all demos."""
    print("=" * 60)
    print("MLUX BASELINE DEMOS")
    print("=" * 60)
    print(f"\nBackend availability:")
    print(f"  MLUX: {MLUX_AVAILABLE}")
    print(f"  MLX_LM: {MLX_LM_AVAILABLE}")

    results = []

    # Demo 1: Basic generation
    results.append(("basic", demo_basic_generation()))

    # Demo 2: With cache
    results.append(("cache", demo_with_cache()))

    # Demo 3: Steering (may fall back)
    results.append(("steering", demo_steering()))

    # Demo 4: Fallback behavior
    results.append(("fallback", demo_fallback_behavior()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "OK" if result.valid else "FAIL"
        print(f"  [{status}] {name}: {result.backend_used}, {result.generation_time:.2f}s")

    return results


if __name__ == "__main__":
    run_all_demos()
