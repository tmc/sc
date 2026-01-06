"""
Model Scaling Experiment: 7B vs 0.5B Steering Comparison

Tests if steering improvements scale with model size.

BASELINE (0.5B):
- Baseline validity: 80%
- hierarchy_boost (1.5x L23H1): 80% valid, 8 states, 4 nested
- Over-amplification (>2.0x) hurts validity

MODEL: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
- 32 layers (vs 24 in 0.5B)
- Layer mapping: Structure L15-20, Hierarchy L28-31

HYPOTHESIS: Larger model = higher baseline + larger steering delta
"""

import json
import time
import gc
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
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


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_7B = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MODEL_0_5B = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Layer mapping for 7B (32 layers)
# Structure heads likely in L15-20 range (mid-layers)
# Hierarchy head likely in final layers (L28-31)
LAYER_MAP_7B = {
    "structure": list(range(15, 21)),  # L15-20
    "hierarchy": list(range(28, 32)),   # L28-31
}

# For 0.5B (24 layers) - from baseline experiments
LAYER_MAP_0_5B = {
    "structure": list(range(10, 16)),  # L10-15
    "hierarchy": [23],  # L23H1
}


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment."""
    model_name: str = MODEL_7B
    max_tokens: int = 256
    temperature: float = 0.3
    num_samples: int = 10  # Per config
    boost_factor: float = 1.5


@dataclass
class GenerationResult:
    """Result of a single generation."""
    prompt: str
    output: str
    is_valid: bool
    error: str
    num_states: int
    hierarchy_depth: int
    generation_time: float


@dataclass
class ConfigResult:
    """Result for a configuration."""
    config_name: str
    results: List[GenerationResult]
    validity_rate: float
    avg_states: float
    avg_depth: float
    avg_time: float


# =============================================================================
# VALIDATION
# =============================================================================

def validate_statechart(text: str) -> Tuple[bool, str, int, int]:
    """
    Validate and analyze statechart.

    Returns:
        (is_valid, error, num_states, hierarchy_depth)
    """
    import re

    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')

    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    start = text.find('{')
    if start == -1:
        return False, "No JSON found", 0, 0

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
        sc = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        return False, f"JSON error: {e}", 0, 0

    if "root_state" not in sc:
        return False, "Missing root_state", 0, 0

    root = sc["root_state"]
    children = root.get("children", [])

    if len(children) == 0:
        return False, "Empty children", 0, 0

    has_initial = any(c.get("is_initial", False) for c in children)
    if not has_initial:
        return False, "No initial state", 0, 0

    # Count states and depth
    def count_states_depth(state, current_depth=0):
        count = 1
        max_depth = current_depth
        for child in state.get("children", []):
            c, d = count_states_depth(child, current_depth + 1)
            count += c
            max_depth = max(max_depth, d)
        return count, max_depth

    total_states, max_depth = count_states_depth(root)

    return True, "Valid", total_states, max_depth


# =============================================================================
# PROMPTS
# =============================================================================

SC_PROMPTS = [
    "Generate a statechart for a traffic light with 3 states",
    "Create a hierarchical statechart for user authentication",
    "Build a statechart for an order processing workflow",
    "Generate a statechart with nested states for a game character",
    "Create a state machine for a vending machine",
    "Build a statechart for a document approval process",
    "Generate a hierarchical statechart for a media player",
    "Create a statechart for a thermostat controller",
    "Build a state machine for a login flow with retry logic",
    "Generate a statechart for an elevator controller",
]

SC_TEMPLATE = """Generate a statechart JSON for: {description}

Use this exact format:
```json
{{
  "root_state": {{
    "label": "__root__",
    "type": 2,
    "children": [
      {{"label": "State1", "type": 1, "is_initial": true}},
      {{"label": "State2", "type": 1}}
    ]
  }},
  "transitions": [
    {{"from": ["State1"], "to": ["State2"], "event": "EVENT"}}
  ]
}}
```

JSON:
```json
"""


# =============================================================================
# STEERING HOOKS
# =============================================================================

def create_boost_hook(scale: float = 1.5) -> callable:
    """Create a hook that boosts activations by scale factor."""
    def hook(args, output, wrapper):
        if output is None:
            return None
        return output * scale
    return hook


def create_layer_hooks(
    layers: List[int],
    scale: float = 1.5,
) -> List[Tuple[str, callable]]:
    """Create hooks for specified layers."""
    hooks = []
    hook_fn = create_boost_hook(scale)
    for layer in layers:
        layer_name = f"model.layers.{layer}"
        hooks.append((layer_name, hook_fn))
    return hooks


# =============================================================================
# SCALING TESTER
# =============================================================================

class ScalingTester:
    """
    Tests steering across model sizes.
    """

    def __init__(
        self,
        model_name: str = MODEL_7B,
        config: ScalingConfig = None,
    ):
        self.model_name = model_name
        self.config = config or ScalingConfig()
        self.model = None
        self._tokenizer = None

        # Determine layer mapping based on model
        if "7B" in model_name or "7b" in model_name:
            self.layer_map = LAYER_MAP_7B
            self.num_layers = 32
        else:
            self.layer_map = LAYER_MAP_0_5B
            self.num_layers = 24

    def load_model(self):
        """Load model with memory management."""
        if self.model is not None:
            return

        print(f"Loading {self.model_name}...")
        print("This may take a moment for the 7B model...")

        # Clear any existing models from memory
        gc.collect()
        if MLX_AVAILABLE:
            mx.metal.clear_cache()

        if MLUX_AVAILABLE:
            self.model = HookedModel.from_pretrained(self.model_name)
            self._tokenizer = self.model.tokenizer
            print(f"Loaded with {self.num_layers} layers")
        else:
            raise RuntimeError("mlux not available")

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self._tokenizer
            self.model = None
            self._tokenizer = None
            gc.collect()
            if MLX_AVAILABLE:
                mx.metal.clear_cache()
            print("Model unloaded")

    def _get_eos_token_id(self) -> int:
        if self._tokenizer is None:
            return 2
        if hasattr(self._tokenizer, 'eos_token_id'):
            return self._tokenizer.eos_token_id
        return 2

    def generate_single(
        self,
        prompt: str,
        hooks: List[Tuple[str, callable]] = None,
    ) -> GenerationResult:
        """Generate a single statechart."""
        self.load_model()

        full_prompt = SC_TEMPLATE.format(description=prompt)

        # Tokenize
        tokens = self.model.tokenize(full_prompt)
        input_ids = mx.array([tokens])

        # Sampler
        sampler = make_sampler(temp=self.config.temperature)

        start_time = time.time()
        generated_tokens = []
        current_ids = input_ids
        eos_id = self._get_eos_token_id()

        for _ in range(self.config.max_tokens):
            if hooks:
                logits = self.model.run_with_hooks(current_ids, hooks=hooks)
            else:
                logits = self.model.forward(current_ids)

            next_logits = logits[:, -1, :]
            next_token = sampler(next_logits)
            next_token_id = int(next_token.item())

            if next_token_id == eos_id:
                break

            generated_tokens.append(next_token_id)
            current_ids = mx.concatenate([current_ids, next_token.reshape(1, 1)], axis=1)

        output = self._tokenizer.decode(generated_tokens)
        gen_time = time.time() - start_time

        is_valid, error, num_states, depth = validate_statechart(output)

        return GenerationResult(
            prompt=prompt,
            output=output,
            is_valid=is_valid,
            error=error,
            num_states=num_states,
            hierarchy_depth=depth,
            generation_time=gen_time,
        )

    def run_config(
        self,
        config_name: str,
        hooks: List[Tuple[str, callable]] = None,
        num_samples: int = None,
        verbose: bool = True,
    ) -> ConfigResult:
        """Run a configuration across prompts."""
        num_samples = num_samples or self.config.num_samples
        prompts = SC_PROMPTS[:num_samples]

        if verbose:
            print(f"\nConfig: {config_name}")
            print(f"Samples: {num_samples}")

        results = []
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"  [{i+1}/{num_samples}] {prompt[:40]}...", end=" ")

            result = self.generate_single(prompt, hooks)
            results.append(result)

            if verbose:
                status = "VALID" if result.is_valid else f"INVALID"
                print(f"{status} ({result.num_states} states, depth={result.hierarchy_depth})")

        # Compute aggregates
        valid_count = sum(1 for r in results if r.is_valid)
        avg_states = sum(r.num_states for r in results) / max(len(results), 1)
        avg_depth = sum(r.hierarchy_depth for r in results) / max(len(results), 1)
        avg_time = sum(r.generation_time for r in results) / max(len(results), 1)

        return ConfigResult(
            config_name=config_name,
            results=results,
            validity_rate=valid_count / max(len(results), 1),
            avg_states=avg_states,
            avg_depth=avg_depth,
            avg_time=avg_time,
        )

    def run_all_configs(
        self,
        num_samples: int = None,
        verbose: bool = True,
    ) -> Dict[str, ConfigResult]:
        """Run all steering configurations."""
        num_samples = num_samples or self.config.num_samples
        boost = self.config.boost_factor

        if verbose:
            print("=" * 70)
            print(f"SCALING EXPERIMENT: {self.model_name}")
            print("=" * 70)
            print(f"Layers: {self.num_layers}")
            print(f"Structure layers: {self.layer_map['structure']}")
            print(f"Hierarchy layers: {self.layer_map['hierarchy']}")
            print(f"Boost factor: {boost}x")

        results = {}

        # 1. Baseline (no steering)
        results["baseline"] = self.run_config("baseline", hooks=None, num_samples=num_samples, verbose=verbose)

        # 2. Hierarchy boost
        hierarchy_hooks = create_layer_hooks(self.layer_map["hierarchy"], scale=boost)
        results["hierarchy_boost"] = self.run_config("hierarchy_boost", hooks=hierarchy_hooks, num_samples=num_samples, verbose=verbose)

        # 3. Structure boost
        structure_hooks = create_layer_hooks(self.layer_map["structure"], scale=boost)
        results["structure_boost"] = self.run_config("structure_boost", hooks=structure_hooks, num_samples=num_samples, verbose=verbose)

        # 4. Full steering (both)
        full_hooks = hierarchy_hooks + structure_hooks
        results["full_steering"] = self.run_config("full_steering", hooks=full_hooks, num_samples=num_samples, verbose=verbose)

        if verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, ConfigResult]):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"{'Config':<20} {'Validity':>10} {'Avg States':>12} {'Avg Depth':>10} {'Avg Time':>10}")
        print("-" * 70)

        for name, result in results.items():
            print(f"{name:<20} {result.validity_rate:>9.1%} {result.avg_states:>11.1f} {result.avg_depth:>9.1f} {result.avg_time:>9.2f}s")

        print("=" * 70)

        # Compare to 0.5B baseline
        baseline = results["baseline"]
        hierarchy = results["hierarchy_boost"]

        print(f"\nComparison to 0.5B baseline (80%):")
        print(f"  7B baseline: {baseline.validity_rate:.1%}")
        print(f"  7B hierarchy_boost: {hierarchy.validity_rate:.1%}")

        if baseline.validity_rate > 0.80:
            print("  [CONFIRMED] Larger model = higher baseline")
        if hierarchy.validity_rate > baseline.validity_rate:
            print("  [CONFIRMED] Steering improves validity")


# =============================================================================
# QUICK ATTENTION ANALYSIS
# =============================================================================

def analyze_attention_heads(model_name: str = MODEL_7B, num_layers: int = 5):
    """Quick attention analysis to find relevant heads."""
    print("=" * 60)
    print("QUICK ATTENTION ANALYSIS")
    print("=" * 60)

    if not MLUX_AVAILABLE:
        print("mlux not available")
        return

    print(f"Loading {model_name}...")
    model = HookedModel.from_pretrained(model_name)

    # Find available hooks
    hooks = model.available_hooks()
    print(f"\nFound {len(hooks)} hook points")

    # Show layer structure
    layer_hooks = [h for h in hooks if "layers" in h]
    print(f"Layer hooks: {len(layer_hooks)}")

    # Sample a few to understand structure
    print("\nSample hooks:")
    for h in layer_hooks[:10]:
        print(f"  {h}")

    # Cleanup
    del model
    gc.collect()
    if MLX_AVAILABLE:
        mx.metal.clear_cache()

    return hooks


# =============================================================================
# MAIN
# =============================================================================

def run_7b_experiment(num_samples: int = 5):
    """Run the 7B scaling experiment."""
    config = ScalingConfig(
        model_name=MODEL_7B,
        num_samples=num_samples,
        boost_factor=1.5,
    )

    tester = ScalingTester(MODEL_7B, config)

    try:
        results = tester.run_all_configs(num_samples=num_samples, verbose=True)

        # Format for report
        baseline = results["baseline"].validity_rate * 100
        hierarchy = results["hierarchy_boost"].validity_rate * 100
        structure = results["structure_boost"].validity_rate * 100
        full = results["full_steering"].validity_rate * 100

        report = f"SCALING_7B: baseline={baseline:.0f}%, hierarchy={hierarchy:.0f}%, structure={structure:.0f}%, full={full:.0f}%"
        print(f"\nReport: {report}")

        return results, report

    finally:
        tester.unload_model()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_attention_heads()
    else:
        num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        run_7b_experiment(num_samples)
