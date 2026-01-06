#!/usr/bin/env python3
"""
Head Amplifier: Amplify or suppress specific attention heads during generation.

Uses mlux hooks to modify attention weights at specific layers/heads,
enabling targeted steering based on discovered structure-aware heads.

Discovered heads from exp_mlux_sc_attention:
- Structure heads: L11H13, L11H7, L9H7 (attend to structural tokens)
- Hierarchy head: L23H1 (tracks parent-child relationships)
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable

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


# Discovered heads from exp_mlux_sc_attention
STRUCTURE_HEADS = [
    (11, 13),  # L11H13 - highest structure score
    (11, 7),   # L11H7
    (9, 7),    # L9H7
]

HIERARCHY_HEADS = [
    (23, 1),   # L23H1 - tracks parent-child (0.27 attention)
]

KEYWORD_HEADS = [
    (0, 6),    # L0H6 - keyword focused
    (1, 4),    # L1H4
    (4, 10),   # L4H10
]


@dataclass
class HeadConfig:
    """Configuration for a single head."""
    layer: int
    head: int
    scale: float = 1.0  # >1 amplify, <1 suppress, 1 unchanged
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"L{self.layer}H{self.head}"


@dataclass
class AmplificationConfig:
    """Configuration for head amplification."""
    heads: List[HeadConfig] = field(default_factory=list)
    default_scale: float = 1.0  # Scale for non-specified heads

    @classmethod
    def amplify_structure(cls, scale: float = 1.5) -> "AmplificationConfig":
        """Create config to amplify structure heads."""
        return cls(heads=[
            HeadConfig(layer=l, head=h, scale=scale, name=f"structure_L{l}H{h}")
            for l, h in STRUCTURE_HEADS
        ])

    @classmethod
    def amplify_hierarchy(cls, scale: float = 1.5) -> "AmplificationConfig":
        """Create config to amplify hierarchy head."""
        return cls(heads=[
            HeadConfig(layer=l, head=h, scale=scale, name=f"hierarchy_L{l}H{h}")
            for l, h in HIERARCHY_HEADS
        ])

    @classmethod
    def amplify_all_discovered(cls, scale: float = 1.3) -> "AmplificationConfig":
        """Amplify all discovered heads."""
        heads = []
        for l, h in STRUCTURE_HEADS:
            heads.append(HeadConfig(l, h, scale, f"structure_L{l}H{h}"))
        for l, h in HIERARCHY_HEADS:
            heads.append(HeadConfig(l, h, scale, f"hierarchy_L{l}H{h}"))
        return cls(heads=heads)

    @classmethod
    def suppress_hierarchy(cls, scale: float = 0.5) -> "AmplificationConfig":
        """Suppress hierarchy head (for comparison)."""
        return cls(heads=[
            HeadConfig(layer=l, head=h, scale=scale, name=f"hierarchy_L{l}H{h}")
            for l, h in HIERARCHY_HEADS
        ])


class HeadAmplifier:
    """
    Amplifies or suppresses specific attention heads during generation.

    Uses mlux pre-hooks on o_proj to modify attention head outputs.
    The o_proj input has shape (batch, seq, n_heads * d_head), where each
    head's output occupies d_head dimensions.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        print("=" * 60)
        print("HEAD AMPLIFIER")
        print("=" * 60)

        if not MLUX_AVAILABLE:
            print("WARNING: mlux not available")
            self.model = None
            self.num_layers = 24
            self.num_heads = 14
            self.d_head = 64
        else:
            from mlux import HookedModel
            print(f"Loading model: {model_name}")
            self.model = HookedModel.from_pretrained(model_name)
            config = self.model.config
            self.num_layers = config.get('num_hidden_layers', config.get('n_layers', 24)) if isinstance(config, dict) else 24
            self.num_heads = config.get('num_attention_heads', config.get('n_heads', 14)) if isinstance(config, dict) else 14
            # d_head is typically hidden_size / n_heads, default 64 for Qwen2.5-0.5B
            hidden = config.get('hidden_size', self.num_heads * 64) if isinstance(config, dict) else self.num_heads * 64
            self.d_head = hidden // self.num_heads if hidden else 64
            print(f"Model loaded. Layers: {self.num_layers}, Heads: {self.num_heads}, d_head: {self.d_head}")

        print("=" * 60)

    def create_head_amplify_pre_hook(
        self,
        head_scales: Dict[int, float],
    ) -> Callable:
        """
        Create a pre-hook that scales specific heads in o_proj input.

        Args:
            head_scales: {head_idx: scale} mapping

        Returns:
            Pre-hook function for mlux (args, kwargs, wrapper) -> (args, kwargs)
        """
        d_head = self.d_head

        def pre_hook(args, kwargs, wrapper):
            """Scale specific head dimensions before o_proj."""
            if not head_scales or not MLX_AVAILABLE:
                return args, kwargs

            x = args[0]  # (batch, seq, n_heads * d_head)

            # Apply scaling to each specified head
            modified_x = x
            for head_idx, scale in sorted(head_scales.items()):
                if head_idx >= self.num_heads:
                    continue

                start = head_idx * d_head
                end = (head_idx + 1) * d_head

                # Scale this head's dimensions
                head_output = modified_x[..., start:end] * scale

                # Reconstruct tensor
                before = modified_x[..., :start]
                after = modified_x[..., end:]
                modified_x = mx.concatenate([before, head_output, after], axis=-1)

            return (modified_x,) + args[1:], kwargs

        return pre_hook

    def create_amplification_hooks(
        self,
        config: AmplificationConfig,
    ) -> List[Tuple[str, Callable]]:
        """
        Create pre-hooks for all layers that need head amplification.

        Args:
            config: Amplification configuration

        Returns:
            List of (hook_name, pre_hook_fn) tuples for mlux
        """
        # Group heads by layer
        layer_heads: Dict[int, Dict[int, float]] = {}
        for h in config.heads:
            if h.layer not in layer_heads:
                layer_heads[h.layer] = {}
            layer_heads[h.layer][h.head] = h.scale

        # Create pre-hooks for each layer
        pre_hooks = []
        for layer, head_scales in layer_heads.items():
            hook_name = f"model.layers.{layer}.self_attn.o_proj"
            pre_hook = self.create_head_amplify_pre_hook(head_scales)
            pre_hooks.append((hook_name, pre_hook))

        return pre_hooks

    def generate_with_amplification(
        self,
        prompt: str,
        config: AmplificationConfig,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with head amplification using pre-hooks on o_proj.

        Args:
            prompt: Input prompt
            config: Amplification configuration
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            (generated_text, metadata)
        """
        if not MLUX_AVAILABLE or self.model is None:
            return self._mock_generate(prompt, config)

        # Build pre-hooks for head amplification
        pre_hooks = self.create_amplification_hooks(config)

        # Generate with pre-hooks
        try:
            # run_with_hooks returns logits, need to decode
            output = self.model.run_with_hooks(
                prompt,
                pre_hooks=pre_hooks,  # Use pre_hooks instead of hooks
            )

            # output is logits tensor, need to decode
            # For generation, we need to use mlx_lm with hooks
            # Actually, run_with_hooks gives us just one forward pass logits
            # For full generation with hooks, we need a different approach

            # Use iterative generation with hooks
            generated_text = self._generate_with_hooks(prompt, pre_hooks, max_tokens, temperature)

            metadata = {
                "amplified_heads": [(h.layer, h.head, h.scale) for h in config.heads],
                "pre_hooks_applied": len(pre_hooks),
                "method": "pre_hooks",
            }

            return generated_text, metadata

        except Exception as e:
            print(f"Generation with pre-hooks failed: {e}")
            # Fallback to normal generation using mlx_lm
            try:
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(temp=temperature)
                output = generate(
                    self.model.model,
                    self.model.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                )
                return output, {"error": str(e), "fallback": True}
            except Exception as e2:
                print(f"Fallback generation also failed: {e2}")
                return f"[ERROR] Generation failed: {e2}", {"error": str(e2)}

    def _generate_with_hooks(
        self,
        prompt: str,
        pre_hooks: List[Tuple[str, Callable]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate text token by token with hooks applied.

        This enables head amplification during autoregressive generation.
        """
        from mlx_lm.sample_utils import make_sampler

        tokenizer = self.model.tokenizer
        sampler = make_sampler(temp=temperature)

        # Encode prompt
        tokens = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)
        generated_tokens = []

        for _ in range(max_tokens):
            # Forward pass with hooks
            logits = self.model.run_with_hooks(tokens, pre_hooks=pre_hooks)

            # Get next token logits (last position)
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Sample next token
            next_token = sampler(next_logits)

            # Check for EOS
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())

            # Append to sequence
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

        # Decode generated tokens
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text

    def _mock_generate(
        self,
        prompt: str,
        config: AmplificationConfig,
    ) -> Tuple[str, Dict[str, Any]]:
        """Mock generation for testing."""
        heads_str = ", ".join(h.name for h in config.heads)
        return (
            f"[MOCK] Amplified heads: {heads_str}\nGenerated for: {prompt[:50]}...",
            {"mock": True, "heads": [(h.layer, h.head, h.scale) for h in config.heads]}
        )

    def compare_amplification(
        self,
        prompt: str,
        configs: Dict[str, AmplificationConfig],
        max_tokens: int = 256,
    ) -> Dict[str, Tuple[str, Dict]]:
        """
        Compare different amplification configurations.

        Args:
            prompt: Input prompt
            configs: Named configurations to compare
            max_tokens: Max tokens per generation

        Returns:
            {config_name: (output, metadata)}
        """
        results = {}

        # Baseline (no amplification)
        if MLUX_AVAILABLE and self.model is not None:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            try:
                sampler = make_sampler(temp=0.3)
                baseline = generate(
                    self.model.model,
                    self.model.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                )
                results["baseline"] = (baseline, {"amplified_heads": []})
            except Exception as e:
                results["baseline"] = (f"[ERROR] {e}", {"error": str(e)})
        else:
            results["baseline"] = (f"[MOCK] Baseline: {prompt[:50]}...", {"mock": True})

        # Each configuration
        for name, config in configs.items():
            output, meta = self.generate_with_amplification(prompt, config, max_tokens)
            results[name] = (output, meta)

        return results


def demo():
    """Demo head amplification."""
    print("=" * 60)
    print("HEAD AMPLIFICATION DEMO")
    print("=" * 60)

    amplifier = HeadAmplifier()

    prompt = '''Generate a hierarchical statechart JSON:
```json
{"root_state": {"label": "__root__", "type": 2, "children": ['''

    configs = {
        "amplify_hierarchy": AmplificationConfig.amplify_hierarchy(scale=1.5),
        "amplify_structure": AmplificationConfig.amplify_structure(scale=1.5),
        "suppress_hierarchy": AmplificationConfig.suppress_hierarchy(scale=0.5),
    }

    print(f"\nPrompt: {prompt[:60]}...")
    print(f"\nConfigurations to test:")
    for name, config in configs.items():
        heads = [(h.layer, h.head, h.scale) for h in config.heads]
        print(f"  {name}: {heads}")

    results = amplifier.compare_amplification(prompt, configs, max_tokens=200)

    print("\n--- Results ---")
    for name, (output, meta) in results.items():
        print(f"\n[{name}]")
        print(f"  Output: {output[:100]}...")
        print(f"  Metadata: {meta}")

    return results


if __name__ == "__main__":
    demo()
