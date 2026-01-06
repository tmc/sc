#!/usr/bin/env python3
"""
Model Scaling Experiment: Test steering on Qwen2.5-Coder-3B-Instruct-4bit.

Tests if steering improvements scale with model size:
- Baseline from 0.5B: 80% validity, hierarchy_boost same
- Hypothesis: Larger model = higher baseline + larger steering delta

Protocol:
1. Discover structure/hierarchy heads for 3B model
2. Test same steering configs as 0.5B
3. Run 20 generations per config
4. Measure validity rate, hierarchy depth, state count
"""

import sys
import json
import time
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


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    num_layers: int
    num_heads: int
    d_head: int
    hidden_size: int


@dataclass
class HeadScore:
    """Score for a single attention head."""
    layer: int
    head: int
    structure_score: float = 0.0
    hierarchy_score: float = 0.0


@dataclass
class ScalingResult:
    """Results from scaling experiment."""
    model_name: str
    config_name: str
    num_samples: int = 0
    valid_json_count: int = 0
    has_hierarchy_count: int = 0
    total_states: int = 0
    total_nested: int = 0
    max_depth: int = 0
    generation_times: List[float] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return self.valid_json_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def hierarchy_rate(self) -> float:
        return self.has_hierarchy_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_states(self) -> float:
        return self.total_states / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_nested(self) -> float:
        return self.total_nested / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_time(self) -> float:
        return sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0


class Model3BScaling:
    """
    Model scaling experiment for 3B Qwen model.

    Tests if attention steering improvements scale with model size.
    """

    MODEL_NAME = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"

    def __init__(self):
        print("=" * 60)
        print("MODEL SCALING EXPERIMENT: 3B")
        print("=" * 60)

        self.model = None
        self.tokenizer = None
        self.config = None

        # Discovered heads (will be populated by analyze_heads)
        self.structure_heads: List[Tuple[int, int]] = []
        self.hierarchy_heads: List[Tuple[int, int]] = []

        if not MLUX_AVAILABLE:
            print("WARNING: mlux not available")
            # Default 3B architecture estimates
            self.config = ModelConfig(
                name=self.MODEL_NAME,
                num_layers=36,
                num_heads=16,
                d_head=128,
                hidden_size=2048,
            )
        else:
            self._load_model()

        print("=" * 60)

    def _load_model(self):
        """Load the 3B model."""
        from mlux import HookedModel

        print(f"Loading model: {self.MODEL_NAME}")
        print("(This may take a moment for 3B model...)")

        start = time.time()
        self.model = HookedModel.from_pretrained(self.MODEL_NAME)
        load_time = time.time() - start

        self.tokenizer = self.model.tokenizer
        cfg = self.model.config

        # Extract architecture
        if isinstance(cfg, dict):
            num_layers = cfg.get('num_hidden_layers', cfg.get('n_layers', 36))
            num_heads = cfg.get('num_attention_heads', cfg.get('n_heads', 16))
            hidden_size = cfg.get('hidden_size', 2048)
        else:
            num_layers = getattr(cfg, 'num_hidden_layers', 36)
            num_heads = getattr(cfg, 'num_attention_heads', 16)
            hidden_size = getattr(cfg, 'hidden_size', 2048)

        d_head = hidden_size // num_heads

        self.config = ModelConfig(
            name=self.MODEL_NAME,
            num_layers=num_layers,
            num_heads=num_heads,
            d_head=d_head,
            hidden_size=hidden_size,
        )

        print(f"Model loaded in {load_time:.1f}s")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Heads: {self.config.num_heads}")
        print(f"  d_head: {self.config.d_head}")
        print(f"  Hidden: {self.config.hidden_size}")

    def analyze_heads(self, num_samples: int = 3) -> Dict[str, List[HeadScore]]:
        """
        Analyze attention heads to find structure and hierarchy heads.

        For 3B model, heads may be at different layer positions than 0.5B.
        """
        print("\n--- Analyzing Heads for 3B Model ---")

        if not MLUX_AVAILABLE or self.model is None:
            print("Using estimated head positions for 3B")
            # Scale head positions from 0.5B (24 layers) to 3B (36 layers)
            # 0.5B structure heads: L11, L9 -> ~L16, L13 for 3B
            # 0.5B hierarchy head: L23 -> ~L34 for 3B
            self.structure_heads = [(16, 13), (16, 7), (13, 7)]
            self.hierarchy_heads = [(34, 1)]
            return {"structure": [], "hierarchy": []}

        # Test prompts for head analysis
        test_prompts = [
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "A", "children": [{"label": "B"}]}]}}',
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Parent", "children": [{"label": "Child1"}, {"label": "Child2"}]}]}}',
        ]

        # Structural tokens to look for
        structural_tokens = ['{', '}', '[', ']', ':', ',', '"children"', '"type"']

        head_scores: Dict[Tuple[int, int], HeadScore] = {}

        for prompt in test_prompts[:num_samples]:
            try:
                # Get attention patterns
                attention = self.model.get_attention_patterns(
                    prompt,
                    layers=list(range(self.config.num_layers)),
                )

                # Tokenize to find structural positions
                tokens = self.tokenizer.encode(prompt)
                token_strs = [self.tokenizer.decode([t]) for t in tokens]

                # Find structural token positions
                struct_positions = []
                for i, t in enumerate(token_strs):
                    if any(st in t for st in structural_tokens):
                        struct_positions.append(i)

                # Score each head
                for layer, attn in attention.items():
                    if not isinstance(layer, int):
                        continue

                    attn_arr = np.array(attn)
                    if len(attn_arr.shape) == 4:
                        attn_arr = attn_arr[0]  # Remove batch dim

                    for head in range(attn_arr.shape[0]):
                        key = (layer, head)
                        if key not in head_scores:
                            head_scores[key] = HeadScore(layer=layer, head=head)

                        head_attn = attn_arr[head]  # (seq, seq)

                        # Structure score: attention to structural tokens
                        if struct_positions:
                            struct_attn = head_attn[:, struct_positions].mean()
                            head_scores[key].structure_score += float(struct_attn)

                        # Hierarchy score: attention from children to parents
                        # Look for diagonal-adjacent attention patterns
                        diag_attn = 0
                        for i in range(1, min(head_attn.shape[0], head_attn.shape[1])):
                            diag_attn += head_attn[i, i-1]  # Attend to previous token
                        head_scores[key].hierarchy_score += float(diag_attn / max(1, head_attn.shape[0]-1))

            except Exception as e:
                print(f"  Analysis error: {e}")
                continue

        # Normalize scores
        for score in head_scores.values():
            score.structure_score /= max(1, num_samples)
            score.hierarchy_score /= max(1, num_samples)

        # Find top structure heads
        by_structure = sorted(head_scores.values(), key=lambda s: s.structure_score, reverse=True)
        self.structure_heads = [(s.layer, s.head) for s in by_structure[:3]]

        # Find top hierarchy heads (prefer later layers)
        by_hierarchy = sorted(head_scores.values(), key=lambda s: s.hierarchy_score, reverse=True)
        self.hierarchy_heads = [(s.layer, s.head) for s in by_hierarchy[:1]]

        print(f"  Structure heads: {self.structure_heads}")
        print(f"  Hierarchy heads: {self.hierarchy_heads}")

        return {
            "structure": by_structure[:10],
            "hierarchy": by_hierarchy[:10],
        }

    def create_pre_hook(self, head_scales: Dict[int, float]) -> Callable:
        """Create pre-hook for head amplification on o_proj."""
        d_head = self.config.d_head
        num_heads = self.config.num_heads

        def pre_hook(args, kwargs, wrapper):
            if not head_scales or not MLX_AVAILABLE:
                return args, kwargs

            x = args[0]  # (batch, seq, n_heads * d_head)
            modified_x = x

            for head_idx, scale in sorted(head_scales.items()):
                if head_idx >= num_heads:
                    continue

                start = head_idx * d_head
                end = (head_idx + 1) * d_head

                head_output = modified_x[..., start:end] * scale
                before = modified_x[..., :start]
                after = modified_x[..., end:]
                modified_x = mx.concatenate([before, head_output, after], axis=-1)

            return (modified_x,) + args[1:], kwargs

        return pre_hook

    def generate_with_steering(
        self,
        prompt: str,
        structure_scale: float = 1.0,
        hierarchy_scale: float = 1.0,
        max_tokens: int = 200,
        temperature: float = 0.3,
    ) -> Tuple[str, float]:
        """
        Generate with head steering.

        Returns (output, generation_time)
        """
        if not MLUX_AVAILABLE or self.model is None:
            return f"[MOCK] structure={structure_scale}, hierarchy={hierarchy_scale}", 0.1

        # Build pre-hooks
        pre_hooks = []

        # Group heads by layer
        layer_heads: Dict[int, Dict[int, float]] = {}

        if structure_scale != 1.0:
            for l, h in self.structure_heads:
                if l not in layer_heads:
                    layer_heads[l] = {}
                layer_heads[l][h] = structure_scale

        if hierarchy_scale != 1.0:
            for l, h in self.hierarchy_heads:
                if l not in layer_heads:
                    layer_heads[l] = {}
                layer_heads[l][h] = hierarchy_scale

        for layer, head_scales in layer_heads.items():
            hook_name = f"model.layers.{layer}.self_attn.o_proj"
            pre_hook = self.create_pre_hook(head_scales)
            pre_hooks.append((hook_name, pre_hook))

        # Generate token by token
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        generated_tokens = []

        start_time = time.time()

        for _ in range(max_tokens):
            if pre_hooks:
                logits = self.model.run_with_hooks(tokens, pre_hooks=pre_hooks)
            else:
                logits = self.model.run_with_hooks(tokens)

            next_logits = logits[:, -1, :]
            next_token = sampler(next_logits)

            if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

        gen_time = time.time() - start_time
        output = self.tokenizer.decode(generated_tokens)

        return output, gen_time

    def analyze_output(self, prompt: str, output: str) -> Dict[str, Any]:
        """Analyze generated output for validity and structure."""
        result = {
            "valid_json": False,
            "has_hierarchy": False,
            "num_states": 0,
            "num_nested": 0,
            "max_depth": 0,
        }

        # Combine prompt and output for complete JSON
        full_text = prompt + output

        try:
            # Find JSON boundaries
            start = full_text.find('{')
            if start == -1:
                return result

            # Balance braces to find end
            depth = 0
            end = start
            for i, c in enumerate(full_text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            json_str = full_text[start:end]
            data = json.loads(json_str)
            result["valid_json"] = True

            # Count states and hierarchy
            def analyze_state(state: Dict, depth: int = 0) -> Tuple[int, int, int]:
                """Returns (total_states, nested_count, max_depth)"""
                total = 1
                nested = 0
                max_d = depth

                children = state.get('children', [])
                if children:
                    nested = 1
                    for child in children:
                        if isinstance(child, dict):
                            c_total, c_nested, c_depth = analyze_state(child, depth + 1)
                            total += c_total
                            nested += c_nested
                            max_d = max(max_d, c_depth)

                return total, nested, max_d

            root = data.get('root_state', data)
            if isinstance(root, dict):
                total, nested, max_d = analyze_state(root)
                result["num_states"] = total
                result["num_nested"] = nested
                result["max_depth"] = max_d
                result["has_hierarchy"] = nested > 0

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return result

    def run_benchmark(self, num_samples: int = 20) -> Dict[str, ScalingResult]:
        """Run full scaling benchmark."""
        print("\n" + "=" * 60)
        print(f"RUNNING 3B BENCHMARK ({num_samples} samples per config)")
        print("=" * 60)

        # First analyze heads if not done
        if not self.structure_heads or not self.hierarchy_heads:
            self.analyze_heads()

        # Test prompts with explicit structure
        prompts = [
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle", "type": 1}, {"label": "Active", "type": 2, "children": [{"label": "Running", "type": 1}, {"label":',
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Off", "type": 1}, {"label": "On", "type": 2, "children": [{"label": "Low", "type": 1}, {"label":',
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Init", "type": 1}, {"label": "Ready", "type": 2, "children": [{"label": "Waiting", "type": 1}, {"label":',
            '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Start", "type": 1}, {"label": "Process", "type": 2, "children": [{"label": "Step1", "type": 1}, {"label":',
        ]

        # Configurations to test
        configs = {
            "baseline": {"structure_scale": 1.0, "hierarchy_scale": 1.0},
            "hierarchy_boost": {"structure_scale": 1.0, "hierarchy_scale": 1.5},
            "structure_boost": {"structure_scale": 1.5, "hierarchy_scale": 1.0},
            "full_steering": {"structure_scale": 1.5, "hierarchy_scale": 1.5},
        }

        results = {name: ScalingResult(self.MODEL_NAME, name) for name in configs}

        for config_name, config in configs.items():
            print(f"\n--- Config: {config_name} ---")
            result = results[config_name]

            for i in range(num_samples):
                prompt = prompts[i % len(prompts)]

                output, gen_time = self.generate_with_steering(
                    prompt,
                    structure_scale=config["structure_scale"],
                    hierarchy_scale=config["hierarchy_scale"],
                )

                analysis = self.analyze_output(prompt, output)

                result.num_samples += 1
                result.generation_times.append(gen_time)

                if analysis["valid_json"]:
                    result.valid_json_count += 1
                if analysis["has_hierarchy"]:
                    result.has_hierarchy_count += 1
                result.total_states += analysis["num_states"]
                result.total_nested += analysis["num_nested"]
                result.max_depth = max(result.max_depth, analysis["max_depth"])

                status = "H" if analysis["has_hierarchy"] else "V" if analysis["valid_json"] else "X"
                print(f"  [{status}] Sample {i+1}/{num_samples}: {analysis['num_states']} states, depth {analysis['max_depth']}")

        return results

    def print_summary(self, results: Dict[str, ScalingResult]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("3B MODEL SCALING RESULTS")
        print("=" * 60)

        print(f"\nModel: {self.MODEL_NAME}")
        print(f"Architecture: {self.config.num_layers} layers, {self.config.num_heads} heads")
        print(f"Structure heads: {self.structure_heads}")
        print(f"Hierarchy heads: {self.hierarchy_heads}")

        print("\n--- Results by Configuration ---")
        print(f"{'Config':<20} {'Valid%':>8} {'Hier%':>8} {'AvgStates':>10} {'AvgNested':>10} {'MaxDepth':>8} {'AvgTime':>8}")
        print("-" * 80)

        for name, r in sorted(results.items()):
            print(f"{name:<20} {r.validity_rate:>7.1%} {r.hierarchy_rate:>7.1%} "
                  f"{r.avg_states:>10.1f} {r.avg_nested:>10.1f} {r.max_depth:>8d} {r.avg_time:>7.2f}s")

        # Compare to 0.5B baseline
        print("\n--- Comparison to 0.5B ---")
        print("0.5B baseline: 80% validity, 80% hierarchy_boost")

        baseline = results.get("baseline")
        hier_boost = results.get("hierarchy_boost")

        if baseline:
            delta = baseline.validity_rate - 0.80
            print(f"3B baseline: {baseline.validity_rate:.1%} ({delta:+.1%} vs 0.5B)")

        if hier_boost and baseline:
            steering_delta = hier_boost.hierarchy_rate - baseline.hierarchy_rate
            print(f"3B steering effect: {steering_delta:+.1%} hierarchy improvement")


def run_experiment():
    """Run the full 3B scaling experiment."""
    exp = Model3BScaling()

    # Analyze heads first
    exp.analyze_heads(num_samples=2)

    # Run benchmark
    results = exp.run_benchmark(num_samples=20)

    # Print summary
    exp.print_summary(results)

    return exp, results


if __name__ == "__main__":
    run_experiment()
