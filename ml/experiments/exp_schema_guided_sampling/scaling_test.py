#!/usr/bin/env python3
"""
Retok-TopK Scaling Test: 1.5B and 3B Models

Tests if Retok-TopK approach scales to larger models.

Baseline findings from 0.5B:
- Retok-TopK: 35ms mask time, 100% valid JSON
- Top-K = 500 candidates checked

Test on:
- Qwen2.5-Coder-1.5B-Instruct-4bit
- Qwen2.5-Coder-3B-Instruct-4bit

Metrics:
- Mask time (ms)
- Validity rate (%)
- Generation time (s)
- Tokens generated
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False

# Import the approaches
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.exp_schema_guided_sampling.approaches import RetokenizationApproach, check_balanced
from experiments.exp_schema_guided_sampling.statechart_sampler import JSONTokenizer


MODELS = {
    "0.5B": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "1.5B": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "3B": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
}

TOP_K = 500  # Only check top 500 candidates


@dataclass
class ScalingResult:
    """Result from testing a model."""
    model_name: str
    mask_time_ms: float
    valid_json: bool
    balanced: bool
    gen_time_s: float
    tokens_generated: int
    output_preview: str


def test_retok_topk(
    model_name: str,
    model: Any,
    tokenizer: Any,
    statechart: Dict,
    prompt: str,
    max_tokens: int = 100,
) -> ScalingResult:
    """
    Test Retok-TopK approach on a model.
    """
    from mlx_lm.sample_utils import make_sampler

    # Initialize approach
    approach = RetokenizationApproach(tokenizer, statechart)
    approach.reset()

    # Process prompt through machine
    json_tok = JSONTokenizer()
    for char in prompt:
        event = json_tok.process_char(char)
        if event:
            approach.machine.send_event(event)
    approach.generated_string = prompt
    approach.json_tokenizer = json_tok

    # Setup
    tokens = mx.array(tokenizer.encode(prompt))[None]
    sampler = make_sampler(temp=0.3)
    generated = []
    mask_times = []

    t0 = time.time()

    for i in range(max_tokens):
        # Forward pass
        logits = model(tokens)
        next_logits = logits[:, -1, :]

        # Get mask (Retok-TopK)
        mt0 = time.time()
        top_k_indices = mx.argpartition(next_logits.squeeze(), -TOP_K)[-TOP_K:]
        top_k_ids = [int(i) for i in top_k_indices.tolist()]
        valid = approach.get_valid_tokens(top_k_ids)
        mask_times.append(time.time() - mt0)

        # Apply mask
        if valid:
            vocab_size = next_logits.shape[-1]
            mask_np = np.full(vocab_size, -100.0)
            for tid in valid:
                if tid < vocab_size:
                    mask_np[tid] = 0.0
            next_logits = next_logits + mx.array(mask_np)

        # Sample
        next_token = sampler(next_logits)
        token_id = next_token.item()

        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        token_str = tokenizer.decode([token_id])
        approach.process_token(token_str)

        # Check completion
        if approach.machine.is_complete():
            break

        if approach.machine.should_force_close():
            # Handle force close
            current = tokenizer.decode(generated)
            if current.rstrip().endswith(','):
                current = current.rstrip().rstrip(',')
                generated = list(tokenizer.encode(current))

            closing = ""
            for item in reversed(approach.machine.context.stack):
                if item == "object":
                    closing += "}"
                elif item == "array":
                    closing += "]"
            if closing:
                closing_tokens = tokenizer.encode(closing)
                generated.extend(closing_tokens)
            break

        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

    gen_time = time.time() - t0
    avg_mask_time = sum(mask_times) / len(mask_times) if mask_times else 0

    # Validate output
    output = tokenizer.decode(generated)
    full = prompt + output

    try:
        json.loads(full)
        valid_json = True
    except json.JSONDecodeError:
        valid_json = False

    balanced = check_balanced(full)

    return ScalingResult(
        model_name=model_name,
        mask_time_ms=avg_mask_time * 1000,
        valid_json=valid_json,
        balanced=balanced,
        gen_time_s=gen_time,
        tokens_generated=len(generated),
        output_preview=output[:80] + "..." if len(output) > 80 else output,
    )


def run_scaling_test(
    model_sizes: Optional[List[str]] = None,
    n_trials: int = 5,
) -> Dict[str, List[ScalingResult]]:
    """
    Run scaling test on multiple models.
    """
    if model_sizes is None:
        model_sizes = ["1.5B", "3B"]

    if not MLX_AVAILABLE:
        print("MLX not available")
        return {}

    from mlx_lm import load

    # Load statechart
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    # Test prompts - IMPORTANT: prompts must end at complete JSON value boundaries
    # (not mid-key or mid-value) to ensure ForceClose can trigger correctly
    prompts = [
        '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}',
        '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "s0", "type": 1, "is_initial": true}',
        '{"root_state": {"label": "Test", "children": [{"label": "A"}',
    ]

    results = {}

    for size in model_sizes:
        if size not in MODELS:
            print(f"Unknown model size: {size}")
            continue

        model_path = MODELS[size]
        print(f"\n{'=' * 60}")
        print(f"Testing {size}: {model_path}")
        print("=" * 60)

        try:
            print("Loading model...")
            model, tokenizer = load(model_path)
            print("Model loaded.")

            results[size] = []

            for trial in range(n_trials):
                prompt = prompts[trial % len(prompts)]
                print(f"\nTrial {trial + 1}/{n_trials}")

                result = test_retok_topk(
                    model_name=size,
                    model=model,
                    tokenizer=tokenizer,
                    statechart=sc_def,
                    prompt=prompt,
                    max_tokens=100,
                )

                results[size].append(result)
                print(f"  Mask: {result.mask_time_ms:.1f}ms, Valid: {result.valid_json}, Tokens: {result.tokens_generated}")

            # Clean up
            del model
            del tokenizer

        except Exception as e:
            print(f"Error testing {size}: {e}")
            import traceback
            traceback.print_exc()

    return results


def compute_summary(results: Dict[str, List[ScalingResult]]) -> Dict[str, Dict]:
    """Compute summary statistics."""
    summary = {}

    for size, trials in results.items():
        if not trials:
            continue

        mask_times = [r.mask_time_ms for r in trials]
        valid_count = sum(1 for r in trials if r.valid_json)
        gen_times = [r.gen_time_s for r in trials]

        summary[size] = {
            "avg_mask_ms": sum(mask_times) / len(mask_times),
            "min_mask_ms": min(mask_times),
            "max_mask_ms": max(mask_times),
            "validity_rate": valid_count / len(trials),
            "avg_gen_time": sum(gen_times) / len(gen_times),
            "n_trials": len(trials),
        }

    return summary


def main():
    """Run the scaling test."""
    print("=" * 70)
    print("RETOK-TOPK SCALING TEST: 1.5B and 3B Models")
    print("=" * 70)

    # Run tests
    results = run_scaling_test(["1.5B", "3B"], n_trials=5)

    if not results:
        print("No results collected")
        return None, None

    # Compute summary
    summary = compute_summary(results)

    # Print results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<10} {'Mask (ms)':<12} {'Valid':<12} {'Gen (s)':<12} {'Trials':<8}")
    print("-" * 60)

    for size, stats in summary.items():
        print(f"{size:<10} {stats['avg_mask_ms']:.1f}ms{'':<5} {stats['validity_rate']:.0%}{'':<6} {stats['avg_gen_time']:.2f}s{'':<5} {stats['n_trials']}")

    # Build report string
    report_parts = []
    for size in ["1.5B", "3B"]:
        if size in summary:
            s = summary[size]
            report_parts.append(f"{size}_mask={s['avg_mask_ms']:.0f}ms")
            report_parts.append(f"{size}_valid={s['validity_rate']:.0%}")

    report = ", ".join(report_parts)
    print(f"\nReport: RETOK_SCALING: {report}")

    return summary, report


if __name__ == "__main__":
    summary, report = main()
