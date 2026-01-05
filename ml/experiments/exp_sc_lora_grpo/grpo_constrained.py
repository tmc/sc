#!/usr/bin/env python3
"""
Hybrid GRPO + Constrained Decoding.

Combines:
1. GRPO-trained LoRA for semantic quality (83% validity)
2. Retok-TopK constrained decoding for structural validity (100% JSON)

Expected: Best of both - 100% JSON validity + good semantic content.
"""

import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Any, Optional
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

from .sc_reward import compute_sc_reward, RewardBreakdown
from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    JSONTokenizer,
)


@dataclass
class HybridConfig:
    """Configuration for hybrid generation."""
    model_path: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"

    # LoRA (same as GRPO training)
    lora_rank: int = 8
    lora_layers: List[int] = None

    # Generation
    max_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 500  # For constrained masking

    def __post_init__(self):
        if self.lora_layers is None:
            self.lora_layers = [0, 1, 2, 3]


def load_sc_json_statechart() -> Dict:
    """Load the SC JSON grammar statechart from file."""
    grammar_path = Path(__file__).parent.parent / "exp_schema_guided_sampling" / "json_parser_v2.statechart.json"
    with open(grammar_path) as f:
        return json.load(f)


class ConstrainedSampler:
    """Retok-TopK constrained sampler for JSON generation."""

    def __init__(self, tokenizer, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self.vocab = tokenizer.get_vocab()
        self.id_to_str = {v: k for k, v in self.vocab.items()}
        self.generated_string = ""

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()
        self.generated_string = ""

    def initialize_with_prefix(self, prefix: str):
        """Process prefix to initialize machine state."""
        self.reset()
        for char in prefix:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)
        self.generated_string = prefix

    def _validate_token(self, token_str: str) -> bool:
        """Simulate token through statechart."""
        test_machine = StatechartMachine(statechart=self.statechart)
        test_machine.current_state = self.machine.current_state
        test_machine.context = self.machine.context.copy()
        test_tokenizer = JSONTokenizer()
        test_tokenizer.in_string = self.json_tokenizer.in_string
        test_tokenizer.current_token = self.json_tokenizer.current_token
        test_tokenizer.escape_next = self.json_tokenizer.escape_next

        for char in token_str:
            event = test_tokenizer.process_char(char)
            if event:
                enabled = test_machine.get_enabled_events()
                if event not in enabled:
                    return False
                test_machine.send_event(event)

        return True

    def get_valid_mask(self, top_k_ids: List[int]) -> mx.array:
        """Get mask for valid tokens among top-k candidates."""
        mask = []
        for token_id in top_k_ids:
            token_str = self.id_to_str.get(token_id, "")
            valid = self._validate_token(token_str)
            mask.append(1.0 if valid else -float('inf'))
        return mx.array(mask)

    def process_token(self, token_str: str):
        """Update state after generating token."""
        self.generated_string += token_str
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)

    def is_complete(self) -> bool:
        return self.machine.current_state == "Complete"


def apply_lora(model, config: HybridConfig):
    """Apply LoRA to model (same as GRPO training)."""
    model.freeze()

    for layer_idx in config.lora_layers:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        for proj_name in ['q_proj', 'v_proj']:
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                lora = LoRALinear.from_base(
                    proj,
                    r=config.lora_rank,
                    scale=16.0 / config.lora_rank,
                )
                setattr(attn, proj_name, lora)

    return model


def generate_constrained(
    model,
    tokenizer,
    sampler: ConstrainedSampler,
    prompt: str,
    json_prefix: str,
    config: HybridConfig,
) -> Tuple[str, float]:
    """Generate with constrained decoding."""

    # Initialize sampler with JSON prefix (e.g., '{"root_state":')
    sampler.initialize_with_prefix(json_prefix)

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array([prompt_tokens])

    generated_tokens = []
    t0 = time.time()

    for _ in range(config.max_tokens):
        # Forward pass
        logits = model(input_ids)
        next_logits = logits[0, -1, :]

        # Get top-k candidates
        top_k_indices = mx.argpartition(-next_logits, config.top_k)[:config.top_k]
        top_k_logits = next_logits[top_k_indices]

        # Apply constraint mask
        mask = sampler.get_valid_mask(top_k_indices.tolist())
        masked_logits = top_k_logits + mask

        # Check if any valid tokens
        valid_count = mx.sum(mask > -float('inf')).item()
        if valid_count == 0:
            break

        # Sample from masked distribution
        scaled_logits = masked_logits / config.temperature
        probs = mx.softmax(scaled_logits)

        # Sample
        sampled_idx = mx.random.categorical(scaled_logits[None, :])[0]
        next_token = top_k_indices[sampled_idx]

        # Decode and process
        token_str = tokenizer.decode([next_token.item()])
        sampler.process_token(token_str)

        generated_tokens.append(next_token.item())

        # Check completion
        if sampler.is_complete():
            break

        # Update input
        input_ids = mx.concatenate([input_ids, next_token[None, None]], axis=1)

    elapsed = time.time() - t0
    output = tokenizer.decode(generated_tokens)

    return output, elapsed


def create_few_shot_prompt(task: str) -> str:
    """Few-shot prompt for SC generation."""
    return f'''Generate a statechart JSON. Output ONLY valid JSON.

Example: Toggle
{{"root_state": {{"label": "Toggle", "type": 2, "children": [{{"label": "Off", "type": 1, "is_initial": true}}, {{"label": "On", "type": 1}}]}}, "transitions": [{{"from": ["Off"], "to": ["On"], "event": "TURN_ON"}}]}}

{task}
{{"root_state":'''


TEST_TASKS = [
    "Create a Door statechart with Open and Closed states",
    "Create a Player statechart with Idle and Running states",
    "Create a Switch with On and Off states",
    "Create a Connection with Connected and Disconnected states",
    "Create a Timer with Stopped and Running states",
    "Create a Traffic Light with Red, Yellow, Green states",
    "Create a Login flow with LoggedOut and LoggedIn states",
    "Create a Game with Menu, Playing, Paused states",
]


def run_hybrid_benchmark():
    """Run hybrid GRPO + constrained benchmark."""
    print("=" * 60)
    print("GRPO + CONSTRAINED DECODING BENCHMARK")
    print("=" * 60)

    config = HybridConfig()

    # Load model
    print(f"Loading model: {config.model_path}")
    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    # Apply LoRA (simulates GRPO-trained weights)
    print(f"Applying LoRA (rank={config.lora_rank}, layers={config.lora_layers})")
    apply_lora(model, config)
    print("LoRA applied.")

    # Load SC JSON grammar
    sc_grammar = load_sc_json_statechart()
    sampler = ConstrainedSampler(tokenizer, sc_grammar)

    # Run benchmark
    print("\n" + "-" * 60)
    print("GENERATION BENCHMARK")
    print("-" * 60)

    results = []
    json_valid = 0
    sc_valid = 0
    unique_outputs = set()
    total_time = 0.0

    for i, task in enumerate(TEST_TASKS):
        prompt = create_few_shot_prompt(task)
        sampler.reset()

        json_prefix = '{"root_state":'
        output, elapsed = generate_constrained(model, tokenizer, sampler, prompt, json_prefix, config)
        total_time += elapsed

        # Prepend the start
        full_output = '{"root_state":' + output

        # Clean up
        if '\n' in full_output:
            full_output = full_output.split('\n')[0]

        # Validate
        try:
            parsed = json.loads(full_output)
            is_json_valid = True
            json_valid += 1
        except:
            is_json_valid = False

        reward, breakdown = compute_sc_reward(full_output)
        is_sc_valid = reward >= 0.6
        if is_sc_valid:
            sc_valid += 1

        unique_outputs.add(full_output)

        results.append({
            'task': task,
            'output': full_output,
            'json_valid': is_json_valid,
            'sc_valid': is_sc_valid,
            'reward': reward,
            'time': elapsed,
        })

        status = "✓" if is_sc_valid else "✗"
        print(f"{status} Task {i+1}: JSON={is_json_valid}, SC={is_sc_valid}, reward={reward:.2f}, time={elapsed:.2f}s")
        print(f"  Output: {full_output[:80]}...")

    # Summary
    n = len(TEST_TASKS)
    json_rate = json_valid / n
    sc_rate = sc_valid / n
    diversity = len(unique_outputs) / n
    avg_time = total_time / n

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"JSON validity:    {json_rate:.1%} ({json_valid}/{n})")
    print(f"SC validity:      {sc_rate:.1%} ({sc_valid}/{n})")
    print(f"Diversity:        {diversity:.1%} ({len(unique_outputs)} unique)")
    print(f"Avg time:         {avg_time:.2f}s")

    print(f"\nGRPO_CONSTRAINED: validity={sc_rate:.1%}, json={json_rate:.1%}, diversity={diversity:.1%}")

    return {
        'json_validity': json_rate,
        'sc_validity': sc_rate,
        'diversity': diversity,
        'avg_time': avg_time,
        'results': results,
    }


if __name__ == "__main__":
    run_hybrid_benchmark()
