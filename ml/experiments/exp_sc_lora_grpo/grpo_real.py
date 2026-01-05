#!/usr/bin/env python3
"""
Real GRPO Training with Actual Gradient Updates.

Uses MLX autodiff to compute gradients and update LoRA weights.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

from .sc_reward import compute_sc_reward, compute_grpo_advantages, RewardBreakdown


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    samples_per_prompt: int = 4
    max_tokens: int = 100
    temperature: float = 0.8

    epochs: int = 3
    learning_rate: float = 1e-4

    num_prompts: int = 5


def apply_lora_to_model(model, config: GRPOConfig):
    """Apply LoRA adapters to specified layers."""
    # Freeze all params first
    model.freeze()

    # Apply to q_proj, v_proj in specified layers
    num_layers = len(model.model.layers)
    lora_layers = config.lora_layers if config.lora_layers else list(range(num_layers))

    lora_count = 0
    for layer_idx in lora_layers:
        if layer_idx >= num_layers:
            continue
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        for proj_name in ['q_proj', 'v_proj']:
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                # Create LoRA wrapper using from_base (works with QuantizedLinear)
                lora = LoRALinear.from_base(
                    proj,
                    r=config.lora_rank,
                    scale=config.lora_alpha / config.lora_rank,
                )
                setattr(attn, proj_name, lora)
                lora_count += 1

    print(f"  Applied LoRA to {lora_count} projections")

    # Unfreeze LoRA parameters only
    # LoRA params are unfrozen by default in LoRALinear

    return model


def get_lora_parameters(model, config: GRPOConfig) -> Dict[str, Any]:
    """Extract only LoRA parameters for optimization."""
    params = {}
    num_layers = len(model.model.layers)
    lora_layers = config.lora_layers if config.lora_layers else list(range(num_layers))

    for layer_idx in lora_layers:
        if layer_idx >= num_layers:
            continue
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'v_proj']:
            proj = getattr(layer.self_attn, proj_name, None)
            if proj is not None and hasattr(proj, 'lora_a'):
                params[f"l{layer_idx}.{proj_name}.lora_a"] = proj.lora_a
                params[f"l{layer_idx}.{proj_name}.lora_b"] = proj.lora_b
    return params


def count_trainable_params(model) -> int:
    """Count trainable parameters in model."""
    total = 0
    for name, param in model.trainable_parameters().items():
        if isinstance(param, dict):
            for k, v in param.items():
                if hasattr(v, 'size'):
                    total += v.size
        elif hasattr(param, 'size'):
            total += param.size
    return total


def generate_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
) -> Tuple[str, mx.array, mx.array]:
    """Generate text and return log probabilities of generated tokens."""

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array([prompt_tokens])

    generated_tokens = []
    log_probs = []

    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_ids)
        next_logits = logits[0, -1, :]  # Last position

        # Apply temperature
        scaled_logits = next_logits / temperature

        # Sample
        probs = mx.softmax(scaled_logits)
        next_token = mx.random.categorical(scaled_logits[None, :])[0]

        # Get log prob of sampled token
        log_prob = mx.log(probs[next_token] + 1e-10)

        generated_tokens.append(next_token.item())
        log_probs.append(log_prob)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to input
        input_ids = mx.concatenate([input_ids, next_token[None, None]], axis=1)

    # Decode
    output_text = tokenizer.decode(generated_tokens)

    return output_text, mx.stack(log_probs), mx.array(generated_tokens)


def compute_grpo_loss(
    model,
    tokenizer,
    prompt_tokens: List[int],
    response_tokens: List[int],
    advantage: float,
) -> mx.array:
    """Compute GRPO loss for a single sample."""

    # Full sequence
    full_tokens = prompt_tokens + response_tokens
    input_ids = mx.array([full_tokens[:-1]])  # All but last
    target_ids = mx.array([full_tokens[1:]])   # All but first

    # Forward pass
    logits = model(input_ids)

    # Only compute loss on response tokens
    prompt_len = len(prompt_tokens) - 1  # -1 because of shift
    response_logits = logits[0, prompt_len:, :]
    response_targets = target_ids[0, prompt_len:]

    # Cross-entropy log probs (manual log_softmax)
    log_probs = response_logits - mx.logsumexp(response_logits, axis=-1, keepdims=True)
    token_log_probs = mx.take_along_axis(
        log_probs,
        response_targets[:, None],
        axis=-1
    ).squeeze(-1)

    # Sum log probs for sequence
    seq_log_prob = mx.sum(token_log_probs)

    # GRPO loss: -advantage * log_prob
    loss = -advantage * seq_log_prob

    return loss


def create_few_shot_prompt(task: str) -> str:
    """Create few-shot prompt for SC generation."""
    return f'''Generate a statechart JSON. Output ONLY valid JSON.

Example: Toggle
{{"root_state": {{"label": "Toggle", "type": 2, "children": [{{"label": "Off", "type": 1, "is_initial": true}}, {{"label": "On", "type": 1}}]}}, "transitions": [{{"from": ["Off"], "to": ["On"], "event": "TURN_ON"}}]}}

Example: Light
{{"root_state": {{"label": "Light", "type": 2, "children": [{{"label": "Red", "type": 1, "is_initial": true}}, {{"label": "Green", "type": 1}}]}}, "transitions": [{{"from": ["Red"], "to": ["Green"], "event": "GO"}}]}}

{task}
{{"root_state":'''


TRAINING_TASKS = [
    "Create a Door statechart with Open and Closed states",
    "Create a Player statechart with Idle and Running states",
    "Create a Switch with On and Off states",
    "Create a Connection with Connected and Disconnected states",
    "Create a Timer with Stopped and Running states",
]


def run_grpo_training():
    """Run GRPO training with real gradients."""
    print("=" * 60)
    print("GRPO TRAINING WITH REAL GRADIENTS")
    print("=" * 60)

    config = GRPOConfig()

    # Load model
    print(f"Loading model: {config.model_path}")
    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    # Apply LoRA
    print(f"Applying LoRA (rank={config.lora_rank}, layers={config.lora_layers})")
    apply_lora_to_model(model, config)

    # Count trainable params
    total_params = count_trainable_params(model)
    print(f"Trainable parameters: {total_params:,}")

    # Optimizer - only update LoRA params
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)

    for epoch in range(config.epochs):
        t0 = time.time()
        epoch_rewards = []
        epoch_losses = []
        valid_count = 0
        total_count = 0

        for task in TRAINING_TASKS[:config.num_prompts]:
            prompt = create_few_shot_prompt(task)
            prompt_tokens = tokenizer.encode(prompt)

            # Generate samples
            samples = []
            for _ in range(config.samples_per_prompt):
                output, log_probs, tokens = generate_with_logprobs(
                    model, tokenizer, prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # Prepend the starting JSON
                full_output = '{"root_state":' + output
                if '\n' in full_output:
                    full_output = full_output.split('\n')[0]

                reward, breakdown = compute_sc_reward(full_output)
                samples.append({
                    'output': full_output,
                    'tokens': tokens.tolist(),
                    'log_probs': log_probs,
                    'reward': reward,
                })
                epoch_rewards.append(reward)
                total_count += 1
                if reward >= 0.6:
                    valid_count += 1

            # Compute advantages
            rewards = [s['reward'] for s in samples]
            advantages = compute_grpo_advantages(rewards)

            # Compute loss and update for each sample
            for sample, advantage in zip(samples, advantages):
                if abs(advantage) < 0.01:
                    continue  # Skip neutral samples

                # Define loss function for nn.value_and_grad
                def loss_fn(model):
                    return compute_grpo_loss(
                        model, tokenizer,
                        prompt_tokens,
                        sample['tokens'],
                        advantage,
                    )

                # Compute gradients using nn.value_and_grad
                loss, grads = nn.value_and_grad(model, loss_fn)(model)

                # Update with optimizer
                optimizer.update(model, grads)
                mx.eval(model.parameters())

                epoch_losses.append(loss.item())

        elapsed = time.time() - t0
        mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        validity = valid_count / total_count if total_count > 0 else 0

        print(f"Epoch {epoch}: reward={mean_reward:.3f}, validity={validity:.1%}, "
              f"loss={mean_loss:.4f}, time={elapsed:.1f}s")

    print("-" * 60)
    print("Training complete!")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final_rewards = []
    final_valid = 0

    for task in TRAINING_TASKS[:3]:
        prompt = create_few_shot_prompt(task)
        for _ in range(2):
            output, _, _ = generate_with_logprobs(
                model, tokenizer, prompt,
                max_tokens=config.max_tokens,
                temperature=0.7,
            )
            full_output = '{"root_state":' + output
            if '\n' in full_output:
                full_output = full_output.split('\n')[0]

            reward, _ = compute_sc_reward(full_output)
            final_rewards.append(reward)
            if reward >= 0.6:
                final_valid += 1
            print(f"  Output: {full_output[:80]}...")
            print(f"  Reward: {reward:.2f}")

    final_validity = final_valid / len(final_rewards) if final_rewards else 0
    final_mean = sum(final_rewards) / len(final_rewards) if final_rewards else 0

    print(f"\nFinal: validity={final_validity:.1%}, reward={final_mean:.3f}")

    return {
        'final_validity': final_validity,
        'final_reward': final_mean,
    }


if __name__ == "__main__":
    run_grpo_training()
