"""
GRPO Trainer: Group Relative Policy Optimization for SC Generation

Algorithm:
1. Generate N=8 samples per prompt
2. Score each with sc_reward()
3. Compute advantages relative to group mean
4. Update LoRA weights toward higher-reward samples

Based on: "GRPO: Group Relative Policy Optimization"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from pathlib import Path
import json
import time
import random
import math

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load, generate
    from mlx_lm.tuner import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from .lora_config import FullConfig, LoRAConfig, get_default_config
from .grpo_reward import SCRewardFunction, sc_reward, batch_reward


@dataclass
class GRPOConfig:
    """GRPO-specific configuration."""
    # Sampling
    n_samples: int = 8           # Samples per prompt
    temperature: float = 0.8     # Generation temperature
    max_tokens: int = 512        # Max tokens per sample

    # GRPO parameters
    beta: float = 0.1            # KL penalty coefficient
    clip_ratio: float = 0.2      # PPO-style clipping

    # Training
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 4          # Prompts per batch

    # Logging
    log_interval: int = 10


@dataclass
class GRPOMetrics:
    """Metrics from a GRPO training step."""
    epoch: int
    step: int
    reward_mean: float
    reward_std: float
    advantage_mean: float
    loss: float
    validity_rate: float
    kl_divergence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "advantage_mean": self.advantage_mean,
            "loss": self.loss,
            "validity_rate": self.validity_rate,
            "kl_divergence": self.kl_divergence,
        }

    def __str__(self) -> str:
        return (f"GRPO_LORA: epoch={self.epoch}, validity={self.validity_rate:.1%}, "
                f"reward_mean={self.reward_mean:.3f}, loss={self.loss:.4f}")


@dataclass
class GRPOResult:
    """Result from GRPO training run."""
    final_validity: float
    baseline_validity: float
    improvement: float
    best_reward: float
    total_epochs: int
    total_steps: int
    training_time_seconds: float
    metrics_history: List[GRPOMetrics]
    adapter_path: Path

    def summary(self) -> str:
        return (f"GRPO Training Complete:\n"
                f"  Baseline validity: {self.baseline_validity:.1%}\n"
                f"  Final validity: {self.final_validity:.1%}\n"
                f"  Improvement: {self.improvement:+.1f}pp\n"
                f"  Best reward: {self.best_reward:.3f}\n"
                f"  Epochs: {self.total_epochs}\n"
                f"  Time: {self.training_time_seconds:.1f}s")


class GRPOTrainer:
    """
    GRPO Trainer for SC-conditioned LoRA fine-tuning.

    Uses group relative policy optimization to update
    LoRA weights toward higher-reward SC generations.
    """

    def __init__(
        self,
        lora_config: LoRAConfig,
        grpo_config: GRPOConfig,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        self.lora_config = lora_config
        self.grpo_config = grpo_config
        self.model_name = model_name

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.reward_fn = SCRewardFunction()

        self.metrics_history: List[GRPOMetrics] = []

    def load_model(self):
        """Load model and apply LoRA."""
        if not MLX_AVAILABLE:
            print("Warning: MLX not available")
            return

        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)

        # Apply LoRA
        print(f"Applying LoRA (rank={self.lora_config.rank})")
        linear_to_lora_layers(
            self.model,
            self.lora_config.rank,
            self.lora_config.target_modules,
        )

        # Setup optimizer
        self.optimizer = optim.Adam(learning_rate=self.grpo_config.learning_rate)

        # Count parameters
        trainable = sum(p.size for p in self.model.trainable_parameters().values())
        total = sum(p.size for p in self.model.parameters().values())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(
        self,
        prompts: List[str],
        output_dir: Optional[Path] = None,
    ) -> GRPOResult:
        """
        Run GRPO training.

        Args:
            prompts: List of SC generation prompts
            output_dir: Directory to save adapter

        Returns:
            GRPOResult with training metrics
        """
        output_dir = output_dir or Path("./grpo_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        # Load model if needed
        if self.model is None:
            self.load_model()

        # Evaluate baseline
        print("Evaluating baseline...")
        baseline_validity = self._evaluate_validity(prompts[:20])
        print(f"Baseline validity: {baseline_validity:.1%}")

        # Training loop
        best_reward = 0.0
        step = 0

        for epoch in range(self.grpo_config.epochs):
            random.shuffle(prompts)

            for batch_start in range(0, len(prompts), self.grpo_config.batch_size):
                batch_prompts = prompts[batch_start:batch_start + self.grpo_config.batch_size]

                # GRPO step
                metrics = self._grpo_step(batch_prompts, epoch, step)
                self.metrics_history.append(metrics)

                if metrics.reward_mean > best_reward:
                    best_reward = metrics.reward_mean

                # Log
                if step % self.grpo_config.log_interval == 0:
                    print(metrics)

                step += 1

            # Save checkpoint after each epoch
            self._save_adapter(output_dir / f"adapter_epoch{epoch}.safetensors")

        # Final evaluation
        print("\nEvaluating final model...")
        final_validity = self._evaluate_validity(prompts[:20])

        # Save final adapter
        adapter_path = output_dir / "adapter_final.safetensors"
        self._save_adapter(adapter_path)

        training_time = time.time() - t0

        return GRPOResult(
            final_validity=final_validity,
            baseline_validity=baseline_validity,
            improvement=(final_validity - baseline_validity) * 100,
            best_reward=best_reward,
            total_epochs=self.grpo_config.epochs,
            total_steps=step,
            training_time_seconds=training_time,
            metrics_history=self.metrics_history,
            adapter_path=adapter_path,
        )

    def _grpo_step(
        self,
        prompts: List[str],
        epoch: int,
        step: int,
    ) -> GRPOMetrics:
        """
        Single GRPO training step.

        1. Generate N samples per prompt
        2. Compute rewards
        3. Compute group-relative advantages
        4. Update with policy gradient
        """
        all_rewards = []
        all_advantages = []
        valid_count = 0
        total_samples = 0

        for prompt in prompts:
            # Generate N samples
            samples = self._generate_samples(prompt)

            # Compute rewards
            rewards = [self.reward_fn.compute_reward(s)[0] for s in samples]
            all_rewards.extend(rewards)

            # Count valid (reward > 0.5 means mostly valid)
            valid_count += sum(1 for r in rewards if r > 0.5)
            total_samples += len(rewards)

            # Compute advantages (group-relative)
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            advantages = [r - mean_reward for r in rewards]
            all_advantages.extend(advantages)

        # Compute metrics
        reward_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        reward_std = self._std(all_rewards)
        advantage_mean = sum(all_advantages) / len(all_advantages) if all_advantages else 0
        validity_rate = valid_count / total_samples if total_samples else 0

        # Compute loss (simplified - actual GRPO uses policy gradient)
        loss = self._compute_grpo_loss(all_rewards, all_advantages)

        # Update model (mock for now without MLX tensors)
        if MLX_AVAILABLE and self.model is not None:
            self._update_weights(loss)

        return GRPOMetrics(
            epoch=epoch,
            step=step,
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=advantage_mean,
            loss=loss,
            validity_rate=validity_rate,
        )

    def _generate_samples(self, prompt: str) -> List[str]:
        """Generate N samples for a prompt."""
        samples = []

        if MLX_AVAILABLE and self.model is not None:
            for _ in range(self.grpo_config.n_samples):
                # Format prompt
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate
                output = generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted,
                    max_tokens=self.grpo_config.max_tokens,
                    temp=self.grpo_config.temperature,
                )
                samples.append(output)
        else:
            # Mock generation for testing
            samples = self._mock_generate(prompt)

        return samples

    def _mock_generate(self, prompt: str) -> List[str]:
        """Mock generation for testing without MLX."""
        samples = []
        for _ in range(self.grpo_config.n_samples):
            # Generate with varying validity
            if random.random() < 0.5:  # 50% valid baseline
                sc = {
                    "name": "Generated",
                    "root_state": {
                        "label": "__root__",
                        "type": 2,
                        "children": [
                            {"label": "State1", "type": 1, "is_initial": True},
                            {"label": "State2", "type": 1},
                        ]
                    },
                    "transitions": [
                        {"from": ["State1"], "to": ["State2"], "event": "EVENT"}
                    ]
                }
                samples.append(json.dumps(sc))
            else:
                samples.append("Invalid output {broken")

        return samples

    def _compute_grpo_loss(
        self,
        rewards: List[float],
        advantages: List[float],
    ) -> float:
        """
        Compute GRPO loss.

        Loss = -E[advantage * log_prob] + beta * KL
        """
        if not advantages:
            return 0.0

        # Simplified loss: negative mean advantage-weighted reward
        weighted = [a * r for a, r in zip(advantages, rewards)]
        loss = -sum(weighted) / len(weighted)

        return loss

    def _update_weights(self, loss: float):
        """Update LoRA weights based on loss."""
        if not MLX_AVAILABLE or self.model is None:
            return

        # In practice, this would use mx.grad and optimizer.update
        # Simplified placeholder
        pass

    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _evaluate_validity(self, prompts: List[str]) -> float:
        """Evaluate validity rate on prompts."""
        valid = 0
        total = 0

        for prompt in prompts:
            samples = self._generate_samples(prompt)
            for sample in samples:
                reward, _ = self.reward_fn.compute_reward(sample)
                if reward > 0.5:
                    valid += 1
                total += 1

        return valid / total if total else 0

    def _save_adapter(self, path: Path):
        """Save LoRA adapter."""
        if MLX_AVAILABLE and self.model is not None:
            lora_params = dict(self.model.trainable_parameters())
            mx.savez(str(path), **lora_params)
        else:
            # Create placeholder
            path.touch()


# SC generation prompts for training
SC_PROMPTS = [
    # Simple (level 1)
    "Generate a statechart JSON for a light switch with On and Off states",
    "Generate a statechart JSON for a door that can be Open or Closed",
    "Generate a statechart JSON for a simple toggle button",
    "Generate a statechart JSON for a power button with Standby and Active states",

    # Medium (level 2)
    "Generate a statechart JSON for a traffic light with Red, Yellow, Green states",
    "Generate a statechart JSON for a media player with Play, Pause, Stop",
    "Generate a statechart JSON for a door lock with Locked and Unlocked states",
    "Generate a statechart JSON for a fan with Off, Low, Medium, High speeds",

    # Complex (level 3)
    "Generate a statechart JSON for user authentication with login, logout, and session timeout",
    "Generate a statechart JSON for an order processing workflow with pending, processing, shipped, delivered",
    "Generate a statechart JSON for a vending machine with idle, selecting, dispensing, returning change",
    "Generate a statechart JSON for an elevator with floors 1-3 and moving/stopped states",

    # Advanced (level 4)
    "Generate a statechart JSON for a smart thermostat with heating, cooling, idle modes and temperature guards",
    "Generate a statechart JSON for a game character with idle, walking, running, jumping, attacking states",
    "Generate a statechart JSON for a washing machine cycle with fill, wash, rinse, spin, drain phases",
    "Generate a statechart JSON for a microwave with cooking modes and timer states",
]


def train_grpo(
    prompts: Optional[List[str]] = None,
    lora_config: Optional[LoRAConfig] = None,
    grpo_config: Optional[GRPOConfig] = None,
    output_dir: Optional[Path] = None,
) -> GRPOResult:
    """
    Convenience function to run GRPO training.

    Args:
        prompts: Training prompts (default: SC_PROMPTS)
        lora_config: LoRA configuration
        grpo_config: GRPO configuration
        output_dir: Output directory

    Returns:
        GRPOResult
    """
    prompts = prompts or SC_PROMPTS
    lora_config = lora_config or LoRAConfig(
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    grpo_config = grpo_config or GRPOConfig()
    output_dir = output_dir or Path("./grpo_outputs")

    trainer = GRPOTrainer(lora_config, grpo_config)
    return trainer.train(prompts, output_dir)


def demo():
    """Demonstrate GRPO training."""
    print("=" * 60)
    print("GRPO TRAINER: SC-Conditioned LoRA Fine-tuning")
    print("=" * 60)

    # Quick config for demo
    lora_config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    grpo_config = GRPOConfig(
        n_samples=4,
        epochs=2,
        batch_size=2,
        log_interval=1,
    )

    # Use subset of prompts
    prompts = SC_PROMPTS[:8]

    print(f"\nConfig:")
    print(f"  LoRA rank: {lora_config.rank}")
    print(f"  Samples/prompt: {grpo_config.n_samples}")
    print(f"  Epochs: {grpo_config.epochs}")
    print(f"  Prompts: {len(prompts)}")

    # Train
    result = train_grpo(
        prompts=prompts,
        lora_config=lora_config,
        grpo_config=grpo_config,
        output_dir=Path("/tmp/grpo_demo"),
    )

    print("\n" + result.summary())

    return result


if __name__ == "__main__":
    demo()
