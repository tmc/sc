#!/usr/bin/env python3
"""
Faithful TRM with ACT (Adaptive Computation Time).

Implements Samsung's Q-learning based halting mechanism for adaptive iteration counts.

Key features:
1. Q-head: Predicts halt/continue logits for each sequence
2. Halting policy: Learned via Q-learning (continue bootstraps from next step)
3. Exploration: During training, randomly halt with configurable probability
4. Per-sequence halting: Different sequences can halt at different H steps

Reference: github.com/SamsungSAILMontreal/TinyRecursiveModels
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math


def truncated_normal(shape: tuple, std: float = 1.0) -> mx.array:
    """Approximate truncated normal."""
    lower, upper = -2 * std, 2 * std
    samples = mx.clip(mx.random.normal(shape=shape) * std, lower, upper)
    return samples


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """RMS normalization."""
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


def log_stablemax(x: mx.array, axis: int = -1) -> mx.array:
    """Log StableMax for numerical stability."""
    s_x = mx.where(x < 0, 1.0 / (1.0 - x + 1e-10), x + 1.0)
    log_s_x = mx.log(s_x + 1e-10)
    log_sum = mx.log(mx.sum(s_x, axis=axis, keepdims=True) + 1e-10)
    return log_s_x - log_sum


@dataclass
class ACTConfig:
    """Config for TRM with ACT."""
    vocab_size: int = 10
    hidden_size: int = 128
    expansion: float = 2.0
    max_H_cycles: int = 5  # Maximum H cycles (ACT can halt earlier)
    L_cycles: int = 6
    L_layers: int = 2
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 81
    use_mlp_t: bool = True
    num_heads: int = 4
    init_std: float = 1.0
    # ACT-specific
    halt_exploration_prob: float = 0.1  # Random halt during training
    q_init_bias: float = -5.0  # Initial bias for Q-head (encourages continuing)
    discount_factor: float = 0.99  # Q-learning discount


class SwiGLU(nn.Module):
    """SwiGLU activation."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = ((int(expansion * hidden_size * 2 / 3) + 255) // 256) * 256
        if inter == 0:
            inter = max(64, int(expansion * hidden_size))
        self.gate_up = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down = nn.Linear(inter, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down(nn.silu(gate) * up)


class ReasoningBlock(nn.Module):
    """Single reasoning block."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        if config.use_mlp_t:
            self.mlp_t = SwiGLU(config.max_seq_len, config.expansion)
        else:
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.mlp = SwiGLU(config.hidden_size, config.expansion)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        if self.config.use_mlp_t:
            h = mx.transpose(x, (0, 2, 1))
            h = self.mlp_t(h)
            h = mx.transpose(h, (0, 2, 1))
            x = rms_norm(x + h, self.config.rms_norm_eps)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            head_dim = D // self.config.num_heads
            q = q.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.config.num_heads, head_dim).transpose(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
            h = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
            h = self.o_proj(h)
            x = rms_norm(x + h, self.config.rms_norm_eps)

        h = self.mlp(x)
        x = rms_norm(x + h, self.config.rms_norm_eps)
        return x


class ReasoningModule(nn.Module):
    """L_level network."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = [ReasoningBlock(config) for _ in range(config.L_layers)]

    def __call__(self, hidden_states: mx.array, input_injection: mx.array) -> mx.array:
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer(x)
        return x


class FaithfulTRMACT(nn.Module):
    """
    Faithful TRM with Adaptive Computation Time (ACT).

    Samsung's ACT uses Q-learning to learn when to halt:
    - Q(state, halt) estimates expected future reward if halting now
    - Q(state, continue) estimates expected future reward if continuing
    - During training, explores with probability halt_exploration_prob
    - Q-values are bootstrapped: Q(continue) = reward + gamma * max(Q_next)
    """

    def __init__(self, config: Optional[ACTConfig] = None):
        super().__init__()
        self.config = config or ACTConfig()

        # Embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.embed_pos = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)

        # Shared L_level network
        self.L_level = ReasoningModule(self.config)

        # Initial states
        self.H_init = truncated_normal((self.config.hidden_size,), std=self.config.init_std)
        self.L_init = truncated_normal((self.config.hidden_size,), std=self.config.init_std)

        # Output head
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Q-head for ACT: predicts [halt_value, continue_value] per sequence
        # Uses pooled representation from z_H
        self.q_head = nn.Linear(self.config.hidden_size, 2, bias=True)

    def _run_h_cycle(
        self,
        z_H: mx.array,
        z_L: mx.array,
        input_embed: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Run one complete H cycle (L_cycles of L_level updates)."""
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embed)
        z_H = self.L_level(z_H, z_L)
        return z_H, z_L

    def _get_q_values(self, z_H: mx.array) -> mx.array:
        """Get Q-values [halt, continue] from z_H state."""
        # Pool z_H across sequence: mean pooling
        pooled = mx.mean(z_H, axis=1)  # [B, D]
        q_values = self.q_head(pooled)  # [B, 2]
        return q_values

    def __call__(
        self,
        input_ids: mx.array,
        training: bool = False,
        return_steps: bool = False
    ) -> dict:
        """
        Forward pass with ACT.

        Args:
            input_ids: [B, 81] input puzzle
            training: If True, uses exploration for halting decisions
            return_steps: If True, return number of steps taken per sequence

        Returns:
            dict with 'logits', 'q_values' (per step), 'steps' (if return_steps)
        """
        B, L = input_ids.shape

        # Embeddings
        positions = mx.arange(L)
        input_embed = self.embed_tokens(input_ids) + self.embed_pos(positions)
        input_embed = self.embed_scale * input_embed

        # Initialize states
        z_H = mx.broadcast_to(self.H_init, (B, L, self.config.hidden_size))
        z_L = mx.broadcast_to(self.L_init, (B, L, self.config.hidden_size))

        # Track halting
        halted = mx.zeros((B,), dtype=mx.bool_)  # Which sequences have halted
        steps = mx.zeros((B,), dtype=mx.int32)  # Steps taken per sequence
        all_q_values = []  # Q-values at each step

        # Final outputs (will be filled as sequences halt)
        final_z_H = mx.zeros((B, L, self.config.hidden_size))

        for h_step in range(self.config.max_H_cycles):
            # Run H cycle
            z_H_new, z_L_new = self._run_h_cycle(z_H, z_L, input_embed)

            # Get Q-values for halting decision
            q_values = self._get_q_values(z_H_new)  # [B, 2]: [halt, continue]
            all_q_values.append(q_values)

            # Decide whether to halt
            if h_step == self.config.max_H_cycles - 1:
                # Force halt on last step
                should_halt = mx.ones((B,), dtype=mx.bool_)
            else:
                # Halt if Q(halt) > Q(continue)
                should_halt = q_values[:, 0] > q_values[:, 1]

                if training:
                    # Exploration: randomly flip some decisions
                    explore_mask = mx.random.uniform(shape=(B,)) < self.config.halt_exploration_prob
                    should_halt = mx.where(explore_mask, ~should_halt, should_halt)

            # Update for sequences that should halt now (and haven't already)
            newly_halted = should_halt & ~halted

            # Store final z_H for newly halted sequences
            # Use where to update only newly halted sequences
            final_z_H = mx.where(
                newly_halted[:, None, None],
                z_H_new,
                final_z_H
            )

            # Update steps for active sequences
            steps = mx.where(~halted, steps + 1, steps)

            # Update halted mask
            halted = halted | should_halt

            # Check if all sequences have halted
            if mx.all(halted):
                break

            # Update states for continuing sequences
            z_H = mx.where(halted[:, None, None], z_H, z_H_new)
            z_L = mx.where(halted[:, None, None], z_L, z_L_new)

        # Output logits from final states
        logits = self.lm_head(final_z_H)

        result = {
            'logits': logits,
            'q_values': mx.stack(all_q_values, axis=1),  # [B, num_steps, 2]
        }
        if return_steps:
            result['steps'] = steps

        return result

    def solve(self, puzzle: mx.array, temperature: float = 0.0, **kwargs) -> dict:
        """Solve a puzzle."""
        result = self(puzzle, training=False, return_steps=True)
        logits = result['logits']

        if temperature > 0:
            probs = mx.softmax(logits / temperature, axis=-1)
            predictions = mx.argmax(probs, axis=-1)
        else:
            predictions = mx.argmax(logits, axis=-1)

        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)

        return {
            "predictions": predictions,
            "logits": logits,
            "steps": result['steps']
        }

    def loss(
        self,
        puzzle: mx.array,
        solution: mx.array,
        **kwargs
    ) -> Tuple[mx.array, dict]:
        """
        Compute loss with ACT Q-learning.

        Loss has two components:
        1. Task loss: cross-entropy on predictions
        2. Q-loss: TD error for Q-learning halting
        """
        result = self(puzzle, training=True, return_steps=True)
        logits = result['logits']
        q_values = result['q_values']  # [B, num_steps, 2]
        steps = result['steps']

        # Task loss (cross-entropy)
        empty_mask = (puzzle == 0).astype(mx.float32)
        log_probs = log_stablemax(logits, axis=-1)
        target_expanded = solution[:, :, None]
        correct_log_probs = mx.take_along_axis(log_probs, target_expanded, axis=-1).squeeze(-1)
        task_loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)

        # Q-learning loss
        # Reward: based on prediction accuracy at halt time
        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean((predictions == solution).astype(mx.float32) * empty_mask, axis=-1)
        reward = accuracy  # [B]

        # TD targets for Q-values
        # For halting: target = reward
        # For continuing: target = reward + gamma * max(Q_next)
        B, num_steps, _ = q_values.shape
        q_loss = mx.array(0.0)

        for t in range(num_steps):
            q_t = q_values[:, t, :]  # [B, 2]

            # Determine if this was the halt step for each sequence
            is_halt_step = (steps == t + 1)  # steps is 1-indexed

            # Target for halt action
            halt_target = reward

            # Target for continue action (bootstrap from next step or use reward if last)
            if t < num_steps - 1:
                q_next = q_values[:, t + 1, :]
                continue_target = reward + self.config.discount_factor * mx.max(q_next, axis=-1)
            else:
                continue_target = reward

            # TD error
            halt_error = (q_t[:, 0] - halt_target) ** 2
            continue_error = (q_t[:, 1] - continue_target) ** 2

            # Only count loss for steps that were actually taken
            step_mask = (t < steps).astype(mx.float32)
            q_loss = q_loss + mx.mean((halt_error + continue_error) * step_mask)

        q_loss = q_loss / num_steps

        # Combined loss
        total_loss = task_loss + 0.1 * q_loss  # Weight Q-loss lower

        # Metrics
        cell_acc = mx.sum((predictions == solution).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        avg_steps = mx.mean(steps.astype(mx.float32))

        return total_loss, {
            "cell_accuracy": cell_acc,
            "task_loss": task_loss,
            "q_loss": q_loss,
            "avg_steps": avg_steps
        }


def create_model(config: Optional[ACTConfig] = None) -> FaithfulTRMACT:
    """Factory function."""
    return FaithfulTRMACT(config)


if __name__ == "__main__":
    print("Testing Faithful TRM with ACT...")

    config = ACTConfig(max_H_cycles=5, L_cycles=6, use_mlp_t=True)
    model = create_model(config)

    batch = mx.zeros((4, 81), dtype=mx.int32)
    solution = mx.ones((4, 81), dtype=mx.int32)

    print(f"\nConfig:")
    print(f"  max_H_cycles={config.max_H_cycles}")
    print(f"  L_cycles={config.L_cycles}")
    print(f"  halt_exploration_prob={config.halt_exploration_prob}")

    # Test forward
    result = model(batch, training=False, return_steps=True)
    print(f"\nForward pass:")
    print(f"  Input: {batch.shape}")
    print(f"  Logits: {result['logits'].shape}")
    print(f"  Q-values shape: {result['q_values'].shape}")
    print(f"  Steps taken: {result['steps'].tolist()}")

    # Test loss
    loss, metrics = model.loss(batch, solution)
    print(f"\nLoss computation:")
    print(f"  Total loss: {float(loss):.4f}")
    print(f"  Task loss: {float(metrics['task_loss']):.4f}")
    print(f"  Q loss: {float(metrics['q_loss']):.4f}")
    print(f"  Avg steps: {float(metrics['avg_steps']):.2f}")
    print(f"  Cell accuracy: {float(metrics['cell_accuracy']):.1%}")

    # Test solve
    solve_result = model.solve(batch)
    print(f"\nSolve:")
    print(f"  Predictions: {solve_result['predictions'].shape}")
    print(f"  Steps: {solve_result['steps'].tolist()}")

    print("\n  ACT enables adaptive computation - harder puzzles get more iterations!")
