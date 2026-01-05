"""
exp_sc_lora_grpo: GRPO Training for SC-Conditioned Generation

Train LoRA adapters using Group Relative Policy Optimization (GRPO)
with statechart validity as the reward signal.

Key Insight: Use the statechart executor as the reward function.
No separate reward model needed - SC validity IS the reward.

Meta-Insight: The reward function itself is modeled as a statechart!
(reward_statechart.json) - Self-referential: SC validates SC using SC.

Components:
- sc_reward.py: Reward function computing SC validity score
- grpo_trainer.py: GRPO training loop with LoRA
- reward_statechart.json: Reward logic as inspectable statechart

Usage:
    python -m experiments.exp_sc_lora_grpo.grpo_trainer
"""

from .sc_reward import (
    compute_sc_reward,
    compute_batch_rewards,
    compute_grpo_advantages,
    RewardBreakdown,
)

from .grpo_trainer import (
    GRPOTrainer,
    GRPOConfig,
    TrainingMetrics,
    generate_training_prompts,
    run_grpo_training,
)

__all__ = [
    'compute_sc_reward',
    'compute_batch_rewards',
    'compute_grpo_advantages',
    'RewardBreakdown',
    'GRPOTrainer',
    'GRPOConfig',
    'TrainingMetrics',
    'generate_training_prompts',
    'run_grpo_training',
]
