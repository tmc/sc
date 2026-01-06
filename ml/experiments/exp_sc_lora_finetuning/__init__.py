"""
exp_sc_lora_finetuning: LoRA Fine-tune Qwen on Statechart Generation

GOAL: Improve statechart generation validity by +15% via LoRA fine-tuning.

Components:
- lora_config.py: LoRA hyperparameters and model configuration
- trainer.py: Fine-tuning loop with mlx_lm
- evaluator.py: Compare base vs fine-tuned validity rates
- grpo_reward.py: SC validation reward function
- grpo_trainer.py: GRPO training loop
- grpo_benchmark.py: Run training and track metrics

Data source: exp_synthetic_sc_dataset output
Model: Qwen2.5-Coder-0.5B-Instruct (or larger variants)
Framework: mlx_lm with LoRA adapters

GRPO Algorithm:
1. Generate N=8 samples per prompt
2. Score each with sc_reward()
3. Compute advantages relative to group mean
4. Update LoRA weights toward higher-reward samples
"""

from .lora_config import (
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)
from .trainer import (
    LoRATrainer,
    TrainingResult,
    train_lora,
)
from .evaluator import (
    ValidityEvaluator,
    EvaluationResult,
    ComparisonResult,
    evaluate_model,
    compare_models,
)
from .grpo_reward import (
    SCRewardFunction,
    RewardBreakdown,
    sc_reward,
    batch_reward,
)
from .grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    GRPOResult,
    GRPOMetrics,
    train_grpo,
    SC_PROMPTS,
)
from .grpo_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    quick_benchmark,
    full_benchmark,
    format_report,
)

__all__ = [
    # Config
    'LoRAConfig',
    'ModelConfig',
    'TrainingConfig',
    'get_default_config',
    # Trainer
    'LoRATrainer',
    'TrainingResult',
    'train_lora',
    # Evaluator
    'ValidityEvaluator',
    'EvaluationResult',
    'ComparisonResult',
    'evaluate_model',
    'compare_models',
    # GRPO Reward
    'SCRewardFunction',
    'RewardBreakdown',
    'sc_reward',
    'batch_reward',
    # GRPO Trainer
    'GRPOConfig',
    'GRPOTrainer',
    'GRPOResult',
    'GRPOMetrics',
    'train_grpo',
    'SC_PROMPTS',
    # GRPO Benchmark
    'BenchmarkConfig',
    'run_benchmark',
    'quick_benchmark',
    'full_benchmark',
    'format_report',
]
