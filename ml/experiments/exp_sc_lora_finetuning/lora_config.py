"""
LoRA Configuration for Statechart Generation Fine-tuning

Defines hyperparameters for:
- LoRA adapter configuration (rank, alpha, dropout)
- Model selection and quantization
- Training parameters (lr, batch size, epochs)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path
import json


class ModelSize(Enum):
    """Available Qwen model sizes."""
    QWEN_0_5B = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    QWEN_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    QWEN_3B = "Qwen/Qwen2.5-Coder-3B-Instruct"
    QWEN_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"


class TargetModules(Enum):
    """Which modules to apply LoRA to."""
    ATTENTION_ONLY = ["q_proj", "v_proj"]
    ATTENTION_KV = ["q_proj", "k_proj", "v_proj"]
    ATTENTION_OUTPUT = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoRAConfig:
    """
    LoRA adapter configuration.

    Key hyperparameters:
    - rank: Dimension of low-rank matrices (higher = more capacity)
    - alpha: Scaling factor (often set to 2*rank)
    - dropout: Regularization dropout rate
    """
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", or "lora_only"

    # Advanced options
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora: bool = False    # Weight-decomposed LoRA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'LoRAConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def small(cls) -> 'LoRAConfig':
        """Small config for quick experiments."""
        return cls(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])

    @classmethod
    def medium(cls) -> 'LoRAConfig':
        """Medium config for balanced training."""
        return cls(rank=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    @classmethod
    def large(cls) -> 'LoRAConfig':
        """Large config for maximum capacity."""
        return cls(rank=64, alpha=128, target_modules=TargetModules.ALL_LINEAR.value)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = ModelSize.QWEN_0_5B.value
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    max_seq_length: int = 2048
    use_flash_attention: bool = True

    # Paths
    base_model_path: Optional[Path] = None
    adapter_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "max_seq_length": self.max_seq_length,
            "use_flash_attention": self.use_flash_attention,
            "base_model_path": str(self.base_model_path) if self.base_model_path else None,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "constant", "linear", "cosine"

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Data
    train_split: float = 0.9
    max_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler": self.lr_scheduler,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "train_split": self.train_split,
            "max_samples": self.max_samples,
        }

    @classmethod
    def fast(cls) -> 'TrainingConfig':
        """Fast config for debugging."""
        return cls(
            batch_size=2,
            num_epochs=1,
            max_samples=100,
            eval_steps=20,
            save_steps=50,
        )

    @classmethod
    def standard(cls) -> 'TrainingConfig':
        """Standard training config."""
        return cls()

    @classmethod
    def thorough(cls) -> 'TrainingConfig':
        """Thorough training for best results."""
        return cls(
            num_epochs=5,
            learning_rate=1e-4,
            eval_steps=50,
            early_stopping_patience=5,
        )


@dataclass
class FullConfig:
    """Complete configuration for LoRA fine-tuning."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "sc_lora_finetune"
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lora": self.lora.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
        }

    def save(self, path: Path):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'FullConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            lora=LoRAConfig.from_dict(data.get("lora", {})),
            model=ModelConfig(**{k: v for k, v in data.get("model", {}).items()
                                if k in ModelConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in data.get("training", {}).items()
                                      if k in TrainingConfig.__dataclass_fields__}),
            experiment_name=data.get("experiment_name", "sc_lora_finetune"),
            output_dir=Path(data.get("output_dir", "./outputs")),
            seed=data.get("seed", 42),
        )


def get_default_config(
    model_size: ModelSize = ModelSize.QWEN_0_5B,
    lora_size: str = "medium",
    training_speed: str = "standard",
) -> FullConfig:
    """
    Get default configuration for common scenarios.

    Args:
        model_size: Which Qwen model to use
        lora_size: "small", "medium", or "large"
        training_speed: "fast", "standard", or "thorough"

    Returns:
        Complete configuration
    """
    # LoRA config
    lora_configs = {
        "small": LoRAConfig.small(),
        "medium": LoRAConfig.medium(),
        "large": LoRAConfig.large(),
    }
    lora = lora_configs.get(lora_size, LoRAConfig.medium())

    # Training config
    training_configs = {
        "fast": TrainingConfig.fast(),
        "standard": TrainingConfig.standard(),
        "thorough": TrainingConfig.thorough(),
    }
    training = training_configs.get(training_speed, TrainingConfig.standard())

    # Model config
    model = ModelConfig(model_name=model_size.value)

    return FullConfig(
        lora=lora,
        model=model,
        training=training,
    )


# Recommended configs for statechart generation
SC_GENERATION_CONFIG = FullConfig(
    lora=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    model=ModelConfig(
        model_name=ModelSize.QWEN_0_5B.value,
        max_seq_length=2048,
    ),
    training=TrainingConfig(
        learning_rate=2e-4,
        batch_size=4,
        num_epochs=3,
        warmup_steps=100,
        gradient_checkpointing=True,
    ),
    experiment_name="sc_generation_lora",
)


def demo():
    """Demonstrate configuration options."""
    print("=" * 60)
    print("LoRA CONFIGURATION for Statechart Generation")
    print("=" * 60)

    # Default config
    print("\n--- Default Config ---")
    config = get_default_config()
    print(f"Model: {config.model.model_name}")
    print(f"LoRA rank: {config.lora.rank}")
    print(f"Learning rate: {config.training.learning_rate}")

    # SC-specific config
    print("\n--- SC Generation Config ---")
    print(f"LoRA rank: {SC_GENERATION_CONFIG.lora.rank}")
    print(f"Target modules: {SC_GENERATION_CONFIG.lora.target_modules}")
    print(f"Epochs: {SC_GENERATION_CONFIG.training.num_epochs}")

    # Config comparison
    print("\n--- LoRA Size Comparison ---")
    for size in ["small", "medium", "large"]:
        cfg = get_default_config(lora_size=size)
        trainable = cfg.lora.rank * len(cfg.lora.target_modules) * 2  # Approximate
        print(f"  {size}: rank={cfg.lora.rank}, ~{trainable}K trainable params")

    return config


if __name__ == "__main__":
    demo()
