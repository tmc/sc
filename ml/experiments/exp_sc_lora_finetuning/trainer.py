"""
LoRA Trainer for Statechart Generation

Fine-tunes Qwen models using LoRA adapters on statechart generation tasks.
Uses mlx_lm for Apple Silicon optimization.

Training data format:
{
    "prompt": "Generate a statechart for: <description>",
    "completion": "<statechart JSON>"
}
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Iterator
from pathlib import Path
import json
import time
import random

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load, generate
    from mlx_lm.tuner import linear_to_lora_layers
    from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
    from mlx_lm.tuner.datasets import load_dataset
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from .lora_config import FullConfig, LoRAConfig, TrainingConfig, get_default_config


@dataclass
class TrainingExample:
    """Single training example."""
    prompt: str
    completion: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class TrainingMetrics:
    """Metrics from a training step or epoch."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class TrainingResult:
    """Result from complete training run."""
    final_loss: float
    best_loss: float
    total_steps: int
    total_epochs: int
    training_time_seconds: float
    adapter_path: Path
    metrics_history: List[TrainingMetrics]

    # Validation metrics
    val_loss: Optional[float] = None
    validity_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "training_time_seconds": self.training_time_seconds,
            "adapter_path": str(self.adapter_path),
            "val_loss": self.val_loss,
            "validity_rate": self.validity_rate,
        }


class StatechartDataset:
    """
    Dataset for statechart generation training.

    Loads from exp_synthetic_sc_dataset or creates synthetic examples.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.max_samples = max_samples
        self.examples: List[TrainingExample] = []

        if data_path and data_path.exists():
            self._load_from_path(data_path)
        else:
            self._create_synthetic_data()

    def _load_from_path(self, path: Path):
        """Load dataset from JSON file or directory."""
        if path.is_file():
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.examples.append(TrainingExample(
                        prompt=item.get("prompt", item.get("input", "")),
                        completion=item.get("completion", item.get("output", "")),
                        metadata=item.get("metadata", {}),
                    ))
        elif path.is_dir():
            for file in path.glob("*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            self.examples.append(TrainingExample(
                                prompt=item.get("prompt", ""),
                                completion=item.get("completion", ""),
                            ))

        if self.max_samples:
            self.examples = self.examples[:self.max_samples]

    def _create_synthetic_data(self):
        """Create synthetic training data for development."""
        templates = [
            {
                "desc": "traffic light controller",
                "sc": {
                    "name": "TrafficLight",
                    "root_state": {
                        "label": "__root__",
                        "type": 2,
                        "children": [
                            {"label": "Red", "type": 1, "is_initial": True},
                            {"label": "Green", "type": 1},
                            {"label": "Yellow", "type": 1},
                        ]
                    },
                    "transitions": [
                        {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
                        {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
                        {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
                    ]
                }
            },
            {
                "desc": "door lock with PIN",
                "sc": {
                    "name": "DoorLock",
                    "root_state": {
                        "label": "__root__",
                        "type": 2,
                        "children": [
                            {"label": "Locked", "type": 1, "is_initial": True},
                            {"label": "Unlocked", "type": 1},
                        ]
                    },
                    "transitions": [
                        {"from": ["Locked"], "to": ["Unlocked"], "event": "CORRECT_PIN"},
                        {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
                    ]
                }
            },
            {
                "desc": "media player",
                "sc": {
                    "name": "MediaPlayer",
                    "root_state": {
                        "label": "__root__",
                        "type": 2,
                        "children": [
                            {"label": "Stopped", "type": 1, "is_initial": True},
                            {"label": "Playing", "type": 1},
                            {"label": "Paused", "type": 1},
                        ]
                    },
                    "transitions": [
                        {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
                        {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                        {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
                        {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
                    ]
                }
            },
        ]

        # Generate variations
        for template in templates:
            for i in range(10):  # 10 variations each
                prompt = f"Generate a statechart JSON for: {template['desc']}"
                if i > 0:
                    prompt += f" (variation {i})"

                self.examples.append(TrainingExample(
                    prompt=prompt,
                    completion=json.dumps(template["sc"], indent=2),
                ))

        if self.max_samples:
            self.examples = self.examples[:self.max_samples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def split(self, train_ratio: float = 0.9) -> Tuple['StatechartDataset', 'StatechartDataset']:
        """Split into train and validation sets."""
        random.shuffle(self.examples)
        split_idx = int(len(self.examples) * train_ratio)

        train_ds = StatechartDataset.__new__(StatechartDataset)
        train_ds.examples = self.examples[:split_idx]
        train_ds.max_samples = None
        train_ds.data_path = None

        val_ds = StatechartDataset.__new__(StatechartDataset)
        val_ds.examples = self.examples[split_idx:]
        val_ds.max_samples = None
        val_ds.data_path = None

        return train_ds, val_ds

    def to_jsonl(self, path: Path):
        """Export dataset to JSONL format for mlx_lm."""
        with open(path, 'w') as f:
            for ex in self.examples:
                f.write(json.dumps({"text": f"{ex.prompt}\n{ex.completion}"}) + "\n")


class LoRATrainer:
    """
    LoRA fine-tuning trainer for statechart generation.

    Uses mlx_lm for efficient training on Apple Silicon.
    """

    def __init__(self, config: FullConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.metrics_history: List[TrainingMetrics] = []

    def load_base_model(self):
        """Load the base model for fine-tuning."""
        if not MLX_AVAILABLE:
            print("Warning: MLX not available, using mock training")
            return

        print(f"Loading base model: {self.config.model.model_name}")
        self.model, self.tokenizer = load(self.config.model.model_name)

        # Apply LoRA layers
        print(f"Applying LoRA with rank={self.config.lora.rank}")
        linear_to_lora_layers(
            self.model,
            self.config.lora.rank,
            self.config.lora.target_modules,
        )

        # Count trainable parameters
        trainable = sum(p.size for n, p in self.model.trainable_parameters().items())
        total = sum(p.size for p in self.model.parameters().values())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(
        self,
        dataset: StatechartDataset,
        output_dir: Optional[Path] = None,
    ) -> TrainingResult:
        """
        Run LoRA fine-tuning.

        Args:
            dataset: Training dataset
            output_dir: Directory to save adapters

        Returns:
            TrainingResult with metrics and adapter path
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        if not MLX_AVAILABLE:
            # Mock training for development
            return self._mock_train(dataset, output_dir)

        # Split dataset
        train_ds, val_ds = dataset.split(self.config.training.train_split)

        # Export to JSONL for mlx_lm
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        train_ds.to_jsonl(train_path)
        val_ds.to_jsonl(val_path)

        # Load base model if not already loaded
        if self.model is None:
            self.load_base_model()

        # Configure training args
        training_args = TrainingArgs(
            batch_size=self.config.training.batch_size,
            iters=len(train_ds) * self.config.training.num_epochs // self.config.training.batch_size,
            val_batches=len(val_ds) // self.config.training.batch_size,
            steps_per_report=self.config.training.logging_steps,
            steps_per_eval=self.config.training.eval_steps,
            steps_per_save=self.config.training.save_steps,
            adapter_file=str(output_dir / "adapters.safetensors"),
            max_seq_length=self.config.model.max_seq_length,
            grad_checkpoint=self.config.training.gradient_checkpointing,
        )

        # Run training
        print("Starting LoRA fine-tuning...")
        mlx_train(
            self.model,
            self.tokenizer,
            training_args,
            train_dataset=str(train_path),
            val_dataset=str(val_path),
        )

        training_time = time.time() - t0

        # Load best checkpoint
        adapter_path = output_dir / "adapters.safetensors"

        return TrainingResult(
            final_loss=self.metrics_history[-1].loss if self.metrics_history else 0.0,
            best_loss=min(m.loss for m in self.metrics_history) if self.metrics_history else 0.0,
            total_steps=len(self.metrics_history),
            total_epochs=self.config.training.num_epochs,
            training_time_seconds=training_time,
            adapter_path=adapter_path,
            metrics_history=self.metrics_history,
        )

    def _mock_train(self, dataset: StatechartDataset, output_dir: Path) -> TrainingResult:
        """Mock training for development without MLX."""
        print("Running mock training (MLX not available)")

        steps = len(dataset) * self.config.training.num_epochs // self.config.training.batch_size
        loss = 2.5

        for step in range(min(steps, 100)):
            loss *= 0.99  # Simulate loss decrease
            self.metrics_history.append(TrainingMetrics(
                step=step,
                epoch=step * self.config.training.batch_size // len(dataset),
                loss=loss,
                learning_rate=self.config.training.learning_rate,
            ))

        adapter_path = output_dir / "adapters.safetensors"
        # Create empty file
        adapter_path.touch()

        return TrainingResult(
            final_loss=loss,
            best_loss=loss,
            total_steps=len(self.metrics_history),
            total_epochs=self.config.training.num_epochs,
            training_time_seconds=5.0,
            adapter_path=adapter_path,
            metrics_history=self.metrics_history,
        )

    def save_adapter(self, path: Path):
        """Save LoRA adapter weights."""
        if self.model is None:
            print("No model loaded")
            return

        if MLX_AVAILABLE:
            # Save only LoRA parameters
            lora_params = {
                k: v for k, v in self.model.trainable_parameters().items()
            }
            mx.savez(str(path), **lora_params)
            print(f"Saved adapter to {path}")

    def load_adapter(self, path: Path):
        """Load LoRA adapter weights."""
        if self.model is None:
            self.load_base_model()

        if MLX_AVAILABLE and path.exists():
            weights = mx.load(str(path))
            self.model.load_weights(list(weights.items()))
            print(f"Loaded adapter from {path}")


def train_lora(
    data_path: Optional[Path] = None,
    config: Optional[FullConfig] = None,
    output_dir: Optional[Path] = None,
) -> TrainingResult:
    """
    Convenience function to run LoRA training.

    Args:
        data_path: Path to training data
        config: Training configuration
        output_dir: Output directory for adapters

    Returns:
        TrainingResult
    """
    config = config or get_default_config()
    output_dir = output_dir or Path("./outputs")

    dataset = StatechartDataset(data_path, max_samples=config.training.max_samples)
    trainer = LoRATrainer(config)
    return trainer.train(dataset, output_dir)


def demo():
    """Demonstrate LoRA training."""
    print("=" * 60)
    print("LoRA TRAINER: Fine-tune Qwen on Statechart Generation")
    print("=" * 60)

    # Use fast config for demo
    config = get_default_config(training_speed="fast")
    config.training.max_samples = 30

    print(f"\nConfig:")
    print(f"  Model: {config.model.model_name}")
    print(f"  LoRA rank: {config.lora.rank}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Max samples: {config.training.max_samples}")

    # Create synthetic dataset
    dataset = StatechartDataset(max_samples=30)
    print(f"\nDataset: {len(dataset)} examples")

    # Train
    output_dir = Path("/tmp/sc_lora_demo")
    result = train_lora(config=config, output_dir=output_dir)

    print(f"\n--- Training Complete ---")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Best loss: {result.best_loss:.4f}")
    print(f"Steps: {result.total_steps}")
    print(f"Time: {result.training_time_seconds:.1f}s")
    print(f"Adapter: {result.adapter_path}")

    return result


if __name__ == "__main__":
    demo()
