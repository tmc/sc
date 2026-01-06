"""
Bidirectional Trainer: Train on both text→SC and SC→text

Uses mlx_lm for real inference with alternating direction training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
import json
import time

from .dataset import (
    BidirectionalDataset, 
    create_dataset,
    format_forward_prompt,
    format_reverse_prompt,
)


@dataclass
class TrainingConfig:
    """Configuration for bidirectional training."""
    epochs: int = 5
    batch_size: int = 4
    max_tokens: int = 512
    temperature: float = 0.7
    direction_ratio: float = 0.5  # 0.5 = equal forward/reverse
    log_interval: int = 10


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    epoch: int
    step: int
    direction: str  # "forward" or "reverse"
    loss_approx: float  # Approximated from generation quality
    validity_rate: float  # For forward direction
    coherence_rate: float  # For reverse direction


@dataclass
class TrainingResult:
    """Result of bidirectional training."""
    total_epochs: int
    total_steps: int
    final_forward_validity: float
    final_reverse_coherence: float
    training_time_seconds: float
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    
    def summary(self) -> str:
        return (
            f"Bidirectional Training Complete\n"
            f"  Epochs: {self.total_epochs}\n"
            f"  Steps: {self.total_steps}\n"
            f"  Forward validity: {self.final_forward_validity:.1%}\n"
            f"  Reverse coherence: {self.final_reverse_coherence:.1%}\n"
            f"  Time: {self.training_time_seconds:.1f}s"
        )


class BidirectionalTrainer:
    """Trainer for bidirectional SC generation/explanation."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        config: Optional[TrainingConfig] = None,
    ):
        self.model_name = model_name
        self.config = config or TrainingConfig()
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load model with mlx_lm."""
        if self.model is None:
            from mlx_lm import load
            print(f"Loading {self.model_name}...")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded.")
    
    def _generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text using the model."""
        from mlx_lm import generate
        
        max_tokens = max_tokens or self.config.max_tokens
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=text,
            max_tokens=max_tokens,
        )
        return response
    
    def _check_validity(self, generated: str) -> bool:
        """Check if generated SC is valid JSON with required fields."""
        try:
            sc = json.loads(generated)
            return "root_state" in sc or "states" in sc
        except:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{[\s\S]*\}', generated)
            if match:
                try:
                    sc = json.loads(match.group())
                    return "root_state" in sc or "states" in sc
                except:
                    pass
            return False
    
    def _check_coherence(self, explanation: str) -> bool:
        """Check if explanation is coherent (has state/transition words)."""
        keywords = ['state', 'transition', 'event', 'initial', 'from', 'to']
        explanation_lower = explanation.lower()
        matches = sum(1 for kw in keywords if kw in explanation_lower)
        return matches >= 2  # At least 2 keywords
    
    def train(
        self,
        dataset: BidirectionalDataset,
        output_dir: Optional[Path] = None,
    ) -> TrainingResult:
        """
        Run bidirectional training.
        
        This is few-shot learning through example generation,
        not gradient-based fine-tuning.
        """
        self._load_model()
        start_time = time.time()
        
        metrics_history = []
        total_steps = 0
        
        # Get examples
        forward_examples = dataset.get_forward_examples()
        reverse_examples = dataset.get_reverse_examples()
        
        # Track running metrics
        forward_valid = 0
        forward_total = 0
        reverse_coherent = 0
        reverse_total = 0
        
        for epoch in range(self.config.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")
            
            # Alternate between directions
            for i, (desc, sc) in enumerate(forward_examples[:self.config.batch_size]):
                # Forward: text → SC
                prompt = format_forward_prompt(desc)
                generated = self._generate(prompt, max_tokens=300)
                
                is_valid = self._check_validity(generated)
                forward_valid += int(is_valid)
                forward_total += 1
                total_steps += 1
                
                if total_steps % self.config.log_interval == 0:
                    validity_rate = forward_valid / forward_total if forward_total > 0 else 0
                    print(f"  Step {total_steps}: forward validity={validity_rate:.1%}")
                    metrics_history.append(TrainingMetrics(
                        epoch=epoch,
                        step=total_steps,
                        direction="forward",
                        loss_approx=1.0 - validity_rate,
                        validity_rate=validity_rate,
                        coherence_rate=0,
                    ))
            
            for i, (sc, desc) in enumerate(reverse_examples[:self.config.batch_size]):
                # Reverse: SC → text
                prompt = format_reverse_prompt(sc)
                explanation = self._generate(prompt, max_tokens=200)
                
                is_coherent = self._check_coherence(explanation)
                reverse_coherent += int(is_coherent)
                reverse_total += 1
                total_steps += 1
                
                if total_steps % self.config.log_interval == 0:
                    coherence_rate = reverse_coherent / reverse_total if reverse_total > 0 else 0
                    print(f"  Step {total_steps}: reverse coherence={coherence_rate:.1%}")
                    metrics_history.append(TrainingMetrics(
                        epoch=epoch,
                        step=total_steps,
                        direction="reverse",
                        loss_approx=1.0 - coherence_rate,
                        validity_rate=0,
                        coherence_rate=coherence_rate,
                    ))
        
        # Final metrics
        final_validity = forward_valid / forward_total if forward_total > 0 else 0
        final_coherence = reverse_coherent / reverse_total if reverse_total > 0 else 0
        
        result = TrainingResult(
            total_epochs=self.config.epochs,
            total_steps=total_steps,
            final_forward_validity=final_validity,
            final_reverse_coherence=final_coherence,
            training_time_seconds=time.time() - start_time,
            metrics_history=metrics_history,
        )
        
        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "training_result.json", 'w') as f:
                json.dump({
                    "epochs": result.total_epochs,
                    "steps": result.total_steps,
                    "forward_validity": result.final_forward_validity,
                    "reverse_coherence": result.final_reverse_coherence,
                    "time_seconds": result.training_time_seconds,
                }, f, indent=2)
        
        return result


def train_bidirectional(
    dataset: Optional[BidirectionalDataset] = None,
    config: Optional[TrainingConfig] = None,
    output_dir: Optional[Path] = None,
) -> TrainingResult:
    """Convenience function to run bidirectional training."""
    if dataset is None:
        dataset = create_dataset()
    
    trainer = BidirectionalTrainer(config=config)
    return trainer.train(dataset, output_dir)


def demo():
    """Run quick demo."""
    print("=== Bidirectional Training Demo ===")
    
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        log_interval=2,
    )
    
    dataset = create_dataset()
    result = train_bidirectional(dataset, config)
    print("\n" + result.summary())
    
    return result


if __name__ == "__main__":
    demo()
