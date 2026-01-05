"""
Constrained LoRA Trainer

Extends LoRA training with statechart constraints during training.
The key insight: Apply constraints during training (not just inference)
to shape model's internal representations toward valid syntax.

Key features:
- Statechart-masked loss: Penalize invalid token predictions
- Validity-weighted loss: Higher weight for syntactically important positions
- Generation-time constraints: Full masking during sampling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import time

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    optim = None

try:
    from .lora_trainer import LoRAConfig, LoRATrainer, load_starlark_corpus
    from .starlark_statechart import StarlarkStatechart
    from .constrained_sampler import StatechartConstrainedSampler, SamplingConfig, LogitMasker
    from .token_mapper import TokenMapper
except ImportError:
    from lora_trainer import LoRAConfig, LoRATrainer, load_starlark_corpus
    from starlark_statechart import StarlarkStatechart
    from constrained_sampler import StatechartConstrainedSampler, SamplingConfig, LogitMasker
    from token_mapper import TokenMapper


@dataclass
class ConstrainedLoRAConfig(LoRAConfig):
    """Configuration for constrained LoRA training."""
    # Constraint settings
    constraint_weight: float = 0.5  # Weight for constraint loss term
    mask_invalid_tokens: bool = True  # Mask invalid tokens in loss
    validity_weighted_loss: bool = True  # Weight loss by syntax importance
    track_validity_stats: bool = True  # Track validity during training

    # Generation settings (for evaluation)
    use_constrained_generation: bool = True
    max_gen_tokens: int = 256


class ConstrainedLoRATrainer(LoRATrainer):
    """
    LoRA trainer with statechart constraints during training.

    Extends base LoRA trainer to:
    1. Mask loss for invalid token predictions
    2. Weight loss by syntactic importance
    3. Track validity statistics during training
    """

    def __init__(self, config: ConstrainedLoRAConfig):
        super().__init__(config)
        self.config = config
        self.statechart = StarlarkStatechart()
        self.mapper = None
        self.masker = None
        self.sampler = None

        # Validity tracking
        self.validity_stats = {
            'total_tokens': 0,
            'valid_tokens': 0,
            'constrained_positions': 0,
        }

    def load_model(self, model_path: Optional[str] = None):
        """Load model and initialize constraint components."""
        super().load_model(model_path)

        # Initialize constraint components
        self.mapper = TokenMapper(self.tokenizer)
        self.masker = LogitMasker(self.mapper, self.statechart)
        self.sampler = StatechartConstrainedSampler(
            self.model, self.tokenizer, self.statechart
        )

    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """Execute single training step with constraints."""
        def loss_fn(model):
            logits = model(batch['input_ids'])

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = batch['labels'][:, 1:]

            # Compute validity mask for each position
            validity_mask = None
            if self.config.mask_invalid_tokens:
                validity_mask = self._compute_validity_mask(batch)

            # Base cross-entropy loss
            base_loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction='none',
            )

            # Apply validity weighting
            if validity_mask is not None and self.config.validity_weighted_loss:
                # Higher weight for positions where model predicts invalid tokens
                weights = 1.0 + validity_mask * self.config.constraint_weight
                base_loss = base_loss * weights.reshape(-1)

            return base_loss.mean()

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)

        # Update parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        self.step += 1

        return {
            'loss': float(loss),
            'step': self.step,
        }

    def _compute_validity_mask(self, batch: Dict[str, mx.array]) -> mx.array:
        """
        Compute validity mask for each position in the batch.

        Returns mask where:
        - 0.0 = valid prediction
        - 1.0 = invalid prediction (should be penalized)
        """
        batch_size, seq_len = batch['input_ids'].shape

        # Track validity for each position
        validity = mx.zeros((batch_size, seq_len - 1))

        for b in range(batch_size):
            self.statechart.reset()

            for t in range(seq_len - 1):
                token_id = int(batch['input_ids'][b, t])
                token = self.mapper.get_token_text(token_id)

                # Get valid tokens at this position
                valid_ids = self.masker.get_valid_token_ids()

                # Check if next token is valid
                next_token_id = int(batch['labels'][b, t + 1])
                if next_token_id not in valid_ids and next_token_id != -100:
                    validity = validity.at[b, t].set(1.0)
                    self.validity_stats['constrained_positions'] += 1

                # Update statechart
                self.statechart.transition(token)
                self.validity_stats['total_tokens'] += 1

        return validity

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        use_constraints: bool = None,
    ) -> Tuple[str, Dict]:
        """Generate with optional constraints."""
        max_tokens = max_tokens or self.config.max_gen_tokens
        use_constraints = use_constraints if use_constraints is not None else self.config.use_constrained_generation

        if use_constraints and self.sampler is not None:
            config = SamplingConfig(max_tokens=max_tokens)
            output, stats = self.sampler.generate(prompt, config)
            return output, stats.to_dict()
        else:
            # Unconstrained generation
            return self._generate_unconstrained(prompt, max_tokens)

    def _generate_unconstrained(
        self,
        prompt: str,
        max_tokens: int,
    ) -> Tuple[str, Dict]:
        """Generate without constraints (baseline)."""
        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array([input_ids])

        generated_ids = []
        for _ in range(max_tokens):
            if generated_ids:
                all_ids = mx.concatenate([input_ids, mx.array([generated_ids])], axis=1)
            else:
                all_ids = input_ids

            logits = self.model(all_ids)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            logits = logits[0, -1, :]

            probs = mx.softmax(logits / 0.7)
            next_id = int(mx.random.categorical(mx.log(probs + 1e-10)))

            generated_ids.append(next_id)

            next_token = self.tokenizer.decode([next_id])
            if next_token in ['\n\n', '<|endoftext|>']:
                break

        output = self.tokenizer.decode(generated_ids)
        return output, {'tokens_generated': len(generated_ids)}

    def evaluate_validity(
        self,
        prompts: List[str],
        use_constraints: bool = True,
    ) -> Dict[str, float]:
        """Evaluate syntax validity of generated outputs."""
        valid_count = 0
        total_count = len(prompts)
        parse_errors = []

        for prompt in prompts:
            output, stats = self.generate(
                prompt,
                use_constraints=use_constraints,
            )

            # Check if output is valid Starlark
            is_valid = self._check_starlark_validity(output)
            if is_valid:
                valid_count += 1
            else:
                parse_errors.append(output[:100])

        return {
            'validity_rate': valid_count / max(1, total_count),
            'valid_count': valid_count,
            'total_count': total_count,
            'constrained': use_constraints,
            'sample_errors': parse_errors[:3],
        }

    def _check_starlark_validity(self, code: str) -> bool:
        """Check if code is valid Starlark syntax."""
        try:
            # Use Python's AST as approximation (Starlark is Python-like)
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def get_validity_stats(self) -> Dict[str, float]:
        """Get validity tracking statistics."""
        total = max(1, self.validity_stats['total_tokens'])
        valid = self.validity_stats['total_tokens'] - self.validity_stats['constrained_positions']

        return {
            'total_tokens': self.validity_stats['total_tokens'],
            'valid_tokens': valid,
            'constrained_positions': self.validity_stats['constrained_positions'],
            'validity_rate': valid / total,
        }

    def reset_validity_stats(self):
        """Reset validity tracking."""
        self.validity_stats = {
            'total_tokens': 0,
            'valid_tokens': 0,
            'constrained_positions': 0,
        }


def train_and_eval(
    trainer: ConstrainedLoRATrainer,
    train_data: List[Dict],
    eval_data: List[Dict],
    epochs: int = 3,
) -> Dict[str, Any]:
    """Train and evaluate constrained LoRA."""
    results = {
        'epochs': [],
        'final_metrics': None,
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_data, eval_data)
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")

        # Evaluate validity
        eval_prompts = [d.get('text', d.get('code', ''))[:50] for d in eval_data[:10]]
        validity_constrained = trainer.evaluate_validity(eval_prompts, use_constraints=True)
        validity_unconstrained = trainer.evaluate_validity(eval_prompts, use_constraints=False)

        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            'validity_constrained': validity_constrained['validity_rate'],
            'validity_unconstrained': validity_unconstrained['validity_rate'],
        }
        results['epochs'].append(epoch_results)

        print(f"  Validity (constrained): {validity_constrained['validity_rate']:.2%}")
        print(f"  Validity (unconstrained): {validity_unconstrained['validity_rate']:.2%}")

    results['final_metrics'] = {
        'train_loss': results['epochs'][-1]['train_loss'],
        'validity_constrained': results['epochs'][-1]['validity_constrained'],
        'validity_unconstrained': results['epochs'][-1]['validity_unconstrained'],
        'validity_improvement': (
            results['epochs'][-1]['validity_constrained'] -
            results['epochs'][-1]['validity_unconstrained']
        ),
    }

    return results


def compare_metrics(
    baseline_metrics: Dict[str, Any],
    constrained_metrics: Dict[str, Any],
):
    """Compare baseline and constrained training metrics."""
    print("\n" + "=" * 60)
    print("TRAINING COMPARISON")
    print("=" * 60)

    print("\nFinal Loss:")
    print(f"  Baseline:    {baseline_metrics['final_metrics']['train_loss']:.4f}")
    print(f"  Constrained: {constrained_metrics['final_metrics']['train_loss']:.4f}")

    print("\nValidity Rates:")
    print(f"  Baseline (unconstrained gen):    {baseline_metrics['final_metrics']['validity_unconstrained']:.2%}")
    print(f"  Constrained (constrained gen):   {constrained_metrics['final_metrics']['validity_constrained']:.2%}")

    improvement = constrained_metrics['final_metrics']['validity_improvement']
    print(f"\nValidity Improvement: {improvement:+.2%}")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("CONSTRAINED LORA TRAINER DEMO")
    print("=" * 60)

    if not HAS_MLX:
        print("\nMLX not available - skipping demo")
    else:
        config = ConstrainedLoRAConfig(
            r=8,
            alpha=16,
            learning_rate=1e-4,
            batch_size=2,
            constraint_weight=0.5,
            mask_invalid_tokens=True,
        )

        print(f"\nConfig:")
        print(f"  Constraint weight: {config.constraint_weight}")
        print(f"  Mask invalid tokens: {config.mask_invalid_tokens}")
        print(f"  Validity weighted loss: {config.validity_weighted_loss}")

        trainer = ConstrainedLoRATrainer(config)
        trainer.load_model()

        print(f"\nComponents initialized:")
        print(f"  Model: {type(trainer.model)}")
        print(f"  Statechart: {type(trainer.statechart)}")
        print(f"  Mapper: {type(trainer.mapper)}")

        # Quick training demo
        train_data = load_starlark_corpus()
        print(f"\nTraining on {len(train_data)} examples...")

        metrics = trainer.train_epoch(train_data)
        print(f"\nEpoch metrics: {metrics}")

        validity_stats = trainer.get_validity_stats()
        print(f"\nValidity stats: {validity_stats}")

        print("\nDemo complete!")
