"""
Hindsight Trainer

Integrates hindsight relabeling from SOAR into LoRA training.
Key insight: Every program execution is "correct" for SOME task.

Workflow:
1. Generate programs with constrained sampling
2. Programs that parse but fail semantically → relabel as correct for synthetic tasks
3. Train on both real + synthetic examples
4. Dramatically increases training data efficiency

Based on exp_soar_statechart/hindsight_relabel.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import random
import hashlib
import time

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

try:
    from .constrained_lora import ConstrainedLoRATrainer, ConstrainedLoRAConfig
    from .starlark_statechart import StarlarkStatechart
    from .constrained_sampler import SamplingConfig
except ImportError:
    from constrained_lora import ConstrainedLoRATrainer, ConstrainedLoRAConfig
    from starlark_statechart import StarlarkStatechart
    from constrained_sampler import SamplingConfig


@dataclass
class SyntheticTask:
    """A synthetic task created from a failed program execution."""
    original_task_id: str
    synthetic_id: str
    input_prompt: str
    actual_output: str  # What the program actually produced
    complexity: float  # How much transformation from input
    diversity: float  # Distance from other synthetic tasks
    source_valid: bool  # Was the output syntactically valid?

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to training example format."""
        return {
            'text': f"{self.input_prompt}\n{self.actual_output}",
            'input': self.input_prompt,
            'output': self.actual_output,
            'synthetic': True,
            'task_id': self.synthetic_id,
        }


@dataclass
class HindsightConfig(ConstrainedLoRAConfig):
    """Configuration for hindsight training."""
    # Hindsight settings
    min_complexity: float = 0.1  # Min output complexity for synthetic task
    max_synthetic_per_prompt: int = 10  # Max synthetic tasks per prompt
    diversity_threshold: float = 0.3  # Min diversity between synthetic tasks
    use_synthetic_data: bool = True  # Include synthetic in training

    # Generation settings
    generations_per_epoch: int = 100  # Programs to generate per epoch
    synthetic_weight: float = 0.5  # Weight for synthetic examples in loss


class HindsightRelabeler:
    """
    Creates synthetic training tasks from program executions.

    Every program that runs (even if it doesn't solve the original task)
    becomes a valid training example for a synthetic task where that
    program IS the correct output.
    """

    def __init__(
        self,
        min_complexity: float = 0.1,
        max_synthetic_per_prompt: int = 10,
        diversity_threshold: float = 0.3,
    ):
        self.min_complexity = min_complexity
        self.max_synthetic_per_prompt = max_synthetic_per_prompt
        self.diversity_threshold = diversity_threshold

        self.synthetic_tasks: Dict[str, List[SyntheticTask]] = {}
        self.all_outputs: List[str] = []

    def collect(
        self,
        prompt: str,
        output: str,
        syntax_valid: bool,
        semantic_valid: bool,
        task_id: str = "",
    ) -> Optional[SyntheticTask]:
        """
        Collect a program execution for potential relabeling.

        Args:
            prompt: Input prompt
            output: Generated output
            syntax_valid: Whether output parsed successfully
            semantic_valid: Whether output solved the original task
            task_id: Original task identifier

        Returns:
            SyntheticTask if created, None otherwise
        """
        # Skip if already solved the original task
        if semantic_valid:
            return None

        # Skip if syntax invalid (can't be training data)
        if not syntax_valid:
            return None

        # Check complexity
        complexity = self._compute_complexity(prompt, output)
        if complexity < self.min_complexity:
            return None

        # Check diversity
        diversity = self._compute_diversity(output)
        if diversity < self.diversity_threshold:
            return None

        # Check max per prompt
        if task_id in self.synthetic_tasks:
            if len(self.synthetic_tasks[task_id]) >= self.max_synthetic_per_prompt:
                return None

        # Create synthetic task
        synthetic = SyntheticTask(
            original_task_id=task_id,
            synthetic_id=self._generate_id(prompt, output),
            input_prompt=prompt,
            actual_output=output,
            complexity=complexity,
            diversity=diversity,
            source_valid=syntax_valid,
        )

        # Store
        if task_id not in self.synthetic_tasks:
            self.synthetic_tasks[task_id] = []
        self.synthetic_tasks[task_id].append(synthetic)
        self.all_outputs.append(output)

        return synthetic

    def _compute_complexity(self, prompt: str, output: str) -> float:
        """Compute complexity of output relative to prompt."""
        # Simple metric: ratio of output length to prompt length
        if len(prompt) == 0:
            return 0.0

        # Count non-trivial tokens
        output_tokens = len(output.split())
        prompt_tokens = len(prompt.split())

        # Complexity = how much the model added
        if prompt_tokens == 0:
            return 1.0

        ratio = output_tokens / prompt_tokens
        return min(1.0, ratio / 5.0)  # Normalize to [0, 1]

    def _compute_diversity(self, output: str) -> float:
        """Compute diversity from existing synthetic outputs."""
        if not self.all_outputs:
            return 1.0

        # Jaccard-like diversity metric
        output_set = set(output.split())
        max_similarity = 0.0

        for existing in self.all_outputs[-100:]:  # Check recent outputs
            existing_set = set(existing.split())
            intersection = len(output_set & existing_set)
            union = len(output_set | existing_set)
            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        return 1.0 - max_similarity

    def _generate_id(self, prompt: str, output: str) -> str:
        """Generate unique ID for synthetic task."""
        content = f"{prompt}||{output}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_synthetic_examples(self) -> List[Dict[str, Any]]:
        """Get all synthetic examples as training data."""
        examples = []
        for task_list in self.synthetic_tasks.values():
            for task in task_list:
                examples.append(task.to_training_example())
        return examples

    def get_statistics(self) -> Dict[str, Any]:
        """Get relabeling statistics."""
        total_synthetic = sum(len(tasks) for tasks in self.synthetic_tasks.values())
        avg_complexity = 0.0
        avg_diversity = 0.0

        if total_synthetic > 0:
            complexities = []
            diversities = []
            for task_list in self.synthetic_tasks.values():
                for task in task_list:
                    complexities.append(task.complexity)
                    diversities.append(task.diversity)
            avg_complexity = sum(complexities) / len(complexities)
            avg_diversity = sum(diversities) / len(diversities)

        return {
            'total_synthetic': total_synthetic,
            'prompts_covered': len(self.synthetic_tasks),
            'avg_complexity': avg_complexity,
            'avg_diversity': avg_diversity,
            'total_outputs_seen': len(self.all_outputs),
        }

    def clear(self):
        """Clear all collected data."""
        self.synthetic_tasks = {}
        self.all_outputs = []


class HindsightTrainer(ConstrainedLoRATrainer):
    """
    LoRA trainer with hindsight relabeling for sample efficiency.

    Extends constrained LoRA trainer to:
    1. Generate programs and collect failed executions
    2. Create synthetic tasks from valid-but-wrong outputs
    3. Train on combined real + synthetic data
    """

    def __init__(self, config: HindsightConfig):
        super().__init__(config)
        self.config = config
        self.relabeler = HindsightRelabeler(
            min_complexity=config.min_complexity,
            max_synthetic_per_prompt=config.max_synthetic_per_prompt,
            diversity_threshold=config.diversity_threshold,
        )
        self.synthetic_buffer: List[Dict[str, Any]] = []

    def generate_and_relabel(
        self,
        prompts: List[str],
        check_semantic: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate programs and collect for hindsight relabeling.

        Args:
            prompts: List of input prompts
            check_semantic: Optional function to check semantic correctness

        Returns:
            Statistics about generation and relabeling
        """
        stats = {
            'total_generated': 0,
            'syntax_valid': 0,
            'semantic_valid': 0,
            'synthetic_created': 0,
        }

        for i, prompt in enumerate(prompts):
            # Generate with constraints
            output, gen_stats = self.generate(prompt, use_constraints=True)
            stats['total_generated'] += 1

            # Check syntax validity
            syntax_valid = self._check_starlark_validity(output)
            if syntax_valid:
                stats['syntax_valid'] += 1

            # Check semantic validity (if function provided)
            semantic_valid = False
            if check_semantic is not None:
                try:
                    semantic_valid = check_semantic(prompt, output)
                except Exception:
                    pass

            if semantic_valid:
                stats['semantic_valid'] += 1

            # Collect for relabeling
            task_id = f"prompt_{i}"
            synthetic = self.relabeler.collect(
                prompt=prompt,
                output=output,
                syntax_valid=syntax_valid,
                semantic_valid=semantic_valid,
                task_id=task_id,
            )

            if synthetic is not None:
                stats['synthetic_created'] += 1
                self.synthetic_buffer.append(synthetic.to_training_example())

        return stats

    def train_epoch(
        self,
        real_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        use_synthetic: bool = None,
    ) -> Dict[str, float]:
        """Train for one epoch with optional synthetic data."""
        use_synthetic = use_synthetic if use_synthetic is not None else self.config.use_synthetic_data

        # Combine real and synthetic data
        if use_synthetic and self.synthetic_buffer:
            combined_data = real_data + self.synthetic_buffer
            random.shuffle(combined_data)
        else:
            combined_data = real_data

        # Train
        metrics = super().train_epoch(combined_data, eval_data)

        # Add synthetic stats
        metrics['synthetic_examples'] = len(self.synthetic_buffer)
        metrics['data_amplification'] = (
            len(combined_data) / max(1, len(real_data))
        )

        return metrics

    def train_with_hindsight(
        self,
        real_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 5,
        generate_per_epoch: int = None,
    ) -> Dict[str, Any]:
        """
        Full training loop with hindsight relabeling.

        Each epoch:
        1. Generate programs and collect synthetic examples
        2. Train on combined real + synthetic data
        3. Evaluate and track metrics
        """
        generate_per_epoch = generate_per_epoch or self.config.generations_per_epoch
        results = {
            'epochs': [],
            'relabeling_stats': [],
        }

        # Extract prompts from real data
        prompts = [d.get('text', d.get('input', ''))[:100] for d in real_data]

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Generate and collect synthetic examples
            sample_prompts = random.sample(
                prompts, min(generate_per_epoch, len(prompts))
            )
            gen_stats = self.generate_and_relabel(sample_prompts)
            print(f"  Generated: {gen_stats['total_generated']}, "
                  f"Syntax valid: {gen_stats['syntax_valid']}, "
                  f"Synthetic: {gen_stats['synthetic_created']}")

            # Train on combined data
            train_metrics = self.train_epoch(real_data, eval_data, use_synthetic=True)
            print(f"  Train loss: {train_metrics['train_loss']:.4f}, "
                  f"Data amplification: {train_metrics['data_amplification']:.2f}x")

            # Store results
            epoch_result = {
                'epoch': epoch + 1,
                **train_metrics,
                **gen_stats,
            }
            results['epochs'].append(epoch_result)

        # Final relabeling stats
        results['relabeling_stats'] = self.relabeler.get_statistics()
        results['final_synthetic_buffer'] = len(self.synthetic_buffer)

        return results

    def compare_with_baseline(
        self,
        real_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare hindsight training with baseline (no synthetic data).

        Returns metrics for both approaches.
        """
        # Baseline: Train without synthetic
        print("\n" + "=" * 60)
        print("BASELINE TRAINING (no synthetic data)")
        print("=" * 60)

        baseline_metrics = []
        for epoch in range(epochs):
            metrics = self.train_epoch(real_data, eval_data, use_synthetic=False)
            baseline_metrics.append(metrics)
            print(f"Epoch {epoch + 1}: loss={metrics['train_loss']:.4f}")

        # Reset model
        self.reset()
        self.relabeler.clear()
        self.synthetic_buffer = []

        # Hindsight: Train with synthetic
        print("\n" + "=" * 60)
        print("HINDSIGHT TRAINING (with synthetic data)")
        print("=" * 60)

        hindsight_results = self.train_with_hindsight(
            real_data, eval_data, epochs=epochs
        )

        return {
            'baseline': {
                'epochs': baseline_metrics,
                'final_loss': baseline_metrics[-1]['train_loss'],
            },
            'hindsight': {
                'epochs': hindsight_results['epochs'],
                'final_loss': hindsight_results['epochs'][-1]['train_loss'],
                'relabeling_stats': hindsight_results['relabeling_stats'],
            },
            'improvement': {
                'loss_reduction': (
                    baseline_metrics[-1]['train_loss'] -
                    hindsight_results['epochs'][-1]['train_loss']
                ),
                'data_amplification': hindsight_results['epochs'][-1]['data_amplification'],
            },
        }


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("HINDSIGHT TRAINER DEMO")
    print("=" * 60)

    if not HAS_MLX:
        print("\nMLX not available - showing relabeler demo only")

    # Demo the relabeler independently
    relabeler = HindsightRelabeler(
        min_complexity=0.1,
        diversity_threshold=0.3,
    )

    # Simulate some program executions
    examples = [
        ("def foo():", "def foo():\n    return 42", True, False),
        ("def bar():", "def bar():\n    x = 1\n    return x", True, False),
        ("def baz():", "def baz():\n    return None", True, True),  # Correct
        ("def qux():", "def qux(:\n    return", False, False),  # Syntax error
    ]

    print("\nSimulating program executions:")
    for prompt, output, syntax, semantic in examples:
        result = relabeler.collect(
            prompt=prompt,
            output=output,
            syntax_valid=syntax,
            semantic_valid=semantic,
            task_id=prompt[:10],
        )
        status = "Created synthetic" if result else "Skipped"
        print(f"  {prompt} -> {status}")

    print(f"\nRelabeler statistics:")
    stats = relabeler.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSynthetic training examples:")
    for example in relabeler.get_synthetic_examples():
        print(f"  {example['synthetic_id']}: {example['input'][:30]}...")

    print("\nDemo complete!")
