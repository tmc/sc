"""
exp_qwen_statechart: Statechart-Guided Fine-Tuning for Qwen Coder

KEY RESEARCH CONTRIBUTION:
Demonstrate that statechart structure improves SOAR sample efficiency on
ARC-style tasks by constraining LLM generation to syntactically valid programs.

Hypothesis: Statechart-guided generation provides:
1. 100% syntactic validity (vs ~75% for unconstrained SOAR)
2. Structure-preserving mutations via statechart traversal
3. Faster convergence through reduced search space

Components:
1. StarlarkStatechart - Starlark syntax state machine (simpler than Python)
2. StatechartConstrainedSampler - Logit masking for valid tokens only
3. ConstrainedLoRATrainer - LoRA with statechart constraints during training
4. ImprovedSOAR - SOAR reimplementation with statechart advantages
5. HindsightTrainer - Hindsight relabeling for sample efficiency

Experiments:
- exp1: Constrained vs unconstrained sampling comparison
- exp2: LoRA vs constrained LoRA training
- exp3: Hindsight relabeling for Starlark
- exp4: Evolved guards for failure prediction
- exp5: SAE probing for syntax state discovery

Based on:
- exp_code_completion/ - Logit masking pattern (+25% validity)
- exp_soar_statechart/ - SOAR + REX + hindsight relabeling
- exp_guard_synthesis/ - Evolutionary guard synthesis (F1=1.0)
- exp_transfer_learning/ - Topology-preserving crossover

Usage:
    from exp_qwen_statechart import (
        StarlarkStatechart,
        StatechartConstrainedSampler,
        ConstrainedLoRATrainer,
    )

    # Create statechart for Starlark syntax
    statechart = StarlarkStatechart()

    # Wrap model with constrained sampling
    sampler = StatechartConstrainedSampler(model, tokenizer, statechart)

    # Generate syntactically valid Starlark
    output = sampler.generate("def traffic_light():")
"""

# Core components
from .starlark_statechart import (
    StarlarkSyntaxState,
    StarlarkStatechart,
    StarlarkContext,
)

from .token_mapper import TokenMapper

from .constrained_sampler import StatechartConstrainedSampler

# Training components (import when available)
try:
    from .lora_trainer import LoRAConfig, LoRATrainer
    from .constrained_lora import ConstrainedLoRATrainer
    from .hindsight_trainer import HindsightTrainer
except ImportError:
    pass

# SOAR improvements (import when available)
try:
    from .soar_improved import ImprovedSOAR
except ImportError:
    pass

__all__ = [
    'StarlarkSyntaxState',
    'StarlarkStatechart',
    'StarlarkContext',
    'TokenMapper',
    'StatechartConstrainedSampler',
    'LoRAConfig',
    'LoRATrainer',
    'ConstrainedLoRATrainer',
    'HindsightTrainer',
    'ImprovedSOAR',
]
