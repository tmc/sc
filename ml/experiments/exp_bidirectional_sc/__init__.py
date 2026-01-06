"""
exp_bidirectional_sc: Bidirectional Statechart Training

GOAL: Train model to BOTH:
1. Generate SCs from descriptions (text → SC)
2. Explain SCs in natural language (SC → text)

HYPOTHESIS: Bidirectional training improves both directions through
shared representation learning.

Uses exp_sc_generation_benchmark's 100 tasks as test set.
"""

from .dataset import BidirectionalDataset, create_dataset
from .trainer import BidirectionalTrainer, train_bidirectional
from .evaluator import BidirectionalEvaluator, evaluate_bidirectional

__all__ = [
    'BidirectionalDataset',
    'create_dataset',
    'BidirectionalTrainer', 
    'train_bidirectional',
    'BidirectionalEvaluator',
    'evaluate_bidirectional',
]
