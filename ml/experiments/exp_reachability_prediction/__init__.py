"""
exp_reachability_prediction: Predict if a target state is reachable.

Problem: Given a statechart, current configuration, and target state,
predict whether the target is reachable and the minimum steps needed.

Approach:
1. BFS baseline: Exhaustive search for ground truth
2. LLM predictor: Use Qwen to predict reachability

Input: (SC_json, current_config, target_state)
Output: (is_reachable: bool, min_steps: int, required_events: list)

Model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from .reachability_predictor import (
    BFSReachabilityAnalyzer,
    LLMReachabilityPredictor,
    ReachabilityResult,
)
from .benchmark import run_benchmark

__all__ = [
    'BFSReachabilityAnalyzer',
    'LLMReachabilityPredictor',
    'ReachabilityResult',
    'run_benchmark',
]
