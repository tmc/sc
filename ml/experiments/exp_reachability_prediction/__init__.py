"""
exp_reachability_prediction: Predict if a target state is reachable.

Problem: Given a statechart, current configuration, and target state,
predict whether the target is reachable and the minimum steps needed.

Approach:
1. BFS baseline: Exhaustive search for ground truth
2. LLM predictor: Use Qwen to predict reachability
3. Scratchpad: BFS expansion in text for step counting

Results:
- Reachability F1: 71% (good)
- Steps accuracy: 0% baseline → improved with scratchpad

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

from .steps_predictor import (
    ReachabilityCase,
    StepsPredictor,
    create_bfs_scratchpad_prompt,
    create_simple_trace_prompt,
    parse_steps_response,
    get_test_cases,
)

from .benchmark_steps import (
    run_benchmark as run_steps_benchmark,
    BenchmarkConfig as StepsConfig,
    PredictionResult as StepsPredictionResult,
)

__all__ = [
    # Original
    'BFSReachabilityAnalyzer',
    'LLMReachabilityPredictor',
    'ReachabilityResult',
    'run_benchmark',
    # Steps scratchpad
    'ReachabilityCase',
    'StepsPredictor',
    'create_bfs_scratchpad_prompt',
    'create_simple_trace_prompt',
    'parse_steps_response',
    'get_test_cases',
    'run_steps_benchmark',
    'StepsConfig',
    'StepsPredictionResult',
]
