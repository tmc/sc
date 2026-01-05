#!/usr/bin/env python3
"""Full transfer learning benchmark with proper training."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_transfer_learning.benchmark import run_transfer_benchmark

results = run_transfer_benchmark(
    n_runs=1,  # Single run for speed
    n_generations=50,
    population_size=30,
    target_accuracy=0.85,  # More realistic target
    test_chain=False,  # Skip Othello for speed
    verbose=True
)

print("\n" + results.summary())
