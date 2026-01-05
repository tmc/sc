#!/usr/bin/env python3
"""Quick benchmark runner."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_transfer_learning.benchmark import quick_benchmark
results = quick_benchmark()
print()
print(results.summary())
