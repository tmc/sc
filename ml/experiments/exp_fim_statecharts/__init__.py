"""
Fill-in-Middle (FIM) Experiment for Statechart Generation

Problem: Standard FIM fails on statecharts with 0% valid insertion because
LLMs don't understand structural constraints when filling holes.

Solution: Structure-aware FIM that provides context about:
- Valid state labels in scope
- Required structural elements (type, is_initial, etc.)
- Transition source/target constraints

Hole Types:
- STATE: Insert child state into existing state
- TRANSITION: Insert transition connecting existing states
- GUARD: Insert guard expression into transition
- ACTION: Insert entry/exit action into state

Uses: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from .fim_generator import FIMGenerator, HoleType, FIMResult
from .benchmark import run_benchmark

__all__ = ['FIMGenerator', 'HoleType', 'FIMResult', 'run_benchmark']
