"""
React State Extraction Experiment

Extract statecharts from React component code by parsing useState/useReducer
patterns and building state machines from the discovered state transitions.

Components:
- react_parser.py: Parse React/JSX code to extract hooks and state patterns
- state_extractor.py: Extract state variables, transitions, and effects
- sc_builder.py: Build statechart proto from extracted state machine
- benchmark.py: Accuracy testing on 10 React components
"""

from .react_parser import ReactParser
from .state_extractor import StateExtractor
from .sc_builder import SCBuilder
from .benchmark import run_benchmark

__all__ = ['ReactParser', 'StateExtractor', 'SCBuilder', 'run_benchmark']
