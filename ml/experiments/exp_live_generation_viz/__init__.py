"""
exp_live_generation_viz: Live Statechart Generation Visualization

GOAL: Integrate SC generation with web visualizer for real-time feedback.

Features:
1. TOKEN-BY-TOKEN STREAMING: Watch statechart build token by token
2. PARTIAL PARSING: Parse incomplete JSON during generation
3. LIVE VALIDATION: Stream validation results as structure forms
4. VISUAL FEEDBACK: Highlight errors, show progress

Uses ml/utils/visualizer_bridge.py for WebSocket communication.

Architecture:
    LLM Generator --> Partial Parser --> Visualizer Bridge --> Web UI
         |               |                    |
         v               v                    v
      tokens        partial SC            WebSocket
                                              |
                                              v
                                        Live Diagram

Usage:
    from ml.experiments.exp_live_generation_viz import (
        LiveGenerator,
        PartialParser,
        GenerationDemo,
    )

    # Create live generator
    demo = GenerationDemo()
    demo.run("Create a traffic light state machine")

    # Watch in browser at http://localhost:8080
"""

from .partial_parser import (
    PartialParser,
    PartialParseResult,
    ParseState,
    parse_partial_json,
    parse_partial_statechart,
)

from .generation_demo import (
    GenerationDemo,
    LiveGenerator,
    DemoConfig,
    MockLLM,
    run_demo,
)

from .benchmark import (
    VizBenchmark,
    BenchmarkResult,
    LatencyTest,
    run_viz_benchmark,
)

__all__ = [
    # Partial parser
    'PartialParser',
    'PartialParseResult',
    'ParseState',
    'parse_partial_json',
    'parse_partial_statechart',
    # Demo
    'GenerationDemo',
    'LiveGenerator',
    'DemoConfig',
    'MockLLM',
    'run_demo',
    # Benchmark
    'VizBenchmark',
    'BenchmarkResult',
    'LatencyTest',
    'run_viz_benchmark',
]
