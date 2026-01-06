"""
exp_terminal_states: Final State and Completion Semantics

Test State.is_final semantics from proto/statecharts/v1/statecharts.proto:
  "bool is_final = 5; // Terminal state: no outgoing transitions"

Key Concepts:
1. Final states signal completion (generate τ event)
2. PARALLEL completes when ALL children reach final
3. Completion transitions (τ) fire automatically on completion
4. Completion propagates up the hierarchy

Components:
- terminal_state.py: Final state handling and validation
- completion_detector.py: Detect when parallel regions complete
- parent_trigger.py: Trigger parent transitions on completion
- benchmark.py: Test completion propagation semantics

Usage:
    from experiments.exp_terminal_states import (
        State, StateType, Configuration, Transition,
        CompletionDetector,
        ParentTriggerEngine,
        run_benchmark,
    )

    # Build statechart with final states
    root, transitions = build_parallel_statechart()

    # Track completion
    detector = CompletionDetector(root)
    events = detector.update(config)

    # Execute with completion chains
    engine = ParentTriggerEngine(root, transitions)
    engine.initialize()
    executions = engine.process_event("done")
"""

from .terminal_state import (
    StateType,
    State,
    Transition,
    Configuration,
    FinalStateValidator,
    FinalStateSemantics,
    build_parallel_statechart,
)

from .completion_detector import (
    CompletionStatus,
    RegionStatus,
    CompletionEvent,
    CompletionDetector,
    MultiLevelCompletionDetector,
    build_nested_parallel,
)

from .parent_trigger import (
    TransitionExecution,
    ParentTriggerEngine,
    CompletionChainTracker,
    build_completion_chain_example,
)

from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    StatechartGenerator,
    CompletionDetectionTests,
    CompletionTriggerTests,
    ValidationTests,
    run_benchmark,
)

__all__ = [
    # Terminal State
    'StateType',
    'State',
    'Transition',
    'Configuration',
    'FinalStateValidator',
    'FinalStateSemantics',
    'build_parallel_statechart',
    # Completion Detector
    'CompletionStatus',
    'RegionStatus',
    'CompletionEvent',
    'CompletionDetector',
    'MultiLevelCompletionDetector',
    'build_nested_parallel',
    # Parent Trigger
    'TransitionExecution',
    'ParentTriggerEngine',
    'CompletionChainTracker',
    'build_completion_chain_example',
    # Benchmark
    'BenchmarkResult',
    'BenchmarkSuite',
    'StatechartGenerator',
    'CompletionDetectionTests',
    'CompletionTriggerTests',
    'ValidationTests',
    'run_benchmark',
]
