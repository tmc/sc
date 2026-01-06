"""
exp_internal_vs_external_events: Internal vs External Event Semantics

Test internal vs external event semantics from proto:
- extensions/v1/annotations.proto: bool internal = 7
- extensions/v1/annotations.proto: string priority = 9 ("low"/"normal"/"high"/"critical")

Key Concepts:
1. External events: Queued, processed in priority order
2. Internal events: Synchronous, processed immediately (completion, raised)
3. Run-to-completion: External events wait for internal processing
4. Priority levels: CRITICAL > HIGH > NORMAL > LOW

Components:
- event_queue.py: Priority-ordered event queue
- internal_event.py: Internal event and RTC processing
- priority_handler.py: Priority aging, starvation prevention
- event_benchmark.py: Test all event semantics

Usage:
    from experiments.exp_internal_vs_external_events import (
        EventQueue, EventPriority,
        InternalEventProcessor, RunToCompletionProcessor,
        PriorityAwareQueue,
        run_benchmark,
    )

    # Basic queue
    queue = EventQueue()
    queue.enqueue_external("click", EventPriority.HIGH)
    queue.enqueue_internal("completion")  # Jumps queue
    event = queue.dequeue()  # Returns internal first

    # Run-to-completion
    rtc = RunToCompletionProcessor()
    rtc.send_external("click")
    rtc.run_to_completion()  # Processes all internal events
"""

from .event_queue import (
    EventPriority,
    Event,
    EventOccurrence,
    EventQueue,
    FIFOEventQueue,
    StrictPriorityQueue,
    EventDispatcher,
)

from .internal_event import (
    InternalEventType,
    InternalEvent,
    InternalEventChain,
    InternalEventProcessor,
    RunToCompletionProcessor,
    CompletionEventGenerator,
)

from .priority_handler import (
    PriorityConfig,
    PriorityMetrics,
    PriorityAwareQueue,
    PriorityInheritanceManager,
    PriorityTransitionHandler,
    PriorityScheduler,
)

from .event_benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    QueueOrderingTests,
    InternalEventTests,
    PriorityTests,
    PerformanceTests,
    run_benchmark,
)

__all__ = [
    # Event Queue
    'EventPriority',
    'Event',
    'EventOccurrence',
    'EventQueue',
    'FIFOEventQueue',
    'StrictPriorityQueue',
    'EventDispatcher',
    # Internal Events
    'InternalEventType',
    'InternalEvent',
    'InternalEventChain',
    'InternalEventProcessor',
    'RunToCompletionProcessor',
    'CompletionEventGenerator',
    # Priority Handler
    'PriorityConfig',
    'PriorityMetrics',
    'PriorityAwareQueue',
    'PriorityInheritanceManager',
    'PriorityTransitionHandler',
    'PriorityScheduler',
    # Benchmark
    'BenchmarkResult',
    'BenchmarkSuite',
    'QueueOrderingTests',
    'InternalEventTests',
    'PriorityTests',
    'PerformanceTests',
    'run_benchmark',
]
