"""
Event Benchmark - Test internal vs external event semantics.

Benchmarks:
1. Queue ordering correctness
2. Priority level enforcement
3. Internal event synchronicity
4. Run-to-completion semantics
5. Starvation prevention
6. Performance metrics
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .event_queue import Event, EventPriority, EventQueue, FIFOEventQueue, StrictPriorityQueue
from .internal_event import (
    InternalEvent, InternalEventType, InternalEventProcessor,
    RunToCompletionProcessor, InternalEventChain,
)
from .priority_handler import (
    PriorityConfig, PriorityAwareQueue, PriorityMetrics,
    PriorityTransitionHandler,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    name: str
    passed: bool
    details: Dict[str, any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    results: List[BenchmarkResult]
    total_time: float

    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def pass_rate(self) -> float:
        return self.passed() / len(self.results) if self.results else 0.0


class QueueOrderingTests:
    """Test event queue ordering semantics."""

    def test_priority_ordering(self) -> BenchmarkResult:
        """Test: Higher priority events dequeued first."""
        queue = EventQueue()

        # Add in reverse priority order
        queue.enqueue_external("low", EventPriority.LOW)
        queue.enqueue_external("normal", EventPriority.NORMAL)
        queue.enqueue_external("high", EventPriority.HIGH)
        queue.enqueue_external("critical", EventPriority.CRITICAL)

        # Dequeue and check order
        order = []
        while not queue.is_empty():
            event = queue.dequeue()
            order.append(event.priority)

        expected = [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW,
        ]

        return BenchmarkResult(
            name="priority_ordering",
            passed=order == expected,
            details={
                "expected": [p.name for p in expected],
                "actual": [p.name for p in order],
            },
        )

    def test_fifo_within_priority(self) -> BenchmarkResult:
        """Test: FIFO ordering within same priority level."""
        queue = EventQueue()

        # Add multiple events at same priority
        queue.enqueue_external("first", EventPriority.NORMAL)
        queue.enqueue_external("second", EventPriority.NORMAL)
        queue.enqueue_external("third", EventPriority.NORMAL)

        order = []
        while not queue.is_empty():
            event = queue.dequeue()
            order.append(event.label)

        expected = ["first", "second", "third"]

        return BenchmarkResult(
            name="fifo_within_priority",
            passed=order == expected,
            details={"order": order},
        )

    def test_internal_jumps_queue(self) -> BenchmarkResult:
        """Test: Internal events processed before external."""
        queue = EventQueue()

        # Add external events first
        queue.enqueue_external("ext1", EventPriority.CRITICAL)
        queue.enqueue_external("ext2", EventPriority.CRITICAL)

        # Add internal event
        queue.enqueue_internal("internal")

        # Internal should come first
        first = queue.dequeue()

        return BenchmarkResult(
            name="internal_jumps_queue",
            passed=first.internal and first.label == "internal",
            details={
                "first_event": first.label,
                "is_internal": first.internal,
            },
        )


class InternalEventTests:
    """Test internal event semantics."""

    def test_completion_chain(self) -> BenchmarkResult:
        """Test: Internal events can trigger more internal events."""
        processor = InternalEventProcessor()
        events_processed = []

        def handler(event: InternalEvent) -> List[InternalEvent]:
            events_processed.append(event.label)
            if event.chain_depth < 3:
                return [InternalEvent(
                    label=f"chain_{event.chain_depth + 1}",
                    event_type=InternalEventType.RAISED,
                )]
            return []

        processor.register_handler("chain_0", handler)
        processor.register_handler("chain_1", handler)
        processor.register_handler("chain_2", handler)
        processor.register_handler("chain_3", handler)

        processor.raise_event("chain_0")
        chain = processor.process_all()

        return BenchmarkResult(
            name="completion_chain",
            passed=len(chain) == 4 and chain.max_depth == 3,
            details={
                "chain_length": len(chain),
                "max_depth": chain.max_depth,
                "events": events_processed,
            },
        )

    def test_run_to_completion(self) -> BenchmarkResult:
        """Test: External events wait for internal processing."""
        rtc = RunToCompletionProcessor()
        processing_log = []

        # External handler raises internal events
        def ext_handler(event: Event) -> List[InternalEvent]:
            processing_log.append(f"EXT:{event.label}")
            return [
                InternalEvent(label="int1", event_type=InternalEventType.RAISED),
                InternalEvent(label="int2", event_type=InternalEventType.RAISED),
            ]

        # Internal handlers
        def int_handler(event: InternalEvent) -> List[InternalEvent]:
            processing_log.append(f"INT:{event.label}")
            return []

        rtc.register_external_handler("click", ext_handler)
        rtc.register_internal_handler("int1", int_handler)
        rtc.register_internal_handler("int2", int_handler)

        # Send two external events
        rtc.send_external("click")
        rtc.send_external("click")

        rtc.run_to_completion()

        # Should process: EXT:click, INT:int1, INT:int2, EXT:click, INT:int1, INT:int2
        # Key: second external waits for first's internal events
        expected_pattern = ["EXT:click", "INT:int1", "INT:int2",
                          "EXT:click", "INT:int1", "INT:int2"]

        return BenchmarkResult(
            name="run_to_completion",
            passed=processing_log == expected_pattern,
            details={
                "expected": expected_pattern,
                "actual": processing_log,
            },
        )

    def test_max_chain_depth(self) -> BenchmarkResult:
        """Test: Chain depth limit prevents infinite loops."""
        processor = InternalEventProcessor()
        processor.MAX_CHAIN_DEPTH = 5

        # Handler that always triggers another event
        def infinite_handler(event: InternalEvent) -> List[InternalEvent]:
            return [InternalEvent(label="loop", event_type=InternalEventType.RAISED)]

        processor.register_handler("loop", infinite_handler)

        processor.raise_event("loop")
        chain = processor.process_all()

        # Should stop at max depth
        return BenchmarkResult(
            name="max_chain_depth",
            passed=chain.max_depth <= processor.MAX_CHAIN_DEPTH,
            details={
                "chain_length": len(chain),
                "max_depth": chain.max_depth,
                "limit": processor.MAX_CHAIN_DEPTH,
            },
        )


class PriorityTests:
    """Test priority handling semantics."""

    def test_starvation_prevention(self) -> BenchmarkResult:
        """Test: Low priority events eventually processed."""
        config = PriorityConfig(
            enable_aging=False,
            enable_starvation_prevention=True,
            starvation_threshold=3,
        )
        queue = PriorityAwareQueue(config)

        # Add one low priority, then many high
        queue.enqueue("low_event", EventPriority.LOW)
        for i in range(10):
            queue.enqueue(f"high_{i}", EventPriority.HIGH)

        # Dequeue all
        order = []
        while not queue.is_empty():
            event = queue.dequeue()
            order.append((event.label, event.priority))

        # Find where low_event was processed
        low_position = next(
            i for i, (label, _) in enumerate(order) if label == "low_event"
        )

        # Low should be processed before all high are done (starvation prevention)
        return BenchmarkResult(
            name="starvation_prevention",
            passed=low_position <= config.starvation_threshold + 1,
            details={
                "low_position": low_position,
                "threshold": config.starvation_threshold,
                "starvation_preventions": queue.metrics.starvation_preventions,
            },
        )

    def test_priority_inheritance(self) -> BenchmarkResult:
        """Test: Events can inherit parent priority."""
        from .priority_handler import PriorityInheritanceManager

        manager = PriorityInheritanceManager()

        parent = Event(label="parent", priority=EventPriority.CRITICAL)
        child = Event(label="child", priority=EventPriority.LOW)

        manager.set_parent(child, parent)

        # Child should inherit parent's priority
        effective = manager.get_effective_priority(child)

        return BenchmarkResult(
            name="priority_inheritance",
            passed=effective == EventPriority.CRITICAL,
            details={
                "original": EventPriority.LOW.name,
                "inherited": effective.name,
            },
        )

    def test_type_based_priority(self) -> BenchmarkResult:
        """Test: Event types map to correct priorities."""
        queue = PriorityAwareQueue()
        handler = PriorityTransitionHandler(queue)

        # Send typed events
        handler.send_typed_event("crash", "error")      # CRITICAL
        handler.send_typed_event("click", "user_action") # HIGH
        handler.send_typed_event("update", "update")     # NORMAL
        handler.send_typed_event("log", "analytics")     # LOW

        # Check priorities
        order = []
        while not queue.is_empty():
            event = queue.dequeue()
            order.append((event.label, event.priority))

        expected_order = [
            ("crash", EventPriority.CRITICAL),
            ("click", EventPriority.HIGH),
            ("update", EventPriority.NORMAL),
            ("log", EventPriority.LOW),
        ]

        return BenchmarkResult(
            name="type_based_priority",
            passed=order == expected_order,
            details={
                "expected": [(l, p.name) for l, p in expected_order],
                "actual": [(l, p.name) for l, p in order],
            },
        )


class PerformanceTests:
    """Test performance characteristics."""

    def test_queue_throughput(self) -> BenchmarkResult:
        """Test: Queue operations are efficient."""
        queue = EventQueue()
        n_events = 1000

        # Enqueue
        start = time.perf_counter()
        for i in range(n_events):
            priority = random.choice(list(EventPriority))
            queue.enqueue_external(f"event_{i}", priority)
        enqueue_time = time.perf_counter() - start

        # Dequeue
        start = time.perf_counter()
        while not queue.is_empty():
            queue.dequeue()
        dequeue_time = time.perf_counter() - start

        total_time = enqueue_time + dequeue_time
        events_per_sec = n_events / total_time

        return BenchmarkResult(
            name="queue_throughput",
            passed=events_per_sec > 10000,  # Should process >10k events/sec
            details={
                "n_events": n_events,
                "enqueue_ms": enqueue_time * 1000,
                "dequeue_ms": dequeue_time * 1000,
                "events_per_sec": int(events_per_sec),
            },
        )

    def test_internal_chain_performance(self) -> BenchmarkResult:
        """Test: Internal event chains are efficient."""
        processor = InternalEventProcessor()
        processor.MAX_CHAIN_DEPTH = 50

        chain_length = 0

        def chain_handler(event: InternalEvent) -> List[InternalEvent]:
            nonlocal chain_length
            chain_length += 1
            if event.chain_depth < 49:
                return [InternalEvent(label="chain", event_type=InternalEventType.RAISED)]
            return []

        processor.register_handler("chain", chain_handler)

        start = time.perf_counter()
        processor.raise_event("chain")
        processor.process_all()
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            name="internal_chain_performance",
            passed=elapsed < 0.01,  # Should complete in <10ms
            details={
                "chain_length": chain_length,
                "elapsed_ms": elapsed * 1000,
            },
        )


def run_benchmark() -> BenchmarkSuite:
    """Run complete benchmark suite."""
    start_time = time.time()
    results = []

    # Queue ordering tests
    ordering_tests = QueueOrderingTests()
    results.append(ordering_tests.test_priority_ordering())
    results.append(ordering_tests.test_fifo_within_priority())
    results.append(ordering_tests.test_internal_jumps_queue())

    # Internal event tests
    internal_tests = InternalEventTests()
    results.append(internal_tests.test_completion_chain())
    results.append(internal_tests.test_run_to_completion())
    results.append(internal_tests.test_max_chain_depth())

    # Priority tests
    priority_tests = PriorityTests()
    results.append(priority_tests.test_starvation_prevention())
    results.append(priority_tests.test_priority_inheritance())
    results.append(priority_tests.test_type_based_priority())

    # Performance tests
    perf_tests = PerformanceTests()
    results.append(perf_tests.test_queue_throughput())
    results.append(perf_tests.test_internal_chain_performance())

    total_time = time.time() - start_time
    return BenchmarkSuite(results=results, total_time=total_time)


def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("EVENT SEMANTICS BENCHMARK")
    print("=" * 60)

    suite = run_benchmark()

    print(f"\nResults: {suite.passed()}/{len(suite.results)} tests passed")
    print(f"Time: {suite.total_time*1000:.1f}ms")
    print("-" * 60)

    for result in suite.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}")
        if result.error:
            print(f"  Error: {result.error}")
        for key, value in result.details.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {suite.pass_rate()*100:.0f}% pass rate")
    print("=" * 60)

    return suite


if __name__ == "__main__":
    demo()
