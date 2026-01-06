"""
Profiler: Memory and Time Profiling for Scalability Testing

Provides:
1. MemoryTracker - Track memory allocation during operations
2. TimeTracker - High-resolution timing with statistics
3. Profiler - Combined profiling with detailed breakdown
4. profile_operation - Decorator for easy profiling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from contextlib import contextmanager
from functools import wraps
import time
import gc
import sys
import tracemalloc
import statistics


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage."""
    timestamp: float
    current_bytes: int
    peak_bytes: int
    label: str = ""

    @property
    def current_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


@dataclass
class TimeSpan:
    """A timed span."""
    label: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        return self.duration * 1000


@dataclass
class ProfileResult:
    """Complete profiling result."""
    operation: str
    n_states: int = 0

    # Time metrics
    total_time: float = 0.0
    time_spans: List[TimeSpan] = field(default_factory=list)

    # Memory metrics
    memory_before: int = 0
    memory_after: int = 0
    memory_peak: int = 0
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)

    # Computed metrics
    throughput: float = 0.0  # States per second
    memory_per_state: float = 0.0  # Bytes per state

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_states > 0 and self.total_time > 0:
            self.throughput = self.n_states / self.total_time

        if self.n_states > 0:
            mem_used = self.memory_after - self.memory_before
            self.memory_per_state = mem_used / self.n_states if mem_used > 0 else 0

    @property
    def time_ms(self) -> float:
        return self.total_time * 1000

    @property
    def memory_delta_mb(self) -> float:
        return (self.memory_after - self.memory_before) / (1024 * 1024)

    @property
    def memory_peak_mb(self) -> float:
        return self.memory_peak / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'n_states': self.n_states,
            'total_time_ms': self.time_ms,
            'memory_delta_mb': self.memory_delta_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'throughput': self.throughput,
            'memory_per_state': self.memory_per_state,
            **self.metadata,
        }


class MemoryTracker:
    """
    Track memory allocation during operations.

    Uses tracemalloc for accurate tracking.
    """

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.is_tracking = False
        self._start_snapshot = None

    def start(self):
        """Start memory tracking."""
        gc.collect()
        tracemalloc.start()
        self.is_tracking = True
        self._start_snapshot = self._take_snapshot("start")

    def stop(self) -> Tuple[int, int]:
        """Stop tracking and return (delta, peak)."""
        if not self.is_tracking:
            return 0, 0

        end_snapshot = self._take_snapshot("end")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.is_tracking = False

        delta = end_snapshot.current_bytes - self._start_snapshot.current_bytes
        return delta, peak

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        return self._take_snapshot(label)

    def _take_snapshot(self, label: str) -> MemorySnapshot:
        """Internal snapshot creation."""
        current, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            current_bytes=current,
            peak_bytes=peak,
            label=label,
        )
        self.snapshots.append(snapshot)
        return snapshot

    @contextmanager
    def track(self, label: str = ""):
        """Context manager for memory tracking."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class TimeTracker:
    """
    High-resolution timing with statistics.

    Supports nested timing spans.
    """

    def __init__(self):
        self.spans: List[TimeSpan] = []
        self._stack: List[TimeSpan] = []

    def start(self, label: str = "main") -> TimeSpan:
        """Start a timing span."""
        span = TimeSpan(label=label, start_time=time.perf_counter())
        self._stack.append(span)
        return span

    def stop(self, label: str = None) -> TimeSpan:
        """Stop a timing span."""
        if not self._stack:
            raise RuntimeError("No active timing span")

        span = self._stack.pop()
        if label and span.label != label:
            raise RuntimeError(f"Expected to stop '{label}', but active span is '{span.label}'")

        span.end_time = time.perf_counter()
        self.spans.append(span)
        return span

    def span_durations(self) -> Dict[str, float]:
        """Get duration for each span label."""
        durations = {}
        for span in self.spans:
            if span.label in durations:
                durations[span.label] += span.duration
            else:
                durations[span.label] = span.duration
        return durations

    @contextmanager
    def time(self, label: str = "main"):
        """Context manager for timing."""
        self.start(label)
        try:
            yield
        finally:
            self.stop(label)

    def total_time(self) -> float:
        """Get total time across all spans."""
        return sum(s.duration for s in self.spans)


class Profiler:
    """
    Combined memory and time profiler.

    Provides detailed breakdown of operations.
    """

    def __init__(self, operation: str):
        self.operation = operation
        self.memory_tracker = MemoryTracker()
        self.time_tracker = TimeTracker()
        self.n_states = 0
        self.metadata: Dict[str, Any] = {}

    def start(self):
        """Start profiling."""
        gc.collect()
        self.memory_tracker.start()
        self.time_tracker.start("total")

    def stop(self) -> ProfileResult:
        """Stop profiling and get result."""
        self.time_tracker.stop("total")
        memory_delta, memory_peak = self.memory_tracker.stop()

        total_time = self.time_tracker.spans[-1].duration if self.time_tracker.spans else 0

        result = ProfileResult(
            operation=self.operation,
            n_states=self.n_states,
            total_time=total_time,
            time_spans=list(self.time_tracker.spans),
            memory_before=self.memory_tracker._start_snapshot.current_bytes if self.memory_tracker._start_snapshot else 0,
            memory_after=self.memory_tracker.snapshots[-1].current_bytes if self.memory_tracker.snapshots else 0,
            memory_peak=memory_peak,
            memory_snapshots=list(self.memory_tracker.snapshots),
            metadata=self.metadata,
        )

        return result

    @contextmanager
    def phase(self, label: str):
        """Profile a phase within the operation."""
        self.time_tracker.start(label)
        self.memory_tracker.snapshot(f"{label}_start")
        try:
            yield
        finally:
            self.memory_tracker.snapshot(f"{label}_end")
            self.time_tracker.stop(label)

    @contextmanager
    def profile(self):
        """Context manager for full profiling."""
        self.start()
        try:
            yield self
        finally:
            pass  # Result retrieved via stop()


def profile_operation(operation_name: str):
    """
    Decorator to profile a function.

    Usage:
        @profile_operation("my_operation")
        def my_function(n_states, ...):
            ...
            return result

        result, profile = my_function(1000)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Any, ProfileResult]:
            profiler = Profiler(operation_name)
            profiler.start()

            # Extract n_states if present
            if 'n_states' in kwargs:
                profiler.n_states = kwargs['n_states']
            elif args and isinstance(args[0], int):
                profiler.n_states = args[0]

            try:
                result = func(*args, **kwargs)
            finally:
                profile_result = profiler.stop()

            return result, profile_result

        return wrapper
    return decorator


class ProfileAggregator:
    """
    Aggregate multiple profile results for statistical analysis.
    """

    def __init__(self):
        self.results: List[ProfileResult] = []

    def add(self, result: ProfileResult):
        """Add a profile result."""
        self.results.append(result)

    def summary(self) -> Dict[str, Any]:
        """Get statistical summary."""
        if not self.results:
            return {}

        times = [r.total_time for r in self.results]
        memories = [r.memory_delta_mb for r in self.results]
        throughputs = [r.throughput for r in self.results if r.throughput > 0]

        return {
            'n_runs': len(self.results),
            'time_mean_ms': statistics.mean(times) * 1000,
            'time_std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            'time_min_ms': min(times) * 1000,
            'time_max_ms': max(times) * 1000,
            'memory_mean_mb': statistics.mean(memories),
            'memory_max_mb': max(memories),
            'throughput_mean': statistics.mean(throughputs) if throughputs else 0,
        }


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    n_runs: int = 5,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark a function with multiple runs.

    Returns statistical summary.
    """
    kwargs = kwargs or {}

    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    aggregator = ProfileAggregator()

    for _ in range(n_runs):
        profiler = Profiler("benchmark")
        profiler.start()
        func(*args, **kwargs)
        result = profiler.stop()
        aggregator.add(result)

    return aggregator.summary()


def demo():
    """Demonstrate profiling capabilities."""
    print("=" * 60)
    print("PROFILER: Memory and Time Profiling")
    print("=" * 60)

    # Example: Profile list creation at scale
    def create_large_list(n: int) -> List[int]:
        return list(range(n))

    print("\n--- Profiling List Creation ---")

    for n in [10_000, 100_000, 1_000_000]:
        profiler = Profiler(f"create_list_{n}")
        profiler.n_states = n
        profiler.start()

        with profiler.phase("allocation"):
            result = create_large_list(n)

        with profiler.phase("sum"):
            total = sum(result)

        profile = profiler.stop()

        print(f"\nn={n:,}:")
        print(f"  Total time: {profile.time_ms:.2f} ms")
        print(f"  Memory delta: {profile.memory_delta_mb:.2f} MB")
        print(f"  Memory peak: {profile.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {profile.throughput:,.0f} items/sec")

        # Phase breakdown
        for span in profile.time_spans:
            if span.label != "total":
                print(f"  Phase '{span.label}': {span.duration_ms:.2f} ms")

    # Benchmark with statistics
    print("\n--- Benchmark with Statistics ---")
    summary = benchmark_function(
        create_large_list,
        args=(100_000,),
        n_runs=5,
        warmup=2,
    )
    print(f"Mean time: {summary['time_mean_ms']:.2f} +/- {summary['time_std_ms']:.2f} ms")

    return profile


if __name__ == "__main__":
    demo()
