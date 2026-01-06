"""
Priority Handler - Event priority management for statecharts.

Priority levels from proto extensions/v1/annotations.proto:
  string priority = 9;  // "low", "normal", "high", "critical"

Priority semantics:
- CRITICAL: Process immediately (internal events, errors)
- HIGH: Process before normal events (user interactions)
- NORMAL: Default processing order
- LOW: Process when queue is otherwise empty (background tasks)

Features:
- Priority inheritance (child events inherit parent priority)
- Priority boosting (age-based promotion)
- Priority inversion prevention
- Starvation prevention
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
from collections import defaultdict
import heapq

from .event_queue import Event, EventPriority, EventQueue


@dataclass
class PriorityConfig:
    """Configuration for priority handling."""
    enable_aging: bool = True
    aging_threshold_ms: float = 100.0  # Boost after 100ms
    max_age_boosts: int = 2  # Max priority levels to boost
    enable_starvation_prevention: bool = True
    starvation_threshold: int = 10  # Process low priority after N high priority


@dataclass
class PriorityMetrics:
    """Track priority-related metrics."""
    events_by_priority: Dict[EventPriority, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    wait_times_by_priority: Dict[EventPriority, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    boosts_applied: int = 0
    starvation_preventions: int = 0

    def record_event(self, event: Event, wait_time: float):
        self.events_by_priority[event.priority] += 1
        self.wait_times_by_priority[event.priority].append(wait_time)

    def avg_wait_time(self, priority: EventPriority) -> float:
        times = self.wait_times_by_priority[priority]
        return sum(times) / len(times) if times else 0.0

    def to_dict(self) -> Dict:
        return {
            'events_by_priority': {
                p.name: c for p, c in self.events_by_priority.items()
            },
            'avg_wait_times': {
                p.name: self.avg_wait_time(p) for p in EventPriority
            },
            'boosts_applied': self.boosts_applied,
            'starvation_preventions': self.starvation_preventions,
        }


class PriorityAwareQueue:
    """Event queue with priority aging and starvation prevention."""

    def __init__(self, config: PriorityConfig = None):
        self.config = config or PriorityConfig()
        self._queues: Dict[EventPriority, List[Tuple[float, int, Event]]] = {
            p: [] for p in EventPriority
        }
        self._sequence = 0
        self._high_priority_streak = 0
        self.metrics = PriorityMetrics()

    def enqueue(
        self,
        label: str,
        priority: EventPriority = EventPriority.NORMAL,
        parameters: Dict[str, Any] = None,
    ) -> Event:
        """Add event to priority queue."""
        self._sequence += 1
        event = Event(
            label=label,
            parameters=parameters or {},
            priority=priority,
            timestamp=time.time(),
            sequence=self._sequence,
        )
        # Store as (timestamp, sequence, event) for heap ordering
        heapq.heappush(
            self._queues[priority],
            (event.timestamp, event.sequence, event)
        )
        return event

    def _apply_aging(self):
        """Boost priority of old events."""
        if not self.config.enable_aging:
            return

        now = time.time()
        threshold = self.config.aging_threshold_ms / 1000.0

        for priority in [EventPriority.LOW, EventPriority.NORMAL, EventPriority.HIGH]:
            queue = self._queues[priority]
            to_boost = []

            # Find events to boost
            new_queue = []
            for ts, seq, event in queue:
                age = now - ts
                if age > threshold and priority.value < EventPriority.CRITICAL.value:
                    to_boost.append((ts, seq, event))
                else:
                    new_queue.append((ts, seq, event))

            self._queues[priority] = new_queue
            heapq.heapify(self._queues[priority])

            # Boost to next priority level
            if to_boost:
                next_priority = EventPriority(min(priority.value + 1, EventPriority.CRITICAL.value))
                for ts, seq, event in to_boost:
                    event.priority = next_priority
                    heapq.heappush(self._queues[next_priority], (ts, seq, event))
                    self.metrics.boosts_applied += 1

    def _check_starvation(self) -> Optional[EventPriority]:
        """Check if low priority events are starving."""
        if not self.config.enable_starvation_prevention:
            return None

        if self._high_priority_streak >= self.config.starvation_threshold:
            # Check if lower priority events are waiting
            for priority in [EventPriority.LOW, EventPriority.NORMAL]:
                if self._queues[priority]:
                    self._high_priority_streak = 0
                    self.metrics.starvation_preventions += 1
                    return priority
        return None

    def dequeue(self) -> Optional[Event]:
        """Get next event respecting priority and anti-starvation."""
        self._apply_aging()

        # Check for starvation prevention
        force_priority = self._check_starvation()

        # Find highest priority non-empty queue
        for priority in reversed(list(EventPriority)):
            if force_priority is not None and priority != force_priority:
                if priority.value > force_priority.value:
                    continue  # Skip higher priority to prevent starvation

            if self._queues[priority]:
                ts, seq, event = heapq.heappop(self._queues[priority])
                wait_time = time.time() - ts
                self.metrics.record_event(event, wait_time)

                # Track high priority streak
                if priority.value >= EventPriority.HIGH.value:
                    self._high_priority_streak += 1
                else:
                    self._high_priority_streak = 0

                return event

        return None

    def is_empty(self) -> bool:
        return all(len(q) == 0 for q in self._queues.values())

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def get_counts(self) -> Dict[str, int]:
        return {p.name: len(q) for p, q in self._queues.items()}


class PriorityInheritanceManager:
    """Manage priority inheritance between related events."""

    def __init__(self):
        self._event_parents: Dict[int, Event] = {}  # event.sequence -> parent
        self._inherited_priorities: Dict[int, EventPriority] = {}

    def set_parent(self, child: Event, parent: Event):
        """Establish parent-child relationship."""
        self._event_parents[child.sequence] = parent
        # Inherit higher priority from parent
        if parent.priority.value > child.priority.value:
            child.priority = parent.priority
            self._inherited_priorities[child.sequence] = parent.priority

    def get_effective_priority(self, event: Event) -> EventPriority:
        """Get effective priority considering inheritance."""
        if event.sequence in self._inherited_priorities:
            return self._inherited_priorities[event.sequence]
        return event.priority


class PriorityTransitionHandler:
    """Handle priority for transition-related events."""

    # Priority mapping for common event types
    EVENT_TYPE_PRIORITIES = {
        'error': EventPriority.CRITICAL,
        'timeout': EventPriority.HIGH,
        'user_action': EventPriority.HIGH,
        'completion': EventPriority.CRITICAL,  # Internal
        'update': EventPriority.NORMAL,
        'refresh': EventPriority.LOW,
        'analytics': EventPriority.LOW,
    }

    def __init__(self, queue: PriorityAwareQueue):
        self.queue = queue
        self._type_overrides: Dict[str, EventPriority] = {}

    def set_event_type_priority(self, event_type: str, priority: EventPriority):
        """Override priority for event type."""
        self._type_overrides[event_type] = priority

    def get_priority_for_type(self, event_type: str) -> EventPriority:
        """Get priority for event type."""
        if event_type in self._type_overrides:
            return self._type_overrides[event_type]
        return self.EVENT_TYPE_PRIORITIES.get(event_type, EventPriority.NORMAL)

    def send_typed_event(
        self,
        label: str,
        event_type: str = None,
        priority: EventPriority = None,
    ) -> Event:
        """Send event with type-based priority."""
        if priority is None:
            priority = self.get_priority_for_type(event_type or label)
        return self.queue.enqueue(label, priority)


class PriorityScheduler:
    """Schedule events with priority-aware timing."""

    def __init__(self, queue: PriorityAwareQueue):
        self.queue = queue
        self._scheduled: List[Tuple[float, str, EventPriority]] = []

    def schedule(
        self,
        label: str,
        delay_ms: float,
        priority: EventPriority = EventPriority.NORMAL,
    ):
        """Schedule event for future delivery."""
        deliver_at = time.time() + (delay_ms / 1000.0)
        heapq.heappush(self._scheduled, (deliver_at, label, priority))

    def check_scheduled(self) -> List[Event]:
        """Check and enqueue any ready scheduled events."""
        now = time.time()
        ready = []

        while self._scheduled and self._scheduled[0][0] <= now:
            deliver_at, label, priority = heapq.heappop(self._scheduled)
            event = self.queue.enqueue(label, priority)
            ready.append(event)

        return ready

    def pending_count(self) -> int:
        return len(self._scheduled)


def demo():
    """Demonstrate priority handling."""
    print("=" * 60)
    print("PRIORITY HANDLER")
    print("=" * 60)

    # Basic priority queue
    print("\n--- Priority Queue ---")
    config = PriorityConfig(enable_aging=False, enable_starvation_prevention=False)
    queue = PriorityAwareQueue(config)

    # Add events in mixed priority order
    queue.enqueue("low_task", EventPriority.LOW)
    queue.enqueue("normal_task1", EventPriority.NORMAL)
    queue.enqueue("critical_alert", EventPriority.CRITICAL)
    queue.enqueue("high_action", EventPriority.HIGH)
    queue.enqueue("normal_task2", EventPriority.NORMAL)

    print("Added 5 events with mixed priorities")
    print(f"Queue counts: {queue.get_counts()}")

    print("\nDequeue order:")
    order = []
    while not queue.is_empty():
        event = queue.dequeue()
        order.append(f"{event.label} ({event.priority.name})")
        print(f"  {event.label} ({event.priority.name})")

    # Starvation prevention
    print("\n--- Starvation Prevention ---")
    config2 = PriorityConfig(
        enable_aging=False,
        enable_starvation_prevention=True,
        starvation_threshold=3,
    )
    queue2 = PriorityAwareQueue(config2)

    # Add many high priority and one low priority
    queue2.enqueue("low_waiting", EventPriority.LOW)
    for i in range(5):
        queue2.enqueue(f"high_{i}", EventPriority.HIGH)

    print("Added 1 LOW + 5 HIGH priority events")
    print("Starvation threshold: 3 high priority events\n")

    print("Dequeue order (low should be forced after 3 high):")
    while not queue2.is_empty():
        event = queue2.dequeue()
        forced = "(STARVATION PREVENTION)" if event.priority == EventPriority.LOW else ""
        print(f"  {event.label} ({event.priority.name}) {forced}")

    print(f"\nMetrics: {queue2.metrics.to_dict()}")

    # Type-based priorities
    print("\n--- Type-Based Priorities ---")
    queue3 = PriorityAwareQueue()
    handler = PriorityTransitionHandler(queue3)

    handler.send_typed_event("click", "user_action")
    handler.send_typed_event("page_view", "analytics")
    handler.send_typed_event("crash", "error")
    handler.send_typed_event("token_expired", "timeout")

    print("Sent events with type-based priorities")
    print("\nDequeue order:")
    while not queue3.is_empty():
        event = queue3.dequeue()
        print(f"  {event.label} ({event.priority.name})")

    return queue, queue2, queue3


if __name__ == "__main__":
    demo()
