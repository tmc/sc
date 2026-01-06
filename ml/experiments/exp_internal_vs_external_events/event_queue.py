"""
Event Queue - Priority-ordered event queue for statecharts.

Two types of events:
1. External events: Queued, processed in order/priority
2. Internal events: Synchronous, processed immediately (completion events)

From proto extensions/v1/annotations.proto:
- bool internal = 7;  // Internal vs external
- string priority = 9;  // "low", "normal", "high", "critical"

Queue semantics:
- External events are queued and processed in priority order
- Internal events jump the queue (processed in current step)
- FIFO within same priority level
- Higher priority events processed first
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
from collections import deque


class EventPriority(Enum):
    """Event priority levels (higher value = higher priority)."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def from_string(cls, s: str) -> 'EventPriority':
        mapping = {
            'low': cls.LOW,
            'normal': cls.NORMAL,
            'high': cls.HIGH,
            'critical': cls.CRITICAL,
        }
        return mapping.get(s.lower(), cls.NORMAL)


@dataclass
class Event:
    """An event in the statechart system."""
    label: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    internal: bool = False
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0  # For FIFO ordering within priority

    def __lt__(self, other: 'Event') -> bool:
        """Compare for priority queue (higher priority first, then FIFO)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher = first
        return self.sequence < other.sequence  # Earlier = first

    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'internal': self.internal,
            'priority': self.priority.name,
            'parameters': self.parameters,
        }


@dataclass
class EventOccurrence:
    """Record of an event being processed."""
    event: Event
    step: int
    processing_order: int
    queue_depth_at_processing: int


class EventQueue:
    """Priority queue for external events with internal event support."""

    def __init__(self):
        self._external_queue: List[Event] = []  # Heap for priority ordering
        self._internal_queue: deque[Event] = deque()  # FIFO for internal
        self._sequence_counter = 0
        self._step = 0
        self._processing_order = 0
        self._history: List[EventOccurrence] = []

    def _next_sequence(self) -> int:
        """Get next sequence number for FIFO ordering."""
        self._sequence_counter += 1
        return self._sequence_counter

    def enqueue_external(
        self,
        label: str,
        priority: EventPriority = EventPriority.NORMAL,
        parameters: Dict[str, Any] = None,
    ) -> Event:
        """Add external event to queue."""
        event = Event(
            label=label,
            parameters=parameters or {},
            internal=False,
            priority=priority,
            sequence=self._next_sequence(),
        )
        heapq.heappush(self._external_queue, event)
        return event

    def enqueue_internal(
        self,
        label: str,
        parameters: Dict[str, Any] = None,
    ) -> Event:
        """Add internal event (processed immediately)."""
        event = Event(
            label=label,
            parameters=parameters or {},
            internal=True,
            priority=EventPriority.CRITICAL,  # Internal = highest priority
            sequence=self._next_sequence(),
        )
        self._internal_queue.append(event)
        return event

    def dequeue(self) -> Optional[Event]:
        """Get next event to process (internal first, then by priority)."""
        self._processing_order += 1

        # Internal events have absolute priority
        if self._internal_queue:
            event = self._internal_queue.popleft()
        elif self._external_queue:
            event = heapq.heappop(self._external_queue)
        else:
            return None

        # Record occurrence
        occurrence = EventOccurrence(
            event=event,
            step=self._step,
            processing_order=self._processing_order,
            queue_depth_at_processing=len(self),
        )
        self._history.append(occurrence)

        return event

    def peek(self) -> Optional[Event]:
        """Look at next event without removing."""
        if self._internal_queue:
            return self._internal_queue[0]
        elif self._external_queue:
            return self._external_queue[0]
        return None

    def is_empty(self) -> bool:
        return len(self._internal_queue) == 0 and len(self._external_queue) == 0

    def __len__(self) -> int:
        return len(self._internal_queue) + len(self._external_queue)

    def advance_step(self):
        """Move to next processing step."""
        self._step += 1

    def get_history(self) -> List[EventOccurrence]:
        """Get event processing history."""
        return self._history.copy()

    def clear_history(self):
        """Clear event history."""
        self._history = []

    def get_queue_state(self) -> Dict:
        """Get current queue state for debugging."""
        return {
            'internal_count': len(self._internal_queue),
            'external_count': len(self._external_queue),
            'step': self._step,
            'total_processed': self._processing_order,
        }


class FIFOEventQueue:
    """Simple FIFO queue (no priority ordering)."""

    def __init__(self):
        self._queue: deque[Event] = deque()
        self._sequence_counter = 0

    def enqueue(self, label: str, internal: bool = False) -> Event:
        self._sequence_counter += 1
        event = Event(
            label=label,
            internal=internal,
            sequence=self._sequence_counter,
        )
        self._queue.append(event)
        return event

    def dequeue(self) -> Optional[Event]:
        return self._queue.popleft() if self._queue else None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


class StrictPriorityQueue:
    """Separate queues per priority level."""

    def __init__(self):
        self._queues: Dict[EventPriority, deque[Event]] = {
            p: deque() for p in EventPriority
        }
        self._sequence_counter = 0

    def enqueue(
        self,
        label: str,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Event:
        self._sequence_counter += 1
        event = Event(
            label=label,
            priority=priority,
            sequence=self._sequence_counter,
        )
        self._queues[priority].append(event)
        return event

    def dequeue(self) -> Optional[Event]:
        """Get highest priority event."""
        # Check from highest to lowest priority
        for priority in reversed(list(EventPriority)):
            if self._queues[priority]:
                return self._queues[priority].popleft()
        return None

    def is_empty(self) -> bool:
        return all(len(q) == 0 for q in self._queues.values())

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def get_counts(self) -> Dict[str, int]:
        return {p.name: len(q) for p, q in self._queues.items()}


class EventDispatcher:
    """Dispatch events to handlers with priority ordering."""

    def __init__(self):
        self.queue = EventQueue()
        self._handlers: Dict[str, List[Callable[[Event], None]]] = {}
        self._wildcard_handlers: List[Callable[[Event], None]] = []

    def register_handler(
        self,
        event_label: str,
        handler: Callable[[Event], None],
    ):
        """Register handler for specific event."""
        if event_label not in self._handlers:
            self._handlers[event_label] = []
        self._handlers[event_label].append(handler)

    def register_wildcard_handler(
        self,
        handler: Callable[[Event], None],
    ):
        """Register handler for all events."""
        self._wildcard_handlers.append(handler)

    def dispatch(self, event: Event):
        """Dispatch event to registered handlers."""
        # Call specific handlers
        if event.label in self._handlers:
            for handler in self._handlers[event.label]:
                handler(event)

        # Call wildcard handlers
        for handler in self._wildcard_handlers:
            handler(event)

    def send(
        self,
        label: str,
        priority: EventPriority = EventPriority.NORMAL,
        internal: bool = False,
    ) -> Event:
        """Send event to queue."""
        if internal:
            return self.queue.enqueue_internal(label)
        else:
            return self.queue.enqueue_external(label, priority)

    def process_next(self) -> Optional[Event]:
        """Process next event in queue."""
        event = self.queue.dequeue()
        if event:
            self.dispatch(event)
        return event

    def process_all(self) -> int:
        """Process all events in queue."""
        count = 0
        while not self.queue.is_empty():
            self.process_next()
            count += 1
        return count


def demo():
    """Demonstrate event queue semantics."""
    print("=" * 60)
    print("EVENT QUEUE SEMANTICS")
    print("=" * 60)

    queue = EventQueue()

    # Add events in mixed order
    print("\nAdding events:")
    queue.enqueue_external("low_event", EventPriority.LOW)
    print("  1. low_event (LOW)")
    queue.enqueue_external("normal_event", EventPriority.NORMAL)
    print("  2. normal_event (NORMAL)")
    queue.enqueue_external("high_event", EventPriority.HIGH)
    print("  3. high_event (HIGH)")
    queue.enqueue_external("critical_event", EventPriority.CRITICAL)
    print("  4. critical_event (CRITICAL)")
    queue.enqueue_external("another_normal", EventPriority.NORMAL)
    print("  5. another_normal (NORMAL)")

    print(f"\nQueue state: {queue.get_queue_state()}")

    print("\nDequeuing (should be priority order):")
    order = []
    while not queue.is_empty():
        event = queue.dequeue()
        order.append(f"{event.label} ({event.priority.name})")
        print(f"  {event.label} ({event.priority.name})")

    print("\n" + "-" * 60)
    print("INTERNAL vs EXTERNAL")
    print("-" * 60)

    queue2 = EventQueue()

    # Add external events first
    queue2.enqueue_external("ext1", EventPriority.HIGH)
    queue2.enqueue_external("ext2", EventPriority.CRITICAL)
    print("\nAdded external: ext1 (HIGH), ext2 (CRITICAL)")

    # Add internal event (should jump queue)
    queue2.enqueue_internal("internal_completion")
    print("Added internal: internal_completion")

    print("\nDequeue order:")
    while not queue2.is_empty():
        event = queue2.dequeue()
        type_str = "INTERNAL" if event.internal else "EXTERNAL"
        print(f"  {event.label} [{type_str}]")

    print("\n" + "-" * 60)
    print("EVENT DISPATCHER")
    print("-" * 60)

    dispatcher = EventDispatcher()
    received = []

    dispatcher.register_handler("click", lambda e: received.append(f"click: {e.label}"))
    dispatcher.register_handler("hover", lambda e: received.append(f"hover: {e.label}"))
    dispatcher.register_wildcard_handler(lambda e: received.append(f"* {e.label}"))

    dispatcher.send("click", EventPriority.HIGH)
    dispatcher.send("hover", EventPriority.LOW)
    dispatcher.send("click", EventPriority.NORMAL)

    print(f"\nSent 3 events, processing...")
    count = dispatcher.process_all()
    print(f"Processed {count} events")
    print(f"Received: {received}")

    return queue


if __name__ == "__main__":
    demo()
