"""
Internal Event - Synchronous event handling for statecharts.

Internal events are processed synchronously within the current step:
1. Completion events (τ) - generated when final state reached
2. Raised events - explicitly raised during action execution
3. Error events - generated on errors

Semantics:
- Internal events processed BEFORE next external event
- Internal events can trigger more internal events (chain)
- Run-to-completion: external events wait for all internal processing

FORMAL DEFINITION [Harel]:
- σ →τ σ' : completion transition (internal)
- σ →e σ' : event-triggered transition (external)
- Internal events are processed until stable (no more enabled τ)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
from collections import deque

from .event_queue import Event, EventPriority, EventQueue


class InternalEventType(Enum):
    """Types of internal events."""
    COMPLETION = auto()     # τ event - state reached final
    RAISED = auto()         # Explicitly raised in action
    ERROR = auto()          # Error condition
    TIMEOUT = auto()        # Timer expired
    CHANGE = auto()         # Data change notification


@dataclass
class InternalEvent:
    """An internal (synchronous) event."""
    label: str
    event_type: InternalEventType
    source_state: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    chain_depth: int = 0  # How deep in internal event chain

    def to_event(self) -> Event:
        """Convert to generic Event."""
        return Event(
            label=self.label,
            parameters=self.parameters,
            internal=True,
            priority=EventPriority.CRITICAL,
        )


@dataclass
class InternalEventChain:
    """Track chain of internal events."""
    events: List[InternalEvent] = field(default_factory=list)
    max_depth: int = 0
    started_at_step: int = 0

    def add(self, event: InternalEvent):
        self.events.append(event)
        self.max_depth = max(self.max_depth, event.chain_depth)

    def __len__(self) -> int:
        return len(self.events)


class InternalEventProcessor:
    """Process internal events synchronously."""

    MAX_CHAIN_DEPTH = 100  # Prevent infinite loops

    def __init__(self):
        self._pending: deque[InternalEvent] = deque()
        self._current_depth = 0
        self._chains: List[InternalEventChain] = []
        self._current_chain: Optional[InternalEventChain] = None
        self._handlers: Dict[str, List[Callable[[InternalEvent], List[InternalEvent]]]] = {}
        self._step = 0

    def register_handler(
        self,
        event_label: str,
        handler: Callable[[InternalEvent], List[InternalEvent]],
    ):
        """
        Register handler for internal event.
        Handler returns list of new internal events to raise.
        """
        if event_label not in self._handlers:
            self._handlers[event_label] = []
        self._handlers[event_label].append(handler)

    def raise_event(
        self,
        label: str,
        event_type: InternalEventType = InternalEventType.RAISED,
        source_state: str = None,
        parameters: Dict[str, Any] = None,
    ) -> InternalEvent:
        """Raise an internal event."""
        event = InternalEvent(
            label=label,
            event_type=event_type,
            source_state=source_state,
            parameters=parameters or {},
            chain_depth=self._current_depth,
        )
        self._pending.append(event)
        return event

    def raise_completion(self, source_state: str) -> InternalEvent:
        """Raise completion (τ) event."""
        return self.raise_event(
            label="τ",
            event_type=InternalEventType.COMPLETION,
            source_state=source_state,
        )

    def raise_error(self, error_message: str, source_state: str = None) -> InternalEvent:
        """Raise error event."""
        return self.raise_event(
            label="error",
            event_type=InternalEventType.ERROR,
            source_state=source_state,
            parameters={"message": error_message},
        )

    def process_all(self) -> InternalEventChain:
        """
        Process all pending internal events until stable.
        Returns the event chain that was processed.
        """
        self._step += 1
        self._current_chain = InternalEventChain(started_at_step=self._step)

        while self._pending and self._current_depth < self.MAX_CHAIN_DEPTH:
            event = self._pending.popleft()
            self._current_chain.add(event)
            self._current_depth = event.chain_depth + 1

            # Call handlers
            if event.label in self._handlers:
                for handler in self._handlers[event.label]:
                    new_events = handler(event)
                    for new_event in new_events:
                        new_event.chain_depth = self._current_depth
                        self._pending.append(new_event)

        if self._current_depth >= self.MAX_CHAIN_DEPTH:
            self.raise_error(f"Internal event chain exceeded max depth {self.MAX_CHAIN_DEPTH}")

        chain = self._current_chain
        self._chains.append(chain)
        self._current_chain = None
        self._current_depth = 0

        return chain

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def get_chains(self) -> List[InternalEventChain]:
        return self._chains.copy()

    def clear_chains(self):
        self._chains = []


class RunToCompletionProcessor:
    """
    Run-to-completion semantics processor.

    RTC step:
    1. Take next external event from queue
    2. Process all resulting internal events
    3. Reach stable configuration
    4. Then take next external event
    """

    def __init__(self):
        self.external_queue = EventQueue()
        self.internal_processor = InternalEventProcessor()
        self._step = 0
        self._rtc_steps: List[Dict] = []

    def send_external(
        self,
        label: str,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Event:
        """Send external event to queue."""
        return self.external_queue.enqueue_external(label, priority)

    def raise_internal(
        self,
        label: str,
        event_type: InternalEventType = InternalEventType.RAISED,
    ) -> InternalEvent:
        """Raise internal event."""
        return self.internal_processor.raise_event(label, event_type)

    def register_external_handler(
        self,
        event_label: str,
        handler: Callable[[Event], List[InternalEvent]],
    ):
        """Register handler for external event that may raise internal events."""
        # Store for external processing
        if not hasattr(self, '_external_handlers'):
            self._external_handlers: Dict[str, List[Callable]] = {}
        if event_label not in self._external_handlers:
            self._external_handlers[event_label] = []
        self._external_handlers[event_label].append(handler)

    def register_internal_handler(
        self,
        event_label: str,
        handler: Callable[[InternalEvent], List[InternalEvent]],
    ):
        """Register handler for internal event."""
        self.internal_processor.register_handler(event_label, handler)

    def step(self) -> Optional[Dict]:
        """
        Execute one RTC step.
        Returns step info or None if no events.
        """
        if self.external_queue.is_empty() and not self.internal_processor.has_pending():
            return None

        self._step += 1
        step_info = {
            'step': self._step,
            'external_event': None,
            'internal_chain': None,
        }

        # Process any pending internal events first
        if self.internal_processor.has_pending():
            chain = self.internal_processor.process_all()
            step_info['internal_chain'] = chain
        # Then take next external event
        elif not self.external_queue.is_empty():
            event = self.external_queue.dequeue()
            step_info['external_event'] = event

            # Call external handlers (may raise internal events)
            if hasattr(self, '_external_handlers') and event.label in self._external_handlers:
                for handler in self._external_handlers[event.label]:
                    internal_events = handler(event)
                    for ie in internal_events:
                        self.internal_processor._pending.append(ie)

            # Process resulting internal events
            if self.internal_processor.has_pending():
                chain = self.internal_processor.process_all()
                step_info['internal_chain'] = chain

        self._rtc_steps.append(step_info)
        return step_info

    def run_to_completion(self) -> List[Dict]:
        """Run until all events processed."""
        steps = []
        while True:
            step_info = self.step()
            if step_info is None:
                break
            steps.append(step_info)
        return steps

    def get_steps(self) -> List[Dict]:
        return self._rtc_steps.copy()


class CompletionEventGenerator:
    """Generate completion events based on state machine status."""

    def __init__(self, processor: InternalEventProcessor):
        self.processor = processor
        self._final_states: Set[str] = set()
        self._active_states: Set[str] = set()

    def register_final_state(self, state_label: str):
        """Register a state as final."""
        self._final_states.add(state_label)

    def enter_state(self, state_label: str):
        """Called when state is entered."""
        self._active_states.add(state_label)

        # Generate completion if final
        if state_label in self._final_states:
            self.processor.raise_completion(state_label)

    def exit_state(self, state_label: str):
        """Called when state is exited."""
        self._active_states.discard(state_label)

    def is_active(self, state_label: str) -> bool:
        return state_label in self._active_states


def demo():
    """Demonstrate internal event semantics."""
    print("=" * 60)
    print("INTERNAL EVENT SEMANTICS")
    print("=" * 60)

    # Simple internal event chain
    print("\n--- Internal Event Chain ---")
    processor = InternalEventProcessor()

    chain_log = []

    # Handler that raises another internal event
    def cascade_handler(event: InternalEvent) -> List[InternalEvent]:
        chain_log.append(f"Handled: {event.label} (depth={event.chain_depth})")
        if event.chain_depth < 3:
            return [InternalEvent(
                label=f"cascade_{event.chain_depth + 1}",
                event_type=InternalEventType.RAISED,
                chain_depth=event.chain_depth + 1,
            )]
        return []

    processor.register_handler("cascade_0", cascade_handler)
    processor.register_handler("cascade_1", cascade_handler)
    processor.register_handler("cascade_2", cascade_handler)
    processor.register_handler("cascade_3", cascade_handler)

    processor.raise_event("cascade_0", InternalEventType.RAISED)
    chain = processor.process_all()

    print(f"Chain length: {len(chain)}")
    print(f"Max depth: {chain.max_depth}")
    for entry in chain_log:
        print(f"  {entry}")

    # Run-to-completion
    print("\n--- Run-to-Completion Semantics ---")
    rtc = RunToCompletionProcessor()

    rtc_log = []

    # External handler that raises internal events
    def click_handler(event: Event) -> List[InternalEvent]:
        rtc_log.append(f"External: {event.label}")
        return [
            InternalEvent(label="validate", event_type=InternalEventType.RAISED),
            InternalEvent(label="update", event_type=InternalEventType.RAISED),
        ]

    # Internal handlers
    def validate_handler(event: InternalEvent) -> List[InternalEvent]:
        rtc_log.append(f"Internal: {event.label}")
        return [InternalEvent(label="validated", event_type=InternalEventType.RAISED)]

    def update_handler(event: InternalEvent) -> List[InternalEvent]:
        rtc_log.append(f"Internal: {event.label}")
        return []

    def validated_handler(event: InternalEvent) -> List[InternalEvent]:
        rtc_log.append(f"Internal: {event.label}")
        return []

    rtc.register_external_handler("click", click_handler)
    rtc.register_internal_handler("validate", validate_handler)
    rtc.register_internal_handler("update", update_handler)
    rtc.register_internal_handler("validated", validated_handler)

    # Send two external events
    rtc.send_external("click", EventPriority.NORMAL)
    rtc.send_external("click", EventPriority.NORMAL)

    print("\nProcessing...")
    steps = rtc.run_to_completion()

    print(f"Total RTC steps: {len(steps)}")
    for i, step in enumerate(steps):
        print(f"\n  Step {i + 1}:")
        if step['external_event']:
            print(f"    External: {step['external_event'].label}")
        if step['internal_chain']:
            chain = step['internal_chain']
            print(f"    Internal chain: {len(chain)} events, depth={chain.max_depth}")

    print("\nExecution log:")
    for entry in rtc_log:
        print(f"  {entry}")

    return processor, rtc


if __name__ == "__main__":
    demo()
