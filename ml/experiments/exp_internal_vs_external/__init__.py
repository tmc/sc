"""
exp_internal_vs_external: Internal vs External Event Semantics

Evolves event classification (INTERNAL/EXTERNAL) and priority assignments
(CRITICAL/HIGH/NORMAL/LOW) without hardcoding.

Key concepts:
- INTERNAL events: synchronous, processed within same RTC step
- EXTERNAL events: queued, processed in subsequent macro-steps
- Priority: determines queue processing order

Uses evolutionary patterns from exp_topology_evolution.
"""

from .event_queue_evolver import (
    EventKind,
    EventPriority,
    EventDescriptor,
    QueuedEvent,
    PriorityEventQueue,
    EventPriorityGenome,
    EventPriorityEvolver,
    TracePriorityLearner,
    EvolutionConfig,
    create_random_genome,
)

__all__ = [
    "EventKind",
    "EventPriority",
    "EventDescriptor",
    "QueuedEvent",
    "PriorityEventQueue",
    "EventPriorityGenome",
    "EventPriorityEvolver",
    "TracePriorityLearner",
    "EvolutionConfig",
    "create_random_genome",
]
