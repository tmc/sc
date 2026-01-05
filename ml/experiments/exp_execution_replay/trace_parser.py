"""
Trace Parser - Parse ExecutionTrace protos for offline learning.

Parses execution traces from proto/statecharts/v1/execution.proto format.
Uses Python dataclasses that mirror the proto structure.

Key types:
- ExecutionTrace: Complete execution history
- TransitionLogEntry: Single step (event, source->target config)
- Configuration: Set of active states
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from enum import Enum, auto
import json
import hashlib
import random


# =============================================================================
# Mirror Proto Types (Pure Python, no protobuf dependency)
# =============================================================================

@dataclass
class Configuration:
    """
    Set of active states.

    Mirrors proto Configuration message.
    σ ∈ P(S) - a subset of all states S.
    """
    active_states: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(tuple(sorted(self.active_states)))

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        return set(self.active_states) == set(other.active_states)

    def to_tuple(self) -> Tuple[str, ...]:
        return tuple(sorted(self.active_states))

    @classmethod
    def from_states(cls, *states: str) -> "Configuration":
        return cls(active_states=list(states))


@dataclass
class Event:
    """
    Triggering event.

    Mirrors proto Event message.
    """
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class TransitionRef:
    """
    Reference to a transition that fired.

    Mirrors proto TransitionRef message.
    """
    label: str
    from_states: List[str] = field(default_factory=list)
    to_states: List[str] = field(default_factory=list)
    event: str = ""


@dataclass
class GuardEvaluation:
    """
    Result of evaluating a guard.

    Mirrors proto GuardEvaluation message.
    """
    guard_expression: str
    result: bool
    bound_values: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class ActionExecution:
    """
    Executed action record.

    Mirrors proto ActionExecution message.
    """
    action_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class TransitionLogEntry:
    """
    Single step in execution trace.

    Mirrors proto TransitionLogEntry message.
    L = (t, seq, e, σ, σ', T, G, A, Γ, Γ')
    """
    id: str
    timestamp: float
    sequence: int

    # Event that triggered this step
    trigger_event: Optional[Event] = None

    # Source and target configurations
    source_config: Optional[Configuration] = None
    target_config: Optional[Configuration] = None

    # What happened
    transitions_fired: List[TransitionRef] = field(default_factory=list)
    guard_results: List[GuardEvaluation] = field(default_factory=list)
    actions_executed: List[ActionExecution] = field(default_factory=list)

    # Context before and after
    context_before: Dict[str, Any] = field(default_factory=dict)
    context_after: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    processing_time_ms: float = 0.0
    error: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceMetadata:
    """
    Trace summary statistics.

    Mirrors proto TraceMetadata message.
    """
    started_at: float = 0.0
    ended_at: float = 0.0
    total_transitions: int = 0
    total_events_processed: int = 0
    total_events_ignored: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    """
    Complete execution history.

    Mirrors proto ExecutionTrace message.
    τ = (id, σ₀, Γ₀, L*, σₙ, Γₙ)
    """
    trace_id: str
    machine_id: str = ""

    # Initial state
    initial_config: Optional[Configuration] = None
    initial_context: Dict[str, Any] = field(default_factory=dict)

    # Log entries
    entries: List[TransitionLogEntry] = field(default_factory=list)

    # Final state
    final_config: Optional[Configuration] = None
    final_context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Optional[TraceMetadata] = None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TransitionLogEntry]:
        return iter(self.entries)

    def get_state_sequence(self) -> List[Configuration]:
        """Get sequence of configurations: σ₀ → σ₁ → ... → σₙ"""
        configs = []
        if self.initial_config:
            configs.append(self.initial_config)
        for entry in self.entries:
            if entry.target_config:
                configs.append(entry.target_config)
        return configs

    def get_transition_pairs(self) -> List[Tuple[Configuration, str, Configuration]]:
        """Get (source, event, target) triples for learning."""
        pairs = []
        for entry in self.entries:
            if entry.source_config and entry.target_config and entry.trigger_event:
                pairs.append((
                    entry.source_config,
                    entry.trigger_event.event_type,
                    entry.target_config
                ))
        return pairs


# =============================================================================
# Trace Parser
# =============================================================================

class TraceParser:
    """
    Parse execution traces from various formats.

    Supports:
    - JSON (serialized proto or direct)
    - In-memory Python objects
    - Synthetic trace generation for testing
    """

    def __init__(self):
        self.traces: List[ExecutionTrace] = []
        self.stats = {
            "traces_parsed": 0,
            "entries_parsed": 0,
            "errors": 0,
        }

    def parse_json(self, json_str: str) -> ExecutionTrace:
        """Parse trace from JSON string."""
        data = json.loads(json_str)
        return self.parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> ExecutionTrace:
        """Parse trace from dictionary."""
        # Parse initial config
        initial_config = None
        if "initial_config" in data:
            ic = data["initial_config"]
            initial_config = Configuration(
                active_states=ic.get("active_states", [])
            )

        # Parse entries
        entries = []
        for entry_data in data.get("entries", []):
            entry = self._parse_entry(entry_data)
            entries.append(entry)

        # Parse final config
        final_config = None
        if "final_config" in data:
            fc = data["final_config"]
            final_config = Configuration(
                active_states=fc.get("active_states", [])
            )

        # Parse metadata
        metadata = None
        if "metadata" in data:
            md = data["metadata"]
            metadata = TraceMetadata(
                started_at=md.get("started_at", 0.0),
                ended_at=md.get("ended_at", 0.0),
                total_transitions=md.get("total_transitions", len(entries)),
                total_events_processed=md.get("total_events_processed", len(entries)),
            )

        trace = ExecutionTrace(
            trace_id=data.get("trace_id", self._generate_id()),
            machine_id=data.get("machine_id", ""),
            initial_config=initial_config,
            initial_context=data.get("initial_context", {}),
            entries=entries,
            final_config=final_config,
            final_context=data.get("final_context", {}),
            metadata=metadata,
        )

        self.traces.append(trace)
        self.stats["traces_parsed"] += 1
        self.stats["entries_parsed"] += len(entries)

        return trace

    def _parse_entry(self, data: Dict[str, Any]) -> TransitionLogEntry:
        """Parse single log entry."""
        # Parse event
        trigger_event = None
        if "trigger_event" in data:
            te = data["trigger_event"]
            trigger_event = Event(
                event_type=te.get("event_type", "unknown"),
                payload=te.get("payload", {}),
                timestamp=te.get("timestamp", 0.0),
            )

        # Parse configs
        source_config = None
        if "source_config" in data:
            sc = data["source_config"]
            source_config = Configuration(
                active_states=sc.get("active_states", [])
            )

        target_config = None
        if "target_config" in data:
            tc = data["target_config"]
            target_config = Configuration(
                active_states=tc.get("active_states", [])
            )

        # Parse transitions fired
        transitions_fired = []
        for tr in data.get("transitions_fired", []):
            transitions_fired.append(TransitionRef(
                label=tr.get("label", ""),
                from_states=tr.get("from", []),
                to_states=tr.get("to", []),
                event=tr.get("event", ""),
            ))

        # Parse guard results
        guard_results = []
        for gr in data.get("guard_results", []):
            guard_results.append(GuardEvaluation(
                guard_expression=gr.get("guard_expression", ""),
                result=gr.get("result", False),
                bound_values=gr.get("bound_values", {}),
            ))

        # Parse actions
        actions_executed = []
        for ae in data.get("actions_executed", []):
            actions_executed.append(ActionExecution(
                action_name=ae.get("action_name", ""),
                parameters=ae.get("parameters", {}),
                duration_ms=ae.get("duration_ms", 0.0),
            ))

        return TransitionLogEntry(
            id=data.get("id", self._generate_id()),
            timestamp=data.get("timestamp", 0.0),
            sequence=data.get("sequence", 0),
            trigger_event=trigger_event,
            source_config=source_config,
            target_config=target_config,
            transitions_fired=transitions_fired,
            guard_results=guard_results,
            actions_executed=actions_executed,
            context_before=data.get("context_before", {}),
            context_after=data.get("context_after", {}),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
        )

    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.md5(str(random.random()).encode()).hexdigest()[:12]

    def get_all_states(self) -> Set[str]:
        """Get all unique states across all traces."""
        states = set()
        for trace in self.traces:
            if trace.initial_config:
                states.update(trace.initial_config.active_states)
            for entry in trace.entries:
                if entry.source_config:
                    states.update(entry.source_config.active_states)
                if entry.target_config:
                    states.update(entry.target_config.active_states)
        return states

    def get_all_events(self) -> Set[str]:
        """Get all unique events across all traces."""
        events = set()
        for trace in self.traces:
            for entry in trace.entries:
                if entry.trigger_event:
                    events.add(entry.trigger_event.event_type)
        return events

    def get_transition_counts(self) -> Dict[Tuple[str, str, str], int]:
        """Count (source, event, target) occurrences."""
        counts: Dict[Tuple[str, str, str], int] = {}
        for trace in self.traces:
            for source, event, target in trace.get_transition_pairs():
                # For simplicity, use first active state
                src = source.active_states[0] if source.active_states else "UNKNOWN"
                tgt = target.active_states[0] if target.active_states else "UNKNOWN"
                key = (src, event, tgt)
                counts[key] = counts.get(key, 0) + 1
        return counts


# =============================================================================
# Synthetic Trace Generator
# =============================================================================

class SyntheticTraceGenerator:
    """
    Generate synthetic execution traces for testing.

    Simulates statechart execution to create realistic traces.
    """

    def __init__(
        self,
        states: List[str],
        events: List[str],
        transitions: List[Tuple[str, str, str]],  # (source, event, target)
    ):
        self.states = states
        self.events = events
        self.transitions = transitions

        # Build transition map: (source, event) -> [targets]
        self.transition_map: Dict[Tuple[str, str], List[str]] = {}
        for src, evt, tgt in transitions:
            key = (src, evt)
            if key not in self.transition_map:
                self.transition_map[key] = []
            self.transition_map[key].append(tgt)

    def generate_trace(
        self,
        initial_state: str,
        num_steps: int = 20,
        noise_prob: float = 0.0,
    ) -> ExecutionTrace:
        """Generate a single execution trace."""
        trace_id = hashlib.md5(str(random.random()).encode()).hexdigest()[:12]

        current_state = initial_state
        entries = []

        for seq in range(num_steps):
            # Pick random event
            event_type = random.choice(self.events)

            # Find valid transition
            key = (current_state, event_type)
            if key in self.transition_map:
                targets = self.transition_map[key]
                target_state = random.choice(targets)
            else:
                # No transition - stay in same state (or skip)
                if random.random() < 0.3:
                    continue
                target_state = current_state

            # Add noise (wrong transitions for testing robustness)
            if noise_prob > 0 and random.random() < noise_prob:
                target_state = random.choice(self.states)

            # Create entry
            entry = TransitionLogEntry(
                id=f"entry_{seq}",
                timestamp=float(seq) * 0.1,
                sequence=seq,
                trigger_event=Event(event_type=event_type),
                source_config=Configuration.from_states(current_state),
                target_config=Configuration.from_states(target_state),
                transitions_fired=[TransitionRef(
                    label=f"{current_state}_to_{target_state}",
                    from_states=[current_state],
                    to_states=[target_state],
                    event=event_type,
                )],
            )
            entries.append(entry)
            current_state = target_state

        return ExecutionTrace(
            trace_id=trace_id,
            machine_id="synthetic",
            initial_config=Configuration.from_states(initial_state),
            entries=entries,
            final_config=Configuration.from_states(current_state),
            metadata=TraceMetadata(
                total_transitions=len(entries),
                total_events_processed=len(entries),
            ),
        )

    def generate_traces(
        self,
        num_traces: int,
        steps_per_trace: int = 20,
        noise_prob: float = 0.0,
    ) -> List[ExecutionTrace]:
        """Generate multiple traces."""
        traces = []
        for _ in range(num_traces):
            initial = random.choice(self.states)
            trace = self.generate_trace(initial, steps_per_trace, noise_prob)
            traces.append(trace)
        return traces


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate trace parsing and generation."""
    print("=" * 60)
    print("Execution Trace Parser Demo")
    print("=" * 60)

    # Define simple statechart
    states = ["IDLE", "RUNNING", "PAUSED", "STOPPED"]
    events = ["start", "pause", "resume", "stop", "reset"]
    transitions = [
        ("IDLE", "start", "RUNNING"),
        ("RUNNING", "pause", "PAUSED"),
        ("RUNNING", "stop", "STOPPED"),
        ("PAUSED", "resume", "RUNNING"),
        ("PAUSED", "stop", "STOPPED"),
        ("STOPPED", "reset", "IDLE"),
    ]

    # Generate synthetic traces
    print("\n--- Generating Synthetic Traces ---")
    generator = SyntheticTraceGenerator(states, events, transitions)
    traces = generator.generate_traces(num_traces=10, steps_per_trace=15)

    print(f"Generated {len(traces)} traces")

    # Analyze traces
    parser = TraceParser()
    for trace in traces:
        parser.traces.append(trace)
        parser.stats["traces_parsed"] += 1
        parser.stats["entries_parsed"] += len(trace.entries)

    print(f"\n--- Trace Analysis ---")
    print(f"Total traces: {len(parser.traces)}")
    print(f"Total entries: {parser.stats['entries_parsed']}")
    print(f"Unique states: {parser.get_all_states()}")
    print(f"Unique events: {parser.get_all_events()}")

    # Transition counts
    print(f"\n--- Transition Counts ---")
    counts = parser.get_transition_counts()
    for (src, evt, tgt), count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src} --[{evt}]--> {tgt}: {count}")

    # Show sample trace
    print(f"\n--- Sample Trace ---")
    sample = traces[0]
    print(f"Trace ID: {sample.trace_id}")
    print(f"Initial: {sample.initial_config.active_states}")
    print(f"Steps: {len(sample.entries)}")
    print(f"Final: {sample.final_config.active_states}")

    print("\nFirst 5 transitions:")
    for entry in sample.entries[:5]:
        src = entry.source_config.active_states[0] if entry.source_config else "?"
        tgt = entry.target_config.active_states[0] if entry.target_config else "?"
        evt = entry.trigger_event.event_type if entry.trigger_event else "?"
        print(f"  {src} --[{evt}]--> {tgt}")

    return parser, traces


if __name__ == "__main__":
    demo()
