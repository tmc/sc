"""
Online vs Offline Learning Comparison

Compare sample efficiency of:
1. Online Learning: Learn while interacting with environment
2. Offline Learning: Learn from pre-collected execution traces

Metrics:
- Sample efficiency: Accuracy vs number of transitions observed
- Learning speed: Time to reach target accuracy
- Generalization: Performance on unseen transitions
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import random
import time
import math

from .trace_parser import (
    ExecutionTrace, TransitionLogEntry, Configuration, Event,
    SyntheticTraceGenerator
)
from .offline_learner import (
    DirectExtractionLearner, LearnedStatechart, LearnedTransition
)


# =============================================================================
# Simulated Environment for Online Learning
# =============================================================================

class StatechartEnvironment:
    """
    Simulated statechart environment for online learning.

    Agent can observe current state and take actions (events).
    Environment responds with new state.
    """

    def __init__(
        self,
        states: List[str],
        events: List[str],
        transitions: List[Tuple[str, str, str]],  # (source, event, target)
        initial_state: str,
    ):
        self.states = states
        self.events = events
        self.initial_state = initial_state
        self.current_state = initial_state

        # Build transition map
        self.transition_map: Dict[Tuple[str, str], str] = {}
        for src, evt, tgt in transitions:
            self.transition_map[(src, evt)] = tgt

        self.step_count = 0

    def reset(self) -> str:
        """Reset to initial state."""
        self.current_state = self.initial_state
        self.step_count = 0
        return self.current_state

    def step(self, event: str) -> Tuple[str, str, bool]:
        """
        Take action (event), return (old_state, new_state, valid).

        valid=False if no transition exists for (current_state, event).
        """
        old_state = self.current_state
        key = (self.current_state, event)

        if key in self.transition_map:
            self.current_state = self.transition_map[key]
            valid = True
        else:
            valid = False

        self.step_count += 1
        return old_state, self.current_state, valid

    def get_valid_events(self) -> List[str]:
        """Get events that have transitions from current state."""
        valid = []
        for evt in self.events:
            if (self.current_state, evt) in self.transition_map:
                valid.append(evt)
        return valid


# =============================================================================
# Online Learner
# =============================================================================

class OnlineLearner:
    """
    Learn statechart while interacting with environment.

    Uses exploration (random) and exploitation (known transitions).
    """

    def __init__(self, exploration_rate: float = 0.3):
        self.exploration_rate = exploration_rate

        # Learning counts
        self.transition_counts: Dict[Tuple[str, str, str], int] = {}
        self.source_event_counts: Dict[Tuple[str, str], int] = {}
        self.state_counts: Dict[str, int] = {}

        # Seen states and events
        self.known_states: Set[str] = set()
        self.known_events: Set[str] = set()

        self.total_steps = 0
        self.valid_steps = 0

    def select_action(self, current_state: str, available_events: List[str]) -> str:
        """Select action using epsilon-greedy strategy."""
        if not available_events:
            return random.choice(list(self.known_events)) if self.known_events else "unknown"

        if random.random() < self.exploration_rate:
            # Explore: random event
            return random.choice(available_events)
        else:
            # Exploit: prefer events we've seen work
            for evt in available_events:
                if (current_state, evt) in self.source_event_counts:
                    return evt
            return random.choice(available_events)

    def observe(self, source: str, event: str, target: str, valid: bool):
        """Observe a transition."""
        self.total_steps += 1
        self.known_states.add(source)
        self.known_states.add(target)
        self.known_events.add(event)

        if valid:
            self.valid_steps += 1
            key = (source, event, target)
            self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

            se_key = (source, event)
            self.source_event_counts[se_key] = self.source_event_counts.get(se_key, 0) + 1

        self.state_counts[source] = self.state_counts.get(source, 0) + 1

    def get_learned_chart(self) -> LearnedStatechart:
        """Get current learned statechart."""
        chart = LearnedStatechart()

        for state in self.known_states:
            from .offline_learner import LearnedState
            chart.states[state] = LearnedState(name=state)

        for (src, evt, tgt), count in self.transition_counts.items():
            total = self.source_event_counts.get((src, evt), 1)
            prob = count / total

            chart.transitions.append(LearnedTransition(
                source=src,
                target=tgt,
                event=evt,
                count=count,
                probability=prob,
            ))
            chart.events.add(evt)

        return chart

    def get_accuracy(self, ground_truth: List[Tuple[str, str, str]]) -> float:
        """Calculate accuracy against ground truth transitions."""
        gt_set = set(ground_truth)
        learned_set = set(self.transition_counts.keys())

        if not gt_set:
            return 0.0

        return len(gt_set & learned_set) / len(gt_set)


# =============================================================================
# Comparison Framework
# =============================================================================

@dataclass
class LearningCurvePoint:
    """Single point on learning curve."""
    num_samples: int
    accuracy: float
    precision: float
    recall: float
    elapsed_time: float


@dataclass
class ComparisonResult:
    """Result of online vs offline comparison."""
    online_curve: List[LearningCurvePoint] = field(default_factory=list)
    offline_curve: List[LearningCurvePoint] = field(default_factory=list)
    online_final_accuracy: float = 0.0
    offline_final_accuracy: float = 0.0
    online_samples_to_90: int = 0  # Samples to reach 90% accuracy
    offline_samples_to_90: int = 0


class OnlineOfflineComparison:
    """
    Compare online and offline learning approaches.
    """

    def __init__(
        self,
        states: List[str],
        events: List[str],
        transitions: List[Tuple[str, str, str]],
        initial_state: str,
    ):
        self.states = states
        self.events = events
        self.transitions = transitions
        self.initial_state = initial_state
        self.gt_set = set(transitions)

    def _compute_metrics(
        self,
        learned_transitions: Set[Tuple[str, str, str]]
    ) -> Tuple[float, float, float]:
        """Compute accuracy, precision, recall."""
        if not learned_transitions:
            return 0.0, 0.0, 0.0

        true_positives = len(self.gt_set & learned_transitions)
        precision = true_positives / len(learned_transitions) if learned_transitions else 0.0
        recall = true_positives / len(self.gt_set) if self.gt_set else 0.0
        accuracy = recall  # For this task, accuracy = recall (finding all GT transitions)

        return accuracy, precision, recall

    def run_online_learning(
        self,
        max_steps: int = 1000,
        eval_interval: int = 50,
        exploration_rate: float = 0.3,
    ) -> List[LearningCurvePoint]:
        """Run online learning experiment."""
        env = StatechartEnvironment(
            self.states, self.events, self.transitions, self.initial_state
        )
        learner = OnlineLearner(exploration_rate=exploration_rate)

        curve = []
        start_time = time.time()

        current_state = env.reset()
        learner.known_states.add(current_state)

        for step in range(max_steps):
            # Select action
            valid_events = env.get_valid_events() or self.events
            event = learner.select_action(current_state, valid_events)

            # Take step
            old_state, new_state, valid = env.step(event)
            learner.observe(old_state, event, new_state, valid)

            current_state = new_state

            # Occasionally reset to explore from different states
            if random.random() < 0.1:
                current_state = env.reset()

            # Evaluate at intervals
            if (step + 1) % eval_interval == 0:
                learned = set(learner.transition_counts.keys())
                acc, prec, rec = self._compute_metrics(learned)

                curve.append(LearningCurvePoint(
                    num_samples=step + 1,
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    elapsed_time=time.time() - start_time,
                ))

        return curve

    def run_offline_learning(
        self,
        max_samples: int = 1000,
        eval_interval: int = 50,
        traces_per_batch: int = 5,
        steps_per_trace: int = 10,
    ) -> List[LearningCurvePoint]:
        """Run offline learning experiment."""
        generator = SyntheticTraceGenerator(
            self.states, self.events, self.transitions
        )

        curve = []
        start_time = time.time()
        all_traces: List[ExecutionTrace] = []
        total_samples = 0

        while total_samples < max_samples:
            # Generate batch of traces
            new_traces = generator.generate_traces(
                num_traces=traces_per_batch,
                steps_per_trace=steps_per_trace,
            )
            all_traces.extend(new_traces)

            # Count samples
            for trace in new_traces:
                total_samples += len(trace.entries)

            # Learn from all traces so far
            learner = DirectExtractionLearner()
            learner.add_traces(all_traces)
            chart = learner.learn()

            # Evaluate
            learned = set(
                (t.source, t.event, t.target)
                for t in chart.transitions
                if t.probability > 0.1
            )
            acc, prec, rec = self._compute_metrics(learned)

            curve.append(LearningCurvePoint(
                num_samples=total_samples,
                accuracy=acc,
                precision=prec,
                recall=rec,
                elapsed_time=time.time() - start_time,
            ))

        return curve

    def run_comparison(
        self,
        max_samples: int = 500,
        eval_interval: int = 25,
    ) -> ComparisonResult:
        """Run full comparison."""
        result = ComparisonResult()

        # Online learning
        result.online_curve = self.run_online_learning(
            max_steps=max_samples,
            eval_interval=eval_interval,
        )

        # Offline learning
        result.offline_curve = self.run_offline_learning(
            max_samples=max_samples,
            eval_interval=eval_interval,
        )

        # Final accuracies
        if result.online_curve:
            result.online_final_accuracy = result.online_curve[-1].accuracy
        if result.offline_curve:
            result.offline_final_accuracy = result.offline_curve[-1].accuracy

        # Samples to reach 90%
        for point in result.online_curve:
            if point.accuracy >= 0.9:
                result.online_samples_to_90 = point.num_samples
                break

        for point in result.offline_curve:
            if point.accuracy >= 0.9:
                result.offline_samples_to_90 = point.num_samples
                break

        return result


# =============================================================================
# Replay Buffer for Hybrid Learning
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for hybrid online/offline learning.

    Stores transitions and samples for training.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Tuple[str, str, str]] = []  # (source, event, target)
        self.priorities: List[float] = []

    def add(self, source: str, event: str, target: str, priority: float = 1.0):
        """Add transition to buffer."""
        if len(self.buffer) >= self.max_size:
            # Remove lowest priority
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

        self.buffer.append((source, event, target))
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Tuple[str, str, str]]:
        """Sample batch from buffer."""
        if len(self.buffer) == 0:
            return []

        # Priority sampling
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]

        indices = random.choices(range(len(self.buffer)), weights=probs, k=min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in indices]

    def load_from_traces(self, traces: List[ExecutionTrace]):
        """Load transitions from offline traces."""
        for trace in traces:
            for entry in trace.entries:
                if not entry.source_config or not entry.target_config or not entry.trigger_event:
                    continue

                src = entry.source_config.active_states[0] if entry.source_config.active_states else "?"
                tgt = entry.target_config.active_states[0] if entry.target_config.active_states else "?"
                evt = entry.trigger_event.event_type

                self.add(src, evt, tgt)

    def __len__(self) -> int:
        return len(self.buffer)


class HybridLearner:
    """
    Hybrid learner that combines online and offline data.

    Uses replay buffer to mix fresh online experience with offline traces.
    """

    def __init__(self, offline_ratio: float = 0.5):
        self.offline_ratio = offline_ratio
        self.replay_buffer = ReplayBuffer()
        self.online_learner = OnlineLearner()

    def load_offline_traces(self, traces: List[ExecutionTrace]):
        """Load offline traces into replay buffer."""
        self.replay_buffer.load_from_traces(traces)

    def observe_online(self, source: str, event: str, target: str, valid: bool):
        """Observe online transition."""
        self.online_learner.observe(source, event, target, valid)
        if valid:
            self.replay_buffer.add(source, event, target, priority=2.0)  # Higher priority for fresh data

    def learn_batch(self, batch_size: int = 32):
        """Learn from mixed batch."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)

        # Update counts
        for src, evt, tgt in batch:
            key = (src, evt, tgt)
            self.online_learner.transition_counts[key] = (
                self.online_learner.transition_counts.get(key, 0) + 1
            )
            self.online_learner.source_event_counts[(src, evt)] = (
                self.online_learner.source_event_counts.get((src, evt), 0) + 1
            )

    def get_learned_chart(self) -> LearnedStatechart:
        """Get learned statechart."""
        return self.online_learner.get_learned_chart()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate online vs offline comparison."""
    print("=" * 60)
    print("Online vs Offline Learning Comparison")
    print("=" * 60)

    # Define ground truth statechart
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

    print(f"\nGround truth: {len(states)} states, {len(events)} events, {len(transitions)} transitions")

    # Run comparison
    comparison = OnlineOfflineComparison(
        states=states,
        events=events,
        transitions=transitions,
        initial_state="IDLE",
    )

    print("\n--- Running Comparison ---")
    result = comparison.run_comparison(max_samples=300, eval_interval=30)

    # Print learning curves
    print("\n--- Online Learning Curve ---")
    print(f"{'Samples':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    for point in result.online_curve:
        print(f"{point.num_samples:>10} {point.accuracy:>10.2%} {point.precision:>10.2%} {point.recall:>10.2%}")

    print("\n--- Offline Learning Curve ---")
    print(f"{'Samples':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    for point in result.offline_curve:
        print(f"{point.num_samples:>10} {point.accuracy:>10.2%} {point.precision:>10.2%} {point.recall:>10.2%}")

    # Summary
    print("\n--- Summary ---")
    print(f"Online final accuracy:  {result.online_final_accuracy:.2%}")
    print(f"Offline final accuracy: {result.offline_final_accuracy:.2%}")

    if result.online_samples_to_90 > 0:
        print(f"Online samples to 90%:  {result.online_samples_to_90}")
    else:
        print(f"Online samples to 90%:  >300 (not reached)")

    if result.offline_samples_to_90 > 0:
        print(f"Offline samples to 90%: {result.offline_samples_to_90}")
    else:
        print(f"Offline samples to 90%: >300 (not reached)")

    # Compare efficiency
    if result.online_samples_to_90 > 0 and result.offline_samples_to_90 > 0:
        ratio = result.offline_samples_to_90 / result.online_samples_to_90
        print(f"\nOffline/Online ratio: {ratio:.2f}x")
        if ratio < 1:
            print("Offline is MORE sample efficient")
        else:
            print("Online is MORE sample efficient")

    return result


if __name__ == "__main__":
    demo()
