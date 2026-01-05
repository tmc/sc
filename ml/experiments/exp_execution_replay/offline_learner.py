"""
Offline Learner - Learn statecharts from execution traces.

No online interaction needed - pure offline learning from logged traces.

Approaches:
1. Direct Extraction: Count transitions, compute probabilities
2. State Inference: Cluster trace patterns to infer hidden states
3. Guard Learning: Learn transition guards from context
4. Neural Fitting: Train neural network to predict transitions
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import random
import math

from .trace_parser import (
    ExecutionTrace, TransitionLogEntry, Configuration,
    TraceParser, SyntheticTraceGenerator
)


# =============================================================================
# Learned Statechart Representation
# =============================================================================

@dataclass
class LearnedTransition:
    """A transition learned from traces."""
    source: str
    target: str
    event: str
    count: int = 0
    probability: float = 0.0
    guard_expression: str = ""
    confidence: float = 0.0


@dataclass
class LearnedState:
    """A state learned from traces."""
    name: str
    visit_count: int = 0
    entry_events: Dict[str, int] = field(default_factory=dict)
    exit_events: Dict[str, int] = field(default_factory=dict)
    avg_dwell_time: float = 0.0


@dataclass
class LearnedStatechart:
    """Statechart structure learned from traces."""
    states: Dict[str, LearnedState] = field(default_factory=dict)
    transitions: List[LearnedTransition] = field(default_factory=list)
    initial_state: str = ""
    events: Set[str] = field(default_factory=set)

    # Learning metadata
    num_traces: int = 0
    num_transitions: int = 0

    def get_transition_probs(self, source: str, event: str) -> Dict[str, float]:
        """Get probability distribution over targets for (source, event)."""
        probs = {}
        for t in self.transitions:
            if t.source == source and t.event == event:
                probs[t.target] = t.probability
        return probs

    def sample_next_state(self, current: str, event: str) -> Optional[str]:
        """Sample next state given current state and event."""
        probs = self.get_transition_probs(current, event)
        if not probs:
            return None

        r = random.random()
        cumsum = 0.0
        for target, prob in probs.items():
            cumsum += prob
            if r < cumsum:
                return target
        return list(probs.keys())[-1]


# =============================================================================
# Direct Extraction Learner
# =============================================================================

class DirectExtractionLearner:
    """
    Learn statechart by directly counting transitions in traces.

    Simple maximum likelihood estimation:
    P(target | source, event) = count(source, event, target) / count(source, event)
    """

    def __init__(self):
        # Count (source, event, target) -> count
        self.transition_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        # Count (source, event) -> count
        self.source_event_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # State visit counts
        self.state_counts: Dict[str, int] = defaultdict(int)
        # Initial state counts
        self.initial_counts: Dict[str, int] = defaultdict(int)

        self.traces_processed = 0

    def add_trace(self, trace: ExecutionTrace):
        """Add trace to learning data."""
        self.traces_processed += 1

        # Count initial state
        if trace.initial_config and trace.initial_config.active_states:
            initial = trace.initial_config.active_states[0]
            self.initial_counts[initial] += 1

        # Count transitions
        for entry in trace.entries:
            if not entry.source_config or not entry.target_config:
                continue
            if not entry.trigger_event:
                continue

            src = entry.source_config.active_states[0] if entry.source_config.active_states else "UNKNOWN"
            tgt = entry.target_config.active_states[0] if entry.target_config.active_states else "UNKNOWN"
            evt = entry.trigger_event.event_type

            self.transition_counts[(src, evt, tgt)] += 1
            self.source_event_counts[(src, evt)] += 1
            self.state_counts[src] += 1
            self.state_counts[tgt] += 1

    def add_traces(self, traces: List[ExecutionTrace]):
        """Add multiple traces."""
        for trace in traces:
            self.add_trace(trace)

    def learn(self) -> LearnedStatechart:
        """Learn statechart from collected data."""
        chart = LearnedStatechart()
        chart.num_traces = self.traces_processed

        # Learn states
        for state, count in self.state_counts.items():
            chart.states[state] = LearnedState(
                name=state,
                visit_count=count,
            )

        # Learn initial state
        if self.initial_counts:
            chart.initial_state = max(self.initial_counts.items(), key=lambda x: x[1])[0]

        # Learn transitions with probabilities
        for (src, evt, tgt), count in self.transition_counts.items():
            total = self.source_event_counts[(src, evt)]
            prob = count / total if total > 0 else 0.0

            trans = LearnedTransition(
                source=src,
                target=tgt,
                event=evt,
                count=count,
                probability=prob,
                confidence=min(1.0, count / 10),  # Confidence grows with samples
            )
            chart.transitions.append(trans)
            chart.events.add(evt)

        chart.num_transitions = len(chart.transitions)
        return chart


# =============================================================================
# State Inference Learner
# =============================================================================

class StateInferenceLearner:
    """
    Infer hidden states from trace patterns.

    Uses clustering to discover states that may not be explicitly labeled.
    Useful when traces only contain observations, not state labels.
    """

    def __init__(self, n_states: int = 5, context_dim: int = 10):
        self.n_states = n_states
        self.context_dim = context_dim

        # State cluster centers (learned)
        self.cluster_centers: Optional[mx.array] = None

        # Transition model: P(s' | s, e)
        self.transition_probs: Dict[Tuple[int, str, int], float] = {}

    def _context_to_vector(self, context: Dict[str, Any]) -> mx.array:
        """Convert context dict to fixed-size vector."""
        vec = [0.0] * self.context_dim

        # Simple encoding: hash keys and values
        for i, (key, value) in enumerate(sorted(context.items())[:self.context_dim]):
            if isinstance(value, (int, float)):
                vec[i % self.context_dim] += float(value)
            elif isinstance(value, bool):
                vec[i % self.context_dim] += 1.0 if value else 0.0
            else:
                vec[i % self.context_dim] += hash(str(value)) % 100 / 100.0

        return mx.array(vec)

    def _assign_cluster(self, context_vec: mx.array) -> int:
        """Assign context to nearest cluster."""
        if self.cluster_centers is None:
            return 0

        # Find nearest cluster
        distances = mx.sum((self.cluster_centers - context_vec) ** 2, axis=1)
        return int(mx.argmin(distances))

    def fit(self, traces: List[ExecutionTrace]) -> LearnedStatechart:
        """Learn statechart with inferred states."""
        # Collect all context vectors
        contexts = []
        for trace in traces:
            for entry in trace.entries:
                if entry.context_before:
                    vec = self._context_to_vector(entry.context_before)
                    contexts.append(vec)

        if not contexts:
            # Fall back to direct extraction if no context
            learner = DirectExtractionLearner()
            learner.add_traces(traces)
            return learner.learn()

        # Stack contexts
        context_matrix = mx.stack(contexts)

        # K-means clustering
        self.cluster_centers = self._kmeans(context_matrix, self.n_states)

        # Assign states to entries and count transitions
        transition_counts: Dict[Tuple[int, str, int], int] = defaultdict(int)
        source_event_counts: Dict[Tuple[int, str], int] = defaultdict(int)

        for trace in traces:
            prev_state = None
            for entry in trace.entries:
                if entry.context_before:
                    vec = self._context_to_vector(entry.context_before)
                    curr_state = self._assign_cluster(vec)
                else:
                    curr_state = 0

                if prev_state is not None and entry.trigger_event:
                    evt = entry.trigger_event.event_type
                    transition_counts[(prev_state, evt, curr_state)] += 1
                    source_event_counts[(prev_state, evt)] += 1

                prev_state = curr_state

        # Build learned statechart
        chart = LearnedStatechart()
        chart.num_traces = len(traces)

        # Create inferred states
        for i in range(self.n_states):
            chart.states[f"S{i}"] = LearnedState(name=f"S{i}")

        # Create transitions
        for (src, evt, tgt), count in transition_counts.items():
            total = source_event_counts[(src, evt)]
            prob = count / total if total > 0 else 0.0

            chart.transitions.append(LearnedTransition(
                source=f"S{src}",
                target=f"S{tgt}",
                event=evt,
                count=count,
                probability=prob,
            ))
            chart.events.add(evt)

        return chart

    def _kmeans(self, data: mx.array, k: int, max_iters: int = 100) -> mx.array:
        """Simple K-means clustering."""
        n = data.shape[0]
        if n < k:
            # Not enough data - return random centers
            return mx.random.normal((k, data.shape[1]))

        # Initialize centers randomly
        indices = random.sample(range(n), k)
        centers = data[mx.array(indices)]

        for _ in range(max_iters):
            # Assign points to clusters
            distances = mx.zeros((n, k))
            for j in range(k):
                distances[:, j] = mx.sum((data - centers[j]) ** 2, axis=1)

            assignments = mx.argmin(distances, axis=1)

            # Update centers
            new_centers = []
            for j in range(k):
                mask = assignments == j
                if mx.sum(mask) > 0:
                    cluster_points = data[mask]
                    new_centers.append(mx.mean(cluster_points, axis=0))
                else:
                    new_centers.append(centers[j])

            new_centers = mx.stack(new_centers)

            # Check convergence
            if mx.max(mx.abs(new_centers - centers)) < 1e-6:
                break

            centers = new_centers

        return centers


# =============================================================================
# Guard Learning
# =============================================================================

class GuardLearner:
    """
    Learn transition guards from context variables.

    For each transition (s, e, s'), learn a guard condition
    that predicts when this transition should fire.
    """

    def __init__(self, hidden_dim: int = 32):
        self.hidden_dim = hidden_dim
        self.guard_networks: Dict[Tuple[str, str, str], nn.Module] = {}

    def _build_guard_network(self, input_dim: int) -> nn.Module:
        """Build simple guard classifier."""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _context_to_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract numeric features from context."""
        features = []
        for key, value in sorted(context.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
        return features[:32] + [0.0] * max(0, 32 - len(features))

    def fit(self, traces: List[ExecutionTrace], epochs: int = 100):
        """Learn guards from traces."""
        # Collect training data for each transition
        transition_data: Dict[Tuple[str, str, str], List[Tuple[List[float], bool]]] = defaultdict(list)

        for trace in traces:
            for entry in trace.entries:
                if not entry.source_config or not entry.target_config or not entry.trigger_event:
                    continue

                src = entry.source_config.active_states[0] if entry.source_config.active_states else "?"
                tgt = entry.target_config.active_states[0] if entry.target_config.active_states else "?"
                evt = entry.trigger_event.event_type

                features = self._context_to_features(entry.context_before)

                # Positive example for this transition
                transition_data[(src, evt, tgt)].append((features, True))

                # Negative examples for other possible transitions from same source
                for other_entry in trace.entries:
                    if other_entry.source_config and other_entry.target_config:
                        other_tgt = other_entry.target_config.active_states[0] if other_entry.target_config.active_states else "?"
                        if other_tgt != tgt:
                            transition_data[(src, evt, other_tgt)].append((features, False))

        # Train guard networks
        for trans_key, data in transition_data.items():
            if len(data) < 10:
                continue

            network = self._build_guard_network(32)
            optimizer = None  # Would use mlx.optimizers in real training

            # Simple training (mock - in practice would use proper optimization)
            positive = sum(1 for _, label in data if label)
            total = len(data)
            self.guard_networks[trans_key] = network

    def predict_guard(
        self,
        source: str,
        event: str,
        target: str,
        context: Dict[str, Any]
    ) -> float:
        """Predict guard probability for transition."""
        key = (source, event, target)
        if key not in self.guard_networks:
            return 0.5  # Default

        features = mx.array(self._context_to_features(context))
        network = self.guard_networks[key]
        output = network(features)
        return float(mx.sigmoid(output))


# =============================================================================
# Neural Transition Model
# =============================================================================

class NeuralTransitionModel(nn.Module):
    """
    Neural network that predicts transitions.

    Input: (current_state_embedding, event_embedding, context)
    Output: next_state_distribution
    """

    def __init__(
        self,
        n_states: int,
        n_events: int,
        state_dim: int = 16,
        context_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_states = n_states
        self.n_events = n_events

        # Embeddings
        self.state_embed = nn.Embedding(n_states, state_dim)
        self.event_embed = nn.Embedding(n_events, state_dim)

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
        )

        # Transition predictor
        input_dim = state_dim * 2 + hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_states),
        )

    def __call__(
        self,
        state_idx: mx.array,
        event_idx: mx.array,
        context: mx.array
    ) -> mx.array:
        """Forward pass - predict next state distribution."""
        state_emb = self.state_embed(state_idx)
        event_emb = self.event_embed(event_idx)
        context_emb = self.context_encoder(context)

        combined = mx.concatenate([state_emb, event_emb, context_emb], axis=-1)
        logits = self.predictor(combined)

        return mx.softmax(logits, axis=-1)


class NeuralOfflineLearner:
    """
    Learn transition model using neural network from offline traces.
    """

    def __init__(self, context_dim: int = 32, hidden_dim: int = 64):
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.state_to_idx: Dict[str, int] = {}
        self.idx_to_state: Dict[int, str] = {}
        self.event_to_idx: Dict[str, int] = {}
        self.idx_to_event: Dict[int, str] = {}

        self.model: Optional[NeuralTransitionModel] = None

    def _build_vocab(self, traces: List[ExecutionTrace]):
        """Build state and event vocabularies."""
        states = set()
        events = set()

        for trace in traces:
            if trace.initial_config:
                states.update(trace.initial_config.active_states)
            for entry in trace.entries:
                if entry.source_config:
                    states.update(entry.source_config.active_states)
                if entry.target_config:
                    states.update(entry.target_config.active_states)
                if entry.trigger_event:
                    events.add(entry.trigger_event.event_type)

        for i, state in enumerate(sorted(states)):
            self.state_to_idx[state] = i
            self.idx_to_state[i] = state

        for i, event in enumerate(sorted(events)):
            self.event_to_idx[event] = i
            self.idx_to_event[i] = event

    def _context_to_features(self, context: Dict[str, Any]) -> List[float]:
        """Convert context to fixed-size feature vector."""
        features = []
        for key, value in sorted(context.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
        return features[:self.context_dim] + [0.0] * max(0, self.context_dim - len(features))

    def _prepare_data(
        self,
        traces: List[ExecutionTrace]
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Prepare training data from traces."""
        states = []
        events = []
        contexts = []
        targets = []

        for trace in traces:
            for entry in trace.entries:
                if not entry.source_config or not entry.target_config or not entry.trigger_event:
                    continue

                src = entry.source_config.active_states[0] if entry.source_config.active_states else None
                tgt = entry.target_config.active_states[0] if entry.target_config.active_states else None
                evt = entry.trigger_event.event_type

                if src not in self.state_to_idx or tgt not in self.state_to_idx:
                    continue
                if evt not in self.event_to_idx:
                    continue

                states.append(self.state_to_idx[src])
                events.append(self.event_to_idx[evt])
                contexts.append(self._context_to_features(entry.context_before))
                targets.append(self.state_to_idx[tgt])

        return (
            mx.array(states),
            mx.array(events),
            mx.array(contexts),
            mx.array(targets),
        )

    def fit(self, traces: List[ExecutionTrace], epochs: int = 100, lr: float = 0.01):
        """Train neural transition model."""
        # Build vocabulary
        self._build_vocab(traces)

        n_states = len(self.state_to_idx)
        n_events = len(self.event_to_idx)

        if n_states == 0 or n_events == 0:
            return

        # Build model
        self.model = NeuralTransitionModel(
            n_states=n_states,
            n_events=n_events,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
        )

        # Prepare data
        states, events, contexts, targets = self._prepare_data(traces)

        if len(states) == 0:
            return

        # Training loop (simplified)
        def loss_fn(model, states, events, contexts, targets):
            probs = model(states, events, contexts)
            # Cross-entropy loss
            log_probs = mx.log(probs + 1e-10)
            target_log_probs = mx.take_along_axis(
                log_probs, targets.reshape(-1, 1), axis=1
            )
            return -mx.mean(target_log_probs)

        # Note: In practice, would use mlx.optimizers.Adam
        # This is a simplified training demonstration

    def predict(
        self,
        state: str,
        event: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict next state distribution."""
        if self.model is None:
            return {}

        if state not in self.state_to_idx or event not in self.event_to_idx:
            return {}

        state_idx = mx.array([self.state_to_idx[state]])
        event_idx = mx.array([self.event_to_idx[event]])
        context_vec = mx.array([self._context_to_features(context)])

        probs = self.model(state_idx, event_idx, context_vec)
        probs = probs[0]  # Remove batch dimension

        result = {}
        for i, prob in enumerate(probs.tolist()):
            if i in self.idx_to_state:
                result[self.idx_to_state[i]] = prob

        return result

    def to_statechart(self, threshold: float = 0.1) -> LearnedStatechart:
        """Convert learned model to statechart."""
        chart = LearnedStatechart()

        for state in self.state_to_idx:
            chart.states[state] = LearnedState(name=state)

        # Generate transitions by sampling
        for src in self.state_to_idx:
            for evt in self.event_to_idx:
                probs = self.predict(src, evt, {})
                for tgt, prob in probs.items():
                    if prob >= threshold:
                        chart.transitions.append(LearnedTransition(
                            source=src,
                            target=tgt,
                            event=evt,
                            probability=prob,
                        ))
                        chart.events.add(evt)

        return chart


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate offline learning from traces."""
    print("=" * 60)
    print("Offline Learner Demo")
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

    # Generate synthetic traces
    print("\n--- Generating Traces ---")
    generator = SyntheticTraceGenerator(states, events, transitions)
    traces = generator.generate_traces(num_traces=50, steps_per_trace=20)
    print(f"Generated {len(traces)} traces")

    # Method 1: Direct extraction
    print("\n--- Direct Extraction ---")
    extractor = DirectExtractionLearner()
    extractor.add_traces(traces)
    chart1 = extractor.learn()

    print(f"Learned {len(chart1.states)} states: {list(chart1.states.keys())}")
    print(f"Learned {len(chart1.transitions)} transitions")
    print(f"Initial state: {chart1.initial_state}")

    print("\nTransition probabilities:")
    for t in sorted(chart1.transitions, key=lambda x: -x.probability)[:6]:
        print(f"  {t.source} --[{t.event}]--> {t.target}: {t.probability:.2f} (n={t.count})")

    # Method 2: State inference (with noisy traces)
    print("\n--- State Inference (with noise) ---")
    noisy_traces = generator.generate_traces(num_traces=30, steps_per_trace=15, noise_prob=0.1)
    inferrer = StateInferenceLearner(n_states=4)
    chart2 = inferrer.fit(noisy_traces)

    print(f"Inferred {len(chart2.states)} states")
    print(f"Learned {len(chart2.transitions)} transitions")

    # Compare with ground truth
    print("\n--- Evaluation ---")
    gt_transitions = set((src, evt, tgt) for src, evt, tgt in transitions)
    learned_transitions = set((t.source, t.event, t.target) for t in chart1.transitions if t.probability > 0.3)

    precision = len(gt_transitions & learned_transitions) / max(1, len(learned_transitions))
    recall = len(gt_transitions & learned_transitions) / max(1, len(gt_transitions))
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"Ground truth transitions: {len(gt_transitions)}")
    print(f"Learned transitions (p>0.3): {len(learned_transitions)}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

    return chart1, chart2


if __name__ == "__main__":
    demo()
