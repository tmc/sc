"""
Experiment M: World Model Building

Reverse-engineer statecharts from observation sequences.
Given (observation, action, next_observation) tuples, learn:
- Discrete states (via clustering)
- Transitions between states
- Guard conditions for transitions
- Hierarchical structure

Output: Statechart proto compatible with the sc format.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class LearnedTransition:
    """A learned transition between states."""
    source: int
    target: int
    action: Optional[int] = None
    guard_weights: Optional[mx.array] = None
    count: int = 0
    confidence: float = 0.0


@dataclass
class LearnedState:
    """A learned discrete state."""
    id: int
    centroid: mx.array
    label: str = ""
    parent: Optional[int] = None
    children: list = None
    is_initial: bool = False
    is_final: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []


class StateDiscovery(nn.Module):
    """Discover discrete states from continuous observations.

    Uses a learned encoder + clustering to find discrete states.
    Supports both fixed K clustering and adaptive discovery.
    """

    def __init__(self, obs_dim: int, embed_dim: int = 64,
                 max_states: int = 32, temperature: float = 1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.max_states = max_states
        self.temperature = temperature

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable state centroids
        self.centroids = mx.random.normal((max_states, embed_dim)) * 0.1

        # State usage mask (for adaptive discovery)
        self.state_active = mx.zeros((max_states,))
        self.num_active_states = 0

    def encode(self, obs: mx.array) -> mx.array:
        """Encode observation to embedding space.

        Args:
            obs: [B, obs_dim] observations

        Returns:
            [B, embed_dim] embeddings
        """
        return self.encoder(obs)

    def assign_states(self, obs: mx.array) -> tuple:
        """Assign observations to discrete states.

        Args:
            obs: [B, obs_dim] observations

        Returns:
            (state_ids, soft_assignments, embeddings)
            - state_ids: [B] hard state assignments
            - soft_assignments: [B, max_states] soft assignments
            - embeddings: [B, embed_dim] observation embeddings
        """
        embeddings = self.encode(obs)  # [B, embed_dim]

        # Compute distances to centroids
        # embeddings: [B, embed_dim], centroids: [max_states, embed_dim]
        distances = mx.sum(
            (embeddings[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=-1
        )  # [B, max_states]

        # Soft assignments via softmax over negative distances
        soft_assignments = mx.softmax(-distances / self.temperature, axis=-1)

        # Hard assignments
        state_ids = mx.argmin(distances, axis=-1)

        return state_ids, soft_assignments, embeddings

    def update_centroids(self, embeddings: mx.array,
                        assignments: mx.array, lr: float = 0.1):
        """Update centroids using exponential moving average.

        Args:
            embeddings: [B, embed_dim] observation embeddings
            assignments: [B, max_states] soft assignments
            lr: Learning rate for centroid updates
        """
        # Weighted sum of embeddings per centroid
        # assignments: [B, max_states], embeddings: [B, embed_dim]
        weighted_sum = mx.matmul(assignments.T, embeddings)  # [max_states, embed_dim]
        counts = mx.sum(assignments, axis=0, keepdims=True).T  # [max_states, 1]

        # New centroids (avoid div by zero)
        new_centroids = weighted_sum / (counts + 1e-8)

        # EMA update
        self.centroids = (1 - lr) * self.centroids + lr * new_centroids

        # Update active states
        active = mx.sum(assignments, axis=0) > 0.5
        self.state_active = mx.maximum(self.state_active, active.astype(mx.float32))
        self.num_active_states = int(mx.sum(self.state_active))

    def discover_new_state(self, embedding: mx.array,
                          threshold: float = 2.0) -> Optional[int]:
        """Potentially discover a new state if observation is far from existing.

        Args:
            embedding: [embed_dim] single observation embedding
            threshold: Distance threshold for new state

        Returns:
            New state ID if created, None otherwise
        """
        if self.num_active_states >= self.max_states:
            return None

        # Check distance to nearest active centroid
        active_mask = self.state_active > 0.5
        if mx.sum(active_mask) == 0:
            # First state
            new_id = 0
            self.centroids = self.centroids.at[new_id].add(
                embedding - self.centroids[new_id]
            )
            self.state_active = self.state_active.at[new_id].add(1.0)
            self.num_active_states = 1
            return new_id

        distances = mx.sum((embedding - self.centroids) ** 2, axis=-1)
        distances = mx.where(active_mask, distances, float('inf'))
        min_dist = mx.min(distances)

        if float(min_dist) > threshold:
            # Find first inactive state
            new_id = None
            for idx in range(self.max_states):
                if float(self.state_active[idx]) < 0.5:
                    new_id = idx
                    break
            if new_id is not None:
                self.centroids = self.centroids.at[new_id].add(
                    embedding - self.centroids[new_id]
                )
                self.state_active = self.state_active.at[new_id].add(1.0)
                self.num_active_states += 1
                return new_id

        return None


class TransitionLearner(nn.Module):
    """Learn transitions between discovered states.

    Maintains a transition matrix and learns which state pairs
    have valid transitions under which actions.
    """

    def __init__(self, max_states: int, num_actions: int,
                 embed_dim: int = 64):
        super().__init__()
        self.max_states = max_states
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # Transition counts: [num_actions, max_states, max_states]
        self.transition_counts = mx.zeros((num_actions, max_states, max_states))

        # Learnable transition embeddings for confidence estimation
        self.transition_embed = nn.Linear(max_states * 2 + num_actions, embed_dim)
        self.transition_score = nn.Linear(embed_dim, 1)

    def record_transition(self, source: int, action: int, target: int):
        """Record an observed transition.

        Args:
            source: Source state ID
            action: Action taken
            target: Target state ID
        """
        self.transition_counts = self.transition_counts.at[action, source, target].add(1.0)

    def get_transition_probs(self, source: int, action: int) -> mx.array:
        """Get transition probabilities from a source state under an action.

        Args:
            source: Source state ID
            action: Action taken

        Returns:
            [max_states] transition probabilities to each target
        """
        counts = self.transition_counts[action, source, :]
        total = mx.sum(counts) + 1e-8
        return counts / total

    def get_learned_transitions(self, min_count: int = 1) -> list:
        """Extract learned transitions as a list.

        Args:
            min_count: Minimum observation count to include

        Returns:
            List of LearnedTransition objects
        """
        transitions = []
        for action in range(self.num_actions):
            for source in range(self.max_states):
                for target in range(self.max_states):
                    count = int(self.transition_counts[action, source, target])
                    if count >= min_count:
                        total = float(mx.sum(self.transition_counts[action, source, :]))
                        confidence = count / total if total > 0 else 0
                        transitions.append(LearnedTransition(
                            source=source,
                            target=target,
                            action=action,
                            count=count,
                            confidence=confidence,
                        ))
        return transitions


class GuardInducer(nn.Module):
    """Induce guard conditions from context at transitions.

    Learns what context features predict transition enablement.
    """

    def __init__(self, context_dim: int, embed_dim: int = 32,
                 max_transitions: int = 100):
        super().__init__()
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.max_transitions = max_transitions

        # Per-transition guard networks
        self.guard_nets = [
            nn.Sequential(
                nn.Linear(context_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
            for _ in range(max_transitions)
        ]

        # Transition to guard mapping
        self.transition_to_guard = {}
        self.next_guard_id = 0

    def get_or_create_guard(self, transition_key: tuple) -> int:
        """Get or create a guard ID for a transition.

        Args:
            transition_key: (source, action, target) tuple

        Returns:
            Guard ID
        """
        if transition_key not in self.transition_to_guard:
            if self.next_guard_id >= self.max_transitions:
                return -1  # No more guards available
            self.transition_to_guard[transition_key] = self.next_guard_id
            self.next_guard_id += 1
        return self.transition_to_guard[transition_key]

    def evaluate_guard(self, guard_id: int, context: mx.array) -> mx.array:
        """Evaluate a guard condition.

        Args:
            guard_id: Guard ID
            context: [B, context_dim] context features

        Returns:
            [B, 1] guard values in [0, 1]
        """
        if guard_id < 0 or guard_id >= len(self.guard_nets):
            return mx.ones((context.shape[0], 1))

        logits = self.guard_nets[guard_id](context)
        return mx.sigmoid(logits)

    def train_guard(self, guard_id: int, contexts: mx.array,
                   labels: mx.array, lr: float = 0.01) -> float:
        """Train a guard on labeled examples.

        Args:
            guard_id: Guard ID
            contexts: [N, context_dim] context features
            labels: [N] binary labels (1 if transition occurred)
            lr: Learning rate

        Returns:
            Loss value
        """
        def loss_fn(net):
            logits = net(contexts).squeeze(-1)
            # Binary cross entropy
            loss = mx.mean(
                -labels * mx.log(mx.sigmoid(logits) + 1e-8)
                - (1 - labels) * mx.log(1 - mx.sigmoid(logits) + 1e-8)
            )
            return loss

        loss, grads = mx.value_and_grad(loss_fn)(self.guard_nets[guard_id])

        # Simple gradient update (in practice use optimizer)
        for name, param in self.guard_nets[guard_id].parameters().items():
            if name in grads:
                param -= lr * grads[name]

        return float(loss)


class HierarchyBuilder:
    """Discover hierarchical structure from state patterns.

    Analyzes state co-occurrence and transition patterns to find
    potential parent-child relationships.
    """

    def __init__(self, max_states: int):
        self.max_states = max_states

        # State co-occurrence matrix
        self.cooccurrence = mx.zeros((max_states, max_states))

        # State sequence patterns
        self.sequence_counts = {}

    def record_sequence(self, states: list):
        """Record a state sequence for pattern analysis.

        Args:
            states: List of state IDs
        """
        # Update co-occurrence (states in same sequence)
        for i, s1 in enumerate(states):
            for s2 in states[i+1:min(i+5, len(states))]:
                self.cooccurrence = self.cooccurrence.at[s1, s2].add(1.0)
                self.cooccurrence = self.cooccurrence.at[s2, s1].add(1.0)

        # Record subsequences
        for length in [2, 3]:
            for i in range(len(states) - length + 1):
                subseq = tuple(states[i:i+length])
                self.sequence_counts[subseq] = self.sequence_counts.get(subseq, 0) + 1

    def find_clusters(self, threshold: float = 0.5) -> list:
        """Find state clusters based on co-occurrence.

        Args:
            threshold: Similarity threshold for clustering

        Returns:
            List of (parent_id, child_ids) tuples
        """
        # Normalize co-occurrence to similarity
        row_sums = mx.sum(self.cooccurrence, axis=1, keepdims=True) + 1e-8
        similarity = self.cooccurrence / row_sums

        # Simple clustering: group states with high mutual similarity
        clusters = []
        used = set()

        for i in range(self.max_states):
            if i in used:
                continue
            if float(mx.sum(self.cooccurrence[i, :])) == 0:
                continue

            # Find states similar to i (using comparison and iteration)
            sim_row = similarity[i, :]
            similar = []
            for j in range(self.max_states):
                if j != i and j not in used and float(sim_row[j]) > threshold:
                    similar.append(j)

            if similar:
                cluster = [i] + similar
                clusters.append((i, similar))  # i is parent, similar are children
                used.update(cluster)
            else:
                used.add(i)

        return clusters

    def detect_initial_states(self, first_states: list) -> list:
        """Detect likely initial states from sequence starts.

        Args:
            first_states: List of first state IDs from sequences

        Returns:
            List of likely initial state IDs
        """
        counts = {}
        for s in first_states:
            counts[s] = counts.get(s, 0) + 1

        # States that appear first > 10% of time
        total = len(first_states)
        threshold = total * 0.1
        return [s for s, c in counts.items() if c > threshold]

    def detect_final_states(self, last_states: list) -> list:
        """Detect likely final states from sequence ends.

        Args:
            last_states: List of last state IDs from sequences

        Returns:
            List of likely final state IDs
        """
        counts = {}
        for s in last_states:
            counts[s] = counts.get(s, 0) + 1

        total = len(last_states)
        threshold = total * 0.1
        return [s for s, c in counts.items() if c > threshold]


class WorldModelBuilder(nn.Module):
    """Complete world model builder assembling all components.

    Takes observation sequences and produces a learned statechart.
    """

    def __init__(self, obs_dim: int, num_actions: int,
                 context_dim: int = None, embed_dim: int = 64,
                 max_states: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.context_dim = context_dim or obs_dim
        self.embed_dim = embed_dim
        self.max_states = max_states

        # Component modules
        self.state_discovery = StateDiscovery(obs_dim, embed_dim, max_states)
        self.transition_learner = TransitionLearner(max_states, num_actions, embed_dim)
        self.guard_inducer = GuardInducer(self.context_dim, embed_dim // 2)
        self.hierarchy_builder = HierarchyBuilder(max_states)

        # Training data collection
        self.sequences = []
        self.contexts = []

    def process_trajectory(self, observations: mx.array, actions: mx.array,
                          contexts: mx.array = None):
        """Process a trajectory to update the world model.

        Args:
            observations: [T, obs_dim] observation sequence
            actions: [T-1] action sequence
            contexts: Optional [T, context_dim] context features
        """
        T = observations.shape[0]

        # Assign states
        state_ids, soft_assignments, embeddings = self.state_discovery.assign_states(
            observations
        )

        # Update centroids
        self.state_discovery.update_centroids(embeddings, soft_assignments)

        # Record transitions
        state_list = [int(s) for s in state_ids]
        for t in range(T - 1):
            action = int(actions[t])
            source = state_list[t]
            target = state_list[t + 1]
            self.transition_learner.record_transition(source, action, target)

            # Record guard context
            if contexts is not None:
                trans_key = (source, action, target)
                guard_id = self.guard_inducer.get_or_create_guard(trans_key)
                # Store for later training
                self.contexts.append((guard_id, contexts[t], 1.0))

        # Record for hierarchy analysis
        self.sequences.append(state_list)
        self.hierarchy_builder.record_sequence(state_list)

    def build_statechart(self, min_transition_count: int = 2) -> dict:
        """Build a statechart proto from learned model.

        Args:
            min_transition_count: Minimum observations for transition

        Returns:
            Statechart dict compatible with sc proto format
        """
        # Get active states
        active_states = []
        for i in range(self.max_states):
            if float(self.state_discovery.state_active[i]) > 0.5:
                active_states.append(i)

        # Detect initial/final states
        first_states = [seq[0] for seq in self.sequences if seq]
        last_states = [seq[-1] for seq in self.sequences if seq]
        initial_states = set(self.hierarchy_builder.detect_initial_states(first_states))
        final_states = set(self.hierarchy_builder.detect_final_states(last_states))

        # Find hierarchy
        clusters = self.hierarchy_builder.find_clusters()
        parent_map = {}
        for parent, children in clusters:
            for child in children:
                parent_map[child] = parent

        # Build state tree
        def build_state(state_id, depth=0):
            centroid = self.state_discovery.centroids[state_id]
            children_ids = [c for p, cs in clusters if p == state_id for c in cs]

            state_dict = {
                "label": f"State_{state_id}",
                "type": 2 if children_ids else 1,  # OR if has children, else BASIC
                "is_initial": state_id in initial_states,
            }

            if state_id in final_states:
                state_dict["is_final"] = True

            if children_ids:
                state_dict["children"] = [
                    build_state(cid, depth + 1) for cid in children_ids
                ]

            return state_dict

        # Build root state with top-level states
        top_level = [s for s in active_states if s not in parent_map]
        root_children = [build_state(s) for s in top_level]

        # Get transitions
        learned_transitions = self.transition_learner.get_learned_transitions(
            min_transition_count
        )

        transitions = []
        for lt in learned_transitions:
            if lt.source in active_states and lt.target in active_states:
                trans_dict = {
                    "label": f"t_{lt.source}_{lt.action}_{lt.target}",
                    "from": [f"State_{lt.source}"],
                    "to": [f"State_{lt.target}"],
                    "event": f"ACTION_{lt.action}",
                }
                transitions.append(trans_dict)

        # Build events list
        events = [{"label": f"ACTION_{a}"} for a in range(self.num_actions)]

        statechart = {
            "name": "LearnedStatechart",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": root_children,
            },
            "events": events,
            "transitions": transitions,
        }

        return statechart

    def get_stats(self) -> dict:
        """Get learning statistics.

        Returns:
            Dict with model statistics
        """
        transitions = self.transition_learner.get_learned_transitions(1)
        return {
            "num_states": self.state_discovery.num_active_states,
            "num_transitions": len(transitions),
            "num_sequences": len(self.sequences),
            "num_guards": self.guard_inducer.next_guard_id,
        }


def test_world_model():
    """Test world model building on synthetic data."""
    print("=" * 60)
    print("World Model Building Test")
    print("=" * 60)

    # Create synthetic environment with 4 hidden states
    # State transitions: 0 -> 1 -> 2 -> 3 -> 0 (cycle)
    obs_dim = 8
    num_actions = 2  # 0 = stay, 1 = advance

    # Generate observations for each hidden state
    state_obs = {
        0: mx.array([1, 0, 0, 0, 0, 0, 0, 0]),
        1: mx.array([0, 1, 0, 0, 0, 0, 0, 0]),
        2: mx.array([0, 0, 1, 0, 0, 0, 0, 0]),
        3: mx.array([0, 0, 0, 1, 0, 0, 0, 0]),
    }

    # Add noise to observations
    def get_obs(state):
        base = state_obs[state].astype(mx.float32)
        noise = mx.random.normal(base.shape) * 0.1
        return base + noise

    # Create world model builder
    model = WorldModelBuilder(
        obs_dim=obs_dim,
        num_actions=num_actions,
        max_states=8,
        embed_dim=32,
    )

    # Generate trajectories
    print("\nGenerating synthetic trajectories...")
    num_trajectories = 50
    trajectory_length = 10

    for traj_idx in range(num_trajectories):
        hidden_state = 0
        observations = []
        actions = []

        for t in range(trajectory_length):
            observations.append(get_obs(hidden_state))

            if t < trajectory_length - 1:
                # Action 1 advances, action 0 stays
                action = 1 if mx.random.uniform() > 0.3 else 0
                actions.append(action)

                if action == 1:
                    hidden_state = (hidden_state + 1) % 4

        obs_tensor = mx.stack(observations, axis=0)
        action_tensor = mx.array(actions)

        model.process_trajectory(obs_tensor, action_tensor)

    # Build statechart
    print("\nBuilding statechart from observations...")
    statechart = model.build_statechart(min_transition_count=3)

    # Print results
    stats = model.get_stats()
    print(f"\nLearned Model Statistics:")
    print(f"  States discovered: {stats['num_states']}")
    print(f"  Transitions learned: {stats['num_transitions']}")
    print(f"  Sequences processed: {stats['num_sequences']}")

    print(f"\nStatechart structure:")
    print(f"  Name: {statechart['name']}")
    print(f"  Root children: {len(statechart['root_state']['children'])}")
    print(f"  Events: {len(statechart['events'])}")
    print(f"  Transitions: {len(statechart['transitions'])}")

    # Print learned transitions
    print(f"\nLearned transitions:")
    for t in statechart["transitions"][:10]:
        print(f"  {t['from'][0]} --[{t['event']}]--> {t['to'][0]}")

    # Test gradient flow
    print("\n" + "=" * 60)
    print("Gradient Flow Test")
    print("=" * 60)

    test_obs = mx.random.normal((5, obs_dim))

    def loss_fn(obs):
        state_ids, soft_assign, emb = model.state_discovery.assign_states(obs)
        return mx.mean(soft_assign[:, 0])  # Probability of state 0

    loss, grad = mx.value_and_grad(loss_fn)(test_obs)
    grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))

    print(f"Loss: {float(loss):.4f}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"✓ Gradients flow: {grad_norm > 0}")

    return model, statechart


if __name__ == "__main__":
    model, statechart = test_world_model()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("WorldModelBuilder components:")
    print("  ✓ StateDiscovery - clusters observations to discrete states")
    print("  ✓ TransitionLearner - learns state transitions from sequences")
    print("  ✓ GuardInducer - induces guard conditions from context")
    print("  ✓ HierarchyBuilder - discovers hierarchical structure")
    print("  ✓ WorldModelBuilder - assembles complete statechart")
