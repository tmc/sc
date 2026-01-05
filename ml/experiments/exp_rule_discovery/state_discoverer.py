"""
State Discoverer - Discovers abstract states from raw observations.

Uses clustering and feature analysis to find meaningful game phases/states.
This is inspired by SAE but simplified for interpretability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import random
import math

try:
    from .trace_collector import Transition, Outcome
except ImportError:
    from trace_collector import Transition, Outcome


@dataclass
class AbstractState:
    """A discovered abstract game state."""
    id: int
    name: str
    centroid: Dict[str, float]  # Feature centroid
    members: List[Transition] = field(default_factory=list)
    description: str = ""

    # Discovered properties
    win_rate: float = 0.0
    loss_rate: float = 0.0
    is_terminal: bool = False
    common_actions: List[Any] = field(default_factory=list)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, AbstractState) and self.id == other.id


class StateDiscoverer:
    """
    Discovers abstract states from game traces.

    Approach:
    1. Extract features from each state
    2. Cluster similar states together
    3. Name clusters based on distinguishing features
    4. Compute statistics for each cluster
    """

    def __init__(self, num_states: int = 5, feature_keys: List[str] = None):
        """
        Args:
            num_states: Target number of abstract states to discover
            feature_keys: List of feature keys to use for clustering
        """
        self.num_states = num_states
        self.feature_keys = feature_keys or []
        self.states: List[AbstractState] = []
        self.state_assignments: Dict[int, AbstractState] = {}  # transition id -> state

    def _extract_default_features(self, trans: Transition) -> Dict[str, float]:
        """Extract default features if none provided."""
        features = trans.features.copy() if trans.features else {}

        # Add basic features from state
        state = trans.state
        if 'board' in state:
            board = state['board']
            # Piece counts
            features['empty_count'] = sum(1 for c in board if c == 0)
            features['p1_count'] = sum(1 for c in board if c == 1)
            features['p2_count'] = sum(1 for c in board if c == 2)
            features['total_pieces'] = features['p1_count'] + features['p2_count']
            features['fill_ratio'] = features['total_pieces'] / len(board)

            # Center control (for 3x3)
            if len(board) == 9:
                features['center_control'] = 1.0 if board[4] != 0 else 0.0
                corners = [board[0], board[2], board[6], board[8]]
                features['corner_count'] = sum(1 for c in corners if c != 0)

        if 'current_player' in state:
            features['is_p1_turn'] = 1.0 if state['current_player'] == 1 else 0.0

        return features

    def _feature_distance(self, f1: Dict[str, float], f2: Dict[str, float]) -> float:
        """Compute distance between two feature vectors."""
        keys = set(f1.keys()) | set(f2.keys())
        if self.feature_keys:
            keys = keys & set(self.feature_keys)

        if not keys:
            return 0.0

        dist = 0.0
        for key in keys:
            v1 = f1.get(key, 0.0)
            v2 = f2.get(key, 0.0)
            dist += (v1 - v2) ** 2

        return math.sqrt(dist)

    def _compute_centroid(self, transitions: List[Transition]) -> Dict[str, float]:
        """Compute feature centroid of a cluster."""
        if not transitions:
            return {}

        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)

        for trans in transitions:
            features = self._extract_default_features(trans)
            for key, value in features.items():
                feature_sums[key] += value
                feature_counts[key] += 1

        return {key: feature_sums[key] / feature_counts[key]
                for key in feature_sums}

    def discover_states(self, transitions: List[Transition]) -> List[AbstractState]:
        """
        Discover abstract states from transitions using k-means-like clustering.

        Returns list of discovered AbstractStates.
        """
        if not transitions:
            return []

        # Extract features for all transitions
        all_features = [self._extract_default_features(t) for t in transitions]

        # Initialize centroids randomly
        indices = random.sample(range(len(transitions)), min(self.num_states, len(transitions)))
        centroids = [all_features[i].copy() for i in indices]

        # K-means iterations
        for _ in range(10):
            # Assign transitions to nearest centroid
            clusters: List[List[int]] = [[] for _ in range(len(centroids))]
            for i, features in enumerate(all_features):
                distances = [self._feature_distance(features, c) for c in centroids]
                nearest = distances.index(min(distances))
                clusters[nearest].append(i)

            # Update centroids
            new_centroids = []
            for cluster_indices in clusters:
                if cluster_indices:
                    cluster_trans = [transitions[i] for i in cluster_indices]
                    new_centroids.append(self._compute_centroid(cluster_trans))
                else:
                    # Empty cluster - reinitialize randomly
                    new_centroids.append(random.choice(all_features).copy())
            centroids = new_centroids

        # Create AbstractState objects
        self.states = []
        for state_id, (centroid, cluster_indices) in enumerate(zip(centroids, clusters)):
            cluster_trans = [transitions[i] for i in cluster_indices]

            # Compute statistics
            win_count = sum(1 for t in cluster_trans if t.outcome == Outcome.WIN)
            loss_count = sum(1 for t in cluster_trans if t.outcome == Outcome.LOSS)
            terminal_count = sum(1 for t in cluster_trans
                                 if t.outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW))

            # Find common actions
            action_counts = defaultdict(int)
            for t in cluster_trans:
                action_counts[str(t.action)] += 1
            common_actions = sorted(action_counts.keys(),
                                    key=lambda a: action_counts[a], reverse=True)[:3]

            # Generate name from distinguishing features
            name = self._generate_state_name(centroid, state_id)

            state = AbstractState(
                id=state_id,
                name=name,
                centroid=centroid,
                members=cluster_trans,
                win_rate=win_count / max(1, len(cluster_trans)),
                loss_rate=loss_count / max(1, len(cluster_trans)),
                is_terminal=terminal_count > len(cluster_trans) * 0.5,
                common_actions=common_actions,
            )
            self.states.append(state)

            # Record assignments
            for i in cluster_indices:
                self.state_assignments[id(transitions[i])] = state

        return self.states

    def _generate_state_name(self, centroid: Dict[str, float], state_id: int) -> str:
        """Generate a human-readable name for a state based on its features."""
        name_parts = []

        # Analyze fill ratio
        fill = centroid.get('fill_ratio', 0)
        if fill < 0.2:
            name_parts.append("Early")
        elif fill < 0.5:
            name_parts.append("Mid")
        elif fill < 0.8:
            name_parts.append("Late")
        else:
            name_parts.append("End")

        # Analyze control
        if centroid.get('center_control', 0) > 0.5:
            name_parts.append("CenterCtrl")

        # Analyze position
        corners = centroid.get('corner_count', 0)
        if corners >= 2:
            name_parts.append("CornerPlay")

        # Fallback
        if len(name_parts) == 1:
            name_parts.append(f"Phase{state_id}")

        return "_".join(name_parts)

    def get_state_for_transition(self, trans: Transition) -> Optional[AbstractState]:
        """Get the abstract state assigned to a transition."""
        return self.state_assignments.get(id(trans))

    def classify_new_state(self, features: Dict[str, float]) -> AbstractState:
        """Classify a new state into one of the discovered abstract states."""
        if not self.states:
            raise ValueError("No states discovered yet")

        distances = [(s, self._feature_distance(features, s.centroid))
                     for s in self.states]
        return min(distances, key=lambda x: x[1])[0]

    def get_transition_matrix(self) -> Dict[Tuple[str, str], int]:
        """
        Build transition matrix between abstract states.

        Returns dict mapping (from_state, to_state) -> count
        """
        # Need to look at consecutive transitions in traces
        # This requires trace structure, simplified version:
        matrix = defaultdict(int)

        # Group transitions by trace (assumes features preserve ordering)
        for state in self.states:
            for trans in state.members:
                from_state = state.name
                # Use next_state features to classify destination
                next_features = self._extract_default_features(
                    Transition(
                        state=trans.next_state,
                        action=None,
                        next_state={},
                        outcome=Outcome.VALID,
                    )
                )
                to_state_obj = self.classify_new_state(next_features)
                to_state = to_state_obj.name
                matrix[(from_state, to_state)] += 1

        return dict(matrix)

    def describe_states(self) -> str:
        """Generate human-readable description of discovered states."""
        lines = ["=== Discovered Abstract States ===\n"]

        for state in self.states:
            lines.append(f"State: {state.name} (ID: {state.id})")
            lines.append(f"  Members: {len(state.members)} transitions")
            lines.append(f"  Win Rate: {state.win_rate*100:.1f}%")
            lines.append(f"  Loss Rate: {state.loss_rate*100:.1f}%")
            lines.append(f"  Terminal: {state.is_terminal}")
            lines.append(f"  Common Actions: {state.common_actions[:3]}")
            lines.append(f"  Key Features:")
            for key, value in sorted(state.centroid.items()):
                lines.append(f"    {key}: {value:.2f}")
            lines.append("")

        return '\n'.join(lines)
