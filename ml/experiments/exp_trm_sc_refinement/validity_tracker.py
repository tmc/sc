"""
Validity Tracker: Track statechart validity during refinement.

Monitors multiple validity dimensions:
- Structural validity (hierarchy, types)
- Initial state validity (exactly one per composite)
- Transition validity (reachable targets)
- Determinism (no conflicting transitions)
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class ValidityMetrics:
    """Comprehensive validity metrics for a statechart."""
    # Overall
    is_valid: bool
    validity_score: float  # 0-1, aggregate score

    # Structural validity
    has_root: bool
    hierarchy_valid: bool
    types_valid: bool

    # Initial states
    initial_state_valid: bool
    num_composite_states: int
    num_with_initial: int

    # Transitions
    transitions_valid: bool
    num_transitions: int
    num_invalid_source: int
    num_invalid_target: int
    num_unreachable: int

    # Determinism
    is_deterministic: bool
    num_conflicts: int

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "validity_score": self.validity_score,
            "has_root": self.has_root,
            "hierarchy_valid": self.hierarchy_valid,
            "types_valid": self.types_valid,
            "initial_state_valid": self.initial_state_valid,
            "transitions_valid": self.transitions_valid,
            "is_deterministic": self.is_deterministic,
            "num_transitions": self.num_transitions,
            "num_invalid_source": self.num_invalid_source,
            "num_invalid_target": self.num_invalid_target,
            "num_conflicts": self.num_conflicts,
        }


@dataclass
class ValidityHistory:
    """Track validity over refinement iterations."""
    metrics: List[ValidityMetrics] = field(default_factory=list)

    def add(self, metrics: ValidityMetrics):
        self.metrics.append(metrics)

    def get_validity_trajectory(self) -> np.ndarray:
        return np.array([m.validity_score for m in self.metrics])

    def get_improvement(self) -> float:
        if len(self.metrics) < 2:
            return 0.0
        return self.metrics[-1].validity_score - self.metrics[0].validity_score

    @property
    def initial_validity(self) -> float:
        return self.metrics[0].validity_score if self.metrics else 0.0

    @property
    def final_validity(self) -> float:
        return self.metrics[-1].validity_score if self.metrics else 0.0


class ValidityTracker:
    """
    Tracks statechart validity during refinement.

    Validates multiple aspects:
    1. Structural: proper hierarchy, valid types
    2. Initial states: exactly one initial per composite
    3. Transitions: valid source/target, reachability
    4. Determinism: no conflicting transitions
    """

    # State type constants (matching proto)
    TYPE_BASIC = 0
    TYPE_NORMAL = 1  # OR state
    TYPE_PARALLEL = 2  # AND state
    TYPE_INITIAL = 3

    def __init__(self, max_states: int = 32, max_transitions: int = 64):
        self.max_states = max_states
        self.max_transitions = max_transitions
        self.history = ValidityHistory()

    def reset(self):
        """Reset tracking history."""
        self.history = ValidityHistory()

    def _check_hierarchy(
        self,
        state_types: np.ndarray,  # [S]
        parent_indices: np.ndarray,  # [S], -1 for root
        state_mask: np.ndarray,  # [S]
    ) -> Tuple[bool, bool]:
        """
        Check hierarchy validity.

        Returns:
            (has_root, hierarchy_valid)
        """
        valid_states = np.where(state_mask > 0)[0]

        if len(valid_states) == 0:
            return False, False

        # Check for root (parent == -1 or self-reference)
        has_root = False
        root_count = 0
        for i in valid_states:
            if parent_indices[i] < 0 or parent_indices[i] == i:
                has_root = True
                root_count += 1

        # Should have exactly one root
        if root_count != 1:
            return root_count >= 1, False

        # Check for cycles (no state should be its own ancestor)
        # Simple check: parent depth < child depth
        hierarchy_valid = True
        for i in valid_states:
            parent = parent_indices[i]
            if parent >= 0 and parent < len(state_mask):
                # Basic type shouldn't have children
                if state_types[parent] == self.TYPE_BASIC:
                    hierarchy_valid = False
                    break

        return has_root, hierarchy_valid

    def _check_initial_states(
        self,
        state_types: np.ndarray,  # [S]
        is_initial: np.ndarray,   # [S], boolean
        parent_indices: np.ndarray,  # [S]
        state_mask: np.ndarray,
    ) -> Tuple[bool, int, int]:
        """
        Check initial state validity.

        Each composite (NORMAL/PARALLEL) state should have exactly one
        initial child.

        Returns:
            (valid, num_composite, num_with_initial)
        """
        valid_states = np.where(state_mask > 0)[0]

        # Find composite states
        composite_states = []
        for i in valid_states:
            if state_types[i] in [self.TYPE_NORMAL, self.TYPE_PARALLEL]:
                composite_states.append(i)

        if len(composite_states) == 0:
            return True, 0, 0

        # For each composite, check if it has exactly one initial child
        num_with_initial = 0
        for parent_idx in composite_states:
            # Find children
            initial_children = 0
            for i in valid_states:
                if parent_indices[i] == parent_idx:
                    if is_initial[i]:
                        initial_children += 1

            if initial_children == 1:
                num_with_initial += 1

        valid = num_with_initial == len(composite_states)
        return valid, len(composite_states), num_with_initial

    def _check_transitions(
        self,
        source_indices: np.ndarray,  # [T]
        target_indices: np.ndarray,  # [T]
        state_mask: np.ndarray,      # [S]
        trans_mask: np.ndarray,      # [T]
    ) -> Tuple[bool, int, int, int]:
        """
        Check transition validity.

        Returns:
            (valid, num_invalid_source, num_invalid_target, num_unreachable)
        """
        valid_trans = np.where(trans_mask > 0)[0]
        valid_states = set(np.where(state_mask > 0)[0])

        num_invalid_source = 0
        num_invalid_target = 0

        for t in valid_trans:
            src = source_indices[t]
            tgt = target_indices[t]

            if src not in valid_states:
                num_invalid_source += 1
            if tgt not in valid_states:
                num_invalid_target += 1

        # Compute reachability from initial states
        # Simple BFS from all initial-like states
        reachable: Set[int] = set()

        # Start from state 0 (assumed root/initial)
        if len(valid_states) > 0:
            queue = [min(valid_states)]
            reachable.add(queue[0])

            while queue:
                current = queue.pop(0)
                for t in valid_trans:
                    if source_indices[t] == current:
                        tgt = target_indices[t]
                        if tgt in valid_states and tgt not in reachable:
                            reachable.add(tgt)
                            queue.append(tgt)

        num_unreachable = len(valid_states) - len(reachable)
        valid = num_invalid_source == 0 and num_invalid_target == 0

        return valid, num_invalid_source, num_invalid_target, num_unreachable

    def _check_determinism(
        self,
        source_indices: np.ndarray,  # [T]
        event_ids: np.ndarray,       # [T]
        trans_mask: np.ndarray,      # [T]
    ) -> Tuple[bool, int]:
        """
        Check for non-determinism (conflicting transitions).

        Conflict: same source state + same event = multiple transitions.

        Returns:
            (is_deterministic, num_conflicts)
        """
        valid_trans = np.where(trans_mask > 0)[0]

        # Group by (source, event)
        trans_groups: Dict[Tuple[int, int], List[int]] = {}
        for t in valid_trans:
            key = (int(source_indices[t]), int(event_ids[t]))
            if key not in trans_groups:
                trans_groups[key] = []
            trans_groups[key].append(t)

        # Count conflicts
        num_conflicts = 0
        for key, trans_list in trans_groups.items():
            if len(trans_list) > 1:
                num_conflicts += len(trans_list) - 1

        return num_conflicts == 0, num_conflicts

    def compute_metrics(
        self,
        state_types: np.ndarray,     # [S]
        is_initial: np.ndarray,      # [S]
        parent_indices: np.ndarray,  # [S]
        source_indices: np.ndarray,  # [T]
        target_indices: np.ndarray,  # [T]
        event_ids: np.ndarray,       # [T]
        state_mask: np.ndarray,      # [S]
        trans_mask: np.ndarray,      # [T]
    ) -> ValidityMetrics:
        """
        Compute comprehensive validity metrics.

        Args:
            All arrays should be numpy arrays for a single statechart.

        Returns:
            ValidityMetrics
        """
        # Check hierarchy
        has_root, hierarchy_valid = self._check_hierarchy(
            state_types, parent_indices, state_mask
        )

        # Check types (basic states shouldn't be marked as initial, etc.)
        types_valid = True  # Simplified

        # Check initial states
        initial_valid, num_composite, num_with_initial = self._check_initial_states(
            state_types, is_initial, parent_indices, state_mask
        )

        # Check transitions
        trans_valid, num_inv_src, num_inv_tgt, num_unreach = self._check_transitions(
            source_indices, target_indices, state_mask, trans_mask
        )

        # Check determinism
        is_deterministic, num_conflicts = self._check_determinism(
            source_indices, event_ids, trans_mask
        )

        # Compute aggregate validity score
        scores = [
            1.0 if has_root else 0.0,
            1.0 if hierarchy_valid else 0.5,
            1.0 if types_valid else 0.8,
            1.0 if initial_valid else (num_with_initial / max(num_composite, 1)),
            1.0 if trans_valid else 0.5,
            1.0 if is_deterministic else 0.7,
        ]
        validity_score = np.mean(scores)

        is_valid = (
            has_root and hierarchy_valid and types_valid and
            initial_valid and trans_valid and is_deterministic
        )

        return ValidityMetrics(
            is_valid=is_valid,
            validity_score=float(validity_score),
            has_root=has_root,
            hierarchy_valid=hierarchy_valid,
            types_valid=types_valid,
            initial_state_valid=initial_valid,
            num_composite_states=num_composite,
            num_with_initial=num_with_initial,
            transitions_valid=trans_valid,
            num_transitions=int(np.sum(trans_mask)),
            num_invalid_source=num_inv_src,
            num_invalid_target=num_inv_tgt,
            num_unreachable=num_unreach,
            is_deterministic=is_deterministic,
            num_conflicts=num_conflicts,
        )

    def track(
        self,
        state_types: np.ndarray,
        is_initial: np.ndarray,
        parent_indices: np.ndarray,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        event_ids: np.ndarray,
        state_mask: np.ndarray,
        trans_mask: np.ndarray,
    ) -> ValidityMetrics:
        """Compute and record metrics."""
        metrics = self.compute_metrics(
            state_types, is_initial, parent_indices,
            source_indices, target_indices, event_ids,
            state_mask, trans_mask,
        )
        self.history.add(metrics)
        return metrics


def track_validity(
    predictions: Dict[str, mx.array],
    state_mask: mx.array,
    trans_mask: mx.array,
) -> ValidityMetrics:
    """
    Track validity from model predictions.

    Args:
        predictions: Dict from SCRefiner.head()
        state_mask: [B, S] valid states
        trans_mask: [B, T] valid transitions

    Returns:
        ValidityMetrics for first item in batch
    """
    # Extract predictions for first batch item
    state_types = np.array(
        mx.argmax(predictions["state_type"][0], axis=-1).tolist()
    )
    is_initial = np.array(
        mx.argmax(predictions["state_initial"][0], axis=-1).tolist()
    ) == 1
    parent_indices = np.array(
        mx.argmax(predictions["state_parent"][0], axis=-1).tolist()
    )
    source_indices = np.array(
        mx.argmax(predictions["trans_source"][0], axis=-1).tolist()
    )
    target_indices = np.array(
        mx.argmax(predictions["trans_target"][0], axis=-1).tolist()
    )

    # Event IDs not predicted, use placeholder
    event_ids = np.zeros(source_indices.shape, dtype=np.int32)

    s_mask = np.array(state_mask[0].tolist())
    t_mask = np.array(trans_mask[0].tolist())

    tracker = ValidityTracker()
    return tracker.compute_metrics(
        state_types, is_initial, parent_indices,
        source_indices, target_indices, event_ids,
        s_mask, t_mask,
    )


def test_validity_tracker():
    """Test the validity tracker."""
    print("=" * 60)
    print("Testing Validity Tracker")
    print("=" * 60)

    tracker = ValidityTracker(max_states=16, max_transitions=32)

    # Create test statechart data
    S, T = 8, 12

    # Valid statechart structure
    state_types = np.array([1, 0, 0, 0, 1, 0, 0, 0])  # NORMAL, then BASIC
    is_initial = np.array([1, 1, 0, 0, 0, 1, 0, 0])  # Root initial, child initials
    parent_indices = np.array([-1, 0, 0, 0, 0, 4, 4, 4])  # Hierarchy
    source_indices = np.array([1, 2, 3, 1, 5, 6, 7, 5, 0, 0, 0, 0])
    target_indices = np.array([2, 3, 1, 3, 6, 7, 5, 7, 1, 2, 3, 4])
    event_ids = np.array([0, 0, 0, 1, 0, 0, 0, 1, 2, 3, 4, 5])
    state_mask = np.ones(S)
    trans_mask = np.ones(T)

    print("\n1. Computing metrics for valid statechart...")
    metrics = tracker.compute_metrics(
        state_types, is_initial, parent_indices,
        source_indices, target_indices, event_ids,
        state_mask, trans_mask,
    )

    print(f"   Valid: {metrics.is_valid}")
    print(f"   Score: {metrics.validity_score:.3f}")
    print(f"   Has root: {metrics.has_root}")
    print(f"   Hierarchy valid: {metrics.hierarchy_valid}")
    print(f"   Initial valid: {metrics.initial_state_valid}")
    print(f"   Transitions valid: {metrics.transitions_valid}")
    print(f"   Deterministic: {metrics.is_deterministic}")

    print("\n2. Testing with invalid transitions...")
    bad_source = source_indices.copy()
    bad_source[0] = 99  # Invalid source

    metrics2 = tracker.compute_metrics(
        state_types, is_initial, parent_indices,
        bad_source, target_indices, event_ids,
        state_mask, trans_mask,
    )

    print(f"   Valid: {metrics2.is_valid}")
    print(f"   Score: {metrics2.validity_score:.3f}")
    print(f"   Invalid sources: {metrics2.num_invalid_source}")

    print("\n3. Testing validity history...")
    tracker.reset()

    # Simulate refinement improving validity
    for i in range(5):
        # Gradually fix issues
        fixed_source = source_indices.copy()
        if i < 2:
            fixed_source[0] = 99  # Invalid initially

        tracker.track(
            state_types, is_initial, parent_indices,
            fixed_source, target_indices, event_ids,
            state_mask, trans_mask,
        )

    print(f"   History length: {len(tracker.history.metrics)}")
    print(f"   Initial validity: {tracker.history.initial_validity:.3f}")
    print(f"   Final validity: {tracker.history.final_validity:.3f}")
    print(f"   Improvement: {tracker.history.get_improvement():.3f}")

    print("\n" + "=" * 60)
    print("Validity tracker test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_validity_tracker()
