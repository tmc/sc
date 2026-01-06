"""
Activation Tracker: Track how representations evolve during TRM refinement.

Uses mlux-style hooks to capture activations at each layer/iteration
and analyze how the model's internal representations develop.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class ActivationHook:
    """Configuration for an activation hook."""
    name: str
    module_path: str           # e.g., "board_encoder", "cell_transition"
    capture_input: bool = False
    capture_output: bool = True
    transform: Optional[Callable] = None  # Optional transform on captured values


@dataclass
class ActivationRecord:
    """Record of captured activations."""
    hook_name: str
    iteration_idx: int
    h_cycle: int
    l_cycle: int
    shape: Tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    l2_norm: float
    sparsity: float  # Fraction of near-zero values


@dataclass
class ActivationEvolution:
    """Tracks how a specific activation evolves over iterations."""
    hook_name: str
    records: List[ActivationRecord] = field(default_factory=list)

    def get_mean_trajectory(self) -> np.ndarray:
        return np.array([r.mean for r in self.records])

    def get_std_trajectory(self) -> np.ndarray:
        return np.array([r.std for r in self.records])

    def get_norm_trajectory(self) -> np.ndarray:
        return np.array([r.l2_norm for r in self.records])

    def get_sparsity_trajectory(self) -> np.ndarray:
        return np.array([r.sparsity for r in self.records])


class ActivationTracker:
    """
    Tracks activations during TRM forward passes.

    Inspired by mlux's HookedModel, but specialized for TRM iteration
    analysis rather than general transformer interpretability.
    """

    def __init__(
        self,
        model,  # SudokuStatechart9x9
        hooks: Optional[List[ActivationHook]] = None,
    ):
        """
        Initialize tracker.

        Args:
            model: The model to track
            hooks: List of hooks to install (default: common hooks)
        """
        self.model = model

        # Default hooks for TRM-style Sudoku model
        if hooks is None:
            hooks = [
                ActivationHook("board_encoder", "board_encoder", capture_output=True),
                ActivationHook("h_context", "h_context_net", capture_output=True),
                ActivationHook("cell_transition", "cell_transition", capture_output=True),
                ActivationHook("output_head", "output_head", capture_output=True),
            ]

        self.hooks = {h.name: h for h in hooks}

        # Activation storage
        self._cache: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._records: Dict[str, List[ActivationRecord]] = defaultdict(list)
        self._current_iteration = 0

    def clear_cache(self):
        """Clear activation cache."""
        self._cache.clear()
        self._records.clear()
        self._current_iteration = 0

    def _compute_stats(
        self,
        arr: np.ndarray,
        hook_name: str,
        h: int,
        l: int,
    ) -> ActivationRecord:
        """Compute activation statistics."""
        flat = arr.flatten()

        return ActivationRecord(
            hook_name=hook_name,
            iteration_idx=self._current_iteration,
            h_cycle=h,
            l_cycle=l,
            shape=arr.shape,
            mean=float(np.mean(flat)),
            std=float(np.std(flat)),
            min_val=float(np.min(flat)),
            max_val=float(np.max(flat)),
            l2_norm=float(np.linalg.norm(flat)),
            sparsity=float(np.mean(np.abs(flat) < 0.01)),
        )

    def capture(
        self,
        hook_name: str,
        activation: mx.array,
        h_cycle: int,
        l_cycle: int,
    ):
        """
        Capture an activation.

        Args:
            hook_name: Name of the hook
            activation: The activation to capture
            h_cycle: Current H cycle
            l_cycle: Current L cycle
        """
        if hook_name not in self.hooks:
            return

        hook = self.hooks[hook_name]

        # Convert to numpy
        arr = np.array(activation.tolist())

        # Apply transform if specified
        if hook.transform:
            arr = hook.transform(arr)

        # Store
        self._cache[hook_name].append(arr)

        # Compute stats
        record = self._compute_stats(arr, hook_name, h_cycle, l_cycle)
        self._records[hook_name].append(record)

    def track_forward(
        self,
        question: mx.array,  # [81]
        H_cycles: int = 3,
        L_cycles: int = 6,
    ) -> Dict[str, ActivationEvolution]:
        """
        Track activations through a full forward pass.

        Args:
            question: Input puzzle [81]
            H_cycles: Number of outer cycles
            L_cycles: Number of inner cycles

        Returns:
            Dict mapping hook names to ActivationEvolution
        """
        self.clear_cache()

        # Convert to soft states
        board = self.model.board_to_soft(question[None, :])

        for h in range(H_cycles):
            # H-level context
            board_context = self.model.encode_board(board)
            self.capture("board_encoder", board_context, h, 0)

            h_context = self.model.h_context_net(board_context)
            self.capture("h_context", h_context, h, 0)

            for l in range(L_cycles):
                # Full context
                context = self.model.encode_board(board) + h_context

                # Track cell transitions (aggregate)
                trans_outputs = []
                for cell_id in range(81):
                    pos_emb = self.model.pos_embed(mx.array([cell_id]))
                    cell_context = context + pos_emb
                    trans_score = self.model.cell_transition(cell_context)
                    trans_outputs.append(trans_score)

                trans_matrix = mx.stack(trans_outputs, axis=1)  # [1, 81, 9]
                self.capture("cell_transition", trans_matrix, h, l)

                # Output head
                logits = self.model.output_head(board)
                self.capture("output_head", logits, h, l)

                # Step model
                board = self.model.step(board, context)

                self._current_iteration += 1

        # Build evolution objects
        evolutions = {}
        for hook_name, records in self._records.items():
            evolutions[hook_name] = ActivationEvolution(
                hook_name=hook_name,
                records=records,
            )

        return evolutions

    def get_activation_matrix(
        self,
        hook_name: str,
    ) -> Optional[np.ndarray]:
        """
        Get cached activations as a matrix.

        Returns:
            [num_iterations, ...] array or None
        """
        if hook_name not in self._cache:
            return None

        return np.stack(self._cache[hook_name], axis=0)

    def compute_representation_drift(
        self,
        hook_name: str,
    ) -> np.ndarray:
        """
        Compute how much representations change between iterations.

        Returns:
            [num_iterations - 1] array of L2 distances
        """
        acts = self.get_activation_matrix(hook_name)
        if acts is None or len(acts) < 2:
            return np.array([])

        # Flatten to vectors
        acts_flat = acts.reshape(len(acts), -1)

        # Compute consecutive distances
        drifts = []
        for i in range(1, len(acts_flat)):
            dist = np.linalg.norm(acts_flat[i] - acts_flat[i-1])
            drifts.append(dist)

        return np.array(drifts)

    def compute_representation_similarity(
        self,
        hook_name: str,
    ) -> np.ndarray:
        """
        Compute cosine similarity between consecutive iterations.

        Returns:
            [num_iterations - 1] array of similarities
        """
        acts = self.get_activation_matrix(hook_name)
        if acts is None or len(acts) < 2:
            return np.array([])

        # Flatten to vectors
        acts_flat = acts.reshape(len(acts), -1)

        # Compute consecutive cosine similarities
        sims = []
        for i in range(1, len(acts_flat)):
            norm1 = np.linalg.norm(acts_flat[i-1])
            norm2 = np.linalg.norm(acts_flat[i])
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(acts_flat[i-1], acts_flat[i]) / (norm1 * norm2)
            else:
                sim = 0.0
            sims.append(sim)

        return np.array(sims)


def track_activations(
    model,
    questions: mx.array,  # [B, 81]
    H_cycles: int = 3,
    L_cycles: int = 6,
) -> List[Dict[str, ActivationEvolution]]:
    """
    Track activations for a batch of puzzles.

    Args:
        model: SudokuStatechart9x9 model
        questions: [B, 81] input puzzles
        H_cycles: Number of outer cycles
        L_cycles: Number of inner cycles

    Returns:
        List of evolution dicts, one per puzzle
    """
    B = questions.shape[0]
    results = []

    tracker = ActivationTracker(model)

    for i in range(B):
        evolutions = tracker.track_forward(questions[i], H_cycles, L_cycles)
        results.append(evolutions)

    return results


def analyze_evolution(evolutions: Dict[str, ActivationEvolution]) -> Dict:
    """Analyze activation evolution patterns."""
    analysis = {}

    for name, evo in evolutions.items():
        analysis[name] = {
            "num_iterations": len(evo.records),
            "mean_trajectory": evo.get_mean_trajectory().tolist(),
            "std_trajectory": evo.get_std_trajectory().tolist(),
            "norm_trajectory": evo.get_norm_trajectory().tolist(),
            "sparsity_trajectory": evo.get_sparsity_trajectory().tolist(),
            "mean_increase": float(
                evo.get_mean_trajectory()[-1] - evo.get_mean_trajectory()[0]
            ) if len(evo.records) > 0 else 0,
            "final_sparsity": float(evo.get_sparsity_trajectory()[-1])
                if len(evo.records) > 0 else 0,
        }

    return analysis


def test_activation_tracker():
    """Test the activation tracker."""
    print("=" * 60)
    print("Testing Activation Tracker")
    print("=" * 60)

    from ..exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
    )

    # Create tracker
    tracker = ActivationTracker(model)

    # Create test data
    mx.random.seed(42)
    question = mx.random.randint(0, 10, (81,))

    print("\n1. Tracking forward pass...")
    evolutions = tracker.track_forward(question, H_cycles=2, L_cycles=3)

    print(f"   Tracked hooks: {list(evolutions.keys())}")

    for name, evo in evolutions.items():
        print(f"\n   {name}:")
        print(f"     Iterations: {len(evo.records)}")
        print(f"     Mean trajectory: {evo.get_mean_trajectory()[:3]}...")
        print(f"     Norm trajectory: {evo.get_norm_trajectory()[:3]}...")

    print("\n2. Computing representation drift...")
    for name in evolutions.keys():
        drift = tracker.compute_representation_drift(name)
        print(f"   {name}: drift = {drift[:3]}...")

    print("\n3. Computing representation similarity...")
    for name in evolutions.keys():
        sim = tracker.compute_representation_similarity(name)
        print(f"   {name}: similarity = {sim[:3]}...")

    print("\n4. Full evolution analysis:")
    analysis = analyze_evolution(evolutions)
    for name, stats in analysis.items():
        print(f"   {name}: final_sparsity={stats['final_sparsity']:.3f}")

    print("\n" + "=" * 60)
    print("Activation tracker test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_activation_tracker()
