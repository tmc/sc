"""
Iteration Analyzer: Analyze TRM reasoning iterations with mlux-style hooks.

Extracts activations at each (H, L) iteration step to understand:
1. How representations evolve during recursive refinement
2. Which iterations are most important for solving
3. Convergence patterns across different puzzle difficulties
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import from exp_trm_sudoku_9x9
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


@dataclass
class IterationSnapshot:
    """Snapshot of model state at a single iteration."""
    h_cycle: int                    # Outer cycle index (0 to H-1)
    l_cycle: int                    # Inner cycle index (0 to L-1)
    iteration_idx: int              # Linear index (h * L + l)

    # Board state
    board_states: np.ndarray        # [81, 10] soft probabilities
    board_entropy: np.ndarray       # [81] entropy per cell
    board_confidence: np.ndarray    # [81] max probability per cell

    # Context embedding
    context: np.ndarray             # [hidden_dim] board context
    h_context: np.ndarray           # [hidden_dim] H-level context

    # Guard values
    guard_values: np.ndarray        # [81, 9] constraint satisfaction
    guard_activations: float        # Mean guard activation

    # Predictions at this step
    predictions: np.ndarray         # [81] argmax predictions
    num_correct: int                # Cells matching target
    num_changed: int                # Cells changed from previous iteration

    @property
    def is_early(self) -> bool:
        return self.h_cycle == 0

    @property
    def is_late(self) -> bool:
        return self.h_cycle >= 2

    def to_dict(self) -> Dict:
        return {
            "h_cycle": self.h_cycle,
            "l_cycle": self.l_cycle,
            "iteration_idx": self.iteration_idx,
            "board_entropy_mean": float(np.mean(self.board_entropy)),
            "board_confidence_mean": float(np.mean(self.board_confidence)),
            "guard_activations": self.guard_activations,
            "num_correct": self.num_correct,
            "num_changed": self.num_changed,
        }


@dataclass
class IterationTrajectory:
    """Full trajectory of a puzzle through all iterations."""
    puzzle_id: int
    question: np.ndarray            # [81] input
    answer: np.ndarray              # [81] target
    snapshots: List[IterationSnapshot] = field(default_factory=list)

    # Final results
    final_prediction: np.ndarray = None
    is_correct: bool = False
    convergence_iteration: Optional[int] = None  # When solution stabilized

    @property
    def num_iterations(self) -> int:
        return len(self.snapshots)

    def get_entropy_trajectory(self) -> np.ndarray:
        """Get mean entropy at each iteration."""
        return np.array([np.mean(s.board_entropy) for s in self.snapshots])

    def get_confidence_trajectory(self) -> np.ndarray:
        """Get mean confidence at each iteration."""
        return np.array([np.mean(s.board_confidence) for s in self.snapshots])

    def get_accuracy_trajectory(self) -> np.ndarray:
        """Get accuracy (fraction correct) at each iteration."""
        return np.array([s.num_correct / 81.0 for s in self.snapshots])

    def get_change_trajectory(self) -> np.ndarray:
        """Get number of cells changed at each iteration."""
        return np.array([s.num_changed for s in self.snapshots])

    def find_convergence(self, threshold: int = 0) -> Optional[int]:
        """Find iteration where predictions stabilize."""
        for i, s in enumerate(self.snapshots[1:], 1):
            if s.num_changed <= threshold:
                return i
        return None

    def to_dict(self) -> Dict:
        return {
            "puzzle_id": self.puzzle_id,
            "is_correct": self.is_correct,
            "convergence_iteration": self.convergence_iteration,
            "num_iterations": self.num_iterations,
            "final_accuracy": self.snapshots[-1].num_correct / 81.0 if self.snapshots else 0,
            "entropy_trajectory": self.get_entropy_trajectory().tolist(),
            "accuracy_trajectory": self.get_accuracy_trajectory().tolist(),
        }


class IterationAnalyzer:
    """
    Analyzes TRM iteration dynamics using mlux-style hooks.

    Hooks into the Sudoku statechart model at each (H, L) iteration
    to capture activations and track reasoning evolution.
    """

    def __init__(
        self,
        model,  # SudokuStatechart9x9
        H_cycles: int = 3,
        L_cycles: int = 6,
    ):
        """
        Initialize analyzer.

        Args:
            model: Trained SudokuStatechart9x9 model
            H_cycles: Number of outer cycles
            L_cycles: Number of inner cycles
        """
        self.model = model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.total_iterations = H_cycles * L_cycles

        # Hooks storage
        self._hooks: List[Callable] = []
        self._activation_cache: Dict[str, Any] = {}

    def add_hook(
        self,
        name: str,
        hook_fn: Callable[[str, int, int, Any], None],
    ):
        """
        Add a hook to be called at each iteration.

        Args:
            name: Hook identifier
            hook_fn: Function(name, h_cycle, l_cycle, activations)
        """
        self._hooks.append((name, hook_fn))

    def _compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute entropy of probability distributions."""
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs), axis=-1)

    def _compute_confidence(self, probs: np.ndarray) -> np.ndarray:
        """Compute max probability (confidence)."""
        return np.max(probs, axis=-1)

    def analyze_puzzle(
        self,
        question: mx.array,  # [81]
        answer: mx.array,    # [81]
        puzzle_id: int = 0,
    ) -> IterationTrajectory:
        """
        Analyze a single puzzle through all iterations.

        Args:
            question: Input puzzle
            answer: Target solution
            puzzle_id: Puzzle identifier

        Returns:
            IterationTrajectory with all snapshots
        """
        # Initialize trajectory
        trajectory = IterationTrajectory(
            puzzle_id=puzzle_id,
            question=np.array(question.tolist()),
            answer=np.array(answer.tolist()),
        )

        # Convert to soft states
        board = self.model.board_to_soft(question[None, :])  # [1, 81, 10]
        prev_predictions = None

        # Run through iterations
        for h in range(self.H_cycles):
            # Get H-level context
            h_context = self.model.encode_board(board)
            h_context = self.model.h_context_net(h_context)

            for l in range(self.L_cycles):
                # Get current context
                context = self.model.encode_board(board) + h_context

                # Collect guard values
                guard_matrix = []
                for cell_id in range(81):
                    guards = self.model.guard(board, cell_id)
                    guard_matrix.append(guards)
                guard_matrix = mx.stack(guard_matrix, axis=1)  # [1, 81, 9]

                # Get predictions
                logits = self.model.output_head(board)
                predictions = mx.argmax(logits, axis=-1)[0]  # [81]

                # Compute metrics
                board_np = np.array(board[0].tolist())
                entropy = self._compute_entropy(board_np)
                confidence = self._compute_confidence(board_np)

                pred_np = np.array(predictions.tolist())
                answer_np = np.array(answer.tolist())
                num_correct = int(np.sum(pred_np == answer_np))

                if prev_predictions is not None:
                    num_changed = int(np.sum(pred_np != prev_predictions))
                else:
                    num_changed = 81

                # Create snapshot
                snapshot = IterationSnapshot(
                    h_cycle=h,
                    l_cycle=l,
                    iteration_idx=h * self.L_cycles + l,
                    board_states=board_np,
                    board_entropy=entropy,
                    board_confidence=confidence,
                    context=np.array(context[0].tolist()),
                    h_context=np.array(h_context[0].tolist()),
                    guard_values=np.array(guard_matrix[0].tolist()),
                    guard_activations=float(np.mean(guard_matrix.tolist())),
                    predictions=pred_np,
                    num_correct=num_correct,
                    num_changed=num_changed,
                )

                trajectory.snapshots.append(snapshot)
                prev_predictions = pred_np

                # Call hooks
                for name, hook_fn in self._hooks:
                    hook_fn(name, h, l, {
                        "board": board_np,
                        "context": snapshot.context,
                        "guards": snapshot.guard_values,
                        "predictions": pred_np,
                    })

                # Step the model
                board = self.model.step(board, context)

        # Final prediction
        logits = self.model.output_head(board)
        final_pred = mx.argmax(logits, axis=-1)[0]
        trajectory.final_prediction = np.array(final_pred.tolist())
        trajectory.is_correct = bool(mx.all(final_pred == answer).item())
        trajectory.convergence_iteration = trajectory.find_convergence()

        return trajectory

    def analyze_batch(
        self,
        questions: mx.array,  # [B, 81]
        answers: mx.array,    # [B, 81]
    ) -> List[IterationTrajectory]:
        """Analyze a batch of puzzles."""
        B = questions.shape[0]
        trajectories = []

        for i in range(B):
            trajectory = self.analyze_puzzle(
                questions[i],
                answers[i],
                puzzle_id=i,
            )
            trajectories.append(trajectory)

        return trajectories


def analyze_iterations(
    model,
    questions: mx.array,
    answers: mx.array,
    H_cycles: int = 3,
    L_cycles: int = 6,
) -> List[IterationTrajectory]:
    """
    Convenience function to analyze iterations.

    Args:
        model: Trained SudokuStatechart9x9
        questions: [B, 81] input puzzles
        answers: [B, 81] target solutions
        H_cycles: Number of outer cycles
        L_cycles: Number of inner cycles

    Returns:
        List of IterationTrajectory for each puzzle
    """
    analyzer = IterationAnalyzer(model, H_cycles, L_cycles)
    return analyzer.analyze_batch(questions, answers)


def summarize_trajectories(trajectories: List[IterationTrajectory]) -> Dict:
    """Generate summary statistics for trajectories."""
    n = len(trajectories)
    if n == 0:
        return {}

    correct_count = sum(1 for t in trajectories if t.is_correct)
    convergence_iters = [t.convergence_iteration for t in trajectories if t.convergence_iteration]

    # Aggregate trajectories
    entropy_trajs = np.array([t.get_entropy_trajectory() for t in trajectories])
    accuracy_trajs = np.array([t.get_accuracy_trajectory() for t in trajectories])
    change_trajs = np.array([t.get_change_trajectory() for t in trajectories])

    return {
        "num_puzzles": n,
        "accuracy": correct_count / n,
        "avg_convergence_iter": np.mean(convergence_iters) if convergence_iters else None,
        "entropy_by_iteration": np.mean(entropy_trajs, axis=0).tolist(),
        "accuracy_by_iteration": np.mean(accuracy_trajs, axis=0).tolist(),
        "changes_by_iteration": np.mean(change_trajs, axis=0).tolist(),
    }


def test_iteration_analyzer():
    """Test the iteration analyzer."""
    print("=" * 60)
    print("Testing Iteration Analyzer")
    print("=" * 60)

    # Import model
    from ..exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
    )

    # Create analyzer
    analyzer = IterationAnalyzer(model, H_cycles=2, L_cycles=3)

    # Add hook
    hook_activations = []
    def my_hook(name, h, l, acts):
        hook_activations.append((h, l, acts["predictions"][:5]))

    analyzer.add_hook("test_hook", my_hook)

    # Create test data
    mx.random.seed(42)
    question = mx.random.randint(0, 10, (81,))
    answer = mx.random.randint(1, 10, (81,))

    print("\n1. Analyzing single puzzle...")
    trajectory = analyzer.analyze_puzzle(question, answer, puzzle_id=0)

    print(f"   Iterations: {trajectory.num_iterations}")
    print(f"   Is correct: {trajectory.is_correct}")
    print(f"   Convergence: {trajectory.convergence_iteration}")
    print(f"   Hook calls: {len(hook_activations)}")

    print("\n2. Trajectory dynamics:")
    entropy_traj = trajectory.get_entropy_trajectory()
    accuracy_traj = trajectory.get_accuracy_trajectory()
    change_traj = trajectory.get_change_trajectory()

    print(f"   Entropy: {entropy_traj}")
    print(f"   Accuracy: {accuracy_traj}")
    print(f"   Changes: {change_traj}")

    print("\n3. Snapshot details (first 3):")
    for snap in trajectory.snapshots[:3]:
        print(f"   H={snap.h_cycle}, L={snap.l_cycle}: "
              f"correct={snap.num_correct}/81, "
              f"changed={snap.num_changed}, "
              f"entropy={np.mean(snap.board_entropy):.3f}")

    print("\n4. Testing batch analysis...")
    questions = mx.random.randint(0, 10, (4, 81))
    answers = mx.random.randint(1, 10, (4, 81))

    trajectories = analyzer.analyze_batch(questions, answers)
    print(f"   Analyzed {len(trajectories)} puzzles")

    summary = summarize_trajectories(trajectories)
    print(f"   Summary: {json.dumps(summary, indent=2)}")

    print("\n" + "=" * 60)
    print("Iteration analyzer test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_iteration_analyzer()
