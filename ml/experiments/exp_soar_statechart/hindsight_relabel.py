"""
Hindsight Relabeling for Statechart Programs

Turns failed program executions into valid training data by relabeling:
- Failed trajectory -> "correct" trajectory for a synthetic task
- The synthetic task has the failed output as its expected output

Key insight: Every program execution is "correct" for SOME task.
This dramatically increases training data efficiency.

SOAR-style HER (Hindsight Experience Replay):
1. Execute program on task (may fail)
2. If output != expected, create synthetic task where output IS expected
3. Use (input, synthetic_output) as positive training example
4. Train model to reproduce this "successful" execution
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
import random
import copy

try:
    from .soar_statechart import (
        StatechartProgram, ProgramState, Transition, Guard, GridAction
    )
except ImportError:
    from soar_statechart import (
        StatechartProgram, ProgramState, Transition, Guard, GridAction
    )


# =============================================================================
# State Trajectories
# =============================================================================

@dataclass
class StateSnapshot:
    """Snapshot of execution state at one step."""
    state_label: str
    grid: mx.array
    step: int
    action_applied: Optional[GridAction] = None
    transition_taken: Optional[Transition] = None


@dataclass
class StateTrajectory:
    """Complete execution trajectory through statechart."""
    program_id: str
    task_id: str
    input_grid: mx.array
    output_grid: mx.array
    expected_grid: mx.array
    snapshots: List[StateSnapshot] = field(default_factory=list)
    success: bool = False

    @property
    def is_correct(self) -> bool:
        """Check if trajectory produced correct output."""
        return bool(mx.all(self.output_grid == self.expected_grid))

    @property
    def partial_match(self) -> float:
        """Fraction of cells matching expected output."""
        if self.output_grid.size == 0:
            return 0.0
        matches = mx.sum(self.output_grid == self.expected_grid)
        return float(matches) / self.output_grid.size

    def get_state_sequence(self) -> List[str]:
        """Get sequence of state labels visited."""
        return [s.state_label for s in self.snapshots]

    def get_action_sequence(self) -> List[GridAction]:
        """Get sequence of actions applied."""
        return [s.action_applied for s in self.snapshots if s.action_applied]


@dataclass
class SyntheticTask:
    """A synthetic task created via hindsight relabeling."""
    original_task_id: str
    synthetic_id: str
    input_grid: mx.array
    output_grid: mx.array  # The "correct" output (what program actually produced)
    source_trajectory: Optional[StateTrajectory] = None

    # Quality metrics
    complexity: float = 0.0  # How different from original
    diversity: float = 0.0   # How different from other synthetics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_task_id": self.original_task_id,
            "synthetic_id": self.synthetic_id,
            "input_grid": self.input_grid.tolist(),
            "output_grid": self.output_grid.tolist(),
            "complexity": self.complexity,
            "diversity": self.diversity,
        }


# =============================================================================
# Trajectory Collector
# =============================================================================

class TrajectoryCollector:
    """Collects execution trajectories from programs."""

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

    def collect(
        self,
        program: StatechartProgram,
        input_grid: mx.array,
        expected_grid: mx.array,
        task_id: str = "",
    ) -> StateTrajectory:
        """
        Execute program and collect trajectory.

        Args:
            program: Statechart program to execute
            input_grid: Input grid
            expected_grid: Expected output
            task_id: Task identifier

        Returns:
            StateTrajectory with execution history
        """
        trajectory = StateTrajectory(
            program_id=str(id(program)),
            task_id=task_id,
            input_grid=input_grid,
            output_grid=input_grid,  # Will be updated
            expected_grid=expected_grid,
        )

        # Reset program state
        program._enter_initial()
        context = {"grid": input_grid}

        for step in range(self.max_steps):
            # Record current state
            current_states = list(program.current_config)
            if not current_states:
                break

            # Find enabled transitions
            enabled = []
            for trans in program.transitions:
                if trans.source in program.current_config:
                    if trans.is_enabled(context):
                        enabled.append(trans)

            if not enabled:
                # Check if done
                is_done = any(
                    program.states[s].is_final
                    for s in program.current_config
                    if s in program.states
                )
                if is_done:
                    break
                continue

            # Select transition
            enabled.sort(key=lambda t: -t.priority)
            trans = enabled[0]

            # Take snapshot before transition
            snapshot = StateSnapshot(
                state_label=trans.source,
                grid=mx.array(context["grid"]),
                step=step,
                transition_taken=trans,
            )

            # Execute transition
            program.current_config.discard(trans.source)
            program._enter_state(trans.target)

            # Execute action
            target_state = program.states.get(trans.target)
            if target_state and target_state.action:
                context["grid"] = target_state.action.apply(context["grid"])
                snapshot.action_applied = target_state.action

            trajectory.snapshots.append(snapshot)

            # Check if done
            is_done = any(
                program.states[s].is_final
                for s in program.current_config
                if s in program.states
            )
            if is_done:
                break

        trajectory.output_grid = context["grid"]
        trajectory.success = trajectory.is_correct

        return trajectory


# =============================================================================
# Hindsight Relabeler
# =============================================================================

class HindsightRelabeler:
    """
    Creates synthetic training data from failed executions.

    For each failed trajectory:
    1. Create synthetic task with actual output as expected
    2. The trajectory is now "correct" for this synthetic task
    3. Use for training to improve program diversity
    """

    def __init__(
        self,
        min_complexity: float = 0.1,
        max_synthetic_per_task: int = 100,
        diversity_threshold: float = 0.3,
    ):
        self.min_complexity = min_complexity
        self.max_synthetic_per_task = max_synthetic_per_task
        self.diversity_threshold = diversity_threshold

        # Storage
        self.trajectories: Dict[str, List[StateTrajectory]] = defaultdict(list)
        self.synthetic_tasks: Dict[str, List[SyntheticTask]] = defaultdict(list)
        self.synthetic_id_counter = 0

    def add_trajectory(self, trajectory: StateTrajectory):
        """Add a trajectory to collection."""
        self.trajectories[trajectory.task_id].append(trajectory)

    def _compute_complexity(
        self,
        input_grid: mx.array,
        output_grid: mx.array,
        expected_grid: mx.array
    ) -> float:
        """
        Compute complexity of synthetic task.

        Higher complexity = more transformation from input to output.
        """
        # Cells changed from input to output
        changed_from_input = float(mx.mean(input_grid != output_grid))

        # How different from original expected
        diff_from_expected = float(mx.mean(output_grid != expected_grid))

        return (changed_from_input + diff_from_expected) / 2

    def _compute_diversity(
        self,
        new_output: mx.array,
        existing_outputs: List[mx.array]
    ) -> float:
        """
        Compute how diverse this output is from existing synthetics.
        """
        if not existing_outputs:
            return 1.0

        min_diff = float('inf')
        for existing in existing_outputs:
            if existing.shape == new_output.shape:
                diff = float(mx.mean(new_output != existing))
                min_diff = min(min_diff, diff)

        return min_diff if min_diff != float('inf') else 1.0

    def relabel_trajectory(
        self,
        trajectory: StateTrajectory
    ) -> Optional[SyntheticTask]:
        """
        Create synthetic task from a (possibly failed) trajectory.

        Returns None if trajectory doesn't meet quality criteria.
        """
        # Don't relabel already-successful trajectories (they're real data)
        if trajectory.is_correct:
            return None

        # Check if output is interesting (not just identity)
        if mx.all(trajectory.output_grid == trajectory.input_grid):
            return None

        # Compute complexity
        complexity = self._compute_complexity(
            trajectory.input_grid,
            trajectory.output_grid,
            trajectory.expected_grid
        )

        if complexity < self.min_complexity:
            return None

        # Check diversity against existing synthetics
        existing_outputs = [
            st.output_grid for st in self.synthetic_tasks[trajectory.task_id]
        ]
        diversity = self._compute_diversity(trajectory.output_grid, existing_outputs)

        if diversity < self.diversity_threshold:
            return None

        # Check max synthetics per task
        if len(self.synthetic_tasks[trajectory.task_id]) >= self.max_synthetic_per_task:
            return None

        # Create synthetic task
        self.synthetic_id_counter += 1
        synthetic = SyntheticTask(
            original_task_id=trajectory.task_id,
            synthetic_id=f"synthetic_{self.synthetic_id_counter}",
            input_grid=trajectory.input_grid,
            output_grid=trajectory.output_grid,
            source_trajectory=trajectory,
            complexity=complexity,
            diversity=diversity,
        )

        self.synthetic_tasks[trajectory.task_id].append(synthetic)
        return synthetic

    def relabel_all(self) -> List[SyntheticTask]:
        """Relabel all collected trajectories."""
        all_synthetic = []

        for task_id, trajectories in self.trajectories.items():
            for traj in trajectories:
                synthetic = self.relabel_trajectory(traj)
                if synthetic:
                    all_synthetic.append(synthetic)

        return all_synthetic

    def get_training_data(
        self,
        include_original: bool = True,
        include_synthetic: bool = True,
    ) -> List[Tuple[mx.array, mx.array, str]]:
        """
        Get combined training data (original + synthetic).

        Returns list of (input, output, task_id) tuples.
        """
        data = []

        if include_original:
            # Add successful trajectories as-is
            for task_id, trajectories in self.trajectories.items():
                for traj in trajectories:
                    if traj.is_correct:
                        data.append((
                            traj.input_grid,
                            traj.output_grid,
                            task_id
                        ))

        if include_synthetic:
            # Add synthetic tasks
            for task_id, synthetics in self.synthetic_tasks.items():
                for syn in synthetics:
                    data.append((
                        syn.input_grid,
                        syn.output_grid,
                        syn.synthetic_id
                    ))

        return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get relabeling statistics."""
        total_trajectories = sum(len(t) for t in self.trajectories.values())
        successful = sum(
            1 for t in self.trajectories.values()
            for traj in t if traj.is_correct
        )
        total_synthetic = sum(len(s) for s in self.synthetic_tasks.values())

        avg_complexity = 0.0
        avg_diversity = 0.0
        if total_synthetic > 0:
            all_syn = [s for ss in self.synthetic_tasks.values() for s in ss]
            avg_complexity = sum(s.complexity for s in all_syn) / total_synthetic
            avg_diversity = sum(s.diversity for s in all_syn) / total_synthetic

        return {
            "total_trajectories": total_trajectories,
            "successful_trajectories": successful,
            "failed_trajectories": total_trajectories - successful,
            "synthetic_tasks": total_synthetic,
            "avg_complexity": avg_complexity,
            "avg_diversity": avg_diversity,
            "data_amplification": (
                (total_synthetic + successful) / max(1, successful)
            ),
        }


# =============================================================================
# Hindsight State Relabeler (for statechart learning)
# =============================================================================

class HindsightStateRelabeler:
    """
    Specialized relabeler for learning statechart transitions.

    Creates synthetic transition data from trajectories:
    - Each (state, action, next_state) triple is a training example
    - Failed trajectories provide negative examples
    - Synthetic tasks provide additional positive examples
    """

    def __init__(self):
        self.transition_examples: List[Dict[str, Any]] = []
        self.state_examples: List[Dict[str, Any]] = []

    def extract_from_trajectory(
        self,
        trajectory: StateTrajectory,
        is_positive: bool = True
    ):
        """Extract transition examples from trajectory."""
        for i, snapshot in enumerate(trajectory.snapshots):
            if snapshot.transition_taken:
                trans = snapshot.transition_taken

                example = {
                    "source_state": trans.source,
                    "target_state": trans.target,
                    "guard": trans.guard.predicate if trans.guard else "true",
                    "action": snapshot.action_applied,
                    "grid_before": snapshot.grid,
                    "grid_after": (
                        trajectory.snapshots[i + 1].grid
                        if i + 1 < len(trajectory.snapshots)
                        else trajectory.output_grid
                    ),
                    "is_positive": is_positive,
                    "task_id": trajectory.task_id,
                }

                self.transition_examples.append(example)

            # State examples for state classification
            self.state_examples.append({
                "state_label": snapshot.state_label,
                "grid": snapshot.grid,
                "step": snapshot.step,
                "task_id": trajectory.task_id,
            })

    def get_guard_training_data(
        self
    ) -> List[Tuple[mx.array, str, bool]]:
        """
        Get training data for guard prediction.

        Returns (grid, guard_predicate, is_correct) tuples.
        """
        return [
            (ex["grid_before"], ex["guard"], ex["is_positive"])
            for ex in self.transition_examples
        ]

    def get_action_training_data(
        self
    ) -> List[Tuple[mx.array, GridAction, mx.array]]:
        """
        Get training data for action prediction.

        Returns (grid_before, action, grid_after) tuples.
        """
        return [
            (ex["grid_before"], ex["action"], ex["grid_after"])
            for ex in self.transition_examples
            if ex["action"] is not None
        ]


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate hindsight relabeling."""
    print("=" * 60)
    print("Hindsight Relabeling for Statecharts")
    print("=" * 60)

    from soar_statechart import SOARStatechart

    # Create some programs
    soar = SOARStatechart()
    programs = [soar.random_program() for _ in range(10)]

    # Create task
    input_grid = mx.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    expected_grid = mx.where(input_grid == 1, 2, input_grid)

    # Collect trajectories
    collector = TrajectoryCollector()
    relabeler = HindsightRelabeler()

    print("\nCollecting trajectories...")
    for i, prog in enumerate(programs):
        traj = collector.collect(
            program=prog,
            input_grid=input_grid,
            expected_grid=expected_grid,
            task_id="task_0"
        )
        relabeler.add_trajectory(traj)
        status = "✓" if traj.is_correct else f"✗ ({traj.partial_match:.1%} match)"
        print(f"  Program {i}: {status}")

    # Relabel
    print("\nRelabeling failed trajectories...")
    synthetics = relabeler.relabel_all()

    print(f"\nCreated {len(synthetics)} synthetic tasks")

    # Statistics
    stats = relabeler.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Successful: {stats['successful_trajectories']}")
    print(f"  Failed: {stats['failed_trajectories']}")
    print(f"  Synthetic tasks: {stats['synthetic_tasks']}")
    print(f"  Avg complexity: {stats['avg_complexity']:.3f}")
    print(f"  Avg diversity: {stats['avg_diversity']:.3f}")
    print(f"  Data amplification: {stats['data_amplification']:.1f}x")

    # Training data
    train_data = relabeler.get_training_data()
    print(f"\nTotal training examples: {len(train_data)}")

    return relabeler


if __name__ == "__main__":
    demo()
