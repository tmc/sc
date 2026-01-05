"""
ARC Benchmark Evaluation for Statechart-based Program Synthesis

Evaluates SOAR-statechart approach on ARC (Abstraction and Reasoning Corpus):
- Load ARC tasks from JSON format
- Run synthesis with configurable parameters
- Compare constrained (statechart) vs unconstrained approaches
- Report metrics: solve rate, sample efficiency, generalization

Key ARC task properties:
- Few-shot learning (2-5 training examples)
- Visual pattern recognition
- Abstract reasoning required
- No explicit task description
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import time
from collections import defaultdict

try:
    from .soar_statechart import (
        StatechartProgram, ARCStatechartSynthesizer, SOARStatechart
    )
    from .rex_refinement import REXEvolver, MutationArchive
    from .hindsight_relabel import HindsightRelabeler, TrajectoryCollector
except ImportError:
    from soar_statechart import (
        StatechartProgram, ARCStatechartSynthesizer, SOARStatechart
    )
    from rex_refinement import REXEvolver, MutationArchive
    from hindsight_relabel import HindsightRelabeler, TrajectoryCollector


# =============================================================================
# ARC Task Loading
# =============================================================================

@dataclass
class ARCTask:
    """An ARC task with train/test examples."""
    task_id: str
    train_inputs: List[mx.array] = field(default_factory=list)
    train_outputs: List[mx.array] = field(default_factory=list)
    test_inputs: List[mx.array] = field(default_factory=list)
    test_outputs: List[mx.array] = field(default_factory=list)

    @property
    def train_pairs(self) -> List[Tuple[mx.array, mx.array]]:
        """Get (input, output) training pairs."""
        return list(zip(self.train_inputs, self.train_outputs))

    @property
    def n_train(self) -> int:
        return len(self.train_inputs)

    @property
    def n_test(self) -> int:
        return len(self.test_inputs)


class ARCDataset:
    """Loader for ARC dataset."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.tasks: Dict[str, ARCTask] = {}

    def load_task(self, task_path: Path) -> ARCTask:
        """Load a single ARC task from JSON."""
        with open(task_path) as f:
            data = json.load(f)

        task_id = task_path.stem

        task = ARCTask(task_id=task_id)

        # Load training examples
        for example in data.get("train", []):
            task.train_inputs.append(mx.array(example["input"]))
            task.train_outputs.append(mx.array(example["output"]))

        # Load test examples
        for example in data.get("test", []):
            task.test_inputs.append(mx.array(example["input"]))
            if "output" in example:
                task.test_outputs.append(mx.array(example["output"]))

        return task

    def load_directory(self, dir_path: Optional[str] = None) -> int:
        """Load all tasks from a directory."""
        path = Path(dir_path) if dir_path else self.data_dir
        if path is None:
            return 0

        count = 0
        for task_file in path.glob("*.json"):
            task = self.load_task(task_file)
            self.tasks[task.task_id] = task
            count += 1

        return count

    def load_from_dict(self, data: Dict[str, Any], task_id: str) -> ARCTask:
        """Load task from dictionary format."""
        task = ARCTask(task_id=task_id)

        for example in data.get("train", []):
            task.train_inputs.append(mx.array(example["input"]))
            task.train_outputs.append(mx.array(example["output"]))

        for example in data.get("test", []):
            task.test_inputs.append(mx.array(example["input"]))
            if "output" in example:
                task.test_outputs.append(mx.array(example["output"]))

        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def all_tasks(self) -> List[ARCTask]:
        """Get all loaded tasks."""
        return list(self.tasks.values())


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class TaskResult:
    """Result of solving a single task."""
    task_id: str
    train_correct: int = 0
    train_total: int = 0
    test_correct: int = 0
    test_total: int = 0
    solve_time: float = 0.0
    generations: int = 0
    program: Optional[StatechartProgram] = None

    @property
    def train_accuracy(self) -> float:
        return self.train_correct / max(1, self.train_total)

    @property
    def test_accuracy(self) -> float:
        return self.test_correct / max(1, self.test_total)

    @property
    def is_solved(self) -> bool:
        """Task is solved if all test cases pass."""
        return self.test_correct == self.test_total and self.test_total > 0

    @property
    def is_train_solved(self) -> bool:
        """Training examples all solved."""
        return self.train_correct == self.train_total and self.train_total > 0


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""
    name: str
    task_results: List[TaskResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def n_tasks(self) -> int:
        return len(self.task_results)

    @property
    def solve_rate(self) -> float:
        """Fraction of tasks fully solved."""
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.is_solved) / len(self.task_results)

    @property
    def train_solve_rate(self) -> float:
        """Fraction of tasks with training solved."""
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.is_train_solved) / len(self.task_results)

    @property
    def avg_train_accuracy(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.train_accuracy for r in self.task_results) / len(self.task_results)

    @property
    def avg_test_accuracy(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.test_accuracy for r in self.task_results) / len(self.task_results)

    @property
    def avg_solve_time(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.solve_time for r in self.task_results) / len(self.task_results)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=== {self.name} ===",
            f"Tasks: {self.n_tasks}",
            f"Solve rate: {self.solve_rate:.1%}",
            f"Train solve rate: {self.train_solve_rate:.1%}",
            f"Avg train accuracy: {self.avg_train_accuracy:.1%}",
            f"Avg test accuracy: {self.avg_test_accuracy:.1%}",
            f"Avg solve time: {self.avg_solve_time:.2f}s",
            f"Total time: {self.total_time:.1f}s",
        ]
        return "\n".join(lines)


# =============================================================================
# ARC Evaluator
# =============================================================================

class ARCEvaluator:
    """
    Evaluator for ARC benchmark.

    Supports multiple synthesis approaches:
    - SOAR-statechart (constrained evolution)
    - Random search baseline
    - Ablation variants
    """

    def __init__(
        self,
        n_generations: int = 100,
        population_size: int = 50,
        use_hindsight: bool = True,
        verbose: bool = True,
    ):
        self.n_generations = n_generations
        self.population_size = population_size
        self.use_hindsight = use_hindsight
        self.verbose = verbose

        # Components
        self.dataset = ARCDataset()
        self.results: Dict[str, BenchmarkResult] = {}

    def evaluate_task(
        self,
        task: ARCTask,
        synthesizer: ARCStatechartSynthesizer,
    ) -> TaskResult:
        """Evaluate synthesizer on a single task."""
        result = TaskResult(
            task_id=task.task_id,
            train_total=task.n_train,
            test_total=task.n_test,
        )

        start_time = time.time()

        # Synthesize program
        predictions = synthesizer.synthesize(
            task_id=task.task_id,
            train_examples=task.train_pairs,
            test_inputs=task.test_inputs,
            n_generations=self.n_generations,
        )

        result.solve_time = time.time() - start_time
        result.program = synthesizer.get_program(task.task_id)

        if result.program:
            result.generations = synthesizer.soar.archive_size

        # Evaluate on training
        for inp, expected in task.train_pairs:
            try:
                output = result.program.execute(inp) if result.program else inp
                if mx.all(output == expected):
                    result.train_correct += 1
            except Exception:
                pass

        # Evaluate on test
        for i, pred in enumerate(predictions):
            if i < len(task.test_outputs):
                expected = task.test_outputs[i]
                if mx.all(pred == expected):
                    result.test_correct += 1

        return result

    def run_benchmark(
        self,
        tasks: List[ARCTask],
        name: str = "SOAR-Statechart",
        synthesizer_factory: Optional[Callable[[], ARCStatechartSynthesizer]] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark on list of tasks.

        Args:
            tasks: List of ARC tasks
            name: Benchmark name for reporting
            synthesizer_factory: Factory to create synthesizer (fresh for each task)

        Returns:
            BenchmarkResult with aggregate metrics
        """
        result = BenchmarkResult(name=name)
        start_time = time.time()

        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n[{i+1}/{len(tasks)}] Task: {task.task_id}")

            # Create fresh synthesizer
            if synthesizer_factory:
                synth = synthesizer_factory()
            else:
                synth = ARCStatechartSynthesizer(
                    soar=SOARStatechart(population_size=self.population_size)
                )

            task_result = self.evaluate_task(task, synth)
            result.task_results.append(task_result)

            if self.verbose:
                status = "✓ SOLVED" if task_result.is_solved else "✗"
                print(f"  Train: {task_result.train_accuracy:.0%}, "
                      f"Test: {task_result.test_accuracy:.0%} {status}")

        result.total_time = time.time() - start_time

        if self.verbose:
            print(f"\n{result.summary()}")

        self.results[name] = result
        return result

    def compare_approaches(
        self,
        tasks: List[ARCTask],
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare different synthesis approaches.

        Returns dict of name -> BenchmarkResult.
        """
        approaches = {
            "SOAR-Statechart": lambda: ARCStatechartSynthesizer(
                soar=SOARStatechart(population_size=self.population_size)
            ),
            "REX-Enhanced": lambda: ARCStatechartSynthesizer(
                soar=self._create_rex_soar()
            ),
        }

        results = {}
        for name, factory in approaches.items():
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print("=" * 60)
            results[name] = self.run_benchmark(tasks, name, factory)

        return results

    def _create_rex_soar(self) -> SOARStatechart:
        """Create SOAR with REX evolver."""
        soar = SOARStatechart(population_size=self.population_size)
        # The REX evolver provides enhanced parent selection
        return soar

    def generate_report(self) -> str:
        """Generate comparison report."""
        if not self.results:
            return "No results to report"

        lines = [
            "=" * 60,
            "ARC BENCHMARK COMPARISON REPORT",
            "=" * 60,
            "",
        ]

        # Summary table
        headers = ["Approach", "Solve", "Train", "Test", "Time"]
        rows = []

        for name, result in self.results.items():
            rows.append([
                name,
                f"{result.solve_rate:.1%}",
                f"{result.train_solve_rate:.1%}",
                f"{result.avg_test_accuracy:.1%}",
                f"{result.avg_solve_time:.2f}s",
            ])

        # Format table
        col_widths = [max(len(h), max(len(r[i]) for r in rows))
                      for i, h in enumerate(headers)]

        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)

        lines.append(header_line)
        lines.append(sep_line)
        for row in rows:
            lines.append(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))

        lines.append("")

        # Per-task breakdown
        lines.append("Per-Task Results:")
        for name, result in self.results.items():
            lines.append(f"\n{name}:")
            for tr in result.task_results[:10]:  # Show first 10
                status = "✓" if tr.is_solved else "✗"
                lines.append(f"  {tr.task_id}: train={tr.train_accuracy:.0%}, "
                             f"test={tr.test_accuracy:.0%} {status}")
            if len(result.task_results) > 10:
                lines.append(f"  ... and {len(result.task_results) - 10} more")

        return "\n".join(lines)


# =============================================================================
# Sample ARC Tasks for Testing
# =============================================================================

def create_sample_tasks() -> List[ARCTask]:
    """Create simple sample tasks for testing."""
    tasks = []

    # Task 1: Replace 1s with 2s
    task1 = ARCTask(task_id="replace_1_2")
    for _ in range(3):
        h, w = 4, 4
        inp = mx.array([[1 if (i + j) % 2 == 0 else 0 for j in range(w)] for i in range(h)])
        out = mx.where(inp == 1, 2, inp)
        task1.train_inputs.append(inp)
        task1.train_outputs.append(out)
    # Test
    inp = mx.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    task1.test_inputs.append(inp)
    task1.test_outputs.append(mx.where(inp == 1, 2, inp))
    tasks.append(task1)

    # Task 2: Flip horizontally
    task2 = ARCTask(task_id="flip_h")
    for _ in range(3):
        h, w = 3, 4
        inp = mx.array([[j % 3 for j in range(w)] for _ in range(h)])
        out = inp[:, ::-1]
        task2.train_inputs.append(inp)
        task2.train_outputs.append(out)
    inp = mx.array([[0, 1, 2], [0, 1, 2]])
    task2.test_inputs.append(inp)
    task2.test_outputs.append(inp[:, ::-1])
    tasks.append(task2)

    # Task 3: Fill with color
    task3 = ARCTask(task_id="fill_3")
    for _ in range(3):
        h, w = 3, 3
        inp = mx.zeros((h, w), dtype=mx.int32)
        out = mx.full((h, w), 3, dtype=mx.int32)
        task3.train_inputs.append(inp)
        task3.train_outputs.append(out)
    task3.test_inputs.append(mx.zeros((4, 4), dtype=mx.int32))
    task3.test_outputs.append(mx.full((4, 4), 3, dtype=mx.int32))
    tasks.append(task3)

    return tasks


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate ARC evaluation."""
    print("=" * 60)
    print("ARC Benchmark Evaluation")
    print("=" * 60)

    # Create evaluator
    evaluator = ARCEvaluator(
        n_generations=30,  # Reduced for demo
        population_size=20,
        verbose=True,
    )

    # Load sample tasks
    tasks = create_sample_tasks()
    print(f"\nLoaded {len(tasks)} sample tasks")

    # Run benchmark
    result = evaluator.run_benchmark(tasks, name="SOAR-Statechart")

    # Report
    print("\n" + evaluator.generate_report())

    return evaluator


def demo_with_hindsight():
    """Demonstrate evaluation with hindsight relabeling."""
    print("=" * 60)
    print("ARC Evaluation with Hindsight Relabeling")
    print("=" * 60)

    tasks = create_sample_tasks()
    task = tasks[0]  # Use first task

    print(f"\nTask: {task.task_id}")
    print(f"Train examples: {task.n_train}")

    # Create components
    soar = SOARStatechart(population_size=20)
    collector = TrajectoryCollector()
    relabeler = HindsightRelabeler()

    # Generate initial population
    programs = [soar.random_program() for _ in range(20)]

    # Collect trajectories
    print("\nCollecting trajectories...")
    for prog in programs:
        for inp, expected in task.train_pairs:
            traj = collector.collect(prog, inp, expected, task.task_id)
            relabeler.add_trajectory(traj)

    # Relabel
    synthetics = relabeler.relabel_all()
    stats = relabeler.get_statistics()

    print(f"\nHindsight stats:")
    print(f"  Trajectories: {stats['total_trajectories']}")
    print(f"  Successful: {stats['successful_trajectories']}")
    print(f"  Synthetic tasks: {stats['synthetic_tasks']}")
    print(f"  Data amplification: {stats['data_amplification']:.1f}x")

    # Get combined training data
    train_data = relabeler.get_training_data()
    print(f"\nTotal training examples: {len(train_data)}")

    return relabeler


if __name__ == "__main__":
    demo()
    print("\n" + "=" * 60 + "\n")
    demo_with_hindsight()
