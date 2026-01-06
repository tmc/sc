"""
Benchmark Runner: Run 100-task SC generation benchmark.

Runs benchmark tasks through a generator and evaluates results.
Supports multiple generator backends (mlux, mlx_lm, API).
"""

import json
import time
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .task_generator import (
    BenchmarkTask,
    TaskCategory,
    Difficulty,
    generate_all_tasks,
    load_tasks_from_file,
)
from .evaluator import (
    StatechartEvaluator,
    EvaluationResult,
    AggregateMetrics,
    compute_aggregate_metrics,
)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SC_TEMPLATE = """Generate a statechart JSON for: {prompt}

Use this exact format:
```json
{{
  "root_state": {{
    "label": "__root__",
    "type": 2,
    "children": [
      {{"label": "State1", "type": 1, "is_initial": true}},
      {{"label": "State2", "type": 1}}
    ]
  }},
  "transitions": [
    {{"from": ["State1"], "to": ["State2"], "event": "EVENT"}}
  ]
}}
```

For hierarchical states, nest children:
```json
{{"label": "Parent", "type": 2, "children": [{{"label": "Child", "type": 1, "is_initial": true}}]}}
```

For parallel states, use type 3:
```json
{{"label": "Parallel", "type": 3, "children": [...]}}
```

For guards, add guard field:
```json
{{"from": ["A"], "to": ["B"], "event": "E", "guard": {{"condition": "[x > 0]"}}}}
```

JSON:
```json
"""


# =============================================================================
# GENERATOR INTERFACE
# =============================================================================

class GeneratorBackend(ABC):
    """Abstract generator backend."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate statechart JSON from prompt."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass


class MockGenerator(GeneratorBackend):
    """Mock generator for testing."""

    def __init__(self, valid_rate: float = 0.8):
        self.valid_rate = valid_rate
        import random
        self._random = random

    def generate(self, prompt: str) -> str:
        if self._random.random() < self.valid_rate:
            return '''```json
{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "State1", "type": 1, "is_initial": true},
      {"label": "State2", "type": 1}
    ]
  },
  "transitions": [
    {"from": ["State1"], "to": ["State2"], "event": "EVENT"}
  ]
}
```'''
        else:
            return "Invalid output"

    def name(self) -> str:
        return "mock"


class MLXLMGenerator(GeneratorBackend):
    """MLX-LM based generator."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
            self._model, self._tokenizer = load(self.model_name)
            self._generate_fn = generate
            self._make_sampler = make_sampler
        except ImportError:
            raise RuntimeError("mlx_lm not available")

    def generate(self, prompt: str) -> str:
        self._load_model()

        full_prompt = SC_TEMPLATE.format(prompt=prompt)
        sampler = self._make_sampler(temp=self.temperature)

        output = self._generate_fn(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
        )

        return output

    def name(self) -> str:
        return f"mlx_lm:{self.model_name}"


class MluxGenerator(GeneratorBackend):
    """Mlux HookedModel generator with optional steering."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.3,
        hooks: List = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.hooks = hooks or []
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from mlux import HookedModel
            self._model = HookedModel.from_pretrained(self.model_name)
        except ImportError:
            raise RuntimeError("mlux not available")

    def generate(self, prompt: str) -> str:
        self._load_model()

        full_prompt = SC_TEMPLATE.format(prompt=prompt)

        if self.hooks:
            output = self._model.generate_with_steering(
                full_prompt,
                max_tokens=self.max_tokens,
                hooks=self.hooks,
            )
        else:
            output = self._model.generate(
                full_prompt,
                max_tokens=self.max_tokens,
            )

        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')

        return output

    def name(self) -> str:
        steering = "_steered" if self.hooks else ""
        return f"mlux{steering}:{self.model_name}"


# =============================================================================
# BENCHMARK RESULT
# =============================================================================

@dataclass
class BenchmarkRun:
    """Result of a single benchmark run."""
    run_id: str
    generator_name: str
    timestamp: str
    config: Dict[str, Any]
    results: List[EvaluationResult]
    aggregate: AggregateMetrics
    total_time: float
    avg_generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generator_name": self.generator_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
            "aggregate": self.aggregate.to_dict(),
            "total_time": self.total_time,
            "avg_generation_time": self.avg_generation_time,
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class SCGenerationBenchmark:
    """
    100-Task Statechart Generation Benchmark.

    Runs tasks through generator, evaluates outputs, aggregates results.
    """

    def __init__(
        self,
        generator: GeneratorBackend = None,
        evaluator: StatechartEvaluator = None,
    ):
        self.generator = generator or MockGenerator()
        self.evaluator = evaluator or StatechartEvaluator()
        self.tasks: List[BenchmarkTask] = []
        self.results: List[EvaluationResult] = []

    def load_tasks(self, filepath: Path = None) -> List[BenchmarkTask]:
        """Load benchmark tasks."""
        if filepath and filepath.exists():
            self.tasks = load_tasks_from_file(filepath)
        else:
            self.tasks = generate_all_tasks()
        return self.tasks

    def run_single(
        self,
        task: BenchmarkTask,
        verbose: bool = True,
    ) -> EvaluationResult:
        """Run a single benchmark task."""
        if verbose:
            print(f"  [{task.id}] {task.prompt[:40]}...", end=" ", flush=True)

        start_time = time.time()
        generated = self.generator.generate(task.prompt)
        gen_time = time.time() - start_time

        result = self.evaluator.evaluate(task, generated)

        if verbose:
            status = "PASS" if result.validity_score >= 1.0 else "FAIL"
            print(f"{status} ({result.overall_score:.1%}, {gen_time:.1f}s)")

        return result

    def run_category(
        self,
        category: TaskCategory,
        verbose: bool = True,
    ) -> List[EvaluationResult]:
        """Run all tasks in a category."""
        if not self.tasks:
            self.load_tasks()

        cat_tasks = [t for t in self.tasks if t.category == category]

        if verbose:
            print(f"\n{'='*60}")
            print(f"CATEGORY: {category.value} ({len(cat_tasks)} tasks)")
            print(f"{'='*60}")

        results = []
        for task in cat_tasks:
            result = self.run_single(task, verbose)
            results.append(result)

        return results

    def run_full(
        self,
        verbose: bool = True,
        save_results: bool = True,
    ) -> BenchmarkRun:
        """Run full 100-task benchmark."""
        if not self.tasks:
            self.load_tasks()

        if verbose:
            print("=" * 70)
            print("SC GENERATION BENCHMARK (100 Tasks)")
            print("=" * 70)
            print(f"Generator: {self.generator.name()}")
            print(f"Tasks: {len(self.tasks)}")

        start_time = time.time()
        self.results = []

        for category in TaskCategory:
            cat_results = self.run_category(category, verbose)
            self.results.extend(cat_results)

        total_time = time.time() - start_time

        # Compute aggregate
        aggregate = compute_aggregate_metrics(self.results, self.tasks)

        # Create run
        from datetime import datetime
        run = BenchmarkRun(
            run_id=f"run_{int(time.time())}",
            generator_name=self.generator.name(),
            timestamp=datetime.now().isoformat(),
            config={
                "num_tasks": len(self.tasks),
                "categories": [c.value for c in TaskCategory],
            },
            results=self.results,
            aggregate=aggregate,
            total_time=total_time,
            avg_generation_time=total_time / max(len(self.tasks), 1),
        )

        if verbose:
            self._print_summary(run)

        if save_results:
            self._save_results(run)

        return run

    def run_quick(
        self,
        tasks_per_category: int = 2,
        verbose: bool = True,
    ) -> BenchmarkRun:
        """Run quick benchmark with fewer tasks per category."""
        if not self.tasks:
            self.load_tasks()

        if verbose:
            print("=" * 60)
            print(f"QUICK SC BENCHMARK ({tasks_per_category}/category)")
            print("=" * 60)
            print(f"Generator: {self.generator.name()}")

        start_time = time.time()
        self.results = []

        for category in TaskCategory:
            cat_tasks = [t for t in self.tasks if t.category == category][:tasks_per_category]

            if verbose:
                print(f"\n{category.value}:")

            for task in cat_tasks:
                result = self.run_single(task, verbose)
                self.results.append(result)

        total_time = time.time() - start_time

        # Get subset of tasks that were run
        run_task_ids = {r.task_id for r in self.results}
        run_tasks = [t for t in self.tasks if t.id in run_task_ids]

        aggregate = compute_aggregate_metrics(self.results, run_tasks)

        from datetime import datetime
        run = BenchmarkRun(
            run_id=f"quick_{int(time.time())}",
            generator_name=self.generator.name(),
            timestamp=datetime.now().isoformat(),
            config={
                "num_tasks": len(self.results),
                "tasks_per_category": tasks_per_category,
                "mode": "quick",
            },
            results=self.results,
            aggregate=aggregate,
            total_time=total_time,
            avg_generation_time=total_time / max(len(self.results), 1),
        )

        if verbose:
            self._print_summary(run)

        return run

    def _print_summary(self, run: BenchmarkRun):
        """Print benchmark summary."""
        agg = run.aggregate

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nOverall Results:")
        print(f"  Tasks: {agg.total_tasks}")
        print(f"  Valid: {agg.valid_count}/{agg.total_tasks} ({agg.validity_rate:.1%})")
        print(f"  Avg Validity: {agg.avg_validity:.1%}")
        print(f"  Avg Completeness: {agg.avg_completeness:.1%}")
        print(f"  Avg Correctness: {agg.avg_correctness:.1%}")
        print(f"  Avg Overall: {agg.avg_overall:.1%}")
        print(f"  Total Time: {run.total_time:.1f}s")
        print(f"  Avg Gen Time: {run.avg_generation_time:.2f}s/task")

        print(f"\nBy Category:")
        print(f"  {'Category':<20} {'Valid':>8} {'Overall':>10}")
        print(f"  {'-'*40}")
        for cat, metrics in agg.by_category.items():
            valid_pct = metrics.get('avg_validity', 0) * 100
            overall_pct = metrics.get('avg_overall', 0) * 100
            print(f"  {cat:<20} {valid_pct:>7.1f}% {overall_pct:>9.1f}%")

        print(f"\nBy Difficulty:")
        print(f"  {'Difficulty':<12} {'Count':>8} {'Valid':>8} {'Overall':>10}")
        print(f"  {'-'*40}")
        for diff, metrics in agg.by_difficulty.items():
            count = metrics.get('count', 0)
            valid_pct = metrics.get('avg_validity', 0) * 100
            overall_pct = metrics.get('avg_overall', 0) * 100
            print(f"  {diff:<12} {count:>8} {valid_pct:>7.1f}% {overall_pct:>9.1f}%")

        print("=" * 70)

    def _save_results(self, run: BenchmarkRun):
        """Save results to file."""
        output_dir = Path(__file__).parent / "runs"
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / f"{run.run_id}.json"
        with open(filepath, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)

        print(f"\nResults saved to: {filepath}")


# =============================================================================
# COMPARISON
# =============================================================================

def compare_generators(
    generators: List[GeneratorBackend],
    tasks_per_category: int = 2,
    verbose: bool = True,
) -> Dict[str, BenchmarkRun]:
    """Compare multiple generators on the same tasks."""
    results = {}

    for generator in generators:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# GENERATOR: {generator.name()}")
            print(f"{'#'*60}")

        benchmark = SCGenerationBenchmark(generator=generator)
        run = benchmark.run_quick(tasks_per_category=tasks_per_category, verbose=verbose)
        results[generator.name()] = run

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Generator':<30} {'Valid':>10} {'Overall':>10} {'Time':>10}")
        print("-" * 70)

        for name, run in results.items():
            valid_pct = run.aggregate.validity_rate * 100
            overall_pct = run.aggregate.avg_overall * 100
            print(f"{name:<30} {valid_pct:>9.1f}% {overall_pct:>9.1f}% {run.total_time:>9.1f}s")

    return results


# =============================================================================
# MAIN
# =============================================================================

def test_benchmark():
    """Test benchmark with mock generator."""
    print("=" * 60)
    print("SC GENERATION BENCHMARK TEST")
    print("=" * 60)

    # Mock generator
    generator = MockGenerator(valid_rate=0.8)
    benchmark = SCGenerationBenchmark(generator=generator)

    # Load tasks
    tasks = benchmark.load_tasks()
    print(f"Loaded {len(tasks)} tasks")

    # Run quick benchmark
    run = benchmark.run_quick(tasks_per_category=1, verbose=True)

    print(f"\nRun ID: {run.run_id}")
    print(f"Generator: {run.generator_name}")
    print(f"Validity Rate: {run.aggregate.validity_rate:.1%}")
    print(f"Overall Score: {run.aggregate.avg_overall:.1%}")

    print("\n[PASS] Benchmark test complete")
    return True


def run_mlx_benchmark(tasks_per_category: int = 5):
    """Run benchmark with MLX-LM generator."""
    print("=" * 60)
    print("SC GENERATION BENCHMARK (MLX-LM)")
    print("=" * 60)

    try:
        generator = MLXLMGenerator(
            model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            max_tokens=512,
            temperature=0.3,
        )
    except Exception as e:
        print(f"Failed to create generator: {e}")
        return None

    benchmark = SCGenerationBenchmark(generator=generator)
    run = benchmark.run_quick(tasks_per_category=tasks_per_category, verbose=True)

    return run


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--mlx":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        run_mlx_benchmark(tasks_per_category=n)
    else:
        test_benchmark()
