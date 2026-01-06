"""
Benchmark Evaluator: Compare Generated vs Gold Standard Statecharts

Metrics:
1. Structural Match: States and transitions match gold standard
2. Semantic Similarity: Names and events similar
3. Feature Correctness: Hierarchy, parallel, guards present
4. Validity: Well-formed statechart
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path
import json
import re


@dataclass
class TaskResult:
    """Result for a single benchmark task."""
    task_id: str
    category: str
    difficulty: int
    
    # Scores (0-1)
    structural_score: float = 0.0
    semantic_score: float = 0.0
    feature_score: float = 0.0
    validity_score: float = 0.0
    
    # Details
    states_matched: int = 0
    states_total: int = 0
    transitions_matched: int = 0
    transitions_total: int = 0
    
    # Metadata
    gold_states: List[str] = field(default_factory=list)
    generated_states: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        return (self.structural_score * 0.4 + 
                self.semantic_score * 0.2 + 
                self.feature_score * 0.2 + 
                self.validity_score * 0.2)
    
    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.5


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    total_tasks: int
    passed_tasks: int
    
    # By category
    category_scores: Dict[str, float]
    category_pass_rates: Dict[str, float]
    
    # By difficulty
    difficulty_scores: Dict[int, float]
    
    # Overall
    avg_structural: float
    avg_semantic: float
    avg_feature: float
    avg_validity: float
    avg_overall: float
    
    # Individual results
    task_results: List[TaskResult]
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SC GENERATION BENCHMARK RESULTS",
            "=" * 60,
            f"Total tasks: {self.total_tasks}",
            f"Passed: {self.passed_tasks} ({self.passed_tasks/self.total_tasks:.1%})",
            f"",
            f"Overall Scores:",
            f"  Structural: {self.avg_structural:.1%}",
            f"  Semantic:   {self.avg_semantic:.1%}",
            f"  Feature:    {self.avg_feature:.1%}",
            f"  Validity:   {self.avg_validity:.1%}",
            f"  Combined:   {self.avg_overall:.1%}",
            f"",
            f"By Category:",
        ]
        for cat, score in self.category_scores.items():
            pass_rate = self.category_pass_rates[cat]
            lines.append(f"  {cat}: {score:.1%} (pass: {pass_rate:.1%})")
        
        return "\n".join(lines)


class BenchmarkEvaluator:
    """Evaluate generated statecharts against gold standards."""
    
    def __init__(self, benchmark_path: Path):
        self.benchmark_path = benchmark_path
        self.tasks = []
        self.gold_standards = {}
        self._load_benchmark()
    
    def _load_benchmark(self):
        """Load benchmark tasks and gold standards."""
        # Load tasks
        tasks_path = self.benchmark_path / "benchmark_tasks.json"
        with open(tasks_path, 'r') as f:
            data = json.load(f)
            self.tasks = data["tasks"]
        
        # Load gold standards
        gold_dir = self.benchmark_path / "gold_standards"
        for task in self.tasks:
            gold_path = gold_dir / f"{task['id']}.json"
            if gold_path.exists():
                with open(gold_path, 'r') as f:
                    self.gold_standards[task['id']] = json.load(f)
    
    def evaluate(self, generations: Dict[str, str]) -> EvaluationResult:
        """
        Evaluate generated statecharts.
        
        Args:
            generations: Dict mapping task_id -> generated JSON string
            
        Returns:
            EvaluationResult with scores
        """
        results = []
        
        for task in self.tasks:
            task_id = task["id"]
            gold = self.gold_standards.get(task_id)
            generated_str = generations.get(task_id, "")
            
            result = self._evaluate_task(task, gold, generated_str)
            results.append(result)
        
        return self._aggregate_results(results)
    
    def _evaluate_task(
        self,
        task: Dict,
        gold: Dict,
        generated_str: str,
    ) -> TaskResult:
        """Evaluate a single task."""
        result = TaskResult(
            task_id=task["id"],
            category=task["category"],
            difficulty=task["difficulty"],
        )
        
        # Parse generated
        generated = self._parse_json(generated_str)
        if generated is None:
            return result  # All zeros
        
        # Validity check
        result.validity_score = self._check_validity(generated)
        
        if gold is None:
            return result
        
        # Extract states
        gold_states = self._extract_states(gold.get("root_state", {}))
        gen_states = self._extract_states(generated.get("root_state", {}))
        
        result.gold_states = list(gold_states)
        result.generated_states = list(gen_states)
        result.states_total = len(gold_states)
        
        # Structural score
        result.structural_score, result.states_matched = self._structural_match(
            gold, generated, gold_states, gen_states
        )
        
        # Semantic score
        result.semantic_score = self._semantic_similarity(gold, generated)
        
        # Feature score
        result.feature_score = self._feature_match(task["category"], generated)
        
        # Transitions
        gold_trans = gold.get("transitions", [])
        gen_trans = generated.get("transitions", [])
        result.transitions_total = len(gold_trans)
        result.transitions_matched = self._count_matching_transitions(gold_trans, gen_trans)
        
        return result
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from text."""
        if not text:
            return None
        
        # Try to extract JSON
        patterns = [
            r'```json\s*([\s\S]*?)```',
            r'```\s*([\s\S]*?)```',
            r'(\{[\s\S]*\})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        
        try:
            return json.loads(text)
        except:
            return None
    
    def _extract_states(self, state: Dict, states: Set[str] = None) -> Set[str]:
        """Extract state labels from hierarchy."""
        if states is None:
            states = set()
        
        label = state.get("label", "")
        if label and not label.startswith("__"):
            states.add(label)
        
        for child in state.get("children", []):
            self._extract_states(child, states)
        
        return states
    
    def _check_validity(self, sc: Dict) -> float:
        """Check statechart validity."""
        score = 0.0
        
        # Has root_state
        if "root_state" in sc:
            score += 0.25
        
        # Root has children or is basic
        root = sc.get("root_state", {})
        if root.get("children") or root.get("type") == 1:
            score += 0.25
        
        # Has transitions
        if sc.get("transitions"):
            score += 0.25
        
        # Has initial state
        if self._has_initial(root):
            score += 0.25
        
        return score
    
    def _has_initial(self, state: Dict) -> bool:
        """Check if there's an initial state."""
        if state.get("is_initial"):
            return True
        for child in state.get("children", []):
            if self._has_initial(child):
                return True
        return False
    
    def _structural_match(
        self,
        gold: Dict,
        generated: Dict,
        gold_states: Set[str],
        gen_states: Set[str],
    ) -> Tuple[float, int]:
        """Calculate structural match score."""
        if not gold_states:
            return 0.0, 0
        
        # State overlap
        matched = gold_states & gen_states
        state_score = len(matched) / len(gold_states)
        
        # Transition overlap (simplified)
        gold_trans = gold.get("transitions", [])
        gen_trans = generated.get("transitions", [])
        
        trans_score = 0.0
        if gold_trans:
            matching = self._count_matching_transitions(gold_trans, gen_trans)
            trans_score = matching / len(gold_trans)
        
        return (state_score * 0.6 + trans_score * 0.4), len(matched)
    
    def _count_matching_transitions(self, gold: List, gen: List) -> int:
        """Count matching transitions."""
        matched = 0
        gen_set = set()
        
        for t in gen:
            key = (tuple(t.get("from", [])), tuple(t.get("to", [])), t.get("event", ""))
            gen_set.add(key)
        
        for t in gold:
            key = (tuple(t.get("from", [])), tuple(t.get("to", [])), t.get("event", ""))
            if key in gen_set:
                matched += 1
        
        return matched
    
    def _semantic_similarity(self, gold: Dict, generated: Dict) -> float:
        """Calculate semantic similarity."""
        score = 0.0
        
        # Name similarity
        gold_name = gold.get("name", "").lower()
        gen_name = generated.get("name", "").lower()
        if gold_name and gen_name:
            if gold_name == gen_name:
                score += 0.5
            elif any(w in gen_name for w in gold_name.split()):
                score += 0.25
        
        # Event similarity
        gold_events = {t.get("event", "") for t in gold.get("transitions", [])}
        gen_events = {t.get("event", "") for t in generated.get("transitions", [])}
        
        if gold_events:
            overlap = len(gold_events & gen_events) / len(gold_events)
            score += 0.5 * overlap
        
        return score
    
    def _feature_match(self, category: str, generated: Dict) -> float:
        """Check if required features are present."""
        root = generated.get("root_state", {})
        
        if category == "simple":
            # Just needs states and transitions
            return 1.0 if self._has_states(root) else 0.0
        
        elif category == "hierarchical":
            # Needs nested states
            return 1.0 if self._has_hierarchy(root) else 0.0
        
        elif category == "parallel":
            # Needs parallel type
            return 1.0 if self._has_parallel(root) else 0.0
        
        elif category == "guards":
            # Needs guards on transitions
            return 1.0 if self._has_guards(generated) else 0.0
        
        elif category == "complex":
            # Needs multiple features
            score = 0.0
            if self._has_hierarchy(root):
                score += 0.33
            if self._has_parallel(root):
                score += 0.33
            if self._has_guards(generated):
                score += 0.34
            return score
        
        return 0.0
    
    def _has_states(self, state: Dict) -> bool:
        return bool(state.get("children") or state.get("label"))
    
    def _has_hierarchy(self, state: Dict, depth: int = 0) -> bool:
        """Check for nested composite states."""
        for child in state.get("children", []):
            if child.get("children"):
                return True
            if self._has_hierarchy(child, depth + 1):
                return True
        return False
    
    def _has_parallel(self, state: Dict) -> bool:
        """Check for parallel type."""
        if state.get("type") == 3:
            return True
        for child in state.get("children", []):
            if self._has_parallel(child):
                return True
        return False
    
    def _has_guards(self, sc: Dict) -> bool:
        """Check for guarded transitions."""
        for t in sc.get("transitions", []):
            if t.get("guard"):
                return True
        return False
    
    def _aggregate_results(self, results: List[TaskResult]) -> EvaluationResult:
        """Aggregate individual results."""
        if not results:
            return EvaluationResult(0, 0, {}, {}, {}, 0, 0, 0, 0, 0, [])
        
        # Category aggregation
        category_scores = {}
        category_pass = {}
        categories = set(r.category for r in results)
        
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            category_scores[cat] = sum(r.overall_score for r in cat_results) / len(cat_results)
            category_pass[cat] = sum(1 for r in cat_results if r.passed) / len(cat_results)
        
        # Difficulty aggregation
        difficulty_scores = {}
        difficulties = set(r.difficulty for r in results)
        for d in difficulties:
            d_results = [r for r in results if r.difficulty == d]
            difficulty_scores[d] = sum(r.overall_score for r in d_results) / len(d_results)
        
        return EvaluationResult(
            total_tasks=len(results),
            passed_tasks=sum(1 for r in results if r.passed),
            category_scores=category_scores,
            category_pass_rates=category_pass,
            difficulty_scores=difficulty_scores,
            avg_structural=sum(r.structural_score for r in results) / len(results),
            avg_semantic=sum(r.semantic_score for r in results) / len(results),
            avg_feature=sum(r.feature_score for r in results) / len(results),
            avg_validity=sum(r.validity_score for r in results) / len(results),
            avg_overall=sum(r.overall_score for r in results) / len(results),
            task_results=results,
        )


def load_benchmark(path: Path = None) -> BenchmarkEvaluator:
    """Load benchmark from default or specified path."""
    if path is None:
        path = Path(__file__).parent
    return BenchmarkEvaluator(path)


def evaluate_generation(
    generations: Dict[str, str],
    benchmark_path: Path = None,
) -> EvaluationResult:
    """Evaluate generated statecharts against benchmark."""
    evaluator = load_benchmark(benchmark_path)
    return evaluator.evaluate(generations)


def run_benchmark(
    generator_fn,
    benchmark_path: Path = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run benchmark with a generator function.
    
    Args:
        generator_fn: Function(prompt) -> generated_json_string
        benchmark_path: Path to benchmark
        verbose: Print progress
    """
    evaluator = load_benchmark(benchmark_path)
    
    generations = {}
    for i, task in enumerate(evaluator.tasks):
        if verbose and i % 20 == 0:
            print(f"Generating {i+1}/{len(evaluator.tasks)}...")
        
        generated = generator_fn(task["prompt"])
        generations[task["id"]] = generated
    
    result = evaluator.evaluate(generations)
    
    if verbose:
        print(result.summary())
    
    return result


def validate_gold_standards(benchmark_path: Path = None) -> Tuple[int, int]:
    """Validate all gold standards."""
    evaluator = load_benchmark(benchmark_path)
    
    valid = 0
    total = len(evaluator.gold_standards)
    
    for task_id, gold in evaluator.gold_standards.items():
        score = evaluator._check_validity(gold)
        if score >= 0.75:
            valid += 1
    
    return valid, total


def demo():
    """Demonstrate evaluator."""
    print("=" * 60)
    print("SC GENERATION BENCHMARK EVALUATOR")
    print("=" * 60)
    
    # Validate gold standards
    valid, total = validate_gold_standards()
    print(f"\nGold standards validated: {valid}/{total} ({valid/total:.0%})")
    
    # Demo with mock generator
    def mock_generator(prompt: str) -> str:
        # Return a simple valid statechart
        return json.dumps({
            "name": "Generated",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "State1", "type": 1, "is_initial": True},
                    {"label": "State2", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["State1"], "to": ["State2"], "event": "EVENT"}
            ]
        })
    
    print("\nRunning benchmark with mock generator...")
    result = run_benchmark(mock_generator, verbose=False)
    print(result.summary())
    
    return result


if __name__ == "__main__":
    demo()
