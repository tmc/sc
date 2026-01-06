"""
Evaluator: Compare Base vs Fine-tuned Model Performance

Metrics:
1. Validity Rate: % of generated statecharts that are valid JSON + valid schema
2. Structural Correctness: Proper state hierarchy, transitions
3. Semantic Quality: Meaningful state/event names, complete coverage
4. Generation Speed: Tokens per second

Target: +15% validity improvement over base model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import time
import re

# MLX imports
try:
    from mlx_lm import load, generate
    from mlx_lm.tuner import linear_to_lora_layers
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .lora_config import FullConfig, get_default_config


@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt: str
    output: str
    generation_time_ms: float
    tokens_generated: int

    # Validity checks
    is_valid_json: bool = False
    is_valid_schema: bool = False
    is_valid_structure: bool = False

    # Quality scores
    structural_score: float = 0.0
    semantic_score: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.is_valid_json and self.is_valid_schema and self.is_valid_structure


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    model_name: str
    is_finetuned: bool
    n_samples: int

    # Validity rates
    json_validity_rate: float
    schema_validity_rate: float
    structural_validity_rate: float
    overall_validity_rate: float

    # Quality scores
    avg_structural_score: float
    avg_semantic_score: float

    # Performance
    avg_generation_time_ms: float
    tokens_per_second: float

    # Individual results
    results: List[GenerationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_finetuned": self.is_finetuned,
            "n_samples": self.n_samples,
            "validity": {
                "json": self.json_validity_rate,
                "schema": self.schema_validity_rate,
                "structural": self.structural_validity_rate,
                "overall": self.overall_validity_rate,
            },
            "quality": {
                "structural": self.avg_structural_score,
                "semantic": self.avg_semantic_score,
            },
            "performance": {
                "avg_time_ms": self.avg_generation_time_ms,
                "tokens_per_sec": self.tokens_per_second,
            },
        }


@dataclass
class ComparisonResult:
    """Comparison between base and fine-tuned models."""
    base_result: EvaluationResult
    finetuned_result: EvaluationResult

    # Improvements
    validity_improvement: float  # Percentage points
    structural_improvement: float
    semantic_improvement: float
    speed_change: float  # % change

    # Target met
    target_validity_improvement: float = 15.0
    meets_target: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base_result.to_dict(),
            "finetuned": self.finetuned_result.to_dict(),
            "improvements": {
                "validity": self.validity_improvement,
                "structural": self.structural_improvement,
                "semantic": self.semantic_improvement,
                "speed_change": self.speed_change,
            },
            "target": self.target_validity_improvement,
            "meets_target": self.meets_target,
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "COMPARISON: Base vs Fine-tuned Model",
            "=" * 60,
            "",
            f"Base validity rate:      {self.base_result.overall_validity_rate:.1%}",
            f"Fine-tuned validity:     {self.finetuned_result.overall_validity_rate:.1%}",
            f"Improvement:             {self.validity_improvement:+.1f}pp",
            "",
            f"Target improvement:      {self.target_validity_improvement:.1f}pp",
            f"Target met:              {'YES' if self.meets_target else 'NO'}",
            "",
            f"Structural improvement:  {self.structural_improvement:+.1f}pp",
            f"Semantic improvement:    {self.semantic_improvement:+.1f}pp",
            f"Speed change:            {self.speed_change:+.1f}%",
        ]
        return "\n".join(lines)


class StatechartValidator:
    """Validate generated statecharts."""

    def validate_json(self, text: str) -> Tuple[bool, Optional[Dict]]:
        """Check if text is valid JSON."""
        # Try to extract JSON from text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return False, None

        try:
            data = json.loads(json_match.group())
            return True, data
        except json.JSONDecodeError:
            return False, None

    def validate_schema(self, data: Dict) -> bool:
        """Check if JSON follows statechart schema."""
        if not isinstance(data, dict):
            return False

        # Must have root_state
        if "root_state" not in data:
            return False

        root = data["root_state"]
        if not isinstance(root, dict):
            return False

        # Root must have children or be a basic state
        if "children" not in root and root.get("type", 1) != 1:
            return False

        return True

    def validate_structure(self, data: Dict) -> Tuple[bool, float]:
        """
        Validate structural correctness.

        Returns (is_valid, score)
        """
        score = 0.0
        checks_passed = 0
        total_checks = 5

        # Check 1: Has states
        states = self._extract_states(data.get("root_state", {}))
        if len(states) > 0:
            checks_passed += 1
            score += 0.2

        # Check 2: Has initial state
        has_initial = any(s.get("is_initial") for s in states)
        if has_initial:
            checks_passed += 1
            score += 0.2

        # Check 3: Transitions reference valid states
        transitions = data.get("transitions", [])
        state_labels = {s.get("label") for s in states}
        valid_trans = 0
        for t in transitions:
            sources = t.get("from", [])
            targets = t.get("to", [])
            if all(s in state_labels for s in sources + targets):
                valid_trans += 1

        if transitions and valid_trans == len(transitions):
            checks_passed += 1
            score += 0.2

        # Check 4: State types are valid
        valid_types = {1, 2, 3}  # BASIC, NORMAL, PARALLEL
        types_valid = all(s.get("type", 1) in valid_types for s in states)
        if types_valid:
            checks_passed += 1
            score += 0.2

        # Check 5: No orphan states (all reachable from initial)
        if has_initial and transitions:
            reachable = self._find_reachable(data)
            if len(reachable) >= len(states) * 0.8:
                checks_passed += 1
                score += 0.2

        is_valid = checks_passed >= 3
        return is_valid, score

    def validate_semantics(self, data: Dict) -> float:
        """
        Evaluate semantic quality.

        Returns score 0-1.
        """
        score = 0.0

        # Check state name quality
        states = self._extract_states(data.get("root_state", {}))
        good_names = 0
        for s in states:
            label = s.get("label", "")
            # Good names are capitalized, not too short, not generic
            if (len(label) >= 3 and
                label[0].isupper() and
                label.lower() not in {"state", "s1", "s2", "state1", "state2"}):
                good_names += 1

        if states:
            score += 0.4 * (good_names / len(states))

        # Check event name quality
        transitions = data.get("transitions", [])
        good_events = 0
        for t in transitions:
            event = t.get("event", "")
            if event and len(event) >= 2 and event.isupper():
                good_events += 1

        if transitions:
            score += 0.4 * (good_events / len(transitions))

        # Check for meaningful structure
        has_name = bool(data.get("name"))
        if has_name:
            score += 0.2

        return score

    def _extract_states(self, state: Dict, states: List = None) -> List[Dict]:
        """Extract all states from hierarchy."""
        if states is None:
            states = []

        label = state.get("label", "")
        if not label.startswith("__"):
            states.append(state)

        for child in state.get("children", []):
            self._extract_states(child, states)

        return states

    def _find_reachable(self, data: Dict) -> set:
        """Find states reachable from initial."""
        states = self._extract_states(data.get("root_state", {}))
        transitions = data.get("transitions", [])

        initial = None
        for s in states:
            if s.get("is_initial"):
                initial = s.get("label")
                break

        if not initial:
            return set()

        reachable = {initial}
        changed = True
        while changed:
            changed = False
            for t in transitions:
                sources = t.get("from", [])
                targets = t.get("to", [])
                if any(s in reachable for s in sources):
                    for tgt in targets:
                        if tgt not in reachable:
                            reachable.add(tgt)
                            changed = True

        return reachable


class ValidityEvaluator:
    """
    Evaluate model validity on statechart generation.
    """

    def __init__(self, config: Optional[FullConfig] = None):
        self.config = config or get_default_config()
        self.validator = StatechartValidator()
        self.model = None
        self.tokenizer = None

    def load_model(self, adapter_path: Optional[Path] = None):
        """Load model with optional LoRA adapter."""
        if not MLX_AVAILABLE:
            print("Warning: MLX not available")
            return

        print(f"Loading model: {self.config.model.model_name}")
        self.model, self.tokenizer = load(self.config.model.model_name)

        if adapter_path and adapter_path.exists():
            print(f"Loading adapter: {adapter_path}")
            # Apply LoRA structure
            linear_to_lora_layers(
                self.model,
                self.config.lora.rank,
                self.config.lora.target_modules,
            )
            # Load weights
            weights = mx.load(str(adapter_path))
            self.model.load_weights(list(weights.items()))

    def evaluate(
        self,
        prompts: List[str],
        adapter_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate model on given prompts.

        Args:
            prompts: List of generation prompts
            adapter_path: Optional path to LoRA adapter

        Returns:
            EvaluationResult with validity rates
        """
        is_finetuned = adapter_path is not None

        if self.model is None:
            self.load_model(adapter_path)

        results = []
        for prompt in prompts:
            result = self._generate_and_validate(prompt)
            results.append(result)

        # Aggregate metrics
        n = len(results)
        json_valid = sum(1 for r in results if r.is_valid_json)
        schema_valid = sum(1 for r in results if r.is_valid_schema)
        struct_valid = sum(1 for r in results if r.is_valid_structure)
        overall_valid = sum(1 for r in results if r.is_valid)

        total_time = sum(r.generation_time_ms for r in results)
        total_tokens = sum(r.tokens_generated for r in results)

        return EvaluationResult(
            model_name=self.config.model.model_name,
            is_finetuned=is_finetuned,
            n_samples=n,
            json_validity_rate=json_valid / n if n else 0,
            schema_validity_rate=schema_valid / n if n else 0,
            structural_validity_rate=struct_valid / n if n else 0,
            overall_validity_rate=overall_valid / n if n else 0,
            avg_structural_score=sum(r.structural_score for r in results) / n if n else 0,
            avg_semantic_score=sum(r.semantic_score for r in results) / n if n else 0,
            avg_generation_time_ms=total_time / n if n else 0,
            tokens_per_second=total_tokens / (total_time / 1000) if total_time else 0,
            results=results,
        )

    def _generate_and_validate(self, prompt: str) -> GenerationResult:
        """Generate and validate a single response."""
        t0 = time.time()

        if MLX_AVAILABLE and self.model is not None:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            output = generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=1024,
                temp=0.7,
            )
            tokens = len(self.tokenizer.encode(output))
        else:
            # Mock generation
            output = self._mock_generate(prompt)
            tokens = len(output.split())

        gen_time = (time.time() - t0) * 1000

        # Validate
        is_json, data = self.validator.validate_json(output)
        is_schema = self.validator.validate_schema(data) if data else False
        is_struct, struct_score = self.validator.validate_structure(data) if data else (False, 0)
        sem_score = self.validator.validate_semantics(data) if data else 0

        return GenerationResult(
            prompt=prompt,
            output=output,
            generation_time_ms=gen_time,
            tokens_generated=tokens,
            is_valid_json=is_json,
            is_valid_schema=is_schema,
            is_valid_structure=is_struct,
            structural_score=struct_score,
            semantic_score=sem_score,
        )

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        import random
        if random.random() < 0.7:  # 70% valid
            return json.dumps({
                "name": "MockStatechart",
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
        else:
            return "Invalid output {broken json"


def compare_models(
    prompts: List[str],
    adapter_path: Path,
    config: Optional[FullConfig] = None,
    target_improvement: float = 15.0,
) -> ComparisonResult:
    """
    Compare base model vs fine-tuned model.

    Args:
        prompts: Evaluation prompts
        adapter_path: Path to LoRA adapter
        config: Model configuration
        target_improvement: Target validity improvement in pp

    Returns:
        ComparisonResult with comparison metrics
    """
    config = config or get_default_config()

    # Evaluate base model
    print("Evaluating base model...")
    base_evaluator = ValidityEvaluator(config)
    base_result = base_evaluator.evaluate(prompts)

    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    ft_evaluator = ValidityEvaluator(config)
    ft_result = ft_evaluator.evaluate(prompts, adapter_path)

    # Calculate improvements
    validity_imp = (ft_result.overall_validity_rate - base_result.overall_validity_rate) * 100
    struct_imp = (ft_result.avg_structural_score - base_result.avg_structural_score) * 100
    sem_imp = (ft_result.avg_semantic_score - base_result.avg_semantic_score) * 100

    base_speed = base_result.tokens_per_second
    ft_speed = ft_result.tokens_per_second
    speed_change = ((ft_speed - base_speed) / base_speed * 100) if base_speed else 0

    return ComparisonResult(
        base_result=base_result,
        finetuned_result=ft_result,
        validity_improvement=validity_imp,
        structural_improvement=struct_imp,
        semantic_improvement=sem_imp,
        speed_change=speed_change,
        target_validity_improvement=target_improvement,
        meets_target=validity_imp >= target_improvement,
    )


def evaluate_model(
    prompts: List[str],
    adapter_path: Optional[Path] = None,
    config: Optional[FullConfig] = None,
) -> EvaluationResult:
    """Convenience function for single model evaluation."""
    evaluator = ValidityEvaluator(config)
    return evaluator.evaluate(prompts, adapter_path)


# Test prompts for evaluation
TEST_PROMPTS = [
    "Generate a statechart JSON for a traffic light controller with Red, Yellow, Green states",
    "Generate a statechart JSON for a door lock system with Locked and Unlocked states",
    "Generate a statechart JSON for a media player with Play, Pause, Stop functionality",
    "Generate a statechart JSON for user authentication with login and logout",
    "Generate a statechart JSON for an order processing workflow",
    "Generate a statechart JSON for a simple game with menu and playing states",
    "Generate a statechart JSON for a thermostat controller",
    "Generate a statechart JSON for an elevator system",
    "Generate a statechart JSON for a vending machine",
    "Generate a statechart JSON for a washing machine cycle",
]


def demo():
    """Demonstrate evaluation."""
    print("=" * 60)
    print("EVALUATOR: Base vs Fine-tuned Validity Comparison")
    print("=" * 60)

    # Use subset of prompts for demo
    prompts = TEST_PROMPTS[:5]
    print(f"\nEvaluating on {len(prompts)} prompts...")

    # Mock adapter path
    adapter_path = Path("/tmp/mock_adapter.safetensors")
    adapter_path.touch()

    # Compare
    result = compare_models(prompts, adapter_path, target_improvement=15.0)

    print(result.summary())

    return result


if __name__ == "__main__":
    demo()
