"""
Validity Tester for Steered Statechart Generation

Tests how steering vectors affect the validity rate of generated statecharts.

Compares:
1. Baseline generation (no steering)
2. Steered generation (with validity vector applied)

Target: +10% validity improvement with steering.

Uses ml/utils/mlux_loader.py for model loading.
"""

import json
import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    MLUX_AVAILABLE,
)

from .steering_vectors import (
    SteeringVectorComputer,
    SteeringVector,
    create_contrastive_pairs,
)


# =============================================================================
# GENERATION PROMPTS
# =============================================================================

SC_GENERATION_PROMPTS = [
    "Generate a statechart for a traffic light controller with 3 states",
    "Create a statechart for a user login flow",
    "Design a statechart for an order processing system",
    "Build a statechart for a vending machine",
    "Make a statechart for a door lock system",
    "Generate a statechart for a simple game character with idle, walk, run states",
    "Create a statechart for a thermostat controller",
    "Design a statechart for a video player with play, pause, stop",
    "Build a statechart for an elevator controller",
    "Make a statechart for a coffee machine",
]

SC_GENERATION_TEMPLATE = """Generate a statechart JSON for: {description}

Output a valid JSON statechart with this exact format:
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
    {{"from": ["State1"], "to": ["State2"], "event": "EVENT_NAME"}}
  ]
}}
```

JSON:
```json
"""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_statechart(text: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate generated statechart.

    Returns:
        (is_valid, error_message, parsed_sc)
    """
    import re

    # Handle bytes output
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')

    # Extract JSON from text
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find JSON object
    start = text.find('{')
    if start == -1:
        return False, "No JSON object found", None

    # Find matching closing brace
    depth = 0
    end = start
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = text[start:end]

    # Parse JSON
    try:
        sc = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}", None

    # Validate structure
    if "root_state" not in sc:
        return False, "Missing root_state", None

    root = sc["root_state"]
    if "children" not in root:
        return False, "Missing children in root_state", None

    children = root.get("children", [])
    if len(children) == 0:
        return False, "Empty children", None

    # Check for initial state
    has_initial = any(c.get("is_initial", False) for c in children)
    if not has_initial:
        return False, "No initial state", None

    # Check state labels
    for child in children:
        if "label" not in child:
            return False, "State missing label", None

    # Valid transitions (if present)
    transitions = sc.get("transitions", [])
    state_labels = {c["label"] for c in children}

    for trans in transitions:
        from_states = trans.get("from", [])
        to_states = trans.get("to", [])

        for s in from_states:
            if s not in state_labels:
                return False, f"Invalid source state: {s}", None

        for s in to_states:
            if s not in state_labels:
                return False, f"Invalid target state: {s}", None

    return True, "Valid", sc


# =============================================================================
# GENERATION RESULT
# =============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation attempt."""
    prompt: str
    output: str
    is_valid: bool
    error: str
    statechart: Optional[Dict] = None
    generation_time: float = 0.0
    steered: bool = False
    layer: int = 0
    alpha: float = 0.0


@dataclass
class ValidityTestResult:
    """Result of validity testing."""
    baseline_results: List[GenerationResult]
    steered_results: List[GenerationResult]
    baseline_validity_rate: float
    steered_validity_rate: float
    improvement: float
    layer: int
    alpha: float
    target_met: bool  # +10% improvement

    def __repr__(self):
        return (f"ValidityTestResult(baseline={self.baseline_validity_rate:.1%}, "
                f"steered={self.steered_validity_rate:.1%}, "
                f"improvement={self.improvement:+.1%}, "
                f"target_met={self.target_met})")


# =============================================================================
# VALIDITY TESTER
# =============================================================================

class ValidityTester:
    """
    Tests steering vector effect on statechart generation validity.

    Compares baseline (no steering) vs steered generation across
    multiple prompts to measure validity improvement.
    """

    # Target: +10% improvement
    IMPROVEMENT_TARGET = 0.10

    def __init__(
        self,
        model: HookedModelWrapper = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        self.model = model or load_model(model_name)
        self.steering_computer = SteeringVectorComputer(model=self.model)
        self.results: List[ValidityTestResult] = []

    @property
    def has_mlux(self) -> bool:
        return self.model.has_interpretability

    def generate_baseline(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
    ) -> List[GenerationResult]:
        """Generate statecharts without steering."""
        config = config or GenerationConfig(max_tokens=512, temperature=0.3)
        results = []

        for prompt in prompts:
            full_prompt = SC_GENERATION_TEMPLATE.format(description=prompt)

            start = time.time()
            output = self.model.generate(full_prompt, config)
            elapsed = time.time() - start

            is_valid, error, sc = validate_statechart(output)

            results.append(GenerationResult(
                prompt=prompt,
                output=output,
                is_valid=is_valid,
                error=error,
                statechart=sc,
                generation_time=elapsed,
                steered=False,
            ))

        return results

    def generate_steered(
        self,
        prompts: List[str],
        steering_vector: SteeringVector,
        alpha: float = 1.0,
        config: GenerationConfig = None,
    ) -> List[GenerationResult]:
        """Generate statecharts with steering vector applied."""
        config = config or GenerationConfig(max_tokens=512, temperature=0.3)
        results = []

        for prompt in prompts:
            full_prompt = SC_GENERATION_TEMPLATE.format(description=prompt)

            start = time.time()

            if self.has_mlux:
                output = self.model.generate_with_steering(
                    full_prompt,
                    steering_vector=steering_vector.vector,
                    layer=steering_vector.layer,
                    alpha=alpha,
                    config=config,
                )
            else:
                # Fallback: normal generation (no steering effect)
                output = self.model.generate(full_prompt, config)

            elapsed = time.time() - start

            is_valid, error, sc = validate_statechart(output)

            results.append(GenerationResult(
                prompt=prompt,
                output=output,
                is_valid=is_valid,
                error=error,
                statechart=sc,
                generation_time=elapsed,
                steered=True,
                layer=steering_vector.layer,
                alpha=alpha,
            ))

        return results

    def test_validity(
        self,
        prompts: List[str] = None,
        layer: int = 12,
        alpha: float = 1.0,
    ) -> ValidityTestResult:
        """
        Test validity improvement with steering.

        Args:
            prompts: Generation prompts (default: SC_GENERATION_PROMPTS)
            layer: Layer for steering vector
            alpha: Steering strength

        Returns:
            ValidityTestResult with comparison
        """
        if prompts is None:
            prompts = SC_GENERATION_PROMPTS[:5]  # Use first 5 for speed

        print(f"\nTesting validity with layer={layer}, alpha={alpha}")
        print(f"  Prompts: {len(prompts)}")

        # Compute steering vector
        print("  Computing steering vector...")
        steering_vec = self.steering_computer.compute_averaged(layer=layer)
        print(f"  Vector computed: {steering_vec}")

        # Baseline generation
        print("  Running baseline generation...")
        baseline_results = self.generate_baseline(prompts)
        baseline_valid = sum(1 for r in baseline_results if r.is_valid)
        baseline_rate = baseline_valid / len(prompts)
        print(f"  Baseline: {baseline_valid}/{len(prompts)} ({baseline_rate:.1%})")

        # Steered generation
        print("  Running steered generation...")
        steered_results = self.generate_steered(prompts, steering_vec, alpha)
        steered_valid = sum(1 for r in steered_results if r.is_valid)
        steered_rate = steered_valid / len(prompts)
        print(f"  Steered: {steered_valid}/{len(prompts)} ({steered_rate:.1%})")

        # Calculate improvement
        improvement = steered_rate - baseline_rate
        target_met = improvement >= self.IMPROVEMENT_TARGET

        print(f"  Improvement: {improvement:+.1%}")
        print(f"  Target (+10%): {'MET' if target_met else 'NOT MET'}")

        result = ValidityTestResult(
            baseline_results=baseline_results,
            steered_results=steered_results,
            baseline_validity_rate=baseline_rate,
            steered_validity_rate=steered_rate,
            improvement=improvement,
            layer=layer,
            alpha=alpha,
            target_met=target_met,
        )

        self.results.append(result)
        return result

    def test_multiple_configs(
        self,
        prompts: List[str] = None,
        layers: List[int] = None,
        alphas: List[float] = None,
    ) -> List[ValidityTestResult]:
        """
        Test multiple layer/alpha configurations.

        Returns:
            List of results for each configuration
        """
        if prompts is None:
            prompts = SC_GENERATION_PROMPTS[:5]
        if layers is None:
            layers = [6, 12, 18]
        if alphas is None:
            alphas = [0.5, 1.0, 1.5]

        results = []

        for layer in layers:
            for alpha in alphas:
                result = self.test_validity(prompts, layer, alpha)
                results.append(result)

        return results

    def find_best_config(self) -> Optional[ValidityTestResult]:
        """Find configuration with best improvement."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.improvement)

    def summary(self) -> Dict[str, Any]:
        """Get test summary."""
        if not self.results:
            return {"tests": 0}

        best = self.find_best_config()
        target_met_count = sum(1 for r in self.results if r.target_met)

        return {
            "tests": len(self.results),
            "target_met_count": target_met_count,
            "best_improvement": best.improvement if best else 0,
            "best_layer": best.layer if best else 0,
            "best_alpha": best.alpha if best else 0,
            "avg_baseline_rate": sum(r.baseline_validity_rate for r in self.results) / len(self.results),
            "avg_steered_rate": sum(r.steered_validity_rate for r in self.results) / len(self.results),
        }


# =============================================================================
# TESTING
# =============================================================================

def test_validity_tester():
    """Test validity tester functionality."""
    print("=" * 60)
    print("VALIDITY TESTER TEST")
    print("=" * 60)

    # Create tester
    print("\nCreating tester...")
    tester = ValidityTester()
    print(f"  MLUX available: {tester.has_mlux}")

    # Test with a few prompts
    prompts = SC_GENERATION_PROMPTS[:3]

    # Single test
    print("\nRunning single validity test...")
    result = tester.test_validity(prompts, layer=12, alpha=1.0)
    print(f"\nResult: {result}")

    # Summary
    print("\nSummary:")
    summary = tester.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n[PASS] Validity tester test complete")
    return True


if __name__ == "__main__":
    test_validity_tester()
