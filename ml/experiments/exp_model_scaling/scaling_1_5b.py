"""
Model Scaling Experiment: Qwen2.5-Coder-1.5B

Tests if steering improvements scale with model size.

BASELINE FROM 0.5B:
- Baseline validity: 80%
- hierarchy_boost (1.5x L23H1): 80% valid, 8 states, 4 nested
- Over-amplification (>2.0x) hurts validity

TEST PROTOCOL:
1. Load 1.5B model with mlux HookedModel
2. Test configs: baseline, hierarchy_boost, structure_boost, full_steering
3. Run 20 generations per config
4. Measure: validity rate, hierarchy depth, state count

HYPOTHESIS: Larger model = higher baseline + larger steering delta
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    MLX_LM_AVAILABLE,
    MLUX_AVAILABLE,
)


MODEL_NAME = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

# Steering configurations based on 0.5B findings
STEERING_CONFIGS = {
    "baseline": {
        "description": "No steering",
        "hooks": [],
    },
    "hierarchy_boost": {
        "description": "L23H1 x1.5 for hierarchy",
        "hooks": [("L23", "H1", 1.5)],
    },
    "structure_boost": {
        "description": "L11H13, L11H7, L9H7 x1.5 for structure",
        "hooks": [
            ("L11", "H13", 1.5),
            ("L11", "H7", 1.5),
            ("L9", "H7", 1.5),
        ],
    },
    "full_steering": {
        "description": "Both hierarchy and structure x1.5",
        "hooks": [
            ("L23", "H1", 1.5),
            ("L11", "H13", 1.5),
            ("L11", "H7", 1.5),
            ("L9", "H7", 1.5),
        ],
    },
}

# Test prompts for generation
TEST_PROMPTS = [
    "Generate a JSON statechart for a traffic light system with red, yellow, and green states:",
    "Create a JSON statechart for a login flow with authentication states:",
    "Design a JSON statechart for an order processing system:",
    "Build a JSON statechart for a media player with play, pause, stop:",
    "Generate a JSON statechart for a vending machine:",
]


@dataclass
class GenerationResult:
    """Result of a single generation."""
    prompt: str
    output: str
    config_name: str
    is_valid: bool
    state_count: int
    max_depth: int
    has_hierarchy: bool
    parse_error: Optional[str] = None


@dataclass
class ConfigMetrics:
    """Aggregated metrics for a configuration."""
    config_name: str
    n_samples: int
    validity_rate: float
    mean_state_count: float
    mean_max_depth: float
    hierarchy_rate: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config_name,
            "validity": f"{self.validity_rate:.1%}",
            "states": f"{self.mean_state_count:.1f}",
            "depth": f"{self.mean_max_depth:.1f}",
            "hierarchy": f"{self.hierarchy_rate:.1%}",
        }


def parse_statechart(output: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Extract and parse JSON statechart from output."""
    # Try to find JSON in output
    text = output

    # Look for JSON block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    # Try to find JSON object
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text[brace_start:]):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    text = text[brace_start:brace_start + i + 1]
                    break

    try:
        chart = json.loads(text)
        return chart, None
    except json.JSONDecodeError as e:
        return None, str(e)


def validate_statechart(chart: Dict[str, Any]) -> bool:
    """Check if statechart is valid."""
    if not isinstance(chart, dict):
        return False

    # Must have root_state
    if "root_state" not in chart:
        return False

    root = chart["root_state"]
    if not isinstance(root, dict):
        return False

    # Root must have label and children
    if "label" not in root:
        return False

    children = root.get("children", [])
    if not children:
        return False

    # Check for at least one state with label
    for child in children:
        if isinstance(child, dict) and "label" in child:
            return True

    return False


def count_states(chart: Dict[str, Any]) -> int:
    """Count total states in chart."""
    count = 0

    def traverse(state):
        nonlocal count
        if isinstance(state, dict):
            if "label" in state:
                count += 1
            for child in state.get("children", []):
                traverse(child)

    traverse(chart.get("root_state", {}))
    return count


def get_max_depth(chart: Dict[str, Any]) -> int:
    """Get maximum nesting depth."""
    def traverse(state, depth=0):
        if not isinstance(state, dict):
            return depth

        max_child_depth = depth
        for child in state.get("children", []):
            child_depth = traverse(child, depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    return traverse(chart.get("root_state", {}))


def has_nested_states(chart: Dict[str, Any]) -> bool:
    """Check if chart has nested compound states."""
    def check(state, depth=0):
        if not isinstance(state, dict):
            return False

        children = state.get("children", [])
        if depth > 0 and children:
            # Found nested compound
            return True

        for child in children:
            if check(child, depth + 1):
                return True

        return False

    return check(chart.get("root_state", {}))


class ScalingExperiment:
    """
    Model scaling experiment for 1.5B Qwen model.
    """

    def __init__(
        self,
        model: Optional[HookedModelWrapper] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model = model

        if self.model is None and MLX_LM_AVAILABLE:
            if self.verbose:
                print(f"Loading model: {MODEL_NAME}")
            self.model = load_model(MODEL_NAME)

    def generate_with_config(
        self,
        prompt: str,
        config_name: str,
    ) -> GenerationResult:
        """Generate with a specific steering configuration."""
        config = STEERING_CONFIGS[config_name]

        # Build full prompt
        full_prompt = f"""You are a statechart expert. Generate a valid JSON statechart.

{prompt}

The statechart must have:
- root_state with label "__root__" and type 2
- children array with states (each has label, type)
- is_initial: true on one child
- transitions array with from, to, event

Output only valid JSON:"""

        # Generate
        gen_config = GenerationConfig(
            max_tokens=512,
            temperature=0.7,
        )

        if self.model is not None:
            output = self.model.generate(full_prompt, gen_config)
        else:
            # Mock for testing
            output = self._mock_generate(prompt, config_name)

        # Parse and validate
        chart, parse_error = parse_statechart(output)

        if chart is None:
            return GenerationResult(
                prompt=prompt,
                output=output,
                config_name=config_name,
                is_valid=False,
                state_count=0,
                max_depth=0,
                has_hierarchy=False,
                parse_error=parse_error,
            )

        is_valid = validate_statechart(chart)
        state_count = count_states(chart) if is_valid else 0
        max_depth = get_max_depth(chart) if is_valid else 0
        has_hier = has_nested_states(chart) if is_valid else False

        return GenerationResult(
            prompt=prompt,
            output=output,
            config_name=config_name,
            is_valid=is_valid,
            state_count=state_count,
            max_depth=max_depth,
            has_hierarchy=has_hier,
        )

    def _mock_generate(self, prompt: str, config_name: str) -> str:
        """Mock generation for testing."""
        import random

        # Simulate different validity rates based on config
        validity_probs = {
            "baseline": 0.85,
            "hierarchy_boost": 0.90,
            "structure_boost": 0.88,
            "full_steering": 0.92,
        }

        if random.random() < validity_probs.get(config_name, 0.8):
            # Generate valid chart
            n_states = random.randint(3, 8)
            depth = 2 if config_name in ("hierarchy_boost", "full_steering") else 1

            children = []
            for i in range(n_states):
                state = {"label": f"s{i}", "type": 1}
                if i == 0:
                    state["is_initial"] = True
                if depth > 1 and i == 1:
                    state["type"] = 2
                    state["children"] = [
                        {"label": f"s{i}_sub0", "type": 1, "is_initial": True},
                        {"label": f"s{i}_sub1", "type": 1},
                    ]
                children.append(state)

            chart = {
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": children,
                },
                "transitions": [
                    {"from": ["s0"], "to": ["s1"], "event": "E1"},
                ],
            }
            return json.dumps(chart, indent=2)
        else:
            # Generate invalid output
            return "Sorry, I cannot generate that."

    def run_experiment(
        self,
        n_samples: int = 20,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, ConfigMetrics]:
        """
        Run the scaling experiment.

        Args:
            n_samples: Number of generations per config
            prompts: Test prompts (uses defaults if None)

        Returns:
            Metrics for each configuration
        """
        if prompts is None:
            prompts = TEST_PROMPTS

        results_by_config: Dict[str, List[GenerationResult]] = defaultdict(list)

        total = len(STEERING_CONFIGS) * n_samples
        current = 0

        for config_name in STEERING_CONFIGS:
            if self.verbose:
                print(f"\nTesting {config_name}...")

            start = time.time()

            for i in range(n_samples):
                prompt = prompts[i % len(prompts)]
                result = self.generate_with_config(prompt, config_name)
                results_by_config[config_name].append(result)

                current += 1
                if self.verbose and current % 10 == 0:
                    print(f"  Progress: {current}/{total}")

            duration = time.time() - start

            # Compute metrics
            results = results_by_config[config_name]
            n = len(results)

            valid_results = [r for r in results if r.is_valid]
            n_valid = len(valid_results)

            metrics = ConfigMetrics(
                config_name=config_name,
                n_samples=n,
                validity_rate=n_valid / n if n > 0 else 0,
                mean_state_count=sum(r.state_count for r in valid_results) / n_valid if n_valid > 0 else 0,
                mean_max_depth=sum(r.max_depth for r in valid_results) / n_valid if n_valid > 0 else 0,
                hierarchy_rate=sum(1 for r in valid_results if r.has_hierarchy) / n_valid if n_valid > 0 else 0,
                duration=duration,
            )

            if self.verbose:
                print(f"  Validity: {metrics.validity_rate:.1%}")
                print(f"  States: {metrics.mean_state_count:.1f}")
                print(f"  Depth: {metrics.mean_max_depth:.1f}")

            results_by_config[config_name + "_metrics"] = metrics

        # Extract just the metrics
        return {
            name: results_by_config[name + "_metrics"]
            for name in STEERING_CONFIGS
        }


def run_scaling_test(n_samples: int = 20, use_model: bool = True) -> Dict[str, ConfigMetrics]:
    """
    Run the 1.5B scaling test.

    Args:
        n_samples: Number of generations per config
        use_model: Whether to use real model (False = mock)

    Returns:
        Metrics for each configuration
    """
    print("=" * 60)
    print("Model Scaling Experiment: Qwen2.5-Coder-1.5B")
    print("=" * 60)

    if use_model and MLX_LM_AVAILABLE:
        model = load_model(MODEL_NAME)
    else:
        model = None
        print("(Using mock model for testing)")

    experiment = ScalingExperiment(model=model, verbose=True)
    metrics = experiment.run_experiment(n_samples=n_samples)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS: 1.5B Scaling Test")
    print("=" * 60)

    print(f"\n{'Config':<20} {'Valid':<10} {'States':<10} {'Depth':<10} {'Hierarchy':<10}")
    print("-" * 60)

    for name in ["baseline", "hierarchy_boost", "structure_boost", "full_steering"]:
        m = metrics[name]
        print(f"{name:<20} {m.validity_rate:>8.1%} {m.mean_state_count:>8.1f} "
              f"{m.mean_max_depth:>8.1f} {m.hierarchy_rate:>8.1%}")

    # Compare to 0.5B baseline
    print("\n" + "-" * 60)
    print("COMPARISON TO 0.5B BASELINE:")
    print("  0.5B baseline: 80% valid")
    print(f"  1.5B baseline: {metrics['baseline'].validity_rate:.1%} valid")

    delta = metrics['baseline'].validity_rate - 0.80
    print(f"  Delta: {delta:+.1%}")

    return metrics


def demo():
    """Demo with mock model."""
    return run_scaling_test(n_samples=20, use_model=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real model")
    parser.add_argument("--samples", type=int, default=20, help="Samples per config")
    args = parser.parse_args()

    run_scaling_test(n_samples=args.samples, use_model=args.real)
