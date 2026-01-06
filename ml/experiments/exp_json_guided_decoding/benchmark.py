#!/usr/bin/env python3
"""
JSON Guided Decoding Benchmark

Tests the effectiveness of grammar-constrained generation for producing
valid JSON output, specifically for statechart generation tasks.

Usage:
    python benchmark.py
    python benchmark.py --model 0.5B
    python benchmark.py --compare-modes
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .json_grammar import JSONParser, get_valid_next_tokens, validate_json_grammar
from .json_guided_sampler import (
    JSONGuidedSampler,
    SamplerConfig,
    sample_with_grammar,
    MockModel,
)

# Try to import model loading utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.mlux_loader import load_model, ModelBackend
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False

try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


# Test prompts for SC JSON generation
SC_GENERATION_PROMPTS = [
    {
        "name": "simple_toggle",
        "prompt": "Generate JSON for a toggle switch with Off and On states",
        "expected_keys": ["root_state", "transitions"],
    },
    {
        "name": "traffic_light",
        "prompt": "Generate JSON for a traffic light with Red, Yellow, Green states",
        "expected_keys": ["root_state", "transitions"],
    },
    {
        "name": "nested_player",
        "prompt": "Generate JSON for a media player with Playing/Paused states where Playing has Normal/FastForward substates",
        "expected_keys": ["root_state", "transitions"],
    },
    {
        "name": "counter",
        "prompt": "Generate JSON for a counter with Counting and Done states",
        "expected_keys": ["root_state"],
    },
]


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    prompt_name: str
    mode: str  # "guided" or "unconstrained"

    # Validity metrics
    is_valid_json: bool = False
    is_balanced: bool = False  # Matching brackets/braces
    is_complete: bool = False  # Parser reached END state
    has_expected_keys: bool = False

    # Structure metrics
    num_objects: int = 0
    num_arrays: int = 0
    max_depth: int = 0
    total_tokens: int = 0

    # Timing
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Output
    raw_output: str = ""
    parsed_output: Optional[Dict] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "prompt_name": self.prompt_name,
            "mode": self.mode,
            "validity": {
                "json": self.is_valid_json,
                "balanced": self.is_balanced,
                "complete": self.is_complete,
                "has_keys": self.has_expected_keys,
            },
            "structure": {
                "objects": self.num_objects,
                "arrays": self.num_arrays,
                "depth": self.max_depth,
                "tokens": self.total_tokens,
            },
            "performance": {
                "time_ms": self.generation_time_ms,
                "tokens_per_sec": self.tokens_per_second,
            },
            "error": self.error_message,
        }


def check_balanced(json_str: str) -> bool:
    """Check if brackets and braces are balanced."""
    stack = []
    pairs = {'{': '}', '[': ']'}
    in_string = False
    escape = False

    for char in json_str:
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue

        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack[-1] != char:
                return False
            stack.pop()

    return len(stack) == 0


def analyze_structure(parsed: Dict, depth: int = 0) -> Tuple[int, int, int]:
    """Analyze JSON structure. Returns (objects, arrays, max_depth)."""
    objects = 0
    arrays = 0
    max_d = depth

    if isinstance(parsed, dict):
        objects = 1
        for value in parsed.values():
            o, a, d = analyze_structure(value, depth + 1)
            objects += o
            arrays += a
            max_d = max(max_d, d)
    elif isinstance(parsed, list):
        arrays = 1
        for item in parsed:
            o, a, d = analyze_structure(item, depth + 1)
            objects += o
            arrays += a
            max_d = max(max_d, d)

    return objects, arrays, max_d


def evaluate_output(
    output: str,
    prompt_config: Dict,
    mode: str,
    generation_time_ms: float,
) -> BenchmarkResult:
    """Evaluate a generation output."""
    result = BenchmarkResult(
        prompt_name=prompt_config["name"],
        mode=mode,
        raw_output=output,
        generation_time_ms=generation_time_ms,
        total_tokens=len(output),
    )

    if generation_time_ms > 0 and len(output) > 0:
        result.tokens_per_second = (len(output) / generation_time_ms) * 1000

    # Check balanced brackets
    result.is_balanced = check_balanced(output)

    # Validate with grammar parser
    is_valid, error = validate_json_grammar(output)
    result.is_complete = is_valid
    if error:
        result.error_message = error

    # Try to parse as JSON
    try:
        parsed = json.loads(output)
        result.is_valid_json = True
        result.parsed_output = parsed

        # Check expected keys
        expected = prompt_config.get("expected_keys", [])
        if isinstance(parsed, dict):
            result.has_expected_keys = all(k in parsed for k in expected)

        # Analyze structure
        result.num_objects, result.num_arrays, result.max_depth = analyze_structure(parsed)

    except json.JSONDecodeError as e:
        result.is_valid_json = False
        if not result.error_message:
            result.error_message = str(e)

    return result


class JSONGuidedBenchmark:
    """Benchmark runner for JSON guided decoding."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        prompts: Optional[List[Dict]] = None,
        config: Optional[SamplerConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts or SC_GENERATION_PROMPTS
        self.config = config or SamplerConfig()
        self.results: List[BenchmarkResult] = []

    def run_guided(self, prompt_config: Dict) -> BenchmarkResult:
        """Run with grammar-guided generation."""
        prompt = self._format_prompt(prompt_config["prompt"])

        start = time.time()

        if self.model is not None:
            output, _ = sample_with_grammar(
                self.model,
                self.tokenizer,
                prompt,
                self.config,
            )
        else:
            # Mock generation
            output = self._mock_guided_generate(prompt_config)

        gen_time = (time.time() - start) * 1000

        return evaluate_output(output, prompt_config, "guided", gen_time)

    def run_unconstrained(self, prompt_config: Dict) -> BenchmarkResult:
        """Run without grammar constraints (baseline)."""
        prompt = self._format_prompt(prompt_config["prompt"])

        start = time.time()

        if HAS_MLX_LM and self.model is not None:
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature,
            )
        else:
            # Mock generation - may produce invalid JSON
            output = self._mock_unconstrained_generate(prompt_config)

        gen_time = (time.time() - start) * 1000

        return evaluate_output(output, prompt_config, "unconstrained", gen_time)

    def _format_prompt(self, task: str) -> str:
        """Format prompt for JSON generation."""
        return f"""Generate valid JSON for a statechart.
Task: {task}
Output only valid JSON, no explanations.
JSON:"""

    def _mock_guided_generate(self, prompt_config: Dict) -> str:
        """Mock guided generation - always produces valid JSON."""
        name = prompt_config["name"]
        templates = {
            "simple_toggle": json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Off", "type": 1, "is_initial": True},
                        {"label": "On", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
                    {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"}
                ]
            }),
            "traffic_light": json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Red", "type": 1, "is_initial": True},
                        {"label": "Yellow", "type": 1},
                        {"label": "Green", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                    {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                    {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"}
                ]
            }),
            "nested_player": json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Paused", "type": 1, "is_initial": True},
                        {
                            "label": "Playing",
                            "type": 2,
                            "children": [
                                {"label": "Normal", "type": 1, "is_initial": True},
                                {"label": "FastForward", "type": 1}
                            ]
                        }
                    ]
                },
                "transitions": [
                    {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
                    {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"}
                ]
            }),
            "counter": json.dumps({
                "root_state": {
                    "label": "__root__",
                    "type": 2,
                    "children": [
                        {"label": "Counting", "type": 1, "is_initial": True},
                        {"label": "Done", "type": 1}
                    ]
                },
                "transitions": [
                    {"from": ["Counting"], "to": ["Done"], "event": "FINISH"}
                ]
            }),
        }
        return templates.get(name, '{"root_state": {"label": "test", "type": 1}}')

    def _mock_unconstrained_generate(self, prompt_config: Dict) -> str:
        """Mock unconstrained generation - may produce invalid JSON."""
        import random

        # Simulate LLM failure modes
        failure_rate = 0.3
        if random.random() < failure_rate:
            failures = [
                '{"root_state": {"label": "test"',  # Missing closing
                '{"root_state: {"label": "test"}}',  # Missing quote
                'Here is the JSON:\n{"root_state": {}}',  # Extra text
                '{"root_state": {"label": "test",}}',  # Trailing comma
            ]
            return random.choice(failures)

        # Otherwise return valid
        return self._mock_guided_generate(prompt_config)

    def run_comparison(self) -> Dict[str, Any]:
        """Run both modes and compare results."""
        guided_results = []
        unconstrained_results = []

        for prompt_config in self.prompts:
            print(f"Testing: {prompt_config['name']}")

            # Guided
            guided = self.run_guided(prompt_config)
            guided_results.append(guided)
            self.results.append(guided)
            status = "✓" if guided.is_valid_json else "✗"
            print(f"  Guided:        {status} (balanced={guided.is_balanced})")

            # Unconstrained
            unconstrained = self.run_unconstrained(prompt_config)
            unconstrained_results.append(unconstrained)
            self.results.append(unconstrained)
            status = "✓" if unconstrained.is_valid_json else "✗"
            print(f"  Unconstrained: {status} (balanced={unconstrained.is_balanced})")

        return self._compute_summary(guided_results, unconstrained_results)

    def _compute_summary(
        self,
        guided: List[BenchmarkResult],
        unconstrained: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        def rate(results: List[BenchmarkResult], attr: str) -> float:
            count = sum(1 for r in results if getattr(r, attr))
            return count / len(results) if results else 0.0

        def avg(results: List[BenchmarkResult], attr: str) -> float:
            vals = [getattr(r, attr) for r in results]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "guided": {
                "valid_json_rate": rate(guided, "is_valid_json"),
                "balanced_rate": rate(guided, "is_balanced"),
                "complete_rate": rate(guided, "is_complete"),
                "has_keys_rate": rate(guided, "has_expected_keys"),
                "avg_tokens": avg(guided, "total_tokens"),
                "avg_time_ms": avg(guided, "generation_time_ms"),
            },
            "unconstrained": {
                "valid_json_rate": rate(unconstrained, "is_valid_json"),
                "balanced_rate": rate(unconstrained, "is_balanced"),
                "complete_rate": rate(unconstrained, "is_complete"),
                "has_keys_rate": rate(unconstrained, "has_expected_keys"),
                "avg_tokens": avg(unconstrained, "total_tokens"),
                "avg_time_ms": avg(unconstrained, "generation_time_ms"),
            },
            "improvement": {
                "json_validity": (
                    rate(guided, "is_valid_json") - rate(unconstrained, "is_valid_json")
                ) * 100,
                "balanced": (
                    rate(guided, "is_balanced") - rate(unconstrained, "is_balanced")
                ) * 100,
            },
        }

    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("JSON GUIDED DECODING RESULTS")
        print("=" * 60)

        print("\nValidity Rates:")
        print(f"  {'Mode':<15} {'JSON%':>10} {'Balanced%':>12} {'Complete%':>12}")
        print("-" * 55)

        g = summary["guided"]
        print(f"  {'Guided':<15} {g['valid_json_rate']*100:>9.1f}% {g['balanced_rate']*100:>11.1f}% {g['complete_rate']*100:>11.1f}%")

        u = summary["unconstrained"]
        print(f"  {'Unconstrained':<15} {u['valid_json_rate']*100:>9.1f}% {u['balanced_rate']*100:>11.1f}% {u['complete_rate']*100:>11.1f}%")

        print("\nImprovement:")
        imp = summary["improvement"]
        print(f"  JSON validity: +{imp['json_validity']:.1f}%")
        print(f"  Balanced:      +{imp['balanced']:.1f}%")

        print("\nPerformance:")
        print(f"  Guided avg tokens:        {g['avg_tokens']:.0f}")
        print(f"  Unconstrained avg tokens: {u['avg_tokens']:.0f}")


def run_benchmark(
    model=None,
    tokenizer=None,
    prompts: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run the full benchmark."""
    benchmark = JSONGuidedBenchmark(model, tokenizer, prompts)
    summary = benchmark.run_comparison()
    benchmark.print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="JSON Guided Decoding Benchmark")
    parser.add_argument("--model", default=None, help="Model size to test")
    parser.add_argument("--compare-modes", action="store_true", help="Compare guided vs unconstrained")
    parser.add_argument("--output", default="./results/json_guided.json", help="Output file")

    args = parser.parse_args()

    # Load model if specified
    model = None
    tokenizer = None

    if args.model and HAS_LOADER:
        from ..exp_model_comparison.model_configs import get_model_config
        config = get_model_config(args.model)
        wrapper = load_model(config.mlx_name)
        model = wrapper
        tokenizer = wrapper.tokenizer

    # Run benchmark
    print("Running JSON Guided Decoding Benchmark")
    print(f"Model: {args.model or 'mock'}")
    print()

    summary = run_benchmark(model, tokenizer)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Return key metrics for reporting
    g = summary["guided"]
    return {
        "validity": g["valid_json_rate"] * 100,
        "balanced": g["balanced_rate"] * 100,
        "avg_tokens": g["avg_tokens"],
    }


if __name__ == "__main__":
    main()
