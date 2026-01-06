#!/usr/bin/env python3
"""
Base vs Instruct Model Comparison for SC Generation.

Tests if base models (non-Instruct) perform differently for
completion-style statechart generation.

HYPOTHESIS: Base models may be better at raw completion tasks.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

# Models to test
MODELS = {
    "0.5B_base": "mlx-community/Qwen2.5-Coder-0.5B-4bit",
    "0.5B_instruct": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "1.5B_base": "mlx-community/Qwen2.5-Coder-1.5B-4bit",
    "1.5B_instruct": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
}

# Completion-style prompts (not instructions)
TEST_PROMPTS = [
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle", "type": 1}, {"label": "Active", "type": 2, "children": [{"label": "Running", "type": 1}, {"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Off", "type": 1}, {"label": "On", "type": 2, "children": [{"label": "Low", "type": 1}, {"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Init", "type": 1}, {"label": "Ready", "type": 2, "children": [{"label": "Waiting", "type": 1}, {"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Start", "type": 1}, {"label": "Process", "type": 2, "children": [{"label": "Step1", "type": 1}, {"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Stopped", "type": 1}, {"label": "Playing", "type": 2, "children": [{"label": "Normal", "type": 1}, {"label":',
]


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    model_type: str  # "base" or "instruct"
    num_samples: int = 0
    valid_json_count: int = 0
    has_hierarchy_count: int = 0
    total_states: int = 0
    total_nested: int = 0
    generation_times: List[float] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return self.valid_json_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def hierarchy_rate(self) -> float:
        return self.has_hierarchy_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_states(self) -> float:
        return self.total_states / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_time(self) -> float:
        return sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0


def parse_and_analyze(prompt: str, output: str) -> Dict[str, Any]:
    """Parse combined prompt+output and analyze structure."""
    result = {
        "valid_json": False,
        "has_hierarchy": False,
        "num_states": 0,
        "num_nested": 0,
    }

    full_text = prompt + output

    try:
        # Find JSON boundaries
        start = full_text.find('{')
        if start == -1:
            return result

        depth = 0
        end = start
        for i, c in enumerate(full_text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = full_text[start:end]
        data = json.loads(json_str)
        result["valid_json"] = True

        def count_states(state: Dict, depth: int = 0) -> Tuple[int, int]:
            total = 1
            nested = 0
            children = state.get('children', [])
            if children:
                nested = 1
                for child in children:
                    if isinstance(child, dict):
                        c_total, c_nested = count_states(child, depth + 1)
                        total += c_total
                        nested += c_nested
            return total, nested

        root = data.get('root_state', data)
        if isinstance(root, dict):
            total, nested = count_states(root)
            result["num_states"] = total
            result["num_nested"] = nested
            result["has_hierarchy"] = nested > 0

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return result


def test_model(model_key: str, model_path: str, n_samples: int = 10) -> ModelResult:
    """Test a single model."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model_type = "base" if "base" in model_key else "instruct"
    result = ModelResult(model_name=model_path, model_type=model_type)

    print(f"\nLoading {model_key}: {model_path}")
    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.3)

    print(f"Testing {n_samples} samples...")

    for i in range(n_samples):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]

        start_time = time.time()
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=150,
            sampler=sampler,
            verbose=False,
        )
        gen_time = time.time() - start_time

        # Analyze
        analysis = parse_and_analyze(prompt, output)

        result.num_samples += 1
        result.generation_times.append(gen_time)

        if analysis["valid_json"]:
            result.valid_json_count += 1
        if analysis["has_hierarchy"]:
            result.has_hierarchy_count += 1
        result.total_states += analysis["num_states"]
        result.total_nested += analysis["num_nested"]

        status = "H" if analysis["has_hierarchy"] else "V" if analysis["valid_json"] else "X"
        print(f"  [{status}] {i+1}/{n_samples}: {analysis['num_states']} states")

    # Free memory
    del model
    del tokenizer

    return result


def run_comparison(n_samples: int = 10, models_to_test: List[str] = None) -> Dict[str, ModelResult]:
    """Run full comparison."""
    print("=" * 60)
    print("BASE vs INSTRUCT COMPARISON")
    print("=" * 60)

    if models_to_test is None:
        models_to_test = list(MODELS.keys())

    results = {}

    for model_key in models_to_test:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            continue

        model_path = MODELS[model_key]
        try:
            results[model_key] = test_model(model_key, model_path, n_samples)
        except Exception as e:
            print(f"Error testing {model_key}: {e}")
            continue

    return results


def print_summary(results: Dict[str, ModelResult]):
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Type':<10} {'Valid%':>8} {'Hier%':>8} {'States':>8} {'Time':>8}")
    print("-" * 70)

    for name, r in sorted(results.items()):
        print(f"{name:<20} {r.model_type:<10} {r.validity_rate:>7.1%} {r.hierarchy_rate:>7.1%} "
              f"{r.avg_states:>8.1f} {r.avg_time:>7.2f}s")

    # Compare base vs instruct
    print("\n--- Base vs Instruct Delta ---")

    for size in ["0.5B", "1.5B"]:
        base_key = f"{size}_base"
        inst_key = f"{size}_instruct"

        if base_key in results and inst_key in results:
            base = results[base_key]
            inst = results[inst_key]
            delta = base.validity_rate - inst.validity_rate
            print(f"  {size}: Base {base.validity_rate:.1%} vs Instruct {inst.validity_rate:.1%} "
                  f"(delta: {delta:+.1%})")


def quick_test():
    """Quick test with just base models."""
    results = run_comparison(n_samples=10, models_to_test=["0.5B_base", "1.5B_base"])
    print_summary(results)
    return results


def full_test():
    """Full comparison of all models."""
    results = run_comparison(n_samples=10)
    print_summary(results)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        full_test()
    else:
        quick_test()
