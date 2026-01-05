#!/usr/bin/env python3
"""
Benchmark: Compare guided vs unguided SC generation.

Measures:
1. JSON validity rate
2. Structure completeness (balanced braces)
3. SC validity (required fields)
4. Generation characteristics
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .guided_sampler import GuidedGenerator


@dataclass
class BenchmarkResult:
    """Results for a benchmark run."""
    name: str
    num_samples: int = 0
    valid_json_count: int = 0
    balanced_braces_count: int = 0
    has_root_state_count: int = 0
    has_label_count: int = 0
    total_states: int = 0
    forced_closes: int = 0
    generation_times: List[float] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return self.valid_json_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def balanced_rate(self) -> float:
        return self.balanced_braces_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def root_state_rate(self) -> float:
        return self.has_root_state_count / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_states(self) -> float:
        return self.total_states / self.num_samples if self.num_samples > 0 else 0

    @property
    def avg_time(self) -> float:
        return sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0


# Test prompts
TEST_PROMPTS = [
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle", "type": 1}, {"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Off"}, {"label": "On", "children": [{"label":',
    '{"root_state": {"label": "__root__", "children": [{"label": "Start"}, {"label": "Running", "children": [{"label":',
    '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Init"}, {"label":',
    '{"root_state": {"label": "__root__", "children": [{"label": "A", "type": 1}, {"label": "B", "children": [{"label":',
]


def check_balanced(text: str) -> bool:
    """Check if braces/brackets are balanced."""
    stack = []
    in_string = False

    for char in text:
        if char == '"' and (not stack or stack[-1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if char in '{[':
                stack.append(char)
            elif char == '}':
                if not stack or stack[-1] != '{':
                    return False
                stack.pop()
            elif char == ']':
                if not stack or stack[-1] != '[':
                    return False
                stack.pop()

    return len(stack) == 0


def analyze_output(prompt: str, output: str) -> Dict[str, Any]:
    """Analyze generated output."""
    full_text = prompt + output

    result = {
        "valid_json": False,
        "balanced": check_balanced(full_text),
        "has_root_state": False,
        "has_label": False,
        "num_states": 0,
        "raw_output": output[:200],
    }

    # Try to parse JSON
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

        # Check SC structure
        if "root_state" in data:
            result["has_root_state"] = True

        # Count states
        def count_labels(obj: Any) -> int:
            if not isinstance(obj, dict):
                return 0
            count = 1 if "label" in obj else 0
            for v in obj.values():
                if isinstance(v, dict):
                    count += count_labels(v)
                elif isinstance(v, list):
                    for item in v:
                        count += count_labels(item)
            return count

        result["num_states"] = count_labels(data)
        result["has_label"] = result["num_states"] > 0

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return result


def run_benchmark(
    n_samples: int = 10,
    model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
) -> Dict[str, BenchmarkResult]:
    """Run guided vs unguided benchmark."""
    print("=" * 60)
    print("SCHEMA-GUIDED SAMPLING BENCHMARK")
    print("=" * 60)

    from mlx_lm import load

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)

    generator = GuidedGenerator(model, tokenizer, max_depth=6, max_states=12)

    results = {
        "unguided": BenchmarkResult(name="unguided"),
        "guided": BenchmarkResult(name="guided"),
    }

    print(f"\nRunning {n_samples} samples per config...")

    for i in range(n_samples):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        print(f"\n--- Sample {i+1}/{n_samples} ---")

        # Unguided
        start = time.time()
        output_ug, meta_ug = generator.generate(prompt, max_tokens=150, use_guidance=False)
        time_ug = time.time() - start

        analysis_ug = analyze_output(prompt, output_ug)
        r = results["unguided"]
        r.num_samples += 1
        r.generation_times.append(time_ug)
        if analysis_ug["valid_json"]:
            r.valid_json_count += 1
        if analysis_ug["balanced"]:
            r.balanced_braces_count += 1
        if analysis_ug["has_root_state"]:
            r.has_root_state_count += 1
        if analysis_ug["has_label"]:
            r.has_label_count += 1
        r.total_states += analysis_ug["num_states"]

        status_ug = "V" if analysis_ug["valid_json"] else ("B" if analysis_ug["balanced"] else "X")
        print(f"  [UG:{status_ug}] {analysis_ug['num_states']} states, balanced={analysis_ug['balanced']}")

        # Guided
        start = time.time()
        output_g, meta_g = generator.generate(prompt, max_tokens=150, use_guidance=True)
        time_g = time.time() - start

        analysis_g = analyze_output(prompt, output_g)
        r = results["guided"]
        r.num_samples += 1
        r.generation_times.append(time_g)
        r.forced_closes += meta_g.get("forced_closes", 0)
        if analysis_g["valid_json"]:
            r.valid_json_count += 1
        if analysis_g["balanced"]:
            r.balanced_braces_count += 1
        if analysis_g["has_root_state"]:
            r.has_root_state_count += 1
        if analysis_g["has_label"]:
            r.has_label_count += 1
        r.total_states += analysis_g["num_states"]

        status_g = "V" if analysis_g["valid_json"] else ("B" if analysis_g["balanced"] else "X")
        print(f"  [G:{status_g}] {analysis_g['num_states']} states, balanced={analysis_g['balanced']}, forced={meta_g.get('forced_closes', 0)}")

    return results


def print_summary(results: Dict[str, BenchmarkResult]):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\n{'Config':<12} {'Valid%':>8} {'Balanced%':>10} {'RootState%':>11} {'States':>8} {'Time':>8}")
    print("-" * 65)

    for name, r in results.items():
        print(f"{name:<12} {r.validity_rate:>7.1%} {r.balanced_rate:>9.1%} "
              f"{r.root_state_rate:>10.1%} {r.avg_states:>8.1f} {r.avg_time:>7.2f}s")

    # Delta
    if "unguided" in results and "guided" in results:
        ug = results["unguided"]
        g = results["guided"]

        print("\n--- Guided vs Unguided Delta ---")
        print(f"  Validity: {g.validity_rate - ug.validity_rate:+.1%}")
        print(f"  Balanced: {g.balanced_rate - ug.balanced_rate:+.1%}")
        print(f"  Forced closes: {g.forced_closes}")


def quick_test():
    """Quick test with few samples."""
    results = run_benchmark(n_samples=5)
    print_summary(results)
    return results


def full_test():
    """Full benchmark."""
    results = run_benchmark(n_samples=20)
    print_summary(results)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        full_test()
    else:
        quick_test()
