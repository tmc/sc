"""
Benchmark: Introspection Token Effectiveness.

Compares SC generation quality with and without introspection tokens.
Uses REAL Qwen-1.5B inference.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from mlx_lm import load, generate


@dataclass
class IntrospectionBenchmarkResult:
    """Results from introspection benchmark."""
    mode: str
    num_samples: int
    sc_valid: int
    json_valid: int
    
    @property
    def sc_rate(self) -> float:
        return self.sc_valid / self.num_samples if self.num_samples > 0 else 0
    
    @property
    def json_rate(self) -> float:
        return self.json_valid / self.num_samples if self.num_samples > 0 else 0


def is_valid_sc_json(text: str) -> Tuple[bool, bool]:
    """
    Check if text contains valid SC JSON.
    Returns (is_valid_json, is_valid_sc).
    """
    try:
        # Extract JSON from markdown if present
        json_match = re.search(r'```json?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            start = text.find('{')
            if start == -1:
                return False, False
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = text[start:end]

        data = json.loads(json_str)
        is_json = True
        
        # Check SC structure
        is_sc = False
        if isinstance(data, dict) and 'root_state' in data:
            root = data['root_state']
            if isinstance(root, dict) and 'label' in root:
                is_sc = True
        
        return is_json, is_sc
    except:
        return False, False


# Test prompts for SC generation
TEST_PROMPTS = [
    "Generate a valid statechart JSON with root_state containing label and type:",
    "Create a state machine JSON with states Off and On:",
    "Write statechart JSON for a toggle switch:",
    "Generate SC JSON with root_state, label, and transitions:",
    "Create a hierarchical statechart JSON with children states:",
]

# Introspection contexts to test
INTROSPECTION_CONTEXTS = {
    "NONE": "",
    "STATE": "\n[SC:STATE=START] [SC:DEPTH=0]\nValid next: { to begin JSON object.\nGenerate:",
    "VALID": "\n[SC:VALID={,root_state,transitions}]\nYou must output valid SC JSON with root_state field.\nGenerate:",
    "FULL": "\n[SC:STATE=START] [SC:VALID={,root_state}] [SC:DEPTH=0]\nConstraints: Must have root_state with label. type is 1,2,3.\nGenerate:",
}


def run_introspection_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    num_samples: int = 5,
) -> Dict[str, IntrospectionBenchmarkResult]:
    """
    Run introspection benchmark with REAL inference.
    
    Compares generation with different introspection modes.
    """
    print("=" * 70)
    print("Introspection Token Benchmark (REAL MLX)")
    print("=" * 70)
    
    # Load model once
    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    print("Model loaded.")
    
    results = {}
    
    for mode, context in INTROSPECTION_CONTEXTS.items():
        print(f"\n--- Testing mode: {mode} ---")
        
        sc_valid = 0
        json_valid = 0
        
        for i, base_prompt in enumerate(TEST_PROMPTS[:num_samples]):
            # Build prompt with introspection context
            if context:
                prompt = base_prompt + context
            else:
                prompt = base_prompt
            
            # Generate
            output = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=150,
                verbose=False,
            )
            
            # Evaluate
            is_json, is_sc = is_valid_sc_json(output)
            
            if is_json:
                json_valid += 1
            if is_sc:
                sc_valid += 1
            
            status = "SC" if is_sc else ("JSON" if is_json else "INVALID")
            print(f"  [{i+1}] {status}: {base_prompt[:40]}...")
        
        results[mode] = IntrospectionBenchmarkResult(
            mode=mode,
            num_samples=num_samples,
            sc_valid=sc_valid,
            json_valid=json_valid,
        )
        
        print(f"  Result: SC={sc_valid}/{num_samples} ({results[mode].sc_rate:.0%})")
    
    return results


def print_benchmark_results(results: Dict[str, IntrospectionBenchmarkResult]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("INTROSPECTION BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Mode':<12} {'JSON Valid':>12} {'SC Valid':>12}")
    print("-" * 40)
    
    baseline_sc = results.get("NONE", IntrospectionBenchmarkResult("NONE", 1, 0, 0)).sc_rate
    
    for mode, result in results.items():
        improvement = result.sc_rate - baseline_sc if mode != "NONE" else 0
        imp_str = f" ({improvement:+.0%})" if mode != "NONE" else ""
        print(f"{mode:<12} {result.json_rate:>11.0%} {result.sc_rate:>11.0%}{imp_str}")
    
    print("=" * 70)
    
    # Calculate best improvement
    best_mode = max(results.items(), key=lambda x: x[1].sc_rate)
    improvement = best_mode[1].sc_rate - baseline_sc
    
    print(f"\nBaseline (NONE): {baseline_sc:.0%}")
    print(f"Best mode: {best_mode[0]} ({best_mode[1].sc_rate:.0%})")
    print(f"Improvement: {improvement:+.0%}")
    
    return improvement


def test_benchmark():
    """Run full benchmark and report."""
    results = run_introspection_benchmark(num_samples=5)
    improvement = print_benchmark_results(results)
    
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)
    print(f"\nINTROSPECTION improvement={improvement:+.0%}")
    
    for mode, result in results.items():
        print(f"  {mode}: sc={result.sc_rate:.0%}, json={result.json_rate:.0%}")
    
    print("=" * 70)
    
    return results, improvement


if __name__ == "__main__":
    test_benchmark()
