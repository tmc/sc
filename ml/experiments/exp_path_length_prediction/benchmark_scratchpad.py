#!/usr/bin/env python3
"""
Benchmark: Compare scratchpad approaches for path length prediction.

Compares:
1. Baseline: Direct prediction (12% baseline)
2. Scratchpad: Explicit step counting
3. BFS simulation: Teach BFS in text
4. ASCII visualization: Graph structure in prompt
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .scratchpad_predictor import (
    PromptStyle,
    StatechartPath,
    ScratchpadPredictor,
    create_scratchpad_few_shot,
    create_bfs_simulation_prompt,
    create_ascii_visualization_prompt,
    parse_path_length,
    get_test_cases,
)
from .path_visualizer import create_ascii_graph


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    max_tokens: int = 300
    test_larger_model: bool = False  # Also test 3B model


@dataclass
class StyleResult:
    """Results for a single prompting style."""
    style: PromptStyle
    correct: int
    total: int
    within_one: int
    predictions: List[Dict]


@dataclass
class BenchmarkResult:
    """Full benchmark results."""
    results_by_style: Dict[PromptStyle, StyleResult]
    model_path: str


def run_benchmark(config: BenchmarkConfig = None) -> BenchmarkResult:
    """Run benchmark comparing all prompting styles."""
    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("PATH LENGTH SCRATCHPAD BENCHMARK")
    print("=" * 60)
    print(f"Model: {config.model_path}")

    # Load model
    print("\nLoading model...")
    from mlx_lm import load, generate as mlx_generate

    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    # Get test cases
    test_cases = get_test_cases()
    print(f"\nTest cases: {len(test_cases)}")
    for i, tc in enumerate(test_cases):
        print(f"  {i+1}. {tc.initial} → {tc.terminal}: expected={tc.expected_length}")

    # Styles to test
    styles = [
        PromptStyle.BASELINE,
        PromptStyle.SCRATCHPAD,
        PromptStyle.BFS_SIMULATION,
        PromptStyle.ASCII_VISUAL,
    ]

    results_by_style = {}

    for style in styles:
        print(f"\n{'=' * 60}")
        print(f"Testing: {style.value.upper()}")
        print("-" * 60)

        correct = 0
        within_one = 0
        predictions = []

        for tc in test_cases:
            # Create prompt based on style
            if style == PromptStyle.BASELINE:
                prompt = _create_baseline_prompt(tc)
            elif style == PromptStyle.SCRATCHPAD:
                prompt = create_scratchpad_few_shot(tc)
            elif style == PromptStyle.BFS_SIMULATION:
                prompt = create_bfs_simulation_prompt(tc)
            elif style == PromptStyle.ASCII_VISUAL:
                ascii_graph = create_ascii_graph(tc)
                prompt = create_ascii_visualization_prompt(tc, ascii_graph)
            else:
                prompt = create_scratchpad_few_shot(tc)

            # Generate
            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
            )

            # Parse
            predicted = parse_path_length(output)
            expected = tc.expected_length

            is_correct = (predicted == expected)
            is_within_one = abs(predicted - expected) <= 1 if predicted >= 0 and expected >= 0 else (predicted == expected)

            if is_correct:
                correct += 1
            if is_within_one:
                within_one += 1

            predictions.append({
                "case": f"{tc.initial}→{tc.terminal}",
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "output_preview": output[:100],
            })

            status = "✓" if is_correct else "✗"
            print(f"  {status} {tc.initial}→{tc.terminal}: expected={expected}, predicted={predicted}")

        accuracy = correct / len(test_cases) * 100
        within_one_acc = within_one / len(test_cases) * 100

        results_by_style[style] = StyleResult(
            style=style,
            correct=correct,
            total=len(test_cases),
            within_one=within_one,
            predictions=predictions,
        )

        print(f"\n  Accuracy: {accuracy:.0f}% ({correct}/{len(test_cases)})")
        print(f"  Within ±1: {within_one_acc:.0f}%")

    return BenchmarkResult(
        results_by_style=results_by_style,
        model_path=config.model_path,
    )


def _create_baseline_prompt(sc: StatechartPath) -> str:
    """Simple baseline prompt without scratchpad."""
    trans_str = []
    for t in sc.transitions:
        trans_str.append(f"{t['from']} → {t['to']}")

    context_str = ""
    if sc.context:
        context_str = f"\nContext: {sc.context}"

    return f"""How many transitions to reach terminal state?

Transitions: {', '.join(trans_str)}
Initial: {sc.initial}
Terminal: {', '.join(sc.terminal)}{context_str}

Answer (number or 'infinite'):"""


def format_report(result: BenchmarkResult) -> str:
    """Format results for reporting."""
    lines = []
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)

    for style, sr in result.results_by_style.items():
        acc = sr.correct / sr.total * 100
        within_one = sr.within_one / sr.total * 100
        lines.append(f"{style.value:15s}: {acc:5.0f}% (within±1: {within_one:.0f}%)")

    # Format for report
    baseline_acc = result.results_by_style[PromptStyle.BASELINE].correct / result.results_by_style[PromptStyle.BASELINE].total * 100
    scratchpad_acc = result.results_by_style[PromptStyle.SCRATCHPAD].correct / result.results_by_style[PromptStyle.SCRATCHPAD].total * 100
    bfs_acc = result.results_by_style[PromptStyle.BFS_SIMULATION].correct / result.results_by_style[PromptStyle.BFS_SIMULATION].total * 100
    ascii_acc = result.results_by_style[PromptStyle.ASCII_VISUAL].correct / result.results_by_style[PromptStyle.ASCII_VISUAL].total * 100

    lines.append("")
    lines.append(f"PATH_SCRATCHPAD baseline={baseline_acc:.0f}%, scratchpad={scratchpad_acc:.0f}%, bfs_sim={bfs_acc:.0f}%, ascii_viz={ascii_acc:.0f}%")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_benchmark()
    print("\n" + format_report(result))
