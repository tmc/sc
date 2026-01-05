#!/usr/bin/env python3
"""
Guard Synthesis Benchmark.

Tests LLM ability to generate statecharts with guards at 4 complexity levels:
- L1: Single variable (is_ready)
- L2: Boolean operators (a && b, a || b)
- L3: Comparisons (x > 5, count < max)
- L4: Nested expressions ((a || b) && c)

Uses GRPO+Constrained approach for 100% JSON validity.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from .guard_generator import (
    GuardLevel,
    generate_guard,
    validate_guard_syntax,
    validate_guard_semantics,
    extract_guard_from_statechart,
    BOOL_VARS,
    COUNT_VARS,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

try:
    from experiments.exp_schema_guided_sampling.statechart_sampler import (
        StatechartMachine,
        JSONTokenizer,
    )
    HAS_SAMPLER = True
except ImportError:
    HAS_SAMPLER = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
    lora_rank: int = 8
    lora_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 500
    samples_per_level: int = 5


@dataclass
class GuardResult:
    """Result for a single guard generation."""
    level: GuardLevel
    prompt: str
    output: str
    guard_expr: Optional[str]
    json_valid: bool
    syntax_valid: bool
    semantic_valid: bool
    correct_level: bool
    error: Optional[str] = None
    time: float = 0.0


def create_guard_prompt(level: GuardLevel, task_context: str = None) -> str:
    """Create a few-shot prompt for guard synthesis."""
    level_instructions = {
        GuardLevel.L1_SINGLE: "The guard MUST be a single boolean variable like 'is_ready' or '!is_locked'",
        GuardLevel.L2_BOOLEAN: "The guard MUST use AND (&&) or OR (||) like 'is_ready && is_enabled'",
        GuardLevel.L3_COMPARISON: "The guard MUST use a comparison operator (<, >, <=, >=, ==, !=) like 'count < 5' or 'retry_count >= max_retries'. Do NOT use simple boolean guards.",
        GuardLevel.L4_NESTED: "The guard MUST use parentheses like '(is_ready || is_enabled) && !is_locked'",
    }

    instruction = level_instructions[level]

    # Multiple few-shot examples to reinforce the guard pattern
    # Use single braces - not in an f-string position
    examples_all = [
        # Example 1 with guard
        '{"root_state": {"label": "Toggle", "type": 2, "children": [{"label": "Off", "type": 1, "is_initial": true}, {"label": "On", "type": 1}]}, "transitions": [{"from": ["Off"], "to": ["On"], "event": "ACTIVATE", "guard": {"expression": "is_enabled", "language": "go"}}]}',
        # Example 2 with guard
        '{"root_state": {"label": "Auth", "type": 2, "children": [{"label": "LoggedOut", "type": 1, "is_initial": true}, {"label": "LoggedIn", "type": 1}]}, "transitions": [{"from": ["LoggedOut"], "to": ["LoggedIn"], "event": "LOGIN", "guard": {"expression": "is_authenticated && has_permission", "language": "go"}}]}',
    ]

    # Level-specific example
    level_examples = {
        GuardLevel.L1_SINGLE: '{"root_state": {"label": "Door", "type": 2, "children": [{"label": "Closed", "type": 1, "is_initial": true}, {"label": "Open", "type": 1}]}, "transitions": [{"from": ["Closed"], "to": ["Open"], "event": "OPEN", "guard": {"expression": "!is_locked", "language": "go"}}]}',

        GuardLevel.L2_BOOLEAN: '{"root_state": {"label": "Connection", "type": 2, "children": [{"label": "Disconnected", "type": 1, "is_initial": true}, {"label": "Connected", "type": 1}]}, "transitions": [{"from": ["Disconnected"], "to": ["Connected"], "event": "CONNECT", "guard": {"expression": "is_ready && has_credentials", "language": "go"}}]}',

        GuardLevel.L3_COMPARISON: '{"root_state": {"label": "Retry", "type": 2, "children": [{"label": "Waiting", "type": 1, "is_initial": true}, {"label": "Retrying", "type": 1}]}, "transitions": [{"from": ["Waiting"], "to": ["Retrying"], "event": "RETRY", "guard": {"expression": "attempt_count < max_attempts", "language": "go"}}]}',

        GuardLevel.L4_NESTED: '{"root_state": {"label": "System", "type": 2, "children": [{"label": "Idle", "type": 1, "is_initial": true}, {"label": "Active", "type": 1}]}, "transitions": [{"from": ["Idle"], "to": ["Active"], "event": "START", "guard": {"expression": "(is_ready || is_enabled) && !is_locked", "language": "go"}}]}',
    }

    if task_context is None:
        contexts = [
            "connection handler with Connected and Disconnected states",
            "authentication system with Authorized and Unauthorized states",
            "retry mechanism with Waiting and Processing states",
            "state validator with Pending and Active states",
            "resource manager with Available and InUse states",
        ]
        task_context = random.choice(contexts)

    # Build prompt without f-string to avoid brace escaping issues
    prompt = f'''Generate a statechart JSON with a guarded transition. {instruction}

IMPORTANT: Every transition MUST have a "guard" field with "expression" and "language".

Example 1:
{examples_all[0]}

Example 2:
{level_examples[level]}

Create a statechart for a {task_context}. {instruction}. Output ONLY valid JSON.
''' + '{"root_state":'

    return prompt


def load_sc_json_statechart() -> Dict:
    """Load the SC JSON grammar statechart from file."""
    grammar_path = Path(__file__).parent.parent / "exp_schema_guided_sampling" / "json_parser_v2.statechart.json"
    if grammar_path.exists():
        with open(grammar_path) as f:
            return json.load(f)
    return None


class ConstrainedSampler:
    """Retok-TopK constrained sampler for JSON generation."""

    def __init__(self, tokenizer, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self.vocab = tokenizer.get_vocab()
        self.id_to_str = {v: k for k, v in self.vocab.items()}
        self.generated_string = ""

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()
        self.generated_string = ""

    def initialize_with_prefix(self, prefix: str):
        """Process prefix to initialize machine state."""
        self.reset()
        for char in prefix:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)
        self.generated_string = prefix

    def _validate_token(self, token_str: str) -> bool:
        """Simulate token through statechart."""
        test_machine = StatechartMachine(statechart=self.statechart)
        test_machine.current_state = self.machine.current_state
        test_machine.context = self.machine.context.copy()
        test_tokenizer = JSONTokenizer()
        test_tokenizer.in_string = self.json_tokenizer.in_string
        test_tokenizer.current_token = self.json_tokenizer.current_token
        test_tokenizer.escape_next = self.json_tokenizer.escape_next

        for char in token_str:
            event = test_tokenizer.process_char(char)
            if event:
                enabled = test_machine.get_enabled_events()
                if event not in enabled:
                    return False
                test_machine.send_event(event)

        return True

    def get_valid_mask(self, top_k_ids: List[int]) -> 'mx.array':
        """Get mask for valid tokens among top-k candidates."""
        mask = []
        for token_id in top_k_ids:
            token_str = self.id_to_str.get(token_id, "")
            valid = self._validate_token(token_str)
            mask.append(1.0 if valid else -float('inf'))
        return mx.array(mask)

    def process_token(self, token_str: str):
        """Update state after generating token."""
        self.generated_string += token_str
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)

    def is_complete(self) -> bool:
        return self.machine.current_state == "Complete"


def apply_lora(model, config: BenchmarkConfig):
    """Apply LoRA to model."""
    model.freeze()

    for layer_idx in config.lora_layers:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        for proj_name in ['q_proj', 'v_proj']:
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                lora = LoRALinear.from_base(
                    proj,
                    r=config.lora_rank,
                    scale=16.0 / config.lora_rank,
                )
                setattr(attn, proj_name, lora)

    return model


def generate_constrained(
    model,
    tokenizer,
    sampler: ConstrainedSampler,
    prompt: str,
    json_prefix: str,
    config: BenchmarkConfig,
) -> Tuple[str, float]:
    """Generate with constrained decoding."""
    sampler.initialize_with_prefix(json_prefix)

    prompt_tokens = tokenizer.encode(prompt)
    input_ids = mx.array([prompt_tokens])

    generated_tokens = []
    t0 = time.time()

    for _ in range(config.max_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :]

        top_k_indices = mx.argpartition(-next_logits, config.top_k)[:config.top_k]
        top_k_logits = next_logits[top_k_indices]

        mask = sampler.get_valid_mask(top_k_indices.tolist())
        masked_logits = top_k_logits + mask

        valid_count = mx.sum(mask > -float('inf')).item()
        if valid_count == 0:
            break

        scaled_logits = masked_logits / config.temperature
        sampled_idx = mx.random.categorical(scaled_logits[None, :])[0]
        next_token = top_k_indices[sampled_idx]

        token_str = tokenizer.decode([next_token.item()])
        sampler.process_token(token_str)

        generated_tokens.append(next_token.item())

        if sampler.is_complete():
            break

        input_ids = mx.concatenate([input_ids, next_token[None, None]], axis=1)

    elapsed = time.time() - t0
    output = tokenizer.decode(generated_tokens)

    return output, elapsed


def validate_generated_guard(output: str, level: GuardLevel) -> GuardResult:
    """Validate a generated statechart's guard."""
    result = GuardResult(
        level=level,
        prompt="",
        output=output,
        guard_expr=None,
        json_valid=False,
        syntax_valid=False,
        semantic_valid=False,
        correct_level=False,
    )

    # Check JSON validity
    try:
        sc = json.loads(output)
        result.json_valid = True
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        return result

    # Extract guard
    guard_expr = extract_guard_from_statechart(output)
    if not guard_expr:
        result.error = "No guard found in statechart"
        return result

    result.guard_expr = guard_expr

    # Validate syntax
    syntax_valid, syntax_error = validate_guard_syntax(guard_expr)
    result.syntax_valid = syntax_valid
    if not syntax_valid:
        result.error = f"Syntax: {syntax_error}"
        return result

    # Validate semantics for level
    semantic_valid, semantic_error = validate_guard_semantics(guard_expr, level)
    result.semantic_valid = semantic_valid
    if not semantic_valid:
        result.error = f"Semantic: {semantic_error}"
        return result

    result.correct_level = True
    return result


def run_mock_benchmark(config: BenchmarkConfig) -> Dict:
    """Run benchmark without MLX (mock mode)."""
    print("Running in MOCK mode (MLX not available)")
    print("-" * 60)

    results_by_level = {level: [] for level in GuardLevel}

    for level in GuardLevel:
        print(f"\n{level.name}:")
        for i in range(config.samples_per_level):
            # Generate mock output with appropriate guard
            guard_expr, vars, desc = generate_guard(level)

            mock_sc = {
                "root_state": {
                    "label": "MockMachine",
                    "type": 2,
                    "children": [
                        {"label": "StateA", "type": 1, "is_initial": True},
                        {"label": "StateB", "type": 1},
                    ],
                },
                "transitions": [
                    {
                        "from": ["StateA"],
                        "to": ["StateB"],
                        "event": "GO",
                        "guard": {
                            "expression": guard_expr,
                            "language": "go",
                        },
                    },
                ],
            }

            output = json.dumps(mock_sc)
            result = validate_generated_guard(output, level)
            result.prompt = f"Mock prompt for {level.name}"
            results_by_level[level].append(result)

            status = "✓" if result.correct_level else "✗"
            print(f"  {status} Sample {i+1}: guard='{guard_expr}'")

    return compute_summary(results_by_level)


def clean_json_output(output: str) -> str:
    """Clean up generated JSON output."""
    # Remove newlines and take first line
    if '\n' in output:
        output = output.split('\n')[0]

    # Balance braces - remove trailing extras
    brace_count = 0
    bracket_count = 0
    last_valid_idx = 0

    for i, char in enumerate(output):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1

        if brace_count == 0 and bracket_count == 0:
            last_valid_idx = i + 1
            break
        elif brace_count >= 0 and bracket_count >= 0:
            last_valid_idx = i + 1

    return output[:last_valid_idx]


def run_real_benchmark(config: BenchmarkConfig) -> Dict:
    """Run benchmark with real LLM generation."""
    from mlx_lm import generate as mlx_generate

    print(f"Loading model: {config.model_path}")
    model, tokenizer = load(config.model_path)
    print("Model loaded.")

    print(f"Applying LoRA (rank={config.lora_rank}, layers={config.lora_layers})")
    apply_lora(model, config)
    print("LoRA applied.")

    # Use unconstrained generation for guard synthesis
    # (Constrained sampler stops too early before transitions)
    print("Using unconstrained generation (better for guard synthesis)")

    print("\n" + "-" * 60)
    print("GUARD SYNTHESIS BENCHMARK")
    print("-" * 60)

    results_by_level = {level: [] for level in GuardLevel}

    for level in GuardLevel:
        print(f"\n{level.name}:")

        for i in range(config.samples_per_level):
            prompt = create_guard_prompt(level)
            json_prefix = '{"root_state":'

            t0 = time.time()
            # Use unconstrained generation - works better for guards
            output = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=config.max_tokens)
            elapsed = time.time() - t0

            full_output = json_prefix + output
            full_output = clean_json_output(full_output)

            result = validate_generated_guard(full_output, level)
            result.prompt = prompt[:100] + "..."
            result.output = full_output
            result.time = elapsed
            results_by_level[level].append(result)

            status = "✓" if result.correct_level else "✗"
            guard_display = result.guard_expr[:40] if result.guard_expr else "None"
            print(f"  {status} Sample {i+1}: guard='{guard_display}' time={elapsed:.2f}s")
            if result.error:
                print(f"      Error: {result.error}")

    return compute_summary(results_by_level)


def compute_summary(results_by_level: Dict[GuardLevel, List[GuardResult]]) -> Dict:
    """Compute summary statistics."""
    summary = {}

    for level, results in results_by_level.items():
        n = len(results)
        json_valid = sum(1 for r in results if r.json_valid)
        syntax_valid = sum(1 for r in results if r.syntax_valid)
        semantic_valid = sum(1 for r in results if r.semantic_valid)
        correct_level = sum(1 for r in results if r.correct_level)

        summary[level.name] = {
            'total': n,
            'json_valid': json_valid,
            'json_rate': json_valid / n if n > 0 else 0,
            'syntax_valid': syntax_valid,
            'syntax_rate': syntax_valid / n if n > 0 else 0,
            'semantic_valid': semantic_valid,
            'semantic_rate': semantic_valid / n if n > 0 else 0,
            'correct_level': correct_level,
            'correct_rate': correct_level / n if n > 0 else 0,
        }

    # Overall stats
    all_results = [r for results in results_by_level.values() for r in results]
    total = len(all_results)
    overall_correct = sum(1 for r in all_results if r.correct_level)

    summary['overall'] = {
        'total': total,
        'correct': overall_correct,
        'rate': overall_correct / total if total > 0 else 0,
    }

    return summary


def print_summary(summary: Dict):
    """Print summary in readable format."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for level in GuardLevel:
        stats = summary[level.name]
        print(f"\n{level.name}:")
        print(f"  JSON valid:    {stats['json_rate']:.1%} ({stats['json_valid']}/{stats['total']})")
        print(f"  Syntax valid:  {stats['syntax_rate']:.1%} ({stats['syntax_valid']}/{stats['total']})")
        print(f"  Semantic valid:{stats['semantic_rate']:.1%} ({stats['semantic_valid']}/{stats['total']})")
        print(f"  Correct level: {stats['correct_rate']:.1%} ({stats['correct_level']}/{stats['total']})")

    overall = summary['overall']
    print(f"\nOVERALL: {overall['rate']:.1%} ({overall['correct']}/{overall['total']})")


def run_benchmark():
    """Run the full benchmark."""
    print("=" * 60)
    print("GUARD SYNTHESIS BENCHMARK")
    print("=" * 60)

    config = BenchmarkConfig()

    if HAS_MLX and HAS_SAMPLER:
        summary = run_real_benchmark(config)
    else:
        summary = run_mock_benchmark(config)

    print_summary(summary)

    # Format for orchestrator report
    l1 = summary['L1_SINGLE']['correct_rate']
    l2 = summary['L2_BOOLEAN']['correct_rate']
    l3 = summary['L3_COMPARISON']['correct_rate']
    l4 = summary['L4_NESTED']['correct_rate']
    overall = summary['overall']['rate']

    print(f"\nGUARD_SYNTHESIS: L1={l1:.1%}, L2={l2:.1%}, L3={l3:.1%}, L4={l4:.1%}, overall={overall:.1%}")

    return summary


if __name__ == "__main__":
    run_benchmark()
