#!/usr/bin/env python3
"""
SC Completion Benchmark - Real MLX Model Testing

Tests completing partial statecharts with missing transitions.
Uses Retok-TopK constrained decoding with real model inference.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

# Import constrained decoding approach
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.exp_schema_guided_sampling.approaches import RetokenizationApproach, check_balanced
from experiments.exp_schema_guided_sampling.statechart_sampler import JSONTokenizer


@dataclass
class CompletionResult:
    """Result of completing a partial SC."""
    partial_sc: str
    completed_sc: str
    valid_json: bool
    valid_structure: bool
    states_found: int
    transitions_found: int
    gen_time_s: float
    tokens_generated: int


# Partial statecharts to complete (missing transitions)
# Include example transition to guide format
PARTIAL_STATECHARTS = [
    # 1. Toggle - has one transition, need reverse
    {
        "partial": '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Off", "type": 1, "is_initial": true}, {"label": "On", "type": 1}]}, "transitions": [{"from": ["Off"], "to": ["On"], "event": "TOGGLE"}',
        "expected_min_transitions": 2,
        "description": "Toggle machine - need reverse TOGGLE"
    },

    # 2. Traffic light - has first transition
    {
        "partial": '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Red", "type": 1, "is_initial": true}, {"label": "Yellow", "type": 1}, {"label": "Green", "type": 1}]}, "transitions": [{"from": ["Red"], "to": ["Yellow"], "event": "NEXT"}',
        "expected_min_transitions": 2,
        "description": "Traffic light - need more cycle transitions"
    },

    # 3. Door FSM - has OPEN, need CLOSE
    {
        "partial": '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Closed", "type": 1, "is_initial": true}, {"label": "Open", "type": 1}]}, "transitions": [{"from": ["Closed"], "to": ["Open"], "event": "OPEN"}',
        "expected_min_transitions": 2,
        "description": "Door FSM - need CLOSE transition"
    },

    # 4. Loading states - has FETCH, need completion transitions
    {
        "partial": '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle", "type": 1, "is_initial": true}, {"label": "Loading", "type": 1}, {"label": "Done", "type": 1}]}, "transitions": [{"from": ["Idle"], "to": ["Loading"], "event": "FETCH"}',
        "expected_min_transitions": 2,
        "description": "Async states - need COMPLETE transition"
    },

    # 5. Counter - has INCREMENT
    {
        "partial": '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Zero", "type": 1, "is_initial": true}, {"label": "Positive", "type": 1}]}, "transitions": [{"from": ["Zero"], "to": ["Positive"], "event": "INCREMENT"}',
        "expected_min_transitions": 1,
        "description": "Counter - complete structure"
    },
]


def validate_statechart(sc_json: str) -> Tuple[bool, int, int]:
    """Validate statechart structure and count states/transitions."""
    try:
        data = json.loads(sc_json)

        # Check required fields
        if "root_state" not in data:
            return False, 0, 0

        # Count states
        def count_states(state: Dict) -> int:
            count = 1
            for child in state.get("children", []):
                count += count_states(child)
            return count

        states = count_states(data["root_state"])

        # Collect all state labels
        all_labels = set()
        def collect_labels(state: Dict):
            all_labels.add(state.get("label", ""))
            for child in state.get("children", []):
                collect_labels(child)
        collect_labels(data["root_state"])

        # Count valid transitions (must have from, to arrays)
        valid_transitions = 0
        for trans in data.get("transitions", []):
            # Check transition format
            if not isinstance(trans, dict):
                continue
            if "from" not in trans or "to" not in trans:
                continue
            if not isinstance(trans.get("from"), list) or not isinstance(trans.get("to"), list):
                continue

            # Validate state references
            valid = True
            for src in trans.get("from", []):
                if src not in all_labels:
                    valid = False
                    break
            for tgt in trans.get("to", []):
                if tgt not in all_labels:
                    valid = False
                    break

            if valid:
                valid_transitions += 1

        return True, states, valid_transitions

    except (json.JSONDecodeError, KeyError, TypeError):
        return False, 0, 0


def complete_statechart(
    model,
    tokenizer,
    partial: str,
    max_tokens: int = 100,
    top_k: int = 500,
) -> CompletionResult:
    """Complete a partial statechart using constrained decoding."""

    # Load JSON parser statechart
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    # Initialize approach
    approach = RetokenizationApproach(tokenizer, sc_def)
    approach.reset()

    # Process partial through machine
    json_tok = JSONTokenizer()
    for char in partial:
        event = json_tok.process_char(char)
        if event:
            approach.machine.send_event(event)
    approach.generated_string = partial
    approach.json_tokenizer = json_tok

    # Setup generation
    tokens = mx.array(tokenizer.encode(partial))[None]
    sampler = make_sampler(temp=0.3)
    generated = []

    t0 = time.time()

    for _ in range(max_tokens):
        logits = model(tokens)
        next_logits = logits[:, -1, :]

        # Get valid tokens (Retok-TopK)
        top_k_indices = mx.argpartition(next_logits.squeeze(), -top_k)[-top_k:]
        top_k_ids = [int(i) for i in top_k_indices.tolist()]
        valid = approach.get_valid_tokens(top_k_ids)

        # Apply mask
        if valid:
            vocab_size = next_logits.shape[-1]
            import numpy as np
            mask_np = np.full(vocab_size, -100.0)
            for tid in valid:
                if tid < vocab_size:
                    mask_np[tid] = 0.0
            next_logits = next_logits + mx.array(mask_np)

        # Sample
        next_token = sampler(next_logits)
        token_id = next_token.item()

        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        token_str = tokenizer.decode([token_id])
        approach.process_token(token_str)

        # Check completion
        if approach.machine.is_complete():
            break

        if approach.machine.should_force_close():
            current = tokenizer.decode(generated)
            if current.rstrip().endswith(','):
                current = current.rstrip().rstrip(',')
                generated = list(tokenizer.encode(current))

            closing = ""
            for item in reversed(approach.machine.context.stack):
                if item == "object":
                    closing += "}"
                elif item == "array":
                    closing += "]"
            if closing:
                closing_tokens = tokenizer.encode(closing)
                generated.extend(closing_tokens)
            break

        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

    gen_time = time.time() - t0

    # Build complete SC
    output = tokenizer.decode(generated)
    completed = partial + output

    # Validate
    valid_json = False
    try:
        json.loads(completed)
        valid_json = True
    except json.JSONDecodeError:
        pass

    valid_structure, states, transitions = validate_statechart(completed)

    return CompletionResult(
        partial_sc=partial[:50] + "...",
        completed_sc=completed,
        valid_json=valid_json,
        valid_structure=valid_structure,
        states_found=states,
        transitions_found=transitions,
        gen_time_s=gen_time,
        tokens_generated=len(generated),
    )


def run_benchmark(model_id: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit") -> Dict:
    """Run SC completion benchmark."""
    print("=" * 70)
    print("SC COMPLETION BENCHMARK - Real MLX Model")
    print("=" * 70)
    print(f"Model: {model_id}")

    print("\nLoading model...")
    model, tokenizer = load(model_id)
    print("Model loaded.\n")

    results = []

    for i, test in enumerate(PARTIAL_STATECHARTS, 1):
        print(f"\n{i}. {test['description']}")

        result = complete_statechart(
            model=model,
            tokenizer=tokenizer,
            partial=test["partial"],
            max_tokens=100,
        )

        # Check if meets minimum transitions
        meets_min = result.transitions_found >= test["expected_min_transitions"]

        status = "PASS" if result.valid_json and result.valid_structure else "FAIL"
        print(f"   Status: {status}")
        print(f"   Valid JSON: {result.valid_json}")
        print(f"   Valid Structure: {result.valid_structure}")
        print(f"   States: {result.states_found}, Transitions: {result.transitions_found} (min: {test['expected_min_transitions']})")
        print(f"   Tokens: {result.tokens_generated}, Time: {result.gen_time_s:.2f}s")

        results.append({
            "description": test["description"],
            "valid_json": result.valid_json,
            "valid_structure": result.valid_structure,
            "states": result.states_found,
            "transitions": result.transitions_found,
            "meets_min": meets_min,
            "completed": result.completed_sc[:200],
        })

    # Summary
    valid_count = sum(1 for r in results if r["valid_json"] and r["valid_structure"])
    accuracy = valid_count / len(results) * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Valid completions: {valid_count}/{len(results)}")
    print(f"Accuracy: {accuracy:.0f}%")

    report = f"SC_COMPLETION accuracy={accuracy:.0f}%"
    print(f"\nReport: {report}")

    return {
        "accuracy": accuracy,
        "valid_count": valid_count,
        "total": len(results),
        "results": results,
        "report": report,
    }


if __name__ == "__main__":
    run_benchmark()
