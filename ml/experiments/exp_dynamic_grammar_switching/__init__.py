"""
exp_dynamic_grammar_switching: Test runtime grammar switching on real tasks

This experiment tests the meta_constrained_sampler's ability to switch
between different constraint grammars mid-generation for multi-modal tasks.

Example: JSON → Code → JSON transitions in a single generation.
"""

import json
import sys
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/grammars')

from meta_constrained_sampler import MetaConstrainedSampler


# Grammar SCs for different output modes
JSON_GRAMMAR_SC = {
    "name": "JSON_Grammar",
    "root_state": {
        "label": "__root__", "type": 2,
        "children": [
            {"label": "Start", "type": 1, "is_initial": True},
            {"label": "InObject", "type": 1},
            {"label": "InArray", "type": 1},
            {"label": "ExpectValue", "type": 1},
            {"label": "ExpectKey", "type": 1},
            {"label": "Complete", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Start"], "to": ["InObject"], "event": "LBRACE"},
        {"from": ["Start"], "to": ["InArray"], "event": "LBRACKET"},
        {"from": ["InObject"], "to": ["ExpectKey"], "event": "STRING_KEY"},
        {"from": ["ExpectKey"], "to": ["ExpectValue"], "event": "COLON"},
        {"from": ["ExpectValue"], "to": ["InObject"], "event": "STRING_VALUE"},
        {"from": ["ExpectValue"], "to": ["InObject"], "event": "NUMBER"},
        {"from": ["InObject"], "to": ["ExpectKey"], "event": "COMMA"},
        {"from": ["InObject"], "to": ["Complete"], "event": "RBRACE"},
        {"from": ["InArray"], "to": ["InArray"], "event": "VALUE"},
        {"from": ["InArray"], "to": ["InArray"], "event": "COMMA"},
        {"from": ["InArray"], "to": ["Complete"], "event": "RBRACKET"}
    ]
}

CODE_GRAMMAR_SC = {
    "name": "Code_Grammar",
    "root_state": {
        "label": "__root__", "type": 2,
        "children": [
            {"label": "Start", "type": 1, "is_initial": True},
            {"label": "InFunction", "type": 1},
            {"label": "InBlock", "type": 1},
            {"label": "InStatement", "type": 1},
            {"label": "Complete", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Start"], "to": ["InFunction"], "event": "DEF"},
        {"from": ["InFunction"], "to": ["InBlock"], "event": "COLON"},
        {"from": ["InBlock"], "to": ["InStatement"], "event": "INDENT"},
        {"from": ["InStatement"], "to": ["InStatement"], "event": "STATEMENT"},
        {"from": ["InStatement"], "to": ["InBlock"], "event": "NEWLINE"},
        {"from": ["InBlock"], "to": ["Complete"], "event": "DEDENT"},
        {"from": ["Start"], "to": ["InStatement"], "event": "STATEMENT"},
        {"from": ["InStatement"], "to": ["Complete"], "event": "EOF"}
    ]
}

MARKDOWN_GRAMMAR_SC = {
    "name": "Markdown_Grammar",
    "root_state": {
        "label": "__root__", "type": 2,
        "children": [
            {"label": "Start", "type": 1, "is_initial": True},
            {"label": "InHeader", "type": 1},
            {"label": "InParagraph", "type": 1},
            {"label": "InCodeBlock", "type": 1},
            {"label": "InList", "type": 1},
            {"label": "Complete", "type": 1}
        ]
    },
    "transitions": [
        {"from": ["Start"], "to": ["InHeader"], "event": "HASH"},
        {"from": ["InHeader"], "to": ["InParagraph"], "event": "NEWLINE"},
        {"from": ["Start"], "to": ["InParagraph"], "event": "TEXT"},
        {"from": ["InParagraph"], "to": ["InParagraph"], "event": "TEXT"},
        {"from": ["InParagraph"], "to": ["InCodeBlock"], "event": "BACKTICKS"},
        {"from": ["InCodeBlock"], "to": ["InCodeBlock"], "event": "CODE"},
        {"from": ["InCodeBlock"], "to": ["InParagraph"], "event": "BACKTICKS"},
        {"from": ["InParagraph"], "to": ["InList"], "event": "BULLET"},
        {"from": ["InList"], "to": ["InList"], "event": "BULLET"},
        {"from": ["InList"], "to": ["InParagraph"], "event": "NEWLINE"},
        {"from": ["InParagraph"], "to": ["Complete"], "event": "EOF"}
    ]
}


def test_json_code_json_switching():
    """Test switching between JSON and Code grammars."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("json", JSON_GRAMMAR_SC)
    sampler.register_sc("code", CODE_GRAMMAR_SC)

    trace = []

    # Start with JSON
    sampler.load_sc("json")
    trace.append({"mode": "json", "state": list(sampler.get_current_states())})

    # Generate some JSON tokens
    for event in ["LBRACE", "STRING_KEY", "COLON", "STRING_VALUE"]:
        sampler.emit_event(event)
        trace.append({"mode": "json", "event": event, "state": list(sampler.get_current_states())})

    # Switch to code (push JSON state)
    sampler.push_sc()
    sampler.load_sc("code")
    trace.append({"mode": "code", "action": "switched", "state": list(sampler.get_current_states())})

    # Generate some code tokens
    for event in ["DEF", "COLON", "INDENT", "STATEMENT"]:
        sampler.emit_event(event)
        trace.append({"mode": "code", "event": event, "state": list(sampler.get_current_states())})

    # Pop back to JSON
    sampler.pop_sc()
    trace.append({"mode": "json", "action": "restored", "state": list(sampler.get_current_states())})

    # Continue JSON
    for event in ["COMMA", "STRING_KEY", "COLON", "NUMBER", "RBRACE"]:
        sampler.emit_event(event)
        trace.append({"mode": "json", "event": event, "state": list(sampler.get_current_states())})

    return trace


def test_triple_nesting():
    """Test three levels of grammar nesting."""
    sampler = MetaConstrainedSampler()
    sampler.register_sc("json", JSON_GRAMMAR_SC)
    sampler.register_sc("code", CODE_GRAMMAR_SC)
    sampler.register_sc("markdown", MARKDOWN_GRAMMAR_SC)

    transitions = []

    # Level 0: Markdown
    sampler.load_sc("markdown")
    sampler.emit_event("HASH")  # Header
    sampler.emit_event("NEWLINE")
    transitions.append({"level": 0, "grammar": "markdown", "depth": sampler.sc_stack.depth()})

    # Level 1: Push, switch to JSON
    sampler.push_sc()
    sampler.load_sc("json")
    sampler.emit_event("LBRACE")
    transitions.append({"level": 1, "grammar": "json", "depth": sampler.sc_stack.depth()})

    # Level 2: Push, switch to code
    sampler.push_sc()
    sampler.load_sc("code")
    sampler.emit_event("DEF")
    transitions.append({"level": 2, "grammar": "code", "depth": sampler.sc_stack.depth()})

    # Pop back through all levels
    sampler.pop_sc()  # Back to JSON
    transitions.append({"level": 1, "grammar": "json_restored", "depth": sampler.sc_stack.depth()})

    sampler.pop_sc()  # Back to Markdown
    transitions.append({"level": 0, "grammar": "markdown_restored", "depth": sampler.sc_stack.depth()})

    return transitions


def measure_switching_overhead():
    """Measure computational overhead of grammar switching."""
    import time

    sampler = MetaConstrainedSampler()
    sampler.register_sc("json", JSON_GRAMMAR_SC)
    sampler.register_sc("code", CODE_GRAMMAR_SC)

    # Baseline: no switching
    sampler.load_sc("json")
    start = time.perf_counter()
    for _ in range(1000):
        sampler.emit_event("LBRACE")
        sampler.emit_event("RBRACE")
        sampler.config = sampler.executor.initial_config()  # Reset
    baseline_time = time.perf_counter() - start

    # With switching
    start = time.perf_counter()
    for _ in range(500):
        sampler.load_sc("json")
        sampler.emit_event("LBRACE")
        sampler.push_sc()
        sampler.load_sc("code")
        sampler.emit_event("DEF")
        sampler.pop_sc()
        sampler.emit_event("RBRACE")
    switching_time = time.perf_counter() - start

    return {
        "baseline_ops_per_sec": 1000 / baseline_time,
        "switching_ops_per_sec": 500 / switching_time,
        "overhead_ratio": (switching_time / 500) / (baseline_time / 1000)
    }


def benchmark():
    """Run dynamic grammar switching benchmark."""
    print("Dynamic Grammar Switching Experiment")
    print("=" * 50)

    # Test 1: JSON-Code-JSON
    print("\n1. JSON → Code → JSON switching:")
    trace = test_json_code_json_switching()
    modes = [t.get("mode") for t in trace if "mode" in t]
    print(f"   Transitions: {len(trace)}")
    print(f"   Mode sequence: {' → '.join(sorted(set(modes)))}")

    # Test 2: Triple nesting
    print("\n2. Triple nesting (Markdown → JSON → Code):")
    transitions = test_triple_nesting()
    for t in transitions:
        print(f"   Level {t['level']}: {t['grammar']} (depth={t['depth']})")

    # Test 3: Overhead measurement
    print("\n3. Switching overhead:")
    overhead = measure_switching_overhead()
    print(f"   Baseline: {overhead['baseline_ops_per_sec']:.0f} ops/sec")
    print(f"   With switching: {overhead['switching_ops_per_sec']:.0f} ops/sec")
    print(f"   Overhead ratio: {overhead['overhead_ratio']:.2f}x")

    return {
        "json_code_json_trace": trace,
        "triple_nesting": transitions,
        "overhead": overhead
    }


if __name__ == "__main__":
    benchmark()
