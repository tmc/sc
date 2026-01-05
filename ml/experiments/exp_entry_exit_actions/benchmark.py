"""
Entry/Exit Actions Benchmark with REAL MLX Inference.

Tests model's ability to generate statecharts with:
- Entry actions
- Exit actions
- Both entry and exit
- Transition actions
- Chained (multiple) actions
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from .action_grammar import (
    validate_sc_with_actions,
    get_few_shot_examples,
    ENTRY_ACTION_EXAMPLE,
    EXIT_ACTION_EXAMPLE,
    BOTH_ACTIONS_EXAMPLE,
    TRANSITION_ACTION_EXAMPLE,
    CHAINED_ACTIONS_EXAMPLE,
)


@dataclass
class TestCase:
    """A test case for action generation."""
    name: str
    prompt: str
    expected_type: str  # "entry", "exit", "both", "transition", "chained"
    expected_actions: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a single test."""
    test_case: TestCase
    generated: str
    parsed_sc: Optional[Dict]
    validation: Optional[Dict]
    correct: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results of the full benchmark."""
    entry_correct: int
    entry_total: int
    exit_correct: int
    exit_total: int
    both_correct: int
    both_total: int
    transition_correct: int
    transition_total: int
    chained_correct: int
    chained_total: int
    all_results: List[TestResult] = field(default_factory=list)

    @property
    def entry_pct(self) -> float:
        return self.entry_correct / self.entry_total * 100 if self.entry_total > 0 else 0

    @property
    def exit_pct(self) -> float:
        return self.exit_correct / self.exit_total * 100 if self.exit_total > 0 else 0

    @property
    def both_pct(self) -> float:
        return self.both_correct / self.both_total * 100 if self.both_total > 0 else 0

    @property
    def transition_pct(self) -> float:
        return self.transition_correct / self.transition_total * 100 if self.transition_total > 0 else 0

    @property
    def chained_pct(self) -> float:
        return self.chained_correct / self.chained_total * 100 if self.chained_total > 0 else 0


# Test cases for each action type
ENTRY_TEST_CASES = [
    TestCase(
        name="loading_spinner",
        prompt="Create a Loading statechart where entering the Loading state starts a spinner",
        expected_type="entry",
        expected_actions=["start_spinner()"]
    ),
    TestCase(
        name="init_log",
        prompt="Create an Init statechart where entering Init logs 'initialized'",
        expected_type="entry",
        expected_actions=["log(initialized)"]
    ),
    TestCase(
        name="connect_setup",
        prompt="Create a Connection statechart where entering Connected calls setup_connection()",
        expected_type="entry",
        expected_actions=["setup_connection()"]
    ),
    TestCase(
        name="audio_start",
        prompt="Create an Audio statechart where entering Playing starts playback",
        expected_type="entry",
        expected_actions=["start_playback()"]
    ),
]

EXIT_TEST_CASES = [
    TestCase(
        name="session_save",
        prompt="Create a Session statechart where leaving Active saves the session",
        expected_type="exit",
        expected_actions=["save_session()"]
    ),
    TestCase(
        name="cleanup",
        prompt="Create a Process statechart where leaving Running calls cleanup()",
        expected_type="exit",
        expected_actions=["cleanup()"]
    ),
    TestCase(
        name="timer_stop",
        prompt="Create a Timer statechart where leaving Running stops the timer",
        expected_type="exit",
        expected_actions=["stop_timer()"]
    ),
    TestCase(
        name="connection_close",
        prompt="Create a Connection statechart where leaving Connected closes the connection",
        expected_type="exit",
        expected_actions=["close_connection()"]
    ),
]

BOTH_TEST_CASES = [
    TestCase(
        name="timer_full",
        prompt="Create a Timer statechart where Running state starts timer on entry and stops on exit",
        expected_type="both",
        expected_actions=["start_timer()", "stop_timer()"]
    ),
    TestCase(
        name="session_full",
        prompt="Create a Session statechart where Active state loads on entry and saves on exit",
        expected_type="both",
        expected_actions=["load_session()", "save_session()"]
    ),
    TestCase(
        name="monitor_full",
        prompt="Create a Monitor statechart where Monitoring state begins monitoring on entry and ends on exit",
        expected_type="both",
        expected_actions=["begin_monitor()", "end_monitor()"]
    ),
]

TRANSITION_TEST_CASES = [
    TestCase(
        name="counter_inc",
        prompt="Create a Counter statechart with INCREMENT transition that calls add(1)",
        expected_type="transition",
        expected_actions=["add(1)"]
    ),
    TestCase(
        name="log_action",
        prompt="Create a State statechart where the NEXT transition logs the transition",
        expected_type="transition",
        expected_actions=["log(transition)"]
    ),
    TestCase(
        name="notify_action",
        prompt="Create a Process statechart where START transition sends a notification",
        expected_type="transition",
        expected_actions=["send_notification()"]
    ),
]

CHAINED_TEST_CASES = [
    TestCase(
        name="init_chain",
        prompt="Create an Init statechart where entering Ready calls init(), setup(), and notify()",
        expected_type="chained",
        expected_actions=["init()", "setup()", "notify()"]
    ),
    TestCase(
        name="shutdown_chain",
        prompt="Create a Shutdown statechart where exiting Running calls save(), cleanup(), and log(done)",
        expected_type="chained",
        expected_actions=["save()", "cleanup()", "log(done)"]
    ),
]


def extract_json_from_response(text: str) -> Optional[str]:
    """Extract JSON from model response (handles markdown blocks)."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)  # Only closing at end
    text = text.strip()

    # Find JSON object - greedy match for nested braces
    brace_count = 0
    start_idx = None
    end_idx = None

    for i, c in enumerate(text):
        if c == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                end_idx = i + 1
                break

    if start_idx is not None and end_idx is not None:
        return text[start_idx:end_idx]

    # Fallback: regex match
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group()
    return None


def create_prompt_with_examples(task: str) -> str:
    """Create few-shot prompt for action generation."""
    examples = get_few_shot_examples()

    return f'''Generate a statechart JSON with actions. Actions use on_entry, on_exit (in states) or action (in transitions).
Output ONLY valid JSON, no explanation.

{examples}
Task: {task}
Output:'''


def run_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    samples_per_test: int = 2,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run full benchmark with REAL MLX inference."""
    from mlx_lm import load, generate

    print("=" * 60)
    print("ENTRY/EXIT ACTIONS BENCHMARK (REAL MLX INFERENCE)")
    print("=" * 60)

    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    print("Model loaded.")

    result = BenchmarkResult(
        entry_correct=0, entry_total=0,
        exit_correct=0, exit_total=0,
        both_correct=0, both_total=0,
        transition_correct=0, transition_total=0,
        chained_correct=0, chained_total=0,
    )

    all_test_cases = [
        ("entry", ENTRY_TEST_CASES),
        ("exit", EXIT_TEST_CASES),
        ("both", BOTH_TEST_CASES),
        ("transition", TRANSITION_TEST_CASES),
        ("chained", CHAINED_TEST_CASES),
    ]

    for action_type, test_cases in all_test_cases:
        print(f"\n--- Testing {action_type.upper()} actions ---")

        for tc in test_cases:
            for sample_idx in range(samples_per_test):
                # Create prompt
                prompt = create_prompt_with_examples(tc.prompt)

                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate
                output = generate(
                    model, tokenizer,
                    prompt=formatted,
                    max_tokens=400,
                    verbose=False,
                )

                # Parse and validate
                json_str = extract_json_from_response(output)

                parsed_sc = None
                validation = None
                correct = False
                error = None

                if json_str:
                    try:
                        parsed_sc = json.loads(json_str)
                        validation = validate_sc_with_actions(parsed_sc)

                        # Check if correct action type was generated
                        if action_type == "entry":
                            correct = validation["entry_count"] > 0 or validation["both_count"] > 0
                        elif action_type == "exit":
                            correct = validation["exit_count"] > 0 or validation["both_count"] > 0
                        elif action_type == "both":
                            correct = validation["both_count"] > 0
                        elif action_type == "transition":
                            correct = validation["transition_action_count"] > 0
                        elif action_type == "chained":
                            correct = validation["chained_count"] > 0

                    except json.JSONDecodeError as e:
                        error = f"JSON parse error: {e}"
                else:
                    error = "No JSON found in output"

                # Record result
                test_result = TestResult(
                    test_case=tc,
                    generated=output[:200],
                    parsed_sc=parsed_sc,
                    validation=validation,
                    correct=correct,
                    error=error,
                )
                result.all_results.append(test_result)

                # Update counts
                if action_type == "entry":
                    result.entry_total += 1
                    if correct:
                        result.entry_correct += 1
                elif action_type == "exit":
                    result.exit_total += 1
                    if correct:
                        result.exit_correct += 1
                elif action_type == "both":
                    result.both_total += 1
                    if correct:
                        result.both_correct += 1
                elif action_type == "transition":
                    result.transition_total += 1
                    if correct:
                        result.transition_correct += 1
                elif action_type == "chained":
                    result.chained_total += 1
                    if correct:
                        result.chained_correct += 1

                if verbose:
                    status = "OK" if correct else "FAIL"
                    print(f"  {tc.name}[{sample_idx}]: {status}")
                    if error:
                        print(f"    Error: {error}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Entry only:  {result.entry_correct}/{result.entry_total} ({result.entry_pct:.0f}%)")
    print(f"Exit only:   {result.exit_correct}/{result.exit_total} ({result.exit_pct:.0f}%)")
    print(f"Both:        {result.both_correct}/{result.both_total} ({result.both_pct:.0f}%)")
    print(f"Transition:  {result.transition_correct}/{result.transition_total} ({result.transition_pct:.0f}%)")
    print(f"Chained:     {result.chained_correct}/{result.chained_total} ({result.chained_pct:.0f}%)")

    return result


def format_report(result: BenchmarkResult) -> str:
    """Format result for orchestrator report."""
    return (
        f"ENTRY_EXIT_ACTIONS entry={result.entry_pct:.0f}%, "
        f"exit={result.exit_pct:.0f}%, both={result.both_pct:.0f}%\n\n"
        f"Results:\n"
        f"- Entry only: {result.entry_correct}/{result.entry_total} correct\n"
        f"- Exit only: {result.exit_correct}/{result.exit_total} correct\n"
        f"- Both: {result.both_correct}/{result.both_total} correct\n"
        f"- Transition actions: {result.transition_correct}/{result.transition_total} correct\n"
        f"- Chained actions: {result.chained_correct}/{result.chained_total} correct"
    )


if __name__ == "__main__":
    result = run_benchmark(samples_per_test=2)
    print("\n" + format_report(result))
