#!/usr/bin/env python3
"""
Scratchpad-based Path Length Prediction.

Uses explicit step-by-step reasoning to help LLMs predict path lengths.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import re


class PromptStyle(Enum):
    """Different prompting strategies."""
    BASELINE = "baseline"           # Direct prediction (12% baseline)
    SCRATCHPAD = "scratchpad"       # Explicit step counting
    BFS_SIMULATION = "bfs_sim"      # Teach BFS in text
    ASCII_VISUAL = "ascii_viz"      # ASCII graph visualization
    DECOMPOSITION = "decompose"     # Break into segments


@dataclass
class StatechartPath:
    """A statechart with path to analyze."""
    states: List[str]
    transitions: List[Dict[str, Any]]  # {from, to, event, guard?}
    initial: str
    terminal: List[str]
    context: Dict[str, Any] = None
    expected_length: int = -1  # -1 for infinite


def create_scratchpad_prompt(sc: StatechartPath) -> str:
    """Create prompt with explicit scratchpad reasoning template."""

    # Format transitions as arrows
    trans_str = []
    for t in sc.transitions:
        arrow = f"{t['from']} --{t.get('event', 'e')}"
        if t.get('guard'):
            arrow += f"[{t['guard']}]"
        arrow += f"--> {t['to']}"
        trans_str.append(arrow)

    transitions = "\n".join(f"  {t}" for t in trans_str)
    terminals = ", ".join(sc.terminal)

    context_str = ""
    if sc.context:
        context_str = f"\nInitial context: {sc.context}"

    prompt = f"""Count the steps from initial state to terminal state.

Statechart:
{transitions}

Initial state: {sc.initial}
Terminal states: {terminals}{context_str}

Let me trace the path step by step:
- Start at {sc.initial}"""

    return prompt


def create_scratchpad_few_shot(sc: StatechartPath) -> str:
    """Few-shot prompt with scratchpad examples."""

    examples = """Example 1:
Statechart:
  A --e1--> B
  B --e2--> C
  C --e3--> D
Initial: A, Terminal: D

Trace:
- Start at A
- Step 1: A --e1--> B (now at B)
- Step 2: B --e2--> C (now at C)
- Step 3: C --e3--> D (now at D, terminal!)
Path length: 3

Example 2:
Statechart:
  Start --BEGIN--> Running
  Running --TICK[count<3]--> Running (count++)
  Running --TICK[count>=3]--> Done
Initial: Start, Terminal: Done, Context: {count: 0}

Trace:
- Start at Start
- Step 1: Start --BEGIN--> Running (now at Running, count=0)
- Step 2: Running --TICK--> Running (count<3 TRUE, count=1)
- Step 3: Running --TICK--> Running (count<3 TRUE, count=2)
- Step 4: Running --TICK--> Running (count<3 TRUE, count=3)
- Step 5: Running --TICK--> Done (count>=3 TRUE, terminal!)
Path length: 5

Example 3:
Statechart:
  A --e1--> B
  B --e2--> A
Initial: A, Terminal: C

Trace:
- Start at A
- Step 1: A --e1--> B (now at B)
- Step 2: B --e2--> A (now at A)
- Step 3: A --e1--> B (now at B)
- ... infinite loop, no path to C
Path length: infinite

"""

    # Format the test case
    trans_str = []
    for t in sc.transitions:
        arrow = f"{t['from']} --{t.get('event', 'e')}"
        if t.get('guard'):
            arrow += f"[{t['guard']}]"
        arrow += f"--> {t['to']}"
        trans_str.append(arrow)

    transitions = "\n  ".join(trans_str)
    terminals = ", ".join(sc.terminal)

    context_str = ""
    if sc.context:
        context_str = f", Context: {sc.context}"

    prompt = f"""{examples}Now solve:
Statechart:
  {transitions}
Initial: {sc.initial}, Terminal: {terminals}{context_str}

Trace:
- Start at {sc.initial}
-"""

    return prompt


def create_bfs_simulation_prompt(sc: StatechartPath) -> str:
    """Teach the model to simulate BFS."""

    example = """To find shortest path, use BFS (Breadth-First Search):

Example:
States: A, B, C, D
Transitions: A→B, A→C, B→D, C→D
Initial: A, Terminal: D

BFS Simulation:
Queue: [A]
Visited: {A: 0}

Pop A (dist=0):
  A→B: add B, Visited: {A:0, B:1}
  A→C: add C, Visited: {A:0, B:1, C:1}
  Queue: [B, C]

Pop B (dist=1):
  B→D: D is terminal! Distance = 1+1 = 2

Shortest path: 2 steps

"""

    # Format test case
    trans_list = []
    for t in sc.transitions:
        trans_list.append(f"{t['from']}→{t['to']}")

    prompt = f"""{example}Now solve:
States: {', '.join(sc.states)}
Transitions: {', '.join(trans_list)}
Initial: {sc.initial}, Terminal: {', '.join(sc.terminal)}

BFS Simulation:
Queue: [{sc.initial}]
Visited: {{{sc.initial}: 0}}

Pop {sc.initial} (dist=0):"""

    return prompt


def create_ascii_visualization_prompt(sc: StatechartPath, ascii_graph: str) -> str:
    """Prompt with ASCII graph visualization."""

    terminals = ", ".join(sc.terminal)

    context_str = ""
    if sc.context:
        context_str = f"\nContext: {sc.context}"

    prompt = f"""Count steps from {sc.initial} to terminal state.

Graph:
{ascii_graph}

Initial: {sc.initial}
Terminal: {terminals}{context_str}

Count the arrows from {sc.initial} to reach a terminal state.
Path length:"""

    return prompt


def create_decomposition_prompt(sc: StatechartPath) -> str:
    """Break path into segments."""

    trans_str = []
    for t in sc.transitions:
        arrow = f"{t['from']} → {t['to']}"
        if t.get('guard'):
            arrow += f" (when {t['guard']})"
        trans_str.append(arrow)

    prompt = f"""Break this path into segments:

Transitions:
{chr(10).join('  ' + t for t in trans_str)}

Initial: {sc.initial}
Terminal: {', '.join(sc.terminal)}

Segment analysis:
1. From {sc.initial}, I can go to:"""

    return prompt


def parse_path_length(output: str) -> int:
    """Parse path length from model output."""
    output_lower = output.lower()

    # Check for infinite
    if any(word in output_lower for word in ['infinite', 'unbounded', 'forever', 'no path', 'unreachable']):
        return -1

    # Look for "path length: N" pattern
    match = re.search(r'path\s*length[:\s]+(\d+)', output_lower)
    if match:
        return int(match.group(1))

    # Look for "N steps" pattern
    match = re.search(r'(\d+)\s*steps?', output_lower)
    if match:
        return int(match.group(1))

    # Look for "distance = N" pattern
    match = re.search(r'distance\s*=\s*(\d+)', output_lower)
    if match:
        return int(match.group(1))

    # Look for standalone number at end
    match = re.search(r':\s*(\d+)\s*$', output.strip())
    if match:
        return int(match.group(1))

    # Find any number
    numbers = re.findall(r'\b(\d+)\b', output)
    if numbers:
        # Return the last number (often the answer)
        return int(numbers[-1])

    return -1


class ScratchpadPredictor:
    """Predictor using scratchpad reasoning."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, sc: StatechartPath, style: PromptStyle, ascii_graph: str = None) -> Tuple[int, str]:
        """Predict path length using specified style."""
        from mlx_lm import generate as mlx_generate

        if style == PromptStyle.BASELINE:
            prompt = self._create_baseline_prompt(sc)
        elif style == PromptStyle.SCRATCHPAD:
            prompt = create_scratchpad_few_shot(sc)
        elif style == PromptStyle.BFS_SIMULATION:
            prompt = create_bfs_simulation_prompt(sc)
        elif style == PromptStyle.ASCII_VISUAL:
            prompt = create_ascii_visualization_prompt(sc, ascii_graph or "")
        elif style == PromptStyle.DECOMPOSITION:
            prompt = create_decomposition_prompt(sc)
        else:
            prompt = create_scratchpad_few_shot(sc)

        output = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=300,  # More tokens for reasoning
        )

        length = parse_path_length(output)
        return length, output

    def _create_baseline_prompt(self, sc: StatechartPath) -> str:
        """Simple baseline prompt without scratchpad."""
        trans_str = []
        for t in sc.transitions:
            trans_str.append(f"{t['from']} → {t['to']}")

        return f"""How many transitions to reach terminal state?

Transitions: {', '.join(trans_str)}
Initial: {sc.initial}
Terminal: {', '.join(sc.terminal)}

Answer (number or 'infinite'):"""


# Test cases
def get_test_cases() -> List[StatechartPath]:
    """Generate test cases for path length prediction."""
    return [
        # Linear path (3 steps)
        StatechartPath(
            states=["A", "B", "C", "D"],
            transitions=[
                {"from": "A", "to": "B", "event": "e1"},
                {"from": "B", "to": "C", "event": "e2"},
                {"from": "C", "to": "D", "event": "e3"},
            ],
            initial="A",
            terminal=["D"],
            expected_length=3,
        ),
        # Short path (2 steps)
        StatechartPath(
            states=["Start", "Middle", "End"],
            transitions=[
                {"from": "Start", "to": "Middle", "event": "GO"},
                {"from": "Middle", "to": "End", "event": "FINISH"},
            ],
            initial="Start",
            terminal=["End"],
            expected_length=2,
        ),
        # Branch (variable)
        StatechartPath(
            states=["Check", "PassPath", "FailPath", "Done"],
            transitions=[
                {"from": "Check", "to": "PassPath", "event": "EVAL", "guard": "score>=50"},
                {"from": "Check", "to": "FailPath", "event": "EVAL", "guard": "score<50"},
                {"from": "PassPath", "to": "Done", "event": "NEXT"},
                {"from": "FailPath", "to": "Done", "event": "NEXT"},
            ],
            initial="Check",
            terminal=["Done"],
            context={"score": 75},
            expected_length=2,
        ),
        # Counter loop (5 steps with count=0, target=3)
        StatechartPath(
            states=["Start", "Counting", "Done"],
            transitions=[
                {"from": "Start", "to": "Counting", "event": "BEGIN"},
                {"from": "Counting", "to": "Counting", "event": "TICK", "guard": "count<3"},
                {"from": "Counting", "to": "Done", "event": "TICK", "guard": "count>=3"},
            ],
            initial="Start",
            terminal=["Done"],
            context={"count": 0},
            expected_length=5,  # BEGIN + 3 TICKs + final TICK
        ),
        # Infinite loop
        StatechartPath(
            states=["A", "B", "C"],
            transitions=[
                {"from": "A", "to": "B", "event": "e1"},
                {"from": "B", "to": "A", "event": "e2"},
            ],
            initial="A",
            terminal=["C"],
            expected_length=-1,  # Infinite
        ),
        # Single step
        StatechartPath(
            states=["Off", "On"],
            transitions=[
                {"from": "Off", "to": "On", "event": "TOGGLE"},
            ],
            initial="Off",
            terminal=["On"],
            expected_length=1,
        ),
    ]


if __name__ == "__main__":
    # Test prompt generation
    cases = get_test_cases()

    print("Scratchpad Prompt Examples")
    print("=" * 60)

    for i, sc in enumerate(cases[:2]):
        print(f"\nTest Case {i+1}: Expected length = {sc.expected_length}")
        print("-" * 40)
        prompt = create_scratchpad_few_shot(sc)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
