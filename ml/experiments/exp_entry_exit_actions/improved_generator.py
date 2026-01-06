"""
Improved Entry/Exit Action Generator.

Strategies to fix exit action generation (50% → 90%+):
1. Exit-heavy few-shot: More exit than entry examples
2. Explicit markers: "EXIT action runs when LEAVING the state"
3. Contrast pairs: Show entry vs exit side-by-side
4. Direct prompt: Simple, focused prompt (learned from L3 success)
"""

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class PromptStrategy(Enum):
    """Prompting strategies for exit action generation."""
    BASELINE = "baseline"
    EXIT_HEAVY = "exit_heavy"
    EXPLICIT = "explicit"
    CONTRAST = "contrast"
    DIRECT = "direct"


@dataclass
class GenerationResult:
    """Result of action generation."""
    strategy: PromptStrategy
    action_type: str  # "entry", "exit", "both"
    description: str
    generated_json: Optional[str]
    parsed_sc: Optional[Dict]
    has_entry: bool
    has_exit: bool
    correct: bool
    raw_output: str
    generation_time_ms: float
    error: Optional[str] = None


# Heavy exit examples (5 exit, 2 entry for bias)
EXIT_HEAVY_EXAMPLES = '''Example 1 - EXIT action (cleanup when leaving):
{"root_state":{"label":"Process","type":2,"children":[{"label":"Running","type":1,"is_initial":true,"on_exit":["cleanup()"]}]},"transitions":[]}

Example 2 - EXIT action (save when leaving):
{"root_state":{"label":"Editor","type":2,"children":[{"label":"Editing","type":1,"is_initial":true,"on_exit":["save_document()"]}]},"transitions":[]}

Example 3 - EXIT action (stop timer when leaving):
{"root_state":{"label":"Timer","type":2,"children":[{"label":"Running","type":1,"is_initial":true,"on_exit":["stop_timer()"]}]},"transitions":[]}

Example 4 - EXIT action (close connection when leaving):
{"root_state":{"label":"Network","type":2,"children":[{"label":"Connected","type":1,"is_initial":true,"on_exit":["close_connection()"]}]},"transitions":[]}

Example 5 - EXIT action (release lock when leaving):
{"root_state":{"label":"Lock","type":2,"children":[{"label":"Held","type":1,"is_initial":true,"on_exit":["release_lock()"]}]},"transitions":[]}

Example 6 - ENTRY action (start when entering):
{"root_state":{"label":"App","type":2,"children":[{"label":"Running","type":1,"is_initial":true,"on_entry":["start_app()"]}]},"transitions":[]}

Example 7 - ENTRY action (init when entering):
{"root_state":{"label":"System","type":2,"children":[{"label":"Ready","type":1,"is_initial":true,"on_entry":["initialize()"]}]},"transitions":[]}
'''

# Contrast examples showing entry vs exit explicitly
CONTRAST_EXAMPLES = '''ENTRY vs EXIT - Know the difference:

ENTRY action (on_entry) - Runs when ENTERING/ARRIVING at the state:
- "start timer when entering Running" → on_entry: ["start_timer()"]
- "initialize when entering Ready" → on_entry: ["initialize()"]

EXIT action (on_exit) - Runs when LEAVING/EXITING the state:
- "stop timer when leaving Running" → on_exit: ["stop_timer()"]
- "cleanup when leaving Active" → on_exit: ["cleanup()"]

Key words for ENTRY: entering, starting, begin, arrive, on entry
Key words for EXIT: leaving, exiting, stop, end, cleanup, on exit, when left

Example ENTRY:
{"root_state":{"label":"Timer","type":2,"children":[{"label":"Running","type":1,"on_entry":["start_timer()"]}]},"transitions":[]}

Example EXIT:
{"root_state":{"label":"Timer","type":2,"children":[{"label":"Running","type":1,"on_exit":["stop_timer()"]}]},"transitions":[]}

Example BOTH:
{"root_state":{"label":"Timer","type":2,"children":[{"label":"Running","type":1,"on_entry":["start_timer()"],"on_exit":["stop_timer()"]}]},"transitions":[]}
'''

# Explicit marker examples
EXPLICIT_EXAMPLES = '''Actions in statecharts:

on_entry = Action that runs when ENTERING the state (arriving, starting)
on_exit = Action that runs when LEAVING the state (departing, stopping)

CRITICAL: "leaving" or "exiting" means on_exit, NOT on_entry!

EXIT example - "stop timer when leaving Running":
{"root_state":{"label":"M","type":2,"children":[{"label":"Running","type":1,"on_exit":["stop_timer()"]}]},"transitions":[]}

EXIT example - "save when leaving Active":
{"root_state":{"label":"M","type":2,"children":[{"label":"Active","type":1,"on_exit":["save()"]}]},"transitions":[]}

EXIT example - "cleanup when exiting Process":
{"root_state":{"label":"M","type":2,"children":[{"label":"Process","type":1,"on_exit":["cleanup()"]}]},"transitions":[]}

ENTRY example - "init when entering Ready":
{"root_state":{"label":"M","type":2,"children":[{"label":"Ready","type":1,"on_entry":["init()"]}]},"transitions":[]}
'''


class ImprovedGenerator:
    """Generator with multiple strategies for entry/exit actions."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        description: str,
        action_type: str,  # "entry", "exit", "both"
        strategy: PromptStrategy,
    ) -> GenerationResult:
        """Generate statechart with specified action type."""
        prompt = self._build_prompt(description, action_type, strategy)

        start = time.time()
        output = self._generate(prompt)
        elapsed_ms = (time.time() - start) * 1000

        # Parse result
        json_str = self._extract_json(output)
        parsed_sc = None
        has_entry = False
        has_exit = False
        error = None

        if json_str:
            try:
                parsed_sc = json.loads(json_str)
                has_entry, has_exit = self._check_actions(parsed_sc)
            except json.JSONDecodeError as e:
                error = f"JSON parse error: {e}"
        else:
            error = "No JSON found"

        # Determine correctness
        correct = False
        if action_type == "entry":
            correct = has_entry
        elif action_type == "exit":
            correct = has_exit
        elif action_type == "both":
            correct = has_entry and has_exit

        return GenerationResult(
            strategy=strategy,
            action_type=action_type,
            description=description,
            generated_json=json_str,
            parsed_sc=parsed_sc,
            has_entry=has_entry,
            has_exit=has_exit,
            correct=correct,
            raw_output=output,
            generation_time_ms=elapsed_ms,
            error=error,
        )

    def _build_prompt(
        self,
        description: str,
        action_type: str,
        strategy: PromptStrategy,
    ) -> str:
        """Build prompt based on strategy."""
        if strategy == PromptStrategy.BASELINE:
            return self._baseline_prompt(description)
        elif strategy == PromptStrategy.EXIT_HEAVY:
            return self._exit_heavy_prompt(description, action_type)
        elif strategy == PromptStrategy.EXPLICIT:
            return self._explicit_prompt(description, action_type)
        elif strategy == PromptStrategy.CONTRAST:
            return self._contrast_prompt(description, action_type)
        elif strategy == PromptStrategy.DIRECT:
            return self._direct_prompt(description, action_type)
        return self._baseline_prompt(description)

    def _baseline_prompt(self, description: str) -> str:
        """Original baseline prompt."""
        return f'''Generate a statechart JSON with the requested action.

Task: {description}

Output valid JSON with root_state and transitions:'''

    def _exit_heavy_prompt(self, description: str, action_type: str) -> str:
        """Prompt with more exit than entry examples."""
        return f'''{EXIT_HEAVY_EXAMPLES}
Task: {description}
JSON:'''

    def _explicit_prompt(self, description: str, action_type: str) -> str:
        """Prompt with explicit markers for exit."""
        action_hint = ""
        if action_type == "exit":
            action_hint = "\nREMEMBER: This needs on_exit (leaving the state), NOT on_entry!"
        elif action_type == "entry":
            action_hint = "\nREMEMBER: This needs on_entry (entering the state)."

        return f'''{EXPLICIT_EXAMPLES}
Task: {description}{action_hint}
JSON:'''

    def _contrast_prompt(self, description: str, action_type: str) -> str:
        """Prompt with entry vs exit contrast."""
        return f'''{CONTRAST_EXAMPLES}
Task: {description}
JSON:'''

    def _direct_prompt(self, description: str, action_type: str) -> str:
        """Simple, direct prompt (based on L3 guard success)."""
        if action_type == "exit":
            return f'''Generate statechart JSON where the state has on_exit action.

on_exit means: action runs when LEAVING the state

Example - "cleanup when leaving Active":
{{"root_state":{{"label":"M","type":2,"children":[{{"label":"Active","type":1,"is_initial":true,"on_exit":["cleanup()"]}}]}},"transitions":[]}}

Now generate for: "{description}"
JSON:'''
        elif action_type == "entry":
            return f'''Generate statechart JSON where the state has on_entry action.

on_entry means: action runs when ENTERING the state

Example - "init when entering Ready":
{{"root_state":{{"label":"M","type":2,"children":[{{"label":"Ready","type":1,"is_initial":true,"on_entry":["init()"]}}]}},"transitions":[]}}

Now generate for: "{description}"
JSON:'''
        else:  # both
            return f'''Generate statechart JSON with BOTH on_entry AND on_exit actions.

on_entry = action when ENTERING
on_exit = action when LEAVING

Example - "start on entry, stop on exit":
{{"root_state":{{"label":"M","type":2,"children":[{{"label":"Running","type":1,"is_initial":true,"on_entry":["start()"],"on_exit":["stop()"]}}]}},"transitions":[]}}

Now generate for: "{description}"
JSON:'''

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=300,
                sampler=sampler,
            )
            return output
        except Exception as e:
            return f"Error: {e}"

    def _extract_json(self, output: str) -> Optional[str]:
        """Extract JSON from output."""
        # Remove markdown
        output = re.sub(r'```json\s*', '', output)
        output = re.sub(r'```\s*', '', output)
        output = output.strip()

        # Find balanced braces
        if '{' not in output:
            return None

        start = output.find('{')
        depth = 0
        for i, c in enumerate(output[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return output[start:i+1]
        return None

    def _check_actions(self, sc: Dict) -> tuple:
        """Check if SC has entry and/or exit actions."""
        has_entry = False
        has_exit = False

        def check_state(state):
            nonlocal has_entry, has_exit
            if not isinstance(state, dict):
                return
            if state.get("on_entry"):
                has_entry = True
            if state.get("on_exit"):
                has_exit = True
            for child in state.get("children", []):
                check_state(child)

        root = sc.get("root_state", {})
        if isinstance(root, dict):
            check_state(root)
        return has_entry, has_exit


# Test cases
ENTRY_TESTS = [
    ("start timer when entering Running", "entry"),
    ("initialize resources when entering Ready", "entry"),
    ("begin logging when entering Active", "entry"),
    ("start playback when entering Playing", "entry"),
]

EXIT_TESTS = [
    ("stop timer when leaving Running", "exit"),
    ("cleanup resources when leaving Active", "exit"),
    ("save state when exiting Editor", "exit"),
    ("close connection when leaving Connected", "exit"),
]

BOTH_TESTS = [
    ("start timer on entry, stop on exit for Running", "both"),
    ("initialize on entry, cleanup on exit for Process", "both"),
    ("connect on entry, disconnect on exit for Session", "both"),
]


if __name__ == "__main__":
    print("Improved Generator Test")
    print("=" * 60)

    gen = ImprovedGenerator()

    print("\nDIRECT prompt for EXIT:")
    print("-" * 40)
    prompt = gen._direct_prompt("stop timer when leaving Running", "exit")
    print(prompt)
