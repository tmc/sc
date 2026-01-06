#!/usr/bin/env python3
"""
Hierarchy Scratchpad: Explicit enumeration for hierarchy understanding.

Applies the scratchpad technique to improve hierarchy reasoning from 27% to 50%+.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from enum import IntEnum


class HierarchyTask(IntEnum):
    """Types of hierarchy reasoning tasks."""
    CONTAINMENT = 1   # Is X inside Y?
    CASCADE_EXIT = 2  # If X exits, what else exits?
    DEFAULT_ENTRY = 3 # Entering X, what's the configuration?
    DEPTH = 4         # How deep is state X?


@dataclass
class HierarchyState:
    """A state in a hierarchical statechart."""
    label: str
    children: List['HierarchyState'] = field(default_factory=list)
    is_initial: bool = False

    def get_all_descendants(self) -> Set[str]:
        """Get all descendant state labels."""
        result = set()
        for child in self.children:
            result.add(child.label)
            result.update(child.get_all_descendants())
        return result


@dataclass
class HierarchyCase:
    """A test case for hierarchy scratchpad."""
    root: HierarchyState
    task: HierarchyTask
    query_state: str
    target_state: Optional[str] = None  # For containment checks
    expected: Any = None  # Expected answer


def build_state_index(root: HierarchyState) -> Dict[str, Tuple[HierarchyState, Optional[str]]]:
    """Build index mapping state label -> (state, parent_label)."""
    index = {}

    def traverse(state: HierarchyState, parent_label: Optional[str]):
        index[state.label] = (state, parent_label)
        for child in state.children:
            traverse(child, state.label)

    traverse(root, None)
    return index


def compute_parent_chain(state_label: str, index: Dict) -> List[str]:
    """Compute chain from state up to root."""
    chain = [state_label]
    current = state_label
    while True:
        state, parent = index.get(current, (None, None))
        if parent is None:
            break
        chain.append(parent)
        current = parent
    return chain


def compute_all_descendants(state_label: str, index: Dict) -> Set[str]:
    """Compute all descendants of a state."""
    state, _ = index.get(state_label, (None, None))
    if state is None:
        return set()
    return state.get_all_descendants()


def format_hierarchy_tree(root: HierarchyState, indent: int = 0) -> str:
    """Format hierarchy as indented tree string."""
    lines = []
    prefix = "  " * indent
    initial_marker = "*" if root.is_initial else ""
    lines.append(f"{prefix}{root.label}{initial_marker}")
    for child in root.children:
        lines.extend(format_hierarchy_tree(child, indent + 1).split("\n"))
    return "\n".join(lines)


def format_hierarchy_nested(root: HierarchyState) -> str:
    """Format hierarchy as nested braces: Root{A{B,C}, D}."""
    if not root.children:
        initial = "*" if root.is_initial else ""
        return f"{root.label}{initial}"

    children_str = ", ".join(format_hierarchy_nested(c) for c in root.children)
    return f"{root.label}{{{children_str}}}"


# ============================================================================
# SCRATCHPAD PROMPTS
# ============================================================================

def create_containment_prompt(case: HierarchyCase) -> str:
    """Create scratchpad prompt for containment check."""
    tree = format_hierarchy_tree(case.root)
    nested = format_hierarchy_nested(case.root)

    examples = """CONTAINMENT: Is X inside Y? Trace UP the parent chain from X.
A state is "inside" another if the other appears in its parent chain.

Example 1:
Tree:
Root
  A
    B
    C
  D

Q: Is B inside Root?
Trace UP from B:
- B's parent is A
- A's parent is Root
Chain: B → A → Root
Is Root in chain? YES
Answer: YES

Example 2:
Tree:
Root
  A
    B
    C
  D

Q: Is D inside A?
Trace UP from D:
- D's parent is Root
Chain: D → Root
Is A in chain? NO
Answer: NO

Example 3:
Tree:
Root
  A
    B
    C
  D

Q: Is C inside A?
Trace UP from C:
- C's parent is A
Chain: C → A
Is A in chain? YES
Answer: YES

Example 4:
Tree:
Root
  A
    B
    C
  D

Q: Is A inside B?
Trace UP from A:
- A's parent is Root
Chain: A → Root
Is B in chain? NO
Answer: NO

"""

    prompt = f"""{examples}Now solve:
Tree:
{tree}

Q: Is {case.query_state} inside {case.target_state}?
Trace UP from {case.query_state}:
-"""

    return prompt


def create_cascade_exit_prompt(case: HierarchyCase) -> str:
    """Create scratchpad prompt for cascade exit."""
    tree = format_hierarchy_tree(case.root)
    nested = format_hierarchy_nested(case.root)

    examples = """CASCADE: List state and ALL descendants (everything below it in tree).

Example 1:
Tree:
Root
  A
    B
    C
  D

Q: If A exits, list A and all below:
From tree, A contains B and C. Neither have children.
Result: A, B, C
Answer: A, B, C

Example 2:
Tree:
Root
  A
    B
    C
  D

Q: If B exits, list B and all below:
From tree, B has no children.
Result: B
Answer: B

Example 3:
Tree:
Root
  A
    B
    C
  D

Q: If Root exits, list Root and all below:
From tree, Root contains A and D. A contains B and C.
Result: Root, A, B, C, D
Answer: Root, A, B, C, D

Example 4:
Tree:
Root
  X
    Y
      Z
  W

Q: If X exits, list X and all below:
From tree, X contains Y. Y contains Z.
Result: X, Y, Z
Answer: X, Y, Z

"""

    prompt = f"""{examples}Now solve:
Tree:
{tree}

Q: If {case.query_state} exits, list {case.query_state} and all below:
From tree,"""

    return prompt


def create_default_entry_prompt(case: HierarchyCase) -> str:
    """Create scratchpad prompt for default entry configuration."""
    tree = format_hierarchy_tree(case.root)
    nested = format_hierarchy_nested(case.root)

    examples = """DEFAULT ENTRY: Follow the * markers DOWN until reaching a leaf.
States marked with * are initial children.

Example 1:
Tree:
Root
  A*
    B*
    C
  D

Q: Entering Root, what's the configuration?
Follow * DOWN:
- Enter Root
- Root's children: A*, D. Pick A* (has *)
- A's children: B*, C. Pick B* (has *)
- B has no children (leaf, stop)
Active: Root, A, B
Answer: Root, A, B

Example 2:
Tree:
Root
  A
    B*
    C
  D*

Q: Entering Root, what's the configuration?
Follow * DOWN:
- Enter Root
- Root's children: A, D*. Pick D* (has *)
- D has no children (leaf, stop)
Active: Root, D
Answer: Root, D

Example 3:
Tree:
Root
  A*
    B
    C*
  D

Q: Entering A, what's the configuration?
Follow * DOWN:
- Enter A
- A's children: B, C*. Pick C* (has *)
- C has no children (leaf, stop)
Active: A, C
Answer: A, C

"""

    prompt = f"""{examples}Now solve:
Tree:
{tree}

Q: Entering {case.query_state}, what's the configuration?
Follow * DOWN:
- Enter {case.query_state}
-"""

    return prompt


def create_depth_prompt(case: HierarchyCase) -> str:
    """Create scratchpad prompt for depth calculation."""
    tree = format_hierarchy_tree(case.root)
    nested = format_hierarchy_nested(case.root)

    examples = """DEPTH: Count steps UP to root. Root = depth 0.

Example 1:
Tree:
Root
  A
    B
    C
  D

Q: What is the depth of B?
Trace UP from B:
- B's parent is A (count 1)
- A's parent is Root (count 2)
- Root has no parent (stop)
Depth = 2
Answer: 2

Example 2:
Tree:
Root
  A
    B
    C
  D

Q: What is the depth of Root?
Trace UP from Root:
- Root has no parent (stop)
Depth = 0
Answer: 0

Example 3:
Tree:
Root
  A
    B
    C
  D

Q: What is the depth of A?
Trace UP from A:
- A's parent is Root (count 1)
- Root has no parent (stop)
Depth = 1
Answer: 1

Example 4:
Tree:
Root
  A
    B
    C
  D

Q: What is the depth of D?
Trace UP from D:
- D's parent is Root (count 1)
- Root has no parent (stop)
Depth = 1
Answer: 1

"""

    prompt = f"""{examples}Now solve:
Tree:
{tree}

Q: What is the depth of {case.query_state}?
Trace UP from {case.query_state}:
-"""

    return prompt


def create_prompt(case: HierarchyCase) -> str:
    """Create appropriate scratchpad prompt based on task type."""
    if case.task == HierarchyTask.CONTAINMENT:
        return create_containment_prompt(case)
    elif case.task == HierarchyTask.CASCADE_EXIT:
        return create_cascade_exit_prompt(case)
    elif case.task == HierarchyTask.DEFAULT_ENTRY:
        return create_default_entry_prompt(case)
    elif case.task == HierarchyTask.DEPTH:
        return create_depth_prompt(case)
    else:
        raise ValueError(f"Unknown task type: {case.task}")


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def _clean_output(output: str) -> str:
    """Clean output by truncating at common stop points."""
    import re
    # Stop at "Example", "Now solve", "Tree:", etc.
    for stop_marker in ["\n\nExample", "\n\nNow solve", "\n\nTree:", "\n\nQ:", "```"]:
        if stop_marker in output:
            output = output.split(stop_marker)[0]
    return output.strip()


def _extract_valid_state_names(text: str) -> List[str]:
    """Extract valid state names (alphanumeric, no newlines/garbage)."""
    import re
    # Match state names: capitalized words or single letters
    matches = re.findall(r'\b([A-Z][a-zA-Z0-9]*)\b', text)
    # Filter out common non-state words
    exclude = {'Answer', 'YES', 'NO', 'Tree', 'Example', 'Collect', 'Active', 'Depth', 'Chain', 'Pick'}
    return [m for m in matches if m not in exclude]


def parse_containment_response(output: str) -> bool:
    """Parse YES/NO answer for containment."""
    output = _clean_output(output)
    output_lower = output.lower()

    # Look for explicit YES/NO
    if "answer: yes" in output_lower or "answer:yes" in output_lower:
        return True
    if "answer: no" in output_lower or "answer:no" in output_lower:
        return False

    # Look for "YES" or "NO" at end
    lines = output.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.lower().strip()
        if line_lower in ("yes", "no"):
            return line_lower == "yes"
        if "yes" in line_lower and "no" not in line_lower:
            return True
        if "no" in line_lower and "yes" not in line_lower:
            return False

    # Look for "is in chain" vs "NOT in chain"
    if "not in chain" in output_lower or "? no" in output_lower:
        return False
    if "in chain" in output_lower or "? yes" in output_lower:
        return True

    return False  # Default


def parse_cascade_response(output: str) -> Set[str]:
    """Parse set of states from cascade exit response."""
    import re

    output = _clean_output(output)

    # Look for "Answer: X, Y, Z" pattern
    match = re.search(r'answer:\s*([A-Za-z0-9_,\s]+)', output, re.IGNORECASE)
    if match:
        states_str = match.group(1).split("\n")[0]  # First line only
        states = _extract_valid_state_names(states_str)
        if states:
            return set(states)

    # Look for "Collect all: X + Y + Z" pattern
    match = re.search(r'collect all:\s*([^\n]+)', output, re.IGNORECASE)
    if match:
        states_str = match.group(1)
        # Handle both "+" and "," separators
        states_str = states_str.replace('+', ',')
        states = _extract_valid_state_names(states_str)
        if states:
            return set(states)

    # Look for state names in last line with comma
    lines = output.strip().split("\n")
    for line in reversed(lines):
        if "," in line or "+" in line:
            parts = line.split(":")[-1] if ":" in line else line
            states = _extract_valid_state_names(parts)
            if states:
                return set(states)

    return set()


def parse_entry_response(output: str) -> Set[str]:
    """Parse configuration set from default entry response."""
    import re

    output = _clean_output(output)

    # Look for "Answer: X, Y, Z" pattern
    match = re.search(r'answer:\s*([A-Za-z0-9_,\s]+)', output, re.IGNORECASE)
    if match:
        states_str = match.group(1).split("\n")[0]  # First line only
        states = _extract_valid_state_names(states_str)
        if states:
            return set(states)

    # Look for "Active: X, Y, Z" pattern
    match = re.search(r'active:\s*([^\n]+)', output, re.IGNORECASE)
    if match:
        states_str = match.group(1)
        states = _extract_valid_state_names(states_str)
        if states:
            return set(states)

    # Look for "Configuration: {X, Y, Z}"
    match = re.search(r'configuration:\s*\{([^}]+)\}', output, re.IGNORECASE)
    if match:
        states_str = match.group(1)
        states = [s.strip() for s in states_str.split(",") if s.strip()]
        return set(states)

    return set()


def parse_depth_response(output: str) -> int:
    """Parse depth number from response."""
    import re

    output = _clean_output(output)

    # Look for "Answer: N" pattern
    match = re.search(r'answer:\s*(\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for "Depth = N" pattern
    match = re.search(r'depth\s*=\s*(\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for "Count: N" pattern
    match = re.search(r'count:\s*(\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for number at end of line
    lines = output.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r'(\d+)\s*$', line)
        if match:
            return int(match.group(1))

    return -1


def parse_response(case: HierarchyCase, output: str) -> Any:
    """Parse response based on task type."""
    if case.task == HierarchyTask.CONTAINMENT:
        return parse_containment_response(output)
    elif case.task == HierarchyTask.CASCADE_EXIT:
        return parse_cascade_response(output)
    elif case.task == HierarchyTask.DEFAULT_ENTRY:
        return parse_entry_response(output)
    elif case.task == HierarchyTask.DEPTH:
        return parse_depth_response(output)
    else:
        raise ValueError(f"Unknown task type: {case.task}")


# ============================================================================
# TEST CASE GENERATION
# ============================================================================

def get_test_cases() -> List[HierarchyCase]:
    """Generate test cases for hierarchy scratchpad benchmark."""
    cases = []

    # =========== HIERARCHY 1: Simple 3-level ===========
    # Root{A{B, C}, D}
    h1_b = HierarchyState(label="B", is_initial=True)
    h1_c = HierarchyState(label="C")
    h1_a = HierarchyState(label="A", children=[h1_b, h1_c], is_initial=True)
    h1_d = HierarchyState(label="D")
    h1_root = HierarchyState(label="Root", children=[h1_a, h1_d])

    # Containment tests
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.CONTAINMENT,
        query_state="B", target_state="Root", expected=True
    ))
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.CONTAINMENT,
        query_state="D", target_state="A", expected=False
    ))
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.CONTAINMENT,
        query_state="C", target_state="A", expected=True
    ))

    # Cascade exit tests
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="A", expected={"A", "B", "C"}
    ))
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="B", expected={"B"}
    ))

    # Default entry tests
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.DEFAULT_ENTRY,
        query_state="Root", expected={"Root", "A", "B"}
    ))
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.DEFAULT_ENTRY,
        query_state="A", expected={"A", "B"}
    ))

    # Depth tests
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.DEPTH,
        query_state="B", expected=2
    ))
    cases.append(HierarchyCase(
        root=h1_root, task=HierarchyTask.DEPTH,
        query_state="Root", expected=0
    ))

    # =========== HIERARCHY 2: 4-level deep ===========
    # Root{A{B{C{D}}}}
    h2_d = HierarchyState(label="D", is_initial=True)
    h2_c = HierarchyState(label="C", children=[h2_d], is_initial=True)
    h2_b = HierarchyState(label="B", children=[h2_c], is_initial=True)
    h2_a = HierarchyState(label="A", children=[h2_b], is_initial=True)
    h2_root = HierarchyState(label="Root", children=[h2_a])

    cases.append(HierarchyCase(
        root=h2_root, task=HierarchyTask.CONTAINMENT,
        query_state="D", target_state="A", expected=True
    ))
    cases.append(HierarchyCase(
        root=h2_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="B", expected={"B", "C", "D"}
    ))
    cases.append(HierarchyCase(
        root=h2_root, task=HierarchyTask.DEPTH,
        query_state="D", expected=4
    ))
    cases.append(HierarchyCase(
        root=h2_root, task=HierarchyTask.DEFAULT_ENTRY,
        query_state="Root", expected={"Root", "A", "B", "C", "D"}
    ))

    # =========== HIERARCHY 3: Wide and shallow ===========
    # Root{A, B, C, D, E}
    h3_a = HierarchyState(label="A", is_initial=True)
    h3_b = HierarchyState(label="B")
    h3_c = HierarchyState(label="C")
    h3_d = HierarchyState(label="D")
    h3_e = HierarchyState(label="E")
    h3_root = HierarchyState(label="Root", children=[h3_a, h3_b, h3_c, h3_d, h3_e])

    cases.append(HierarchyCase(
        root=h3_root, task=HierarchyTask.CONTAINMENT,
        query_state="C", target_state="Root", expected=True
    ))
    cases.append(HierarchyCase(
        root=h3_root, task=HierarchyTask.CONTAINMENT,
        query_state="A", target_state="B", expected=False
    ))
    cases.append(HierarchyCase(
        root=h3_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="Root", expected={"Root", "A", "B", "C", "D", "E"}
    ))
    cases.append(HierarchyCase(
        root=h3_root, task=HierarchyTask.DEPTH,
        query_state="E", expected=1
    ))

    # =========== HIERARCHY 4: Mixed structure ===========
    # Root{Menu{Settings{Audio, Video}, Help}, Game{Playing, Paused}}
    h4_audio = HierarchyState(label="Audio", is_initial=True)
    h4_video = HierarchyState(label="Video")
    h4_settings = HierarchyState(label="Settings", children=[h4_audio, h4_video])
    h4_help = HierarchyState(label="Help")
    h4_menu = HierarchyState(label="Menu", children=[h4_settings, h4_help], is_initial=True)
    h4_playing = HierarchyState(label="Playing", is_initial=True)
    h4_paused = HierarchyState(label="Paused")
    h4_game = HierarchyState(label="Game", children=[h4_playing, h4_paused])
    h4_root = HierarchyState(label="Root", children=[h4_menu, h4_game])

    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.CONTAINMENT,
        query_state="Audio", target_state="Menu", expected=True
    ))
    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.CONTAINMENT,
        query_state="Playing", target_state="Menu", expected=False
    ))
    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="Settings", expected={"Settings", "Audio", "Video"}
    ))
    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.CASCADE_EXIT,
        query_state="Menu", expected={"Menu", "Settings", "Audio", "Video", "Help"}
    ))
    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.DEFAULT_ENTRY,
        query_state="Game", expected={"Game", "Playing"}
    ))
    cases.append(HierarchyCase(
        root=h4_root, task=HierarchyTask.DEPTH,
        query_state="Video", expected=3
    ))

    return cases


if __name__ == "__main__":
    # Test prompt generation
    cases = get_test_cases()

    print("Hierarchy Scratchpad Prompts")
    print("=" * 60)

    for task_type in HierarchyTask:
        test_case = next((c for c in cases if c.task == task_type), None)
        if test_case:
            print(f"\n{'=' * 60}")
            print(f"Task: {task_type.name}")
            print("=" * 60)
            prompt = create_prompt(test_case)
            print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
