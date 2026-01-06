#!/usr/bin/env python3
"""
Context Guards Comprehensive: Guard types, context variables, and interactions.

Covers all guard semantics from the proto specification:
- Boolean guards (is_locked, has_permission)
- Comparison guards (count > 5, temp <= 100)
- Compound guards ((a && b) || c)
- Guard-action chains
- Transition priority with overlapping guards
- In-state guards for parallel regions
- Nested object/array access
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import IntEnum
import re


class GuardCategory(IntEnum):
    """Categories of guard expressions."""
    BOOLEAN = 1       # is_locked, has_key
    COMPARISON = 2    # count > 5, temp <= 100
    COMPOUND = 3      # (a && b) || c
    CHAIN = 4         # Guard depends on prior action's effect
    PRIORITY = 5      # Multiple guards, select by priority
    IN_STATE = 6      # in(State.Active)
    NESTED = 7        # user.role == "admin"
    ARRAY = 8         # items.length > 0, items[0].valid


@dataclass
class Transition:
    """A transition with guard and optional action."""
    source: str
    target: str
    event: str
    guard: Optional[str] = None
    action: Optional[str] = None
    priority: int = 0


@dataclass
class GuardTestCase:
    """A test case for guard evaluation."""
    category: GuardCategory
    description: str
    transitions: List[Transition]
    initial_context: Dict[str, Any]
    event_sequence: List[str]
    expected_final_state: str
    expected_final_context: Dict[str, Any]
    active_states: Set[str] = field(default_factory=set)  # For in-state guards


def evaluate_guard_expr(guard: str, context: Dict[str, Any], active: Set[str] = None) -> bool:
    """Evaluate a guard expression against context (ground truth)."""
    if not guard:
        return True

    active = active or set()

    # Replace in() checks
    def in_state_check(match):
        state_path = match.group(1)
        state_name = state_path.split('.')[-1] if '.' in state_path else state_path
        return str(state_name in active)

    guard = re.sub(r'in\(([^)]+)\)', in_state_check, guard)

    # Replace logical operators for Python eval
    guard_py = guard.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')

    # Handle nested access (user.role -> context['user']['role'])
    def resolve_nested(match):
        path = match.group(0)
        parts = path.split('.')
        result = "context"
        for part in parts:
            # Handle array access like items[0]
            if '[' in part:
                name, idx = part.split('[')
                idx = idx.rstrip(']')
                result = f"{result}['{name}'][{idx}]"
            else:
                result = f"{result}['{part}']"
        return result

    # Match identifiers (including nested paths)
    guard_py = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\])*)\b', resolve_nested, guard_py)

    # Clean up double conversions
    guard_py = guard_py.replace("context['True']", "True").replace("context['False']", "False")
    guard_py = guard_py.replace("context['true']", "True").replace("context['false']", "False")

    try:
        return bool(eval(guard_py))
    except Exception:
        return False


def execute_action(action: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an action expression and return updated context."""
    if not action:
        return context

    new_context = context.copy()

    # Parse action: var=expr or var++, var--
    for stmt in action.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue

        if '+=' in stmt:
            var, expr = stmt.split('+=')
            var = var.strip()
            expr = expr.strip()
            new_context[var] = new_context.get(var, 0) + eval(expr, {}, new_context)
        elif '-=' in stmt:
            var, expr = stmt.split('-=')
            var = var.strip()
            expr = expr.strip()
            new_context[var] = new_context.get(var, 0) - eval(expr, {}, new_context)
        elif '++' in stmt:
            var = stmt.replace('++', '').strip()
            new_context[var] = new_context.get(var, 0) + 1
        elif '--' in stmt:
            var = stmt.replace('--', '').strip()
            new_context[var] = new_context.get(var, 0) - 1
        elif '=' in stmt:
            var, expr = stmt.split('=', 1)
            var = var.strip()
            expr = expr.strip()
            # Handle expressions like x*2
            try:
                new_context[var] = eval(expr, {}, new_context)
            except Exception:
                new_context[var] = expr  # String assignment

    return new_context


def simulate_machine(
    transitions: List[Transition],
    initial_state: str,
    initial_context: Dict[str, Any],
    events: List[str],
    active_states: Set[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Simulate a state machine with guards and actions."""
    state = initial_state
    context = initial_context.copy()
    active = active_states.copy() if active_states else {initial_state}

    for event in events:
        # Find matching transitions (source matches, event matches, guard passes)
        candidates = []
        for t in transitions:
            if t.source == state and t.event == event:
                if evaluate_guard_expr(t.guard, context, active):
                    candidates.append(t)

        if not candidates:
            continue  # Event ignored

        # Select by priority (lowest number = highest priority)
        candidates.sort(key=lambda t: t.priority)
        selected = candidates[0]

        # Execute action
        if selected.action:
            context = execute_action(selected.action, context)

        # Update state
        state = selected.target
        active.discard(selected.source)
        active.add(selected.target)

    return state, context


# ============================================================================
# PROMPT GENERATION
# ============================================================================

def format_transitions(transitions: List[Transition]) -> str:
    """Format transitions for prompt."""
    lines = []
    for t in transitions:
        line = f"{t.source} --{t.event}"
        if t.guard:
            line += f"[{t.guard}]"
        if t.action:
            line += f"/{t.action}"
        line += f"--> {t.target}"
        if t.priority > 0:
            line += f" (priority {t.priority})"
        lines.append(line)
    return "\n".join(lines)


def create_guard_eval_prompt(case: GuardTestCase) -> str:
    """Create prompt for guard evaluation prediction."""
    trans_str = format_transitions(case.transitions)
    ctx_str = ", ".join(f"{k}={v}" for k, v in case.initial_context.items())
    events_str = ", ".join(case.event_sequence)
    active_str = ", ".join(case.active_states) if case.active_states else "none"

    examples = """Evaluate guards and trace state machine execution.

Example 1 (BOOLEAN):
Transitions:
Locked --unlock[has_key]--> Unlocked
Locked --unlock[!has_key]--> Locked
Context: has_key=True
Events: unlock
Trace:
- At Locked, event=unlock
- Guard [has_key]: has_key=True → TRUE
- Take Locked→Unlocked
Final state: Unlocked
Answer: Unlocked

Example 2 (COMPARISON):
Transitions:
Counter --inc[count<max]/count++--> Counter
Counter --inc[count>=max]--> Full
Context: count=2, max=3
Events: inc, inc
Trace:
- At Counter, event=inc
- Guard [count<max]: 2<3 → TRUE
- Action: count++ → count=3
- At Counter, event=inc
- Guard [count<max]: 3<3 → FALSE
- Guard [count>=max]: 3>=3 → TRUE
- Take Counter→Full
Final state: Full
Answer: Full

Example 3 (COMPOUND):
Transitions:
Door --open[(unlocked && !alarm) || override]--> Open
Door --open[!((unlocked && !alarm) || override)]--> Door
Context: unlocked=True, alarm=False, override=False
Events: open
Trace:
- At Door, event=open
- Guard: unlocked=T, alarm=F, override=F
- (T && !F) || F = (T && T) || F = T || F = TRUE
- Take Door→Open
Final state: Open
Answer: Open

Example 4 (CHAIN):
Transitions:
A --e1/x=1--> B
B --e2[x>0]/y=x*2--> C
C --e3[y==2]--> D
Context: x=0, y=0
Events: e1, e2, e3
Trace:
- At A, event=e1: action x=1, go to B, ctx={x:1, y:0}
- At B, event=e2: guard [x>0] 1>0=TRUE, action y=1*2=2, go to C, ctx={x:1, y:2}
- At C, event=e3: guard [y==2] 2==2=TRUE, go to D
Final state: D
Answer: D

Example 5 (PRIORITY):
Transitions:
S --e[x>5]--> A (priority 0)
S --e[x>0]--> B (priority 1)
Context: x=10
Events: e
Trace:
- At S, event=e
- Guard [x>5]: 10>5 → TRUE (priority 0)
- Guard [x>0]: 10>0 → TRUE (priority 1)
- Both true, select priority 0
- Take S→A
Final state: A
Answer: A

"""

    prompt = f"""{examples}Now solve:
Transitions:
{trans_str}
Context: {ctx_str}
Active states: {active_str}
Events: {events_str}
Trace:
-"""

    return prompt


def create_context_mutation_prompt(case: GuardTestCase) -> str:
    """Create prompt for context mutation prediction."""
    trans_str = format_transitions(case.transitions)
    ctx_str = ", ".join(f"{k}={v}" for k, v in case.initial_context.items())
    events_str = ", ".join(case.event_sequence)

    examples = """Track context mutations through transitions.

Example 1:
Transitions:
A --e1/x=1--> B
B --e2/y=x+1--> C
Context: x=0, y=0
Events: e1, e2
Trace:
- e1: action x=1 → {x:1, y:0}
- e2: action y=x+1=2 → {x:1, y:2}
Final context: {x:1, y:2}
Answer: x=1, y=2

Example 2:
Transitions:
Counter --inc/count++--> Counter
Counter --reset/count=0--> Counter
Context: count=5
Events: inc, inc, reset
Trace:
- inc: count++ → {count:6}
- inc: count++ → {count:7}
- reset: count=0 → {count:0}
Final context: {count:0}
Answer: count=0

"""

    prompt = f"""{examples}Now solve:
Transitions:
{trans_str}
Context: {ctx_str}
Events: {events_str}
Trace:
-"""

    return prompt


def parse_state_response(output: str) -> str:
    """Parse final state from model output."""
    output = output.strip()

    # Look for "Answer: State" or "Final state: State"
    match = re.search(r'(?:answer|final state)[:\s]+([A-Za-z][A-Za-z0-9_]*)', output, re.IGNORECASE)
    if match:
        return match.group(1)

    # Look for state name at end of line
    lines = output.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'\b([A-Z][a-zA-Z0-9_]*)\s*$', line)
        if match:
            return match.group(1)

    return ""


def parse_context_response(output: str) -> Dict[str, Any]:
    """Parse final context from model output."""
    result = {}

    # Look for "Answer: x=1, y=2" pattern
    match = re.search(r'answer[:\s]+(.+)', output, re.IGNORECASE)
    if match:
        ctx_str = match.group(1)
        for pair in ctx_str.split(','):
            if '=' in pair:
                key, val = pair.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    result[key] = eval(val)
                except Exception:
                    result[key] = val

    return result


# ============================================================================
# TEST CASES
# ============================================================================

def get_test_cases() -> List[GuardTestCase]:
    """Generate comprehensive test cases for all guard categories."""
    cases = []

    # TC1: Simple boolean guard
    cases.append(GuardTestCase(
        category=GuardCategory.BOOLEAN,
        description="Simple boolean guard (has_key)",
        transitions=[
            Transition("Locked", "Unlocked", "unlock", guard="has_key"),
            Transition("Locked", "Locked", "unlock", guard="!has_key"),
        ],
        initial_context={"has_key": True},
        event_sequence=["unlock"],
        expected_final_state="Unlocked",
        expected_final_context={"has_key": True},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.BOOLEAN,
        description="Boolean guard false case",
        transitions=[
            Transition("Locked", "Unlocked", "unlock", guard="has_key"),
            Transition("Locked", "Locked", "unlock", guard="!has_key"),
        ],
        initial_context={"has_key": False},
        event_sequence=["unlock"],
        expected_final_state="Locked",
        expected_final_context={"has_key": False},
    ))

    # TC2: Numeric comparison
    cases.append(GuardTestCase(
        category=GuardCategory.COMPARISON,
        description="Numeric less-than guard",
        transitions=[
            Transition("Counter", "Counter", "inc", guard="count<max", action="count++"),
            Transition("Counter", "Full", "inc", guard="count>=max"),
        ],
        initial_context={"count": 2, "max": 3},
        event_sequence=["inc", "inc"],
        expected_final_state="Full",
        expected_final_context={"count": 3, "max": 3},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.COMPARISON,
        description="Greater-than comparison",
        transitions=[
            Transition("Temp", "Hot", "check", guard="temp>100"),
            Transition("Temp", "Cold", "check", guard="temp<=100"),
        ],
        initial_context={"temp": 150},
        event_sequence=["check"],
        expected_final_state="Hot",
        expected_final_context={"temp": 150},
    ))

    # TC3: Compound guards (AND/OR/NOT)
    cases.append(GuardTestCase(
        category=GuardCategory.COMPOUND,
        description="Compound AND guard",
        transitions=[
            Transition("Door", "Open", "open", guard="unlocked && !alarm"),
            Transition("Door", "Door", "open", guard="!(unlocked && !alarm)"),
        ],
        initial_context={"unlocked": True, "alarm": False},
        event_sequence=["open"],
        expected_final_state="Open",
        expected_final_context={"unlocked": True, "alarm": False},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.COMPOUND,
        description="Compound OR guard",
        transitions=[
            Transition("Access", "Granted", "try", guard="admin || has_pass"),
            Transition("Access", "Denied", "try", guard="!(admin || has_pass)"),
        ],
        initial_context={"admin": False, "has_pass": True},
        event_sequence=["try"],
        expected_final_state="Granted",
        expected_final_context={"admin": False, "has_pass": True},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.COMPOUND,
        description="Complex compound guard",
        transitions=[
            Transition("Gate", "Pass", "enter", guard="(vip || paid) && !blocked"),
            Transition("Gate", "Reject", "enter", guard="!((vip || paid) && !blocked)"),
        ],
        initial_context={"vip": False, "paid": True, "blocked": False},
        event_sequence=["enter"],
        expected_final_state="Pass",
        expected_final_context={"vip": False, "paid": True, "blocked": False},
    ))

    # TC4: Guard-action chains
    cases.append(GuardTestCase(
        category=GuardCategory.CHAIN,
        description="Simple guard-action chain",
        transitions=[
            Transition("A", "B", "e1", action="x=1"),
            Transition("B", "C", "e2", guard="x>0", action="y=x*2"),
            Transition("C", "D", "e3", guard="y==2"),
        ],
        initial_context={"x": 0, "y": 0},
        event_sequence=["e1", "e2", "e3"],
        expected_final_state="D",
        expected_final_context={"x": 1, "y": 2},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.CHAIN,
        description="Chain with conditional branch",
        transitions=[
            Transition("Start", "Check", "go", action="val=5"),
            Transition("Check", "High", "eval", guard="val>3"),
            Transition("Check", "Low", "eval", guard="val<=3"),
        ],
        initial_context={"val": 0},
        event_sequence=["go", "eval"],
        expected_final_state="High",
        expected_final_context={"val": 5},
    ))

    # TC5: Transition priority with overlapping guards
    cases.append(GuardTestCase(
        category=GuardCategory.PRIORITY,
        description="Priority: more specific wins",
        transitions=[
            Transition("S", "A", "e", guard="x>5", priority=0),
            Transition("S", "B", "e", guard="x>0", priority=1),
        ],
        initial_context={"x": 10},
        event_sequence=["e"],
        expected_final_state="A",
        expected_final_context={"x": 10},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.PRIORITY,
        description="Priority: fallback when first fails",
        transitions=[
            Transition("S", "A", "e", guard="x>5", priority=0),
            Transition("S", "B", "e", guard="x>0", priority=1),
        ],
        initial_context={"x": 3},
        event_sequence=["e"],
        expected_final_state="B",
        expected_final_context={"x": 3},
    ))

    # TC6: In-state guards (parallel regions)
    cases.append(GuardTestCase(
        category=GuardCategory.IN_STATE,
        description="In-state guard (parallel)",
        transitions=[
            Transition("X", "Y", "sync", guard="in(B)"),
            Transition("X", "X", "sync", guard="!in(B)"),
        ],
        initial_context={},
        event_sequence=["sync"],
        expected_final_state="Y",
        expected_final_context={},
        active_states={"X", "B"},  # Parallel: X and B both active
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.IN_STATE,
        description="In-state guard false",
        transitions=[
            Transition("X", "Y", "sync", guard="in(B)"),
            Transition("X", "X", "sync", guard="!in(B)"),
        ],
        initial_context={},
        event_sequence=["sync"],
        expected_final_state="X",
        expected_final_context={},
        active_states={"X", "A"},  # A is active, not B
    ))

    # TC7: Nested object access
    cases.append(GuardTestCase(
        category=GuardCategory.NESTED,
        description="Nested object access",
        transitions=[
            Transition("Check", "Admin", "auth", guard='user.role == "admin"'),
            Transition("Check", "User", "auth", guard='user.role != "admin"'),
        ],
        initial_context={"user": {"role": "admin", "name": "Alice"}},
        event_sequence=["auth"],
        expected_final_state="Admin",
        expected_final_context={"user": {"role": "admin", "name": "Alice"}},
    ))

    cases.append(GuardTestCase(
        category=GuardCategory.NESTED,
        description="Deep nested access",
        transitions=[
            Transition("Init", "Valid", "check", guard="config.auth.enabled"),
            Transition("Init", "Invalid", "check", guard="!config.auth.enabled"),
        ],
        initial_context={"config": {"auth": {"enabled": True}}},
        event_sequence=["check"],
        expected_final_state="Valid",
        expected_final_context={"config": {"auth": {"enabled": True}}},
    ))

    # TC8: Array/collection guards
    cases.append(GuardTestCase(
        category=GuardCategory.ARRAY,
        description="Array length check",
        transitions=[
            Transition("Queue", "Process", "next", guard="items.length > 0"),
            Transition("Queue", "Empty", "next", guard="items.length == 0"),
        ],
        initial_context={"items": {"length": 3}},  # Simplified: items.length
        event_sequence=["next"],
        expected_final_state="Process",
        expected_final_context={"items": {"length": 3}},
    ))

    # TC9: Guard with action side-effect
    cases.append(GuardTestCase(
        category=GuardCategory.CHAIN,
        description="Action affects subsequent guard",
        transitions=[
            Transition("S1", "S2", "e", action="ready=1"),
            Transition("S2", "S3", "f", guard="ready==1"),
            Transition("S2", "S2", "f", guard="ready!=1"),
        ],
        initial_context={"ready": 0},
        event_sequence=["e", "f"],
        expected_final_state="S3",
        expected_final_context={"ready": 1},
    ))

    # TC10: Multiple guards same event (conflict resolution)
    cases.append(GuardTestCase(
        category=GuardCategory.PRIORITY,
        description="Three-way priority",
        transitions=[
            Transition("S", "A", "e", guard="level>=10", priority=0),
            Transition("S", "B", "e", guard="level>=5", priority=1),
            Transition("S", "C", "e", guard="level>=0", priority=2),
        ],
        initial_context={"level": 7},
        event_sequence=["e"],
        expected_final_state="B",  # 7>=10 false, 7>=5 true
        expected_final_context={"level": 7},
    ))

    return cases


if __name__ == "__main__":
    # Test evaluation functions
    print("Context Guards Test")
    print("=" * 60)

    cases = get_test_cases()
    for case in cases[:3]:
        print(f"\n{case.category.name}: {case.description}")
        state, ctx = simulate_machine(
            case.transitions,
            case.transitions[0].source,
            case.initial_context,
            case.event_sequence,
            case.active_states,
        )
        print(f"  Expected: {case.expected_final_state}, Got: {state}")
        print(f"  Context: {ctx}")
