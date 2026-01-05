"""
Action Grammar Extension for Statecharts.

Extends SC JSON grammar to support entry and exit actions:
- on_entry: Actions executed when entering a state
- on_exit: Actions executed when leaving a state
- action: Actions executed during a transition

Action format: "function_name(args)" or ["action1()", "action2()"]
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any


# Extended state fields to include actions
EXTENDED_STATE_FIELDS = {
    "label", "type", "children", "is_initial",
    "on_entry", "on_exit"  # NEW: action fields
}

# Extended transition fields
EXTENDED_TRANS_FIELDS = {
    "from", "to", "event", "guard", "action"
}

# Valid action patterns
ACTION_PATTERN = re.compile(r'^[a-z_][a-z0-9_]*\([^)]*\)$', re.IGNORECASE)


@dataclass
class ActionValidation:
    """Result of action validation."""
    is_valid: bool
    action_type: str  # "entry", "exit", "transition", "chained"
    actions: List[str]
    errors: List[str] = field(default_factory=list)


def validate_action_syntax(action: str) -> bool:
    """Check if action has valid syntax: function_name(args)."""
    return bool(ACTION_PATTERN.match(action.strip()))


def validate_actions_list(actions: Any) -> Tuple[bool, List[str]]:
    """Validate a list of actions or single action."""
    if actions is None:
        return True, []

    if isinstance(actions, str):
        actions = [actions]

    if not isinstance(actions, list):
        return False, [f"Actions must be string or list, got {type(actions).__name__}"]

    errors = []
    for i, action in enumerate(actions):
        if not isinstance(action, str):
            errors.append(f"Action {i}: must be string")
        elif not validate_action_syntax(action):
            errors.append(f"Action {i}: invalid syntax '{action}'")

    return len(errors) == 0, errors


def validate_state_actions(state: Dict) -> ActionValidation:
    """Validate actions in a state definition."""
    errors = []
    action_type = "none"
    all_actions = []
    has_entry = False
    has_exit = False
    entry_count = 0
    exit_count = 0

    # Check on_entry
    on_entry = state.get("on_entry")
    if on_entry:
        valid, errs = validate_actions_list(on_entry)
        if not valid:
            errors.extend([f"on_entry: {e}" for e in errs])
        else:
            has_entry = True
            if isinstance(on_entry, str):
                all_actions.append(on_entry)
                entry_count = 1
            else:
                all_actions.extend(on_entry)
                entry_count = len(on_entry)

    # Check on_exit
    on_exit = state.get("on_exit")
    if on_exit:
        valid, errs = validate_actions_list(on_exit)
        if not valid:
            errors.extend([f"on_exit: {e}" for e in errs])
        else:
            has_exit = True
            if isinstance(on_exit, str):
                all_actions.append(on_exit)
                exit_count = 1
            else:
                all_actions.extend(on_exit)
                exit_count = len(on_exit)

    # Determine action type - "both" takes priority when entry AND exit present
    if has_entry and has_exit:
        action_type = "both"
    elif has_entry:
        action_type = "entry"
    elif has_exit:
        action_type = "exit"

    # "chained" only applies when single type has multiple actions
    if action_type == "entry" and entry_count > 1:
        action_type = "chained"
    elif action_type == "exit" and exit_count > 1:
        action_type = "chained"

    return ActionValidation(
        is_valid=len(errors) == 0,
        action_type=action_type,
        actions=all_actions,
        errors=errors
    )


def validate_transition_action(trans: Dict) -> ActionValidation:
    """Validate action in a transition."""
    action = trans.get("action")
    if not action:
        return ActionValidation(is_valid=True, action_type="none", actions=[])

    valid, errors = validate_actions_list(action)
    actions = [action] if isinstance(action, str) else (action or [])

    return ActionValidation(
        is_valid=valid,
        action_type="transition" if valid and actions else "none",
        actions=actions,
        errors=errors
    )


def validate_sc_with_actions(sc: Dict) -> Dict[str, Any]:
    """Validate entire statechart for action correctness."""
    results = {
        "valid": True,
        "entry_count": 0,
        "exit_count": 0,
        "both_count": 0,
        "transition_action_count": 0,
        "chained_count": 0,
        "errors": [],
        "states_with_actions": [],
        "transitions_with_actions": [],
    }

    def check_state(state: Dict, path: str = "root"):
        """Recursively check states."""
        val = validate_state_actions(state)
        if not val.is_valid:
            results["valid"] = False
            results["errors"].extend([f"{path}: {e}" for e in val.errors])

        if val.action_type == "entry":
            results["entry_count"] += 1
            results["states_with_actions"].append((path, "entry", val.actions))
        elif val.action_type == "exit":
            results["exit_count"] += 1
            results["states_with_actions"].append((path, "exit", val.actions))
        elif val.action_type == "both":
            results["both_count"] += 1
            results["states_with_actions"].append((path, "both", val.actions))
        elif val.action_type == "chained":
            results["chained_count"] += 1
            results["states_with_actions"].append((path, "chained", val.actions))

        # Recurse into children
        children = state.get("children", [])
        for i, child in enumerate(children):
            child_path = f"{path}.children[{i}]"
            check_state(child, child_path)

    # Check root state
    root = sc.get("root_state", {})
    check_state(root)

    # Check transitions
    transitions = sc.get("transitions", [])
    for i, trans in enumerate(transitions):
        val = validate_transition_action(trans)
        if not val.is_valid:
            results["valid"] = False
            results["errors"].extend([f"transition[{i}]: {e}" for e in val.errors])

        if val.action_type == "transition":
            results["transition_action_count"] += 1
            results["transitions_with_actions"].append((i, val.actions))

    return results


# Example SCs with actions for few-shot prompting
ENTRY_ACTION_EXAMPLE = {
    "root_state": {
        "label": "Loading",
        "type": 2,
        "children": [
            {
                "label": "Idle",
                "type": 1,
                "is_initial": True,
                "on_entry": ["reset_state()"]
            },
            {
                "label": "Loading",
                "type": 1,
                "on_entry": ["start_spinner()", "fetch_data()"]
            },
            {
                "label": "Done",
                "type": 1,
                "on_entry": ["log(completed)"]
            }
        ]
    },
    "transitions": [
        {"from": ["Idle"], "to": ["Loading"], "event": "START"},
        {"from": ["Loading"], "to": ["Done"], "event": "COMPLETE"}
    ]
}

EXIT_ACTION_EXAMPLE = {
    "root_state": {
        "label": "Session",
        "type": 2,
        "children": [
            {
                "label": "Active",
                "type": 1,
                "is_initial": True,
                "on_exit": ["save_state()"]
            },
            {
                "label": "Inactive",
                "type": 1,
                "on_exit": ["cleanup()"]
            }
        ]
    },
    "transitions": [
        {"from": ["Active"], "to": ["Inactive"], "event": "TIMEOUT"}
    ]
}

BOTH_ACTIONS_EXAMPLE = {
    "root_state": {
        "label": "Timer",
        "type": 2,
        "children": [
            {
                "label": "Stopped",
                "type": 1,
                "is_initial": True
            },
            {
                "label": "Running",
                "type": 1,
                "on_entry": ["start_timer()"],
                "on_exit": ["stop_timer()"]
            }
        ]
    },
    "transitions": [
        {"from": ["Stopped"], "to": ["Running"], "event": "START"},
        {"from": ["Running"], "to": ["Stopped"], "event": "STOP"}
    ]
}

TRANSITION_ACTION_EXAMPLE = {
    "root_state": {
        "label": "Counter",
        "type": 2,
        "children": [
            {
                "label": "Ready",
                "type": 1,
                "is_initial": True
            }
        ]
    },
    "transitions": [
        {"from": ["Ready"], "to": ["Ready"], "event": "INCREMENT", "action": "counter_add(1)"},
        {"from": ["Ready"], "to": ["Ready"], "event": "DECREMENT", "action": "counter_sub(1)"}
    ]
}

CHAINED_ACTIONS_EXAMPLE = {
    "root_state": {
        "label": "Process",
        "type": 2,
        "children": [
            {
                "label": "Init",
                "type": 1,
                "is_initial": True,
                "on_entry": ["log(starting)", "init_resources()", "notify(ready)"]
            },
            {
                "label": "Running",
                "type": 1,
                "on_entry": ["start_process()", "monitor_begin()"],
                "on_exit": ["stop_process()", "monitor_end()", "cleanup()"]
            }
        ]
    },
    "transitions": [
        {"from": ["Init"], "to": ["Running"], "event": "START", "action": "record_time()"}
    ]
}


def get_few_shot_examples() -> str:
    """Get few-shot examples for prompting."""
    examples = [
        ("Entry action (start spinner on Loading)", json.dumps(ENTRY_ACTION_EXAMPLE, separators=(',', ':'))),
        ("Exit action (save on leaving Active)", json.dumps(EXIT_ACTION_EXAMPLE, separators=(',', ':'))),
        ("Both entry and exit (timer start/stop)", json.dumps(BOTH_ACTIONS_EXAMPLE, separators=(',', ':'))),
        ("Transition action (increment counter)", json.dumps(TRANSITION_ACTION_EXAMPLE, separators=(',', ':'))),
    ]

    lines = []
    for desc, sc in examples:
        lines.append(f"Example: {desc}")
        lines.append(sc)
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test validation
    print("Testing action validation...")

    for name, sc in [
        ("Entry", ENTRY_ACTION_EXAMPLE),
        ("Exit", EXIT_ACTION_EXAMPLE),
        ("Both", BOTH_ACTIONS_EXAMPLE),
        ("Transition", TRANSITION_ACTION_EXAMPLE),
        ("Chained", CHAINED_ACTIONS_EXAMPLE),
    ]:
        result = validate_sc_with_actions(sc)
        print(f"\n{name}: valid={result['valid']}")
        print(f"  entry={result['entry_count']}, exit={result['exit_count']}, both={result['both_count']}")
        print(f"  transition_actions={result['transition_action_count']}, chained={result['chained_count']}")
