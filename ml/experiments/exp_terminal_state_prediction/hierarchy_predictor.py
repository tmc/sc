"""
Hierarchy Predictor - Terminal state prediction with cascade semantics.

Teaches LLM to understand:
1. Composite states contain children (OR-decomposition)
2. When entering a composite state, you enter its initial child
3. Configurations only contain LEAF states
4. When leaving a composite state, all descendant states are exited
"""

import json
import re
from typing import Set, List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class HierarchyPredictionResult:
    """Result of hierarchy-aware prediction."""
    predicted_states: Set[str]
    raw_output: str
    parse_success: bool
    method: str  # 'cascade', 'viz', 'baseline'


def build_ascii_tree(sc_json: dict) -> str:
    """Build ASCII tree showing state hierarchy."""
    lines = []

    def render_state(state: dict, prefix: str = "", is_last: bool = True):
        label = state.get("label", "")
        if label.startswith("__"):
            # Root - just render children
            children = state.get("children", [])
            for i, child in enumerate(children):
                render_state(child, prefix, i == len(children) - 1)
            return

        state_type = state.get("type", 1)
        is_initial = state.get("is_initial", False)

        # State type indicator
        type_marker = ""
        if state_type == 2:
            type_marker = " [OR]"
        elif state_type == 3:
            type_marker = " [AND]"

        initial_marker = " *" if is_initial else ""

        # Connector
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{label}{type_marker}{initial_marker}")

        # Render children
        children = state.get("children", [])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            render_state(child, new_prefix, i == len(children) - 1)

    render_state(sc_json.get("root_state", {}))
    return "\n".join(lines)


def get_state_info(sc_json: dict) -> Dict[str, dict]:
    """Extract state information including parent/child relationships."""
    states = {}

    def collect(state: dict, parent: Optional[str] = None):
        label = state.get("label", "")
        if label and not label.startswith("__"):
            children = [c.get("label") for c in state.get("children", []) if c.get("label")]
            initial_child = None
            for c in state.get("children", []):
                if c.get("is_initial"):
                    initial_child = c.get("label")
                    break
            if children and not initial_child:
                initial_child = children[0]

            states[label] = {
                "label": label,
                "type": state.get("type", 1),
                "parent": parent,
                "children": children,
                "is_initial": state.get("is_initial", False),
                "initial_child": initial_child,
                "is_leaf": len(children) == 0,
            }

        for child in state.get("children", []):
            collect(child, label if label and not label.startswith("__") else parent)

    collect(sc_json.get("root_state", {}))
    return states


def get_leaf_descendants(states: Dict[str, dict], state_label: str) -> Set[str]:
    """Get all leaf state descendants of a composite state."""
    info = states.get(state_label, {})
    if info.get("is_leaf", True):
        return {state_label}

    result = set()
    for child in info.get("children", []):
        result |= get_leaf_descendants(states, child)
    return result


class HierarchyPredictor:
    """Predicts terminal states with hierarchy-aware prompting."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the LLM model."""
        if self.model is None and MLX_AVAILABLE:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded.")

    def _build_cascade_prompt(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
    ) -> str:
        """Build prompt with explicit cascade semantics."""
        states = get_state_info(sc_json)

        # Format transitions with cascade explanation
        rules = []
        for t in sc_json.get("transitions", []):
            from_s = t.get("from", ["?"])[0]
            to_s = t.get("to", ["?"])[0]
            event = t.get("event", "?")

            from_info = states.get(from_s, {})
            to_info = states.get(to_s, {})

            # Add cascade info for composite targets
            if to_info.get("children"):
                initial = to_info.get("initial_child", "?")
                rules.append(f"{from_s} + {event} -> {to_s} (enter {initial})")
            else:
                rules.append(f"{from_s} + {event} -> {to_s}")

        rules_str = "\n".join(rules)

        # Find composite states to explain
        composite_states = [
            label for label, info in states.items()
            if not info.get("is_leaf") and info.get("type") == 2
        ]

        cascade_explanation = ""
        if composite_states:
            explanations = []
            for cs in composite_states[:3]:  # Limit to 3
                info = states[cs]
                children = info.get("children", [])
                initial = info.get("initial_child")
                explanations.append(
                    f"- {cs} contains: {', '.join(children)}. "
                    f"Entering {cs} means entering {initial} (its initial child)"
                )
            cascade_explanation = "\n".join(explanations)

        prompt = f"""Execute state transitions with HIERARCHY semantics.

CRITICAL RULES:
1. Configurations contain ONLY leaf states (states with no children)
2. When entering a composite state, ALWAYS enter its initial child
3. The final answer must be the LEAF state, not composite state

{f"Hierarchy structure:{chr(10)}{cascade_explanation}" if cascade_explanation else ""}

Example with hierarchy:
States: Idle, Active (composite with Running*, Paused), Done
Rules: Idle + START -> Active (enter Running)
       Running + PAUSE -> Paused
       Active + STOP -> Done

Start: Idle, Events: START
Step 1: Idle + START -> Active (enter Running)
Final: Running  <-- NOT "Active"! Running is the leaf state.

Now solve:
Rules:
{rules_str}

Start: {initial_state}
Events: {', '.join(events)}

Trace each step. Answer with LEAF state(s) only:"""

        return prompt

    def _build_viz_prompt(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
    ) -> str:
        """Build prompt with ASCII tree visualization."""
        states = get_state_info(sc_json)
        tree = build_ascii_tree(sc_json)

        # Format transitions
        rules = []
        for t in sc_json.get("transitions", []):
            from_s = t.get("from", ["?"])[0]
            to_s = t.get("to", ["?"])[0]
            event = t.get("event", "?")
            rules.append(f"{from_s} + {event} -> {to_s}")
        rules_str = "\n".join(rules)

        # Mark leaf vs composite states
        leaf_states = [l for l, i in states.items() if i.get("is_leaf")]
        composite_states = [l for l, i in states.items() if not i.get("is_leaf")]

        prompt = f"""Execute state transitions on this hierarchical state machine.

STATE HIERARCHY (* = initial):
{tree}

LEAF states (valid final states): {', '.join(leaf_states)}
COMPOSITE states (NOT valid final states): {', '.join(composite_states) or 'none'}

RULE: When entering a composite state, cascade to its initial (*) child until reaching a leaf.

Transitions:
{rules_str}

Start: {initial_state}
Events: {', '.join(events)}

Execute step by step. Final answer must be from LEAF states only:"""

        return prompt

    def _build_stepwise_prompt(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
    ) -> str:
        """Build prompt with explicit step-by-step cascade reasoning."""
        states = get_state_info(sc_json)

        # Format rules
        rules = []
        for t in sc_json.get("transitions", []):
            from_s = t.get("from", ["?"])[0]
            to_s = t.get("to", ["?"])[0]
            event = t.get("event", "?")
            rules.append(f"{from_s} + {event} -> {to_s}")
        rules_str = "\n".join(rules)

        # Build entry cascade map
        entry_cascade = {}
        for label, info in states.items():
            if not info.get("is_leaf"):
                # Trace down to leaf
                current = label
                path = [current]
                while current:
                    current_info = states.get(current, {})
                    initial = current_info.get("initial_child")
                    if initial:
                        path.append(initial)
                        current = initial
                    else:
                        break
                if len(path) > 1:
                    entry_cascade[label] = path

        cascade_info = ""
        if entry_cascade:
            lines = []
            for composite, path in entry_cascade.items():
                lines.append(f"  {composite} -> {' -> '.join(path[1:])} (cascade)")
            cascade_info = "Entry cascades:\n" + "\n".join(lines)

        prompt = f"""Execute transitions with hierarchical state entry.

Rules:
{rules_str}

{cascade_info if cascade_info else ""}

IMPORTANT: When entering a composite state, follow the cascade to the leaf.
Example: If "Active" has initial child "Running", entering Active means entering Running.

Start: {initial_state}
Events: {', '.join(events)}

Show each step including cascades. Final state must be a LEAF:"""

        return prompt

    def predict(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
        method: str = "cascade",
        max_tokens: int = 250,
    ) -> HierarchyPredictionResult:
        """Predict terminal configuration with hierarchy awareness."""
        self.load_model()

        if not self.model or not self.tokenizer:
            return self._fallback_predict(sc_json, initial_state, events, method)

        # Select prompt method
        if method == "cascade":
            prompt = self._build_cascade_prompt(sc_json, initial_state, events)
        elif method == "viz":
            prompt = self._build_viz_prompt(sc_json, initial_state, events)
        elif method == "stepwise":
            prompt = self._build_stepwise_prompt(sc_json, initial_state, events)
        else:
            prompt = self._build_cascade_prompt(sc_json, initial_state, events)

        # Format for chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Parse output
        predicted, parse_success = self._parse_output(output, sc_json)

        return HierarchyPredictionResult(
            predicted_states=predicted,
            raw_output=output,
            parse_success=parse_success,
            method=method,
        )

    def _parse_output(
        self,
        output: str,
        sc_json: dict,
    ) -> Tuple[Set[str], bool]:
        """Parse LLM output to extract predicted states."""
        states = get_state_info(sc_json)
        leaf_states = {l for l, i in states.items() if i.get("is_leaf")}
        all_states = set(states.keys())

        found = set()

        # Method 1: Look for "Final:" line
        final_match = re.search(r'[Ff]inal[:\s]+(\w+)', output)
        if final_match:
            state = final_match.group(1)
            if state in leaf_states:
                return {state}, True
            # If composite, cascade to leaf
            if state in states and not states[state].get("is_leaf"):
                leaf = self._cascade_to_leaf(states, state)
                if leaf:
                    return {leaf}, True

        # Method 2: "(now at X)" pattern
        now_matches = re.findall(r'\(now at (\w+)\)', output)
        if now_matches:
            state = now_matches[-1]
            if state in leaf_states:
                return {state}, True
            if state in states:
                leaf = self._cascade_to_leaf(states, state)
                if leaf:
                    return {leaf}, True

        # Method 3: Last line with state
        lines = output.strip().split('\n')
        for line in reversed(lines):
            for state in leaf_states:
                if re.search(rf'\b{re.escape(state)}\b', line):
                    return {state}, True

        # Method 4: Any leaf state mentioned
        for state in leaf_states:
            if re.search(rf'\b{re.escape(state)}\b', output):
                found.add(state)

        if found:
            return found, True

        # Method 5: Any state, then cascade
        for state in all_states:
            if re.search(rf'\b{re.escape(state)}\b', output):
                leaf = self._cascade_to_leaf(states, state)
                if leaf:
                    return {leaf}, True

        return set(), False

    def _cascade_to_leaf(self, states: Dict[str, dict], state_label: str) -> Optional[str]:
        """Cascade from composite to leaf state."""
        current = state_label
        visited = set()
        while current and current not in visited:
            visited.add(current)
            info = states.get(current, {})
            if info.get("is_leaf"):
                return current
            initial = info.get("initial_child")
            if initial:
                current = initial
            else:
                break
        return None

    def _fallback_predict(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
        method: str,
    ) -> HierarchyPredictionResult:
        """Fallback to execution-based prediction."""
        from . import TraceExecutor

        executor = TraceExecutor(sc_json)
        final = executor.execute_trace(events)

        return HierarchyPredictionResult(
            predicted_states=final,
            raw_output="[fallback execution]",
            parse_success=True,
            method=method,
        )


def demo():
    """Demonstrate hierarchy predictor."""
    from . import HIERARCHY_SC, NESTED_HIERARCHY_SC

    print("=" * 70)
    print("Hierarchy Predictor Demo")
    print("=" * 70)

    predictor = HierarchyPredictor()

    test_cases = [
        (HIERARCHY_SC, "Idle", ["START"]),  # Should be Running, not Active
        (HIERARCHY_SC, "Idle", ["START", "PAUSE"]),  # Should be Paused
        (HIERARCHY_SC, "Idle", ["START", "STOP"]),  # Should be Done
        (NESTED_HIERARCHY_SC, "Off", ["POWER"]),  # Should be Waiting
    ]

    for sc, initial, events in test_cases:
        print(f"\n--- {sc['name']}: {initial} + {events} ---")
        print(f"Tree:\n{build_ascii_tree(sc)}")

        for method in ["cascade", "viz", "stepwise"]:
            result = predictor.predict(sc, initial, events, method=method)
            print(f"\n  [{method}] Predicted: {result.predicted_states}")
            if not result.parse_success:
                print(f"    Raw: {result.raw_output[:100]}...")


if __name__ == "__main__":
    demo()
