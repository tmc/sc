"""
Detail Generator: Detailed Transition and State Explanations

Generates detailed, context-aware explanations of:
- Individual transitions (trigger, guard, action, effect)
- Individual states (role, behavior, connections)
- Sequences and paths through the statechart
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto

# MLX imports
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class TransitionType(Enum):
    """Classification of transition types."""
    SIMPLE = "simple"          # Basic A -> B
    GUARDED = "guarded"        # A -> B [guard]
    TRIGGERED = "triggered"    # A -> B on event
    COMPOUND = "compound"      # Multiple sources or targets
    SELF = "self"              # A -> A
    COMPLETION = "completion"  # A -> B (no event)


class StateRole(Enum):
    """Role of a state in the statechart."""
    INITIAL = "initial"        # Entry point
    FINAL = "final"            # Exit point
    JUNCTION = "junction"      # High connectivity
    PROCESSING = "processing"  # Active work state
    WAITING = "waiting"        # Waiting for event
    ERROR = "error"            # Error/exception state


@dataclass
class TransitionExplanation:
    """Detailed explanation of a single transition."""
    transition_id: str
    source_states: List[str]
    target_states: List[str]
    event: str
    guard: str
    action: str

    # Classification
    transition_type: TransitionType

    # Natural language
    short_description: str    # One line
    full_description: str     # Paragraph
    preconditions: List[str]  # What must be true before
    postconditions: List[str] # What is true after
    side_effects: List[str]   # Observable effects

    def to_text(self) -> str:
        """Convert to readable text."""
        lines = [f"**{self.short_description}**"]
        lines.append(f"")
        lines.append(self.full_description)

        if self.preconditions:
            lines.append(f"")
            lines.append("Preconditions:")
            for p in self.preconditions:
                lines.append(f"  - {p}")

        if self.postconditions:
            lines.append(f"")
            lines.append("Postconditions:")
            for p in self.postconditions:
                lines.append(f"  - {p}")

        return "\n".join(lines)


@dataclass
class StateExplanation:
    """Detailed explanation of a single state."""
    state_label: str
    state_type: int           # 1=BASIC, 2=NORMAL, 3=PARALLEL
    parent: Optional[str]

    # Classification
    role: StateRole

    # Connections
    incoming_transitions: List[str]
    outgoing_transitions: List[str]
    incoming_events: List[str]
    outgoing_events: List[str]

    # Natural language
    short_description: str
    full_description: str
    behavior_while_active: str
    entry_conditions: List[str]
    exit_conditions: List[str]

    def to_text(self) -> str:
        """Convert to readable text."""
        lines = [f"**State: {self.state_label}**"]
        lines.append(f"Role: {self.role.value}")
        lines.append(f"")
        lines.append(self.full_description)

        if self.behavior_while_active:
            lines.append(f"")
            lines.append(f"While active: {self.behavior_while_active}")

        lines.append(f"")
        lines.append(f"Incoming: {', '.join(self.incoming_events) or 'none'}")
        lines.append(f"Outgoing: {', '.join(self.outgoing_events) or 'none'}")

        return "\n".join(lines)


class DetailGenerator:
    """
    Generate detailed explanations of statechart elements.

    Uses structured analysis and LLM enhancement for natural language.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if MLX_AVAILABLE:
            try:
                self.model, self.tokenizer = load(model_name)
            except Exception:
                pass

    def explain_transition(
        self,
        transition: Dict[str, Any],
        context: Dict[str, Any],
    ) -> TransitionExplanation:
        """Generate detailed explanation for a transition."""
        sources = transition.get("from", [])
        targets = transition.get("to", [])
        event = transition.get("event", "")
        guard = transition.get("guard", "")
        action = transition.get("action", "")

        # Classify transition type
        trans_type = self._classify_transition(sources, targets, event, guard)

        # Generate descriptions
        short_desc = self._generate_transition_short(sources, targets, event, guard)
        full_desc = self._generate_transition_full(
            sources, targets, event, guard, action, trans_type
        )

        # Infer conditions
        preconditions = self._infer_preconditions(sources, guard, context)
        postconditions = self._infer_postconditions(targets, action, context)
        side_effects = self._infer_side_effects(action)

        return TransitionExplanation(
            transition_id=f"{'-'.join(sources)}_to_{'-'.join(targets)}",
            source_states=sources,
            target_states=targets,
            event=event,
            guard=guard,
            action=action,
            transition_type=trans_type,
            short_description=short_desc,
            full_description=full_desc,
            preconditions=preconditions,
            postconditions=postconditions,
            side_effects=side_effects,
        )

    def explain_state(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StateExplanation:
        """Generate detailed explanation for a state."""
        label = state.get("label", "")
        state_type = state.get("type", 1)
        parent = state.get("parent")
        is_initial = state.get("is_initial", False)
        is_final = state.get("is_final", False)

        # Find connections
        incoming = []
        outgoing = []
        in_events = []
        out_events = []

        for t in context.get("transitions", []):
            if label in t.get("to", []):
                src = ", ".join(t.get("from", []))
                incoming.append(f"from {src}")
                if t.get("event"):
                    in_events.append(t["event"])
            if label in t.get("from", []):
                tgt = ", ".join(t.get("to", []))
                outgoing.append(f"to {tgt}")
                if t.get("event"):
                    out_events.append(t["event"])

        # Classify role
        role = self._classify_state_role(
            label, state_type, is_initial, is_final,
            len(incoming), len(outgoing), context
        )

        # Generate descriptions
        short_desc = self._generate_state_short(label, role)
        full_desc = self._generate_state_full(label, role, state_type, context)
        behavior = self._infer_state_behavior(label, role, out_events)

        # Infer conditions
        entry = self._infer_entry_conditions(label, incoming, in_events)
        exit = self._infer_exit_conditions(label, outgoing, out_events)

        return StateExplanation(
            state_label=label,
            state_type=state_type,
            parent=parent,
            role=role,
            incoming_transitions=incoming,
            outgoing_transitions=outgoing,
            incoming_events=list(set(in_events)),
            outgoing_events=list(set(out_events)),
            short_description=short_desc,
            full_description=full_desc,
            behavior_while_active=behavior,
            entry_conditions=entry,
            exit_conditions=exit,
        )

    def explain_path(
        self,
        path: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Explain a path through the statechart."""
        if len(path) < 2:
            return "Single state, no path to explain."

        explanations = []
        transitions = context.get("transitions", [])

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]

            # Find matching transition
            for t in transitions:
                if src in t.get("from", []) and tgt in t.get("to", []):
                    event = t.get("event") or "completion"
                    guard = t.get("guard")

                    step = f"From '{src}' to '{tgt}' on {event}"
                    if guard:
                        step += f" (when {guard})"
                    explanations.append(step)
                    break
            else:
                explanations.append(f"From '{src}' to '{tgt}' (no explicit transition)")

        narrative = f"This path starts at '{path[0]}' and ends at '{path[-1]}'.\n\n"
        narrative += "Steps:\n"
        for i, exp in enumerate(explanations, 1):
            narrative += f"{i}. {exp}\n"

        return narrative

    def _classify_transition(
        self,
        sources: List[str],
        targets: List[str],
        event: str,
        guard: str,
    ) -> TransitionType:
        """Classify transition type."""
        if sources == targets and len(sources) == 1:
            return TransitionType.SELF

        if len(sources) > 1 or len(targets) > 1:
            return TransitionType.COMPOUND

        if not event:
            return TransitionType.COMPLETION

        if guard:
            return TransitionType.GUARDED

        if event:
            return TransitionType.TRIGGERED

        return TransitionType.SIMPLE

    def _classify_state_role(
        self,
        label: str,
        state_type: int,
        is_initial: bool,
        is_final: bool,
        n_incoming: int,
        n_outgoing: int,
        context: Dict,
    ) -> StateRole:
        """Classify state role."""
        if is_initial:
            return StateRole.INITIAL
        if is_final:
            return StateRole.FINAL

        # Check for error-like names
        if any(w in label.lower() for w in ["error", "fail", "exception"]):
            return StateRole.ERROR

        # High connectivity = junction
        if n_incoming >= 3 or n_outgoing >= 3:
            return StateRole.JUNCTION

        # Check for waiting-like names
        if any(w in label.lower() for w in ["wait", "pending", "idle", "ready"]):
            return StateRole.WAITING

        return StateRole.PROCESSING

    def _generate_transition_short(
        self,
        sources: List[str],
        targets: List[str],
        event: str,
        guard: str,
    ) -> str:
        """Generate short transition description."""
        src = " and ".join(sources)
        tgt = " and ".join(targets)

        if event:
            desc = f"On '{event}': {src} → {tgt}"
        else:
            desc = f"Completion: {src} → {tgt}"

        if guard:
            desc += f" [if {guard}]"

        return desc

    def _generate_transition_full(
        self,
        sources: List[str],
        targets: List[str],
        event: str,
        guard: str,
        action: str,
        trans_type: TransitionType,
    ) -> str:
        """Generate full transition description."""
        src = " and ".join(sources)
        tgt = " and ".join(targets)

        if trans_type == TransitionType.SELF:
            if event:
                desc = f"When the '{event}' event occurs while in the '{src}' state, "
                desc += "the system remains in this state"
            else:
                desc = f"The '{src}' state has a self-transition"

        elif trans_type == TransitionType.COMPLETION:
            desc = f"When the '{src}' state completes its activities, "
            desc += f"the system automatically transitions to '{tgt}'"

        elif trans_type == TransitionType.GUARDED:
            desc = f"When '{event}' occurs and the condition '{guard}' is satisfied, "
            desc += f"the system moves from '{src}' to '{tgt}'"

        else:
            if event:
                desc = f"When the '{event}' event is received while in '{src}', "
                desc += f"the system transitions to the '{tgt}' state"
            else:
                desc = f"The system can move from '{src}' to '{tgt}'"

        if action:
            desc += f", executing the action: {action}"

        return desc + "."

    def _generate_state_short(self, label: str, role: StateRole) -> str:
        """Generate short state description."""
        role_desc = {
            StateRole.INITIAL: "Starting point",
            StateRole.FINAL: "End state",
            StateRole.JUNCTION: "Decision point",
            StateRole.PROCESSING: "Active state",
            StateRole.WAITING: "Waiting state",
            StateRole.ERROR: "Error state",
        }
        return f"{label}: {role_desc[role]}"

    def _generate_state_full(
        self,
        label: str,
        role: StateRole,
        state_type: int,
        context: Dict,
    ) -> str:
        """Generate full state description."""
        type_desc = {
            1: "basic",
            2: "composite",
            3: "parallel",
        }.get(state_type, "")

        if role == StateRole.INITIAL:
            return f"'{label}' is the {type_desc} initial state where the system begins execution. " \
                   f"This is the first state entered when the statechart is activated."

        elif role == StateRole.FINAL:
            return f"'{label}' is a {type_desc} final state representing successful completion. " \
                   f"Once this state is reached, the statechart terminates."

        elif role == StateRole.ERROR:
            return f"'{label}' is an error handling state. " \
                   f"The system enters this state when an exceptional condition occurs."

        elif role == StateRole.JUNCTION:
            return f"'{label}' is a {type_desc} junction state with multiple connections. " \
                   f"It serves as a decision point or convergence point in the flow."

        elif role == StateRole.WAITING:
            return f"'{label}' is a {type_desc} waiting state. " \
                   f"The system pauses here until a triggering event occurs."

        else:
            return f"'{label}' is a {type_desc} processing state. " \
                   f"The system performs activities while in this state."

    def _infer_preconditions(
        self,
        sources: List[str],
        guard: str,
        context: Dict,
    ) -> List[str]:
        """Infer preconditions for a transition."""
        conditions = []

        for src in sources:
            conditions.append(f"System must be in '{src}' state")

        if guard:
            conditions.append(f"Condition '{guard}' must be true")

        return conditions

    def _infer_postconditions(
        self,
        targets: List[str],
        action: str,
        context: Dict,
    ) -> List[str]:
        """Infer postconditions after a transition."""
        conditions = []

        for tgt in targets:
            conditions.append(f"System will be in '{tgt}' state")

        if action:
            conditions.append(f"Action '{action}' will have executed")

        return conditions

    def _infer_side_effects(self, action: str) -> List[str]:
        """Infer side effects from action."""
        if not action:
            return []

        effects = []

        # Common action patterns
        if "send" in action.lower():
            effects.append("Message sent to external system")
        if "notify" in action.lower():
            effects.append("Notification triggered")
        if "update" in action.lower():
            effects.append("Data updated")
        if "log" in action.lower():
            effects.append("Event logged")
        if "start" in action.lower() or "begin" in action.lower():
            effects.append("Process initiated")
        if "stop" in action.lower() or "end" in action.lower():
            effects.append("Process terminated")

        return effects or [f"Executes: {action}"]

    def _infer_state_behavior(
        self,
        label: str,
        role: StateRole,
        out_events: List[str],
    ) -> str:
        """Infer behavior while in state."""
        if role == StateRole.WAITING:
            if out_events:
                return f"Waits for one of: {', '.join(out_events)}"
            return "Waits for external stimulus"

        if role == StateRole.PROCESSING:
            return "Executes associated activities"

        if role == StateRole.FINAL:
            return "Terminal state - no further activity"

        if role == StateRole.ERROR:
            return "Handles error condition"

        return "Maintains current state"

    def _infer_entry_conditions(
        self,
        label: str,
        incoming: List[str],
        events: List[str],
    ) -> List[str]:
        """Infer conditions to enter state."""
        if not incoming:
            return ["Initial state or unreachable"]

        conditions = []
        for trans in incoming:
            conditions.append(f"Transition {trans}")

        return conditions

    def _infer_exit_conditions(
        self,
        label: str,
        outgoing: List[str],
        events: List[str],
    ) -> List[str]:
        """Infer conditions to exit state."""
        if not outgoing:
            return ["Final state or no exits defined"]

        conditions = []
        for event in events:
            conditions.append(f"Event '{event}' received")

        return conditions or ["Completion of activities"]


def generate_details(
    statechart: Dict[str, Any],
) -> Tuple[List[TransitionExplanation], List[StateExplanation]]:
    """Generate detailed explanations for all elements."""
    generator = DetailGenerator()

    # Build context
    context = {"transitions": statechart.get("transitions", [])}

    # Explain transitions
    trans_explanations = []
    for t in statechart.get("transitions", []):
        trans_explanations.append(generator.explain_transition(t, context))

    # Explain states
    state_explanations = []

    def process_state(state: Dict, parent: Optional[str] = None):
        label = state.get("label", "")
        if not label.startswith("__"):
            state["parent"] = parent
            state_explanations.append(generator.explain_state(state, context))

        for child in state.get("children", []):
            process_state(child, label if not label.startswith("__") else parent)

    process_state(statechart.get("root_state", {}))

    return trans_explanations, state_explanations


def demo():
    """Demonstrate detail generation."""
    print("=" * 60)
    print("DETAIL GENERATOR: Transition & State Explanations")
    print("=" * 60)

    # Example: Order processing
    order_sc = {
        "name": "Order Processing",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Pending", "type": 1, "is_initial": True},
                {"label": "Validated", "type": 1},
                {"label": "Processing", "type": 1},
                {"label": "Shipped", "type": 1},
                {"label": "Delivered", "type": 1, "is_final": True},
                {"label": "Cancelled", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Pending"], "to": ["Validated"], "event": "VALIDATE",
             "guard": "items_available"},
            {"from": ["Pending"], "to": ["Cancelled"], "event": "CANCEL"},
            {"from": ["Validated"], "to": ["Processing"], "event": "PROCESS",
             "action": "charge_payment()"},
            {"from": ["Processing"], "to": ["Shipped"], "event": "SHIP",
             "action": "send_tracking()"},
            {"from": ["Shipped"], "to": ["Delivered"], "event": "DELIVER"},
        ]
    }

    trans_exps, state_exps = generate_details(order_sc)

    print("\n--- Transition Explanations ---")
    for exp in trans_exps[:3]:  # Show first 3
        print(f"\n{exp.to_text()}")
        print("-" * 40)

    print("\n--- State Explanations ---")
    for exp in state_exps[:3]:  # Show first 3
        print(f"\n{exp.to_text()}")
        print("-" * 40)

    # Explain a path
    generator = DetailGenerator()
    path = ["Pending", "Validated", "Processing", "Shipped", "Delivered"]
    context = {"transitions": order_sc["transitions"]}

    print("\n--- Path Explanation ---")
    print(generator.explain_path(path, context))

    return trans_exps, state_exps


if __name__ == "__main__":
    demo()
