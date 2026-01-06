"""
Transition Explainer: Generate NL explanations for state transitions.

Uses Qwen2.5-Coder-0.5B-Instruct to generate fluent explanations
of why transitions occur.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

# MLX imports
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class ExplanationStyle(Enum):
    """Style of explanation."""
    TECHNICAL = "technical"      # Formal, precise
    CASUAL = "casual"           # Friendly, accessible
    STEP_BY_STEP = "step_by_step"  # Detailed reasoning


@dataclass
class ExplainerConfig:
    """Configuration for explainer."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 256
    style: ExplanationStyle = ExplanationStyle.CASUAL


@dataclass
class TransitionExplanation:
    """Detailed explanation of a transition."""
    summary: str
    preconditions: List[str]
    postconditions: List[str]
    reasoning_steps: List[str]
    guard_explanation: Optional[str] = None
    action_explanation: Optional[str] = None
    alternatives_considered: List[str] = field(default_factory=list)


class TransitionExplainer:
    """
    Generate natural language explanations for state transitions.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        self.config = config or ExplainerConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Qwen model."""
        if MLX_AVAILABLE:
            try:
                self.model, self.tokenizer = load(self.config.model_name)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def explain_transition(
        self,
        statechart: Dict[str, Any],
        from_state: str,
        to_state: str,
        event: str,
        transition: Optional[Dict] = None,
    ) -> TransitionExplanation:
        """
        Generate explanation for a specific transition.

        Args:
            statechart: Statechart definition
            from_state: Source state
            to_state: Target state
            event: Triggering event
            transition: Optional transition dict with guard/action

        Returns:
            TransitionExplanation with detailed breakdown
        """
        # Find transition if not provided
        if transition is None:
            for t in statechart.get("transitions", []):
                if (from_state in t.get("from", []) and
                    to_state in t.get("to", []) and
                    t.get("event") == event):
                    transition = t
                    break

        # Build explanation components
        preconditions = [
            f"System must be in state '{from_state}'",
            f"Event '{event}' must be received",
        ]

        postconditions = [
            f"System will be in state '{to_state}'",
        ]

        reasoning = [
            f"Starting in state '{from_state}'",
            f"Event '{event}' arrives",
        ]

        guard_expl = None
        action_expl = None

        if transition:
            guard = transition.get("guard", "")
            action = transition.get("action", "")

            if guard:
                preconditions.append(f"Guard condition '{guard}' must be true")
                guard_expl = f"The transition only fires if {guard}"
                reasoning.append(f"Evaluating guard: {guard}")

            if action:
                postconditions.append(f"Action '{action}' will be executed")
                action_expl = f"Upon transition, {action} is executed"
                reasoning.append(f"Executing action: {action}")

        reasoning.append(f"Transitioning to state '{to_state}'")

        # Check for alternatives
        alternatives = []
        for t in statechart.get("transitions", []):
            if (from_state in t.get("from", []) and
                t.get("event") == event and
                to_state not in t.get("to", [])):
                alt_target = t.get("to", ["?"])[0]
                alt_guard = t.get("guard", "")
                if alt_guard:
                    alternatives.append(f"Could go to '{alt_target}' if {alt_guard}")
                else:
                    alternatives.append(f"Alternative: '{alt_target}'")

        # Generate summary
        summary = self._generate_summary(
            from_state, to_state, event,
            transition.get("guard") if transition else None,
            transition.get("action") if transition else None,
        )

        return TransitionExplanation(
            summary=summary,
            preconditions=preconditions,
            postconditions=postconditions,
            reasoning_steps=reasoning,
            guard_explanation=guard_expl,
            action_explanation=action_expl,
            alternatives_considered=alternatives,
        )

    def _generate_summary(
        self,
        from_state: str,
        to_state: str,
        event: str,
        guard: Optional[str],
        action: Optional[str],
    ) -> str:
        """Generate natural language summary."""
        if self.model is not None:
            return self._llm_summary(from_state, to_state, event, guard, action)
        return self._template_summary(from_state, to_state, event, guard, action)

    def _template_summary(
        self,
        from_state: str,
        to_state: str,
        event: str,
        guard: Optional[str],
        action: Optional[str],
    ) -> str:
        """Generate template-based summary."""
        base = f"When the '{event}' event occurs while in the '{from_state}' state"

        if guard:
            base += f" and {guard}"

        base += f", the system transitions to the '{to_state}' state"

        if action:
            base += f" and executes {action}"

        return base + "."

    def _llm_summary(
        self,
        from_state: str,
        to_state: str,
        event: str,
        guard: Optional[str],
        action: Optional[str],
    ) -> str:
        """Generate LLM-based summary."""
        prompt = f"""Explain this state machine transition in one sentence:
- From: {from_state}
- To: {to_state}
- Event: {event}
- Guard: {guard or 'none'}
- Action: {action or 'none'}

Write a clear, natural explanation:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=100,
            )
            # Clean up response
            response = response.strip()
            if response:
                # Take first sentence
                if "." in response:
                    response = response.split(".")[0] + "."
                return response
        except Exception:
            pass

        return self._template_summary(from_state, to_state, event, guard, action)

    def explain_no_transition(
        self,
        statechart: Dict[str, Any],
        current_state: str,
        event: str,
    ) -> str:
        """Explain why no transition occurred."""
        transitions = statechart.get("transitions", [])

        # Check if any transition from current state exists
        from_current = [
            t for t in transitions
            if current_state in t.get("from", [])
        ]

        if not from_current:
            return (f"No outgoing transitions are defined from state '{current_state}'. "
                    f"The event '{event}' has no effect.")

        # Check if event matches any
        matching_event = [
            t for t in from_current
            if t.get("event") == event
        ]

        if not matching_event:
            available = set(t.get("event", "") for t in from_current)
            return (f"State '{current_state}' has transitions, but not on event '{event}'. "
                    f"Available events: {', '.join(available)}.")

        # Event matches but guards failed
        guards = [t.get("guard", "") for t in matching_event if t.get("guard")]
        if guards:
            return (f"Transition exists from '{current_state}' on '{event}', "
                    f"but guard conditions may have failed: {', '.join(guards)}.")

        return f"No transition from '{current_state}' on '{event}'."

    def explain_sequence(
        self,
        statechart: Dict[str, Any],
        states: List[str],
        events: List[str],
    ) -> str:
        """Explain a sequence of transitions."""
        if len(states) != len(events) + 1:
            return "Invalid sequence: states and events don't match."

        explanations = []
        for i, event in enumerate(events):
            from_s = states[i]
            to_s = states[i + 1]

            if from_s == to_s:
                explanations.append(
                    f"Event '{event}' in '{from_s}': no transition, state unchanged"
                )
            else:
                explanations.append(
                    f"Event '{event}': {from_s} → {to_s}"
                )

        return ". ".join(explanations) + "."


def explain_transition(
    statechart: Dict[str, Any],
    from_state: str,
    to_state: str,
    event: str,
) -> TransitionExplanation:
    """Convenience function for explaining a transition."""
    explainer = TransitionExplainer()
    return explainer.explain_transition(statechart, from_state, to_state, event)


def demo():
    """Demonstrate transition explanation."""
    print("=" * 60)
    print("TRANSITION EXPLAINER: NL Explanations")
    print("=" * 60)

    # Door lock example
    door_lock = {
        "name": "Smart Door Lock",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Locked", "type": 1, "is_initial": True},
                {"label": "Unlocked", "type": 1},
                {"label": "Alarming", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Locked"], "to": ["Unlocked"], "event": "CORRECT_CODE"},
            {"from": ["Locked"], "to": ["Alarming"], "event": "WRONG_CODE",
             "guard": "attempts >= 3"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "TIMEOUT",
             "action": "autolock()"},
            {"from": ["Alarming"], "to": ["Locked"], "event": "RESET"},
        ]
    }

    explainer = TransitionExplainer()

    print("\n--- Transition Explanations ---")

    # Explain successful unlock
    expl1 = explainer.explain_transition(
        door_lock, "Locked", "Unlocked", "CORRECT_CODE"
    )
    print(f"\nLocked -> Unlocked on CORRECT_CODE:")
    print(f"  Summary: {expl1.summary}")
    print(f"  Preconditions: {expl1.preconditions}")
    print(f"  Reasoning: {expl1.reasoning_steps}")

    # Explain guarded transition
    expl2 = explainer.explain_transition(
        door_lock, "Locked", "Alarming", "WRONG_CODE"
    )
    print(f"\nLocked -> Alarming on WRONG_CODE:")
    print(f"  Summary: {expl2.summary}")
    print(f"  Guard: {expl2.guard_explanation}")

    # Explain no transition
    print("\n--- No Transition Explanation ---")
    no_trans = explainer.explain_no_transition(door_lock, "Locked", "TIMEOUT")
    print(f"  {no_trans}")

    return explainer


if __name__ == "__main__":
    demo()
