"""
Statechart Explainer: Main SC->NL Conversion Interface

Converts statechart specifications to natural language explanations
using Qwen2.5-Coder-0.5B-Instruct via mlx_lm.

Approach:
1. Parse statechart structure
2. Generate prompt with structured context
3. Use LLM to produce fluent explanation
4. Post-process for clarity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto
import json
import re

# MLX imports (with fallback for testing)
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    def load(*args, **kwargs):
        return None, None
    def generate(*args, **kwargs):
        return "MLX not available - using template-based generation"


class ExplanationLevel(Enum):
    """Level of detail in explanation."""
    BRIEF = "brief"          # One-sentence summary
    STANDARD = "standard"    # Paragraph overview
    DETAILED = "detailed"    # Full explanation with all transitions
    TECHNICAL = "technical"  # Include formal semantics


class ExplanationStyle(Enum):
    """Style of explanation."""
    CONVERSATIONAL = "conversational"  # Friendly, accessible
    FORMAL = "formal"                  # Technical, precise
    TUTORIAL = "tutorial"              # Educational, step-by-step


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    level: ExplanationLevel = ExplanationLevel.STANDARD
    style: ExplanationStyle = ExplanationStyle.CONVERSATIONAL
    max_tokens: int = 512
    temperature: float = 0.7
    include_examples: bool = True
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


@dataclass
class Explanation:
    """Generated explanation of a statechart."""
    summary: str
    state_descriptions: Dict[str, str]
    transition_descriptions: List[str]
    full_text: str
    clarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert explanation to markdown format."""
        lines = [f"# Statechart Explanation\n"]
        lines.append(f"## Summary\n{self.summary}\n")

        if self.state_descriptions:
            lines.append("## States\n")
            for state, desc in self.state_descriptions.items():
                lines.append(f"- **{state}**: {desc}")
            lines.append("")

        if self.transition_descriptions:
            lines.append("## Transitions\n")
            for desc in self.transition_descriptions:
                lines.append(f"- {desc}")
            lines.append("")

        return "\n".join(lines)


class StatechartExplainer:
    """
    Main interface for generating NL explanations of statecharts.

    Uses Qwen2.5-Coder-0.5B-Instruct via mlx_lm for generation.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen model via mlx_lm."""
        if MLX_AVAILABLE:
            try:
                self.model, self.tokenizer = load(self.config.model_name)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                self.model = None
                self.tokenizer = None

    def explain(self, statechart: Dict[str, Any]) -> Explanation:
        """
        Generate natural language explanation of a statechart.

        Args:
            statechart: Statechart definition as dict (sc proto format)

        Returns:
            Explanation with summary, state descriptions, and transitions
        """
        # Parse statechart structure
        parsed = self._parse_statechart(statechart)

        # Generate prompt
        prompt = self._build_prompt(parsed)

        # Generate explanation
        if self.model is not None:
            raw_output = self._generate_with_model(prompt)
        else:
            raw_output = self._generate_template(parsed)

        # Parse and structure output
        explanation = self._parse_output(raw_output, parsed)

        return explanation

    def _parse_statechart(self, sc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse statechart into structured representation."""
        parsed = {
            "name": sc.get("name", "Unnamed"),
            "states": [],
            "transitions": [],
            "hierarchy": {},
            "initial_states": [],
            "final_states": [],
        }

        # Extract states from root
        root = sc.get("root_state", {})
        self._extract_states(root, parsed, parent=None, depth=0)

        # Extract transitions
        for trans in sc.get("transitions", []):
            parsed["transitions"].append({
                "from": trans.get("from", []),
                "to": trans.get("to", []),
                "event": trans.get("event", ""),
                "guard": trans.get("guard", ""),
                "action": trans.get("action", ""),
            })

        return parsed

    def _extract_states(
        self,
        state: Dict,
        parsed: Dict,
        parent: Optional[str],
        depth: int,
    ):
        """Recursively extract states from hierarchy."""
        label = state.get("label", "")
        if label.startswith("__"):
            label = None  # Skip root markers

        state_info = {
            "label": label,
            "type": state.get("type", 1),  # 1=BASIC, 2=NORMAL, 3=PARALLEL
            "parent": parent,
            "depth": depth,
            "is_initial": state.get("is_initial", False),
            "is_final": state.get("is_final", False),
        }

        if label:
            parsed["states"].append(state_info)
            if state_info["is_initial"]:
                parsed["initial_states"].append(label)
            if state_info["is_final"]:
                parsed["final_states"].append(label)

        # Recurse into children
        for child in state.get("children", []):
            self._extract_states(child, parsed, label, depth + 1)

    def _build_prompt(self, parsed: Dict) -> str:
        """Build prompt for LLM generation."""
        state_list = ", ".join(s["label"] for s in parsed["states"] if s["label"])

        trans_list = []
        for t in parsed["transitions"]:
            src = ", ".join(t["from"])
            tgt = ", ".join(t["to"])
            event = t["event"] or "completion"
            guard = f" [if {t['guard']}]" if t["guard"] else ""
            trans_list.append(f"{src} --({event}){guard}--> {tgt}")

        trans_str = "\n".join(trans_list) if trans_list else "No transitions defined"

        level_instruction = {
            ExplanationLevel.BRIEF: "Write a single sentence summary.",
            ExplanationLevel.STANDARD: "Write a clear paragraph explanation.",
            ExplanationLevel.DETAILED: "Write a comprehensive explanation covering all aspects.",
            ExplanationLevel.TECHNICAL: "Write a technical explanation with formal semantics.",
        }[self.config.level]

        style_instruction = {
            ExplanationStyle.CONVERSATIONAL: "Use friendly, accessible language.",
            ExplanationStyle.FORMAL: "Use precise, technical language.",
            ExplanationStyle.TUTORIAL: "Explain step-by-step as if teaching.",
        }[self.config.style]

        prompt = f"""You are explaining a statechart (state machine) to a user.

Statechart: {parsed["name"]}
States: {state_list}
Initial: {", ".join(parsed["initial_states"]) or "Not specified"}
Final: {", ".join(parsed["final_states"]) or "None"}

Transitions:
{trans_str}

{level_instruction} {style_instruction}

Explain what this statechart does:"""

        return prompt

    def _generate_with_model(self, prompt: str) -> str:
        """Generate explanation using the loaded model."""
        if self.tokenizer is None:
            return self._generate_template({})

        try:
            # Format as chat for instruct model
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            output = generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=self.config.max_tokens,
            )
            return output
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_template({})

    def _generate_template(self, parsed: Dict) -> str:
        """Fallback template-based generation."""
        states = parsed.get("states", [])
        transitions = parsed.get("transitions", [])
        name = parsed.get("name", "This statechart")

        # Build template explanation
        parts = []

        # Summary
        n_states = len([s for s in states if s.get("label")])
        n_trans = len(transitions)
        parts.append(
            f"{name} is a state machine with {n_states} states and {n_trans} transitions."
        )

        # Initial state
        initial = parsed.get("initial_states", [])
        if initial:
            parts.append(f"It starts in the '{initial[0]}' state.")

        # Describe key transitions
        if transitions:
            parts.append("The following transitions are defined:")
            for t in transitions[:5]:  # Limit to first 5
                src = ", ".join(t.get("from", []))
                tgt = ", ".join(t.get("to", []))
                event = t.get("event") or "automatically"
                guard = t.get("guard")

                if guard:
                    parts.append(
                        f"- From '{src}' to '{tgt}' when '{event}' occurs (if {guard})."
                    )
                else:
                    parts.append(
                        f"- From '{src}' to '{tgt}' when '{event}' occurs."
                    )

        # Final states
        final = parsed.get("final_states", [])
        if final:
            parts.append(f"The machine terminates when it reaches '{final[0]}'.")

        return " ".join(parts)

    def _parse_output(self, raw: str, parsed: Dict) -> Explanation:
        """Parse LLM output into structured Explanation."""
        # Extract summary (first sentence or paragraph)
        sentences = raw.split(". ")
        summary = sentences[0] + "." if sentences else raw[:200]

        # Generate state descriptions
        state_descs = {}
        for state in parsed.get("states", []):
            label = state.get("label")
            if label:
                state_descs[label] = self._describe_state(state, parsed)

        # Generate transition descriptions
        trans_descs = []
        for t in parsed.get("transitions", []):
            trans_descs.append(self._describe_transition(t))

        return Explanation(
            summary=summary,
            state_descriptions=state_descs,
            transition_descriptions=trans_descs,
            full_text=raw,
            metadata={
                "model": self.config.model_name,
                "level": self.config.level.value,
                "style": self.config.style.value,
            }
        )

    def _describe_state(self, state: Dict, parsed: Dict) -> str:
        """Generate description for a single state."""
        label = state.get("label", "")
        state_type = state.get("type", 1)

        type_desc = {
            1: "basic state",
            2: "composite state",
            3: "parallel state",
        }.get(state_type, "state")

        parts = [f"A {type_desc}"]

        if state.get("is_initial"):
            parts.append("that serves as the initial state")
        if state.get("is_final"):
            parts.append("that is a final/accepting state")

        # Find outgoing transitions
        outgoing = [
            t for t in parsed.get("transitions", [])
            if label in t.get("from", [])
        ]
        if outgoing:
            events = [t.get("event") or "completion" for t in outgoing]
            parts.append(f"with transitions on: {', '.join(set(events))}")

        return ". ".join(parts) + "."

    def _describe_transition(self, trans: Dict) -> str:
        """Generate description for a single transition."""
        src = " and ".join(trans.get("from", ["?"]))
        tgt = " and ".join(trans.get("to", ["?"]))
        event = trans.get("event") or "completion"
        guard = trans.get("guard")
        action = trans.get("action")

        desc = f"When '{event}' occurs"
        if guard:
            desc += f" and {guard}"
        desc += f", transition from {src} to {tgt}"
        if action:
            desc += f", executing {action}"

        return desc + "."


def explain_statechart(
    statechart: Dict[str, Any],
    level: ExplanationLevel = ExplanationLevel.STANDARD,
    style: ExplanationStyle = ExplanationStyle.CONVERSATIONAL,
) -> Explanation:
    """
    Convenience function to explain a statechart.

    Args:
        statechart: Statechart definition as dict
        level: Detail level (brief, standard, detailed, technical)
        style: Explanation style (conversational, formal, tutorial)

    Returns:
        Explanation object with full NL explanation
    """
    config = ExplanationConfig(level=level, style=style)
    explainer = StatechartExplainer(config)
    return explainer.explain(statechart)


def demo():
    """Demonstrate statechart explanation."""
    print("=" * 60)
    print("STATECHART EXPLAINER: SC -> Natural Language")
    print("=" * 60)

    # Example: Traffic light statechart
    traffic_light = {
        "name": "Traffic Light Controller",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    print("\n--- Traffic Light ---")
    print(json.dumps(traffic_light, indent=2))

    explanation = explain_statechart(traffic_light)
    print("\n--- Explanation ---")
    print(f"Summary: {explanation.summary}")
    print(f"\nStates:")
    for state, desc in explanation.state_descriptions.items():
        print(f"  {state}: {desc}")
    print(f"\nTransitions:")
    for desc in explanation.transition_descriptions:
        print(f"  - {desc}")

    # Example: Door lock statechart
    print("\n" + "=" * 60)
    door_lock = {
        "name": "Smart Door Lock",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Locked", "type": 1, "is_initial": True},
                {"label": "Unlocked", "type": 1},
                {"label": "Error", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Locked"], "to": ["Unlocked"], "event": "CORRECT_CODE"},
            {"from": ["Locked"], "to": ["Error"], "event": "WRONG_CODE",
             "guard": "attempts >= 3"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "TIMEOUT"},
        ]
    }

    print("\n--- Door Lock ---")
    explanation2 = explain_statechart(
        door_lock,
        level=ExplanationLevel.DETAILED,
        style=ExplanationStyle.TUTORIAL,
    )
    print(f"Summary: {explanation2.summary}")
    print(f"\nFull text:\n{explanation2.full_text}")

    return explanation, explanation2


if __name__ == "__main__":
    demo()
