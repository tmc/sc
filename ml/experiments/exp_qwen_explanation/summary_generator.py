"""
Summary Generator: High-Level Statechart Summaries

Generates concise, high-level summaries of statecharts that capture:
- Purpose and domain
- Key states and their roles
- Overall behavior pattern
- Complexity assessment
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


class BehaviorPattern(Enum):
    """Common statechart behavior patterns."""
    SEQUENTIAL = "sequential"      # Linear state progression
    CYCLIC = "cyclic"              # Repeating cycle
    BRANCHING = "branching"        # Multiple paths
    HIERARCHICAL = "hierarchical"  # Nested states
    PARALLEL = "parallel"          # Concurrent regions
    REACTIVE = "reactive"          # Event-driven


class ComplexityLevel(Enum):
    """Statechart complexity assessment."""
    SIMPLE = "simple"          # 1-3 states, linear
    MODERATE = "moderate"      # 4-10 states, some branching
    COMPLEX = "complex"        # 10+ states, hierarchy
    VERY_COMPLEX = "very_complex"  # Parallel regions, guards


@dataclass
class StatechartSummary:
    """High-level summary of a statechart."""
    one_liner: str                    # Single sentence
    purpose: str                      # What it does
    domain: str                       # Application domain
    behavior_pattern: BehaviorPattern
    complexity: ComplexityLevel

    # Metrics
    n_states: int
    n_transitions: int
    n_events: int
    max_depth: int

    # Key insights
    key_states: List[str]
    key_events: List[str]
    notable_features: List[str]

    def to_text(self) -> str:
        """Convert summary to readable text."""
        lines = [
            f"**Summary**: {self.one_liner}",
            f"",
            f"**Purpose**: {self.purpose}",
            f"**Domain**: {self.domain}",
            f"**Pattern**: {self.behavior_pattern.value}",
            f"**Complexity**: {self.complexity.value}",
            f"",
            f"**Metrics**: {self.n_states} states, {self.n_transitions} transitions, "
            f"{self.n_events} events, depth {self.max_depth}",
            f"",
            f"**Key States**: {', '.join(self.key_states)}",
            f"**Key Events**: {', '.join(self.key_events)}",
        ]

        if self.notable_features:
            lines.append(f"**Notable**: {', '.join(self.notable_features)}")

        return "\n".join(lines)


class SummaryGenerator:
    """
    Generate high-level summaries of statecharts.

    Combines structural analysis with LLM-based natural language generation.
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

    def generate(self, statechart: Dict[str, Any]) -> StatechartSummary:
        """Generate comprehensive summary of statechart."""
        # Analyze structure
        analysis = self._analyze_structure(statechart)

        # Determine behavior pattern
        pattern = self._detect_pattern(analysis)

        # Assess complexity
        complexity = self._assess_complexity(analysis)

        # Generate natural language components
        one_liner = self._generate_one_liner(statechart, analysis)
        purpose = self._infer_purpose(statechart, analysis)
        domain = self._infer_domain(statechart, analysis)

        # Identify key elements
        key_states = self._identify_key_states(analysis)
        key_events = self._identify_key_events(analysis)
        features = self._identify_features(analysis)

        return StatechartSummary(
            one_liner=one_liner,
            purpose=purpose,
            domain=domain,
            behavior_pattern=pattern,
            complexity=complexity,
            n_states=analysis["n_states"],
            n_transitions=analysis["n_transitions"],
            n_events=analysis["n_events"],
            max_depth=analysis["max_depth"],
            key_states=key_states,
            key_events=key_events,
            notable_features=features,
        )

    def _analyze_structure(self, sc: Dict) -> Dict[str, Any]:
        """Analyze statechart structure."""
        analysis = {
            "name": sc.get("name", "Unnamed"),
            "states": [],
            "transitions": [],
            "events": set(),
            "n_states": 0,
            "n_transitions": 0,
            "n_events": 0,
            "max_depth": 0,
            "has_parallel": False,
            "has_guards": False,
            "has_hierarchy": False,
            "cycles": [],
            "initial": [],
            "final": [],
        }

        # Extract states
        root = sc.get("root_state", {})
        self._analyze_states(root, analysis, depth=0)

        # Extract transitions
        for t in sc.get("transitions", []):
            analysis["transitions"].append(t)
            if t.get("event"):
                analysis["events"].add(t["event"])
            if t.get("guard"):
                analysis["has_guards"] = True

        analysis["n_states"] = len(analysis["states"])
        analysis["n_transitions"] = len(analysis["transitions"])
        analysis["n_events"] = len(analysis["events"])

        # Detect cycles
        analysis["cycles"] = self._detect_cycles(analysis)

        return analysis

    def _analyze_states(self, state: Dict, analysis: Dict, depth: int):
        """Recursively analyze state hierarchy."""
        label = state.get("label", "")
        state_type = state.get("type", 1)

        if not label.startswith("__"):
            analysis["states"].append({
                "label": label,
                "type": state_type,
                "depth": depth,
                "is_initial": state.get("is_initial", False),
                "is_final": state.get("is_final", False),
            })

            if state.get("is_initial"):
                analysis["initial"].append(label)
            if state.get("is_final"):
                analysis["final"].append(label)

            if depth > analysis["max_depth"]:
                analysis["max_depth"] = depth

            if state_type == 3:  # PARALLEL
                analysis["has_parallel"] = True

        children = state.get("children", [])
        if children and not label.startswith("__"):
            analysis["has_hierarchy"] = True

        for child in children:
            self._analyze_states(child, analysis, depth + 1)

    def _detect_cycles(self, analysis: Dict) -> List[List[str]]:
        """Detect cycles in transition graph."""
        cycles = []

        # Build adjacency
        adj: Dict[str, Set[str]] = {}
        for t in analysis["transitions"]:
            for src in t.get("from", []):
                if src not in adj:
                    adj[src] = set()
                for tgt in t.get("to", []):
                    adj[src].add(tgt)

        # Simple cycle detection via DFS
        visited = set()
        path = []

        def dfs(node: str, path_set: Set[str]):
            if node in path_set:
                # Found cycle
                idx = path.index(node)
                cycles.append(path[idx:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path_set.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                dfs(neighbor, path_set)

            path.pop()
            path_set.remove(node)

        for start in adj:
            dfs(start, set())

        return cycles

    def _detect_pattern(self, analysis: Dict) -> BehaviorPattern:
        """Detect dominant behavior pattern."""
        if analysis["has_parallel"]:
            return BehaviorPattern.PARALLEL

        if analysis["has_hierarchy"]:
            return BehaviorPattern.HIERARCHICAL

        if analysis["cycles"]:
            return BehaviorPattern.CYCLIC

        # Check for branching
        out_degrees = {}
        for t in analysis["transitions"]:
            for src in t.get("from", []):
                out_degrees[src] = out_degrees.get(src, 0) + 1

        if any(d > 1 for d in out_degrees.values()):
            return BehaviorPattern.BRANCHING

        if analysis["n_events"] > 0:
            return BehaviorPattern.REACTIVE

        return BehaviorPattern.SEQUENTIAL

    def _assess_complexity(self, analysis: Dict) -> ComplexityLevel:
        """Assess statechart complexity."""
        score = 0

        # State count
        n = analysis["n_states"]
        if n <= 3:
            score += 1
        elif n <= 10:
            score += 2
        else:
            score += 3

        # Features
        if analysis["has_parallel"]:
            score += 2
        if analysis["has_hierarchy"]:
            score += 1
        if analysis["has_guards"]:
            score += 1
        if len(analysis["cycles"]) > 1:
            score += 1

        if score <= 2:
            return ComplexityLevel.SIMPLE
        elif score <= 4:
            return ComplexityLevel.MODERATE
        elif score <= 6:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _generate_one_liner(self, sc: Dict, analysis: Dict) -> str:
        """Generate one-sentence summary."""
        name = analysis["name"]
        n_states = analysis["n_states"]
        pattern = self._detect_pattern(analysis)

        pattern_desc = {
            BehaviorPattern.SEQUENTIAL: "progresses through",
            BehaviorPattern.CYCLIC: "cycles through",
            BehaviorPattern.BRANCHING: "branches between",
            BehaviorPattern.HIERARCHICAL: "organizes",
            BehaviorPattern.PARALLEL: "coordinates",
            BehaviorPattern.REACTIVE: "reacts across",
        }[pattern]

        if self.model is not None:
            # Use LLM for better summary
            prompt = f"""Write a single sentence summary for this statechart:
Name: {name}
States: {n_states}
Pattern: {pattern.value}
Events: {', '.join(analysis['events']) or 'none'}

One sentence summary:"""

            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                output = generate(
                    self.model, self.tokenizer,
                    prompt=formatted, max_tokens=50, temp=0.5
                )
                return output.strip().split("\n")[0]
            except Exception:
                pass

        # Fallback template
        return f"{name} {pattern_desc} {n_states} states to model its behavior."

    def _infer_purpose(self, sc: Dict, analysis: Dict) -> str:
        """Infer the purpose of the statechart."""
        name = analysis["name"].lower()
        events = [e.lower() for e in analysis["events"]]

        # Common purpose patterns
        if any(w in name for w in ["controller", "control"]):
            return "Controls and coordinates system behavior"
        if any(w in name for w in ["workflow", "process"]):
            return "Manages a multi-step workflow or process"
        if any(w in name for w in ["ui", "interface", "menu"]):
            return "Manages user interface states and navigation"
        if any(w in name for w in ["auth", "login", "session"]):
            return "Handles authentication and session management"
        if any(w in events for w in ["error", "fail", "retry"]):
            return "Manages error handling and recovery"

        return "Defines behavior through state transitions"

    def _infer_domain(self, sc: Dict, analysis: Dict) -> str:
        """Infer the application domain."""
        name = analysis["name"].lower()
        states = [s["label"].lower() for s in analysis["states"]]
        all_text = name + " " + " ".join(states)

        domains = {
            "UI/UX": ["button", "menu", "dialog", "screen", "view", "click"],
            "Networking": ["connect", "disconnect", "send", "receive", "socket"],
            "Authentication": ["login", "logout", "auth", "session", "token"],
            "E-commerce": ["cart", "checkout", "payment", "order", "shipping"],
            "Gaming": ["play", "pause", "game", "level", "score", "player"],
            "IoT/Hardware": ["sensor", "actuator", "device", "power", "sleep"],
            "Workflow": ["pending", "approved", "review", "submit", "complete"],
        }

        for domain, keywords in domains.items():
            if any(kw in all_text for kw in keywords):
                return domain

        return "General"

    def _identify_key_states(self, analysis: Dict) -> List[str]:
        """Identify the most important states."""
        key = []

        # Initial and final states are always key
        key.extend(analysis["initial"])
        key.extend(analysis["final"])

        # States with many transitions
        in_count = {}
        out_count = {}
        for t in analysis["transitions"]:
            for src in t.get("from", []):
                out_count[src] = out_count.get(src, 0) + 1
            for tgt in t.get("to", []):
                in_count[tgt] = in_count.get(tgt, 0) + 1

        # Add high-connectivity states
        for state in analysis["states"]:
            label = state["label"]
            total = in_count.get(label, 0) + out_count.get(label, 0)
            if total >= 3 and label not in key:
                key.append(label)

        return key[:5]  # Limit to top 5

    def _identify_key_events(self, analysis: Dict) -> List[str]:
        """Identify the most important events."""
        event_count = {}
        for t in analysis["transitions"]:
            event = t.get("event")
            if event:
                event_count[event] = event_count.get(event, 0) + 1

        # Sort by frequency
        sorted_events = sorted(event_count.items(), key=lambda x: -x[1])
        return [e[0] for e in sorted_events[:5]]

    def _identify_features(self, analysis: Dict) -> List[str]:
        """Identify notable features."""
        features = []

        if analysis["has_parallel"]:
            features.append("Concurrent regions (AND-states)")
        if analysis["has_hierarchy"]:
            features.append("Hierarchical nesting")
        if analysis["has_guards"]:
            features.append("Guarded transitions")
        if analysis["cycles"]:
            features.append(f"{len(analysis['cycles'])} cycles detected")
        if analysis["final"]:
            features.append("Has final/accepting states")

        return features


def generate_summary(statechart: Dict[str, Any]) -> StatechartSummary:
    """Convenience function to generate summary."""
    generator = SummaryGenerator()
    return generator.generate(statechart)


def demo():
    """Demonstrate summary generation."""
    print("=" * 60)
    print("SUMMARY GENERATOR: High-Level Statechart Summaries")
    print("=" * 60)

    # Example: E-commerce checkout
    checkout = {
        "name": "E-commerce Checkout Flow",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Cart", "type": 1, "is_initial": True},
                {"label": "Shipping", "type": 1},
                {"label": "Payment", "type": 1},
                {"label": "Review", "type": 1},
                {"label": "Confirmed", "type": 1, "is_final": True},
                {"label": "Cancelled", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Cart"], "to": ["Shipping"], "event": "CHECKOUT"},
            {"from": ["Shipping"], "to": ["Payment"], "event": "CONTINUE"},
            {"from": ["Payment"], "to": ["Review"], "event": "PAY"},
            {"from": ["Review"], "to": ["Confirmed"], "event": "CONFIRM"},
            {"from": ["Cart"], "to": ["Cancelled"], "event": "CANCEL"},
            {"from": ["Shipping"], "to": ["Cart"], "event": "BACK"},
            {"from": ["Payment"], "to": ["Shipping"], "event": "BACK"},
        ]
    }

    print("\n--- E-commerce Checkout ---")
    summary = generate_summary(checkout)
    print(summary.to_text())

    # Example: Game character
    print("\n" + "=" * 60)
    game_char = {
        "name": "Game Character Controller",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "Walking", "type": 1},
                {"label": "Running", "type": 1},
                {"label": "Jumping", "type": 1},
                {"label": "Attacking", "type": 1},
                {"label": "Dead", "type": 1, "is_final": True},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Walking"], "event": "MOVE"},
            {"from": ["Walking"], "to": ["Running"], "event": "SPRINT"},
            {"from": ["Walking"], "to": ["Idle"], "event": "STOP"},
            {"from": ["Running"], "to": ["Walking"], "event": "WALK"},
            {"from": ["Idle", "Walking", "Running"], "to": ["Jumping"], "event": "JUMP"},
            {"from": ["Jumping"], "to": ["Idle"], "event": "LAND"},
            {"from": ["Idle", "Walking"], "to": ["Attacking"], "event": "ATTACK"},
            {"from": ["Attacking"], "to": ["Idle"], "event": "ATTACK_END"},
            {"from": ["Idle", "Walking", "Running", "Jumping"], "to": ["Dead"], "event": "DIE"},
        ]
    }

    print("\n--- Game Character ---")
    summary2 = generate_summary(game_char)
    print(summary2.to_text())

    return summary, summary2


if __name__ == "__main__":
    demo()
