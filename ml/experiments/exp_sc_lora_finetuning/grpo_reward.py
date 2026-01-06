"""
SC Validation Reward Function for GRPO

Scores generated statecharts on:
- JSON validity (0.3)
- Root state presence (0.2)
- Valid transitions (0.3)
- Valid hierarchy (0.2)

Total possible: 1.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json
import subprocess
import re


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown."""
    json_score: float = 0.0       # 0.3 max
    root_state_score: float = 0.0  # 0.2 max
    transitions_score: float = 0.0 # 0.3 max
    hierarchy_score: float = 0.0   # 0.2 max

    @property
    def total(self) -> float:
        return (self.json_score + self.root_state_score +
                self.transitions_score + self.hierarchy_score)

    def to_dict(self) -> Dict[str, float]:
        return {
            "json": self.json_score,
            "root_state": self.root_state_score,
            "transitions": self.transitions_score,
            "hierarchy": self.hierarchy_score,
            "total": self.total,
        }


class SCRewardFunction:
    """
    Reward function for statechart generation.

    Uses SC executor (./sc validate) when available,
    falls back to Python validation.
    """

    def __init__(self, sc_binary_path: str = "./sc"):
        self.sc_binary = sc_binary_path
        self._check_sc_binary()

    def _check_sc_binary(self):
        """Check if SC binary is available."""
        try:
            result = subprocess.run(
                [self.sc_binary, "--help"],
                capture_output=True,
                timeout=5,
            )
            self.use_binary = result.returncode == 0
        except Exception:
            self.use_binary = False

    def compute_reward(self, output: str) -> Tuple[float, RewardBreakdown]:
        """
        Compute reward for generated output.

        Args:
            output: Model-generated text (should contain JSON)

        Returns:
            (total_reward, breakdown)
        """
        breakdown = RewardBreakdown()

        # Try to extract JSON from output
        sc_data = self._extract_json(output)
        if sc_data is None:
            return 0.0, breakdown

        # JSON is valid
        breakdown.json_score = 0.3

        # Check root state
        if self._has_valid_root_state(sc_data):
            breakdown.root_state_score = 0.2

        # Check transitions
        breakdown.transitions_score = self._score_transitions(sc_data)

        # Check hierarchy
        breakdown.hierarchy_score = self._score_hierarchy(sc_data)

        return breakdown.total, breakdown

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        # Try to find JSON in text
        patterns = [
            r'```json\s*([\s\S]*?)```',  # Markdown code block
            r'```\s*([\s\S]*?)```',       # Generic code block
            r'(\{[\s\S]*\})',             # Raw JSON
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Try parsing entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _has_valid_root_state(self, sc: Dict) -> bool:
        """Check if statechart has valid root_state."""
        if "root_state" not in sc:
            return False

        root = sc["root_state"]
        if not isinstance(root, dict):
            return False

        # Must have label or children
        return "label" in root or "children" in root

    def _score_transitions(self, sc: Dict) -> float:
        """Score transitions (0.0 to 0.3)."""
        transitions = sc.get("transitions", [])
        if not transitions:
            return 0.1  # Partial credit for having structure

        # Collect valid state labels
        states = set()
        self._collect_states(sc.get("root_state", {}), states)

        if not states:
            return 0.1

        valid_count = 0
        for t in transitions:
            sources = t.get("from", [])
            targets = t.get("to", [])

            # Check if all referenced states exist
            all_exist = all(s in states for s in sources + targets)
            has_event = bool(t.get("event"))

            if all_exist and (has_event or len(sources) > 0):
                valid_count += 1

        if len(transitions) == 0:
            return 0.1

        validity_ratio = valid_count / len(transitions)
        return 0.3 * validity_ratio

    def _score_hierarchy(self, sc: Dict) -> float:
        """Score hierarchy validity (0.0 to 0.2)."""
        root = sc.get("root_state", {})
        if not root:
            return 0.0

        score = 0.0

        # Has initial state
        if self._has_initial_state(root):
            score += 0.1

        # Valid state types
        if self._valid_state_types(root):
            score += 0.05

        # Proper nesting
        if self._valid_nesting(root):
            score += 0.05

        return score

    def _collect_states(self, state: Dict, states: set):
        """Recursively collect state labels."""
        label = state.get("label", "")
        if label and not label.startswith("__"):
            states.add(label)

        for child in state.get("children", []):
            self._collect_states(child, states)

    def _has_initial_state(self, state: Dict, depth: int = 0) -> bool:
        """Check if there's an initial state."""
        if state.get("is_initial"):
            return True

        for child in state.get("children", []):
            if self._has_initial_state(child, depth + 1):
                return True

        # Root level without explicit initial is ok
        return depth == 0 and state.get("children")

    def _valid_state_types(self, state: Dict) -> bool:
        """Check if state types are valid (1, 2, or 3)."""
        state_type = state.get("type", 1)
        if state_type not in {1, 2, 3}:
            return False

        for child in state.get("children", []):
            if not self._valid_state_types(child):
                return False

        return True

    def _valid_nesting(self, state: Dict, depth: int = 0) -> bool:
        """Check for valid nesting (no too-deep hierarchies)."""
        if depth > 10:  # Suspiciously deep
            return False

        for child in state.get("children", []):
            if not self._valid_nesting(child, depth + 1):
                return False

        return True

    def validate_with_binary(self, sc_json: str) -> Tuple[bool, str]:
        """Validate using SC binary if available."""
        if not self.use_binary:
            return False, "Binary not available"

        try:
            result = subprocess.run(
                [self.sc_binary, "validate", "-"],
                input=sc_json,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0, result.stdout or result.stderr
        except Exception as e:
            return False, str(e)


def sc_reward(output: str) -> float:
    """
    Convenience function for GRPO integration.

    Args:
        output: Generated text from model

    Returns:
        Reward score 0.0 to 1.0
    """
    reward_fn = SCRewardFunction()
    reward, _ = reward_fn.compute_reward(output)
    return reward


def batch_reward(outputs: List[str]) -> List[float]:
    """
    Compute rewards for a batch of outputs.

    Args:
        outputs: List of generated texts

    Returns:
        List of reward scores
    """
    reward_fn = SCRewardFunction()
    return [reward_fn.compute_reward(o)[0] for o in outputs]


def demo():
    """Demonstrate reward function."""
    print("=" * 60)
    print("SC REWARD FUNCTION for GRPO")
    print("=" * 60)

    reward_fn = SCRewardFunction()

    # Test cases
    test_cases = [
        # Good statechart
        {
            "name": "TrafficLight",
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Green", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
                {"from": ["Green"], "to": ["Red"], "event": "TIMER"},
            ]
        },
        # Partial - no transitions
        {
            "name": "Minimal",
            "root_state": {
                "label": "State1",
                "type": 1,
                "is_initial": True,
            }
        },
        # Bad - invalid state references
        {
            "name": "Bad",
            "root_state": {"label": "A", "type": 1},
            "transitions": [
                {"from": ["X"], "to": ["Y"], "event": "E"}
            ]
        },
    ]

    print("\n--- Test Cases ---")
    for i, sc in enumerate(test_cases):
        output = json.dumps(sc)
        reward, breakdown = reward_fn.compute_reward(output)
        print(f"\nCase {i+1}: {sc.get('name', 'Unknown')}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Breakdown: {breakdown.to_dict()}")

    # Test invalid JSON
    print("\n--- Invalid JSON ---")
    reward, breakdown = reward_fn.compute_reward("not json {broken")
    print(f"  Reward: {reward:.2f}")

    # Test with markdown code block
    print("\n--- Markdown Code Block ---")
    md_output = """Here's a statechart:
```json
{"name": "Test", "root_state": {"label": "S", "type": 1, "is_initial": true}}
```
"""
    reward, breakdown = reward_fn.compute_reward(md_output)
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown.to_dict()}")

    return reward_fn


if __name__ == "__main__":
    demo()
