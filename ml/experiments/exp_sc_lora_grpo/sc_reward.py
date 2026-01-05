#!/usr/bin/env python3
"""
SC Reward Function for GRPO Training.

Computes a reward score for generated statechart outputs based on:
1. Valid JSON structure (0.2)
2. Has root_state field (0.2)
3. Valid state hierarchy (0.2)
4. Has transitions array (0.2)
5. Transitions reference valid states (0.2)

Total possible reward: 1.0
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown."""
    valid_json: float = 0.0
    has_root_state: float = 0.0
    valid_hierarchy: float = 0.0
    has_transitions: float = 0.0
    valid_transitions: float = 0.0

    @property
    def total(self) -> float:
        return (self.valid_json + self.has_root_state +
                self.valid_hierarchy + self.has_transitions +
                self.valid_transitions)

    def to_dict(self) -> Dict[str, float]:
        return {
            'valid_json': self.valid_json,
            'has_root_state': self.has_root_state,
            'valid_hierarchy': self.valid_hierarchy,
            'has_transitions': self.has_transitions,
            'valid_transitions': self.valid_transitions,
            'total': self.total,
        }


def extract_state_labels(state: Dict, labels: Set[str] = None) -> Set[str]:
    """Recursively extract all state labels from hierarchy."""
    if labels is None:
        labels = set()

    if 'label' in state:
        labels.add(state['label'])

    for child in state.get('children', []):
        extract_state_labels(child, labels)

    return labels


def validate_state_hierarchy(state: Dict, depth: int = 0, max_depth: int = 10) -> Tuple[bool, str]:
    """
    Validate state hierarchy structure.

    Returns (is_valid, error_message)
    """
    if depth > max_depth:
        return False, f"Max depth {max_depth} exceeded"

    # Must have label
    if 'label' not in state:
        return False, "State missing 'label'"

    label = state['label']

    # Type should be valid (1=basic, 2=composite, 3=parallel)
    state_type = state.get('type', 1)
    if state_type not in (1, 2, 3):
        return False, f"Invalid state type {state_type} for {label}"

    # Check children
    children = state.get('children', [])

    # Composite states should have children
    if state_type in (2, 3) and not children:
        # This is a warning, not an error
        pass

    # Validate children recursively
    for child in children:
        valid, err = validate_state_hierarchy(child, depth + 1, max_depth)
        if not valid:
            return False, err

    # Check for exactly one initial state among children (if any)
    if children:
        initial_count = sum(1 for c in children if c.get('is_initial', False))
        # Having 0 or 1 initial is fine, more than 1 is invalid
        if initial_count > 1:
            return False, f"Multiple initial states in {label}"

    return True, ""


def validate_transitions(transitions: List[Dict], state_labels: Set[str]) -> Tuple[float, str]:
    """
    Validate transitions reference valid states.

    Returns (score 0-1, error_message)
    """
    if not transitions:
        return 1.0, ""  # No transitions is valid (but maybe not useful)

    valid_count = 0
    total_count = len(transitions)

    for i, trans in enumerate(transitions):
        # Must have from and to
        if 'from' not in trans or 'to' not in trans:
            continue

        from_states = trans['from']
        to_states = trans['to']

        # Check if all states exist
        from_valid = all(s in state_labels for s in from_states)
        to_valid = all(s in state_labels for s in to_states)

        if from_valid and to_valid:
            valid_count += 1

    return valid_count / total_count if total_count > 0 else 1.0, ""


def compute_sc_reward(output: str) -> Tuple[float, RewardBreakdown]:
    """
    Compute reward for a generated statechart output.

    Args:
        output: Generated text (should be JSON)

    Returns:
        (total_reward, breakdown)
    """
    breakdown = RewardBreakdown()

    # 1. Valid JSON (0.2)
    try:
        sc = json.loads(output)
        breakdown.valid_json = 0.2
    except json.JSONDecodeError:
        return 0.0, breakdown

    # Must be a dict
    if not isinstance(sc, dict):
        return breakdown.total, breakdown

    # 2. Has root_state (0.2)
    if 'root_state' in sc and isinstance(sc['root_state'], dict):
        breakdown.has_root_state = 0.2
        root_state = sc['root_state']
    else:
        return breakdown.total, breakdown

    # 3. Valid hierarchy (0.2)
    valid, err = validate_state_hierarchy(root_state)
    if valid:
        breakdown.valid_hierarchy = 0.2

    # Extract state labels for transition validation
    state_labels = extract_state_labels(root_state)

    # 4. Has transitions array (0.2)
    if 'transitions' in sc and isinstance(sc['transitions'], list):
        breakdown.has_transitions = 0.2
        transitions = sc['transitions']

        # 5. Valid transitions (0.2)
        trans_score, _ = validate_transitions(transitions, state_labels)
        breakdown.valid_transitions = 0.2 * trans_score
    else:
        # No transitions is allowed, give partial credit
        breakdown.has_transitions = 0.1
        breakdown.valid_transitions = 0.1

    return breakdown.total, breakdown


def compute_batch_rewards(outputs: List[str]) -> List[Tuple[float, RewardBreakdown]]:
    """Compute rewards for a batch of outputs."""
    return [compute_sc_reward(output) for output in outputs]


def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """
    Compute GRPO advantages (relative to group mean).

    advantages[i] = rewards[i] - mean(rewards)
    """
    if not rewards:
        return []

    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the reward function."""
    print("=" * 60)
    print("SC REWARD FUNCTION DEMO")
    print("=" * 60)

    test_outputs = [
        # Perfect SC
        '''{"root_state": {"label": "Main", "type": 2, "children": [{"label": "Idle", "type": 1, "is_initial": true}, {"label": "Active", "type": 1}]}, "transitions": [{"from": ["Idle"], "to": ["Active"], "event": "START"}]}''',

        # Valid JSON but missing transitions
        '''{"root_state": {"label": "Simple", "type": 1}}''',

        # Invalid JSON
        '''{"root_state": {"label": "Broken"''',

        # Valid JSON but wrong structure
        '''{"states": ["A", "B", "C"]}''',

        # Transitions referencing invalid states
        '''{"root_state": {"label": "Main", "type": 2, "children": [{"label": "A", "type": 1}]}, "transitions": [{"from": ["X"], "to": ["Y"], "event": "GO"}]}''',
    ]

    print("\nTest outputs and rewards:\n")

    for i, output in enumerate(test_outputs):
        reward, breakdown = compute_sc_reward(output)
        print(f"Output {i+1}: {output[:60]}...")
        print(f"  Reward: {reward:.2f}")
        print(f"  Breakdown: {breakdown.to_dict()}")
        print()

    # Test GRPO advantages
    rewards = [r for r, _ in compute_batch_rewards(test_outputs)]
    advantages = compute_grpo_advantages(rewards)

    print("\nGRPO Advantages:")
    print(f"  Rewards: {[f'{r:.2f}' for r in rewards]}")
    print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
    print(f"  Advantages: {[f'{a:+.2f}' for a in advantages]}")


if __name__ == "__main__":
    demo()
