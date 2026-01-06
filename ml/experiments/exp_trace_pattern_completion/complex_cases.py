"""
Complex Test Cases for Trace Pattern Completion

These cases challenge the simple transition-map approach:

1. Context-dependent: Same state leads to different next states
2. Counter-based: Need to count events to predict
3. Hierarchical: Nested patterns with position awareness
4. Noisy: Pattern with occasional variations
5. Long-range: Dependencies spanning many events
6. Conditional: Branching based on history
"""

from typing import List, Dict, Any


# Complex test cases that should break simple transition maps
COMPLEX_TEST_CASES = [
    # === Context-Dependent (need 2+ state history) ===
    {
        "name": "context_2gram",
        "description": "A→B but B→C after A, B→A after C (need 2-gram context)",
        "examples": [
            ["A", "B", "C", "B", "A", "B", "C", "B", "A"],
            ["A", "B", "C", "B", "A", "B", "C"],
            ["A", "B", "C", "B", "A"],
        ],
        "partial": ["A", "B", "C", "B"],
        "expected_1": ["A"],  # After C,B → A
        "expected_3": ["A", "B", "C"],
        "challenge": "context",
        "context_needed": 2,
    },
    {
        "name": "context_aba_vs_cba",
        "description": "After B: if preceded by A→C, if preceded by C→A",
        "examples": [
            ["A", "B", "C", "B", "A", "B", "C", "B", "A"],
            ["C", "B", "A", "B", "C", "B", "A"],
        ],
        "partial": ["A", "B", "C", "B", "A", "B"],
        "expected_1": ["C"],  # A,B → C
        "expected_3": ["C", "B", "A"],
        "challenge": "context",
        "context_needed": 2,
    },

    # === Counter-Based (need to count repetitions) ===
    {
        "name": "counter_3x",
        "description": "Repeat X three times then Y",
        "examples": [
            ["X", "X", "X", "Y", "X", "X", "X", "Y"],
            ["X", "X", "X", "Y", "X", "X", "X", "Y", "X", "X", "X", "Y"],
        ],
        "partial": ["X", "X", "X", "Y", "X", "X"],
        "expected_1": ["X"],  # Need one more X before Y
        "expected_3": ["X", "Y", "X"],
        "challenge": "counter",
        "counter_value": 3,
    },
    {
        "name": "counter_increment",
        "description": "1 A, then 2 As, then 3 As (incrementing)",
        "examples": [
            ["A", "B", "A", "A", "B", "A", "A", "A", "B"],
            ["A", "B", "A", "A", "B", "A", "A", "A", "B", "A", "A", "A", "A", "B"],
        ],
        "partial": ["A", "B", "A", "A", "B", "A", "A", "A", "B", "A", "A", "A", "A"],
        "expected_1": ["B"],  # 4 As done, now B
        "expected_3": ["B", "A", "A"],
        "challenge": "counter",
        "counter_value": "increment",
    },

    # === Hierarchical (nested patterns) ===
    {
        "name": "hier_nested_loop",
        "description": "Outer: START-END, Inner: A,B,C repeats twice",
        "examples": [
            ["START", "A", "B", "C", "A", "B", "C", "END", "START", "A", "B", "C", "A", "B", "C", "END"],
            ["START", "A", "B", "C", "A", "B", "C", "END"],
        ],
        "partial": ["START", "A", "B", "C", "A", "B"],
        "expected_1": ["C"],
        "expected_3": ["C", "END", "START"],
        "challenge": "hierarchical",
        "inner_period": 3,
        "inner_reps": 2,
    },
    {
        "name": "hier_brackets",
        "description": "Bracketed expressions: OPEN, content, CLOSE pairs",
        "examples": [
            ["OPEN", "X", "Y", "CLOSE", "OPEN", "X", "Y", "CLOSE"],
            ["OPEN", "X", "Y", "CLOSE", "OPEN", "X", "Y", "CLOSE", "OPEN"],
        ],
        "partial": ["OPEN", "X", "Y", "CLOSE", "OPEN", "X"],
        "expected_1": ["Y"],
        "expected_3": ["Y", "CLOSE", "OPEN"],
        "challenge": "hierarchical",
    },

    # === Noisy (mostly pattern with variations) ===
    {
        "name": "noisy_cycle",
        "description": "A→B→C cycle but sometimes skips B",
        "examples": [
            ["A", "B", "C", "A", "B", "C", "A", "C", "A", "B", "C"],  # One skip
            ["A", "B", "C", "A", "B", "C", "A", "B", "C"],  # Clean
        ],
        "partial": ["A", "B", "C", "A"],
        "expected_1": ["B"],  # Most likely
        "expected_3": ["B", "C", "A"],
        "challenge": "noisy",
        "noise_rate": 0.1,
    },
    {
        "name": "noisy_majority",
        "description": "Usually A→B, occasionally A→C",
        "examples": [
            ["A", "B", "A", "B", "A", "C", "A", "B", "A", "B"],
            ["A", "B", "A", "B", "A", "B", "A", "C", "A", "B"],
        ],
        "partial": ["A", "B", "A", "B", "A"],
        "expected_1": ["B"],  # Majority vote
        "expected_3": ["B", "A", "B"],
        "challenge": "noisy",
        "noise_rate": 0.2,
    },

    # === Long-Range Dependencies ===
    {
        "name": "long_range_echo",
        "description": "Event at position N determines event at N+5",
        "examples": [
            ["A", "X", "X", "X", "X", "A", "B", "X", "X", "X", "X", "B"],
            ["C", "X", "X", "X", "X", "C", "A", "X", "X", "X", "X", "A"],
        ],
        "partial": ["A", "X", "X", "X", "X", "A", "B", "X", "X", "X", "X"],
        "expected_1": ["B"],  # Echo of position 6
        "expected_3": ["B", "C", "X"],
        "challenge": "long_range",
        "echo_distance": 5,
    },

    # === Conditional Branching ===
    {
        "name": "cond_flag_set",
        "description": "SET_FLAG changes subsequent behavior",
        "examples": [
            ["NORMAL", "A", "B", "A", "B", "SET_FLAG", "A", "C", "A", "C"],
            ["NORMAL", "A", "B", "A", "B", "A", "B"],  # No flag
        ],
        "partial": ["NORMAL", "A", "B", "SET_FLAG", "A"],
        "expected_1": ["C"],  # Flag changes A→? to A→C
        "expected_3": ["C", "A", "C"],
        "challenge": "conditional",
    },
    {
        "name": "cond_mode_switch",
        "description": "MODE_A vs MODE_B determines pattern",
        "examples": [
            ["MODE_A", "X", "Y", "X", "Y", "MODE_B", "X", "Z", "X", "Z"],
            ["MODE_B", "X", "Z", "X", "Z", "MODE_A", "X", "Y", "X", "Y"],
        ],
        "partial": ["MODE_A", "X", "Y", "MODE_B", "X"],
        "expected_1": ["Z"],  # MODE_B means X→Z
        "expected_3": ["Z", "X", "Z"],
        "challenge": "conditional",
    },

    # === Fibonacci-like (depends on previous 2) ===
    {
        "name": "fib_pattern",
        "description": "Next = concat of prev two positions (simplified)",
        "examples": [
            ["A", "B", "AB", "BAB", "ABBAB"],  # Fib-like strings
        ],
        "partial": ["A", "B", "AB"],
        "expected_1": ["BAB"],
        "expected_3": ["BAB", "ABBAB"],
        "challenge": "fibonacci",
    },

    # === State Machine with Memory ===
    {
        "name": "sm_with_stack",
        "description": "PUSH/POP with matching closes",
        "examples": [
            ["PUSH_A", "PUSH_B", "POP_B", "POP_A"],
            ["PUSH_A", "PUSH_B", "PUSH_C", "POP_C", "POP_B", "POP_A"],
        ],
        "partial": ["PUSH_A", "PUSH_B", "PUSH_C", "POP_C"],
        "expected_1": ["POP_B"],  # Stack: [A, B] → pop B
        "expected_3": ["POP_B", "POP_A"],
        "challenge": "stack",
    },
]


def get_challenge_cases(challenge_type: str = None) -> List[Dict[str, Any]]:
    """Get test cases, optionally filtered by challenge type."""
    if challenge_type is None:
        return COMPLEX_TEST_CASES
    return [c for c in COMPLEX_TEST_CASES if c.get("challenge") == challenge_type]


def get_all_challenge_types() -> List[str]:
    """Get all unique challenge types."""
    return list(set(c.get("challenge", "unknown") for c in COMPLEX_TEST_CASES))


if __name__ == "__main__":
    print("Complex Test Cases")
    print("=" * 60)

    for challenge_type in sorted(get_all_challenge_types()):
        cases = get_challenge_cases(challenge_type)
        print(f"\n{challenge_type.upper()} ({len(cases)} cases):")
        for case in cases:
            print(f"  - {case['name']}: {case['description']}")
